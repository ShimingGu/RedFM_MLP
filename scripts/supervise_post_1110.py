#!/usr/bin/env python3
"""Persistent post-training waiter: dev selection + main calibration, then a hard gate.

It deliberately stops at `gated_route_decision`.  It never creates an evaluation
gate, never reads P/E labels and never runs final inference.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
DEVICE = "MIG-ed1476fa-775b-52cc-85d6-6b7ad1ef2323"
LOCK_ROOT = Path("/scratch/.tmp-gsm/aion_output/gpu_locks") / str(os.getuid())


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def run(cmd, env, log):
    with log.open("a") as fh:
        fh.write(f"\n[{now()}] RUN {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.call([str(x) for x in cmd], cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)


def acquire(handle):
    for _ in range(120):
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            time.sleep(5)
    return False


def main():
    status_path = OUT / "status/posttrain.json"
    log_path = OUT / "logs/posttrain.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write(status_path, dict(utc=now(), stage="waiting_for_training", pid=os.getpid()))
    while True:
        train = json.loads((OUT / "training/main/status.json").read_text()) if (OUT / "training/main/status.json").exists() else {}
        if train.get("stage") == "complete":
            break
        if train.get("stage") == "failed":
            write(status_path, dict(utc=now(), stage="blocked_training_failed", training=train, pid=os.getpid()))
            return 1
        time.sleep(30)
    write(status_path, dict(utc=now(), stage="training_complete_starting_dev_selection", training=train, pid=os.getpid()))
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=str(ROOT))
    code = run([sys.executable, ROOT / "scripts/select_checkpoint_1110.py", "--execute"], env, log_path)
    if code != 0:
        write(status_path, dict(utc=now(), stage="failed_dev_selection", returncode=code, pid=os.getpid()))
        return 1
    write(status_path, dict(utc=now(), stage="dev_selection_complete_starting_calibration", training=train, pid=os.getpid()))
    uuid = DEVICE.removeprefix("MIG-")
    lock = (LOCK_ROOT / f"{uuid}.lock").open("a+")
    if not acquire(lock):
        write(status_path, dict(utc=now(), stage="waiting_for_gpu_lease", device=DEVICE, pid=os.getpid()))
        return 2
    lock.seek(0); lock.truncate(); lock.write(json.dumps(dict(owner="redshift-anchor-1110-post", pid=os.getpid(), started_utc=now(), device=DEVICE))); lock.flush()
    gpu_env = dict(env, CUDA_VISIBLE_DEVICES=DEVICE, HF_HOME=str(OUT / "cache/hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", ANCHOR_DEVICE_LEASE_FD=str(lock.fileno()))
    code = run([sys.executable, ROOT / "scripts/calibrate_main_1110.py", "--execute"], gpu_env, log_path)
    lock.close()
    if code != 0:
        write(status_path, dict(utc=now(), stage="failed_main_calibration", returncode=code, pid=os.getpid()))
        return 1
    write(status_path, dict(
        utc=now(), stage="gated_route_decision", pid=os.getpid(), device=DEVICE,
        message="Dev selection and main calibration are complete. A human/agent must review the main-vs-revision route. No P/E gate was created; final inference is not authorised.",
        model_selection=str(OUT / "evaluation/model_selection.json"), calibrator=str(OUT / "evaluation/main_calibrator.json"),
        next_manual_action="write evaluation/route_decision.json, then run the explicit freeze/evaluation entry points",
    ))
    print("POSTTRAIN_GATED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
