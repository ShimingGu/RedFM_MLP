#!/usr/bin/env python3
"""Persistent single-slice lease and detached supervisor for 1110 training."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
LOCK_ROOT = Path("/scratch/.tmp-gsm/aion_output/gpu_locks") / str(os.getuid())
DEVICE = "MIG-ed1476fa-775b-52cc-85d6-6b7ad1ef2323"


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def verify_device():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=DEVICE, PYTHONNOUSERSITE="1")
    guard = ROOT / "scripts/qwen_survey_gpu_guard.py"
    out = subprocess.check_output([sys.executable, str(guard), "--resolve"], env=env, text=True)
    resolved = json.loads(out)
    expected = [DEVICE.removeprefix("MIG-")]
    if resolved != expected:
        raise RuntimeError(f"Device {DEVICE} resolved to {resolved}, expected {expected}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=DEVICE)
    args = ap.parse_args()
    device = args.device
    if device != DEVICE:
        raise SystemExit("Only the campaign-owned slice is authorised")
    verify_device()
    uuid = device.removeprefix("MIG-")
    lock_path = LOCK_ROOT / f"{uuid}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = OUT / "status/supervisor.json"
    if (OUT / "training/main/status.json").exists():
        old = json.loads((OUT / "training/main/status.json").read_text())
        if old.get("stage") == "complete":
            print("TRAINING_ALREADY_COMPLETE", flush=True)
            return 0
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip()
        write(status_path, dict(utc=now(), stage="waiting_for_lease", device=device, lock=str(lock_path), owner=owner, pid=os.getpid()))
        print(f"LEASE_BUSY {owner}", flush=True)
        return 2
    supervisor_start = now()
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(dict(owner="redshift-anchor-1110-main", supervisor_pid=os.getpid(), started_utc=supervisor_start, device=device)))
    handle.flush()
    log_path = OUT / "logs/train_main.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=device,
        PYTHONNOUSERSITE="1",
        PYTHONPATH=str(ROOT),
        ANCHOR_DEVICE_LEASE_FD=str(handle.fileno()),
        HF_HOME=str(OUT / "cache/hf"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    cmd = [sys.executable, str(ROOT / "scripts/train_anchor_1110.py"), "--resume"]
    write(status_path, dict(utc=now(), stage="launching", device=device, lock=str(lock_path), supervisor_pid=os.getpid(), started_utc=supervisor_start, command=cmd))
    child = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, pass_fds=(handle.fileno(),))
    write(status_path, dict(utc=now(), stage="running", device=device, lock=str(lock_path), supervisor_pid=os.getpid(), child_pid=child.pid, started_utc=supervisor_start, log=str(log_path), command=cmd))

    def forward(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, forward)
    started = time.time()
    while True:
        code = child.poll()
        status = json.loads((OUT / "training/main/status.json").read_text()) if (OUT / "training/main/status.json").exists() else {}
        write(status_path, dict(
            utc=now(), stage="running" if code is None else ("complete" if code == 0 else "failed"),
            device=device, lock=str(lock_path), supervisor_pid=os.getpid(), child_pid=child.pid,
            started_utc=supervisor_start, elapsed_seconds=time.time() - started, log=str(log_path), child_returncode=code,
            training_status=status,
        ))
        if code is not None:
            break
        time.sleep(10)
    log.close()
    handle.close()
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
