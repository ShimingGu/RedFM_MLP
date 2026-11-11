#!/usr/bin/env python3
"""Post-gate report supervisor; deliberately does nothing until an explicit eval gate exists."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    status_path = OUT / "status/report_supervisor.json"
    log_path = OUT / "logs/report_supervisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write(status_path, dict(utc=now(), stage="waiting_for_eval_gate", pid=os.getpid()))
    while True:
        gate = OUT / "evaluation/eval_gate.json"
        if gate.exists() and json.loads(gate.read_text()).get("authorized") is True:
            break
        time.sleep(30)
    required = [OUT / "evaluation/final/P/complete.json", OUT / "evaluation/final/E/complete.json"]
    write(status_path, dict(utc=now(), stage="waiting_for_frozen_predictions", pid=os.getpid(), required=[str(x) for x in required]))
    while not all(x.exists() for x in required):
        time.sleep(30)
    write(status_path, dict(utc=now(), stage="running_report", pid=os.getpid()))
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONPATH=str(ROOT), HF_HOME=str(OUT / "cache/hf"), TRANSFORMERS_OFFLINE="1")
    with log_path.open("a") as log:
        code = subprocess.call([sys.executable, str(ROOT / "scripts/evaluate_frozen_1110.py"), "report"], cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    write(status_path, dict(utc=now(), stage="complete" if code == 0 else "failed", returncode=code, pid=os.getpid(), log=str(log_path)))
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
