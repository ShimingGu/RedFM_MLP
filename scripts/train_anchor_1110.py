#!/usr/bin/env python3
"""Launch or resume the 1110 main mixed-data training run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
H1 = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1102/highz-hsc-pilot-zcut0p6")
sys.path.insert(0, str(ROOT))

from aion_magnitude.redshift_anchor_training import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    h1cfg = json.loads((H1 / "protocol/configs/H1-A.json").read_text())
    gate = json.loads((H1 / "evaluation/final_checkpoints_frozen.json").read_text())
    state_path = H1 / "training/H1-A/final/trainable_state.pt"
    state_sha = gate["models"]["H1-A"]["primary_trainable_state_sha256"]
    import hashlib
    actual = hashlib.sha256(state_path.read_bytes()).hexdigest()
    assert actual == state_sha, (actual, state_sha)

    training = dict(h1cfg["training"])
    training.update(z_min=0.0, z_max=4.5, n_z_bins=300, seed=1110, device="cuda")
    config = dict(
        name="RA-1110-main",
        seed=1110,
        training=training,
        initial_trainable_state=str(state_path),
        initial_trainable_state_sha256=state_sha,
        initialisation="H1-A fixed final adapter and head hidden trunk; final 300-bin linear layer reinitialised",
        replay_validation_path=str(H1 / "splits/lowz_only_validation.npz"),
        replay_predictions_path=str(H1 / "training/H1-A/final/validation_predictions.pt"),
        presentations=100000,
        head_only_presentations=10240,
        accumulation=16,
        component_weights=[0.5, 0.3, 0.2],
        head_lr=2.0e-4,
        adapter_lr=1.0e-5,
        head_warmup_updates=100,
        checkpoint_interval_presentations=1024,
        validation_interval_presentations=10240,
        loss="CE + 0.5*CRPS on the native 300-bin [0,4.5] grid",
        data="50% synthetic / 30% source low-z / 20% real fitting anchors, averaged within component per 16-presentation window",
        firewall="P, E, high-z V and all development spectra excluded from training",
    )
    cfg_path = OUT / "protocol/configs/main.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(config, indent=2) + "\n")
    else:
        old = json.loads(cfg_path.read_text())
        assert old["presentations"] == config["presentations"] and old["initial_trainable_state_sha256"] == state_sha
        config = old
    out = OUT / "training/main"
    has_checkpoints = any((out / "checkpoints").glob("resume-slot*.pt"))
    resume = bool(args.resume or (not args.fresh and has_checkpoints)) and has_checkpoints
    print(f"START resume={resume} has_checkpoints={has_checkpoints} output={out}", flush=True)
    train(config, out, OUT / "data/training", resume=resume)


if __name__ == "__main__":
    main()
