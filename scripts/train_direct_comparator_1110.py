#!/usr/bin/env python3
"""Gated direct limited-anchor QLoRA comparator (no spectral augmentation)."""
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
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not args.execute:
        print(json.dumps(dict(stage="gated", message="direct comparator code is ready but not launched; requires route_decision.json authorisation after main-route decision"), indent=2))
        return
    route = OUT / "evaluation/route_decision.json"
    if not route.exists() or not json.loads(route.read_text()).get("direct_comparator_authorized"):
        raise SystemExit("Direct comparator not authorised by evaluation/route_decision.json")
    h1cfg = json.loads((H1 / "protocol/configs/H1-A.json").read_text())
    state = H1 / "training/H1-A/final/trainable_state.pt"
    import hashlib
    state_sha = hashlib.sha256(state.read_bytes()).hexdigest()
    training = dict(h1cfg["training"], z_min=0.0, z_max=4.5, n_z_bins=300, seed=1110, device="cuda")
    config = dict(
        name="RA-1110-direct-limited-anchor", seed=1110, training=training,
        initial_trainable_state=str(state), initial_trainable_state_sha256=state_sha,
        initialisation="H1-A fixed final adapter and head trunk; final 300-bin layer reinitialised",
        replay_validation_path=str(H1 / "splits/lowz_only_validation.npz"),
        replay_predictions_path=str(H1 / "training/H1-A/final/validation_predictions.pt"),
        presentations=100000, head_only_presentations=10240, accumulation=16,
        component_weights=[0.5, 0.3, 0.2], head_lr=2.0e-4, adapter_lr=1.0e-5,
        head_warmup_updates=100, checkpoint_interval_presentations=1024,
        validation_interval_presentations=10240,
        loss="CE + 0.5*CRPS on native 300-bin grid",
        data="80% real low-z source / 20% real fitting anchors, no synthetic augmentation",
        firewall="same real fitting objects and initialisation route as main; no development/P/E",
    )
    path = OUT / "protocol/configs/direct_comparator.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(config, indent=2) + "\n")
    out = OUT / "training/direct"
    resume = args.resume or any((out / "checkpoints").glob("resume-slot*.pt"))
    train(config, out, OUT / "data/training_direct", resume=resume)


if __name__ == "__main__":
    main()
