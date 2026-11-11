#!/usr/bin/env python3
"""Gated dev-only checkpoint selection for the 1110 main run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    status = json.loads((OUT / "training/main/status.json").read_text())
    if not args.execute:
        print(json.dumps(dict(stage="gated", training=status.get("stage"), message="dev selection is implemented but not executed; pass --execute after training completes"), indent=2))
        return
    if status.get("stage") != "complete":
        raise SystemExit(f"Training is not complete: {status.get('stage')}")
    files = sorted((OUT / "training/main").glob("validation_*.json"))
    if not files:
        raise FileNotFoundError("No development validation files")
    candidates = []
    for path in files:
        value = json.loads(path.read_text())
        by_group = value.get("by_group", {})
        by_stratum = value.get("by_stratum", {})
        low = by_group.get("lowz_V", {})
        macro = []
        for name in ("[0.8,1.0)", "[1.0,1.5)"):
            cell = by_stratum.get(name)
            if cell and cell.get("n", 0) >= 8:
                macro.append(cell["mae"])
        macro_mae = float(np.mean(macro)) if macro else float("inf")
        macro_crps = float(np.mean([by_stratum[n]["crps"] for n in ("[0.8,1.0)", "[1.0,1.5)") if by_stratum.get(n, {}).get("n", 0) >= 8])) if macro else float("inf")
        penalty = 0.0
        if low:
            penalty += 2.0 * max(0.0, low.get("mae", 0.0) - 0.035)
            penalty += 2.0 * max(0.0, abs(low.get("bias", 0.0)) - 0.01)
            penalty += 1.0 * max(0.0, low.get("nmad", 0.0) - 0.040)
            penalty += 1.0 * max(0.0, low.get("outlier", 0.0) - 0.012)
            penalty += 1.0 * max(0.0, low.get("crps", 0.0) - 0.032)
        candidates.append(dict(
            position=int(value["position"]), path=str(path), validation_sha256=sha(path),
            macro_mae_primary=macro_mae, macro_crps_primary=macro_crps,
            lowz_V=low, dev_by_stratum=by_stratum, live_selection_score=float(value.get("selection_score", np.nan)),
            corrected_selection_score=float(macro_mae + penalty), selection_score=float(macro_mae + penalty),
        ))
    candidates.sort(key=lambda c: (c["corrected_selection_score"], c["macro_crps_primary"], c["position"]))
    best = candidates[0]
    checkpoint = None
    for path in sorted((OUT / "training/main/checkpoints").glob("resume-slot*.pt")):
        proof = path.with_suffix(".sha256")
        if not proof.exists() or sha(path) != proof.read_text().strip():
            continue
        import torch
        value = torch.load(path, map_location="cpu", weights_only=False)
        if int(value.get("position", -1)) == best["position"]:
            checkpoint = (path, value)
            break
    if checkpoint is None:
        raise FileNotFoundError(f"No verified checkpoint at selected position {best['position']}")
    destination = OUT / "training/main/selected_frozen"
    destination.mkdir(parents=True, exist_ok=True)
    import torch
    tmp_state = destination / "trainable_state.pt.partial"
    torch.save(checkpoint[1]["trainable_state"], tmp_state)
    tmp_state.replace(destination / "trainable_state.pt")
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        selected_position=best["position"],
        selected_checkpoint=str(checkpoint[0]),
        selected_checkpoint_sha256=sha(checkpoint[0]),
        selected_state_sha256=sha(destination / "trainable_state.pt"),
        candidates=candidates,
        rule="minimum macro MAE over [0.8,1.0) and [1.0,1.5) on 64 development anchors, subject to all predeclared low-z V safeguards (MAE<=.035, |bias|<=.01, NMAD<=.040, outlier<=.012, CRPS<=.032); penalties 2/2/1/1/1; CRPS tie-breaker; no P/E. The live training score omitted the outlier and CRPS terms; this corrected post-training selector is used.",
        dev_only=True,
    )
    write(OUT / "evaluation/model_selection.json", result)
    write(OUT / "status/model_selection.json", dict(utc=datetime.now(timezone.utc).isoformat(), stage="complete", selected_position=best["position"], state=str(destination / "trainable_state.pt")))
    print(json.dumps(dict(selected_position=best["position"], score=best["selection_score"], state=result["selected_state_sha256"]), indent=2))


if __name__ == "__main__":
    main()
