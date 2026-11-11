#!/usr/bin/env python3
"""Physical template-likelihood comparator on development photometry only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
H1 = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1102/highz-hsc-pilot-zcut0p6")
sys.path.insert(0, str(ROOT))

from aion_magnitude.redshift_anchor_comparator import summarize_pdf, template_pdf  # noqa: E402
from aion_magnitude.redshift_anchor_physics import load_hsc_filters  # noqa: E402


def read(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def main():
    cache = read(OUT / "data/physics/template_band_flux.npz")
    filters = load_hsc_filters(ROOT)
    pivots = np.array([filters[b]["pivot_nm"] * 10.0 for b in "grizy"])
    dev = read(OUT / "data/frozen/anchors.npz")
    devm = dev["role"] == "development"
    lowv = read(H1 / "splits/lowz_only_validation.npz")
    paired = read("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1028/matched-training-design/paired_split/paired_cohort.npz")
    feat_of = {int(t): paired["hsc_features"][i] for i, t in enumerate(paired["targetid"])}
    rng = np.random.default_rng(1110)
    take = np.sort(rng.choice(len(lowv["z"]), size=min(1000, len(lowv["z"])), replace=False))
    rows = []
    for i in np.flatnonzero(devm):
        rows.append((str(int(dev["targetid"][i])), float(dev["z"][i]), dev["features"][i], "dev_anchor"))
    for i in take:
        rows.append((str(int(lowv["targetid"][i])), float(lowv["z"][i]), feat_of[int(lowv["targetid"][i])], "lowz_V"))
    out = []
    for n, (tid, z, features, group) in enumerate(rows):
        p = template_pdf(features, cache, cache["z_grid"].astype(float), pivots=pivots)
        s = summarize_pdf(p, cache["z_grid"].astype(float), z)
        if s is None:
            continue
        out.append(dict(targetid=tid, z=z, group=group, **s))
    path = OUT / "evaluation/comparator_dev.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    for group in ("dev_anchor", "lowz_V"):
        vals = [r for r in out if r["group"] == group]
        if vals:
            summary[group] = dict(n=len(vals), mae=float(np.mean(np.abs([r["delta"] for r in vals]))), bias=float(np.mean([r["delta"] for r in vals])), outlier=float(np.mean(np.abs([r["delta"] for r in vals]) > 0.15)), crps=float(np.mean([r["crps"] for r in vals])), coverage68=float(np.mean([r["coverage68"] for r in vals])))
    result = dict(summary=summary, rows=out, note="development photometry only; no P/E data; template comparator for reference")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
