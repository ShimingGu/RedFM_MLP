#!/usr/bin/env python3
"""Development-only colour and missingness check of synthetic descendants."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
sys.path.insert(0, str(ROOT))


def read(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def colors(features):
    f = np.asarray(features, float)
    flux = f[:, 0:10:2]
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(flux > 0, 22.5 - 2.5 * np.log10(flux), np.nan)
    return np.column_stack([mag[:, i] - mag[:, i + 1] for i in range(4)]), mag


def summary(x, name):
    return dict(name=name, n=int(len(x)), median=[float(v) for v in np.nanmedian(x, axis=0)], p16=[float(v) for v in np.nanpercentile(x, 16, axis=0)], p84=[float(v) for v in np.nanpercentile(x, 84, axis=0)])


def main():
    syn = read(OUT / "data/synthetic/synthetic.npz")
    paired = read("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1028/matched-training-design/paired_split/paired_cohort.npz")
    vrows = np.flatnonzero((paired["category"] == "V") & (paired["z"] < 0.6))
    vc, vm = colors(paired["hsc_features"][vrows])
    low = syn["z"] < 0.6
    sc, sm = colors(syn["features"][low])
    anchors = read(OUT / "data/frozen/anchors.npz")
    dev = anchors["role"] == "development"
    dc, dm = colors(anchors["features"][dev])
    high = (syn["z"] >= 0.8) & (syn["z"] < 1.5)
    hc, hm = colors(syn["features"][high])
    result = dict(
        lowz=dict(real=summary(vc, "V_lowz"), synthetic=summary(sc, "synthetic_z_lt_0.6"), median_offset=[float(a) for a in (np.nanmedian(sc, 0) - np.nanmedian(vc, 0))], missing_fraction=float(1 - np.isfinite(syn["features"][low][:, 0:10:2]).mean())),
        highz=dict(development_anchors=summary(dc, "dev_anchors"), synthetic=summary(hc, "synthetic_0.8_1.5"), median_offset=[float(a) for a in (np.nanmedian(hc, 0) - np.nanmedian(dc, 0))], missing_fraction=float(1 - np.isfinite(syn["features"][high][:, 0:10:2]).mean())),
        note="low-z V and the 64 budgeted development anchors are development photometry; only aggregate summaries are reported; no P/E data",
    )
    (OUT / "protocol/checks_synthetic_colors.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
