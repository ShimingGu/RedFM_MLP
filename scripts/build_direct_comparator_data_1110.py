#!/usr/bin/env python3
"""Build the no-augmentation direct limited-anchor comparator dataset (not launched)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
H1 = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1102/highz-hsc-pilot-zcut0p6")
sys.path.insert(0, str(ROOT))

from aion_magnitude.redshift_anchor_physics import serialize_features  # noqa: E402

SEED = 1110


def read(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    low = read(H1 / "splits/lowz_only_train.npz")
    anchors = read(OUT / "data/frozen/anchors.npz")
    fit = anchors["role"] == "fitting"
    val = read(OUT / "data/training/validation.npz")
    n_low = len(low["targetid"])
    n_anchor = int(fit.sum())
    assert n_low == 10489 and n_anchor == 192
    texts = [str(x) for x in low["text"]] + [serialize_features(anchors["features"][i], ROOT) for i in np.flatnonzero(fit)]
    z = np.concatenate([low["z"], anchors["z"][fit]]).astype(np.float32)
    ids = np.concatenate([low["targetid"], anchors["targetid"][fit]]).astype(np.int64)
    comp = np.concatenate([np.zeros(n_low, np.int8), np.ones(n_anchor, np.int8) * 2])
    assert len(texts) == len(z) == len(ids) == len(comp) == n_low + n_anchor

    rng = np.random.default_rng(SEED + 60)
    parts, phases = [], []
    for phase, start in (("head", 0), ("joint", n_low + n_anchor)):
        ns, nl, na = (5120, 3072, 2048) if phase == "head" else (44880, 26928, 17952)
        # Two low-z components deliberately mirror the main loop's 50/30 weighting;
        # both draw from the same permitted 10489 real source objects.
        c0 = rng.choice(n_low, ns, replace=True)
        c1 = rng.choice(n_low, nl, replace=True)
        c2 = rng.choice(n_anchor, na, replace=True) + n_low
        part = np.concatenate([c0, c1, c2]).astype(np.int64)
        part_component = np.concatenate([np.zeros(ns, np.int8), np.ones(nl, np.int8), np.full(na, 2, np.int8)])
        perm = rng.permutation(len(part))
        parts.append(part[perm])
        phases.append(np.full(len(part), 0 if phase == "head" else 1, dtype=np.int8))
        if "components" not in locals():
            components = []
        components.append(part_component[perm])
    order = np.concatenate(parts)
    phase = np.concatenate(phases)
    component = np.concatenate(components)
    assert len(order) == 100000
    data_dir = OUT / "data/training_direct"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_dir / "pool.npz", text=np.array(texts, dtype=str), z=z, object_id=np.array([str(int(x)) for x in ids]), component=comp, parent_targetid=ids)
    np.savez_compressed(data_dir / "sequence.npz", pool_index=order, phase=phase, component=component)
    np.savez_compressed(data_dir / "validation.npz", **{k: val[k] for k in val})
    counts = {p: np.bincount(component[phase == i], minlength=3).tolist() for p, i in (("head", 0), ("joint", 1))}
    meta = dict(version="direct_comparator_1110_v1", created_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), seed=SEED,
                presentations=100000, components=["lowz_source_A", "lowz_source_B", "anchor_fitting"], effective_real_mix="80% low-z source / 20% real fitting anchors",
                phase_counts=counts, pool=dict(lowz=n_low, anchor=n_anchor), no_synthetic=True,
                firewall="only the same 10489 low-z source and 192 fitting anchors; no development/P/E; no spectra beyond the already permitted fitting sets",
                hashes={n: sha(data_dir / n) for n in ("pool.npz", "sequence.npz", "validation.npz")})
    (data_dir / "training_data.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
