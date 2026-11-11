#!/usr/bin/env python3
"""Reconstruct and freeze the independent-galaxy E membership (no labels used).

The recipe is fixed by redshift_generalisation_execution_1110.md: start from
parent rows with independent & ~original_or_matched_train_val_exposure, then
remove every paired T/V/P identity, matching HSC group, and any parent HSC or
DESI-target position within one arcsecond of a paired HSC or Legacy position.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ARC = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints")
PAIRED = ARC / "1028/matched-training-design/paired_split/paired_cohort.npz"
PARENT = ARC / "1102/highz-feasibility/raw/hsc_parent/cohort.npz"
FLAGS = ARC / "1102/highz-feasibility/flags/hsc_parent.npz"
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
BINS = np.array([0.0, 0.6, 0.8, 1.0, 1.5, 2.0, 4.5])
EXPECTED = np.array([19758, 1522, 536, 311, 12, 0])
ARCSEC = 1.0 / 3600.0


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def unit(ra, dec):
    ra = np.deg2rad(np.asarray(ra, float))
    dec = np.deg2rad(np.asarray(dec, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def read(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def main():
    paired = read(PAIRED)
    parent = read(PARENT)
    flags = read(FLAGS)
    assert {c: int((paired["category"] == c).sum()) for c in "TVP"} == dict(T=16600, V=4257, P=61696)
    assert np.array_equal(parent["targetid"], flags["targetid"])
    base = flags["independent"] & ~flags["original_or_matched_train_val_exposure"]
    print("base", int(base.sum()), flush=True)

    # Exact identities and HSC groups.
    paired_ids = set(int(x) for x in paired["targetid"])
    paired_groups = set(str(x) for x in paired["galaxy_id"])
    id_excluded = np.array([int(x) in paired_ids for x in parent["targetid"]])
    group_excluded = np.array([str(x) in paired_groups for x in parent["group"]])
    print("id/group excluded", int(id_excluded.sum()), int(group_excluded.sum()), flush=True)

    trees = []
    for ra, dec in (
        (paired["ra"], paired["dec"]),
        (paired["legacy_ra"], paired["legacy_dec"]),
    ):
        good = np.isfinite(ra) & np.isfinite(dec)
        trees.append(cKDTree(unit(ra[good], dec[good])))
    chord = 2.0 * np.sin(np.deg2rad(ARCSEC) / 2.0)
    pos_excluded = np.zeros(len(parent["targetid"]), bool)
    for ra, dec in ((parent["ra"], parent["dec"]), (parent["target_ra"], parent["target_dec"])):
        xyz = unit(ra, dec)
        for tree in trees:
            dist, _ = tree.query(xyz, k=1, distance_upper_bound=chord * 1.000001)
            pos_excluded |= np.isfinite(dist)
    print("position excluded", int(pos_excluded.sum()), flush=True)

    keep = base & ~id_excluded & ~group_excluded & ~pos_excluded
    z = parent["z"].astype(np.float64)
    counts = np.histogram(z[keep], bins=BINS)[0]
    print("E", int(keep.sum()), counts.tolist(), flush=True)
    if not np.array_equal(counts, EXPECTED):
        raise SystemExit(f"E count mismatch: {counts.tolist()} vs {EXPECTED.tolist()}; investigate provenance")

    rows = np.flatnonzero(keep)
    out = dict(
        targetid=parent["targetid"][rows],
        group=parent["group"][rows],
        features=parent["features"][rows],
        ra=parent["ra"][rows],
        dec=parent["dec"][rows],
        target_ra=parent["target_ra"][rows],
        target_dec=parent["target_dec"][rows],
        z=parent["z"][rows],
        zerr=parent["zerr"][rows],
        z_exact=parent["z_exact"][rows],
        parent_row=rows,
        strata=np.digitize(z[rows], BINS[1:-1]),
    )
    path = OUT / "data/frozen/E_membership.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    meta = dict(
        recipe="independent & ~original_or_matched_train_val_exposure; exclude paired T/V/P targetid, paired galaxy_id group, and parent HSC/DESI-target positions within 1 arcsec of paired HSC or Legacy positions",
        parent_rows=len(parent["targetid"]),
        base_rows=int(base.sum()),
        exact_id_excluded=int(id_excluded.sum()),
        group_excluded=int(group_excluded.sum()),
        position_excluded=int(pos_excluded.sum()),
        n=int(keep.sum()),
        bins=BINS.tolist(),
        counts=counts.tolist(),
        expected_counts=EXPECTED.tolist(),
        feature_columns=[f"HSC_PDR2_cmodel_{b}_{s}" for b in "grizy" for s in ("flux", "flux_error")]
        + ["MW_EBV"],
        membership_sha256=sha(out["targetid"]),
        targetid_sha256=sha(out["targetid"]),
        features_sha256=sha(out["features"]),
        z_sha256=sha(out["z"]),
        label_access="sealed behind final evaluator; constructed before model development",
        note="E includes the parent catalogue's broader DESI survey selection; no historical paired Legacy restrictions were added.",
    )
    (OUT / "protocol/ledgers").mkdir(parents=True, exist_ok=True)
    (OUT / "protocol/ledgers/E_membership.json").write_text(json.dumps(meta, indent=2) + "\n")
    np.savetxt(OUT / "protocol/ledgers/E_strata.csv", np.column_stack([BINS[:-1], BINS[1:], counts]), delimiter=",", header="z_low,z_high,count", comments="", fmt=["%.2f", "%.2f", "%d"])
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
