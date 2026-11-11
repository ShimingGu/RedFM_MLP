#!/usr/bin/env python3
"""Generate physical descendants from permitted fits and freeze their lineage."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
sys.path.insert(0, str(ROOT))

from aion_magnitude.redshift_anchor_fit import generate_descendants  # noqa: E402
from aion_magnitude.redshift_anchor_physics import CModelNoise, load_hsc_filters  # noqa: E402


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def main():
    started = time.time()
    basis = read(OUT / "data/physics/fsps_basis.npz")
    filters = load_hsc_filters(ROOT)
    paired = read("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1028/matched-training-design/paired_split/paired_cohort.npz")
    trows = np.flatnonzero(paired["category"] == "T")
    noise = CModelNoise(paired["hsc_features"][trows], paired["hsc_valid"][trows])
    ebv_pool = paired["hsc_features"][trows, 10].astype(np.float64)

    lowz_files = sorted((OUT / "data/fits/lowz").glob("*.npz"))
    anchor_files = sorted((OUT / "data/fits/anchor").glob("*.npz"))
    files = [(p, "lowz") for p in lowz_files] + [(p, "anchor") for p in anchor_files]
    assert len(lowz_files) >= 2048, len(lowz_files)
    assert len(anchor_files) >= 192, len(anchor_files)
    parent_ids = np.array([int(read(p)["targetid"]) for p, _ in files], dtype=np.int64)
    roles = np.array([r for _, r in files])
    z = np.array([float(read(p)["z"]) for p, _ in files])
    a_tot = np.array([float(read(p)["a_tot"]) for p, _ in files])
    w_draws = np.stack([read(p)["w_draws"] for p, _ in files])
    print("parents", len(files), "lowz", len(lowz_files), "anchor", len(anchor_files), flush=True)

    # A dev-anchor sanity check: no development identity may be a parent.
    anchors = read(OUT / "data/frozen/anchors.npz")
    dev = set(int(x) for x in anchors["targetid"][anchors["role"] == "development"])
    assert not (set(parent_ids.tolist()) & dev)
    assert not any(r not in ("lowz", "anchor") for r in roles)

    rng = np.random.default_rng(1110)
    rows = generate_descendants(
        basis=basis,
        filters=filters,
        noise=noise,
        fits=None,
        parent_ids=parent_ids,
        parent_roles=roles,
        parent_z=z,
        parent_a_tot=a_tot,
        parent_w_draws=w_draws,
        parent_n=len(files),
        ebv_pool=ebv_pool,
        rng=rng,
        repo_root=ROOT,
        descendants_per_role={"lowz": 15, "anchor": 150},
    )
    if not rows:
        raise SystemExit("No synthetic rows generated")
    data = {}
    for key in rows[0]:
        data[key] = np.asarray([row[key] for row in rows])
    path = OUT / "data/synthetic/synthetic.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    zbins = np.array([0.0, 0.4, 0.6, 0.8, 1.0, 1.5, 1.7, 4.5])
    counts = np.histogram(data["z"], bins=zbins)[0]
    parent_counts = {r: int((data["parent_role"] == r).sum()) for r in ("lowz", "anchor")}
    unique_parents = len(np.unique(data["targetid"]))
    meta = dict(
        created_utc=now(),
        path=str(path),
        n=int(len(data["z"])),
        parents=int(len(files)),
        parent_counts=parent_counts,
        unique_parents=int(unique_parents),
        z_bins=zbins.tolist(),
        z_counts=counts.tolist(),
        proposal="parent-local transport: low-z 90% uniform parent_z+[-0.05,0.12] in [0.4,0.85], 10% broad [0.4,0.85]; anchors uniform parent_z+[-0.08,0.15] clipped [0.6,1.7]; 15 descendants per low-z parent and 150 per anchor",
        proposal_density_recorded=True,
        lineage="parent targetid, parent role, parent redshift, SED draw, luminosity offset, Av offset and EBV are stored per row",
        age_rule="templates older than 0.95 age_universe(z_gen) are zeroed for that descendant",
        amplitude="parent fitted a_tot times lognormal luminosity offset (0.25 dex) times luminosity-distance ratio; no deterministic flux-redshift lookup",
        measurement="T-constrained cModel magnitude error relation with faint-end photon-noise extrapolation and explicit logistic dropout; at least three valid bands",
        extinction="Milky-Way F99 applied at observed wavelengths; intrinsic F99 dust is in the FSPS templates plus per-descendant Av jitter",
        no_evaluation_data="P/E/high-z V and development spectra were never read; only source-training and anchor-fitting identities are parents",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    (OUT / "data/synthetic/synthetic.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    print("elapsed", time.time() - started, flush=True)


if __name__ == "__main__":
    main()
