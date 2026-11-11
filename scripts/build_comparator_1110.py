#!/usr/bin/env python3
"""Cache FSPS template band fluxes on the native 300-bin redshift grid."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
sys.path.insert(0, str(ROOT))

from aion_magnitude.redshift_anchor_physics import band_flux_cache, load_hsc_filters  # noqa: E402
from aion_magnitude.utils import make_redshift_grid  # noqa: E402


def main():
    basis = np.load(OUT / "data/physics/fsps_basis.npz", allow_pickle=False)
    filters = load_hsc_filters(ROOT)
    edges, centers = make_redshift_grid(0.0, 4.5, 300)
    z_grid = centers.numpy().astype(np.float64)
    started = time.time()
    cache = band_flux_cache(basis["wave"], basis["flux"].astype(np.float64), z_grid, filters)
    path = OUT / "data/physics/template_band_flux.npz"
    np.savez_compressed(
        path,
        z_grid=z_grid.astype(np.float32),
        band_flux=cache,
        age_gyr=basis["age_gyr"],
        metallicity=basis["metallicity"],
        av=basis["av"],
        nebular=basis["nebular"],
        normalization_scale=basis["normalization_scale"],
    )
    meta = dict(
        created_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        n_templates=int(cache.shape[0]), n_z=int(cache.shape[1]), bands=list("grizy"),
        z_min=float(z_grid[0]), z_max=float(z_grid[-1]), elapsed_seconds=time.time() - started,
        normalization="median rest-frame 4000-6000 A luminosity set to unity; amplitude is marginalised analytically at evaluation",
        use="physical template comparator only; not a training input",
        sha256=__import__("hashlib").sha256(path.read_bytes()).hexdigest(),
    )
    (OUT / "data/physics/template_band_flux.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
