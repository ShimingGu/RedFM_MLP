# v0.2 snapshot

Date: 2026-07-07

This directory is a lightweight code/documentation snapshot for the current
AION + all-magnitude fusion workflow after the M-adapter feasibility round and
before implementing AION partial fine-tuning experiments.

Included:

- `aion_magnitude.py`
- `aion_extra_bands.py`
- `clauds_bands.py`
- `aion_u_band.py`
- `magnitude_ILC.py`
- `agy_purpose_analysis.md`
- `notebooks/aion_mlp_test.ipynb`
- `notebooks/clauds_aion_ILC.ipynb`
- `logs/aion_main_module.md`
- `logs/aion_u_band.md`
- `logs/all_mag_fusion.md`

Intentionally excluded:

- catalogue/data files: `data/`, `provabgs_desi_ls.hdf5`, etc.
- caches/checkpoints/split products: `cache/`, `cache_0704/`, `clauds_split/`
- environment: `aion_env/`
- generated image outputs: `*.jpeg`, `*.jpg`, `*.png`, `*.avif`
- notebook checkpoints and Python bytecode caches

This snapshot is meant for code review, provenance, and handoff. To run it,
use the original workspace data/cache setup or rebuild the required cache from
the catalogue files.
