# aion-magnitude v0.5.0

Updated: 2026-07-23

This directory is a lightweight code/documentation snapshot for the current
AION + all-magnitude fusion workflow after the M-adapter feasibility round and
before implementing AION partial fine-tuning experiments.

Core package modules include:

- `aion_magnitude.config`, `dataset`, `caching`, `models`, `training`, and `metrics`
- `aion_magnitude.extra_bands` for non-AION photometric features
- `aion_magnitude.ilc` for magnitude-adjustment experiments
- `aion_magnitude.FM_Qwen` for Qwen catalogue embeddings
- `aion_magnitude.morphology` for CLAUDS image-token experiments

## Morphology module

The morphology path uses AION only as a fixed image tokenizer:

```text
CLAUDS u cutout -> AION HSC-G image codec -> FSQ token IDs -> image MLP
photometric features + image-MLP features -> CLAUDS-supervised photo-z head
```

The photometric branch can use either catalogue magnitudes or the frozen grizy
AION magnitude embedding. It does not use the AION image/redshift transformer
embedding. The `HSC-G` name is an AION codec interface alias for the CLAUDS `u`
image, not a claim that the source image is physical HSC g-band data.

Python usage:

```python
from aion_magnitude.morphology import (
    AIONMorphologyConfig,
    cache_aion_morphology_tokens,
    run_morphology_experiment,
)

config = AIONMorphologyConfig(
    max_rows=20_000,
    sample_mode="random",
    image_flux_scale=30.0,
)
product = cache_aion_morphology_tokens(config)
```

After installing the package, the same workflow is available through:

```bash
aion-morphology cache --max-rows 20000 --sample-mode random --image-flux-scale 30
```

`aion` itself is required when image tokens are generated, but it is imported
lazily so catalogue utilities and token-factor models can be used without
loading AION weights.

### Persistent morphology catalogue

`aion_magnitude.morphology_catalogue` builds a reusable FITS catalogue rather
than a photo-z experiment cache. It trains the documented two-layer probe on
the exact `astronolan/galaxy10-aion` benchmark split, calibrates its softmax,
and writes:

- `p_spiral`, `p_bar`, and `p_elliptical_type` from the frozen AION encoder;
- `axis_ellipticity`, `concentration_C`, and `asymmetry_A` from the raw 96x96
  CLAUDS pixels;
- `possible_morphological_mismatch` and `morphology_available` quality flags.

The target AION input is a five-band HSC proxy. Each object's CLAUDS u/uS
cutout supplies the shared spatial morphology, while its catalogue HSC grizy
cmodel fluxes supply the five relative band amplitudes and the ZP 23 to ZP 27
normalization expected by AION's HSC codec. This is more informative than the
single-band tokenizer experiment above, but it is still a proxy rather than
true five-band imaging; the output FITS records that limitation in `HISTORY`.

Run the resumable complete workflow with:

```bash
pixi run python -m aion_magnitude.morphology_catalogue all --device cuda
```

Intermediate embeddings, the trained probe, tile assignments, and per-column
memmaps live under `cache/aion_morphology_catalogue/`. Rows without adequate
image coverage, pixel S/N, or at least three valid HSC fluxes retain NaN
probabilities. The mismatch flag is diagnostic: it marks
`abs(p_elliptical_type - (1 - axis_ellipticity)) >= 0.5`; it does not declare
elongated ellipticals or round face-on spirals erroneous.

### Band-resolved multiband morphology catalogue

`aion_magnitude.multiband_morphology_catalogue` creates a separate catalogue
from the original Phosphoros table, preserving its object IDs and row order.
It reads CLAUDS `u` tiles locally and HSC PDR3 `grizy` patches directly from
`/arc/projects/ots/pdr3_dud/`; it does not copy those HSC images to scratch.
The output is
`data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological_multiband.fits`.

For each suffix `x` in `u`, `g`, `r`, `i`, `z`, and `y`, it adds these 12
columns (72 columns total):

| Column | Definition |
|---|---|
| `p_spiral_x` | Sum of the calibrated Galaxy10 probabilities for barred, tight/loose unbarred, and both edge-on spiral classes. |
| `p_bar_x` | Calibrated probability of the Galaxy10 barred-spiral class. |
| `p_elliptical_type_x` | Sum of the calibrated Galaxy10 probabilities for the three elliptical classes. |
| `axis_ellipticity_x` | `1 - sqrt(lambda_minor/lambda_major)` from positive, background-subtracted 96x96 flux-weighted second moments. |
| `concentration_C_x` | `5 log10(r80/r20)` from positive, background-subtracted 96x96 pixels. |
| `asymmetry_A_x` | Noise-corrected absolute residual from a 180-degree rotation, divided by absolute source flux; signed background-subtracted pixels are retained. |
| `possible_morphological_mismatch_x` | Diagnostic flag for `abs(p_elliptical_type_x - (1 - axis_ellipticity_x)) >= 0.5`; this is not a physical error label. |
| `surface_brightness_24_x` | Signed sum of `raw - local_background` over valid central 24x24 pixels. Despite its historical name, this is integrated aperture flux. |
| `surface_brightness_96_x` | Signed sum of `raw - local_background` over valid full 96x96 pixels; also integrated cutout flux. |
| `mean_per_sqarcsec_12_x` | Unsubtracted raw mean over valid central 12x12 pixels, divided by the WCS pixel area in square arcseconds. |
| `mean_per_sqarcsec_24_x` | Unsubtracted raw mean over valid central 24x24 pixels, divided by the WCS pixel area in square arcseconds. |
| `morphology_available_x` | True only when the band has adequate usable coverage and all probability, pixel-morphology, and brightness values are valid. |

The local background is a sigma-clipped median of valid border pixels. HSC
validity uses the science, mask, and variance planes; negative residuals are
not clipped for integrated brightness or asymmetry. Pixel area is the absolute
WCS projected-plane determinant converted to arcsec2. `mean_per_sqarcsec_*`
intentionally includes the local sky, while `surface_brightness_*` subtracts
it.

The probability probe is trained and temperature-calibrated on Galaxy10 with
one DES band exposed to AION at a time. At catalogue inference, `g`, `r`, `i`,
`z`, and `y` use the corresponding AION HSC channel; the CLAUDS `u` image uses
AION `HSC-G` as the agreed codec proxy. No other image band or catalogue flux
ratio is mixed into a band's probabilities. This remains a cross-survey
transfer: Galaxy10 has no matching `u` or HSC-Y training channel, and the first
catalogue version does not PSF-match bands.

Image values are multiplied by the corresponding `--u-flux-scale` through
`--y-flux-scale` option before measurement (all default to 1). FITS headers
inspected here do not declare `BUNIT`, so the default brightness columns are in
scaled native image units, not a guaranteed common physical unit. Set the six
scale options from an external photometric calibration before quantitative
cross-band brightness comparisons; each selected scale and this warning are
retained in run metadata.

The full workflow is resumable. It stores only a compact WCS manifest,
assignments, the single-band probe, and per-column memmaps under
`cache/aion_multiband_morphology_catalogue/`. It refuses to write the final
FITS file until every assigned row has a terminal status in all six bands, and
writes through an atomic `.partial` file.

Run it later with:

```bash
./scripts/create_multiband_morphology_catalogue.sh all --device cuda
```

Useful staged or development commands include:

```bash
./scripts/create_multiband_morphology_catalogue.sh manifest
./scripts/create_multiband_morphology_catalogue.sh train-head --device cuda
./scripts/create_multiband_morphology_catalogue.sh features --bands u,g --max-target-rows 100 --stop-after-processed-rows 10
./scripts/create_multiband_morphology_catalogue.sh catalogue
```

The row-limited command is only a smoke test and cannot be used to write the
final catalogue.

Intentionally excluded:

- catalogue/data files: `data/`, `provabgs_desi_ls.hdf5`, etc.
- caches/checkpoints/split products: `cache/`, `cache_0704/`, `clauds_split/`
- environment: `aion_env/`
- generated image outputs: `*.jpeg`, `*.jpg`, `*.png`, `*.avif`
- notebook checkpoints and Python bytecode caches

This snapshot is meant for code review, provenance, and handoff. To run it,
use the original workspace data/cache setup or rebuild the required cache from
the catalogue files.

## Single- and multi-GPU evaluation cases

The package includes a scheduler-aware case runner under
`aion_magnitude.evaluation`. Its `auto` strategy follows these rules:

- one worker runs all cases sequentially on the available CPU/GPU;
- several workers distribute independent cases across workers, with one
  visible GPU per worker;
- fewer cases than GPUs leave extra workers idle unless the task explicitly
  declares support for splitting one case across workers.

The included Qwen task currently supports case parallelism. It deliberately
rejects case sharding until Qwen embedding shards can be merged deterministically
by `object_id`.

Inside a Slurm allocation with four GPUs, run the four controlled physical
context cases with:

```bash
export AION_EVAL_WORKERS=4
export AION_MAX_ROWS=20000
export AION_EPOCHS=3
bash scripts/run-qwen-evals-multigpu.sh
```

Alternatively, edit the time/account/partition for the target cluster and
submit `scripts/slurm-qwen-evals.sbatch` directly.

Slurm launches one process per GPU. Each process sees its assigned card as
`cuda:0`, writes an independent JSON artifact, and the parent process writes
`summary.json`. Set `AION_EVAL_OUTPUT_DIR`, `AION_CACHE_ROOT`, `AION_CATALOGUE`,
and `AION_MORPHOLOGY_DIR` to override the cluster paths. Shared photometry and
morphology caches should be built before starting workers concurrently.
`AION_EVAL_WORKERS` is optional: the launcher otherwise uses
`SLURM_GPUS_ON_NODE`, then `CUDA_VISIBLE_DEVICES`, and finally falls back to one
worker. Outside Slurm, multiple visible GPUs use `torchrun` automatically.

The generic CLI can inspect a plan without running a model:

```bash
aion-eval plan \
  --manifest configs/evals/qwen_physical_context.json \
  --worker-count 4
```

Future case-internal multi-GPU tasks receive `case_rank`, `case_world_size`, and
`context.shard_bounds(n_rows)`. Once a task implements deterministic merging,
run it with `--supports-case-sharding --gpus-per-case N`; the cluster launcher
and artifact format do not need to change.

Pydantic Evals reporting is optional and runs over completed JSON artifacts,
not inside GPU workers. Install it with `pip install -e '.[evals]'`; the
launcher then writes `pydantic_report.json` with embedding, optimization, and
photo-z diagnostics. See [`multigpu.md`](multigpu.md) for the complete cluster
handoff, output layout, resume behavior, and the future one-case/multi-GPU
extension contract.
