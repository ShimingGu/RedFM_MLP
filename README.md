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
