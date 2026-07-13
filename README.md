# aion-magnitude v0.2.1

Updated: 2026-07-13

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
CLAUDS u cutout -> AION HSC-G image codec -> FSQ token IDs
non-AION magnitudes + token factors -> CLAUDS-supervised photo-z MLP
```

It does not use the AION image/redshift transformer embedding. The `HSC-G`
name is an AION codec interface alias for the CLAUDS `u` image, not a claim
that the source image is physical HSC g-band data.

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

Intentionally excluded:

- catalogue/data files: `data/`, `provabgs_desi_ls.hdf5`, etc.
- caches/checkpoints/split products: `cache/`, `cache_0704/`, `clauds_split/`
- environment: `aion_env/`
- generated image outputs: `*.jpeg`, `*.jpg`, `*.png`, `*.avif`
- notebook checkpoints and Python bytecode caches

This snapshot is meant for code review, provenance, and handoff. To run it,
use the original workspace data/cache setup or rebuild the required cache from
the catalogue files.
