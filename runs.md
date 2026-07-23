# Run guide

This file describes what the launchers in `scripts/` actually test. It is a
guide to the current code, not an interpretation of the filenames. In
particular, `full`, `raw`, `image`, and `comparison` have accumulated historical
meanings that are not always obvious from their names.

## How to read these experiments

Most scientific launchers build a matched pair of photo-z models. Unless a
section says otherwise, the two arms share the selected galaxies, random split,
target, redshift bins, training budget, evaluation code, and plotting code.

The important distinctions are:

- **What enters the frozen language model?** Magnitudes may be serialized as
  terse values or as text with physical context. Image tokens normally bypass
  Qwen; only one current launcher serializes them into Qwen's text input.
- **What is frozen?** A frozen Qwen/IoTFM/AION/timm encoder is only a feature
  extractor. Training a photo-z head on its cached embeddings is not language
  model fine-tuning.
- **What is trainable?** The photo-z head, tabular MLP, and downstream AION-token
  image MLP are normally trained. Only the QLoRA post-training run updates Qwen
  adapter parameters.
- **What does “AION image” mean?** In the morphology runs, AION is used as an
  image codec. Its FSQ token IDs or decoded token factors feed a separately
  trained image path. AION's own image-to-redshift embedding is not used.
- **What does a row limit mean?** Qwen/IoTFM short runs use seeded random row
  sampling, not the first rows of the FITS catalogue.

The usual target is catalogue `ZPHOT`. Most classification-style runs predict a
300-bin redshift PDF and report point, PDF, PIT, n(z), and tomographic
diagnostics. Environment variables shown below are defaults; extra command-line
arguments are forwarded to the Python entry point by most launchers.

## Quick map

| Launcher | Main question | Magnitudes | Image path | Trainable encoder? |
| --- | --- | --- | --- | --- |
| `standard_comparison.sh` | Frozen AION magnitude features or a direct MLP? | HSC grizy | None | No |
| `standard_full_comparison.sh` | How should grizy be represented when extra bands are held fixed? | grizy + u/u*/Y/J/H/Ks | None | No |
| `image-on_comparison_aion.sh` | Do u-image tokens improve frozen AION grizy features? | HSC grizy | Downstream AION FSQ image MLP | Image MLP only |
| `image-on_comparison_mlp.sh` | Do u-image tokens improve a grizy MLP? | HSC grizy | Downstream AION FSQ image MLP | Image MLP only |
| `image-on_comparison_all_mlp.sh` | Do u-image tokens improve an all-magnitude MLP? | All 11 bands | Downstream AION FSQ image MLP | Image MLP only |
| `qwen-aion_comparison.sh` | Frozen Qwen or frozen AION for grizy? | HSC grizy | None | No |
| `qwen-mlp_full_comparison.sh` | Frozen Qwen or MLP for magnitudes, with the same image path? | All 11 bands, terse Qwen text | Same downstream AION FSQ image MLP in both arms | Image/heads only |
| `qwen-mlp_full_raw_comparison.sh` | The preceding experiment with fixed run-size/context defaults | Same as preceding row | Same as preceding row | Same as preceding row |
| `qwen-mlp_full_image_comparison.sh` | What happens when image token IDs and physical magnitude text enter Qwen? | All 11 bands, physical text | Inside Qwen in one arm; downstream image MLP in control | No Qwen updates |
| `qwen-qwen_comparison.sh` | Does physical magnitude wording help frozen Qwen? | Physical text versus terse text | None | No |
| `qwen_aion-timm_raw_comparison.sh` | AION tokens or a conventional timm image embedding downstream of the same Qwen magnitudes? | Same terse Qwen embedding | AION FSQ versus frozen timm | AION image MLP/heads only |
| `qwen-qwen_posttraining_comparison_multigpu.sh` | Does QLoRA photo-z post-training beat a frozen Qwen representation? | Same terse text | None | QLoRA arm only |
| `iotfm_mlp*.sh` | Frozen inference-optimized transformer or matched tabular MLP? | Selected catalogue columns | None | No transformer updates |
| `table_models/tabxx_*.sh` | How do TabPFN, TabFM, or TabICL complete masked photo-z values under five matched feature ablations? | 11 magnitudes or the 121-column photometry table | Optional AION token IDs or frozen timm embeddings | Standard MLP image path only where stated |

## The three similarly named Qwen–MLP runs

These are easiest to understand as data-flow definitions:

| Launcher | Qwen receives | Qwen arm after Qwen | Control arm | Actual distinction |
| --- | --- | --- | --- | --- |
| `qwen-mlp_full_comparison.sh` | Terse/raw all-band magnitudes | Qwen magnitude embedding **plus a downstream AION-token image MLP** | All-band magnitude MLP **plus the same image MLP** | Qwen versus MLP magnitude representation; image path held constant |
| `qwen-mlp_full_raw_comparison.sh` | The same terse/raw all-band magnitudes | Same | Same | A wrapper that fixes 200k rows, length 2048, batch size 1, and last pooling; it is not currently a different architecture |
| `qwen-mlp_full_image_comparison.sh` | Physical all-band magnitude text **and serialized AION image token IDs** | Qwen embedding and photo-z head; no second image encoder | All-band magnitude MLP plus downstream AION-token image MLP | Moves image information inside Qwen and, by default, also changes raw magnitude text to physical-context text |

Consequently, the `raw` wrapper is presently a run preset for
`qwen-mlp_full_comparison.sh`. It should not be interpreted as a clean raw-text
ablation against the base launcher. Likewise, `full_image` is not just the base
run with “more image”: it changes both where image information is processed and
the default magnitude serialization. If `QWEN_PHYSICAL_CONTEXT=0` is supplied to
`qwen-mlp_full_image_comparison.sh`, the shell delegates to the raw wrapper; it
does not retain the image-inside-Qwen architecture with physical prose removed.

## Standard magnitude baselines

### `scripts/standard_comparison.sh`

**Question:** Is a frozen AION representation of HSC grizy magnitudes more useful
than feeding those same magnitudes directly to a trainable MLP?

- Arm 1: grizy tabular MLP and photo-z PDF head.
- Arm 2: frozen AION grizy magnitude embedding and photo-z PDF head.
- No image input and no extra u/u*/Y/J/H/Ks bands.
- Default run: full catalogue, 10 epochs, 300 redshift bins, z range 0–6,
  train/test/validation fractions 0.20/0.75/0.05.
- Entry point: `notebooks/aion_mlp_test.py --mode standard-comparison`.

### `scripts/standard_full_comparison.sh`

**Question:** With the six extra bands held fixed, should grizy go through frozen
AION or through the tabular MLP?

- Arm 1: frozen AION grizy embedding fused with an MLP over u, u*, Y, J, H, Ks.
- Arm 2: one tabular MLP over grizy plus the same six extra bands.
- No image input.
- Defaults and split fractions match `standard_comparison.sh`.
- “Full” means the added six bands, not image data or the complete FITS schema.

## Image-value tests using the AION codec

All three launchers use CLAUDS u-band cutouts. The AION codec produces FSQ image
tokens; decoded token factors enter a trainable downstream image MLP. The tokens
do not enter a language model, and AION's image-to-redshift model is not used.
Defaults are the full eligible catalogue, 10 epochs, 300 z bins, z range 0–6,
token batch size 64, train batch size 256, and evaluation batch size 512.

### `scripts/image-on_comparison_aion.sh`

- Control: frozen AION grizy magnitude embedding plus photo-z head.
- Image arm: the same frozen magnitude embedding plus the u-image token MLP.
- This isolates the incremental value of morphology on top of an AION magnitude
  representation.

### `scripts/image-on_comparison_mlp.sh`

- Control: grizy magnitude MLP plus photo-z head.
- Image arm: the same grizy MLP plus the u-image token MLP.
- This isolates the incremental value of morphology on top of a conventional
  magnitude MLP.

### `scripts/image-on_comparison_all_mlp.sh`

- Control: all-magnitude MLP over grizy + u/u*/Y/J/H/Ks.
- Image arm: the same all-magnitude MLP plus the u-image token MLP.
- `--preserve-photometry-splits` keeps the established magnitude cohort/splits
  aligned when adding the image requirement.

## Qwen representation experiments

Unless stated otherwise, Qwen is loaded frozen, commonly in 4-bit form, and its
last non-padding token is pooled. Only the downstream photo-z components train.
The launchers use right padding and default to last pooling.

### `scripts/qwen-aion_comparison.sh`

**Question:** For the same grizy-only input, is a frozen Qwen representation or a
frozen AION magnitude representation better for the same photo-z task?

- Arm 1: grizy serialized into frozen Qwen, then a photo-z PDF head.
- Arm 2: grizy passed through frozen AION, then the matched head.
- No image features and no extra bands.
- Defaults: full catalogue, 10 epochs, `Qwen3-8B-Base`, 4-bit Qwen,
  maximum text length 256, last pooling.

### `scripts/qwen-mlp_full_comparison.sh`

**Question:** With an identical morphology path in both arms, should all-band
magnitudes be represented by frozen Qwen or by an MLP?

- Qwen arm: u/u*/g/r/i/z/y/Y/J/H/Ks magnitudes are serialized as terse raw
  values (`galaxy all_magnitudes_ab`) and embedded by frozen Qwen.
- MLP arm: the same magnitudes enter a trainable tabular MLP.
- Both arms additionally receive the same downstream trainable u-image encoder
  built from AION FSQ tokens. The image tokens never enter Qwen.
- Faint-end magnitude cuts and redshift-range cuts are disabled in this
  comparison. A row cap, when supplied, is a seeded random sample.
- Defaults: full catalogue, 10 epochs, `Qwen3-8B-Base`, 4-bit Qwen, maximum
  length 256, last pooling, min-max tabular feature scaling.

### `scripts/qwen-mlp_full_raw_comparison.sh`

This is a thin preset around `qwen-mlp_full_comparison.sh`, not an independent
model definition.

- It keeps the same two arms and the same terse/raw Qwen serialization.
- It defaults to 200,000 randomly sampled rows, Qwen embedding batch size 1,
  maximum length 2048, last pooling, and a separate output directory.
- Because the parent run is already terse/raw in current code, compare results
  only after checking that row count, context length, cache, model, and seed are
  actually matched.

### `scripts/qwen-mlp_full_image_comparison.sh`

**Question:** Can frozen Qwen use serialized AION morphology tokens together
with physically explained magnitudes, relative to the established MLP plus
downstream image-token control?

- Qwen arm: all-band magnitudes are described with physical context; AION FSQ
  token IDs from the u image are serialized into the same Qwen text record.
  The resulting Qwen embedding feeds a photo-z head without a second image MLP.
- Control arm: all-band magnitude MLP plus the standard downstream trainable
  AION-token image encoder.
- Default image serialization is a 16×16 center crop of the token grid. The
  prompt code performs preflight/truncation checks against the 2048-token
  context.
- Defaults: 200,000 random rows, 10 epochs, `Qwen3-8B-Base`, 4-bit Qwen,
  embedding batch size 1, maximum length 2048, last pooling.
- This is not a single-variable test against the raw/base launcher because the
  prompt semantics and image placement both change.

### `scripts/qwen-qwen_comparison.sh`

**Question:** Does adding physical meaning to magnitude text improve a frozen
Qwen photo-z representation?

- Arm 1: physical-context magnitude serialization.
- Arm 2: terse/raw magnitude serialization.
- Both use the same frozen Qwen model, pooling, selected rows, splits, and
  photo-z head design.
- No image representation enters either Qwen arm or either downstream head.
- Defaults: full catalogue, 10 epochs, `Qwen3-8B-Base`, 4-bit Qwen, maximum
  length 2048, last pooling, seed 42.
- The program supports staged preparation and separate physical/terse embedding
  extraction so the expensive caches can be generated independently.

### `scripts/qwen-qwen_comparison_short.sh`

This is the same scientific test as `qwen-qwen_comparison.sh`, with exactly
300,000 randomly selected galaxies and an isolated output directory. It is not
the top or first 300,000 catalogue rows.

### `scripts/qwen-qwen_comparison_short_mutigpu.sh`

This is the same 300,000-row test, despite the historical `mutigpu` typo in its
filename.

- It first prepares one shared cohort and split.
- It extracts the physical and terse Qwen caches concurrently, one per GPU.
- It then trains/evaluates the paired heads from those caches on the first GPU.
- Default device list is `AION_QWEN_GPU_DEVICES`, then
  `CUDA_VISIBLE_DEVICES`, then `0,1`; exactly two IDs are required.
- Multi-GPU execution changes scheduling, not the experiment.

### `scripts/run-qwen-evals-multigpu.sh`

This is a manifest-driven physical-context sweep, not the same fixed pair as a
single `qwen-qwen_comparison.sh` invocation.

- Default manifest: `configs/evals/qwen_physical_context.json`.
- Cases: `physical_global`, `physical_compact`, `physical_full`, and
  `physical_full_summary`; every case repeats its own terse baseline.
- Controlled settings in the manifest are last pooling and no embedding
  normalization.
- It plans shards, launches one worker per visible/requested GPU with
  `torch.distributed` or Slurm `srun`, resumes completed work by default,
  collects `summary.json`, and optionally writes a Pydantic Evals report.
- `AION_EVAL_WORKERS` selects worker count; otherwise Slurm allocation or
  `CUDA_VISIBLE_DEVICES` is used.

### `scripts/slurm-qwen-evals.sbatch`

This is only a Slurm resource/preset wrapper for the preceding evaluation
harness. It requests one node, four GPUs/tasks, eight CPUs per task, and 12
hours. Its default smoke/sweep settings are four workers, 20,000 rows, and three
epochs. Cluster account and partition still need to be supplied at submission.

## Image-encoder comparison after Qwen magnitudes

### `scripts/qwen_aion-timm_raw_comparison.sh`

**Question:** Given the same frozen raw/terse Qwen magnitude embedding, is the
u-image better represented by the existing AION-token path or by a conventional
frozen timm vision backbone?

- Shared magnitude path: all magnitudes serialized tersely into frozen Qwen.
- AION arm: Qwen magnitude embedding plus trainable downstream decoding/MLP of
  AION FSQ image tokens.
- timm arm: the same Qwen magnitude embedding plus a frozen timm embedding of
  the raw u-band cutout, followed by trainable projection/fusion layers.
- Neither AION nor timm image representation enters Qwen. AION's own
  image-to-redshift representation is not used.
- The primary vision factory is `timm.create_model`. Default backbone is
  `hf-hub:timm/convnext_tiny.dinov3_lvd1689m`, pretrained, one input channel,
  classifier removed (`num_classes=0`), global average pooling, input size 224,
  and signed-asinh/99th-percentile image preprocessing.
- Defaults: 300,000 random rows, seed 42, 10 epochs,
  `Qwen3.5-4B-Base`, Qwen maximum length 2048 and last pooling.
- Qwen embedding extraction and timm embedding extraction run concurrently on
  two GPUs. The training comparison starts from the resulting caches.

## Qwen post-training

### `scripts/qwen-qwen_posttraining_comparison_multigpu.sh`

**Question:** Does supervised QLoRA adaptation of Qwen itself improve photo-z
over using Qwen as a frozen feature extractor?

- Shared input: the same terse all-magnitude records, matched cohort, split,
  redshift bins, and last-token pooling.
- Frozen arm: 4-bit frozen Qwen embeddings plus a supervised photo-z PDF head.
- Post-trained arm: 4-bit Qwen with trainable LoRA adapters, jointly optimized
  with an otherwise matched photo-z head using binned photo-z cross-entropy.
- This is the only current launcher where Qwen parameters—adapter parameters,
  not the base weights—are updated.
- No image features enter either arm. The morphology product is used to define
  a compatible/matched galaxy cohort, not as an input feature.
- Defaults: 300,000 random rows, seed 42, `Qwen3.5-4B-Base`, maximum length
  2048, last pooling; frozen head 10 epochs; QLoRA 3 epochs, microbatch 1,
  gradient accumulation 16, learning rate 2e-4, rank 8, alpha 16, dropout 0.05.
- Two GPUs are required: one frozen-feature job and one QLoRA job run
  concurrently. Progress is summarized every 30 seconds by default.

## IoTFM versus tabular MLP

Here IoTFM means “inference-optimized transformer feature mapping.” The default
model is `GLM-5.2-0.8B-A0.8B`. It is frozen and used to encode serialized
catalogue rows; the comparison MLP receives matched tabular values. AION is not
used in these experiments.

### `scripts/iotfm_mlp.sh`

**Question:** Does a frozen transformer representation of the non-photo-z
catalogue columns outperform a direct tabular MLP?

- Input selection excludes photo-z result columns, ID, sky/location columns by
  default. Other catalogue columns, including flags and classification fields,
  remain unless an ablation below removes them.
- IoTFM arm: serialized catalogue record through the frozen transformer plus a
  photo-z PDF head.
- MLP arm: numerically/categorically encoded matched columns through a tabular
  MLP and the same kind of photo-z head.
- Missing values are explicit by default; numerical MLP features get missing
  indicators.
- Defaults: 200,000 random rows, seed 42, 10 epochs, maximum length 2048,
  **mean pooling**, no 4-bit load unless requested, split 0.63/0.32/0.05,
  300 z bins.

### `scripts/iotfm_mlp_magonly.sh`

The same IoTFM-versus-MLP test restricted to 11 AB magnitudes:
u/u*/g/r/i/z/y/Y/J/H/Ks. Magnitudes are derived from the corresponding flux
columns with zero point 23. ID and location remain excluded.

### `scripts/iotfm_mlp_missnormal.sh`

The same full-column test with missingness signal removed: absent fields are
omitted from IoTFM text, while the MLP uses train-median imputation without
missing-value indicator features. This tests whether missingness itself was
providing predictive information.

### `scripts/iotfm_mlp_noflags.sh`

The full-column test after excluding the detection, quality, mask, and field
flag groups defined by `column_types.md`; it also removes missingness indicators.
With `--no-classification`, it instead keeps only the 121 photometric flux,
flux-error, and Kron-radius inputs and removes rows where catalogue `isStar` is
true. The latter mode also has a separate default output directory.

### `scripts/iotfm_mlp_gnlll.sh`

**Question:** Under a matched heteroscedastic Gaussian objective, does frozen
IoTFM improve the predicted redshift mean over a direct MLP?

- Both arms predict a Gaussian photo-z distribution rather than directly
  learning a 300-class PDF.
- Mean signal: 55 flux measurements plus 11 Kron radii. IoTFM serializes and
  embeds these 66 values; the MLP receives them directly.
- Variance signal shared by both arms: 55 matching flux errors plus the detached
  inferred redshift mean.
- Architecture has separate mean and variance networks. Training starts with
  mean-only SmoothL1 warm-up, then jointly uses Gaussian negative log likelihood.
- Missing/sentinel values are train-median imputed without indicators. Frozen
  IoTFM embeddings are standardized using training rows.
- Gaussian outputs are converted to repository-compatible binned PDFs for the
  normal comparison plots.
- Defaults: 200,000 random rows, seed 42, mean pooling, 10 warm-up epochs,
  20 GNLLL epochs, learning rate 1e-3, and no physical feature scaling.

### `scripts/iotfm_mlp_gnlll_scaling.sh`

The identical GNLLL experiment with `physical_robust` scaling enabled for
signals, errors, and target, and separate cache/output suffixes. It is a scaling
ablation, not a different model family.

## Pretrained table-model comparisons

The launchers in `scripts/table_models/` share one leakage-safe table-completion
engine and require `--model=tabpfn`, `--model=tabfm`, or `--model=tabicl`. The
shell name remains generic (`tabxx_*`), while each result directory begins with
the selected backend, such as `tabicl_magonly-fulltable/`.

Every run starts from a seeded random catalogue sample (50,000 rows by default),
uses the common 0.63/0.32/0.05 train/test/validation split, and exposes `ZPHOT`
only as the training target. Validation/test `ZPHOT` values are stored as NaN in
the model-facing completion table. All other detected redshift, posterior, and
likelihood columns are excluded. Missing feature values are median-imputed from
the training split only.

- `tabxx_noimage-aion_comparison.sh`: 11 magnitudes versus the same magnitudes
  plus the 24-by-24 AION FSQ token-ID grid as 576 named table columns.
- `tabxx_aion-timm_comparison.sh`: AION token columns versus a frozen timm image
  embedding, with the same 11 magnitude columns and matched galaxies.
- `tabxx-mlp_noimage_comparison.sh`: a magnitude-only pretrained table model
  versus the repository's standard magnitude-only PDF MLP.
- `tabxx-mlp_aionimage_comparison.sh`: magnitude+AION-token table completion
  versus the standard magnitude MLP with its trainable decoded-token image path.
- `tabxx_magonly-fulltable.sh`: 11 magnitudes versus exactly 55 fluxes, 55 flux
  errors, and 11 Kron radii, without images.

Each arm writes masked, inferred, filled, and evaluation-truth redshift columns,
plus metrics and a schema. `--prepare-only --save-input-table` audits the exact
matrix without loading a table-model checkpoint. Full operational details and
license cautions are in `scripts/table_models/README.md`.

## Data utilities (not scientific runs)

### `scripts/copydata.py`

Copies the CLAUDS directory tree from
`/arc/projects/ots/Cosmic_Imprint_of_Time/clauds` into `data/clauds` by default.
It can symlink instead, overwrite, dry-run, and use a configurable thread count
(default eight). Existing same-size files are skipped unless overwrite is set.

### `scripts/download_clauds_tiles.py`

Lists CLAUDS `tilesv5` from CANFAR VOSpace and downloads public science FITS
tiles into `data/clauds/images/tilesv5`. Weight maps and catalogues are excluded
unless requested. It supports listing only, resuming lexically with
`--start-after`, limiting the file count, and overwrite. Each download is first
written as `.part` and then renamed.

## Common output and comparison cautions

- A result directory should be interpreted together with its run manifest,
  embedding-cache metadata, row count, seed, split counts, model resolution,
  context length, pooling, and serialization schema. The directory name alone
  is not sufficient provenance.
- Cached frozen embeddings are reusable only when their metadata matches the
  current row IDs and representation settings. A cache hit can make a launcher
  appear much faster without changing the scientific comparison.
- `R²` and Spearman `rho` measure association/ranking, while
  `sigma_NMAD`/scatter and outlier fraction `eta` measure residual quality. A run
  can improve the first pair without improving the latter pair; report both
  rather than calling that an unconditional improvement.
- Multi-GPU wrappers parallelize independent extraction/training branches. They
  do not average models, use data-parallel Qwen inference, or change the
  hypothesis unless their documented defaults differ.
- Qwen launchers default to last pooling. IoTFM launchers default to mean
  pooling; that difference is intentional and should be recorded when comparing
  model families.
