# Qwen physical-context and image-token work — 2026-07-18

## Purpose

This log is a restart-ready handoff for the CLAUDS/Qwen experiments completed
around the lost cluster session. No training run was started and none of the
work described here was committed by Codex.

## Recovered work after the cluster-session loss

The previous uncommitted `qwen-mlp_full_comparison` implementation was
recovered from the local Codex rollout cache:

- `scripts/qwen-mlp_full_comparison.sh`
- `notebooks/qwen_mlp_full_comparison.py`
- supporting feature-scaling and Qwen morphology dispatch in
  `aion_magnitude/morphology.py`
- train-only min-max/z-score tests in `tests/test_morphology.py`

The recovered full comparison:

- uses all available magnitudes;
- disables faint-end magnitude cuts;
- applies no explicit redshift-range cut other than requiring a finite target;
- derives the redshift grid from the selected catalogue;
- supports `none`, `minmax`, and `zscore` feature scaling;
- defaults scaling to `none`, while its launcher selects train-set `minmax`;
- keeps validation/test values unclipped and reuses only training statistics;
- uses tokenized CLAUDS u images without AION's image-to-redshift embedding;
- writes a dedicated run directory and population report.

Static recovery checks passed. The recreated system environment did not have
pytest, so the focused tests could not be rerun after recovery.

## CLAUDS band research

The physically informed Qwen serialization uses these survey meanings:

| Band | Facility/instrument | Wavelength information | Region |
|---|---|---|---|
| u | CFHT/MegaCam new u | central 3538 A; bandwidth 868 A | near-UV |
| u* | CFHT/MegaCam old u* | central 3743 A; bandwidth 758 A | near-UV |
| g | Subaru/HSC | effective wavelength about 4740 A | blue optical |
| r | Subaru/HSC | effective wavelength about 6170 A | optical |
| i | Subaru/HSC | effective wavelength about 7650 A | red optical |
| z | Subaru/HSC | effective wavelength about 8890 A | very-red optical |
| y | Subaru/HSC | effective wavelength about 9760 A | very-red optical |
| Y | VISTA/VIRCAM | representative wavelength about 1.0214 um | near-IR |
| J | VISTA/VIRCAM | representative wavelength about 1.2535 um | near-IR |
| H | VISTA/VIRCAM | representative wavelength about 1.6453 um | near-IR |
| Ks | VISTA/VIRCAM | representative wavelength about 2.1540 um | near-IR |

Important distinctions encoded in the prompt:

- u and u* are separate filters; the newer u is bluer, while old u* has a
  weak red leak near 5000 A.
- Lowercase HSC y and uppercase VISTA Y are separate instruments/passbands.
- Magnitudes are described as AB magnitudes: lower magnitude means greater
  observed flux density, and magnitude differences are colours.
- Missing measurements are not zero flux. They can result from footprint,
  depth, masking, or measurement failure.
- The magnitudes are presented in wavelength order.

Sources consulted during the search:

- Sawicki et al. (2019), CLAUDS survey paper:
  https://academic.oup.com/mnras/article/489/4/5202/5566343
- Desprez et al. (2023), combined CLAUDS/HSC catalogue:
  https://arxiv.org/abs/2301.13750
- HSC effective filter wavelengths:
  https://prc.nao.ac.jp/citizen-science/hscv/hscdata.html
- COSMOS VISTA filter set:
  https://cosmos.astro.caltech.edu/page/filterset

## `aion_magnitude/FM_Qwen3.py`

Added a new physically informed Qwen module. `FM_Qwen.py` remains unchanged as
the terse baseline.

Main interfaces:

- `CLAUDS_BAND_DESCRIPTIONS`
- `Qwen3SerializationConfig`
- `Qwen3EmbeddingConfig`
- `serialize_qwen3_observation(...)`
- `serialize_qwen3_batch(...)`
- `serialize_tokenized_galaxy_image(...)`
- `qwen3_embedding_metadata(...)`

The module serializes physical magnitude descriptions first and optionally
appends an ordered AION 24x24 image-token grid.

The image is introduced neutrally as `tokenized galaxy image`. The prompt says
that an image tokenizer converted the observed cutout into an ordered grid,
that the grid retains image information, and that it may provide visual
context complementary to photometry. It does not impose a predefined meaning
on individual token IDs. This intentionally leaves open the possibility that
Qwen or a future adapter can learn morphology from token combinations and
spatial ordering.

The default context length in `Qwen3EmbeddingConfig` is 2048 tokens because a
24x24 image-token grid plus physical magnitude descriptions is much longer
than the previous terse 256-token input.

## Physical Qwen versus unchanged image-token MLP

Added:

- `scripts/qwen-mlp_full_image_comparison.sh`
- `notebooks/qwen_mlp_full_image_comparison.py`

Comparison:

1. `physical-all-magnitude-Qwen+tokenized-galaxy-image`
2. `all-magnitude-MLP+tokenized-galaxy-image`

The Qwen branch reads both the physical column descriptions and the ordered
image tokens through `FM_Qwen3`. Its downstream photo-z head receives only the
resulting frozen Qwen representation, so the image is not supplied twice.

The MLP branch is unchanged: all magnitude features and the existing AION
image-token encoder feed the MLP photo-z model.

The physical-image embedding cache has a `physical_image_` tag so it cannot
silently reuse the old terse-Qwen cache.

Default launcher context length:

```bash
QWEN_MAX_LENGTH=2048
```

## Physical Qwen versus terse Qwen

Added:

- `scripts/qwen-qwen_comparison.sh`
- `notebooks/qwen_qwen_comparison.py`

Comparison:

1. `physical-all-magnitude-Qwen`
2. `terse-all-magnitude-Qwen`

Both branches:

- receive the same all-magnitude values;
- use the same Qwen model, quantization, context length, pooling, splits,
  optimizer settings, and downstream head architecture;
- do not serialize image tokens;
- reset the random seed before constructing the downstream head;
- save checkpoints into separate `physical_qwen/` and `terse_qwen/`
  directories;
- use separate physical and terse embedding caches.

The comparison currently reuses the full-comparison cohort and cache machinery
to guarantee identical selected rows and split labels. Although image-token
availability determines that shared cohort, neither Qwen branch receives the
image tokens in this magnitude-only ablation.

## Validation performed

Passed:

- Python byte-code compilation for `FM_Qwen3.py` and both new runners;
- shell syntax checks for both new launchers;
- executable permissions on both launchers;
- `git diff --check`.

Not performed:

- no Qwen checkpoint load;
- no embedding extraction;
- no GPU/CUDA run;
- no photo-z training;
- no full pytest run because pytest/PyTorch were unavailable in the current
  system Python used for lightweight validation.

## Useful next commands

Physical Qwen with image tokens versus unchanged image-token MLP:

```bash
./scripts/qwen-mlp_full_image_comparison.sh
```

Physical magnitude descriptions versus terse magnitude serialization:

```bash
./scripts/qwen-qwen_comparison.sh
```

Useful environment overrides include:

```bash
AION_MAX_ROWS=1000
AION_EPOCHS=1
QWEN_EMBEDDING_BATCH_SIZE=1
QWEN_MAX_LENGTH=2048
AION_DEVICE=cuda
QWEN_MODEL=qwen3_8b_base
```

Use a small `AION_MAX_ROWS` smoke run before the full catalogue. The physical
image serialization is substantially longer and more expensive than the old
terse input, and Qwen3-8B previously operated very close to the roughly 11 GiB
MIG memory limit.

## Worktree caution

At the end of this session, the new Qwen files and scripts were uncommitted.
The `data/` directory was also visible as untracked and belongs to the user's
workspace; it was not modified as part of this work. Review `git status` before
committing and do not add large catalogues or images.

The Codex patch helper intermittently failed because this cluster kernel has
unprivileged user namespaces disabled. For updates to existing files, a
carefully scoped standard `patch` fallback was used. This is also why restarting
Codex to persist the user's preferred permission settings may be useful.
