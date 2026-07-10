# AION all-extra-band magnitude fusion work log

Date: 2026-07-03

Workspace: `/Users/shiminggu/Documents/Science/aion_tutorial`

## Scope

This log covers the upgrade from a single `u_mag` late-fusion experiment to a general extra-band magnitude fusion workflow.

Core idea:

- Keep the frozen AION encoder input as HSC `grizy`.
- Add non-AION bands only as magnitude-like scalar inputs to the MLP/fusion branch.
- Allow MLP-only / tabular-only training without AION embeddings.
- Support comparing two configs with paired diagnostic plots.

The protected backup files such as `aion_magnitude_usable_0703.py` were not modified.

## Extra Bands

The intended filter sequence from `clauds_filters.avif` is:

```text
u, u*, g, r, i, z, y, Y, J, H, Ks
```

AION currently consumes:

```text
g, r, i, z, y
```

New optional MLP/fusion extra bands:

| canonical name | plotted label | FITS flux column |
|---|---|---|
| `u` | `u` | `FLUX_CMODEL_MegaCam-u` |
| `u_star` | `u*` | `FLUX_CMODEL_MegaCam-uS` |
| `Y` | `Y` | `FLUX_CMODEL_VIRCAM-Y` |
| `J` | `J` | `FLUX_CMODEL_VIRCAM-J` |
| `H` | `H` | `FLUX_CMODEL_VIRCAM-H` |
| `Ks` | `Ks` | `FLUX_CMODEL_VIRCAM-Ks` |

Catalogue availability confirmed:

- COSMOS has all extra bands: `u`, `u*`, `Y`, `J`, `H`, `Ks`.
- DEEP23 currently has only `u`.

If a selected band is absent, the code warns and fills the feature column. Its valid flag is false if valid flags are enabled.

## `clauds_bands.py`

Updated split-cache schema:

- Added `OPTIONAL_EXTRA_BAND_FLUX_COLUMNS`.
- Added `OPTIONAL_EXTRA_BAND_ERROR_COLUMNS`.
- Added `OPTIONAL_EXTRA_FLAG_COLUMNS`.
- Added combined maps:
  - `ALL_BAND_FLUX_COLUMNS`
  - `ALL_BAND_ERROR_COLUMNS`
  - `ALL_FLAG_COLUMNS`

The split products now include placeholders for optional bands even when a catalogue lacks them:

- missing optional flux/error: `NaN`
- missing `is_clean_*`: `False`
- missing `has_bad_photometry_*`, `is_no_data_*`, `not_observed_*`: `True`

This lets DEEP23 use the same schema as COSMOS without failing on missing VIRCAM / `u*` columns.

## `aion_extra_bands.py`

New module created.

Purpose:

- Generalize `aion_u_band.py` from one `u_mag` scalar to multiple selected extra-band magnitudes.
- Keep AION embeddings unchanged.
- Build `extra_features` for late-fusion or tabular-only models.

Important functions:

- `resolve_extra_band_names(...)`
  - Accepts aliases such as `u*`, `ustar`, `u_star`, `Ks`, `ks`.

- `extract_extra_band_magnitudes_from_table(...)`
  - Converts selected band fluxes to AB magnitudes.
  - Applies quality flags when present.
  - Warns for missing bands.

- `extract_extra_band_magnitudes_from_split_arrays(...)`
  - Same idea, but from split-cache `.npy` arrays.

- `make_extra_band_feature_matrix(...)`
  - Fills invalid magnitudes by `median`, `max_valid`, or numeric fill.
  - Can append `*_mag_valid` columns.

- `make_extra_band_product(...)`
  - Replaces cached product `extra_features` with selected extra-band magnitude features.

- `run_extra_band_ablation(...)`
  - Supports variants:
    - `no_extra`
    - `with_extra`
    - `extra_only`

Feature names for all bands with valid flags disabled:

```text
u_mag, u_star_mag, Y_mag, J_mag, H_mag, Ks_mag
```

With valid flags enabled, also:

```text
u_mag_valid, u_star_mag_valid, Y_mag_valid, J_mag_valid, H_mag_valid, Ks_mag_valid
```

## `aion_magnitude.py`

Main training config was extended.

New `AIONMagnitudeConfig` fields:

```python
extra_bands=("u", "u_star", "Y", "J", "H", "Ks")
extra_band_invalid_fill="median"
extra_band_include_valid_flags=False
use_aion_embedding=True
aion_input_bands=("g", "r", "i", "z", "y")
```

Behavior:

- `extra_bands` controls which non-AION bands enter the MLP/fusion branch.
- `use_aion_embedding=False` enables MLP-only / tabular-only training.
- If `use_aion_embedding=False`, config normalization forces:

```python
model_kinds=("tabular",)
```

and warns if `aion` or `fusion` was requested.

- If trying to change `aion_input_bands`, the code warns:

```text
Disabling individual HSC grizy bands is not currently supported because the frozen AION embedding expects the full grizy input; the requested AION band selection will be ignored for now.
```

The current implementation therefore always uses full `grizy` when AION is enabled.

Training guards added:

- `tabular` and `fusion` require at least one selected extra feature.
- `aion` and `fusion` require AION embeddings.
- This prevents silent training with empty MLP inputs or empty AION vectors.

Cache behavior:

- Existing AION embedding cache can be reused.
- On cache hit, catalogue-side tensors are refreshed:
  - `extra_features`
  - `feature_names`
  - `z_spec`
  - `redshift_reference`
  - extra-band metadata
- If existing split cache lacks the new extra-band schema, it is rebuilt automatically.

Path behavior:

- AION embedding cache path is still based on catalogue / zeropoint / row count.
- Baseline output directories include the selected extra-band tag, so different extra-band configs do not overwrite each other.
- No-AION runs use a `clauds_noaion_catalogue_*` cache prefix.

## Config Pair Comparison

Added comparison helpers to `aion_magnitude.py`.

Existing single-plot functions now accept optional `ax=...` while keeping old behavior unchanged:

- `plot_zpred_vs_zphot(..., ax=None)`
- `plot_pit_histogram(..., ax=None)`
- `plot_redshift_probability_distribution(..., ax=None)`
- `plot_nz_lensing_alike(..., ax=None)`

New pairwise comparison functions:

- `compare_zpred_vs_zphot(...)`
  - left/right subplots

- `compare_pit_histogram(...)`
  - left/right subplots

- `compare_redshift_probability_distribution(...)`
  - top/bottom subplots

- `compare_nz_lensing_alike(...)`
  - top/bottom subplots

- `run_config_pair(config_1, config_2, ...)`
  - Runs both configs and returns:
    - `run_1`, `run_2`
    - `evaluation_1`, `evaluation_2`
    - `model_kind_1`, `model_kind_2`

Default evaluated model:

- `fusion` if `use_aion_embedding=True`
- `tabular` if `use_aion_embedding=False`

## `aion_mlp_test.ipynb`

Notebook was refactored but kept simple.

Main config section now exposes:

```python
EXTRA_BANDS = ("u", "u_star", "Y", "J", "H", "Ks")
USE_AION_EMBEDDING = True
AION_INPUT_BANDS = ("g", "r", "i", "z", "y")
EVALUATE_MODEL_KIND = "fusion" if USE_AION_EMBEDDING else "tabular"
MODEL_KINDS = ("tabular", "aion", "fusion") if USE_AION_EMBEDDING else ("tabular",)
```

The main run still uses:

```python
run = am.run_training_and_evaluation(
    config,
    model_kind=EVALUATE_MODEL_KIND,
    split="test",
)
```

Current notebook comparison section:

```python
RUN_CONFIG_COMPARISON = True
config_1 = config
```

Current `config_2` is MLP-only:

```python
EXTRA_BANDS2 = ("u", "u_star", "Y", "J", "H", "Ks")
USE_AION_EMBEDDING2 = False
MODEL_KINDS2 = ("tabular",)
comparison_labels = ("tabular+aion", "tabular")
```

Current comparison output prefix:

```python
prefix = "aion_improvement"
```

Generated comparison files if the section is run:

- `aion_improvement_scatter.jpeg`
- `aion_improvement_pit.jpeg`
- `aion_improvement_nz.jpeg`
- `aion_improvement_nztomo.jpeg`

The single-run tomographic plot currently saves:

- `tomo_experiment2.jpeg`

## Cache Notes

Deleting all of `cache/` is allowed from a code-flow perspective, but expensive:

- removes split caches
- removes AION embedding caches
- removes checkpoints
- removes generated diagnostic plots
- may require model/cache re-downloads depending on local environment

Gentler alternatives:

```text
cache/clauds_split_*
cache/clauds_aion_embeddings_*
```

However, a full clean cache is reasonable when wanting to force the new all-extra-band schema from scratch.

## Verification Already Run

Passed:

```text
./aion_env/bin/python -m py_compile clauds_bands.py aion_extra_bands.py aion_magnitude.py
./aion_env/bin/python -m py_compile aion_magnitude.py
```

Synthetic / smoke checks passed:

- `clauds_bands.bands_dtype()` includes `flux_cmodel_u_star` and `flux_cmodel_Ks`.
- `clauds_bands.flags_dtype()` includes `is_no_data_Ks`.
- Missing optional bands in a mini FITS split are written as:
  - flux: `NaN`
  - no-data flag: `True`
  - clean flag: `False`
- `aion_extra_bands.build_extra_band_feature_matrix_from_table(...)` produced expected all-band feature names and valid-flag columns.
- `use_aion_embedding=False` normalizes to `model_kinds=("tabular",)`.
- `build_baseline_model("tabular", aion_dim=0, extra_feature_dim=..., n_z_bins=...)` produced expected logits shape.
- All four comparison plotting helpers ran on synthetic evaluation dictionaries.
- `aion_mlp_test.ipynb` parses as valid JSON.

Not fully run:

- Full real COSMOS / DEEP23 FITS split-cache rebuild was not completed during smoke testing because Astropy table metadata initialization was slow. A mini FITS schema test passed, and the implementation is expected to rebuild real split caches when the notebook is run.

## Useful Next Steps

1. Run the current notebook comparison:
   - `config_1`: AION + all extra-band fusion
   - `config_2`: MLP-only with all extra bands

2. Record metrics from:
   - `run["baseline_results"]`
   - `pair["run_1"]["baseline_results"]`
   - `pair["run_2"]["baseline_results"]`

3. Try clean ablations:
   - all extra bands + AION fusion
   - `u` only + AION fusion
   - all extra bands tabular-only
   - `u` only tabular-only
   - with and without `extra_band_include_valid_flags=True`

4. Apply best COSMOS-trained setting to DEEP23.
   - DEEP23 only has `u`; selected missing bands will warn and be filled.
   - If applying a model trained with all COSMOS extra bands to DEEP23, keep an eye on the missing-band valid flags / fill values.

## Update: AION-only and grizy-in-MLP switches

Added after reviewing the first AION-vs-tabular comparison.

Important correction:

- The earlier no-AION `tabular` baseline did not include direct `g,r,i,z,y` feature columns.
- It only used selected extra-band magnitudes unless `grizy` was explicitly added.
- This explained why the tabular scatter and tomographic assignment were very poor despite decent-looking population `n(z)` and PIT.

New config fields:

```python
use_mlp_features=True
include_grizy_in_mlp=None
```

Behavior:

- `use_mlp_features=False` means AION-only mode.
  - Requires `use_aion_embedding=True`.
  - Forces `model_kinds=("aion",)`.
  - Uses no MLP extra-feature branch.

- `include_grizy_in_mlp=None` means automatic default.
  - If `use_aion_embedding=True`, it resolves to `False`; `grizy` is already represented by AION and is not duplicated in the MLP.
  - If `use_aion_embedding=False`, it resolves to `True`; tabular-only baselines use direct `g_mag,r_mag,i_mag,z_mag,y_mag` features.

The notebook comparison section now uses:

```python
config_1: AION grizy + extra-band fusion
config_2: grizy-only tabular baseline
```

For grizy-only tabular, the notebook uses:

```python
EXTRA_BANDS2 = ()
USE_AION_EMBEDDING2 = False
USE_MLP_FEATURES2 = True
INCLUDE_GRIZY_IN_MLP2 = None
```

Note:

- `EXTRA_BANDS2 = None` means "use default extra bands", not "use no extra bands".
- Use `EXTRA_BANDS2 = ()` for no `u/u*/Y/J/H/Ks` extra-band columns.

## Update: 2026-07-03 late comparison results

Two real COSMOS comparison series were inspected:

```text
aion_additional_*
aion_improvement_*
```

These are the most useful conclusions from this version.

### `aion_additional_*`

Comparison:

```text
uu*grizyYJHKs-MLP + grizy-AION
vs
uu*grizyYJHKs-MLP
```

Interpretation:

- This asks whether frozen AION `grizy` embeddings add useful information once the MLP already sees direct `u,u*,g,r,i,z,y,Y,J,H,Ks` scalar magnitudes.
- In the current run, they do not.

Scatter metrics from `aion_additional_scatter.jpeg`:

| model | sigma_NMAD | outlier eta | R2 | Pearson rho |
|---|---:|---:|---:|---:|
| `uu*grizyYJHKs-MLP` | 0.0520 | 14.90% | 0.5551 | 0.7595 |
| `uu*grizyYJHKs-MLP + grizy-AION` | 0.0578 | 16.21% | 0.5217 | 0.7368 |

Qualitative read:

- Loss curves finish in a similar range, but the AION-fusion validation curve is not clearly better.
- PIT is less convincing with AION; it has stronger right-side accumulation.
- Overall `n(z)` and tomographic `n(z)` do not show a clear win from adding AION.

Working conclusion:

- When direct `grizy` magnitudes are already in the MLP, frozen AION embeddings are at best redundant in this setup, and in this run slightly degrade object-level photo-z metrics.

### `aion_improvement_*`

Comparison:

```text
uu*YJHKs-MLP + grizy-AION
vs
uu*grizyYJHKs-MLP
```

Interpretation:

- This asks whether AION `grizy` embeddings can replace direct scalar `g,r,i,z,y` magnitudes.
- In the current run, they cannot.

Scatter metrics from `aion_improvement_scatter.jpeg`:

| model | sigma_NMAD | outlier eta | R2 | Pearson rho |
|---|---:|---:|---:|---:|
| `uu*grizyYJHKs-MLP` | 0.0520 | 14.90% | 0.5551 | 0.7595 |
| `uu*YJHKs-MLP + grizy-AION` | 0.0884 | 23.32% | 0.4545 | 0.6886 |

Qualitative read:

- Object-level scatter is much worse when direct `grizy` scalar features are removed and replaced by frozen AION embeddings.
- The aggregate `n(z)` can still look surprisingly good, but that is not enough: scatter and tomographic assignment show that the object-level predictions are degraded.
- This is another reminder that population-level `n(z)` agreement can hide poor per-object photo-z behavior.

Working conclusion:

- Direct scalar `grizy` magnitudes are currently essential for the best MLP/tabular baseline.
- Frozen AION embeddings do not currently substitute for direct `grizy` magnitudes in this late-fusion setup.

### Current best empirical baseline

Based on these images, the strongest and cleanest baseline in this version is:

```text
uu*grizyYJHKs-MLP
```

That is, use all available direct magnitude-like scalar features in the tabular MLP, including `g,r,i,z,y`.

Current statement to carry forward:

- The all-magnitude MLP baseline is strong.
- Frozen AION `grizy` embeddings have not yet shown a measurable gain in this implementation.
- We should not yet conclude that AION is intrinsically unhelpful; the next dedicated session should investigate why the frozen AION branch is not helping here.

Likely next debugging targets for the next session:

- Confirm AION embedding extraction / pooling is sensible for these magnitude inputs.
- Compare AION-only `grizy` against direct `grizy`-only MLP.
- Check scale/normalization mismatch between AION embeddings and MLP scalar features.
- Check whether the fusion architecture gives the AION branch enough capacity or whether it is being ignored / harming calibration.
- Inspect whether checkpoints and cached products are cleanly separated by config.

### Config fixes made during this round

The following implementation details were fixed after hitting edge-case errors:

- `EXTRA_BANDS = ()` is now a valid zero-extra-band selection and returns an `(N, 0)` feature matrix instead of triggering `torch.stack([])`.
- `include_grizy_in_mlp=None` is resolved at the point of use:
  - no AION -> direct `grizy` enters the MLP by default;
  - AION enabled -> direct `grizy` is not duplicated into the MLP by default.
- If AION is enabled, extra bands are empty, and `grizy` is not duplicated into the MLP, the config is treated as AION-only:

```python
use_mlp_features=False
model_kinds=("aion",)
```

This prevents accidentally training `tabular` or `fusion` models with zero MLP input features.

## Update: 2026-07-04 antigravity review fixes

Two real issues from the antigravity review were confirmed and fixed in the root
`aion_magnitude.py`.

### Cache-hit AION embedding consistency

Confirmed issue:

- In the `build_and_cache_aion_embeddings(...)` cache-hit path, the existing
  cached product was loaded and only catalogue-side tensors were refreshed.
- `refresh_cached_product_catalogue_features(...)` updated:
  - `extra_features`
  - `feature_names`
  - `z_spec`
  - `redshift_reference`
  - metadata
- It did not update or validate `product["aion_embedding"]`.
- Therefore, if a user manually reused the same `cache_path` while switching
  `use_aion_embedding` from `True` to `False`, the product could keep an old
  non-empty AION embedding tensor while metadata said AION was disabled.

Fix:

- On cache refresh with `use_aion_embedding=False`, the cached product now
  forcibly replaces:

```python
product["aion_embedding"] = torch.empty((n_rows, 0), dtype=torch.float32)
```

- The AION-related metadata is also reset:

```python
aion_model = None
aion_embedding_pooling = None
embedding_batch_size = None
```

- On cache refresh with `use_aion_embedding=True`, the code now verifies that
  the cached product has a usable non-empty 2D AION embedding tensor with the
  correct row count.
- If the tensor is missing, empty, or row-mismatched, it raises a `RuntimeError`
  instructing the user to rerun with `force_recompute_embeddings=True` or use a
  cache path built with AION enabled.

Reasoning:

- The default path resolver already separates normal AION and no-AION cache
  prefixes, so the bug is most likely when users manually override `cache_path`.
- The fix makes this edge case explicit and prevents silent metadata/tensor
  inconsistency.

### `ensure_cached_product_redshift_reference(...)` undefined variables

Confirmed issue:

- `ensure_cached_product_redshift_reference(...)` passed these variables into
  `build_raw_clauds_photoz_dataset(...)`:

```python
extra_bands
extra_band_invalid_fill
extra_band_include_valid_flags
```

- Those names were not present in the function signature or local scope, so any
  execution of that branch would raise `NameError`.

Fix:

- Added these keyword parameters to the function signature:

```python
extra_bands: Sequence[str] | None = None
extra_band_invalid_fill: str | float = "median"
extra_band_include_valid_flags: bool = False
```

### Verification

Passed:

```text
./aion_env/bin/python -m py_compile aion_magnitude.py aion_extra_bands.py clauds_bands.py
```

Synthetic smoke test passed:

- cache refresh with `use_aion_embedding=False` converts an old `(N, 512)`
  cached AION embedding to `(N, 0)`;
- metadata `aion_model` is reset to `None`;
- cache refresh with `use_aion_embedding=True` and an empty AION tensor raises;
- `ensure_cached_product_redshift_reference(...)` accepts `extra_bands=()` and
  no longer hits undefined-variable names.

## Update: 2026-07-07 M-adapter status

We explored an AION input-side M adapter because direct CLAUDS/HSC magnitudes
may not match the selection/calibration function seen by native AION training.
The idea was to adjust only the grizy magnitudes passed into frozen AION:
`m_aion = m_grizy + M @ standardized([grizy, u, u*, Y, J, Ks])`.

Gradient training failed because the AION magnitude codec does not expose a
useful autograd path back to M. Finite-step and SPSA tests confirmed that M can
change AION outputs, but the learned matrices were unstable and did not improve
validation CE relative to the zero-M AION baseline.

We therefore stop treating learned M as a primary photo-z direction. The
interface is retained as experimental code heritage for externally supplied
photometric adjustments or future calibration-matrix tests.
