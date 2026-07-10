# AION + CLAUDS u-band / u-magnitude work log

Date: 2026-07-03
Workspace: `/Users/shiminggu/Documents/Science/aion_tutorial`

## Scope and constraints

- Active notebook: `aion_mlp_different.ipynb`
- Frozen notebook: `aion_mlp.ipynb`
  - This was intentionally frozen/read-only earlier and should not be edited unless explicitly requested.
- Do not use or reference `Tutorial_marked.ipynb` in future continuation prompts.
- The current useful catalogue direction is:
  - train on `data/clauds/COSMOS-HSCpipe-Phosphoros.fits`
  - apply/validate on `data/clauds/DEEP23-HSCpipe-Phosphoros.fits`
- Main cached COSMOS product:
  - `cache/clauds_aion_embeddings_COSMOS_HSCpipe_Phosphoros_zp23p0_all.pt`
  - usable rows: `723865`
  - `mag_zero_point = 23.0`
  - grizy faint-end cuts: `g=24.5, r=24.5, i=24.0, z=24.5, y=24.5`

## Key clarification

The requested experiment is not about the high-z clump issue.

The intended comparison is:

- `no_u`: use AION frozen encoder embedding from HSC `grizy` only.
- `with_u`: use the same frozen AION `grizy` embedding, plus `u_mag` as an extra MLP/fusion input.
- The `u` band is not passed into the AION encoder.
- The `u` input should be a magnitude-like scalar analogous to `g_mag`, not the older tabular set of `asinh_flux_cmodel_u`, u flux error, u flags, etc.

## New module

Created/updated:

- `aion_u_band_ablation.py`

Despite the filename, the current semantics are u-magnitude ablation.

Important functions:

- `load_u_magnitude_from_split_cache(product, ...)`
  - Reloads split CLAUDS arrays from `metadata["split_output_dir"]`.
  - Rebuilds the same grizy-based usable mask used by the cached AION product.
  - Returns `u_mag` and `valid_u` aligned row-by-row with `product["aion_embedding"]`.
  - Does not use u-band for the usable-row selection.

- `make_no_extra_feature_product(product)`
  - Creates the `no_u` product.
  - Keeps AION embeddings.
  - Sets `extra_features` to shape `(N, 0)`.
  - `feature_names = []`.

- `make_u_magnitude_product(product, u_magnitude, valid_u=..., invalid_fill="median", include_valid_flag=False)`
  - Creates the `with_u` product.
  - Keeps AION embeddings.
  - Sets `extra_features` to `["u_mag"]`, shape `(N, 1)`.
  - If `include_valid_flag=True`, adds `u_mag_valid` as a second feature.
  - Invalid u magnitudes are filled by default with the median valid `u_mag`.

- `run_u_magnitude_ablation(...)`
  - Trains matched baselines:
    - `no_u`: `model_kind="aion"`
    - `with_u`: `model_kind="fusion"`
  - Supports:
    - `require_valid_u=False` by default, same sample for both variants.
    - `require_valid_u=True` to restrict both variants to valid-u rows.
    - `include_valid_flag=True` to tell the MLP whether `u_mag` was real or filled.

- Compatibility aliases remain:
  - `run_u_band_ablation = run_u_magnitude_ablation`
  - `format_u_band_ablation_summary = format_u_magnitude_ablation_summary`
  - These exist only to avoid breaking older notebook state.

Validation already done:

- `aion_env/bin/python -m py_compile aion_u_band_ablation.py`
- Cache alignment checked:
  - `no_u extra shape: (723865, 0)`
  - `with_u extra shape: (723865, 1)`
  - `with_u feature_names: ["u_mag"]`
  - valid `u_mag`: `320693`
  - invalid/filled `u_mag`: `403172`
  - median fill value: about `23.9634`

## Notebook updates

Updated `aion_mlp_different.ipynb`.

Relevant section:

- `# U-magnitude ablation module`

This section now:

- Uses `importlib.reload(aion_u_band_ablation)` so a stale Jupyter kernel can pick up edits without restart.
- Defines:
  - `RUN_U_BAND_ABLATION = True`
  - `U_BAND_ABLATION_OUTPUT_DIR = Path("cache") / f"u_magnitude_ablation_{RUN_TAG}"`
  - `U_BAND_REQUIRE_VALID_U = False`
  - `U_BAND_INCLUDE_VALID_FLAG = False`
  - `U_BAND_INVALID_FILL = "median"`
- Loads aligned `u_mag` with `load_u_magnitude_from_split_cache(product_for_u_ablation)`.
- Runs `run_u_magnitude_ablation(...)`.

The earlier error:

```text
TypeError: run_u_band_ablation() got an unexpected keyword argument 'no_u_model_kind'
```

was caused by a stale/old module object in the notebook kernel. The reload cell should prevent this now.

## Current u-magnitude ablation result

User-reported result:

| variant | nmad | catastrophic_outlier_fraction | cross_entropy | mean_crps | p16_p84_coverage | pit_mean |
|---|---:|---:|---:|---:|---:|---:|
| no_u | 0.10556 | 0.29290 | 3.91065 | 0.24693 | 0.68872 | 0.50921 |
| with_u | 0.08978 | 0.24615 | 3.78171 | 0.21653 | 0.69143 | 0.51372 |
| delta_with_minus_no_u | -0.01578 | -0.04675 | -0.12894 | -0.03040 | 0.00271 | 0.00451 |

Interpretation:

- Adding `u_mag` is worthwhile for this setup.
- NMAD improves by about 15% relative.
- Catastrophic outlier fraction drops by about 4.7 percentage points.
- Cross entropy and CRPS both improve.
- Coverage is essentially stable/slightly better.
- Recommended next variant: try `U_BAND_INCLUDE_VALID_FLAG = True`.
- Also useful: try `U_BAND_REQUIRE_VALID_U = True` to see the valid-u-only upper-bound improvement.

## Existing checkpoints

Current u-magnitude ablation checkpoints:

- `cache/u_magnitude_ablation_COSMOS_HSCpipe_Phosphoros_zp23p0_all/no_u/aion_baseline.pt`
- `cache/u_magnitude_ablation_COSMOS_HSCpipe_Phosphoros_zp23p0_all/with_u/fusion_baseline.pt`

Existing per-variant plots from training:

- `cache/u_magnitude_ablation_COSMOS_HSCpipe_Phosphoros_zp23p0_all/no_u/aion_val_zpred_vs_zphot.png`
- `cache/u_magnitude_ablation_COSMOS_HSCpipe_Phosphoros_zp23p0_all/no_u/aion_val_pit_histogram.png`
- `cache/u_magnitude_ablation_COSMOS_HSCpipe_Phosphoros_zp23p0_all/with_u/fusion_val_zpred_vs_zphot.png`
- `cache/u_magnitude_ablation_COSMOS_HSCpipe_Phosphoros_zp23p0_all/with_u/fusion_val_pit_histogram.png`

## Added comparison plots

Added plotting cell after the u-magnitude ablation section in `aion_mlp_different.ipynb`.

It produces:

1. `z_p50 vs z_phot`
   - left subplot: without u
   - right subplot: with u

2. `z_mean vs z_phot`
   - left subplot: without u
   - right subplot: with u

3. Redshift distribution / mean p(z)
   - three curves on the same axes:
     - true `z_phot`
     - without-u mean `p(z)`
     - with-u mean `p(z)`
   - each curve is smoothed with a Gaussian filter of `sigma=1 bin`
   - implemented with numpy convolution, no scipy dependency

Generated files:

- `cache/u_magnitude_ablation_COSMOS_HSCpipe_Phosphoros_zp23p0_all/u_mag_ablation_z_p50_vs_zphot_subplots.png`
- `cache/u_magnitude_ablation_COSMOS_HSCpipe_Phosphoros_zp23p0_all/u_mag_ablation_z_mean_vs_zphot_subplots.png`
- `cache/u_magnitude_ablation_COSMOS_HSCpipe_Phosphoros_zp23p0_all/u_mag_ablation_pz_distribution_smoothed.png`

The plots were generated successfully using existing checkpoints; no retraining was triggered.

## Continuation suggestions

Good next steps for a new session:

1. Run `U_BAND_INCLUDE_VALID_FLAG = True`.
   - Compare against current `with_u` one-feature result.
   - This tests whether the model benefits from knowing which `u_mag` values are real versus median-filled.

2. Run `U_BAND_REQUIRE_VALID_U = True`.
   - This compares only valid-u rows.
   - It answers the clean question: how much does real u-band magnitude help when available?

3. Apply the best `with_u` checkpoint to the second catalogue.
   - Need to ensure `u_mag` for the apply catalogue is reconstructed using the apply catalogue's split cache.
   - If the apply catalogue has very different u-depth or missingness, keep the `no_u` checkpoint as fallback.

4. Consider renaming variables in notebook:
   - `u_band_ablation` variable currently stores the u-magnitude ablation result.
   - It works, but `u_magnitude_ablation` would be clearer.

5. If copying the module elsewhere:
   - The clean API is `run_u_magnitude_ablation(...)`.
   - Provide a cached AION product plus either:
     - explicit `u_magnitude`, `valid_u`, or
     - product metadata with `split_output_dir`, so `load_u_magnitude_from_split_cache(...)` can reconstruct it.

