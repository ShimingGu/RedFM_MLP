# AION Main Module Work Log

Date: 2026-07-03

Workspace: `/Users/shiminggu/Documents/Science/aion_tutorial`

## Files

- `aion_magnitude.py`: main extracted training/evaluation/plotting module.
- `aion_mlp_test.ipynb`: lightweight notebook that imports `aion_magnitude.py`, trains/evaluates, and plots diagnostics on the test set.
- `aion_mlp.ipynb`: original locked notebook, kept as source reference only.

## Main Goal

Refactor the training and plotting workflow from `aion_mlp.ipynb` into a reusable Python module, so tunable parameters can be changed from the notebook/function calls instead of editing global notebook cells or reopening the `.py` file.

## Important Design Decisions

- Removed the old pattern of public module-level tunable globals such as:
  - `DEFAULT_HSC_MAG_FAINT_LIMITS`
  - `HSC_MAG_FAINT_LIMITS`
  - `DEFAULT_FLUX_MAG_ZEROPOINT`
  - `FLUX_MAG_ZEROPOINT`
  - `Z_MIN`, `Z_MAX`, `N_Z_BINS`
- Defaults now live in `AIONMagnitudeConfig`, function signatures, or default-factory helpers.
- Importing `aion_magnitude.py` should not silently set notebook-level tunable quantities.
- The import-time `set_random_seed()` call was removed. Training entry points still call `set_random_seed(config.seed)`.
- `build_and_cache_aion_embeddings()` now accepts `n_z_bins`, `z_min`, and `z_max`, so redshift-grid settings propagate correctly.

## Split Logic

Random train/test/validation split was added.

Current default split fractions:

- train: `0.20`
- test: `0.75`
- val: `0.05`

Validation set is used during training/model selection diagnostics. Test set is kept for final reporting plots. The test notebook now calls:

```python
run = am.run_training_and_evaluation(
    config,
    model_kind="fusion",
    split="test",
)
```

Final plots use `test_eval`, not `val_eval`.

## Main Config Pattern

Notebook-side parameters are adjusted through:

```python
config = am.AIONMagnitudeConfig(
    catalogue_path=Path("data/clauds/COSMOS-HSCpipe-Phosphoros.fits"),
    max_rows=None,
    z_min=0.0,
    z_max=6.0,
    n_z_bins=300,
    split_strategy="random",
    train_fraction=0.20,
    test_fraction=0.75,
    val_fraction=0.05,
    baseline_epochs=10,
    hsc_mag_faint_limits={"g": 24.5, "r": 24.5, "i": 24.0, "z": 24.5, "y": 24.5},
    device_choice="auto",
)
```

This is the intended place to change those values.

## Redshift PDF Plot

Added:

- `gaussian_kernel_1d()`
- `gaussian_smooth_1d()`
- `redshift_probability_distribution()`
- `plot_redshift_probability_distribution()`

Important correction: this plot compares `true` vs `recovered` under the same smoothing kernel. It does not compare smoothed vs unsmoothed curves.

Current behavior:

- `true`: histogram of `evaluation["z_spec"]` on the redshift grid.
- `recovered`: mean of all per-object `evaluation["pz"]`.
- Both are smoothed with the same `gaussian_sigma_bins`.
- Raw arrays are still returned in `pdf_data`, but not plotted by default.

Notebook call:

```python
gaussian_sigma_bins = 2.0

fig, ax, pdf_data = am.plot_redshift_probability_distribution(
    test_eval,
    gaussian_sigma_bins=gaussian_sigma_bins,
    include_true=True,
    title="Fusion Model Test: redshift probability distribution",
)
```

## Lensing-Like Tomographic n(z)

Added:

- `plot_nz_lensing_alike()`
- `validate_zphot_bins()`
- `tomographic_bin_labels()`
- `assign_tomographic_bins()`
- `sample_lognormal_from_percentiles()`
- `sample_catalogue_redshift_per_object()`
- `sample_inferred_redshift_per_object()`

Required parameter:

```python
zphot_bin=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
```

This produces seven tomographic bins:

- `(0, 0.5]`
- `(0.5, 1]`
- `(1, 1.5]`
- `(1.5, 2]`
- `(2, 2.5]`
- `(2.5, 3]`
- `(3, infinity]`

Objects with `z <= 0` are dropped from tomographic assignment.

Input catalogue side:

- Uses catalogue `zphot`, `z_low68`, `z_high68`, and optionally `z_peak`.
- Assumes a lognormal per-galaxy redshift distribution.
- Draws `n_samples_per_object=10` by default.

Inferred/model side:

- Uses `evaluation["pz"]` per galaxy if available and samples from the discrete p(z).
- If `pz` is absent, falls back to a lognormal approximation from `z_p50`, `z_p16`, `z_p84`, and optionally `z_mode`.
- Tomographic assignment defaults to `inferred_bin_key="z_p50"`, but can be changed to `"z_mean"` or `"z_mode"`.

Plot styling:

- Same tomographic bin uses the same color for input and inferred.
- Input linestyle: `":"`
- Inferred linestyle: `"--"`
- All tomographic bins are plotted on one axis.

Notebook call:

```python
zphot_bin = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

fig, ax, nz_data = am.plot_nz_lensing_alike(
    test_eval,
    zphot_bin=zphot_bin,
    inferred_bin_key="z_p50",
    n_samples_per_object=10,
    title="Fusion Model Test: tomographic n(z)",
)
```

## Redshift Reference Columns

The cache/evaluation pipeline now carries catalogue redshift reference columns through:

```python
redshift_reference = {
    "zphot": ...,
    "z_low68": ...,
    "z_high68": ...,
    "z_peak": ...,
}
```

These are extracted from `clauds_bands.REDSHIFT_COLUMNS`:

- `ZPHOT`
- `Z_LOW68`
- `Z_HIGH68`
- `Z_PEAK`

Cached products now include:

- `object_id`
- `field`
- `z_spec`
- `redshift_reference`
- `aion_embedding`
- `extra_features`
- `feature_names`
- `split_labels`
- `metadata`

For old caches without `redshift_reference`, `ensure_cached_product_redshift_reference()` tries to rebuild only the catalogue-side reference columns and attach them to the existing cache. This avoids rerunning AION embeddings. If object order does not match, rerun with:

```python
force_recompute_embeddings=True
```

## Current Test Notebook State

`aion_mlp_test.ipynb` cells:

1. Import/reload `aion_magnitude`.
2. Build `AIONMagnitudeConfig`.
3. Run training and evaluate on `split="test"`.
4. Plot `z_p50` vs catalogue redshift and PIT histogram.
5. Plot `z_mean` and `z_mode` diagnostics.
6. Plot true vs recovered redshift PDF with shared Gaussian smoothing.
7. Plot lensing-like tomographic `n(z)`.

Notebook outputs were cleared after edits to avoid stale embedded plots/base64 outputs.

## Verification Already Run

Commands/checks that passed:

```text
./aion_env/bin/python -m py_compile aion_magnitude.py
```

Synthetic smoke tests:

- `plot_redshift_probability_distribution()` produced labels:
  - `true (Gaussian sigma=2 bins)`
  - `recovered (Gaussian sigma=2 bins)`
- `plot_nz_lensing_alike()` with 7 tomographic bins produced 14 lines.
- `plot_nz_lensing_alike()` works both with `pz` sampling and fallback lognormal sampling.
- `aion_mlp_test.ipynb` parses as valid JSON.

Search checks:

- No old public default/global tunables found:
  - `DEFAULT_`
  - `HSC_MAG_FAINT_LIMITS`
  - `FLUX_MAG_ZEROPOINT`
  - `Z_MIN`
  - `Z_MAX`
  - `N_Z_BINS`
- No import-time `set_random_seed()` call found.

## Notes For The Next Session

- If the notebook uses an old cache, the new code should try to attach `redshift_reference` without recomputing embeddings.
- If that fails with an object-order mismatch, use `force_recompute_embeddings=True` in `AIONMagnitudeConfig`.
- The current module assumes `Z_LOW68` and `Z_HIGH68` are 16/84 percentile redshift values around the median-like `ZPHOT`.
- `plot_nz_lensing_alike()` normalizes each tomographic bin's sampled redshift distribution to a probability density.
- The Gaussian smoothing parameter in the redshift PDF plot is measured in redshift-bin widths, not physical redshift units.
