# COSMOS GNLLL scaling design

## Decision

Use a per-column, training-split-only, physically informed robust transform
for the scaled IoTFM-versus-MLP GNLLL experiment.  Do not use raw min-max
scaling for these catalogue measurements.

GNLLL means **Gaussian negative log-likelihood loss**.  The intended model
separation remains:

- redshift mean signal: 55 flux estimators plus 11 Kron radii;
- redshift variance signal: 55 matching flux errors plus the detached inferred
  redshift mean.

The scaled and unscaled experiments must otherwise use the same inputs,
split, architecture, training schedule, and evaluation routine.

## Why min-max scaling is unsuitable

The randomized 200,000-row COSMOS cache shows highly skewed distributions,
very bright outliers, valid negative noisy fluxes, and finite `-99` missing
sentinels.  For example, HSC-g cModel flux has a 99th percentile near 7 but a
maximum near 896.  Some flux maxima exceed 26,000.  Several VIRCAM columns are
dominated by `-99` values.  Raw min-max scaling therefore lets sentinels and a
few extrema compress nearly all ordinary galaxies into a narrow numerical
range.

## Missing-value policy

- Treat non-finite values and the exact finite sentinel `-99` as missing.
- Do not classify every negative flux as missing: small negative fluxes are
  physically valid noisy measurements.
- Flux errors must be positive; non-positive flux errors are missing after
  sentinel handling.
- Kron radii must be non-negative; negative radii are missing.
- Do not add missingness-indicator features for this experiment.
- After robust centering, impute missing transformed values to zero.  Zero is
  the training median in scaled space, so missingness is not intentionally
  supplied as a separate signal.

## Feature transformations

All statistics below are fitted on training rows only and then reused without
refitting for validation and test rows.

### Flux measurements

For each flux estimator column `j`, use a fixed softening scale derived from
the matching error column:

```text
s_j = median(positive valid FLUXERR_j values in the training split)
t_flux = asinh(flux / s_j)
```

This keeps valid negative and faint fluxes, compresses bright outliers, and
uses only a population-level training constant.  It does not give an object's
individual flux error to the redshift-mean head, so the intended mean/variance
information separation is preserved.

### Flux uncertainties

For positive valid errors:

```text
t_error = log(error)
```

### Kron radii

For each radius column, with `r_j` equal to its positive training median:

```text
t_radius = log1p(radius / r_j)
```

### Robust standardization

After the group-specific transformation, standardize every feature column:

```text
x_scaled = (t - training_median(t)) / max(training_IQR(t) / 1.349, epsilon)
```

Clip only after this transformation, using the deliberately wide interval
`[-8, 8]`.  This limits pathological numerical leverage without aggressively
removing bright objects.

## Redshift target

For the scaled experiment, standardize `ZPHOT` affinely using the training
mean and standard deviation:

```text
y = (ZPHOT - mean_train) / std_train
```

Train both the mean and variance in this standardized target space.  Convert
predictions back before metrics and figures:

```text
predicted_z = mean_train + std_train * predicted_mean
predicted_variance_z = std_train^2 * predicted_variance
```

An affine target transform preserves the Gaussian interpretation.  A rank or
quantile transform would not.

## Frozen IoTFM embeddings

Continue applying ordinary train-fitted mean/std standardization to the frozen
IoTFM embeddings.  This is separate from catalogue scaling and is used by both
experiment variants, because embedding coordinates are already substantially
less pathological than raw catalogue measurements.

## Experiment controls

- `scripts/iotfm_mlp_gnlll.sh`: no physical catalogue/target scaling.
- `scripts/iotfm_mlp_gnlll_scaling.sh`: the physical robust transformations
  specified above.

The two launchers should otherwise remain matched.  They need distinct output
and cache directories only to prevent one experiment from overwriting the
other.  Scaling metadata and all fitted statistics must be saved in the run
manifest/checkpoints.

## Loader prerequisite

The existing curated CLAUDS split cache retains only the 11 cModel fluxes and
their 11 errors.  It cannot supply this experiment's full 55 flux, 11 radius,
and 55 error columns.  The GNLLL routine must use or create a dedicated
random-row cache that preserves all 121 required FITS columns and `ID`/`ZPHOT`.
