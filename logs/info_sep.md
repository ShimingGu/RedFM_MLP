# Separating redshift signal from redshift uncertainty

Date: 2026-07-21

## Seb's likely idea

Use a heteroscedastic Gaussian regression model with separate paths for the
predicted redshift and its per-galaxy uncertainty.

The mean-redshift path uses the measured signal and size information:

```text
55 flux measurements + 11 Kron radii
                    |
                    v
        frozen IoTFM representation
                    |
                    v
          trainable redshift head
                    |
                    v
             predicted mean mu_z
```

The 55 columns are technically five flux estimators in each of 11 bands, not
55 pure morphology columns. Differences among aperture, PSF, Kron, and cModel
fluxes can nevertheless contain morphological, blending, and aperture
information. The 11 `RADIUS_KRON_*` columns are direct size proxies.

The uncertainty path uses the matching measurement errors and the inferred
redshift:

```text
55 FLUXERR columns + predicted mu_z
                    |
                    v
          trainable variance head
                    |
                    v
          predicted variance sigma_z^2
```

Feeding `mu_z` into the variance head allows the same photometric uncertainty
to map to different redshift uncertainty at different redshifts, where filter
coverage and colour-redshift degeneracies differ.

## Gaussian negative log likelihood

PyTorch's `GaussianNLLLoss` treats the target as a sample from a Gaussian with
a mean and variance predicted by the network:

```text
loss = 0.5 * [log(sigma_z^2) + (z_target - mu_z)^2 / sigma_z^2]
```

The residual term rewards an accurate redshift. The logarithmic term prevents
the model from minimizing the residual penalty merely by predicting infinite
uncertainty.

The variance head must produce a positive variance, not a standard deviation
or raw log-variance. A stable parameterization is:

```python
variance = softplus(raw_variance) + epsilon
loss = torch.nn.GaussianNLLLoss()(mean_redshift, target_redshift, variance)
```

The flux errors are not themselves redshift errors. Training against observed
redshift residuals teaches the model how photometric measurement uncertainty
propagates statistically into redshift uncertainty.

Reference:

- <https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.GaussianNLLLoss.html>

## Recommended architecture refinement

The variance head should probably receive more than the 55 flux errors and
`mu_z`. Formal flux errors alone cannot identify colour degeneracy, unusual
morphology, blending, or an out-of-distribution object.

A stronger first design is:

```text
mean head:
    signal features -> mu_z

variance head:
    signal representation + flux errors + stop_gradient(mu_z) -> sigma_z^2
```

Using `stop_gradient(mu_z)` (PyTorch `mu_z.detach()`) prevents the variance
head's use of the mean as an input from directly changing the mean network.
The Gaussian NLL still couples the two outputs through the likelihood.

## Safer training sequence

Naive joint heteroscedastic training can allow the variance head to explain
poor mean predictions by increasing the predicted variance. Large predicted
variance then downweights the corresponding mean residual, potentially
compromising the mean fit.

A safer experiment is:

1. Train the mean-redshift head with MSE, Huber loss, or the existing photo-z
   objective.
2. Hold the mean model fixed and train the variance head from its residuals.
3. Optionally fine-tune both jointly with Gaussian NLL and monitor mean quality,
   NLL, coverage, PIT, and calibration.

Relevant discussion of heteroscedastic-regression failure modes:

- <https://openreview.net/pdf?id=aPOpXlnV1T>

## Scientific limitations

### A single Gaussian may be insufficient

Photo-z posteriors can be asymmetric or multimodal because different redshifts
can produce similar colours. A single `(mu_z, sigma_z^2)` Gaussian cannot
represent catastrophic secondary solutions. Gaussian NLL should therefore be
tested first as a validation branch or auxiliary loss rather than immediately
replacing the existing 300-bin `p(z)` classifier.

If the Gaussian assumption fails, later alternatives include a Gaussian
mixture-density head or retaining the categorical `p(z)` head while predicting
an auxiliary uncertainty/calibration quantity.

### Meaning of the target

The current project target is catalogue `ZPHOT`. Consequently, the learned
variance describes scatter relative to that catalogue photo-z target. It is
not automatically uncertainty relative to the true physical redshift.
Spectroscopic targets are required for that stronger interpretation.

### Aleatoric versus epistemic uncertainty

This design primarily estimates heteroscedastic aleatoric uncertainty: how
noisy or ambiguous a particular observation is. It does not by itself quantify
epistemic uncertainty from limited training coverage or model ignorance.

## Minimal scientific comparison

Keep the galaxies, split, preprocessing, and mean-head capacity matched, then
compare:

1. deterministic mean regression;
2. a joint mean/variance Gaussian-NLL model;
3. a staged mean-then-variance model;
4. the existing 300-bin `p(z)` model.

Report point-estimate metrics, Gaussian NLL, interval coverage, PIT/calibration,
and catastrophic outliers. A useful uncertainty model should improve or retain
mean-redshift quality while assigning larger calibrated uncertainty to the
objects that are genuinely difficult.
