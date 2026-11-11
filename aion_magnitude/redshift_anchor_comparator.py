"""Cheap physical template-likelihood comparator on the native redshift grid."""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, logsumexp

from .redshift_anchor_physics import BANDS, age_universe, mw_attenuation


def template_pdf(features, cache, z_grid, *, ebv=None, model_error=0.05, max_age_fraction=0.95, pivots=None, amplitude="profile"):
    """Marginalise nonnegative amplitude and the permitted SED templates.

    The returned array is a normalized PDF on the supplied native centers.  It is
    a comparator, not a training input, and uses no redshift label beyond the
    object's own permitted catalogue entry when its score is computed.
    """
    features = np.asarray(features, dtype=np.float64)
    flux = features[0:10:2]
    sigma = features[1:10:2]
    if ebv is None:
        ebv = float(features[10])
    valid = np.isfinite(flux) & np.isfinite(sigma) & (flux > 0) & (sigma > 0)
    if valid.sum() < 3:
        return np.full(len(z_grid), np.nan)
    pivot = np.asarray(pivots, dtype=np.float64) if pivots is not None else np.array([479.1, 623.1, 771.9, 891.6, 976.2]) * 10.0
    mw = mw_attenuation(pivot, ebv)[valid]
    d = flux[valid]
    sig = np.sqrt(sigma[valid] ** 2 + (model_error * np.abs(d)) ** 2)
    template_flux = cache["band_flux"][:, :, :][:, :, valid].astype(np.float64)
    age = cache["age_gyr"].astype(np.float64)
    logp = np.full(len(z_grid), -np.inf)
    for j, z in enumerate(z_grid):
        allowed = age <= max(0.02, max_age_fraction * age_universe(z))
        if allowed.sum() < 3:
            continue
        m = template_flux[allowed, j, :] * mw[None, :]
        precision = 1.0 / sig ** 2
        C = np.sum(m ** 2 * precision[None, :], axis=1)
        B = np.sum(d[None, :] * m * precision[None, :], axis=1)
        D = np.sum(d ** 2 * precision)
        good = C > 0
        if not good.any():
            continue
        C, B = C[good], B[good]
        if amplitude == "flat":
            log_i = -0.5 * np.maximum(D - B ** 2 / C, 0.0) + 0.5 * np.log(np.pi / (2.0 * C)) + log_ndtr(B / np.sqrt(C))
        else:
            # Profile over the nonnegative amplitude.  The flat-amplitude
            # marginalisation carries a 1/sqrt(C) volume term that biases the
            # development PDFs high; the profile is the stable approximation
            # used for the comparator reference.
            log_i = -0.5 * np.maximum(D - B ** 2 / C, 0.0)
        logp[j] = logsumexp(log_i) - np.log(len(C))
    if not np.isfinite(logp).any():
        return np.full(len(z_grid), np.nan)
    logp -= np.max(logp)
    p = np.exp(logp)
    p[~np.isfinite(p)] = 0.0
    total = p.sum()
    return p / total if total > 0 else np.full(len(z_grid), np.nan)


def summarize_pdf(p, z_grid, z_spec):
    p = np.asarray(p, float)
    if not np.isfinite(p).all() or p.sum() <= 0:
        return None
    p = p / p.sum()
    cdf = np.cumsum(p)
    q = {name: float(z_grid[np.searchsorted(cdf, qq)]) for name, qq in (("p16", 0.16), ("p50", 0.50), ("p84", 0.84))}
    delta = (q["p50"] - z_spec) / (1.0 + z_spec)
    return dict(z_p50=q["p50"], z_p16=q["p16"], z_p84=q["p84"], delta=float(delta), coverage68=bool(q["p16"] <= z_spec <= q["p84"]), crps=float(np.sum((cdf - (z_grid >= z_spec).astype(float)) ** 2) * np.mean(np.diff(z_grid))))
