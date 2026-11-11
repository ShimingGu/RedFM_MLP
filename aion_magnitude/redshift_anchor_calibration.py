"""Simple PDF calibration: one temperature plus <=4 smooth redshift-prior terms."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

Z_MIN, Z_MAX, N_Z = 0.0, 4.5, 300
EDGES = np.linspace(Z_MIN, Z_MAX, N_Z + 1)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])


def z_bin(z):
    return np.clip(np.searchsorted(EDGES, np.asarray(z), side="right") - 1, 0, N_Z - 1)


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=-1, keepdims=True)


def _design(z, n_prior):
    x = 2.0 * np.asarray(z, dtype=np.float64) / Z_MAX - 1.0
    cols = [np.ones_like(x)]
    if n_prior >= 1:
        cols.append(x)
    if n_prior >= 2:
        cols.append(0.5 * (3.0 * x ** 2 - 1.0))
    if n_prior >= 3:
        cols.append(0.5 * (5.0 * x ** 3 - 3.0 * x))
    return np.column_stack(cols[: n_prior + 1])


def apply_calibration(pz, temperature, prior_coeffs):
    pz = np.asarray(pz, dtype=np.float64)
    logp = np.log(np.clip(pz, 1e-300, None)) / float(temperature)
    if prior_coeffs is not None and len(prior_coeffs):
        logp = logp + _design(CENTERS, len(prior_coeffs) - 1) @ np.asarray(prior_coeffs, dtype=np.float64)
    logp = logp - logp.max(axis=-1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=-1, keepdims=True)


def fit_calibration(pz, z, weights, *, n_prior=4, temperature_only=False, l2=1e-3):
    pz = np.asarray(pz, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    target = z_bin(z)
    n = len(z)

    def objective(theta):
        log_t, coeffs = theta[0], theta[1:]
        logp = np.log(np.clip(pz, 1e-300, None)) / np.exp(log_t)
        if n_prior:
            logp = logp + _design(CENTERS, n_prior) @ coeffs
        logp = logp - logp.max(axis=1, keepdims=True)
        lognorm = np.log(np.exp(logp).sum(axis=1))
        ll = logp[np.arange(n), target] - lognorm
        return float(-np.sum(w * ll) + l2 * np.sum(coeffs ** 2))

    if temperature_only:
        x0 = np.array([0.0])
        bounds = [(-2.0, 2.5)]
        n_prior_used = 0
    else:
        x0 = np.zeros(1 + n_prior)
        bounds = [(-2.0, 2.5)] + [(-3.0, 3.0)] * n_prior
        n_prior_used = n_prior
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=500, ftol=1e-12))
    theta = result.x
    if temperature_only:
        coeffs = np.zeros(0)
    else:
        coeffs = theta[1:]
    return dict(
        temperature=float(np.exp(theta[0])),
        log_temperature=float(theta[0]),
        prior_coeffs=coeffs.tolist(),
        n_prior=int(n_prior_used),
        objective=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        n_iter=int(result.nit),
        l2=float(l2),
    )


def pdf_metrics(pz, z):
    pz = np.asarray(pz, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    cdf = np.cumsum(pz, axis=1)
    p16 = CENTERS[np.argmax(cdf >= 0.16, axis=1)]
    p50 = CENTERS[np.argmax(cdf >= 0.50, axis=1)]
    p84 = CENTERS[np.argmax(cdf >= 0.84, axis=1)]
    delta = (p50 - z) / (1.0 + z)
    target = z_bin(z)
    ce = -np.log(np.clip(pz[np.arange(len(z)), target], 1e-300, None))
    observed = (CENTERS[None, :] >= z[:, None]).astype(np.float64)
    crps = np.sum((cdf - observed) ** 2, axis=1) * np.mean(np.diff(CENTERS))
    med = float(np.median(delta))
    return dict(
        n=int(len(z)),
        mae=float(np.mean(np.abs(delta))),
        bias=float(np.mean(delta)),
        median_bias=float(med),
        outlier=float(np.mean(np.abs(delta) > 0.15)),
        nmad=float(1.48 * np.median(np.abs(delta - med))),
        ce=float(np.mean(ce)),
        crps=float(np.mean(crps)),
        coverage68=float(np.mean((z >= p16) & (z <= p84))),
        mean_width68=float(np.mean(p84 - p16)),
        p50=p50,
    )


def kish_ess(weights):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return float(1.0 / np.sum(w ** 2))
