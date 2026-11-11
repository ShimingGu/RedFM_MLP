"""Joint spectrum + HSC photometry fitting and physical descendant generation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import nnls

from .redshift_anchor_physics import (
    BANDS,
    CModelNoise,
    age_universe,
    attenuation,
    band_flux,
    f99_curve,
    band_flux_many,
    bin_spectrum,
    cosmology,
    interp_template_matrix,
    luminosity_distance_ratio,
    mw_attenuation,
    proposal_density,
    proposal_redshift,
    serialize_features,
)

SPEC_EDGES = np.linspace(3600.0, 9824.0, 801)


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def fit_quality(w, a_spec, a_tot, cal, spec, sigma_spec, phot, sigma_phot, tmat, fmat):
    model_spec = a_spec * cal * (tmat.T @ w)
    model_phot = a_tot * (fmat.T @ w)
    rs = (spec - model_spec) / np.maximum(sigma_spec, 1e-30)
    rp = (phot - model_phot) / np.maximum(sigma_phot, 1e-30)
    return dict(
        chi2_spec=float(np.mean(rs ** 2)),
        chi2_phot=float(np.mean(rp ** 2)),
        rms_spec=float(np.sqrt(np.mean(rs ** 2))),
        rms_phot=float(np.sqrt(np.mean(rp ** 2))),
        median_frac_spec=float(np.median(np.abs((spec - model_spec) / np.maximum(np.abs(spec), 1e-30)))),
    )


def _calibration(x, spec, model, sigma, mask, iterations=3):
    cal = np.ones_like(spec)
    good = mask & np.isfinite(spec) & np.isfinite(model) & (spec > 0) & (model > 0) & np.isfinite(sigma) & (sigma > 0)
    if good.sum() < 10:
        return cal
    for _ in range(iterations):
        ratio = np.log(np.clip(spec[good] / (model[good] * cal[good]), 0.2, 5.0))
        weight = (model[good] * cal[good] / sigma[good]) ** 2
        weight = np.clip(weight, 0, np.percentile(weight, 99.0))
        coeff = np.polyfit(x[good], ratio, 2, w=np.sqrt(weight))
        test = np.polyval(coeff, x[good])
        resid = ratio - test
        keep = np.abs(resid) < max(0.15, 3.0 * np.median(np.abs(resid - np.median(resid))) * 1.4826)
        if keep.sum() >= 10:
            coeff = np.polyfit(x[good][keep], ratio[keep], 2, w=np.sqrt(weight[keep]))
        poly = np.polyval(coeff, x)
        poly -= np.median(poly)
        cal = np.exp(np.clip(poly, -0.7, 0.7))
    return cal


def fit_object(
    *,
    basis,
    filters,
    z,
    ebv,
    wave,
    flux,
    ivar,
    phot_flux,
    phot_sigma,
    valid_bands,
    rng,
    n_draws=16,
    smooth_sigma=2.0,
):
    spec, sigma, _, good = bin_spectrum(wave, flux, ivar, SPEC_EDGES)
    centers = 0.5 * (SPEC_EDGES[:-1] + SPEC_EDGES[1:])
    # Correct the common Milky-Way foreground from both observables. HSC PDR2
    # cmodel fluxes and DESI coadds are as-observed; the fit is rest-frame.
    att = mw_attenuation(centers, ebv)
    spec_raw = spec / att
    sigma_raw = sigma / att
    good &= np.isfinite(spec_raw) & np.isfinite(sigma_raw) & (sigma_raw > 0)
    if good.sum() < 30:
        return None
    spec_s = gaussian_filter1d(np.nan_to_num(spec_raw), smooth_sigma, mode="nearest")
    spec_s[~good] = np.nan
    sigma_s = np.sqrt(gaussian_filter1d(np.nan_to_num(sigma_raw ** 2), smooth_sigma, mode="nearest"))
    sigma_s[~good] = np.nan
    # A small model-error term prevents over-weighting template mismatch.
    sigma_eff = np.sqrt(sigma_s ** 2 + (0.03 * np.abs(spec_s)) ** 2)

    pivot = np.array([filters[b]["pivot_nm"] * 10.0 for b in BANDS])
    phot = np.asarray(phot_flux, float).copy()
    psig = np.asarray(phot_sigma, float).copy()
    att_p = mw_attenuation(pivot, ebv)
    phot = phot / att_p
    psig = psig / att_p
    valid = np.asarray(valid_bands, bool) & np.isfinite(phot) & np.isfinite(psig) & (psig > 0)
    if valid.sum() < 3:
        return None
    psig_eff = np.sqrt(psig ** 2 + (0.04 * np.abs(phot)) ** 2)

    rest = centers / (1.0 + z)
    # A stellar population cannot be older than the Universe at the object's
    # redshift.  Restrict the basis before fitting; otherwise the fit can put
    # most weight on 11-Gyr templates at z~0.5 and transport becomes unphysical.
    allowed = basis["age_gyr"] <= max(0.02, 0.95 * age_universe(z))
    if allowed.sum() < 3:
        allowed = basis["age_gyr"] <= float(np.min(basis["age_gyr"])) * 2.0
    sub_flux = basis["flux"][allowed]
    tmat = interp_template_matrix(basis["wave"], sub_flux, rest)
    fmat = band_flux_many(basis["wave"], sub_flux, z, filters)
    n_t = tmat.shape[0]
    x = (np.log10(centers) - np.log10(5000.0)) / 0.2

    # Initialise from photometry alone.
    ap = fmat[:, valid].T / psig_eff[valid][:, None]
    bp = phot[valid] / psig_eff[valid]
    w, _ = nnls(ap, bp)
    if w.sum() <= 0:
        w = np.ones(n_t) / n_t
    w /= w.sum()
    model_band = fmat.T @ w
    a_tot = float(np.sum(model_band[valid] * phot[valid] / psig_eff[valid] ** 2) /
                  np.sum(model_band[valid] ** 2 / psig_eff[valid] ** 2))
    model_spec = tmat.T @ w
    a_spec = float(np.sum(spec_s[good] * model_spec[good] / sigma_eff[good] ** 2) /
                   np.sum(model_spec[good] ** 2 / sigma_eff[good] ** 2))
    cal = np.ones_like(centers)

    def solve_w(extra_mask=None):
        nonlocal w, a_spec, a_tot, cal
        mask = good.copy()
        if extra_mask is not None:
            mask &= extra_mask
        model_spec = tmat.T @ w
        cal = _calibration(x, spec_s, a_spec * model_spec, sigma_eff, mask)
        for _ in range(4):
            model_spec = tmat.T @ w
            a_spec = float(np.sum(cal[mask] * model_spec[mask] * spec_s[mask] / sigma_eff[mask] ** 2) /
                           np.sum((cal[mask] * model_spec[mask]) ** 2 / sigma_eff[mask] ** 2))
            model_band = fmat.T @ w
            a_tot = float(np.sum(model_band[valid] * phot[valid] / psig_eff[valid] ** 2) /
                          np.sum(model_band[valid] ** 2 / psig_eff[valid] ** 2))
            aspec_cols = (a_spec * cal[mask, None] * tmat[:, mask].T) / sigma_eff[mask, None]
            aphot_cols = (a_tot * fmat[:, valid].T) / psig_eff[valid, None]
            design = np.vstack([aspec_cols, aphot_cols])
            target = np.concatenate([spec_s[mask] / sigma_eff[mask], phot[valid] / psig_eff[valid]])
            w_new, _ = nnls(design, target, maxiter=1000)
            if w_new.sum() <= 0:
                break
            w_new /= w_new.sum()
            if np.max(np.abs(w_new - w)) < 1e-4:
                w = w_new
                break
            w = w_new
        model_spec = tmat.T @ w
        cal = _calibration(x, spec_s, a_spec * model_spec, sigma_eff, mask)
        model_spec = tmat.T @ w
        a_spec = float(np.sum(cal[mask] * model_spec[mask] * spec_s[mask] / sigma_eff[mask] ** 2) /
                       np.sum((cal[mask] * model_spec[mask]) ** 2 / sigma_eff[mask] ** 2))
        model_band = fmat.T @ w
        a_tot = float(np.sum(model_band[valid] * phot[valid] / psig_eff[valid] ** 2) /
                      np.sum(model_band[valid] ** 2 / psig_eff[valid] ** 2))

    solve_w()
    # One robust pass that removes sky-line/residual outliers from the spectrum.
    model_spec = a_spec * cal * (tmat.T @ w)
    resid = (spec_s - model_spec) / np.maximum(sigma_eff, 1e-30)
    extra = np.abs(resid) < 4.0
    extra &= np.isfinite(extra)
    solve_w(extra)

    quality = fit_quality(w, a_spec, a_tot, cal, spec_s, sigma_eff, phot, psig_eff, tmat, fmat)
    # Covariance on the non-negative active set; a small simplex floor retains
    # genuine SED diversity when the active set is nearly singular.
    active = np.flatnonzero(w > 1e-6)
    draws = np.empty((n_draws, n_t), dtype=np.float32)
    if len(active) == 0:
        draws[:] = w
    else:
        aspec_cols = (a_spec * cal[good, None] * tmat[:, good].T) / sigma_eff[good, None]
        aphot_cols = (a_tot * fmat[:, valid].T) / psig_eff[valid, None]
        design = np.vstack([aspec_cols, aphot_cols])
        a_act = design[:, active]
        cov = np.linalg.pinv(a_act.T @ a_act + 1e-6 * np.eye(len(active)))
        scale = max(1.0, quality["chi2_spec"])
        try:
            chol = np.linalg.cholesky(cov * scale + 1e-10 * np.eye(len(active)))
        except np.linalg.LinAlgError:
            chol = np.diag(np.sqrt(np.maximum(np.diag(cov * scale), 0.0)))
        for j in range(n_draws):
            delta = chol @ rng.normal(size=len(active))
            wd = w.copy()
            wd[active] = np.clip(wd[active] + delta, 0.0, None)
            if wd.sum() <= 0:
                wd = w.copy()
            wd /= wd.sum()
            wd = 0.97 * wd + 0.03 * rng.dirichlet(np.ones(n_t))
            wd /= wd.sum()
            draws[j] = wd.astype(np.float32)

    n_all = basis["flux"].shape[0]
    w_full = np.zeros(n_all, dtype=np.float64)
    w_full[allowed] = w
    draws_full = np.zeros((n_draws, n_all), dtype=np.float32)
    draws_full[:, allowed] = draws
    age = basis["age_gyr"].astype(np.float64)
    zmet = basis["metallicity"].astype(np.float64)
    av = basis["av"].astype(np.float64)
    neb = basis["nebular"].astype(np.float64)
    coverage_lo = 3600.0 / (1.0 + z)
    coverage_hi = 9824.0 / (1.0 + z)
    coverage = max(0.0, (min(coverage_hi, 10000.0) - max(coverage_lo, 1000.0)) / 9000.0)
    return dict(
        w=w_full.astype(np.float32),
        w_draws=draws_full,
        a_spec=float(a_spec),
        a_tot=float(a_tot),
        cal=cal.astype(np.float32),
        quality=quality,
        ebv=float(ebv),
        coverage_fraction=float(coverage),
        rest_lo=float(coverage_lo),
        rest_hi=float(coverage_hi),
        n_spec_bins=int(good.sum()),
        median_snr=float(np.nanmedian(spec_s / np.maximum(sigma_s, 1e-30))),
        mean_age=float(np.sum(w_full * age)),
        mean_zmet=float(np.sum(w_full * zmet)),
        mean_av=float(np.sum(w_full * av)),
        mean_neb=float(np.sum(w_full * neb)),
        effective_templates=float(1.0 / np.sum(w_full ** 2)),
        n_templates_allowed=int(allowed.sum()),
        valid_bands=valid.copy(),
        phot_dered=phot.astype(np.float32),
        phot_sigma_dered=psig.astype(np.float32),
        spec_wave=centers.astype(np.float32),
        spec_flux=spec_s.astype(np.float32),
        spec_sigma=sigma_eff.astype(np.float32),
        cal_mask=good.copy(),
    )


def generate_descendants(
    *,
    basis,
    filters,
    noise,
    fits,
    parent_ids,
    parent_roles,
    parent_z,
    parent_a_tot,
    parent_w_draws,
    parent_n,
    ebv_pool,
    rng,
    repo_root,
    descendants_per_role=None,
    anchor_nearby_fraction=0.9,
):
    """Generate descendants with parent-local redshift transport.

    Low-z source SEDs are transported only mildly and mostly downward in z; the
    real high-z anchors supply the [0.8,1.5) population.  This avoids using a
    low-z galaxy as an unchanged high-z galaxy, which would be an evolution claim
    the data do not support.
    """
    cosmo = cosmology()
    z_table = np.linspace(0.3, 1.75, 146)
    age_table = np.asarray([cosmo.age(z).to_value("Gyr") for z in z_table])
    dl_table = np.asarray([cosmo.luminosity_distance(z).to_value("cm") for z in z_table])
    dl_parent = np.interp(parent_z, z_table, dl_table)
    flux64 = basis["flux"].astype(np.float64)
    k_basis = f99_curve(basis["wave"])
    counts = descendants_per_role or {"lowz": 15, "anchor": 150}
    tasks = np.repeat(np.arange(parent_n), [int(counts[str(r)]) for r in parent_roles])
    rng.shuffle(tasks)
    rows = []
    reject = 0
    attempts = 0
    for p in tasks:
        role = str(parent_roles[p])
        attempts += 1
        if role == "lowz":
            zg = float(np.clip(float(parent_z[p]) + rng.uniform(-0.10, 0.02), 0.02, 0.62))
            pd = 1.0 / 0.12
        else:
            zg = float(np.clip(float(parent_z[p]) + rng.uniform(-0.10, 0.15), 0.6, 1.7))
            pd = 1.0 / 0.25
        draw = int(rng.integers(parent_w_draws.shape[1]))
        w = parent_w_draws[p, draw].astype(np.float64)
        tau = float(np.interp(zg, z_table, age_table))
        allowed = basis["age_gyr"] <= max(0.02, 0.95 * tau)
        if allowed.sum() < 3:
            allowed = basis["age_gyr"] <= float(np.min(basis["age_gyr"])) * 2.0
        w = np.where(allowed, w, 0.0)
        if w.sum() <= 0:
            w = allowed.astype(np.float64)
        w /= w.sum()
        sed = w @ flux64
        dav = float(np.clip(rng.normal(0.0, 0.12), -0.2, 0.3))
        sed = sed * 10.0 ** (-0.4 * dav * k_basis)
        dlum = float(rng.normal(0.0, 0.25))
        amp = float(parent_a_tot[p]) * 10.0 ** dlum * (float(dl_parent[p]) / float(np.interp(zg, z_table, dl_table))) ** 2
        ebv = float(np.clip(rng.choice(ebv_pool) * np.exp(rng.normal(0.0, 0.30)), 0.001, 0.5))
        k_obs = np.interp(basis["wave"] * (1.0 + zg), basis["wave"], k_basis)
        sed = sed * 10.0 ** (-0.4 * ebv * 3.1 * k_obs)
        model = amp * band_flux(basis["wave"], sed, zg, filters)
        sampled = noise.sample(model, rng)
        if sampled[0] is None:
            reject += 1
            continue
        flux, sigma = sampled
        if np.isfinite(flux).sum() < 3:
            reject += 1
            continue
        features = np.empty(11, dtype=np.float64)
        features[0:10:2] = flux
        features[1:10:2] = sigma
        features[10] = ebv
        text = serialize_features(features, repo_root)
        if len(rows) % 5000 == 0:
            print(f"generation rows={len(rows)} attempts={attempts} rejected={reject}", flush=True)
        rows.append(dict(
            synthetic_id=np.int64(-(len(rows) + 1)),
            targetid=np.int64(parent_ids[p]),
            parent_role=role,
            parent_index=np.int64(p),
            parent_z=np.float32(parent_z[p]),
            z=np.float32(zg),
            proposal_density=np.float32(pd),
            sed_draw=np.int16(draw),
            luminosity_offset=np.float32(dlum),
            av_offset=np.float32(dav),
            ebv=np.float32(ebv),
            valid_bands=np.int8(np.isfinite(flux).sum()),
            age_masked=int((~allowed).sum()),
            features=features.astype(np.float32),
            text=text,
        ))
    print(f"generated={len(rows)} rejected={reject} attempts={attempts}", flush=True)
    return rows
