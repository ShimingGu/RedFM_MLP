"""Physical SED transport for the 1110 anchor-recovery campaign.

The module is deliberately label-free: it only knows observed photometry, DESI
spectra, FSPS templates and HSC response curves.  Redshift labels enter only in
the fit/generation driver, never in serialized text.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

BANDS = ("g", "r", "i", "z", "y")
C_ANGSTROM_PER_S = 2.99792458e18
AB_ZERO_CGS = 3.631e-29
MAG_ZERO = 22.5
LN10_OVER_25 = np.log(10.0) / 2.5
HSC_FEATURE_NAMES = [f"HSC_PDR2_cmodel_{b}_{s}" for b in BANDS for s in ("flux", "flux_error")] + ["MW_EBV"]


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_hsc_filters(repo_root):
    """Load the versioned photon-throughput curves; wavelengths converted nm->Angstrom."""
    repo_root = Path(repo_root)
    registry = json.loads((repo_root / "configs/filter_registry_v1.json").read_text())
    out = {}
    for band in BANDS:
        entry = registry["filters"][f"HSC.{band}"]
        table = np.loadtxt(repo_root / "configs" / entry["curve_path"], delimiter=",", skiprows=1)
        wave_nm, response = table[:, 0], table[:, 1]
        response = np.clip(response, 0.0, None)
        good = np.isfinite(wave_nm) & np.isfinite(response)
        wave = wave_nm[good] * 10.0
        response = response[good]
        order = np.argsort(wave)
        wave, response = wave[order], response[order]
        denom = C_ANGSTROM_PER_S * np.trapezoid(response / wave, wave)
        out[band] = dict(
            wave=wave,
            response=response,
            denominator=denom,
            pivot_nm=entry["pivot_nm"],
            curve_sha256=entry["curve_sha256"],
            approximation=entry.get("approximation"),
        )
    return out


def band_flux_from_fnu(f_nu, filters, band):
    """Flat-f_nu sanity transform; returns the same f_nu for a constant spectrum."""
    f = filters[band]
    f_lambda = f_nu * C_ANGSTROM_PER_S / f["wave"] ** 2
    numerator = np.trapezoid(f_lambda * f["wave"] * f["response"], f["wave"])
    return numerator / f["denominator"]


def band_flux(wave_rest, luminosity, z, filters):
    """Observed photon-band flux for L_lambda(rest) transported to z.

    f_lambda(obs) = L_lambda(rest)/(1+z).  The result is in the same arbitrary
    luminosity-per-Angstrom units as ``luminosity``; a constant f_nu input
    therefore returns (1+z)*f_nu, as required by the observer-frame band
    integral.
    """
    out = np.empty(len(BANDS), dtype=np.float64)
    for j, band in enumerate(BANDS):
        f = filters[band]
        rest = f["wave"] / (1.0 + z)
        lum = np.interp(rest, wave_rest, luminosity, left=0.0, right=0.0)
        out[j] = np.trapezoid((lum / (1.0 + z)) * f["wave"] * f["response"], f["wave"]) / f["denominator"]
    return out


def band_flux_many(wave_rest, templates, z, filters):
    """Band fluxes for a template matrix (n_template, n_wave)."""
    out = np.empty((templates.shape[0], len(BANDS)), dtype=np.float64)
    for i in range(templates.shape[0]):
        out[i] = band_flux(wave_rest, templates[i], z, filters)
    return out


def band_flux_cache(wave_rest, templates, z_grid, filters):
    """Band flux cache used by the physical template comparator."""
    out = np.empty((templates.shape[0], len(z_grid), len(BANDS)), dtype=np.float32)
    for j, z in enumerate(z_grid):
        out[:, j, :] = band_flux_many(wave_rest, templates, float(z), filters)
    return out


_F99_MODEL = None


def _f99_curve(wave_ang):
    """F99 A(lambda)/A(V) with clamping outside the published 0.3-10 1/micron range."""
    global _F99_MODEL
    if _F99_MODEL is None:
        import warnings
        from dust_extinction.parameter_averages import F99
        _F99_MODEL = F99(Rv=3.1)
        warnings.filterwarnings("ignore", message="x has no units")
    wave_um = np.asarray(wave_ang, dtype=np.float64) / 1e4
    x = np.clip(1.0 / wave_um, 0.3000001, 9.999999)
    return _F99_MODEL(x)


def f99_curve(wave_ang):
    return _f99_curve(wave_ang)


def attenuation(wave_ang, av):
    """Intrinsic attenuation 10**(-0.4 A(lambda)) for A(V)=av."""
    return 10.0 ** (-0.4 * av * _f99_curve(wave_ang))


def mw_attenuation(wave_ang, ebv):
    """Milky-Way foreground attenuation for E(B-V)=ebv, R_V=3.1."""
    return 10.0 ** (-0.4 * ebv * 3.1 * _f99_curve(wave_ang))


def normalize_templates(templates, wave_rest, pivot=(4000.0, 6000.0)):
    """Normalise each template to unit median L_lambda in the rest optical pivot."""
    mask = (wave_rest >= pivot[0]) & (wave_rest <= pivot[1])
    scale = np.median(templates[:, mask], axis=1)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return templates / scale[:, None], scale


def interp_template_matrix(wave_rest, templates, wave_obs):
    """Interpolate every template onto observer-frame wavelengths."""
    out = np.empty((templates.shape[0], len(wave_obs)), dtype=np.float64)
    for i in range(templates.shape[0]):
        out[i] = np.interp(wave_obs, wave_rest, templates[i], left=0.0, right=0.0)
    return out


def bin_spectrum(wave, flux, ivar, edges):
    """Inverse-variance binning; returns flux, sigma, effective counts and mask."""
    flux = np.asarray(flux, dtype=np.float64)
    ivar = np.asarray(ivar, dtype=np.float64)
    good = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
    idx = np.searchsorted(edges, wave, side="right") - 1
    valid = good & (idx >= 0) & (idx < len(edges) - 1)
    idx = idx[valid]
    w = ivar[valid]
    f = flux[valid]
    sumw = np.bincount(idx, weights=w, minlength=len(edges) - 1)
    sumf = np.bincount(idx, weights=w * f, minlength=len(edges) - 1)
    count = np.bincount(idx, minlength=len(edges) - 1)
    out_flux = np.full(len(edges) - 1, np.nan)
    out_sigma = np.full(len(edges) - 1, np.nan)
    ok = sumw > 0
    out_flux[ok] = sumf[ok] / sumw[ok]
    out_sigma[ok] = 1.0 / np.sqrt(sumw[ok])
    return out_flux, out_sigma, count, ok


def read_spectrum(h5, row):
    wave = np.asarray(h5["spectrum/wave"][row], dtype=np.float64)
    flux = np.asarray(h5["spectrum/flux"][row], dtype=np.float64)
    ivar = np.asarray(h5["spectrum/ivar"][row], dtype=np.float64)
    return wave, flux, ivar


def spectrum_hdf5_index(h5_file):
    ids = np.array([int(x) for x in h5_file["targetid"][:]], dtype=np.int64)
    order = np.argsort(ids)
    return ids[order], order


def extinction_corrected_ebv(features):
    return np.asarray(features[..., 10], dtype=np.float64)


def features_from_fluxes(flux, sigma, ebv):
    out = np.empty(11, dtype=np.float64)
    out[0:10:2] = flux
    out[1:10:2] = sigma
    out[10] = ebv
    return out


_REGISTRY_CACHE = {}


def serialize_features(features, repo_root):
    import sys

    root = str(Path(repo_root))
    if root not in sys.path:
        sys.path.insert(0, root)
    from aion_magnitude.filter_photometry import load_registry, serialize_photometry

    if root not in _REGISTRY_CACHE:
        _REGISTRY_CACHE[root] = load_registry(Path(repo_root) / "configs/filter_registry_v1.json", verify_curves=True)
    value = dict(zip(HSC_FEATURE_NAMES, np.asarray(features, dtype=np.float64)))
    return serialize_photometry(
        value,
        catalogue="galaxiesml-spec",
        level=1,
        registry=_REGISTRY_CACHE[root],
    )


class CModelNoise:
    """Empirical cModel magnitude/error/missingness approximation constrained on T.

    T has five valid bands for every row, so the empirical valid fraction carries
    no censoring information.  The error-vs-magnitude relation is interpolated
    over T and extrapolated at the faint end with a photon-noise power law; a
    documented faint-end logistic dropout supplies the explicit censoring
    approximation.  Its mismatch is quantified on low-z V development photometry.
    """

    def __init__(self, features, valid, band_mag_edges=None):
        features = np.asarray(features, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)
        self.edges = band_mag_edges or np.arange(14.0, 29.01, 0.25)
        self.bands = {}
        for j, band in enumerate(BANDS):
            flux = features[:, 2 * j]
            sigma = features[:, 2 * j + 1]
            good = valid[:, j] & np.isfinite(flux) & np.isfinite(sigma) & (flux > 0) & (sigma > 0)
            mag = np.full(len(flux), np.nan)
            mag[good] = MAG_ZERO - 2.5 * np.log10(flux[good])
            ok = good & np.isfinite(mag)
            idx = np.digitize(mag[ok], self.edges) - 1
            keep = (idx >= 0) & (idx < len(self.edges) - 1)
            idx = idx[keep]
            sm = sigma[ok][keep]
            mm = mag[ok][keep]
            nbin = len(self.edges) - 1
            count = np.bincount(idx, minlength=nbin)
            med_sigma = np.full(nbin, np.nan)
            med_mag = np.full(nbin, np.nan)
            for b in range(nbin):
                m = idx == b
                if m.sum() >= 10:
                    med_sigma[b] = np.median(sm[m])
                    med_mag[b] = np.median(mm[m])
            good_bins = np.isfinite(med_mag) & np.isfinite(med_sigma) & (med_sigma > 0)
            self.bands[band] = dict(
                med_sigma=med_sigma,
                med_mag=med_mag,
                count=count,
                faint_mag=float(np.nanpercentile(mm, 99.0)),
                faint_sigma=float(np.nanmedian(sm[mm >= np.nanpercentile(mm, 97.0)])),
                dropout_center=float(np.nanpercentile(mm, 99.0) + 0.8),
                dropout_width=0.45,
                random_dropout=0.005,
            )

    def _pvalid(self, band, mag):
        d = self.bands[band]
        return 1.0 - d["random_dropout"] - (1.0 - d["random_dropout"]) / (
            1.0 + np.exp(-(np.asarray(mag) - d["dropout_center"]) / d["dropout_width"])
        )

    def _sigma(self, band, mag):
        d = self.bands[band]
        good = np.isfinite(d["med_mag"]) & np.isfinite(d["med_sigma"])
        if good.sum() == 0:
            return np.full_like(mag, np.nan)
        base = np.interp(mag, d["med_mag"][good], d["med_sigma"][good], left=d["med_sigma"][good][0], right=np.nan)
        faint = np.asarray(mag) > d["faint_mag"]
        if np.any(faint):
            # Background-limited flux errors are nearly constant; allow a modest
            # photon-noise rise rather than inventing arbitrary precision.
            base = np.asarray(base, dtype=np.float64)
            base[faint] = d["faint_sigma"] * 10.0 ** (0.2 * (np.asarray(mag)[faint] - d["faint_mag"]))
        return base

    def sample(self, model_flux, rng, min_valid=3, max_tries=20):
        model_flux = np.asarray(model_flux, dtype=np.float64)
        for _ in range(max_tries):
            flux = np.full(5, np.nan)
            sigma_out = np.full(5, np.nan)
            for j, band in enumerate(BANDS):
                f = model_flux[j]
                if not np.isfinite(f) or f <= 0:
                    continue
                mag = MAG_ZERO - 2.5 * np.log10(f)
                if not (0.0 < mag < 40.0):
                    continue
                if rng.random() > self._pvalid(band, mag):
                    continue
                s = self._sigma(band, mag)
                if not np.isfinite(s) or s <= 0:
                    continue
                s = s * np.exp(rng.normal(0.0, 0.15))
                obs = f + s * rng.normal()
                if obs <= 0:
                    continue
                obs_mag = MAG_ZERO - 2.5 * np.log10(obs)
                if not (0.0 < obs_mag < 40.0):
                    continue
                if s >= 9.21 * obs:
                    continue
                flux[j], sigma_out[j] = obs, s
            if np.isfinite(flux).sum() >= min_valid:
                return flux, sigma_out
        return None, None


PROPOSAL_BINS = np.array([0.4, 0.6, 0.8, 1.0, 1.5, 1.7])
PROPOSAL_PROBS = np.array([0.10, 0.15, 0.27, 0.30, 0.18])
PROPOSAL_PROBS = PROPOSAL_PROBS / PROPOSAL_PROBS.sum()


def proposal_density(z):
    """Explicit piecewise-constant density of the predeclared descendant proposal."""
    z = np.asarray(z, dtype=np.float64)
    j = np.clip(np.digitize(z, PROPOSAL_BINS[1:-1]), 0, len(PROPOSAL_PROBS) - 1)
    return PROPOSAL_PROBS[j] / (PROPOSAL_BINS[j + 1] - PROPOSAL_BINS[j])


def proposal_redshift(rng, n, kind="lowz"):
    """Predeclared descendant proposal: broad support with enriched 0.8-1.5."""
    which = rng.choice(len(PROPOSAL_PROBS), size=n, p=PROPOSAL_PROBS)
    z = np.empty(n)
    for j in range(len(PROPOSAL_PROBS)):
        m = which == j
        z[m] = rng.uniform(PROPOSAL_BINS[j], PROPOSAL_BINS[j + 1], size=m.sum())
    return z, proposal_density(z), PROPOSAL_BINS, PROPOSAL_PROBS


_COSMO = None


def cosmology():
    global _COSMO
    if _COSMO is None:
        from astropy.cosmology import FlatLambdaCDM
        _COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=2.725)
    return _COSMO


def age_universe(z, cosmo=None):
    cosmo = cosmo or cosmology()
    return cosmo.age(z).to_value("Gyr")


def luminosity_distance_ratio(z_from, z_to, cosmo=None):
    cosmo = cosmo or cosmology()
    return (cosmo.luminosity_distance(z_from).to_value("cm") / cosmo.luminosity_distance(z_to).to_value("cm")) ** 2
