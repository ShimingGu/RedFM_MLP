#!/usr/bin/env python3
"""Bounded physical/implementation correctness checks required before training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
sys.path.insert(0, str(ROOT))

from aion_magnitude.redshift_anchor_physics import (  # noqa: E402
    BANDS,
    CModelNoise,
    C_ANGSTROM_PER_S,
    band_flux,
    load_hsc_filters,
    mw_attenuation,
    serialize_features,
)


def read(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def check_flat_fnu(filters):
    wave = np.linspace(100.0, 30000.0, 30000)
    worst = 0.0
    for fnu in (1.0, 1e-29):
        luminosity = fnu * C_ANGSTROM_PER_S / wave ** 2
        for z in (0.0, 0.5, 1.0, 1.5):
            got = band_flux(wave, luminosity, z, filters)
            expected = fnu * (1.0 + z)
            worst = max(worst, float(np.max(np.abs(got / expected - 1.0))))
    assert worst < 1e-6, worst
    return dict(max_relative_error=worst, n_cases=8)


def check_identity_and_shift(filters):
    wave = np.linspace(500.0, 20000.0, 20000)
    luminosity = (wave / 5000.0) ** -1.2 * np.exp(-((wave - 4000.0) / 800.0) ** 2)
    direct = []
    for band in BANDS:
        f = filters[band]
        lum = np.interp(f["wave"], wave, luminosity)
        direct.append(np.trapezoid(lum * f["wave"] * f["response"], f["wave"]) / f["denominator"])
    got = band_flux(wave, luminosity, 0.0, filters)
    err = float(np.max(np.abs(got / np.asarray(direct) - 1.0)))
    assert err < 1e-8, err
    # A rest-frame line must move with (1+z); compare the analytic band integral
    # against a dense observer-frame resampling (independent of the code path).
    line = np.exp(-0.5 * ((wave - 6563.0) / 8.0) ** 2)
    sed = luminosity + 0.2 * line
    z = 0.9
    dense = np.linspace(3000.0, 12000.0, 90001)
    f_obs = np.interp(dense / (1.0 + z), wave, sed) / (1.0 + z)
    shift_err = 0.0
    for band in BANDS:
        f = filters[band]
        direct_shift = np.trapezoid(
            np.interp(f["wave"], dense, f_obs) * f["wave"] * f["response"], f["wave"]
        ) / f["denominator"]
        got_shift = band_flux(wave, sed, z, filters)[BANDS.index(band)]
        shift_err = max(shift_err, abs(got_shift / direct_shift - 1.0))
    assert shift_err < 1e-5, shift_err
    return dict(identity_max_relative_error=err, shift_max_relative_error=float(shift_err))


def check_measurement_masks(filters, paired):
    rows = np.flatnonzero(paired["category"] == "T")
    features = paired["hsc_features"][rows]
    noise = CModelNoise(features, paired["hsc_valid"][rows])
    rng = np.random.default_rng(1110)
    n_ok = 0
    mins = []
    texts = 0
    for i in range(2000):
        f = features[i]
        model = f[0:10:2].astype(np.float64)
        got, sig = noise.sample(model, rng)
        if got is None:
            continue
        assert np.isfinite(got).sum() >= 3
        assert np.all(np.isfinite(got) == np.isfinite(sig))
        assert np.all(got[np.isfinite(got)] > 0)
        assert np.all(sig[np.isfinite(sig)] > 0)
        out = np.empty(11)
        out[0:10:2], out[1:10:2], out[10] = got, sig, f[10]
        text = serialize_features(out, ROOT)
        lines = [line for line in text.splitlines() if line.startswith("filter=")]
        assert len(lines) == 5
        assert text.count("=NA") == 2 * (5 - int(np.isfinite(got).sum()))
        n_ok += 1
        mins.append(int(np.isfinite(got).sum()))
        texts += 1
    assert n_ok > 1800, n_ok
    return dict(sampled=n_ok, min_valid_bands=int(min(mins)), texts=texts, serializer_na_consistent=True)


def check_galaxy_separation(paired):
    frozen = read(OUT / "data/frozen/anchors.npz")
    lowz = read(OUT / "data/frozen/lowz_fit_subset.npz")
    e = read(OUT / "data/frozen/E_membership.npz")
    paired_ids = set(int(x) for x in paired["targetid"])
    assert not (set(int(x) for x in e["targetid"]) & paired_ids)
    assert not (set(int(x) for x in frozen["targetid"]) & set(int(x) for x in lowz["targetid"]))
    assert set(np.unique(frozen["role"])) <= {"fitting", "development"}
    fit_ids = set(int(x) for x in frozen["targetid"][frozen["role"] == "fitting"])
    dev_ids = set(int(x) for x in frozen["targetid"][frozen["role"] == "development"])
    assert not (fit_ids & dev_ids)
    assert not (set(int(x) for x in lowz["targetid"]) & dev_ids)
    return dict(E_not_paired=True, anchors_disjoint=True, fit_dev_disjoint=True, E_objects=int(len(e["targetid"])))


def check_serializer_identity(paired):
    lowz = read(OUT / "data/frozen/lowz_fit_subset.npz")
    h1 = np.load("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1102/highz-hsc-pilot-zcut0p6/splits/lowz_only_train.npz", allow_pickle=False)
    idx = {int(t): i for i, t in enumerate(h1["targetid"])}
    identical = 0
    for j in range(64):
        i = idx[int(lowz["targetid"][j])]
        text = serialize_features(lowz["features"][j], ROOT)
        identical += text == str(h1["text"][i])
    assert identical == 64, identical
    return dict(checked=64, byte_identical=identical)


def check_cmodel_development(paired):
    rows = np.flatnonzero(paired["category"] == "V")
    features = paired["hsc_features"][rows]
    noise = CModelNoise(paired["hsc_features"][paired["category"] == "T"], paired["hsc_valid"][paired["category"] == "T"])
    rng = np.random.default_rng(1111)
    ratios = []
    for i in range(0, len(rows), 7):
        model = features[i, 0:10:2].astype(np.float64)
        got, sig = noise.sample(model, rng)
        if got is None:
            continue
        ok = np.isfinite(got) & np.isfinite(features[i, 0:10:2])
        ratios.extend((sig[ok] / features[i, 1:10:2][ok]).tolist())
    ratios = np.asarray(ratios)
    return dict(n_pairs=int(len(ratios)), median_sigma_ratio=float(np.median(ratios)), p16=float(np.percentile(ratios, 16)), p84=float(np.percentile(ratios, 84)), note="V photometry is development data; only aggregate noise mismatch is reported")


def main():
    filters = load_hsc_filters(ROOT)
    paired = read("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1028/matched-training-design/paired_split/paired_cohort.npz")
    result = dict(
        flat_fnu=check_flat_fnu(filters),
        identity_shift=check_identity_and_shift(filters),
        measurement_masks=check_measurement_masks(filters, paired),
        galaxy_separation=check_galaxy_separation(paired),
        serializer_identity=check_serializer_identity(paired),
        cmodel_development=check_cmodel_development(paired),
    )
    path = OUT / "protocol/checks_physics.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
