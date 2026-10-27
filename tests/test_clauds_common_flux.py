"""Physical units, missingness, native replay, and fixed band identities."""
import json
from pathlib import Path
import re

import numpy as np
import pytest

from aion_magnitude.filter_photometry import (
    CLAUDS_BANDS, clauds_flux_to_nanomaggy, magnitude_to_nanomaggy,
    load_registry, serialize_rows,
)
from aion_magnitude.clauds_common_flux import serialize_clauds_common_flux

ROOT = Path(__file__).resolve().parents[1]


def test_microjy_ab_roundtrip_and_uncertainty_jacobian():
    raw = np.array([0.01, 1., 100.])
    error = raw * .1
    f, sigma, valid = clauds_flux_to_nanomaggy(raw, error)
    # AB magnitudes of microJy measurements, including the near-infrared bands.
    magnitude = 23.9 - 2.5 * np.log10(raw)
    sigma_mag = 2.5 / np.log(10) * error / raw
    expected_f, expected_sigma = magnitude_to_nanomaggy(magnitude, sigma_mag)
    np.testing.assert_allclose(f, expected_f, rtol=1e-13)
    np.testing.assert_allclose(sigma, expected_sigma, rtol=1e-13)
    assert valid.all()
    assert f[1] == pytest.approx(10 ** ((22.5 - 23.9) / 2.5))
    assert f[1] != pytest.approx(10 ** ((22.5 - 23.0) / 2.5))


def test_observation_flags_signed_flux_and_missing_errors():
    flux = [-2, 0, -99, np.nan, 10, 11, 12, 13]
    error = [1, 2, 3, 4, 0, -99, np.nan, 1]
    f, sigma, valid = clauds_flux_to_nanomaggy(
        flux, error, not_observed=[False]*7+[True], no_data=[False]*6+[True,False])
    assert valid.tolist() == [True, True, False, False, True, True, False, False]
    assert f[0] < 0 and f[1] == 0
    assert np.isnan(f[[2,3,6,7]]).all()
    assert np.isnan(sigma[2:]).all()
    with pytest.raises(ValueError, match="align"):
        clauds_flux_to_nanomaggy([1,2], [1,2], no_data=[False])


def test_hsc_subset_deletes_only_actual_non_hsc_bands():
    features = {f"{b}_mag": 25. for b in CLAUDS_BANDS}
    f = np.arange(11, dtype=float) - 2
    e = np.ones(11); f[0] = np.nan; e[0] = np.nan
    texts = {}
    for level in (1,2):
        full = serialize_clauds_common_flux(features, f, e, level=level)
        hsc = serialize_clauds_common_flux(features, f, e, level=level, band_configuration="hsc")
        full_lines = full.splitlines()
        assert hsc.splitlines() == [line for line in full_lines
                                   if not line.startswith("filter=") or line.startswith("filter=HSC.")]
        assert [line.split()[0] for line in hsc.splitlines()[1:-1]] == [f"filter=HSC.{b}" for b in ("g","r","i","z","y")]
        assert "EBV=NA" in full and "flux=NA sigma=NA" in full
        assert "filter=MegaCam.u_star" in full and "filter=VIRCAM.Y" in full
        assert "unit=AB_nanomaggy sigma_unit=AB_nanomaggy" in full
        texts[level] = full
    assert re.sub(r" pivot_nm=\d+\.\d+ width_nm=\d+\.\d+", "", texts[2]) == texts[1]


def test_imputed_magnitudes_cannot_change_flux_strings():
    first = {f"{b}_mag": 10. for b in CLAUDS_BANDS}
    second = {f"{b}_mag": 30. for b in CLAUDS_BANDS}
    flux = np.full(11, np.nan); sigma = flux.copy()
    assert serialize_clauds_common_flux(first, flux, sigma, level=2) == serialize_clauds_common_flux(second, flux, sigma, level=2)
    assert "mag=" not in serialize_clauds_common_flux(first, flux, sigma, level=2)
    with pytest.raises(ValueError, match="uncertainties"):
        serialize_clauds_common_flux(first, np.ones(11), np.zeros(11), level=1)
    with pytest.raises(ValueError, match="leakage"):
        serialize_clauds_common_flux(dict(first, ZPHOT=2.), flux, sigma, level=1)


@pytest.mark.parametrize("level", (0,1,2))
def test_completed_native_target_inputs_remain_byte_identical(level):
    fast = ROOT / "fast-check-runs/1023/hsc-qlora-to-legacy-clauds"
    if not fast.exists():
        pytest.skip("local audited query archive unavailable")
    meta = json.loads((fast/'summary.json').read_text())["targets"]["clauds"]
    with np.load(fast/'clauds_predictions.npz', allow_pickle=False) as a:
        texts = serialize_rows(dict(features=a['features']), meta, catalogue="clauds", level=level)
    if level == 0:
        expected = json.loads((fast/'clauds_inputs.json').read_text())["texts"]
    else:
        p = ROOT/f"filter-aware-runs/1023/hsc-warmstart/level{level}/clauds_inputs.jsonl"
        expected = [json.loads(line)["text"] for line in p.read_text().splitlines()]
    assert texts == expected
