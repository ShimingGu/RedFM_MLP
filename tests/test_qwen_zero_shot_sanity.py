"""Validation-only selection and native-input contracts for the transfer pilot."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))
spec = importlib.util.spec_from_file_location(
    "zero_shot_sanity", NOTEBOOKS / "qwen_zero_shot_sanity.py"
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_selection_excludes_source_objects_other_splits_duplicates_and_out_of_range():
    data = dict(
        targetid=np.arange(10, 18, dtype=np.int64),
        split=np.array(["val", "val", "train", "test", "val", "val", "val", "val"]),
        z=np.array([0.1, 0.2, 0.3, 0.4, 4.6, np.nan, 0.7, 0.8]),
        group=np.array(["a", "b", "c", "d", "e", "f", "b", "h"]),
    )
    selected = pilot.select_queries(data, np.array([10]), 2, 1023)
    assert set(data["targetid"][selected]) == {11, 17}
    np.testing.assert_array_equal(
        selected, pilot.select_queries(data, np.array([10]), 2, 1023)
    )
    with pytest.raises(ValueError, match="eligible"):
        pilot.select_queries(data, np.array([10]), 3, 1023)


def test_native_serialization_keeps_survey_band_meaning_and_units():
    row = {"features": np.array([[-1.5, 0.2, np.nan]], dtype=np.float32)}
    text = pilot.native_texts(
        "desi-ls",
        row,
        {
            "feature_names": [
                "LS_DR9_tractor_g_flux",
                "LS_DR9_tractor_g_flux_error",
                "MW_EBV",
            ]
        },
    )[0]
    assert "schema=desi-ls_native_photometry_specz_v1" in text
    assert "LS_DR9_tractor_g_flux=-1.50000" in text
    assert "MW_EBV=NA" in text
    assert "spectroscopic_redshift" not in text
    clauds = pilot.native_texts(
        "clauds", {"features": np.array([[24.5]])}, {"feature_names": ["u_mag"]}
    )[0]
    assert "all_magnitudes_ab" in clauds
    assert "u_mag=24.50000" in clauds
    assert "ZPHOT" not in clauds


def test_prompt_audit_fails_instead_of_silently_truncating():
    def tokenizer(text, truncation):
        assert truncation is False
        return {"input_ids": list(range(len(text)))}

    assert pilot.audit_lengths(tokenizer, ["abc", "a"], 3)["truncated"] == 0
    with pytest.raises(ValueError, match="limit"):
        pilot.audit_lengths(tokenizer, ["abcd"], 3)


def test_interpolated_pit_handles_bin_boundaries_and_endpoints():
    spec = importlib.util.spec_from_file_location(
        "plot_zero_shot", NOTEBOOKS / "plot_qwen_zero_shot_sanity.py"
    )
    plotting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotting)
    pz = np.full((5, 2), 0.5)
    z = np.array([0, 0.5, 1, 1.5, 2])
    np.testing.assert_allclose(
        plotting.continuous_pit(pz, z, np.array([0, 1, 2])),
        np.array([0, 0.25, 0.5, 0.75, 1]),
    )


def test_reverse_hsc_coordinate_lookup_preserves_large_ids_and_query_order(tmp_path):
    import h5py

    spec = importlib.util.spec_from_file_location(
        "clauds_zero_shot", NOTEBOOKS / "qwen_clauds_zero_shot_sanity.py"
    )
    reverse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reverse)
    path = tmp_path / "hsc.hdf5"
    ids = np.array([9007199254740993, 9007199254740995], dtype=np.int64)
    with h5py.File(path, "w") as f:
        f["targetid"] = ids.astype("S20")
        f["DESI_fibermap/TARGET_RA"] = [10.0, 20.0]
        f["DESI_fibermap/TARGET_DEC"] = [-1.0, -2.0]
    actual = reverse.hsc_positions(ids[::-1], {"sources": [{"path": str(path)}]})
    np.testing.assert_array_equal(actual, [[20.0, -2.0], [10.0, -1.0]])
