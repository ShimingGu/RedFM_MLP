"""Data-contract tests for the new native-catalogue, spec-z experiments."""

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from astropy.table import Table
import torch

from aion_magnitude import survey_catalogues as s

ROOT = Path(__file__).resolve().parents[1]


def runner():
    spec = importlib.util.spec_from_file_location(
        "survey_posttraining_test", ROOT / "notebooks/qwen_survey_posttraining.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def catalogues(tmp_path):
    n = 120
    ids = 39627762704586359 + np.arange(n, dtype=np.int64)
    z = np.linspace(0.1, 1.0, n)
    root = tmp_path / "desi"
    zpath = root / s.ZCAT_RELATIVE
    zpath.parent.mkdir(parents=True)
    rows = {
        "TARGETID": ids,
        "Z": z,
        "ZERR": np.full(n, 0.0001),
        "ZCAT_PRIMARY": np.ones(n, bool),
        "SPECTYPE": np.full(n, "GALAXY", dtype="U8"),
        "OBJTYPE": np.full(n, "TGT", dtype="U3"),
        "ZWARN": np.zeros(n, np.int64),
        "COADD_FIBERSTATUS": np.zeros(n, np.int64),
        "DELTACHI2": np.full(n, 100.0),
        "SURVEY": np.full(n, "main", dtype="U7"),
        "PROGRAM": np.full(n, "dark", dtype="U6"),
        "ZPHOT": np.full(n, 99.0),
    }
    rows["SPECTYPE"][0] = "QSO"
    rows["SPECTYPE"][1] = "STAR"
    rows["ZWARN"][2] = 1
    rows["ZCAT_PRIMARY"][3] = False
    rows["DELTACHI2"][4] = 20.0
    rows["COADD_FIBERSTATUS"][5] = 1
    rows["SURVEY"][6] = "sv3"
    Table(rows).write(zpath)
    photo = {
        "TARGETID": ids,
        "RELEASE": np.full(n, 9000),
        "BRICKID": np.ones(n, int),
        "OBJID": np.arange(n),
        "RA": np.arange(n, dtype=float),
        "DEC": np.zeros(n),
        "EBV": np.full(n, 0.01),
    }
    for b in s.BANDS["desi-ls"]:
        photo[f"FLUX_{b.upper()}"] = np.arange(n, dtype=float) + 1
        photo[f"FLUX_IVAR_{b.upper()}"] = np.full(n, 4.0)
        photo[f"NOBS_{b.upper()}"] = np.ones(n, int)
    photo["FLUX_G"][7] = -3.0
    photo["NOBS_W1"][8] = 0
    # A second target for the same LS imaging object must not cross splits.
    photo["OBJID"][-1] = photo["OBJID"][-2]
    directory = root / s.TRACTOR_RELATIVE
    directory.mkdir(parents=True)
    Table({k: v[::-1] for k, v in photo.items()}).write(
        directory / "tractorphot-nside4-hp000-iron.fits"
    )
    hpath = tmp_path / "hsc.hdf5"
    with h5py.File(hpath, "w") as f:
        f["targetid"] = ids.astype("S20")
        objects = 40955030018199425 + np.arange(n, dtype=np.int64)
        objects[-1] = objects[-2]
        f["object_id"] = objects.astype("S20")
        native = z.copy()
        native[9] = -0.1
        native[10] += 0.1
        f["DESI_fibermap/DESI_redshift"] = native
        f["DESI_fibermap/DESI_redshift_err"] = np.full(n, 0.0001)
        f["DESI_fibermap/EBV"] = np.full(n, 0.01)
        for b in s.BANDS["galaxiesml-spec"]:
            f[f"HSC_metadata/{b}_cmodel_mag"] = np.full(n, 22.5)
            f[f"HSC_metadata/{b}_cmodel_magsigma"] = np.full(n, 0.1)
        # No image/spectrum datasets: successful preparation proves neither is needed.
    return root, hpath, ids, z


def test_units_negative_flux_and_exact_large_ids():
    f, error, valid = s.flux_features([-2.0, 1.0, 2.0], [4.0, 0.0, 9.0])
    assert f[0] == -2 and error[0] == 0.5 and valid[0]
    assert np.isnan(f[1]) and not valid[1]
    f, error, valid = s.magnitude_features([22.5, 25.0, 99.0], [0.1, 0.2, 0.1])
    np.testing.assert_allclose(f[:2], [1.0, 0.1])
    np.testing.assert_allclose(error[0], 0.1 * np.log(10) / 2.5)
    assert not valid[2] and np.isnan(f[2])
    assert s.exact_ids(np.array([b"39627762704586359"]))[0] == 39627762704586359
    with pytest.raises(ValueError, match="never floating"):
        s.exact_ids(np.array([39627762704586359.0]))


def test_desi_joins_by_id_uses_specz_and_does_not_require_hsc(catalogues):
    root, hsc, ids, z = catalogues
    data, meta = s.build_cohort(
        kind="desi-ls",
        desi_root=root,
        hsc_file=hsc.with_name("absent.hdf5"),
        max_rows=None,
    )
    assert not set(ids[:7]) & set(data["targetid"])
    row = np.flatnonzero(data["targetid"] == ids[7])[0]
    assert data["features"][row, 0] == -3.0
    assert data["features"][row, 1] == 0.5
    assert data["z"][row] == pytest.approx(z[7])
    row = np.flatnonzero(data["targetid"] == ids[8])[0]
    assert np.isnan(data["features"][row, 6])
    assert len(np.unique(data["group"])) == len(data["group"])
    assert meta["duplicate_imaging_rows_removed_after_sampling"] == 1
    assert meta["target"] == "Z"
    assert meta["bands"] == ["g", "r", "z", "W1", "W2"]
    assert all("ZPHOT" not in name for name in meta["feature_names"])
    assert all(x["path"] != str(hsc) for x in meta["sources"])


def test_hsc_native_bands_quality_and_repeated_image_groups(catalogues):
    root, hsc, ids, z = catalogues
    data, meta = s.build_cohort(
        kind="galaxiesml-spec", desi_root=root, hsc_file=hsc, max_rows=None
    )
    assert not set(ids[:6]) & set(data["targetid"])
    assert ids[6] in data["targetid"]  # SV objects can legitimately appear in HSC.
    assert ids[9] not in data["targetid"]  # Negative native spec-z.
    assert (
        ids[10] not in data["targetid"]
    )  # Inconsistent archive/current reference label.
    assert meta["native_reference_redshift_disagreements_removed"] == 1
    assert meta["duplicate_imaging_rows_removed_after_sampling"] == 1
    assert meta["bands"] == ["g", "r", "i", "z", "y"]
    np.testing.assert_allclose(data["features"][:, 0], 1.0)
    assert not meta["image_pixels_read"] and not meta["spectral_pixels_read"]
    assert len(set(data["group"])) == len(data["group"])


def test_method_independent_cohort_cache_and_resume_contract(catalogues, tmp_path):
    module = runner()
    root, hsc, ids, z = catalogues
    common = [
        "--stage",
        "prepare",
        "--catalogue-kind",
        "galaxiesml-spec",
        "--desi-root",
        str(root),
        "--hsc-file",
        str(hsc),
        "--max-rows",
        "all",
        "--cache-root",
        str(tmp_path / "cache"),
        "--skip-tokenizer-audit",
    ]
    a = module.parser().parse_args(
        common + ["--method", "qlora", "--output-dir", str(tmp_path / "qlora")]
    )
    b = module.parser().parse_args(
        common + ["--method", "dora", "--output-dir", str(tmp_path / "dora")]
    )
    da, ma, fa = module.load_cohort(a, build=True)
    db, mb, fb = module.load_cohort(b, build=True)
    assert fa == fb and ma["cache_key"] == mb["cache_key"]
    np.testing.assert_array_equal(da["targetid"], db["targetid"])
    np.testing.assert_array_equal(da["split"], db["split"])
    config = module.training_config(a)
    assert (config.epochs, config.head_warmup_epochs, config.lora_rank) == (10, 3, 8)
    assert config.lora_learning_rate == 1e-5
    texts = module.texts_for(a, da, ma)
    assert "HSC_PDR2_cmodel_g_flux=" in texts[0]
    assert "3962776" not in texts[0]
    module.assert_contract(a, ma, create=True)
    a.seed += 1
    with pytest.raises(ValueError, match="different cohort"):
        module.assert_contract(a, ma)


def test_frozen_control_records_cohort_provenance_without_requiring_result_metadata(
    catalogues, tmp_path
):
    module = runner()
    root, hsc, _, _ = catalogues
    args = module.parser().parse_args(
        [
            "--stage",
            "frozen",
            "--catalogue-kind",
            "galaxiesml-spec",
            "--method",
            "qlora",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    data, metadata = s.build_cohort(
        kind="galaxiesml-spec", desi_root=root, hsc_file=hsc, max_rows=None
    )
    metadata["cache_key"] = "fixture"
    identity = {
        "cohort": "fixture",
        "model": args.qwen_model,
        "model_source": module.model_source(args),
        "max_length": args.max_length,
        "serialization": module.asdict(module.serialization(args)),
        "state": "kbit_prepared_padding_safe_last_v1",
    }
    cache = tmp_path / f"frozen_{module.digest(identity)[:24]}.pt"
    torch.save(
        {
            "identity": identity,
            "object_id": data["targetid"].tolist(),
            "embedding": torch.zeros(len(data["z"]), 4),
        },
        cache,
    )

    def train(*a, **kw):
        kw["output_dir"].mkdir(parents=True)
        return {"final_metrics": {}}

    with patch.object(module, "train_single_baseline", side_effect=train):
        module.frozen(args, data, metadata, tmp_path)
    result = torch.load(args.output_dir / "frozen/result.pt", weights_only=False)
    assert result["metadata"]["survey_cohort_cache_key"] == "fixture"
