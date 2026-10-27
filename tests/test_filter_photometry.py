"""Physical mappings, provenance, label isolation and ablation consistency."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from aion_magnitude import filter_photometry as fp
from aion_magnitude.FM_Qwen import serialize_qwen_feature_row

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))


def runner():
    spec = importlib.util.spec_from_file_location("filter_aware_runner", ROOT / "notebooks/qwen_filter_aware.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def features(kind):
    row = {}
    for i, (_, value, error) in enumerate(fp.feature_columns(kind)):
        row[value] = 20 + i if kind == "clauds" else (-2.5 if i == 0 else i + .5)
        if error:
            row[error] = .1 + i
    if kind != "clauds":
        row["MW_EBV"] = .03
    return row


@pytest.mark.parametrize("kind", tuple(fp.SURVEY_BANDS))
def test_level0_is_byte_exact_and_rejects_labels(kind):
    f = features(kind)
    expected = serialize_qwen_feature_row(f, serialization=fp.legacy_config(kind))
    assert fp.serialize_photometry(f, catalogue=kind, level=0) == expected
    for leak in ("Z", "ZPHOT", "z_spec", "DESI_redshift", "redshift"):
        with pytest.raises(ValueError, match="leakage"):
            fp.serialize_photometry(dict(f, **{leak: .9}), catalogue=kind, level=2)


def test_registry_curves_reproduce_metadata_and_hashes():
    spec = importlib.util.spec_from_file_location("build_registry", ROOT / "scripts/build_filter_registry.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    registry = fp.load_registry(verify_curves=True)
    assert len(registry["filters"]) == 21
    for entry in registry["filters"].values():
        curve = np.loadtxt(fp.DEFAULT_REGISTRY.parent / entry["curve_path"], delimiter=",", skiprows=1)
        summary = module.response_summary(curve[:, 0], curve[:, 1])
        assert summary["pivot_nm"] == pytest.approx(entry["pivot_nm"], rel=1e-9)
        assert summary["width_nm"] == pytest.approx(entry["width_nm"], rel=1e-9)
        assert entry["wavelength_unit"] == "nm"
        assert len(entry["source_commit"]) == 40
    # Infrared W1 must not be confused with an optical or near-IR filter.
    assert registry["filters"]["WISE.W1"]["pivot_nm"] > 3000
    assert registry["filters"]["MegaCam.u"]["pivot_nm"] < registry["filters"]["MegaCam.u_star"]["pivot_nm"]


@pytest.mark.parametrize("release, optical", [(9010, ("DECam.g", "DECam.r", "DECam.z")),
    (9012, ("DECam.g", "DECam.r", "DECam.z")), (9011, ("BASS.g", "BASS.r", "MzLS.z"))])
def test_legacy_mapping_is_release_aware(release, optical):
    text = fp.serialize_photometry(features("desi-ls"), catalogue="desi-ls", level=2, legacy_release=release)
    keys = [line.split()[0] for line in text.splitlines()[1:-1]]
    assert keys == [f"filter={key}" for key in (*optical, "WISE.W1", "WISE.W2")]
    assert "flux=-2.50000 sigma=0.10000" in text
    assert "unit=AB_nanomaggy sigma_unit=AB_nanomaggy" in text
    assert fp.legacy_release_from_group(f"ls:{release}:10:2") == release


def test_unknown_legacy_instrument_fails_closed():
    for release in (None, 9000, 10000):
        with pytest.raises(ValueError, match="RELEASE"):
            fp.serialize_photometry(features("desi-ls"), catalogue="desi-ls", level=1, legacy_release=release)


def test_named_feature_mapping_survives_column_reordering():
    f = features("desi-ls")
    reordered = dict(reversed(list(f.items())))
    assert fp.serialize_photometry(f, catalogue="desi-ls", level=2, legacy_release=9011) == fp.serialize_photometry(reordered, catalogue="desi-ls", level=2, legacy_release=9011)


@pytest.mark.parametrize("kind", tuple(fp.SURVEY_BANDS))
def test_level1_level2_differ_only_by_physical_metadata(kind):
    registry = fp.load_registry()
    kwargs = dict(catalogue=kind, legacy_release=9010, registry=registry)
    one = fp.serialize_photometry(features(kind), level=1, **kwargs)
    two = fp.serialize_photometry(features(kind), level=2, **kwargs)
    import re
    assert re.sub(r" pivot_nm=\d+\.\d+ width_nm=\d+\.\d+", "", two) == one
    for line in two.splitlines()[1:-1]:
        key = line.split()[0].removeprefix("filter=")
        entry = registry["filters"][key]
        assert f"pivot_nm={entry['pivot_nm']:.3f}" in line
        assert f"width_nm={entry['width_nm']:.3f}" in line


def test_missing_flux_and_error_stay_missing_without_discarding_signed_flux():
    f = features("galaxiesml-spec")
    f["HSC_PDR2_cmodel_r_flux"] = np.nan
    f["HSC_PDR2_cmodel_r_flux_error"] = np.nan
    text = fp.serialize_photometry(f, catalogue="galaxiesml-spec", level=2)
    assert "flux=NA sigma=NA" in text
    assert "flux=-2.50000" in text


def test_clauds_raw_flux_harmonisation_keeps_negative_zero_and_masks():
    flux, sigma, valid = fp.clauds_flux_to_nanomaggy([-2, 0, 10, -99, np.nan], [1, 2, -99, 4, 1])
    np.testing.assert_allclose(flux[:3], np.array([-2, 0, 10]) * 10**(-.56))
    assert np.isnan(flux[3:]).all() and valid.tolist() == [True, True, True, False, False]
    assert np.isnan(sigma[2:]).all()
    f, e = fp.magnitude_to_nanomaggy([22.5, 25, np.nan], [.1, .2, .1])
    np.testing.assert_allclose(f[:2], [1, .1])
    assert e[0] == pytest.approx(np.log(10) / 2.5 * .1)
    with pytest.raises(ValueError, match="Raw CLAUDS"):
        fp.serialize_photometry(features("clauds"), catalogue="clauds", level=2, numerics="flux")
    with pytest.raises(ValueError, match="exact legacy"):
        fp.serialize_photometry(features("clauds"), catalogue="clauds", level=0, numerics="flux")


def test_same_cohort_ids_masks_and_labels_in_all_representations():
    m = runner()
    f = features("galaxiesml-spec")
    data = dict(features=np.tile(list(f.values()), (12, 1)), targetid=np.arange(2**53+1, 2**53+13, dtype=np.int64),
                group=np.array([f"hsc:{i}" for i in range(12)]), z=np.arange(12) / 10,
                split=np.array(["train"]*6 + ["val"]*4 + ["test"]*2), missing_mask=np.zeros((12,11),bool))
    small = m.select_pilot(data, 4, 3, 1023)
    before = m.sha_arrays(small)
    for level in (0,1,2):
        texts = fp.serialize_rows(small, {"feature_names":list(f)}, catalogue="galaxiesml-spec", level=level)
        assert len(texts) == 7
        assert m.sha_arrays(small) == before
        assert all("redshift=" not in t and "ZPHOT" not in t for t in texts)
    assert not np.any(small["split"] == "test")
    np.testing.assert_array_equal(small["targetid"], m.select_pilot(data, 4, 3, 1023)["targetid"])
    changed = dict(small, z=small["z"] + 99)
    assert fp.serialize_rows(changed, {"feature_names":list(f)}, catalogue="galaxiesml-spec", level=2) == texts


def test_token_audit_fails_before_truncation():
    def tok(texts, truncation, padding):
        assert not truncation and not padding
        return {"input_ids":[list(range(len(t))) for t in texts]}
    assert fp.audit_tokens(tok, ["ab", "abc"], 3)["truncated"] == 0
    with pytest.raises(ValueError, match="Complete records"):
        fp.audit_tokens(tok, ["abcd"], 3)


def test_portable_cohort_hash_rejects_python_object_arrays():
    m = runner()
    with pytest.raises(ValueError, match="non-object arrays"):
        m.sha_arrays({"split":np.array(["train","val"], dtype=object)})


def test_archived_fast_check_level0_inputs_are_identical():
    folder = ROOT / "fast-check-runs/1023/hsc-qlora-to-legacy-clauds"
    if not folder.exists():
        pytest.skip("local archived validation artifacts not bundled")
    summary = json.loads((folder / "summary.json").read_text())
    for kind in ("desi-ls", "clauds"):
        with np.load(folder / f"{kind}_predictions.npz", allow_pickle=False) as a:
            text = fp.serialize_rows({"features":a["features"]}, summary["targets"][kind], catalogue=kind, level=0)
        assert text == json.loads((folder / f"{kind}_inputs.json").read_text())["texts"]


def test_prepared_clauds_native_and_flux_bundles_are_portable_and_identical():
    folder = ROOT / "filter-aware-runs/1023/clauds-validation-portable"
    if not (folder / "manifest.json").exists():
        pytest.skip("local validated catalogue artifacts not bundled")
    m = runner()
    fingerprints = set()
    for level, numerics in [(0,"native"),(1,"native"),(2,"native"),(1,"flux"),(2,"flux")]:
        data, manifest, texts = m.load_prepared(folder, level, numerics)
        assert all(not a.dtype.hasobject for a in data.values())
        fingerprints.add(m.sha_arrays(data))
        assert not np.any(data["split"] == "test")
        assert len(texts) == len(data["z"])
    assert len(fingerprints) == 1


def test_trainer_none_test_dataset_does_not_evaluate_or_report_test(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch
    from torch import nn
    from aion_magnitude import qwen_posttraining as qp

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__(); self.embedding = nn.Embedding(16, 8)
        def forward(self, input_ids, **kw):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
        def save_pretrained(self, path):
            Path(path).mkdir()
    def tokenizer(texts, **kw):
        return dict(input_ids=torch.ones((len(texts),2),dtype=torch.long),
                    attention_mask=torch.ones((len(texts),2),dtype=torch.long))
    model = qp.QwenPhotoZModel(Backbone(), hidden_size=8, n_z_bins=5, head_hidden_dim=8)
    monkeypatch.setattr(qp, "create_qlora_photoz_model", lambda *a, **kw:(model, tokenizer, torch.device("cpu")))
    monkeypatch.setattr(qp.torch, "autocast", lambda *a, **kw:nullcontext())
    data = qp.TextRedshiftDataset(["a","b"], [.1,.2], ["1","2"])
    config = qp.QwenPosttrainingConfig(epochs=1, head_warmup_epochs=0, n_z_bins=5, batch_size=2, gradient_accumulation_steps=1)
    result = qp.train_qlora_photoz(train_dataset=data, val_dataset=data, test_dataset=None,
                                  output_dir=tmp_path, config=config)
    assert result["metadata"]["test_evaluated"] is False
    assert set(result["final_metrics"]) == {"val"}
    assert "test_evaluation" not in result
