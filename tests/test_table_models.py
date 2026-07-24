from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from aion_magnitude import table_models as tm


def make_catalogue_data(n_rows: int = 10) -> tm.CatalogueData:
    magnitudes = pd.DataFrame({
        f"{band}_mag": np.linspace(20.0, 24.0, n_rows, dtype=np.float32)
        for band in tm.MAGNITUDE_BANDS
    })
    return tm.CatalogueData(
        object_id=np.arange(100, 100 + n_rows, dtype=np.int64),
        source_row=np.arange(n_rows, dtype=np.int64),
        target=np.linspace(0.1, 1.0, n_rows, dtype=np.float32),
        magnitude_features=magnitudes,
        full_features=None,
        morphology_features=None,
        detected_redshift_columns=["ZPHOT", "Z_LOW68"],
    )


def test_redshift_columns_are_conservatively_detected_and_rejected():
    for name in (
        "ZPHOT", "ZPHOT_NIR", "Z_LOW68", "Z_HIGH68", "Z_CHI", "Z_PEAK",
        "Posterior-Log", "Likelihood-Log", "Likelihood-Log_star", "best_redshift",
    ):
        assert tm.is_redshift_related_column(name)
    for name in ("FLUX_CMODEL_HSC-G", "RADIUS_KRON_HSC-I", "u_mag"):
        assert not tm.is_redshift_related_column(name)

    try:
        tm.assert_no_redshift_features(["g_mag", "ZPHOT_NIR"])
    except RuntimeError as error:
        assert "ZPHOT_NIR" in str(error)
    else:
        raise AssertionError("redshift leakage was not rejected")


def test_full121_selection_requires_exact_feature_groups():
    flux = [f"{prefix}BAND{index}" for prefix in tm.FLUX_PREFIXES for index in range(11)]
    errors = [f"{prefix}BAND{index}" for prefix in tm.FLUX_ERROR_PREFIXES for index in range(11)]
    radii = [f"RADIUS_KRON_BAND{index}" for index in range(11)]
    selected = tm.select_full121_columns([*radii, *errors, *flux, "ZPHOT"])
    assert tuple(map(len, selected)) == (55, 55, 11)


def test_masking_and_imputation_use_training_rows_only():
    frame = pd.DataFrame({
        "u_mag": [20.0, 22.0, np.nan, 100.0],
        "g_mag": [1.0, 3.0, 9.0, np.nan],
    })
    split = np.array(["train", "train", "val", "test"], dtype=object)
    output, metadata = tm.impute_from_training(frame, split)
    assert output.loc[2, "u_mag"] == 21.0
    assert output.loc[3, "g_mag"] == 2.0
    assert metadata["columns"]["u_mag"]["fit_split"] == "train"

    target = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    masked = tm.build_masked_target(target, split)
    np.testing.assert_allclose(masked[:2], target[:2])
    assert np.isnan(masked[2:]).all()


def test_comparison_matrix_uses_explicit_image_representations():
    current = tm.COMPARISONS["noimage-aion_comparison"]
    assert [arm.feature_mode for arm in current] == ["magonly", "morphology"]
    assert [arm.image_mode for arm in current] == ["none", "none"]

    legacy = tm.COMPARISONS["noimage-aion_comparison_old1"]
    assert [arm.image_mode for arm in legacy] == ["none", "aion_fsq_unpooled"]

    first, second = tm.COMPARISONS["mlp_aionimage_comparison"]
    assert first.runner == "table" and first.image_mode == "aion_fsq_unpooled"
    assert second.runner == "mlp" and second.image_mode == "aion_mlp"
    assert second.feature_mode == "magonly"

    broad = tm.COMPARISONS["aion-original-compact_comparison"]
    assert [arm.image_mode for arm in broad] == [
        "none", "aion_raw", "aion_compact", "aion_compact_shuffled",
    ]
    shuffle = tm.COMPARISONS["image_shuffle-or-not_comparison"]
    assert [arm.image_mode for arm in shuffle] == [
        "aion_compact", "aion_compact_shuffled",
    ]


def test_morphology_arm_adds_catalogue_summaries_without_raw_tokens():
    data = make_catalogue_data(4)
    data.morphology_features = pd.DataFrame({
        name: np.linspace(0.1, 0.4, 4, dtype=np.float32)
        for name in tm.MORPHOLOGY_FEATURE_COLUMNS
    })
    split = np.array(["train", "train", "val", "test"], dtype=object)
    arm = tm.prepare_arm(
        tm.ArmSpec("aion_morphology", "morphology"),
        data,
        split,
        aion_features=None,
        timm_features=None,
    )
    assert list(arm.features.columns) == [
        *(f"{band}_mag" for band in tm.MAGNITUDE_BANDS),
        *tm.MORPHOLOGY_FEATURE_COLUMNS,
    ]
    assert arm.image_metadata["representation"] == "catalogue_morphology_summary"
    assert arm.image_metadata["n_image_features"] == len(
        tm.MORPHOLOGY_FEATURE_COLUMNS
    )


class MeanRegressor:
    def fit(self, features, target):
        assert "ZPHOT" not in features.columns
        self.mean = float(np.mean(target))
        return self

    def predict(self, features):
        return np.full(len(features), self.mean, dtype=np.float32)


def test_table_arm_fills_only_held_out_redshifts_and_saves_truth_separately():
    data = make_catalogue_data(10)
    split = np.array(["train"] * 6 + ["val"] + ["test"] * 3, dtype=object)
    features, imputation = tm.impute_from_training(data.magnitude_features, split)
    arm = tm.PreparedArm(
        spec=tm.ArmSpec("noimage", "magonly"),
        features=features,
        imputation=imputation,
    )
    args = argparse.Namespace(
        model="tabicl", n_estimators=1, model_path=None, seed=42,
        resume=False, save_input_table=True,
    )
    with TemporaryDirectory() as directory:
        result = tm.run_table_arm(
            arm, data, split, args, Path(directory),
            estimator_factory=lambda model, namespace: MeanRegressor(),
        )
        assert np.isnan(result.predictions[:6]).all()
        assert np.isfinite(result.predictions[6:]).all()
        saved = np.load(Path(directory) / "redshift_completion.npz")
        assert np.isnan(saved["masked_redshift"][6:]).all()
        np.testing.assert_allclose(saved["true_redshift"], data.target)
        np.testing.assert_allclose(saved["filled_redshift"][:6], data.target[:6])
        assert (Path(directory) / "model_input_table.npz").exists()


def test_image_tokens_are_decoded_into_named_fsq_factor_columns():
    with TemporaryDirectory() as directory:
        token_path = Path(directory) / "tokens.npy"
        tokens = np.array([
            [0, 1, 2, 3],
            [4, 5, 0, 1],
            [2, 3, 4, 5],
        ], dtype=np.uint16)
        np.save(token_path, tokens)
        product = {
            "image_token_ids_path": str(token_path),
            "image_token_row_indices": np.array([2, 0], dtype=np.int64),
            "metadata": {"aion_image_quantizer_levels": [3, 2]},
        }
        values, names = tm._image_token_matrix(product)
        assert values.shape == (2, 8)
        assert names[0] == "aion_fsq_f00_r00_c00"
        assert names[-1] == "aion_fsq_f01_r01_c01"
        np.testing.assert_array_equal(values[0, :4], [1.0, -1.0, 0.0, 1.0])
        np.testing.assert_array_equal(values[0, 4:], [-1.0, 0.0, 0.0, 0.0])


def test_legacy_compact_and_splitwise_shuffled_features_coexist():
    with TemporaryDirectory() as directory:
        token_path = Path(directory) / "tokens.npy"
        tokens = np.arange(6 * 16, dtype=np.uint16).reshape(6, 16)
        np.save(token_path, tokens)
        product = {
            "image_token_ids_path": str(token_path),
            "image_token_row_indices": np.arange(5, -1, -1, dtype=np.int64),
            "metadata": {"aion_image_quantizer_levels": [3, 3, 3, 3, 3]},
        }
        split = np.array(["train"] * 3 + ["test"] * 3, dtype=object)
        modes = ("aion_raw", "aion_compact", "aion_compact_shuffled")
        features = tm.build_aion_table_features(product, modes, split, seed=17)

        raw, raw_names, raw_metadata = features["aion_raw"]
        np.testing.assert_array_equal(raw, tokens[::-1].astype(np.float32))
        assert raw.shape == (6, 16)
        assert raw_names[0] == "aion_token_r00_c00"
        assert raw_metadata["source_commit"] == "c4ed1f0"

        compact, compact_names, compact_metadata = features["aion_compact"]
        shuffled, shuffled_names, shuffled_metadata = features[
            "aion_compact_shuffled"
        ]
        assert compact.shape == (6, 50)
        assert compact_names[0] == "aion_fsq_f00_global_mean"
        assert compact_names[-1] == "aion_fsq_f04_region_r1_c1_std"
        assert shuffled_names == compact_names
        assert compact_metadata["n_image_features"] == 50
        assert shuffled_metadata["shuffle_scope"] == "within_split"
        assert shuffled_metadata["shuffle_method"] == "single_cycle_derangement"

        for label in ("train", "test"):
            indices = np.flatnonzero(split == label)
            assert {
                tuple(row) for row in compact[indices]
            } == {
                tuple(row) for row in shuffled[indices]
            }
            assert not np.any(np.all(compact[indices] == shuffled[indices], axis=1))

        repeated = tm.build_aion_table_features(
            product, ("aion_compact_shuffled",), split, seed=17,
        )["aion_compact_shuffled"][0]
        np.testing.assert_array_equal(shuffled, repeated)


def test_image_token_decoder_rejects_ids_outside_fsq_vocabulary():
    with TemporaryDirectory() as directory:
        token_path = Path(directory) / "tokens.npy"
        np.save(token_path, np.array([[0, 1, 2, 6]], dtype=np.uint16))
        product = {
            "image_token_ids_path": str(token_path),
            "image_token_row_indices": np.array([0], dtype=np.int64),
            "metadata": {"aion_image_quantizer_levels": [3, 2]},
        }
        try:
            tm._image_token_matrix(product)
        except RuntimeError as error:
            assert "must be in [0, 6)" in str(error)
        else:
            raise AssertionError("out-of-vocabulary AION token ID was accepted")
