from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from aion_magnitude.clauds_bands import select_catalogue_row_indices
from aion_magnitude.dataset import clauds_redshift_filter_mask
from aion_magnitude.morphology import (
    AIONMagnitudeMorphologyResidualPhotoZModel,
    AIONMorphologyConfig,
    FSQTokenDecoder,
    MorphologyResidualPhotoZModel,
    build_morphology_population_report,
    discover_morphology_image_paths,
    format_morphology_population_report,
    make_magnitude_config,
    resolve_morphology_paths,
)


class MorphologyModuleTest(unittest.TestCase):
    def test_morphology_config_preserves_sampling_and_open_max(self) -> None:
        config = AIONMorphologyConfig(
            max_rows=100,
            sample_mode="random",
            sample_seed=7,
            redshift_include_max=False,
        ).normalized()

        paths = resolve_morphology_paths(config)
        self.assertIn("random_s7", str(paths["morphology_tag"]))
        self.assertIn("openmax", str(paths["morphology_tag"]))
        self.assertIn("scale_1", str(paths["morphology_tag"]))
        self.assertIn("cov_0p9", str(paths["morphology_tag"]))

    def test_discovers_every_science_tile_and_requires_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "nested"
            nested.mkdir()
            science_paths = [
                root / "Mega-uS_10054_0c0.fits",
                nested / "Mega-u_9813_8c8.fits",
            ]
            for science_path in science_paths:
                science_path.touch()
                science_path.with_name(f"{science_path.stem}.weight.fits").touch()

            discovered = discover_morphology_image_paths(root)
            self.assertEqual(discovered, sorted(science_paths))

            science_paths[1].with_name(f"{science_paths[1].stem}.weight.fits").unlink()
            with self.assertRaises(FileNotFoundError):
                discover_morphology_image_paths(root)

    def test_grizy_only_mlp_config_has_no_other_band_features(self) -> None:
        config = AIONMorphologyConfig(
            max_rows=None,
            extra_bands=(),
            model_kinds=("photometry", "morphology"),
        )
        magnitude_config = make_magnitude_config(config)
        self.assertFalse(magnitude_config.use_aion_embedding)
        self.assertTrue(magnitude_config.use_mlp_features)
        self.assertTrue(magnitude_config.include_grizy_in_mlp)
        self.assertEqual(tuple(magnitude_config.extra_bands), ())
        self.assertIsNone(magnitude_config.max_rows)

    def test_all_magnitude_mlp_config_preserves_photometry_splits(self) -> None:
        config = AIONMorphologyConfig(
            max_rows=None,
            model_kinds=("photometry", "morphology"),
            preserve_photometry_splits=True,
        ).normalized()
        magnitude_config = make_magnitude_config(config)

        self.assertEqual(
            tuple(magnitude_config.extra_bands),
            ("u", "u_star", "Y", "J", "H", "Ks"),
        )
        self.assertTrue(config.preserve_photometry_splits)
        self.assertIsNone(magnitude_config.max_rows)

    def test_population_report_uses_original_split_denominators(self) -> None:
        product = {
            "split_labels": ["train"] * 5 + ["val"] * 3 + ["test"] * 2,
        }
        morphology_available = np.asarray(
            [True, False, True, False, True, True, False, False, True, False]
        )
        extra_band_valid = np.asarray(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [0, 1, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        report = build_morphology_population_report(
            product,
            morphology_available=morphology_available,
            extra_band_valid=extra_band_valid,
            extra_bands=("u", "u_star", "Y", "J", "Ks"),
        )

        self.assertEqual(report["split_counts"]["train"]["n_gal"], 5)
        self.assertEqual(report["split_counts"]["train"]["percent"], 50.0)
        self.assertEqual(report["morphology_counts"]["train"]["n_gal"], 3)
        self.assertEqual(report["morphology_counts"]["train"]["percent"], 60.0)
        self.assertEqual(report["morphology_counts"]["val"]["n_gal"], 1)
        self.assertAlmostEqual(
            report["morphology_counts"]["val"]["percent"],
            100.0 / 3.0,
        )
        self.assertEqual(report["valid_band_counts"]["train"]["u"]["n_gal"], 3)
        self.assertEqual(report["valid_band_counts"]["val"]["u_star"]["n_gal"], 2)

        formatted = format_morphology_population_report(report)
        self.assertIn("train: n_gal=5 (50.00%)", formatted)
        self.assertIn("train usable matched morphology: n_gal=3 (60.00%)", formatted)
        self.assertIn("validation valid u*: n_gal=2 (66.67%)", formatted)
        self.assertNotIn("valid H:", formatted)

    def test_seeded_random_sampling_is_deterministic(self) -> None:
        first = select_catalogue_row_indices(
            100,
            max_rows=12,
            sample_mode="random",
            row_start=10,
            row_stop=80,
            seed=13,
        )
        second = select_catalogue_row_indices(
            100,
            max_rows=12,
            sample_mode="random",
            row_start=10,
            row_stop=80,
            seed=13,
        )
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all((first >= 10) & (first < 80)))

    def test_redshift_filter_supports_open_upper_bound(self) -> None:
        mask = clauds_redshift_filter_mask(
            [0.0, 1.0, 2.5, np.nan],
            z_min=0.0,
            z_max=2.5,
            include_min=True,
            include_max=False,
        )
        np.testing.assert_array_equal(mask, [True, True, False, False])

    def test_fsq_decoder_and_residual_model_shapes(self) -> None:
        tokens = torch.arange(24 * 24, dtype=torch.long).remainder(4375).unsqueeze(0)
        decoded = FSQTokenDecoder()(tokens)
        self.assertEqual(tuple(decoded.shape), (1, 5, 24, 24))
        self.assertTrue(torch.isfinite(decoded).all())

        model = MorphologyResidualPhotoZModel(
            extra_feature_dim=11,
            n_z_bins=100,
            quantizer_levels=(7, 5, 5, 5, 5),
            image_hidden_dim=32,
            image_embedding_dim=16,
            photometry_hidden_dim=24,
            head_hidden_dim=32,
        )
        logits = model(
            torch.randn(2, 11),
            torch.randint(0, 4375, (2, 24 * 24)),
        )
        self.assertEqual(tuple(logits.shape), (2, 100))

    def test_aion_magnitude_mode_uses_embeddings_and_token_mlp(self) -> None:
        config = AIONMorphologyConfig(
            use_aion_magnitude_embedding=True,
            model_kinds=("aion", "aion_morphology"),
        )
        magnitude_config = make_magnitude_config(config)
        self.assertTrue(magnitude_config.use_aion_embedding)
        self.assertFalse(magnitude_config.use_mlp_features)
        self.assertEqual(tuple(magnitude_config.extra_bands), ())

        model = AIONMagnitudeMorphologyResidualPhotoZModel(
            aion_dim=12,
            n_z_bins=100,
            quantizer_levels=(7, 5, 5, 5, 5),
            image_hidden_dim=32,
            image_embedding_dim=16,
            head_hidden_dim=24,
        )
        logits = model(
            torch.randn(2, 12),
            torch.randint(0, 4375, (2, 24 * 24)),
        )
        self.assertEqual(tuple(logits.shape), (2, 100))


if __name__ == "__main__":
    unittest.main()
