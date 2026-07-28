from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from aion_magnitude.morphology import compute_pixel_morphology_batch

from aion_magnitude.multiband_morphology_catalogue import (
    AION_HSC_BAND_BY_TARGET,
    FLOAT_FEATURE_STEMS,
    MULTIBAND_FEATURE_STEMS,
    MULTIBAND_MORPHOLOGY_BANDS,
    STATUS_PENDING,
    STATUS_QUALITY_REJECTED,
    BandImageTile,
    MultibandMorphologyConfig,
    _group_stratified_indices,
    _store_cutout_batch,
    compute_brightness_features,
    hsc_invalid_mask_bits,
    multiband_column_names,
    parse_bands,
)


class MultibandMorphologyCatalogueTest(unittest.TestCase):
    def test_column_contract_and_aion_channel_mapping(self) -> None:
        names = multiband_column_names()
        self.assertEqual(len(names), 72)
        self.assertEqual(len(set(names)), 72)
        for band in MULTIBAND_MORPHOLOGY_BANDS:
            self.assertIn(f"p_spiral_{band}", names)
            self.assertIn(f"surface_brightness_96_{band}", names)
            self.assertIn(f"mean_per_sqarcsec_12_{band}", names)
        self.assertEqual(AION_HSC_BAND_BY_TARGET["u"], "HSC-G")
        for band in "grizy":
            self.assertEqual(AION_HSC_BAND_BY_TARGET[band], f"HSC-{band.upper()}")
        self.assertEqual(parse_bands("y,u,r"), ("u", "r", "y"))
        with self.assertRaises(ValueError):
            parse_bands("u,u")
        with self.assertRaises(ValueError):
            parse_bands("u,j")

    def test_brightness_definitions_keep_raw_mean_and_subtract_sums(self) -> None:
        raw = np.full((1, 96, 96), 10.0, dtype=np.float32)
        valid = np.ones_like(raw, dtype=bool)
        measured = compute_brightness_features(
            raw,
            valid,
            np.asarray([2.0]),
            np.asarray([0.25]),
        )
        self.assertAlmostEqual(float(measured["surface_brightness_24"][0]), 8.0 * 24**2)
        self.assertAlmostEqual(float(measured["surface_brightness_96"][0]), 8.0 * 96**2)
        self.assertAlmostEqual(float(measured["mean_per_sqarcsec_12"][0]), 40.0)
        self.assertAlmostEqual(float(measured["mean_per_sqarcsec_24"][0]), 40.0)
        self.assertTrue(bool(measured["brightness_valid"][0]))

    def test_aperture_coverage_rejects_only_affected_measurements(self) -> None:
        raw = np.ones((1, 96, 96), dtype=np.float32)
        valid = np.ones_like(raw, dtype=bool)
        valid[:, 36:60, 36:60] = False
        measured = compute_brightness_features(
            raw,
            valid,
            np.asarray([0.0]),
            np.asarray([1.0]),
            min_aperture_coverage=0.9,
        )
        self.assertTrue(np.isnan(measured["surface_brightness_24"][0]))
        self.assertTrue(np.isnan(measured["mean_per_sqarcsec_12"][0]))
        self.assertTrue(np.isfinite(measured["surface_brightness_96"][0]))
        self.assertFalse(bool(measured["brightness_valid"][0]))

    def test_hsc_invalid_mask_plane_bits(self) -> None:
        header = fits.Header()
        header["MP_BAD"] = 0
        header["MP_SAT"] = 3
        bits = hsc_invalid_mask_bits(header)
        self.assertEqual(bits, (1 << 0) | (1 << 3))

    def test_directional_mask_is_rejected_before_shape_is_published(self) -> None:
        yy, xx = np.mgrid[:96, :96]
        image = np.exp(
            -((xx - 47.5) ** 2 + (yy - 47.5) ** 2) / (2.0 * 12.0**2)
        ).astype(np.float32)
        valid = np.ones_like(image, dtype=bool)
        valid[:, 31:33] = False
        masked = image.copy()
        masked[~valid] = 0.0
        measured = compute_pixel_morphology_batch(
            np.stack([image, masked]),
            valid_masks=np.stack([np.ones_like(valid), valid]),
            variance_cutouts=np.full((2, 96, 96), 0.01),
            min_signal_to_noise=5.0,
            min_valid_fraction=0.98,
        )
        self.assertTrue(bool(measured["morphology_pixel_valid"][0]))
        self.assertFalse(bool(measured["morphology_pixel_valid"][1]))
        self.assertTrue(np.isnan(measured["asymmetry_A"][1]))
        self.assertTrue(np.isnan(measured["axis_ellipticity"][1]))

    def test_variance_controls_signal_to_noise(self) -> None:
        yy, xx = np.mgrid[:96, :96]
        image = np.exp(
            -((xx - 47.5) ** 2 + (yy - 47.5) ** 2) / (2.0 * 8.0**2)
        ).astype(np.float32)
        measured = compute_pixel_morphology_batch(
            np.stack([image, image]),
            variance_cutouts=np.stack(
                [np.full_like(image, 0.01), np.full_like(image, 100.0)]
            ),
            min_signal_to_noise=5.0,
        )
        self.assertTrue(bool(measured["morphology_pixel_valid"][0]))
        self.assertFalse(bool(measured["morphology_pixel_valid"][1]))

    @staticmethod
    def _feature_arrays(n_rows: int = 1) -> dict[str, np.ndarray]:
        arrays = {
            stem: np.full(n_rows, np.nan, dtype=np.float32)
            for stem in FLOAT_FEATURE_STEMS
        }
        arrays.update(
            {
                "possible_morphological_mismatch": np.zeros(n_rows, dtype=bool),
                "morphology_available": np.zeros(n_rows, dtype=bool),
                "cutout_coverage": np.full(n_rows, np.nan, dtype=np.float32),
                "signal_coverage": np.full(n_rows, np.nan, dtype=np.float32),
                "asymmetry_pair_coverage": np.full(n_rows, np.nan, dtype=np.float32),
                "pixel_valid": np.zeros(n_rows, dtype=bool),
                "brightness_valid": np.zeros(n_rows, dtype=bool),
                "probability_valid": np.zeros(n_rows, dtype=bool),
                "status": np.full(n_rows, STATUS_PENDING, dtype=np.uint8),
            }
        )
        return arrays

    def test_torchfits_matches_astropy_on_compressed_hsc_subsets(self) -> None:
        try:
            import torchfits  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("torchfits is unavailable")
        rng = np.random.default_rng(5)
        science = rng.normal(size=(64, 64)).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.int32)
        mask[30, 30] = 1
        variance = rng.uniform(0.1, 0.3, size=(64, 64)).astype(np.float32)
        header = fits.Header()
        header["CRPIX1"] = 32.0
        header["CRPIX2"] = 32.0
        header["CRVAL1"] = 150.0
        header["CRVAL2"] = 2.0
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CD1_1"] = -0.2 / 3600.0
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 0.2 / 3600.0
        mask_header = fits.Header()
        mask_header["MP_BAD"] = 0
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "calexp-HSC-G-test.fits"
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.CompImageHDU(science, header=header),
                    fits.CompImageHDU(mask, header=mask_header),
                    fits.CompImageHDU(variance),
                ]
            ).writeto(image_path)
            astropy_tile = BandImageTile(
                image_path, "hsc_pdr3", fits_backend="astropy"
            )
            torchfits_tile = BandImageTile(
                image_path, "hsc_pdr3", fits_backend="torchfits"
            )
            try:
                expected = astropy_tile.extract(32.0, 32.0, size=32)
                actual = torchfits_tile.extract(32.0, 32.0, size=32)
            finally:
                astropy_tile.close()
                torchfits_tile.close()
        np.testing.assert_array_equal(actual["valid"], expected["valid"])
        np.testing.assert_allclose(actual["raw"], expected["raw"], rtol=0, atol=0)
        np.testing.assert_allclose(
            actual["variance"], expected["variance"], rtol=0, atol=0, equal_nan=True
        )
        self.assertEqual(actual["background"], expected["background"])

    def test_nonfinite_probabilities_are_terminal_rejections(self) -> None:
        class Encoder:
            def encode_hsc_band(self, images, band):
                return torch.zeros((len(images), 4))

        class Head:
            def predict_proba(self, embeddings):
                return torch.full((len(embeddings), 10), torch.nan)

        yy, xx = np.mgrid[:96, :96]
        image = np.exp(
            -((xx - 47.5) ** 2 + (yy - 47.5) ** 2) / (2.0 * 8.0**2)
        ).astype(np.float32)
        cutout = {
            "raw": image,
            "valid": np.ones_like(image, dtype=bool),
            "variance": np.full_like(image, 0.01, dtype=np.float64),
            "background": 0.0,
            "background_subtracted": image,
            "coverage": 1.0,
            "pixel_area_arcsec2": 1.0,
        }
        arrays = self._feature_arrays()
        _store_cutout_batch(
            rows=[0],
            cutouts=[cutout],
            arrays=arrays,
            band="g",
            config=MultibandMorphologyConfig(),
            encoder=Encoder(),
            head=Head(),
        )
        self.assertEqual(int(arrays["status"][0]), int(STATUS_QUALITY_REJECTED))
        self.assertFalse(bool(arrays["morphology_available"][0]))
        self.assertFalse(bool(arrays["probability_valid"][0]))
        for stem in FLOAT_FEATURE_STEMS:
            self.assertTrue(np.isnan(arrays[stem][0]), stem)

    def test_inference_failure_leaves_row_pending(self) -> None:
        class Encoder:
            def encode_hsc_band(self, images, band):
                raise RuntimeError("simulated interruption")

        yy, xx = np.mgrid[:96, :96]
        image = np.exp(
            -((xx - 47.5) ** 2 + (yy - 47.5) ** 2) / (2.0 * 8.0**2)
        ).astype(np.float32)
        cutout = {
            "raw": image,
            "valid": np.ones_like(image, dtype=bool),
            "variance": np.full_like(image, 0.01, dtype=np.float64),
            "background": 0.0,
            "background_subtracted": image,
            "coverage": 1.0,
            "pixel_area_arcsec2": 1.0,
        }
        arrays = self._feature_arrays()
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            _store_cutout_batch(
                rows=[0],
                cutouts=[cutout],
                arrays=arrays,
                band="g",
                config=MultibandMorphologyConfig(),
                encoder=Encoder(),
                head=object(),
            )
        self.assertEqual(int(arrays["status"][0]), int(STATUS_PENDING))
        self.assertFalse(bool(arrays["morphology_available"][0]))

    def test_group_split_has_no_four_band_galaxy_leakage(self) -> None:
        source_labels = np.repeat(np.arange(3), 10)
        labels = np.tile(source_labels, 4)
        groups = np.tile(np.arange(len(source_labels)), 4)
        train, validation = _group_stratified_indices(
            labels, groups, validation_fraction=0.2, seed=9
        )
        self.assertFalse(set(groups[train]).intersection(groups[validation]))
        self.assertEqual(set(labels[validation]), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
