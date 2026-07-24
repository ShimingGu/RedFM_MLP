from __future__ import annotations

import unittest

import numpy as np
from astropy.io import fits

from aion_magnitude.multiband_morphology_catalogue import (
    AION_HSC_BAND_BY_TARGET,
    MULTIBAND_MORPHOLOGY_BANDS,
    _group_stratified_indices,
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
