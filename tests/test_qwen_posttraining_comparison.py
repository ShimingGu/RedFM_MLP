from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from astropy.table import Table


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    notebook_dir = str(ROOT / "notebooks")
    if notebook_dir not in sys.path:
        sys.path.insert(0, notebook_dir)
    path = ROOT / "notebooks/qwen_posttraining_comparison.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_posttraining_comparison_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QwenPosttrainingComparisonTest(unittest.TestCase):
    def test_defaults_are_token_free_and_morphology_is_opt_in(self) -> None:
        module = load_module()
        defaults = module.build_parser().parse_args(["--stage", "prepare"])
        self.assertFalse(defaults.use_morphology)
        self.assertFalse(hasattr(defaults, "morphology_dir"))
        self.assertFalse(hasattr(defaults, "token_batch_size"))
        self.assertEqual(
            defaults.catalogue.name,
            "COSMOS-HSCpipe-Phosphoros_morphological_multiband_updated.fits",
        )
        self.assertEqual(defaults.qlora_epochs, 10)
        self.assertEqual(defaults.head_warmup_epochs, 3)
        self.assertLess(defaults.qlora_learning_rate, defaults.head_learning_rate)

    def test_catalogue_product_optionally_adds_all_six_band_morphology(self) -> None:
        module = load_module()
        n_rows = 30
        columns = {
            "ID": np.arange(1000, 1000 + n_rows, dtype=np.int64),
            "ZPHOT": np.linspace(0.1, 1.5, n_rows, dtype=np.float32),
        }
        for name in module.tm.ALL_BAND_FLUX_COLUMNS.values():
            columns[name] = np.linspace(1.0, 3.0, n_rows, dtype=np.float32)
        for offset, name in enumerate(module.tm.MORPHOLOGY_FEATURE_COLUMNS):
            columns[name] = np.linspace(
                0.01 + offset * 0.001,
                0.50 + offset * 0.001,
                n_rows,
                dtype=np.float32,
            )
        for name in module.tm.MORPHOLOGY_AVAILABILITY_COLUMNS:
            columns[name] = np.ones(n_rows, dtype=bool)
        columns[module.tm.MORPHOLOGY_AVAILABILITY_COLUMNS[2]][1] = False

        with tempfile.TemporaryDirectory() as directory:
            catalogue = Path(directory) / "multiband.fits"
            Table(columns).write(catalogue)
            parser = module.build_parser()
            magnitude_args = parser.parse_args(
                ["--stage", "prepare", "--catalogue", str(catalogue), "--max-rows", "all"]
            )
            magnitude_product, magnitude_prep, _ = module.build_product(magnitude_args)
            morphology_args = parser.parse_args(
                [
                    "--stage", "prepare",
                    "--catalogue", str(catalogue),
                    "--max-rows", "all",
                    "--use-morphology",
                ]
            )
            morphology_product, morphology_prep, _ = module.build_product(morphology_args)

        self.assertEqual(magnitude_product["extra_features"].shape, (30, 11))
        self.assertEqual(magnitude_prep["n_morphology_features"], 0)
        self.assertEqual(morphology_product["extra_features"].shape, (29, 53))
        self.assertEqual(morphology_prep["n_morphology_features"], 42)
        self.assertNotIn(1001, morphology_product["object_id"])
        for product in (magnitude_product, morphology_product):
            self.assertNotIn("image_token_ids_path", product)
            self.assertNotIn("image_token_row_indices", product)
            self.assertFalse(product["metadata"]["image_tokens_read"])
            self.assertFalse(product["metadata"]["image_cutouts_read"])


if __name__ == "__main__":
    unittest.main()
