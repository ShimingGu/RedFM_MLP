from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from astropy.table import Table


ROOT = Path(__file__).resolve().parents[1]


def load_comparison_module():
    path = ROOT / "notebooks/qwen_mlp_full_comparison.py"
    spec = importlib.util.spec_from_file_location("qwen_mlp_full_comparison_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_image_comparison_module():
    notebook_dir = str(ROOT / "notebooks")
    if notebook_dir not in sys.path:
        sys.path.insert(0, notebook_dir)
    path = ROOT / "notebooks/qwen_mlp_full_image_comparison.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_mlp_full_image_comparison_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QwenManifestTest(unittest.TestCase):
    def test_run_manifest_is_single_valid_json_document(self) -> None:
        module = load_comparison_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalogue = root / "catalogue.fits"
            catalogue.touch()
            morphology_dir = root / "morphology"
            morphology_dir.mkdir()
            path = root / "run.json"
            module.save_run_manifest(
                path,
                args=Namespace(
                    catalogue=catalogue,
                    morphology_dir=morphology_dir,
                    output_dir=root,
                    feature_scaling="none",
                ),
                morphology_paths={"morphology_product_path": root / "product.pt"},
                qwen_cache_path=root / "qwen.pt",
                redshift_bounds=(0.0, 6.0),
                artifacts={},
            )
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["redshift_bin_bounds_from_selected_catalogue"], [0.0, 6.0])
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            self.assertFalse(path.read_bytes().endswith(b"\\n"))


class QwenMultibandMorphologyTest(unittest.TestCase):
    def test_missing_revised_catalogue_uses_verified_fallback(self) -> None:
        module = load_image_comparison_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revised = root / "multiband_updated.fits"
            fallback = root / "multiband.fits"
            fallback.touch()
            module.DEFAULT_CATALOGUE = revised
            module.FALLBACK_MULTIBAND_CATALOGUE = fallback
            resolved = module.resolve_catalogue_path(revised)
        self.assertEqual(resolved, fallback.resolve())

    def test_product_and_prompt_use_catalogue_morphology_without_tokens(self) -> None:
        module = load_image_comparison_module()
        parser = module.build_parser()
        defaults = parser.parse_args([])
        self.assertEqual(
            defaults.catalogue.name,
            "COSMOS-HSCpipe-Phosphoros_morphological_multiband_updated.fits",
        )
        self.assertFalse(hasattr(defaults, "morphology_dir"))
        self.assertFalse(hasattr(defaults, "token_batch_size"))

        n_rows = 7
        columns = {
            "ID": np.arange(100, 100 + n_rows, dtype=np.int64),
            "ZPHOT": np.linspace(0.1, 0.7, n_rows, dtype=np.float32),
        }
        for name in module.tm.ALL_BAND_FLUX_COLUMNS.values():
            columns[name] = np.linspace(1.0, 2.0, n_rows, dtype=np.float32)
        for name in module.tm.MORPHOLOGY_FEATURE_COLUMNS:
            columns[name] = np.linspace(0.1, 0.7, n_rows, dtype=np.float32)
        for name in module.tm.MORPHOLOGY_AVAILABILITY_COLUMNS:
            columns[name] = np.ones(n_rows, dtype=bool)
        columns[module.tm.MORPHOLOGY_AVAILABILITY_COLUMNS[2]][1] = False

        with tempfile.TemporaryDirectory() as directory:
            catalogue = Path(directory) / "multiband.fits"
            Table(columns).write(catalogue)
            args = parser.parse_args(
                ["--catalogue", str(catalogue), "--max-rows", "6", "--prepare-only"]
            )
            product, preparation, _ = module.build_catalogue_product(args)

        self.assertEqual(preparation["n_rows"], 6)
        self.assertEqual(preparation["n_magnitude_features"], 11)
        self.assertEqual(preparation["n_morphology_features"], 42)
        self.assertEqual(product["extra_features"].shape, (6, 53))
        self.assertNotIn(101, product["object_id"])
        self.assertNotIn("image_token_ids_path", product)
        self.assertNotIn("image_token_row_indices", product)
        self.assertFalse(product["metadata"]["image_cutouts_read"])
        self.assertFalse(product["metadata"]["image_tokens_read"])

        _, serialization = module.qwen_settings(args)
        text = module.serialize_qwen3_batch(
            product["extra_features"][:1],
            product["feature_names"],
            config=serialization,
        )[0]
        for name in module.tm.MORPHOLOGY_FEATURE_COLUMNS:
            self.assertIn(f"additional measured feature {name}=", text)
        self.assertNotIn("ordered_token_rows", text)


if __name__ == "__main__":
    unittest.main()
