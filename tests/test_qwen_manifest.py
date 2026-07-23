from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_comparison_module():
    path = ROOT / "notebooks/qwen_mlp_full_comparison.py"
    spec = importlib.util.spec_from_file_location("qwen_mlp_full_comparison_test", path)
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


if __name__ == "__main__":
    unittest.main()
