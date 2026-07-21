import importlib.util
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import torch
import numpy as np

from aion_magnitude.models import AIONOnlyPhotoZModel, build_baseline_model

path = Path(__file__).resolve().parents[1] / "notebooks" / "iotfm_mlp.py"
spec = importlib.util.spec_from_file_location("iotfm_mlp", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestColumns(unittest.TestCase):
    def test_iotfm_model_kind_uses_generic_embedding_head(self):
        model = build_baseline_model("iotfm", aion_dim=16, extra_feature_dim=0, n_z_bins=20)
        self.assertIsInstance(model, AIONOnlyPhotoZModel)

    def test_magnitude_only_inputs_use_requested_bands_and_ab_conversion(self):
        table = {
            module.ALL_BAND_FLUX_COLUMNS[band]: np.array([1.0, 10.0, 0.0])
            for band in module.MAGNITUDE_ONLY_BANDS
        }
        inputs = module.build_magnitude_input_table(table)

        self.assertEqual(
            list(inputs),
            ["u_mag", "u_star_mag", "g_mag", "r_mag", "i_mag",
             "z_mag", "y_mag", "Y_mag", "J_mag", "H_mag", "Ks_mag"],
        )
        for values in inputs.values():
            np.testing.assert_allclose(values[:2], [23.0, 20.5])
            self.assertTrue(np.isnan(values[2]))
        self.assertTrue(module.parser().parse_args(["--magnitudes-only"]).magnitudes_only)

    def test_default_exclusions(self):
        names = ["ID", "RA", "DEC", "tract", "patch", "FLUX", "ZPHOT", "ZPHOT_NIR", "Likelihood-Log", "Likelihood-Log_star"]
        included, excluded = module.select_input_columns(names)
        self.assertEqual(included, ["FLUX", "Likelihood-Log_star"])
        self.assertEqual(excluded, ["ID", "RA", "DEC", "tract", "patch", "ZPHOT", "ZPHOT_NIR", "Likelihood-Log"])

    def test_independent_switches(self):
        names = ["ID", "RA", "tract", "FLUX"]
        self.assertEqual(module.select_input_columns(names, include_id=True)[0], ["ID", "FLUX"])
        self.assertEqual(module.select_input_columns(names, include_location=True)[0], ["RA", "tract", "FLUX"])


class TestComparisonArtifacts(unittest.TestCase):
    def test_standard_comparison_figures_are_written(self):
        edges = torch.linspace(0.0, 2.0, 5)
        centers = (edges[:-1] + edges[1:]) / 2
        z_true = torch.tensor([0.2, 0.8, 1.2, 1.8])
        pz = torch.full((len(z_true), len(centers)), 1.0 / len(centers))
        evaluation = {
            "pz": pz,
            "z_spec": z_true,
            "z_p16": centers[0].repeat(len(z_true)),
            "z_p50": centers,
            "z_p84": centers[-1].repeat(len(z_true)),
            "redshift_edges": edges,
            "redshift_centers": centers,
        }
        results = {
            "iotfm": {"history": [{"epoch": 0, "train_loss": 1.0, "val_cross_entropy": 1.1}],
                      "model_kind": "iotfm"},
            "mlp": {"history": [{"epoch": 0, "train_loss": 1.2, "val_cross_entropy": 1.3}],
                    "model_kind": "tabular"},
        }
        args = SimpleNamespace(model="test-transformer", eval_batch_size=4,
                               n_z_bins=4, tomographic_samples=2, device="cpu")

        with TemporaryDirectory() as tmp, patch.object(
            module, "evaluate_trained_branch", side_effect=(evaluation, evaluation)
        ):
            artifacts = module.save_comparison_artifacts(
                results, {}, {}, "test", args, edges, centers, Path(tmp)
            )
            self.assertEqual(set(artifacts), {"loss", "scatter", "pit", "nz", "nztomo"})
            for artifact in artifacts.values():
                self.assertTrue(Path(artifact).is_file(), artifact)

if __name__ == "__main__": unittest.main()
