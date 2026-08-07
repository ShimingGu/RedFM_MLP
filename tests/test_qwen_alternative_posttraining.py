from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch

from aion_magnitude.qwen_alternative_posttraining import (
    EmbeddingRedshiftDataset,
    ResidualEmbeddingAdapterConfig,
    ResidualEmbeddingPhotoZModel,
    _comma_names,
)


ROOT = Path(__file__).resolve().parents[1]


def load_runner_module():
    notebook_dir = str(ROOT / "notebooks")
    if notebook_dir not in sys.path:
        sys.path.insert(0, notebook_dir)
    path = ROOT / "notebooks/qwen_alternative_posttraining.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_alternative_posttraining_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QwenAlternativePosttrainingTest(unittest.TestCase):
    def test_residual_adapter_starts_as_identity_after_input_norm(self) -> None:
        model = ResidualEmbeddingPhotoZModel(
            embedding_dim=12,
            n_z_bins=7,
            bottleneck_dim=4,
            head_hidden_dim=5,
            dropout=0.0,
        )
        embedding = torch.randn(3, 12)
        torch.testing.assert_close(
            model.adapted_embedding(embedding),
            model.input_norm(embedding),
        )
        self.assertEqual(tuple(model(embedding).shape), (3, 7))

    def test_embedding_dataset_keeps_rows_aligned(self) -> None:
        dataset = EmbeddingRedshiftDataset(
            torch.arange(12, dtype=torch.float32).reshape(3, 4),
            torch.tensor([0.1, 0.2, 0.3]),
            ["a", "b", "c"],
        )
        self.assertEqual(dataset[1]["object_id"], "b")
        self.assertAlmostEqual(float(dataset[1]["z_spec"]), 0.2, places=6)
        torch.testing.assert_close(
            dataset[1]["embedding"],
            torch.tensor([4.0, 5.0, 6.0, 7.0]),
        )

    def test_residual_config_validation(self) -> None:
        config = ResidualEmbeddingAdapterConfig().normalized()
        self.assertEqual(config.bottleneck_dim, 256)
        self.assertEqual(config.epochs, 10)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.eval_batch_size, 8)
        self.assertEqual(config.head_warmup_epochs, 3)
        self.assertEqual(config.learning_rate, 2.0e-4)
        self.assertEqual(config.adapter_learning_rate, 1.0e-5)
        with self.assertRaises(ValueError):
            ResidualEmbeddingAdapterConfig(batch_size=0).normalized()
        with self.assertRaises(ValueError):
            ResidualEmbeddingAdapterConfig(dropout=1.0).normalized()

    def test_ia3_module_name_validation(self) -> None:
        self.assertEqual(_comma_names("k_proj, v_proj", setting="targets"), ["k_proj", "v_proj"])
        with self.assertRaises(ValueError):
            _comma_names(" , ", setting="targets")

    def test_runner_exposes_distinct_method_defaults(self) -> None:
        module = load_runner_module()
        parser = module.build_parser()
        ia3 = parser.parse_args(["--stage", "ia3"])
        residual = parser.parse_args(["--stage", "embedding-adapter"])
        self.assertEqual(ia3.ia3_epochs, ia3.qlora_epochs)
        self.assertEqual(ia3.ia3_batch_size, ia3.qlora_batch_size)
        self.assertEqual(
            ia3.ia3_gradient_accumulation_steps, ia3.gradient_accumulation_steps
        )
        self.assertEqual(ia3.ia3_learning_rate, ia3.qlora_learning_rate)
        self.assertEqual(ia3.ia3_head_warmup_epochs, ia3.head_warmup_epochs)
        self.assertEqual(ia3.ia3_max_grad_norm, ia3.lora_max_grad_norm)
        self.assertEqual(ia3.ia3_target_modules, "k_proj,v_proj,down_proj")
        self.assertEqual(ia3.ia3_feedforward_modules, "down_proj")
        self.assertEqual(residual.embedding_adapter_epochs, residual.qlora_epochs)
        self.assertEqual(
            residual.embedding_adapter_batch_size,
            residual.qlora_batch_size * residual.gradient_accumulation_steps,
        )
        self.assertEqual(
            residual.embedding_adapter_eval_batch_size, residual.eval_batch_size
        )
        self.assertEqual(
            residual.embedding_adapter_head_warmup_epochs,
            residual.head_warmup_epochs,
        )
        self.assertEqual(
            residual.embedding_adapter_learning_rate, residual.head_learning_rate
        )
        self.assertEqual(
            residual.embedding_adapter_adapter_learning_rate,
            residual.qlora_learning_rate,
        )


if __name__ == "__main__":
    unittest.main()
