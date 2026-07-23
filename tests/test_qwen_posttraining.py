from types import SimpleNamespace
import unittest

import torch
from torch import nn

from aion_magnitude.qwen_posttraining import (
    QwenPhotoZModel,
    QwenPosttrainingConfig,
    TextRedshiftDataset,
    _cpu_byte_rng_state,
    trainable_parameter_summary,
)


class _ToyQwen(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids, attention_mask, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class QwenPosttrainingTest(unittest.TestCase):
    def test_defaults_require_last_pooling_and_language_only_targets(self) -> None:
        config = QwenPosttrainingConfig().normalized()
        self.assertEqual(config.pooling, "last")
        self.assertNotIn("all-linear", config.lora_target_modules)
        with self.assertRaises(ValueError):
            QwenPosttrainingConfig(pooling="mean").normalized()

    def test_photoz_model_uses_last_non_padding_token(self) -> None:
        model = QwenPhotoZModel(
            _ToyQwen(hidden_size=12),
            hidden_size=12,
            n_z_bins=7,
            head_hidden_dim=9,
        )
        logits = model(
            input_ids=torch.tensor([[1, 2, 0], [3, 4, 5]]),
            attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        )
        self.assertEqual(tuple(logits.shape), (2, 7))
        summary = trainable_parameter_summary(model)
        self.assertGreater(summary["trainable_parameters"], 0)

    def test_text_dataset_keeps_rows_aligned(self) -> None:
        dataset = TextRedshiftDataset(
            ["first", "second"],
            torch.tensor([0.2, 0.8]),
            ["a", "b"],
        )
        self.assertEqual(dataset[1]["text"], "second")
        self.assertEqual(dataset[1]["object_id"], "b")
        self.assertAlmostEqual(float(dataset[1]["z_spec"]), 0.8, places=6)

    def test_checkpoint_rng_state_is_a_cpu_byte_tensor(self) -> None:
        state = torch.get_rng_state()
        restored = _cpu_byte_rng_state(state, name="torch_rng_state")
        self.assertEqual(restored.device.type, "cpu")
        self.assertEqual(restored.dtype, torch.uint8)
        with self.assertRaisesRegex(TypeError, "torch.uint8"):
            _cpu_byte_rng_state(state.float(), name="torch_rng_state")


if __name__ == "__main__":
    unittest.main()
