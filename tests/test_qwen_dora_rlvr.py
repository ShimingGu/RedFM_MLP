from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
import unittest

import torch

from aion_magnitude.qwen_posttraining import (
    create_qlora_photoz_model,
    train_qlora_photoz,
)
from aion_magnitude.qwen_rlvr import (
    RLVRConfig,
    expected_verifier_reward,
    rlvr_group_policy_loss,
    verifier_rewards,
)


ROOT = Path(__file__).resolve().parents[1]


def load_runner_module():
    notebook_dir = str(ROOT / "notebooks")
    if notebook_dir not in sys.path:
        sys.path.insert(0, notebook_dir)
    path = ROOT / "notebooks/qwen_dora_rlvr_posttraining.py"
    spec = importlib.util.spec_from_file_location(
        "qwen_dora_rlvr_posttraining_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QwenDoRAndRLVRTest(unittest.TestCase):
    def test_supervised_trainer_dora_switch_is_backward_compatible(self) -> None:
        self.assertFalse(inspect.signature(create_qlora_photoz_model).parameters["use_dora"].default)
        self.assertFalse(inspect.signature(train_qlora_photoz).parameters["use_dora"].default)

    def test_verifier_rewards_prefer_accurate_redshifts(self) -> None:
        centers = torch.tensor([0.0, 0.5, 1.0, 2.0])
        sampled = torch.tensor([[2, 3, 1]])
        rewards, errors = verifier_rewards(
            sampled,
            torch.tensor([1.0]),
            centers,
            reward_scale=0.05,
            outlier_threshold=0.15,
            outlier_penalty=0.5,
        )
        self.assertGreater(float(rewards[0, 0]), float(rewards[0, 2]))
        self.assertGreater(float(rewards[0, 2]), float(rewards[0, 1]))
        self.assertEqual(float(errors[0, 0]), 0.0)

    def test_group_policy_loss_is_finite_and_differentiable(self) -> None:
        torch.manual_seed(7)
        logits = torch.tensor(
            [[0.1, 0.8, -0.4, 0.2], [0.7, -0.2, 0.3, 0.0]],
            requires_grad=True,
        )
        reference = torch.log_softmax(logits.detach(), dim=-1)
        loss, diagnostics = rlvr_group_policy_loss(
            logits,
            reference,
            torch.tensor([0.5, 0.0]),
            torch.tensor([0.0, 0.5, 1.0, 1.5]),
            group_samples=16,
            reward_scale=0.05,
            outlier_threshold=0.15,
            outlier_penalty=0.5,
            kl_beta=0.02,
            entropy_coefficient=0.001,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(diagnostics["kl"]), 0.0, places=6)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_expected_reward_uses_full_pdf(self) -> None:
        evaluation = {
            "pz": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            "z_spec": torch.tensor([1.0, 0.0]),
            "redshift_centers": torch.tensor([0.0, 1.0]),
        }
        reward = expected_verifier_reward(
            evaluation,
            reward_scale=0.05,
            outlier_threshold=0.15,
            outlier_penalty=0.5,
        )
        self.assertAlmostEqual(reward, 1.0, places=6)

    def test_rlvr_config_validation(self) -> None:
        config = RLVRConfig().normalized()
        self.assertEqual(config.epochs, 1)
        self.assertEqual(config.group_samples, 8)
        with self.assertRaises(ValueError):
            RLVRConfig(group_samples=0).normalized()
        with self.assertRaises(ValueError):
            RLVRConfig(kl_beta=-1.0).normalized()

    def test_runner_keeps_dora_matched_and_rlvr_is_continuation(self) -> None:
        module = load_runner_module()
        parser = module.build_parser()
        dora = parser.parse_args(["--stage", "dora"])
        rlvr = parser.parse_args(["--stage", "rlvr"])
        self.assertEqual(dora.dora_epochs, dora.qlora_epochs)
        self.assertEqual(dora.dora_batch_size, dora.qlora_batch_size)
        self.assertEqual(
            dora.dora_gradient_accumulation_steps,
            dora.gradient_accumulation_steps,
        )
        self.assertEqual(dora.dora_learning_rate, dora.qlora_learning_rate)
        self.assertEqual(dora.dora_rank, dora.lora_rank)
        self.assertEqual(dora.dora_alpha, dora.lora_alpha)
        self.assertEqual(dora.dora_dropout, dora.lora_dropout)
        self.assertEqual(dora.dora_head_warmup_epochs, dora.head_warmup_epochs)
        self.assertEqual(rlvr.rlvr_epochs, 1)
        self.assertEqual(rlvr.rlvr_group_samples, 8)
        self.assertTrue(str(rlvr.rlvr_source_adapter_dir).endswith("qlora/adapter"))
        self.assertTrue(str(rlvr.rlvr_source_head_path).endswith("qlora/photoz_head.pt"))


if __name__ == "__main__":
    unittest.main()
