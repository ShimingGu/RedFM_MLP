from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from aion_magnitude.aion_embedding_methods import AIONEmbeddingMethodConfig
from aion_magnitude.aion_qlora_rlvr import load_aion_qlora_source_artifact


ROOT = Path(__file__).resolve().parents[1]


class AIONQLoRARLVRTest(unittest.TestCase):
    def _write_source(self, root: Path, *, model_kind: str) -> Path:
        config = AIONEmbeddingMethodConfig(
            method="qlora",
            n_z_bins=7,
            epochs=2,
            head_warmup_epochs=1,
            rank=4,
            alpha=8.0,
            device="cuda",
        ).normalized()
        checkpoint = root / "qlora_aion_encoder_adapter.pt"
        torch.save(
            {
                "config": asdict(config),
                "trainable_state_dict": {
                    "aion.encoder.0.attn.qkv.lora_a.weight": torch.ones(1),
                    "pooler.query": torch.ones(1),
                    "photoz_head.network.0.weight": torch.ones(1),
                },
            },
            checkpoint,
        )
        result = root / "result.pt"
        torch.save(
            {
                "model_kind": model_kind,
                "metadata": {
                    "adaptation_scope": "aion_encoder",
                    "posttraining_method": "aion_encoder_qlora_cross_entropy",
                    "config": asdict(config),
                    "checkpoint": str(checkpoint),
                    "aion_input_mode": "photometry",
                },
            },
            result,
        )
        return result

    def test_source_loader_accepts_only_encoder_level_qlora(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._write_source(
                Path(directory), model_kind="aion_qlora_encoder_photoz"
            )
            config, state, result, checkpoint = load_aion_qlora_source_artifact(
                source
            )
        self.assertEqual(config.method, "qlora")
        self.assertEqual(result["metadata"]["adaptation_scope"], "aion_encoder")
        self.assertIn("pooler.query", state)
        self.assertEqual(checkpoint.name, "qlora_aion_encoder_adapter.pt")

    def test_source_loader_rejects_residual_vector_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._write_source(
                Path(directory),
                model_kind="aion_residual_embedding_adapter_photoz",
            )
            with self.assertRaisesRegex(ValueError, "cached-vector residual"):
                load_aion_qlora_source_artifact(source)

    def test_current_launcher_delegates_to_qlora_rlvr(self) -> None:
        entrypoint = (ROOT / "scripts/aion-rlvr_posttraining.sh").read_text()
        launcher = (
            ROOT / "scripts/aion-qlora_rlvr_posttraining.sh"
        ).read_text()
        self.assertIn("aion-qlora_rlvr_posttraining.sh", entrypoint)
        self.assertNotIn("embedding-adapter", entrypoint)
        self.assertIn("/qlora/result.pt", launcher)
        self.assertIn("--rlvr-source-result-path", launcher)
        self.assertIn("AION_RLVR_BATCH_SIZE:-1", launcher)
        self.assertIn("AION_RLVR_EVAL_BATCH_SIZE:-2", launcher)
        self.assertNotIn("--stage embedding-adapter", launcher)


if __name__ == "__main__":
    unittest.main()
