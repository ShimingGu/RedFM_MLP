from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from aion_magnitude.aion_embedding_methods import (
    AIONAttentivePooler,
    AIONEmbeddingMethodConfig,
    AIONTokenRedshiftDataset,
    DoRALinear,
    IA3FeedForwardLinear,
    IA3PackedQKVLinear,
    QLoRALinear,
    inject_aion_encoder_adapters,
)
from aion_magnitude.aion_embedding_rlvr import load_supervised_embedding_adapter
from aion_magnitude.aion_image_embeddings import _source_rows_for_object_ids
from aion_magnitude.caching import extract_aion_embeddings_to_memory
from aion_magnitude.models import extract_hsc_aion_image_embedding
from aion_magnitude.qwen_alternative_posttraining import (
    ResidualEmbeddingAdapterConfig,
    ResidualEmbeddingPhotoZModel,
)
from notebooks.plot_aion_posttraining import main as plot_main


ROOT = Path(__file__).resolve().parents[1]


def load_runner_module():
    path = ROOT / "notebooks/aion_posttraining.py"
    spec = importlib.util.spec_from_file_location("aion_posttraining_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeCodecManager:
    def encode(self, *modalities):
        output = {}
        for modality in modalities:
            if modality.token_key == "tok_image_hsc":
                batch = modality.flux.shape[0]
                output[modality.token_key] = torch.zeros(batch, 576, dtype=torch.long)
            else:
                batch = modality.value.shape[0]
                output[modality.token_key] = torch.zeros(batch, 1, dtype=torch.long)
        return output


class FakeAION:
    def __init__(self):
        self.num_encoder_tokens = None

    def encode(self, tokens, *, num_encoder_tokens):
        self.num_encoder_tokens = num_encoder_tokens
        batch = next(iter(tokens.values())).shape[0]
        return torch.ones(batch, num_encoder_tokens, 4)



class FakeAttention(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qkv = torch.nn.Linear(dim, 3 * dim)
        self.proj = torch.nn.Linear(dim, dim)


class FakeMLP(torch.nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, dim)
        self.fc3 = torch.nn.Linear(dim, hidden)


class FakeBlock(torch.nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.attn = FakeAttention(dim)
        self.mlp = FakeMLP(dim, hidden)


class FakeEncoderAION(torch.nn.Module):
    def __init__(self, dim: int = 12, hidden: int = 20, depth: int = 2):
        super().__init__()
        self.encoder = torch.nn.ModuleList([FakeBlock(dim, hidden) for _ in range(depth)])
        self.decoder_proj_context = torch.nn.Linear(dim, dim)



class AIONPosttrainingTest(unittest.TestCase):
    def _config(self, method: str) -> AIONEmbeddingMethodConfig:
        return AIONEmbeddingMethodConfig(
            method=method, n_z_bins=7, epochs=2, head_warmup_epochs=1,
            rank=4, alpha=8.0, dropout=0.0, quantization_block_size=64,
            device="cpu",
        ).normalized()

    def test_ia3_is_injected_into_every_encoder_attention_and_mlp(self) -> None:
        aion = FakeEncoderAION(depth=3)
        sample = torch.randn(2, 4, 12)
        qkv_before = aion.encoder[0].attn.qkv(sample)
        ff_before = aion.encoder[0].mlp.fc2(torch.randn(2, 4, 20))
        ff_input = torch.randn(2, 4, 20)
        ff_before = aion.encoder[0].mlp.fc2(ff_input)
        names = inject_aion_encoder_adapters(aion, self._config("ia3"))
        self.assertEqual(len(names), 6)
        self.assertTrue(all(isinstance(block.attn.qkv, IA3PackedQKVLinear) for block in aion.encoder))
        self.assertTrue(all(isinstance(block.mlp.fc2, IA3FeedForwardLinear) for block in aion.encoder))
        torch.testing.assert_close(aion.encoder[0].attn.qkv(sample), qkv_before)
        torch.testing.assert_close(aion.encoder[0].mlp.fc2(ff_input), ff_before)
        trainable = [name for name, parameter in aion.named_parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all("gate" in name for name in trainable))

    def test_dora_wraps_real_encoder_weights_and_is_exact_at_initialization(self) -> None:
        aion = FakeEncoderAION(depth=2)
        sample = torch.randn(2, 4, 12)
        original = aion.encoder[0].attn.proj(sample)
        names = inject_aion_encoder_adapters(aion, self._config("dora"))
        self.assertEqual(len(names), 11)
        wrapped = aion.encoder[0].attn.proj
        self.assertIsInstance(wrapped, DoRALinear)
        torch.testing.assert_close(wrapped(sample), original, rtol=1e-5, atol=1e-6)
        self.assertFalse(any("base_weight" == name for name, _ in wrapped.named_parameters()))
        self.assertTrue(wrapped.magnitude.requires_grad)

    def test_qlora_quantizes_actual_encoder_linears_and_trains_only_lora(self) -> None:
        aion = FakeEncoderAION(depth=2)
        original_weight = aion.encoder[0].attn.proj.weight.detach().clone()
        names = inject_aion_encoder_adapters(aion, self._config("qlora"))
        self.assertEqual(len(names), 11)
        wrapped = aion.encoder[0].attn.proj
        self.assertIsInstance(wrapped, QLoRALinear)
        self.assertEqual(wrapped.base.weight.quant_type, "nf4")
        self.assertTrue(wrapped.base.weight.compress_statistics)
        self.assertEqual(tuple(wrapped.base.weight.shape), tuple(original_weight.shape))
        self.assertFalse(wrapped.base.weight.requires_grad)
        self.assertTrue(wrapped.lora_a.weight.requires_grad)
        self.assertTrue(wrapped.lora_b.weight.requires_grad)

    def test_attentive_pooler_and_token_dataset_preserve_sequence_inputs(self) -> None:
        pooler = AIONAttentivePooler(12, num_heads=3)
        self.assertEqual(tuple(pooler(torch.randn(2, 7, 12)).shape), (2, 12))
        dataset = AIONTokenRedshiftDataset(
            {"tok_mag_g": torch.arange(6).view(3, 2),
             "tok_mag_r": torch.arange(3).view(3, 1)},
            torch.tensor([0.1, 0.2, 0.3]), ["a", "b", "c"],
        )
        self.assertEqual(len(dataset), 3)
        self.assertEqual(set(dataset[1]["tokens"]), {"tok_mag_g", "tok_mag_r"})

    def test_empty_aion_cohort_fails_before_torch_cat(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty CLAUDS dataset"):
            extract_aion_embeddings_to_memory(
                [], None, None, device="cpu"
            )


    def test_object_id_lookup_handles_unsorted_source(self) -> None:
        source = np.asarray([40, 10, 30, 20])
        target = np.asarray([20, 40, 10])
        np.testing.assert_array_equal(
            _source_rows_for_object_ids(source, target),
            np.asarray([3, 0, 1]),
        )
        with self.assertRaises(KeyError):
            _source_rows_for_object_ids(source, np.asarray([99]))
        with self.assertRaises(ValueError):
            _source_rows_for_object_ids(source, np.asarray([20, 20]))

    def test_joint_grizy_call_keeps_image_and_magnitude_tokens(self) -> None:
        model = FakeAION()
        batch = {
            f"{band}_mag": torch.full((2,), 22.0)
            for band in ("g", "r", "i", "z", "y")
        }
        embedding = extract_hsc_aion_image_embedding(
            batch,
            torch.zeros(2, 5, 96, 96),
            model,
            FakeCodecManager(),
            device="cpu",
        )
        self.assertEqual(model.num_encoder_tokens, 581)
        self.assertEqual(tuple(embedding.shape), (2, 4))
        with self.assertRaises(ValueError):
            extract_hsc_aion_image_embedding(
                batch,
                torch.zeros(2, 4, 96, 96),
                model,
                FakeCodecManager(),
                device="cpu",
            )

    def test_supervised_adapter_can_be_restored_as_rlvr_source(self) -> None:
        config = ResidualEmbeddingAdapterConfig(
            n_z_bins=7,
            bottleneck_dim=4,
            head_hidden_dim=5,
            epochs=2,
            head_warmup_epochs=1,
            device="cpu",
        ).normalized()
        model = ResidualEmbeddingPhotoZModel(
            embedding_dim=12,
            n_z_bins=7,
            bottleneck_dim=4,
            head_hidden_dim=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "residual_embedding_adapter.pt"
            result_path = root / "result.pt"
            torch.save(model.state_dict(), checkpoint)
            torch.save(
                {
                    "metadata": {
                        "config": config.__dict__,
                        "embedding_dim": 12,
                        "checkpoint": str(checkpoint),
                    }
                },
                result_path,
            )
            restored, restored_config, _ = load_supervised_embedding_adapter(
                result_path,
                device="cpu",
            )
        self.assertEqual(restored_config.n_z_bins, 7)
        self.assertEqual(restored.input_norm.normalized_shape, (12,))

    def test_runner_exposes_both_native_aion_arms(self) -> None:
        module = load_runner_module()
        parser = module.build_parser()
        common = [
            "--stage", "prepare",
            "--catalogue", "/tmp/catalogue.fits",
            "--output-dir", "/tmp/output",
        ]
        photometry = parser.parse_args(
            [*common, "--input-mode", "photometry"]
        )
        images = parser.parse_args(
            [*common, "--input-mode", "photometry-images"]
        )
        self.assertEqual(photometry.input_mode, "photometry")
        self.assertEqual(images.input_mode, "photometry-images")
        self.assertEqual(images.image_embedding_batch_size, 8)
        stage_action = next(
            action for action in parser._actions if action.dest == "stage"
        )
        self.assertTrue(
            {"ia3", "embedding-adapter", "dora", "qlora", "rlvr"}.issubset(
                set(stage_action.choices)
            )
        )

    def test_aion_plotter_writes_matched_comparison(self) -> None:
        edges = torch.linspace(0.0, 3.0, 7)
        centers = 0.5 * (edges[:-1] + edges[1:])
        z_spec = torch.tensor([0.2, 0.8, 1.2, 1.8, 2.2, 2.8])
        pz = torch.softmax(-torch.abs(z_spec[:, None] - centers[None, :]), dim=-1)
        result = {
            "history": [{"epoch": 0, "train_loss": 1.0, "val_cross_entropy": 0.9}],
            "test_evaluation": {
                "pz": pz,
                "z_spec": z_spec,
                "z_p50": centers[torch.argmax(pz, dim=-1)],
                "redshift_edges": edges,
                "redshift_centers": centers,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "head_only.pt"
            candidate = root / "posttrained.pt"
            summary = root / "run.json"
            torch.save(result, baseline)
            torch.save(result, candidate)
            summary.write_text("{}\n")
            status = plot_main(
                [
                    "--baseline-result-path", str(baseline),
                    "--result-path", str(candidate),
                    "--output-dir", str(root),
                    "--prefix", "aion_test_comparison",
                    "--label", "posttrained AION",
                    "--summary-path", str(summary),
                    "--tomographic-samples", "2",
                ]
            )
            self.assertEqual(status, 0)
            artifacts = json.loads(summary.read_text())["artifacts"]
            self.assertEqual(
                set(artifacts),
                {"loss", "scatter", "pit", "nz", "nztomo"},
            )
            self.assertTrue(all(Path(path).is_file() for path in artifacts.values()))

    def test_all_five_launchers_exist_and_use_no_qwen_model(self) -> None:
        for name in (
            "aion-ia3_posttraining.sh",
            "aion-residual_embedding_adapter_posttraining.sh",
            "aion-dora_posttraining.sh",
            "aion-rlvr_posttraining.sh",
            "aion-qlora_posttraining.sh",
        ):
            text = (ROOT / "scripts" / name).read_text()
            self.assertNotIn("QWEN_MODEL", text)
        shared = (
            ROOT / "scripts" / "aion-embedding_method_posttraining.sh"
        ).read_text()
        self.assertIn("AION_INPUT_MODES:-photometry photometry-images", shared)
        self.assertIn("plot_aion_posttraining.py", shared)
        self.assertIn("--stage \"$METHOD\"", shared)
        self.assertIn("--stage attentive-head-only", shared)
        self.assertIn("attentive_head_only/result.pt", shared)


if __name__ == "__main__":
    unittest.main()
