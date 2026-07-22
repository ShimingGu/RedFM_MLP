from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from aion_magnitude.evaluation.execution import (
    CaseAssignment,
    CaseExecutionContext,
    WorkerRuntime,
)
from aion_magnitude.evaluation.qwen_tasks import _embedding_diagnostics, run_qwen_qwen_case
from aion_magnitude.evaluation.runner import ExperimentCase


class TestQwenEvaluationTask(unittest.TestCase):
    def _context(self, output_dir: str, *, case_world_size: int = 1) -> CaseExecutionContext:
        return CaseExecutionContext(
            experiment_name="qwen_eval",
            assignment=CaseAssignment(
                case_index=0,
                case_name="last normalized",
                worker_rank=0,
                case_rank=0,
                case_world_size=case_world_size,
            ),
            runtime=WorkerRuntime(
                rank=0,
                world_size=1,
                local_rank=0,
                local_world_size=1,
                visible_cuda_devices=("5",),
                source="test",
            ),
            device="cuda:0",
            output_dir=output_dir,
        )

    def test_case_isolates_child_gpu_cache_and_metric_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory) / "cache"
            case = ExperimentCase(
                name="last normalized",
                inputs={
                    "physical_context_mode": "compact",
                    "summary_marker": True,
                    "pooling": "last",
                    "normalize": True,
                    "cache_root": str(cache_root),
                },
                metadata={},
            )
            context = self._context(directory)
            captured = {}

            def fake_run(command, **kwargs):
                captured["command"] = command
                captured["env"] = kwargs["env"]
                summary_path = Path(directory) / "000_last_normalized" / "qwen_mlp_full_results.pt"
                branch = {
                    "model_kind": "qwen",
                    "checkpoint_path": "checkpoint.pt",
                    "history": [
                        {"epoch": 0, "train_loss": 1.0, "val_cross_entropy": 0.9},
                        {"epoch": 1, "train_loss": 0.8, "val_cross_entropy": 0.7},
                    ],
                    "final_metrics": {"val": {"nmad": 0.1}},
                }
                torch.save({"qwen_morphology": branch, "morphology": branch}, summary_path)
                cache_arg = command.index("--qwen-cache-path") + 1
                physical_cache = Path(command[cache_arg])
                embedding = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
                torch.save({"embedding": embedding}, physical_cache)
                torch.save(
                    {"embedding": embedding * 2},
                    physical_cache.with_name(physical_cache.stem + "_terse.pt"),
                )
                return SimpleNamespace(returncode=0)

            with (
                patch("aion_magnitude.evaluation.qwen_tasks.subprocess.run", side_effect=fake_run),
                patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "5"}, clear=False),
            ):
                output = run_qwen_qwen_case(case, context)

            self.assertEqual(captured["env"]["CUDA_VISIBLE_DEVICES"], "5")
            self.assertEqual(captured["env"]["AION_DEVICE"], "cuda")
            self.assertEqual(captured["env"]["QWEN_NORMALIZE"], "1")
            self.assertIn("last_normalized/qwen_embeddings.pt", " ".join(captured["command"]))
            self.assertIn("--qwen-physical-context-mode", captured["command"])
            self.assertIn("compact", captured["command"])
            self.assertIn("--qwen-summary-marker", captured["command"])
            self.assertEqual(output["physical_context_mode"], "compact")
            self.assertEqual(output["physical"]["training"]["best_val_loss"], 0.7)
            self.assertTrue(output["physical"]["training"]["loss_decreased"])
            self.assertEqual(output["terse"]["final_metrics"]["val"]["nmad"], 0.1)
            self.assertEqual(output["embedding_diagnostics"]["physical"]["finite_fraction"], 1.0)

    def test_embedding_diagnostics_detect_constant_embeddings(self):
        collapsed = _embedding_diagnostics(torch.ones(8, 4))
        varied = _embedding_diagnostics(torch.eye(8))
        self.assertEqual(collapsed["effective_rank"], 0.0)
        self.assertEqual(collapsed["mean_feature_std"], 0.0)
        self.assertGreater(varied["effective_rank"], 1.0)

    def test_current_qwen_task_rejects_case_sharding(self):
        case = ExperimentCase(name="physical", inputs={}, metadata={})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(NotImplementedError, "does not yet merge"):
                run_qwen_qwen_case(case, self._context(directory, case_world_size=2))


if __name__ == "__main__":
    unittest.main()
