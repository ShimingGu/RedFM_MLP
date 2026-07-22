import importlib.util
import tempfile
import unittest

from aion_magnitude.evaluation.execution import WorkerRuntime
from aion_magnitude.evaluation.qwen_evaluators import evaluate_qwen_case_artifacts
from aion_magnitude.evaluation.runner import ExperimentManifest, run_worker_cases


@unittest.skipUnless(importlib.util.find_spec("pydantic_evals"), "optional pydantic-evals is absent")
class TestQwenEvaluators(unittest.TestCase):
    def test_completed_artifact_gets_three_layer_diagnostics(self):
        manifest = ExperimentManifest.from_mapping(
            {"name": "physical_context", "cases": [{"name": "compact", "inputs": {}}]}
        )
        metrics = {
            "bias": 0.01,
            "median_bias": 0.0,
            "nmad": 0.10,
            "catastrophic_outlier_fraction": 0.05,
            "cross_entropy": 3.0,
            "mean_log_score": 3.0,
            "mean_crps": 0.08,
            "p16_p84_coverage": 0.68,
            "pit_mean": 0.5,
        }

        def task(case, context):
            branch = {
                "training": {
                    "history_finite": True,
                    "loss_decreased": True,
                    "train_loss_reduction_fraction": 0.2,
                },
                "final_metrics": {"test": metrics},
            }
            embedding = {
                "finite_fraction": 1.0,
                "effective_rank": 4.0,
                "mean_feature_std": 0.1,
            }
            return {
                "physical": branch,
                "terse": branch,
                "embedding_diagnostics": {
                    "physical": embedding,
                    "terse": embedding,
                    "physical_to_terse": {
                        "effective_rank_ratio": 1.0,
                        "feature_std_ratio": 1.0,
                        "cosine_distance_ratio": 1.0,
                    },
                },
            }

        runtime = WorkerRuntime(
            rank=0,
            world_size=1,
            local_rank=0,
            local_world_size=1,
            visible_cuda_devices=(),
            source="test",
        )
        with tempfile.TemporaryDirectory() as directory:
            run_worker_cases(manifest, task, output_dir=directory, runtime=runtime)
            report = evaluate_qwen_case_artifacts(
                manifest,
                output_dir=directory,
                worker_count=1,
            )
        self.assertFalse(report.failures)
        self.assertFalse(report.cases[0].evaluator_failures)
        self.assertTrue(report.cases[0].assertions["embedding_physical_not_constant"].value)
        self.assertEqual(report.cases[0].scores["photoz_physical_nmad"].value, 0.10)
        self.assertEqual(report.cases[0].scores["photoz_delta_nmad"].value, 0.0)


if __name__ == "__main__":
    unittest.main()
