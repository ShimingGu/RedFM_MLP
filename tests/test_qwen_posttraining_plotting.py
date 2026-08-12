from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from notebooks.plot_qwen_posttraining import (
    _normalize_train_loss,
    _validate_matching_cohort,
    main,
)


class QwenPosttrainingPlottingTest(unittest.TestCase):
    @staticmethod
    def _result(*, rlvr_history: bool = False) -> dict:
        edges = torch.linspace(0.0, 3.0, 7)
        centers = 0.5 * (edges[:-1] + edges[1:])
        z_spec = torch.tensor([0.2, 0.8, 1.2, 1.8, 2.2, 2.8])
        logits = -torch.abs(z_spec[:, None] - centers[None, :])
        pz = torch.softmax(logits, dim=-1)
        train_key = "loss" if rlvr_history else "train_loss"
        return {
            "model_kind": "test_posttraining",
            "history": [
                {"epoch": 0, train_key: 1.2, "val_cross_entropy": 1.1},
                {"epoch": 1, train_key: 0.9, "val_cross_entropy": 0.8},
            ],
            "test_evaluation": {
                "pz": pz,
                "z_spec": z_spec,
                "z_p50": centers[torch.argmax(pz, dim=-1)],
                "redshift_edges": edges,
                "redshift_centers": centers,
            },
        }

    def test_head_only_comparison_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            baseline_path = output_dir / "baseline.pt"
            candidate_path = output_dir / "candidate.pt"
            summary_path = output_dir / "run.json"
            torch.save(self._result(), baseline_path)
            torch.save(self._result(rlvr_history=True), candidate_path)
            summary_path.write_text("{}\n")

            status = main(
                [
                    "--baseline-result-path", str(baseline_path),
                    "--result-path", str(candidate_path),
                    "--output-dir", str(output_dir),
                    "--prefix", "qwen_method_comparison",
                    "--label", "candidate-Qwen+photo-z-head",
                    "--summary-path", str(summary_path),
                    "--tomographic-samples", "2",
                ]
            )
            self.assertEqual(status, 0)
            artifacts = json.loads(summary_path.read_text())["artifacts"]
            self.assertEqual(
                set(artifacts),
                {"loss", "scatter", "pit", "nz", "nztomo"},
            )
            for path in artifacts.values():
                self.assertIn("qwen_method_comparison", path)
                self.assertGreater(Path(path).stat().st_size, 0)

    def test_cohort_mismatch_is_rejected(self) -> None:
        baseline = self._result()
        candidate = self._result()
        candidate["test_evaluation"]["z_spec"][0] += 0.1
        with self.assertRaisesRegex(ValueError, "identical z_spec"):
            _validate_matching_cohort(baseline, candidate)

    def test_rlvr_loss_is_normalized_for_standard_loss_plot(self) -> None:
        result = self._result(rlvr_history=True)
        _normalize_train_loss(result)
        self.assertEqual(result["history"][0]["train_loss"], 1.2)


if __name__ == "__main__":
    unittest.main()
