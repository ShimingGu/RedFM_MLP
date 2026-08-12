#!/usr/bin/env python3
"""Compare a saved AION post-training result with its matched head-only baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aion_magnitude.morphology import save_morphology_comparison_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-result-path", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--baseline-label", default="head-only AION")
    parser.add_argument("--label", required=True)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--tomographic-samples", type=int, default=10)
    return parser


def _evaluation(result: dict) -> dict:
    evaluation = result.get("test_evaluation")
    if evaluation is None:
        evaluation = result.get("val_evaluation")
    if evaluation is None:
        raise KeyError("Result contains neither test_evaluation nor val_evaluation.")
    return evaluation


def _validate_matching_cohort(baseline: dict, candidate: dict) -> None:
    baseline_evaluation = _evaluation(baseline)
    candidate_evaluation = _evaluation(candidate)
    for key in ("z_spec", "redshift_edges", "redshift_centers"):
        baseline_value = torch.as_tensor(baseline_evaluation[key])
        candidate_value = torch.as_tensor(candidate_evaluation[key])
        if not torch.equal(baseline_value, candidate_value):
            raise ValueError(
                f"Baseline and candidate evaluations do not share identical {key}."
            )


def _normalize_train_loss(result: dict) -> None:
    for row in result.get("history", []):
        if "train_loss" not in row and "loss" in row:
            row["train_loss"] = row["loss"]


def _update_summary(path: Path, artifacts: dict[str, str]) -> None:
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary["artifacts"] = artifacts
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tomographic_samples < 1:
        raise ValueError("--tomographic-samples must be positive.")
    for role, path in (
        ("Head-only", args.baseline_result_path),
        ("Post-trained", args.result_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{role} result not found: {path}")
    baseline = torch.load(
        args.baseline_result_path,
        map_location="cpu",
        weights_only=False,
    )
    candidate = torch.load(args.result_path, map_location="cpu", weights_only=False)
    _validate_matching_cohort(baseline, candidate)
    _normalize_train_loss(baseline)
    _normalize_train_loss(candidate)
    artifacts = save_morphology_comparison_artifacts(
        {"baseline": baseline, "candidate": candidate},
        model_kinds=("baseline", "candidate"),
        output_dir=args.output_dir,
        tomographic_samples=args.tomographic_samples,
        comparison_labels=(args.baseline_label, args.label),
        comparison_prefix=args.output_dir / args.prefix,
    )
    if args.summary_path is not None:
        _update_summary(args.summary_path, artifacts)
    print(json.dumps(artifacts, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
