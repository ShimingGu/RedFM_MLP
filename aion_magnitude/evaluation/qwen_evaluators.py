"""Scientific evaluators for physical-prompt Qwen/photo-z case artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from .execution import ExecutionStrategy
from .pydantic_adapter import evaluate_case_artifacts, require_pydantic_evals
from .runner import ExperimentManifest


PHOTOZ_METRIC_KEYS = (
    "bias",
    "median_bias",
    "nmad",
    "catastrophic_outlier_fraction",
    "cross_entropy",
    "mean_log_score",
    "mean_crps",
    "p16_p84_coverage",
    "pit_mean",
)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _photoz_metrics(branch: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    final_metrics = branch.get("final_metrics", {})
    if not isinstance(final_metrics, Mapping):
        return "missing", {}
    for split in ("test", "val"):
        metrics = final_metrics.get(split)
        if isinstance(metrics, Mapping):
            return split, metrics
    return "missing", {}


def build_qwen_photoz_evaluators() -> list[Any]:
    """Build optional Pydantic evaluators without importing the extra eagerly."""
    require_pydantic_evals()
    from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

    @dataclass
    class EmbeddingHealth(Evaluator):
        min_effective_rank: float = 2.0
        min_feature_std: float = 1e-8

        def evaluate(self, ctx: EvaluatorContext) -> Mapping[str, Any]:
            diagnostics = ctx.output["embedding_diagnostics"]
            physical = diagnostics["physical"]
            terse = diagnostics["terse"]
            comparison = diagnostics["physical_to_terse"]
            finite_fraction = _finite_float(physical.get("finite_fraction")) or 0.0
            effective_rank = _finite_float(physical.get("effective_rank")) or 0.0
            feature_std = _finite_float(physical.get("mean_feature_std")) or 0.0
            result: dict[str, Any] = {
                "embedding_physical_finite": EvaluationReason(
                    finite_fraction == 1.0,
                    f"finite fraction={finite_fraction:.6g}",
                ),
                "embedding_physical_not_constant": EvaluationReason(
                    effective_rank >= self.min_effective_rank
                    and feature_std > self.min_feature_std,
                    f"effective rank={effective_rank:.6g}, mean feature std={feature_std:.6g}",
                ),
                "embedding_physical_effective_rank": effective_rank,
                "embedding_physical_feature_std": feature_std,
            }
            for output_name, source, key in (
                ("embedding_terse_effective_rank", terse, "effective_rank"),
                ("embedding_terse_feature_std", terse, "mean_feature_std"),
                ("embedding_effective_rank_ratio", comparison, "effective_rank_ratio"),
                ("embedding_feature_std_ratio", comparison, "feature_std_ratio"),
                ("embedding_cosine_distance_ratio", comparison, "cosine_distance_ratio"),
            ):
                value = _finite_float(source.get(key))
                if value is not None:
                    result[output_name] = value
            return result

    @dataclass
    class TrainingHealth(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> Mapping[str, Any]:
            result: dict[str, Any] = {}
            for branch_name in ("physical", "terse"):
                training = ctx.output[branch_name]["training"]
                result[f"training_{branch_name}_history_finite"] = bool(
                    training.get("history_finite", False)
                )
                result[f"training_{branch_name}_loss_decreased"] = bool(
                    training.get("loss_decreased", False)
                )
                reduction = _finite_float(training.get("train_loss_reduction_fraction"))
                if reduction is not None:
                    result[f"training_{branch_name}_loss_reduction_fraction"] = reduction
            return result

    @dataclass
    class PhotoZComparison(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> Mapping[str, Any]:
            physical_split, physical = _photoz_metrics(ctx.output["physical"])
            terse_split, terse = _photoz_metrics(ctx.output["terse"])
            result: dict[str, Any] = {
                "photoz_metric_split": (
                    physical_split if physical_split == terse_split else f"{physical_split}/{terse_split}"
                ),
            }
            complete = physical_split != "missing" and terse_split != "missing"
            for key in PHOTOZ_METRIC_KEYS:
                physical_value = _finite_float(physical.get(key))
                terse_value = _finite_float(terse.get(key))
                if physical_value is None or terse_value is None:
                    complete = False
                    continue
                result[f"photoz_physical_{key}"] = physical_value
                result[f"photoz_terse_{key}"] = terse_value
                result[f"photoz_delta_{key}"] = physical_value - terse_value
            result["photoz_metrics_complete_and_finite"] = EvaluationReason(
                complete,
                "Both branches must contain finite test (preferred) or validation photo-z metrics.",
            )
            return result

    return [EmbeddingHealth(), TrainingHealth(), PhotoZComparison()]


def evaluate_qwen_case_artifacts(
    manifest: ExperimentManifest,
    *,
    output_dir: str | Path,
    worker_count: int = 1,
    strategy: ExecutionStrategy = "auto",
):
    """Evaluate completed Qwen GPU artifacts in one lightweight process."""
    return evaluate_case_artifacts(
        manifest,
        output_dir=output_dir,
        worker_count=worker_count,
        strategy=strategy,
        evaluators=build_qwen_photoz_evaluators(),
        max_concurrency=1,
    )


__all__ = [
    "PHOTOZ_METRIC_KEYS",
    "build_qwen_photoz_evaluators",
    "evaluate_qwen_case_artifacts",
]
