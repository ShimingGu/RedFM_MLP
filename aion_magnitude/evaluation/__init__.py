"""Cluster-friendly experiment execution and optional Pydantic Evals adapters."""

from .execution import (
    EXECUTION_STRATEGIES,
    CaseAssignment,
    CaseExecutionContext,
    ExecutionPlan,
    WorkerRuntime,
    build_execution_plan,
    detect_worker_runtime,
    shard_bounds,
)
from .runner import (
    ExperimentCase,
    ExperimentManifest,
    collect_case_artifacts,
    load_experiment_manifest,
    run_worker_cases,
)
from .pydantic_adapter import evaluate_case_artifacts, require_pydantic_evals
from .qwen_evaluators import build_qwen_photoz_evaluators, evaluate_qwen_case_artifacts

__all__ = [
    "EXECUTION_STRATEGIES",
    "CaseAssignment",
    "CaseExecutionContext",
    "ExecutionPlan",
    "WorkerRuntime",
    "build_execution_plan",
    "detect_worker_runtime",
    "shard_bounds",
    "ExperimentCase",
    "ExperimentManifest",
    "collect_case_artifacts",
    "load_experiment_manifest",
    "run_worker_cases",
    "evaluate_case_artifacts",
    "require_pydantic_evals",
    "build_qwen_photoz_evaluators",
    "evaluate_qwen_case_artifacts",
]
