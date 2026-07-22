"""Generic manifest runner for scheduler-managed experiment cases."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import tempfile
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .execution import (
    EXECUTION_STRATEGIES,
    CaseAssignment,
    CaseExecutionContext,
    ExecutionStrategy,
    WorkerRuntime,
    build_execution_plan,
    detect_worker_runtime,
)


CASE_ARTIFACT_SCHEMA_VERSION = 1
CaseTask = Callable[["ExperimentCase", CaseExecutionContext], Any]


@dataclass(frozen=True)
class ExperimentCase:
    """One named experiment configuration loaded from a JSON manifest."""

    name: str
    inputs: dict[str, Any]
    metadata: dict[str, Any]
    expected_output: Any = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int) -> "ExperimentCase":
        name = str(value.get("name", "")).strip()
        if not name:
            raise ValueError(f"Case at index {index} is missing a non-empty name.")
        inputs = value.get("inputs", {})
        metadata = value.get("metadata", {})
        if not isinstance(inputs, Mapping):
            raise TypeError(f"Case {name!r} inputs must be a mapping.")
        if not isinstance(metadata, Mapping):
            raise TypeError(f"Case {name!r} metadata must be a mapping.")
        return cls(
            name=name,
            inputs=dict(inputs),
            metadata=dict(metadata),
            expected_output=value.get("expected_output"),
        )


@dataclass(frozen=True)
class ExperimentManifest:
    """Version-controlled collection of independently executable cases."""

    name: str
    cases: tuple[ExperimentCase, ...]
    metadata: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentManifest":
        name = str(value.get("name", "")).strip()
        if not name:
            raise ValueError("Experiment manifest is missing a non-empty name.")
        raw_cases = value.get("cases")
        if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
            raise TypeError("Experiment manifest cases must be a sequence.")
        cases = tuple(ExperimentCase.from_mapping(item, index) for index, item in enumerate(raw_cases))
        if not cases:
            raise ValueError("Experiment manifest must contain at least one case.")
        case_names = [case.name for case in cases]
        if len(set(case_names)) != len(case_names):
            raise ValueError("Experiment manifest case names must be unique.")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("Experiment manifest metadata must be a mapping.")
        return cls(name=name, cases=cases, metadata=dict(metadata))


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"Experiment manifest must contain a JSON object: {path}")
    return ExperimentManifest.from_mapping(value)


def load_case_task(spec: str) -> CaseTask:
    """Load a ``module:function`` task callable."""
    if ":" not in spec:
        raise ValueError("Task must use the form 'python.module:function'.")
    module_name, attribute_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_name)
    task = getattr(module, attribute_name)
    if not callable(task):
        raise TypeError(f"Loaded task {spec!r} is not callable.")
    return task


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return slug or "case"


def case_artifact_path(output_dir: str | Path, assignment: CaseAssignment) -> Path:
    stem = f"{assignment.case_index:03d}_{_slug(assignment.case_name)}"
    if assignment.case_world_size > 1:
        stem += f".shard{assignment.case_rank:03d}-of-{assignment.case_world_size:03d}"
    return Path(output_dir) / f"{stem}.json"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    raise TypeError(
        f"Case task output contains unsupported JSON value {type(value).__name__}; "
        "return paths, scalars, mappings, arrays, tensors, dataclasses, or Pydantic models."
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _task_name(task: CaseTask) -> str:
    module = getattr(task, "__module__", type(task).__module__)
    name = getattr(task, "__qualname__", type(task).__qualname__)
    return f"{module}:{name}"


def _is_resumable_artifact(
    path: Path,
    *,
    experiment_name: str,
    experiment_metadata: Mapping[str, Any],
    case: ExperimentCase,
    assignment: CaseAssignment,
    task_name: str,
) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    execution = value.get("execution", {})
    saved_assignment = execution.get("assignment", {})
    return (
        value.get("schema_version") == CASE_ARTIFACT_SCHEMA_VERSION
        and value.get("status") == "success"
        and value.get("experiment_name") == experiment_name
        and value.get("experiment_metadata") == _jsonable(experiment_metadata)
        and value.get("case_name") == case.name
        and value.get("inputs") == _jsonable(case.inputs)
        and value.get("metadata") == _jsonable(case.metadata)
        and value.get("expected_output") == _jsonable(case.expected_output)
        and value.get("task") == task_name
        and saved_assignment.get("case_rank") == assignment.case_rank
        and saved_assignment.get("case_world_size") == assignment.case_world_size
    )


def _run_task(task: CaseTask, case: ExperimentCase, context: CaseExecutionContext) -> Any:
    output = task(case, context)
    if inspect.isawaitable(output):
        return asyncio.run(output)
    return output


def _bind_torch_device(device: str) -> None:
    if device.startswith("cuda:") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))


def run_worker_cases(
    manifest: ExperimentManifest,
    task: CaseTask,
    *,
    output_dir: str | Path,
    runtime: WorkerRuntime | None = None,
    strategy: ExecutionStrategy = "auto",
    supports_case_sharding: bool = False,
    gpus_per_case: int | None = None,
    resume: bool = False,
    cuda_device_count: int | None = None,
) -> list[dict[str, Any]]:
    """Run cases assigned to the current worker and persist one artifact per shard."""
    runtime = detect_worker_runtime() if runtime is None else runtime
    plan = build_execution_plan(
        [case.name for case in manifest.cases],
        worker_count=runtime.world_size,
        strategy=strategy,
        supports_case_sharding=supports_case_sharding,
        gpus_per_case=gpus_per_case,
    )
    assignments = plan.for_worker(runtime.rank)
    output_dir = Path(output_dir)
    device = runtime.device(cuda_device_count=cuda_device_count)
    _bind_torch_device(device)
    task_name = _task_name(task)
    records: list[dict[str, Any]] = []
    for assignment in assignments:
        case = manifest.cases[assignment.case_index]
        artifact_path = case_artifact_path(output_dir, assignment)
        if resume and _is_resumable_artifact(
            artifact_path,
            experiment_name=manifest.name,
            experiment_metadata=manifest.metadata,
            case=case,
            assignment=assignment,
            task_name=task_name,
        ):
            records.append({"case_name": case.name, "status": "skipped", "artifact_path": str(artifact_path)})
            continue
        context = CaseExecutionContext(
            experiment_name=manifest.name,
            assignment=assignment,
            runtime=runtime,
            device=device,
            output_dir=str(output_dir),
        )
        started_at = datetime.now(timezone.utc)
        start = time.perf_counter()
        try:
            output = _run_task(task, case, context)
            record = {
                "schema_version": CASE_ARTIFACT_SCHEMA_VERSION,
                "status": "success",
                "experiment_name": manifest.name,
                "experiment_metadata": manifest.metadata,
                "case_name": case.name,
                "task": task_name,
                "inputs": case.inputs,
                "metadata": case.metadata,
                "expected_output": case.expected_output,
                "execution": context.as_dict(),
                "started_at": started_at.isoformat(),
                "duration_seconds": time.perf_counter() - start,
                "output": output,
            }
        except Exception as exc:
            record = {
                "schema_version": CASE_ARTIFACT_SCHEMA_VERSION,
                "status": "failure",
                "experiment_name": manifest.name,
                "experiment_metadata": manifest.metadata,
                "case_name": case.name,
                "task": task_name,
                "inputs": case.inputs,
                "metadata": case.metadata,
                "expected_output": case.expected_output,
                "execution": context.as_dict(),
                "started_at": started_at.isoformat(),
                "duration_seconds": time.perf_counter() - start,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        _write_json_atomic(artifact_path, record)
        record = _jsonable(record)
        record["artifact_path"] = str(artifact_path)
        records.append(record)
    return records


def collect_case_artifacts(
    manifest: ExperimentManifest,
    *,
    output_dir: str | Path,
    worker_count: int = 1,
    strategy: ExecutionStrategy = "auto",
    supports_case_sharding: bool = False,
    gpus_per_case: int | None = None,
) -> dict[str, Any]:
    """Collect and validate all artifacts expected by an execution plan."""
    plan = build_execution_plan(
        [case.name for case in manifest.cases],
        worker_count=worker_count,
        strategy=strategy,
        supports_case_sharding=supports_case_sharding,
        gpus_per_case=gpus_per_case,
    )
    artifacts: list[dict[str, Any]] = []
    missing: list[str] = []
    failures: list[str] = []
    for assignment in plan.assignments:
        path = case_artifact_path(output_dir, assignment)
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        value["artifact_path"] = str(path)
        artifacts.append(value)
        if value.get("status") != "success":
            error = value.get("error", {})
            failures.append(f"{path}: {error.get('type', 'failure')}: {error.get('message', '')}")
    return {
        "schema_version": CASE_ARTIFACT_SCHEMA_VERSION,
        "experiment_name": manifest.name,
        "complete": not missing and not failures,
        "execution_plan": plan.as_dict(),
        "artifacts": artifacts,
        "missing": missing,
        "failures": failures,
    }


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--strategy", choices=EXECUTION_STRATEGIES, default="auto")
    parser.add_argument("--supports-case-sharding", action="store_true")
    parser.add_argument("--gpus-per-case", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Print the resolved execution plan as JSON.")
    _add_plan_arguments(plan_parser)
    plan_parser.add_argument("--worker-count", type=int)

    worker_parser = subparsers.add_parser("worker", help="Run cases assigned to this scheduler process.")
    _add_plan_arguments(worker_parser)
    worker_parser.add_argument("--task", required=True, help="Task callable as python.module:function.")
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    worker_parser.add_argument("--resume", action="store_true")

    collect_parser = subparsers.add_parser("collect", help="Collect worker artifacts into one summary.")
    _add_plan_arguments(collect_parser)
    collect_parser.add_argument("--output-dir", type=Path, required=True)
    collect_parser.add_argument("--worker-count", type=int)
    collect_parser.add_argument("--summary-path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_experiment_manifest(args.manifest)
    runtime = detect_worker_runtime()
    worker_count = args.worker_count if hasattr(args, "worker_count") and args.worker_count else runtime.world_size
    if args.command == "plan":
        plan = build_execution_plan(
            [case.name for case in manifest.cases],
            worker_count=worker_count,
            strategy=args.strategy,
            supports_case_sharding=args.supports_case_sharding,
            gpus_per_case=args.gpus_per_case,
        )
        print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "worker":
        records = run_worker_cases(
            manifest,
            load_case_task(args.task),
            output_dir=args.output_dir,
            runtime=runtime,
            strategy=args.strategy,
            supports_case_sharding=args.supports_case_sharding,
            gpus_per_case=args.gpus_per_case,
            resume=args.resume,
        )
        print(json.dumps(_jsonable(records), indent=2, sort_keys=True))
        return 1 if any(item.get("status") == "failure" for item in records) else 0

    summary = collect_case_artifacts(
        manifest,
        output_dir=args.output_dir,
        worker_count=worker_count,
        strategy=args.strategy,
        supports_case_sharding=args.supports_case_sharding,
        gpus_per_case=args.gpus_per_case,
    )
    if args.summary_path is not None:
        _write_json_atomic(args.summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
