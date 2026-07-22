"""Optional Pydantic Evals reporting over completed GPU case artifacts.

GPU work is intentionally completed by scheduler workers first.  This adapter
then evaluates the small JSON outputs in one process, avoiding accidental model
loads caused by framework-level concurrency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .execution import ExecutionStrategy
from .runner import ExperimentManifest, collect_case_artifacts


def require_pydantic_evals():
    try:
        from pydantic_evals import Case, Dataset
    except ImportError as exc:
        raise ImportError(
            "Pydantic Evals reporting requires the optional 'evals' dependency. "
            "Install the package with: pip install -e '.[evals]'"
        ) from exc
    return Case, Dataset


def evaluate_case_artifacts(
    manifest: ExperimentManifest,
    *,
    output_dir: str | Path,
    worker_count: int = 1,
    strategy: ExecutionStrategy = "auto",
    supports_case_sharding: bool = False,
    gpus_per_case: int | None = None,
    evaluators: Sequence[Any] = (),
    report_evaluators: Sequence[Any] = (),
    max_concurrency: int = 1,
):
    """Build one Pydantic Evals report from completed worker artifacts.

    A sharded case is returned as ``{"shards": [...]}`` until a domain task
    supplies a merged case artifact.  This keeps the distributed contract
    usable now without pretending that arbitrary model outputs can be merged
    generically.
    """
    Case, Dataset = require_pydantic_evals()
    summary = collect_case_artifacts(
        manifest,
        output_dir=output_dir,
        worker_count=worker_count,
        strategy=strategy,
        supports_case_sharding=supports_case_sharding,
        gpus_per_case=gpus_per_case,
    )
    if not summary["complete"]:
        raise RuntimeError(
            "Cannot evaluate incomplete case artifacts: "
            f"missing={len(summary['missing'])}, failures={len(summary['failures'])}."
        )

    artifacts_by_case: dict[str, list[dict[str, Any]]] = {
        case.name: [] for case in manifest.cases
    }
    for artifact in summary["artifacts"]:
        artifacts_by_case[str(artifact["case_name"])].append(artifact)

    def load_completed_output(paths: tuple[str, ...]) -> Any:
        import json

        outputs = []
        for path in paths:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
            outputs.append(value["output"])
        return outputs[0] if len(outputs) == 1 else {"shards": outputs}

    cases = []
    for case in manifest.cases:
        artifacts = sorted(
            artifacts_by_case[case.name],
            key=lambda item: int(item["execution"]["assignment"]["case_rank"]),
        )
        cases.append(
            Case(
                name=case.name,
                inputs=tuple(str(item["artifact_path"]) for item in artifacts),
                expected_output=case.expected_output,
                metadata={
                    **manifest.metadata,
                    **case.metadata,
                    "case_inputs": case.inputs,
                },
            )
        )
    dataset = Dataset(
        name=manifest.name,
        cases=cases,
        evaluators=list(evaluators),
        report_evaluators=list(report_evaluators),
    )
    return dataset.evaluate_sync(load_completed_output, max_concurrency=max_concurrency)
