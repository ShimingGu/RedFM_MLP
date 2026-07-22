"""Create a Pydantic Evals report from completed Qwen case artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from .qwen_evaluators import evaluate_qwen_case_artifacts
from .runner import load_experiment_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--strategy", choices=("auto", "single", "case_parallel", "case_sharded"), default="auto")
    parser.add_argument("--report-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_experiment_manifest(args.manifest)
    report = evaluate_qwen_case_artifacts(
        manifest,
        output_dir=args.output_dir,
        worker_count=args.worker_count,
        strategy=args.strategy,
    )
    report.print(include_input=False, include_output=False, include_reasons=True)
    report_path = args.report_path or args.output_dir / "pydantic_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    from pydantic_evals.reporting import EvaluationReportAdapter

    report_path.write_bytes(EvaluationReportAdapter.dump_json(report, indent=2))
    print(f"Pydantic Evals report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
