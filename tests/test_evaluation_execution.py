import json
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import tempfile
import unittest

from aion_magnitude.evaluation import (
    ExperimentCase,
    ExperimentManifest,
    WorkerRuntime,
    build_execution_plan,
    collect_case_artifacts,
    detect_worker_runtime,
    run_worker_cases,
    shard_bounds,
)


def _runtime(rank: int, world_size: int) -> WorkerRuntime:
    return WorkerRuntime(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        local_world_size=world_size,
        visible_cuda_devices=(str(rank),),
        source="test",
    )


class TestExecutionPlanning(unittest.TestCase):
    def test_detects_slurm_and_process_local_gpu(self):
        runtime = detect_worker_runtime(
            {
                "SLURM_PROCID": "2",
                "SLURM_NTASKS": "4",
                "SLURM_LOCALID": "2",
                "SLURM_NTASKS_PER_NODE": "4(x2)",
                "CUDA_VISIBLE_DEVICES": "7",
            }
        )
        self.assertEqual(runtime.source, "slurm")
        self.assertEqual((runtime.rank, runtime.world_size), (2, 4))
        self.assertEqual(runtime.visible_cuda_devices, ("7",))
        self.assertEqual(runtime.device(cuda_device_count=1), "cuda:0")

    def test_auto_prefers_case_parallelism(self):
        plan = build_execution_plan(
            ["terse", "physical", "compact", "marker", "mean_last"],
            worker_count=4,
        )
        self.assertEqual(plan.strategy, "case_parallel")
        self.assertEqual(
            [item.case_name for item in plan.for_worker(0)],
            ["terse", "mean_last"],
        )
        self.assertTrue(all(item.case_world_size == 1 for item in plan.assignments))

    def test_auto_does_not_shard_unsupported_case(self):
        plan = build_execution_plan(["physical"], worker_count=4)
        self.assertEqual(plan.strategy, "case_parallel")
        self.assertEqual(plan.idle_worker_ranks, (1, 2, 3))

    def test_auto_can_use_all_workers_for_shardable_case(self):
        plan = build_execution_plan(
            ["physical"],
            worker_count=4,
            supports_case_sharding=True,
        )
        self.assertEqual(plan.strategy, "case_sharded")
        self.assertEqual(
            [(item.worker_rank, item.case_rank, item.case_world_size) for item in plan.assignments],
            [(0, 0, 4), (1, 1, 4), (2, 2, 4), (3, 3, 4)],
        )

    def test_fixed_two_gpu_groups_run_cases_round_robin(self):
        plan = build_execution_plan(
            ["a", "b", "c"],
            worker_count=4,
            strategy="case_sharded",
            supports_case_sharding=True,
            gpus_per_case=2,
        )
        case_workers = {
            name: [item.worker_rank for item in plan.assignments if item.case_name == name]
            for name in ("a", "b", "c")
        }
        self.assertEqual(case_workers, {"a": [0, 1], "b": [2, 3], "c": [0, 1]})

    def test_auto_honors_explicit_gpus_per_case(self):
        plan = build_execution_plan(
            ["physical"],
            worker_count=4,
            strategy="auto",
            supports_case_sharding=True,
            gpus_per_case=2,
        )
        self.assertEqual(plan.strategy, "case_sharded")
        self.assertEqual([item.worker_rank for item in plan.assignments], [0, 1])
        self.assertEqual(plan.idle_worker_ranks, (2, 3))

    def test_shard_bounds_cover_rows_once(self):
        bounds = [shard_bounds(10, index, 4) for index in range(4)]
        self.assertEqual(bounds, [(0, 3), (3, 6), (6, 8), (8, 10)])
        rows = [row for start, stop in bounds for row in range(start, stop)]
        self.assertEqual(rows, list(range(10)))


class TestCaseRunner(unittest.TestCase):
    def setUp(self):
        self.manifest = ExperimentManifest(
            name="prompt_ablation",
            cases=tuple(
                ExperimentCase(name=name, inputs={"value": index}, metadata={})
                for index, name in enumerate(("terse", "physical", "compact"))
            ),
            metadata={"cohort": "fixed"},
        )

    def test_workers_write_and_collect_independent_cases(self):
        def task(case, context):
            return {
                "value": case.inputs["value"],
                "device": context.device,
                "case_world_size": context.case_world_size,
            }

        with tempfile.TemporaryDirectory() as directory:
            for rank in range(2):
                run_worker_cases(
                    self.manifest,
                    task,
                    output_dir=directory,
                    runtime=_runtime(rank, 2),
                    cuda_device_count=0,
                )
            summary = collect_case_artifacts(
                self.manifest,
                output_dir=directory,
                worker_count=2,
            )
            self.assertTrue(summary["complete"])
            self.assertEqual(len(summary["artifacts"]), 3)
            self.assertTrue(all(item["output"]["device"] == "cpu" for item in summary["artifacts"]))

    def test_sharded_task_receives_future_row_partition_contract(self):
        manifest = ExperimentManifest(
            name="one_case",
            cases=(ExperimentCase(name="physical", inputs={"rows": 10}, metadata={}),),
            metadata={},
        )

        def task(case, context):
            return {"bounds": context.shard_bounds(case.inputs["rows"])}

        with tempfile.TemporaryDirectory() as directory:
            for rank in range(4):
                run_worker_cases(
                    manifest,
                    task,
                    output_dir=directory,
                    runtime=_runtime(rank, 4),
                    supports_case_sharding=True,
                    cuda_device_count=0,
                )
            summary = collect_case_artifacts(
                manifest,
                output_dir=directory,
                worker_count=4,
                supports_case_sharding=True,
            )
            self.assertTrue(summary["complete"])
            bounds = [tuple(item["output"]["bounds"]) for item in summary["artifacts"]]
            self.assertEqual(bounds, [(0, 3), (3, 6), (6, 8), (8, 10)])

    def test_resume_reuses_only_matching_case_inputs(self):
        calls = []

        def task(case, context):
            calls.append(case.inputs["value"])
            return {"value": case.inputs["value"]}

        with tempfile.TemporaryDirectory() as directory:
            run_worker_cases(
                self.manifest,
                task,
                output_dir=directory,
                runtime=_runtime(0, 1),
                cuda_device_count=0,
            )
            skipped = run_worker_cases(
                self.manifest,
                task,
                output_dir=directory,
                runtime=_runtime(0, 1),
                resume=True,
                cuda_device_count=0,
            )
            self.assertTrue(all(item["status"] == "skipped" for item in skipped))
            changed = ExperimentManifest(
                name=self.manifest.name,
                cases=(
                    ExperimentCase(name="terse", inputs={"value": 99}, metadata={}),
                    *self.manifest.cases[1:],
                ),
                metadata=self.manifest.metadata,
            )
            records = run_worker_cases(
                changed,
                task,
                output_dir=directory,
                runtime=_runtime(0, 1),
                resume=True,
                cuda_device_count=0,
            )
            self.assertEqual(records[0]["status"], "success")
            self.assertEqual([item["status"] for item in records[1:]], ["skipped", "skipped"])
            self.assertEqual(calls, [0, 1, 2, 99])

    def test_manifest_loader_rejects_duplicate_case_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "name": "duplicates",
                        "cases": [{"name": "same"}, {"name": "same"}],
                    }
                ),
                encoding="utf-8",
            )
            from aion_magnitude.evaluation import load_experiment_manifest

            with self.assertRaisesRegex(ValueError, "unique"):
                load_experiment_manifest(path)

    @unittest.skipUnless(importlib.util.find_spec("pydantic_evals"), "optional pydantic-evals is not installed")
    def test_pydantic_adapter_evaluates_completed_artifacts(self):
        from pydantic_evals.evaluators import Evaluator, EvaluatorContext

        from aion_magnitude.evaluation import evaluate_case_artifacts

        @dataclass
        class ValueScore(Evaluator):
            def evaluate(self, ctx: EvaluatorContext) -> float:
                return float(ctx.output["value"])

        def task(case, context):
            return {"value": case.inputs["value"]}

        with tempfile.TemporaryDirectory() as directory:
            run_worker_cases(
                self.manifest,
                task,
                output_dir=directory,
                runtime=_runtime(0, 1),
                cuda_device_count=0,
            )
            report = evaluate_case_artifacts(
                self.manifest,
                output_dir=directory,
                evaluators=[ValueScore()],
            )
            self.assertEqual(len(report.cases), 3)
            self.assertEqual(
                [item.scores["ValueScore"].value for item in report.cases],
                [0.0, 1.0, 2.0],
            )


if __name__ == "__main__":
    unittest.main()
