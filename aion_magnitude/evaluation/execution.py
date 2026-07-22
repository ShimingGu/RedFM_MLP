"""Execution planning for local, case-parallel, and future case-sharded runs.

The planner treats one scheduler process as one GPU worker.  The recommended
cluster layout therefore gives every process exactly one visible GPU.  A task
that can split one case across several workers may opt in to case sharding and
use ``case_rank``/``case_world_size`` plus :func:`shard_bounds`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import re
from typing import Any, Literal, Mapping, Sequence


ExecutionStrategy = Literal["auto", "single", "case_parallel", "case_sharded"]
EXECUTION_STRATEGIES: tuple[str, ...] = (
    "auto",
    "single",
    "case_parallel",
    "case_sharded",
)


def _env_int(env: Mapping[str, str], names: Sequence[str], default: int) -> int:
    for name in names:
        value = env.get(name)
        if value not in (None, ""):
            match = re.fullmatch(r"(\d+)(?:\(x\d+\))?", value)
            if match is not None:
                return int(match.group(1))
            raise ValueError(f"Environment variable {name} must be an integer, got {value!r}.")
    return default


def _visible_cuda_devices(env: Mapping[str, str]) -> tuple[str, ...]:
    value = env.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return ()
    value = value.strip()
    if not value or value == "-1":
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class WorkerRuntime:
    """Scheduler/runtime identity for one process."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_world_size: int = 1
    visible_cuda_devices: tuple[str, ...] = ()
    source: str = "local"

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be at least 1.")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {self.rank}.")
        if self.local_world_size < 1:
            raise ValueError("local_world_size must be at least 1.")
        if not 0 <= self.local_rank < self.local_world_size:
            raise ValueError(
                f"local_rank must be in [0, {self.local_world_size}), got {self.local_rank}."
            )

    def device(self, cuda_device_count: int | None = None) -> str:
        """Return the process-local torch device without touching CUDA eagerly."""
        if cuda_device_count is None:
            try:
                import torch

                cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            except ImportError:
                cuda_device_count = 0
        cuda_device_count = int(cuda_device_count)
        if cuda_device_count <= 0:
            return "cpu"
        # Slurm commonly exposes one physical card per process; it is always cuda:0
        # inside that isolated namespace.  When all cards remain visible, local_rank
        # selects the process-local card.
        device_index = 0 if cuda_device_count == 1 else self.local_rank % cuda_device_count
        return f"cuda:{device_index}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_worker_runtime(env: Mapping[str, str] | None = None) -> WorkerRuntime:
    """Detect Slurm, torchrun, or a local single-process runtime."""
    env = os.environ if env is None else env
    if "SLURM_PROCID" in env or "SLURM_NTASKS" in env:
        source = "slurm"
        rank = _env_int(env, ("SLURM_PROCID",), 0)
        world_size = _env_int(env, ("SLURM_STEP_NUM_TASKS", "SLURM_NTASKS"), 1)
        local_rank = _env_int(env, ("SLURM_LOCALID",), 0)
        local_world_size = _env_int(env, ("SLURM_NTASKS_PER_NODE",), world_size)
    elif "RANK" in env or "WORLD_SIZE" in env:
        source = "torchrun"
        rank = _env_int(env, ("RANK",), 0)
        world_size = _env_int(env, ("WORLD_SIZE",), 1)
        local_rank = _env_int(env, ("LOCAL_RANK",), rank)
        local_world_size = _env_int(env, ("LOCAL_WORLD_SIZE",), world_size)
    else:
        source = "local"
        rank = 0
        world_size = 1
        local_rank = 0
        local_world_size = 1
    return WorkerRuntime(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        visible_cuda_devices=_visible_cuda_devices(env),
        source=source,
    )


@dataclass(frozen=True)
class CaseAssignment:
    """One case executed by one member of its worker group."""

    case_index: int
    case_name: str
    worker_rank: int
    case_rank: int = 0
    case_world_size: int = 1

    @property
    def is_case_leader(self) -> bool:
        return self.case_rank == 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionPlan:
    """Resolved mapping from cases to scheduler worker ranks."""

    strategy: str
    worker_count: int
    assignments: tuple[CaseAssignment, ...]
    idle_worker_ranks: tuple[int, ...] = ()

    def for_worker(self, rank: int) -> tuple[CaseAssignment, ...]:
        if not 0 <= rank < self.worker_count:
            raise ValueError(f"rank must be in [0, {self.worker_count}), got {rank}.")
        return tuple(item for item in self.assignments if item.worker_rank == rank)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "worker_count": self.worker_count,
            "assignments": [item.as_dict() for item in self.assignments],
            "idle_worker_ranks": list(self.idle_worker_ranks),
        }


@dataclass(frozen=True)
class CaseExecutionContext:
    """Context passed to a case task by the generic worker runner."""

    experiment_name: str
    assignment: CaseAssignment
    runtime: WorkerRuntime
    device: str
    output_dir: str

    @property
    def case_rank(self) -> int:
        return self.assignment.case_rank

    @property
    def case_world_size(self) -> int:
        return self.assignment.case_world_size

    @property
    def is_case_leader(self) -> bool:
        return self.assignment.is_case_leader

    def shard_bounds(self, length: int) -> tuple[int, int]:
        return shard_bounds(length, self.case_rank, self.case_world_size)

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "assignment": self.assignment.as_dict(),
            "runtime": self.runtime.as_dict(),
            "device": self.device,
            "output_dir": self.output_dir,
        }


def _balanced_worker_groups(worker_count: int, group_count: int) -> list[tuple[int, ...]]:
    group_count = max(1, min(group_count, worker_count))
    base, remainder = divmod(worker_count, group_count)
    groups: list[tuple[int, ...]] = []
    start = 0
    for group_index in range(group_count):
        size = base + (1 if group_index < remainder else 0)
        groups.append(tuple(range(start, start + size)))
        start += size
    return groups


def _fixed_worker_groups(worker_count: int, gpus_per_case: int) -> tuple[list[tuple[int, ...]], tuple[int, ...]]:
    if gpus_per_case < 1:
        raise ValueError("gpus_per_case must be at least 1.")
    if gpus_per_case > worker_count:
        raise ValueError(
            f"gpus_per_case={gpus_per_case} exceeds worker_count={worker_count}."
        )
    usable = worker_count - (worker_count % gpus_per_case)
    groups = [
        tuple(range(start, start + gpus_per_case))
        for start in range(0, usable, gpus_per_case)
    ]
    return groups, tuple(range(usable, worker_count))


def build_execution_plan(
    case_names: Sequence[str],
    *,
    worker_count: int = 1,
    strategy: ExecutionStrategy = "auto",
    supports_case_sharding: bool = False,
    gpus_per_case: int | None = None,
) -> ExecutionPlan:
    """Choose the safest useful mapping for the available workers.

    ``auto`` prioritizes independent case parallelism.  It only assigns several
    workers to one case when the task explicitly declares
    ``supports_case_sharding=True`` and there are more workers than cases.
    """
    case_names = tuple(str(name) for name in case_names)
    if not case_names:
        raise ValueError("At least one case is required.")
    if len(set(case_names)) != len(case_names):
        raise ValueError("Case names must be unique.")
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1.")
    if strategy not in EXECUTION_STRATEGIES:
        raise ValueError(f"strategy must be one of {EXECUTION_STRATEGIES}, got {strategy!r}.")
    if gpus_per_case is not None and gpus_per_case > 1 and not supports_case_sharding:
        raise ValueError("gpus_per_case > 1 requires a task that supports case sharding.")
    if gpus_per_case is not None and gpus_per_case > 1 and strategy in {"single", "case_parallel"}:
        raise ValueError("gpus_per_case > 1 requires strategy='auto' or 'case_sharded'.")

    resolved = strategy
    if resolved == "auto":
        if gpus_per_case is not None and gpus_per_case > 1:
            resolved = "case_sharded"
        elif worker_count == 1:
            resolved = "single"
        elif supports_case_sharding and len(case_names) < worker_count:
            resolved = "case_sharded"
        else:
            resolved = "case_parallel"
    if resolved == "case_sharded" and not supports_case_sharding:
        raise ValueError("case_sharded execution requires supports_case_sharding=True.")

    assignments: list[CaseAssignment] = []
    idle_workers: tuple[int, ...] = ()
    if resolved == "single":
        assignments.extend(
            CaseAssignment(case_index=index, case_name=name, worker_rank=0)
            for index, name in enumerate(case_names)
        )
        idle_workers = tuple(range(1, worker_count))
    elif resolved == "case_parallel":
        assignments.extend(
            CaseAssignment(
                case_index=index,
                case_name=name,
                worker_rank=index % worker_count,
            )
            for index, name in enumerate(case_names)
        )
        used = {item.worker_rank for item in assignments}
        idle_workers = tuple(rank for rank in range(worker_count) if rank not in used)
    else:
        if gpus_per_case is None:
            groups = _balanced_worker_groups(worker_count, min(len(case_names), worker_count))
        else:
            groups, idle_workers = _fixed_worker_groups(worker_count, gpus_per_case)
        for case_index, name in enumerate(case_names):
            group = groups[case_index % len(groups)]
            for case_rank, worker_rank in enumerate(group):
                assignments.append(
                    CaseAssignment(
                        case_index=case_index,
                        case_name=name,
                        worker_rank=worker_rank,
                        case_rank=case_rank,
                        case_world_size=len(group),
                    )
                )

    used_workers = {item.worker_rank for item in assignments}
    idle_workers = tuple(rank for rank in range(worker_count) if rank not in used_workers)
    return ExecutionPlan(
        strategy=resolved,
        worker_count=worker_count,
        assignments=tuple(assignments),
        idle_worker_ranks=idle_workers,
    )


def shard_bounds(length: int, shard_index: int, shard_count: int) -> tuple[int, int]:
    """Return balanced contiguous ``[start, stop)`` bounds for one data shard."""
    if length < 0:
        raise ValueError("length must be non-negative.")
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1.")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must be in [0, {shard_count}), got {shard_index}.")
    base, remainder = divmod(length, shard_count)
    start = shard_index * base + min(shard_index, remainder)
    stop = start + base + (1 if shard_index < remainder else 0)
    return start, stop
