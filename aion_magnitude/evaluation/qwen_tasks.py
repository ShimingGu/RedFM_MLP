"""Cluster case tasks backed by the existing Qwen/photo-z comparison script."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch

from ..FM_Qwen3 import QWEN_PHYSICAL_CONTEXT_MODES
from .execution import CaseExecutionContext
from .runner import ExperimentCase


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
QWEN_COMPARISON_SCRIPT = REPOSITORY_ROOT / "scripts" / "qwen-qwen_comparison.sh"


def _scalar_tree(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _scalar_tree(value.item())
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("Only scalar tensors belong in a metric summary.")
        return _scalar_tree(value.detach().cpu().item())
    if isinstance(value, Mapping):
        return {str(key): _scalar_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scalar_tree(item) for item in value]
    raise TypeError(f"Unsupported metric summary value: {type(value).__name__}")


def _branch_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    history = list(result.get("history", []))
    train_losses = [
        float(row["train_loss"])
        for row in history
        if row.get("train_loss") is not None and math.isfinite(float(row["train_loss"]))
    ]
    val_losses = [
        float(row["val_cross_entropy"])
        for row in history
        if row.get("val_cross_entropy") is not None
        and math.isfinite(float(row["val_cross_entropy"]))
    ]
    initial_train_loss = train_losses[0] if train_losses else None
    final_train_loss = train_losses[-1] if train_losses else None
    train_loss_reduction = (
        initial_train_loss - final_train_loss
        if initial_train_loss is not None and final_train_loss is not None
        else None
    )
    return {
        "model_kind": str(result.get("model_kind", "")),
        "checkpoint_path": str(result.get("checkpoint_path", "")),
        "final_metrics": _scalar_tree(result.get("final_metrics", {})),
        "training": {
            "epochs_completed": len(history),
            "first_epoch": _scalar_tree(history[0]) if history else None,
            "last_epoch": _scalar_tree(history[-1]) if history else None,
            "best_val_loss": min(val_losses) if val_losses else None,
            "history_finite": len(train_losses) == len(history) and len(val_losses) == len(history),
            "initial_train_loss": initial_train_loss,
            "final_train_loss": final_train_loss,
            "train_loss_reduction": train_loss_reduction,
            "train_loss_reduction_fraction": (
                train_loss_reduction / abs(initial_train_loss)
                if train_loss_reduction is not None and initial_train_loss != 0
                else None
            ),
            "loss_decreased": (
                final_train_loss < initial_train_loss
                if initial_train_loss is not None and final_train_loss is not None
                else False
            ),
        },
    }


def _embedding_diagnostics(
    embedding: torch.Tensor | np.ndarray,
    *,
    max_rows: int = 512,
) -> dict[str, Any]:
    """Return bounded, deterministic diagnostics for representation collapse."""
    values = torch.as_tensor(embedding, dtype=torch.float32).detach().cpu()
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Qwen embeddings must be a non-empty 2D tensor.")
    finite = torch.isfinite(values)
    finite_fraction = float(finite.float().mean().item())
    clean = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if len(clean) > max_rows:
        indices = torch.linspace(0, len(clean) - 1, steps=max_rows).round().long()
        sample = clean.index_select(0, indices)
    else:
        sample = clean

    row_norms = torch.linalg.vector_norm(sample, dim=1)
    feature_std = sample.std(dim=0, unbiased=False)
    centered = sample - sample.mean(dim=0, keepdim=True)
    gram = centered @ centered.T / max(int(sample.shape[1]), 1)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    total = eigenvalues.sum()
    if float(total.item()) > 0:
        probabilities = eigenvalues / total
        positive = probabilities > 0
        effective_rank = float(
            torch.exp(-(probabilities[positive] * probabilities[positive].log()).sum()).item()
        )
        numerical_rank = int((eigenvalues > eigenvalues.max() * 1e-6).sum().item())
    else:
        effective_rank = 0.0
        numerical_rank = 0

    if len(sample) > 1:
        normalized = torch.nn.functional.normalize(sample, dim=1)
        cosine_distance = 1.0 - (normalized[:-1] * normalized[1:]).sum(dim=1)
        mean_cosine_distance = float(cosine_distance.mean().item())
        p05_cosine_distance = float(torch.quantile(cosine_distance, 0.05).item())
    else:
        mean_cosine_distance = 0.0
        p05_cosine_distance = 0.0
    return {
        "rows": int(values.shape[0]),
        "sampled_rows": int(sample.shape[0]),
        "dimensions": int(values.shape[1]),
        "finite_fraction": finite_fraction,
        "mean_row_norm": float(row_norms.mean().item()),
        "mean_feature_std": float(feature_std.mean().item()),
        "median_feature_std": float(feature_std.median().item()),
        "effective_rank": effective_rank,
        "numerical_rank": numerical_rank,
        "mean_consecutive_cosine_distance": mean_cosine_distance,
        "p05_consecutive_cosine_distance": p05_cosine_distance,
    }


def _terse_cache_path(physical_cache_path: Path) -> Path:
    return physical_cache_path.with_name(physical_cache_path.stem + "_terse.pt")


def _load_embedding_diagnostics(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Qwen comparison did not produce its embedding cache: {cache_path}")
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(cached, Mapping) or "embedding" not in cached:
        raise RuntimeError(f"Qwen embedding cache is malformed: {cache_path}")
    return _embedding_diagnostics(cached["embedding"])


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0 else None


def _set_optional_env(env: dict[str, str], inputs: Mapping[str, Any], key: str, env_name: str) -> None:
    value = inputs.get(key)
    if value is not None:
        env[env_name] = str(value)


def run_qwen_qwen_case(case: ExperimentCase, context: CaseExecutionContext) -> dict[str, Any]:
    """Run one physical-versus-terse Qwen comparison on one assigned GPU.

    This task deliberately declares a one-worker contract today.  Future
    row-sharded Qwen extraction can use the same context once it implements
    deterministic shard merge by object ID.
    """
    if context.case_world_size != 1:
        raise NotImplementedError(
            "run_qwen_qwen_case does not yet merge multi-GPU row shards; "
            "run it with case_parallel execution."
        )
    inputs = case.inputs
    physical_context_mode = str(inputs.get("physical_context_mode", "full"))
    if physical_context_mode not in QWEN_PHYSICAL_CONTEXT_MODES:
        raise ValueError(
            f"physical_context_mode must be one of {QWEN_PHYSICAL_CONTEXT_MODES}."
        )
    summary_marker = bool(inputs.get("summary_marker", False))
    case_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", case.name).strip("._") or "case"
    case_output_dir = Path(context.output_dir) / f"{context.assignment.case_index:03d}_{case_slug}"
    case_output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(
        str(inputs.get("cache_root") or os.environ.get("AION_CACHE_ROOT", "/scratch/.tmp-gsm/aion_output/cache"))
    ).expanduser()
    qwen_cache_path = cache_root / "qwen_eval_cases" / case_slug / "qwen_embeddings.pt"
    qwen_cache_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHON_BIN"] = sys.executable
    env["AION_OUTPUT_DIR"] = str(case_output_dir)
    env["AION_CACHE_ROOT"] = str(cache_root)
    if context.device.startswith("cuda:"):
        logical_index = int(context.device.split(":", 1)[1])
        visible = [item.strip() for item in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
        env["CUDA_VISIBLE_DEVICES"] = (
            visible[logical_index] if logical_index < len(visible) else str(logical_index)
        )
        # The child process now sees exactly one card, always addressed as cuda:0.
        env["AION_DEVICE"] = "cuda"
    else:
        env["AION_DEVICE"] = context.device
    _set_optional_env(env, inputs, "catalogue", "AION_CATALOGUE")
    _set_optional_env(env, inputs, "morphology_dir", "AION_MORPHOLOGY_DIR")
    _set_optional_env(env, inputs, "max_rows", "AION_MAX_ROWS")
    _set_optional_env(env, inputs, "epochs", "AION_EPOCHS")
    _set_optional_env(env, inputs, "qwen_model", "QWEN_MODEL")
    _set_optional_env(env, inputs, "max_length", "QWEN_MAX_LENGTH")
    _set_optional_env(env, inputs, "pooling", "QWEN_POOLING")
    _set_optional_env(env, inputs, "embedding_batch_size", "QWEN_EMBEDDING_BATCH_SIZE")
    _set_optional_env(env, inputs, "train_batch_size", "AION_TRAIN_BATCH_SIZE")
    _set_optional_env(env, inputs, "eval_batch_size", "AION_EVAL_BATCH_SIZE")
    if "force_recompute_embeddings" in inputs:
        env["AION_FORCE_RECOMPUTE_EMBEDDINGS"] = (
            "1" if bool(inputs["force_recompute_embeddings"]) else "0"
        )
    if "normalize" in inputs:
        env["QWEN_NORMALIZE"] = "1" if bool(inputs["normalize"]) else "0"
    if "load_in_4bit" in inputs:
        env["QWEN_LOAD_IN_4BIT"] = "1" if bool(inputs["load_in_4bit"]) else "0"

    extra_args = inputs.get("extra_args", [])
    if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
        raise TypeError("case.inputs.extra_args must be a list of strings.")
    if any(item.startswith("--qwen-physical-context-mode") or item == "--qwen-summary-marker" for item in extra_args):
        raise ValueError(
            "Set physical_context_mode and summary_marker as case inputs, not via extra_args."
        )
    command = [
        "bash",
        str(QWEN_COMPARISON_SCRIPT),
        "--qwen-cache-path",
        str(qwen_cache_path),
        "--qwen-physical-context-mode",
        physical_context_mode,
        *(["--qwen-summary-marker"] if summary_marker else []),
        *extra_args,
    ]
    log_path = case_output_dir / "worker.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-40:])
        raise RuntimeError(
            f"Qwen comparison failed with exit code {completed.returncode}. "
            f"Log: {log_path}\n{tail}"
        )

    summary_path = case_output_dir / "qwen_mlp_full_results.pt"
    if not summary_path.exists():
        raise FileNotFoundError(f"Qwen comparison did not produce its summary: {summary_path}")
    results = torch.load(summary_path, map_location="cpu", weights_only=False)
    physical_result = results["qwen_morphology"]
    terse_result = results["morphology"]
    physical_diagnostics = _load_embedding_diagnostics(qwen_cache_path)
    terse_cache_path = _terse_cache_path(qwen_cache_path)
    terse_diagnostics = _load_embedding_diagnostics(terse_cache_path)
    return {
        "physical_context_mode": physical_context_mode,
        "summary_marker": summary_marker,
        "pooling": str(inputs.get("pooling", env.get("QWEN_POOLING", "last"))),
        "physical": _branch_summary(physical_result),
        "terse": _branch_summary(terse_result),
        "embedding_diagnostics": {
            "physical": physical_diagnostics,
            "terse": terse_diagnostics,
            "physical_to_terse": {
                "effective_rank_ratio": _safe_ratio(
                    physical_diagnostics["effective_rank"],
                    terse_diagnostics["effective_rank"],
                ),
                "feature_std_ratio": _safe_ratio(
                    physical_diagnostics["mean_feature_std"],
                    terse_diagnostics["mean_feature_std"],
                ),
                "cosine_distance_ratio": _safe_ratio(
                    physical_diagnostics["mean_consecutive_cosine_distance"],
                    terse_diagnostics["mean_consecutive_cosine_distance"],
                ),
            },
        },
        "comparison_summary_path": str(summary_path),
        "qwen_cache_path": str(qwen_cache_path),
        "terse_qwen_cache_path": str(terse_cache_path),
        "log_path": str(log_path),
    }
