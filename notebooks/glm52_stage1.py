#!/usr/bin/env python3
"""Run the GLM Stage 1 cached/live by baseline/direct frozen-head controls."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from peft import prepare_model_for_kbit_training

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = Path(__file__).resolve().parent
for import_path in (ROOT, NOTEBOOK_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import glm52_posttraining as glm_run
from aion_magnitude.Inference_Opt_TFM import (
    InferenceOptimizedEmbeddingConfig,
    load_inference_optimized_transformer,
    pool_hidden_states,
    serialize_catalogue_row,
)
from aion_magnitude.metrics import (
    predict_photoz_from_logits,
    redshift_cross_entropy_loss,
    summarize_pdf_metrics,
)
from aion_magnitude.models import PhotoZHead
from aion_magnitude.qwen_posttraining import _linear_warmup_decay_lambda
from aion_magnitude.utils import make_redshift_grid, set_random_seed


DEFAULT_RUN_ROOT = Path(
    "/arc/home/gsm/aion_output/figures/glm52-posttraining-paddingfix-0905/"
    "photometry"
)
DEFAULT_OUTPUT_DIR = Path(
    "/arc/home/gsm/aion_output/figures/"
    "glm52-failure-analysis-paddingfix-0905/stage1_head_controls"
)
ARM_LABELS = {
    "A_cached_baseline": "A: cached + baseline protocol",
    "B_cached_direct": "B: cached + direct protocol",
    "C_live_baseline": "C: live + baseline protocol",
    "D_live_direct": "D: live + direct protocol",
}
BASELINE_ARMS = ("A_cached_baseline", "C_live_baseline")
DIRECT_ARMS = ("B_cached_direct", "D_live_direct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-json", type=Path, default=DEFAULT_RUN_ROOT / "prepared.json"
    )
    parser.add_argument(
        "--frozen-result",
        type=Path,
        default=DEFAULT_RUN_ROOT / "head_only/result.pt",
        help="Corrected existing cached-baseline result used as an external reference.",
    )
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--baseline-effective-batch-size", type=int, default=256)
    parser.add_argument("--direct-effective-batch-size", type=int, default=16)
    parser.add_argument("--head-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--baseline-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--direct-weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--direct-warmup-fraction", type=float, default=0.03)
    parser.add_argument("--direct-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-updates", type=int, default=250)
    parser.add_argument(
        "--train-limit",
        type=int,
        help="Optional leading-row limit for smoke tests; omit for Stage 1.",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        help="Optional leading-row limit for smoke tests; omit for Stage 1.",
    )
    parser.add_argument(
        "--no-resume", action="store_false", dest="resume", default=True
    )
    return parser.parse_args()


def sha256_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def prepare_pipeline_args(prepared: dict[str, Any], seed: int) -> argparse.Namespace:
    command = [
        "--stage",
        "prepare",
        "--catalogue",
        str(prepared["catalogue"]),
        "--output-dir",
        str(Path(prepared["catalogue"]).parent),
        "--max-rows",
        str(prepared["max_rows"]),
        "--seed",
        str(seed),
        "--glm-model",
        str(prepared["resolved_model_path"]),
        "--glm-max-length",
        str(prepared["max_length"]),
    ]
    if prepared.get("use_morphology"):
        command.append("--use-morphology")
    return glm_run.build_parser().parse_args(command)


def serialize_rows(
    product: dict[str, Any],
    rows: np.ndarray,
    serialization,
) -> list[str]:
    values = torch.as_tensor(product["extra_features"], dtype=torch.float32)
    names = [str(name) for name in product["feature_names"]]
    return [
        serialize_catalogue_row(
            {
                name: values[int(row), column]
                for column, name in enumerate(names)
            },
            config=serialization,
        )
        for row in rows
    ]


def accumulation_steps(effective_batch_size: int, micro_batch_size: int) -> int:
    if micro_batch_size < 1 or effective_batch_size < 1:
        raise ValueError("Batch sizes must be positive.")
    if effective_batch_size % micro_batch_size:
        raise ValueError(
            f"Effective batch {effective_batch_size} must be divisible by "
            f"micro batch {micro_batch_size}."
        )
    return effective_batch_size // micro_batch_size


def batch_loss_scale(
    batch_index: int,
    *,
    n_examples: int,
    n_batches: int,
    micro_batch_size: int,
    accumulation: int,
) -> float:
    group_start_batch = (batch_index // accumulation) * accumulation
    group_end_batch = min(group_start_batch + accumulation, n_batches)
    group_start = group_start_batch * micro_batch_size
    group_end = min(group_end_batch * micro_batch_size, n_examples)
    batch_start = batch_index * micro_batch_size
    batch_end = min(batch_start + micro_batch_size, n_examples)
    return float(batch_end - batch_start) / float(group_end - group_start)


def is_update_batch(
    batch_index: int, *, n_batches: int, accumulation: int
) -> bool:
    return (batch_index + 1) % accumulation == 0 or batch_index + 1 == n_batches


@torch.no_grad()
def live_embeddings(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    *,
    max_length: int,
) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in encoded.items()}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**inputs, use_cache=False, return_dict=True)
        pooled = pool_hidden_states(
            output.last_hidden_state,
            inputs["attention_mask"],
            pooling="last",
        )
    return pooled.float()


def paired_logits(
    cached_head: PhotoZHead,
    live_head: PhotoZHead,
    cached_embedding: torch.Tensor,
    live_embedding: torch.Tensor,
    *,
    device: torch.device,
    autocast_head: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng_state = torch.cuda.get_rng_state(device)
    if autocast_head:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cached_logits = cached_head(cached_embedding)
        torch.cuda.set_rng_state(rng_state, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            live_logits = live_head(live_embedding)
    else:
        cached_logits = cached_head(cached_embedding)
        torch.cuda.set_rng_state(rng_state, device)
        live_logits = live_head(live_embedding)
    return cached_logits.float(), live_logits.float()


def evaluation_from_logits(
    logits: torch.Tensor,
    z_spec: torch.Tensor,
    redshift_edges: torch.Tensor,
    redshift_centers: torch.Tensor,
    object_ids: list[str],
) -> dict[str, Any]:
    evaluation: dict[str, Any] = {
        "logits": logits.float().cpu(),
        "z_spec": z_spec.float().cpu(),
        "object_id": object_ids,
        "redshift_edges": redshift_edges.float().cpu(),
        "redshift_centers": redshift_centers.float().cpu(),
    }
    evaluation["loss"] = float(
        redshift_cross_entropy_loss(
            evaluation["logits"],
            evaluation["z_spec"],
            edges=evaluation["redshift_edges"],
        ).item()
    )
    evaluation.update(
        predict_photoz_from_logits(
            evaluation["logits"], centers=evaluation["redshift_centers"]
        )
    )
    return evaluation


@torch.no_grad()
def evaluate_arms(
    *,
    heads: dict[str, PhotoZHead],
    cached_embeddings: torch.Tensor,
    texts: list[str],
    z_spec: torch.Tensor,
    object_ids: list[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    batch_size: int,
    redshift_edges: torch.Tensor,
    redshift_centers: torch.Tensor,
    progress_label: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]]]:
    for head in heads.values():
        head.eval()
    logits_parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in ARM_LABELS
    }
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(texts), batch_size)):
        stop = min(start + batch_size, len(texts))
        cached = cached_embeddings[start:stop].to(device)
        live = live_embeddings(
            texts[start:stop],
            tokenizer,
            model,
            device,
            max_length=max_length,
        )
        logits_parts["A_cached_baseline"].append(
            heads["A_cached_baseline"](cached).float().cpu()
        )
        logits_parts["C_live_baseline"].append(
            heads["C_live_baseline"](live).float().cpu()
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits_parts["B_cached_direct"].append(
                heads["B_cached_direct"](cached).float().cpu()
            )
            logits_parts["D_live_direct"].append(
                heads["D_live_direct"](live).float().cpu()
            )
        if batch_index == 0 or batch_index + 1 == n_batches or (
            (batch_index + 1) % 500 == 0
        ):
            print(
                f"{progress_label}: validation batch {batch_index + 1}/{n_batches}",
                flush=True,
            )

    evaluations = {
        name: evaluation_from_logits(
            torch.cat(parts),
            z_spec,
            redshift_edges,
            redshift_centers,
            object_ids,
        )
        for name, parts in logits_parts.items()
    }
    metrics = {
        name: summarize_pdf_metrics(evaluation)
        for name, evaluation in evaluations.items()
    }
    return evaluations, metrics


def state_on_cpu(head: PhotoZHead) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in head.state_dict().items()
    }


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }


def tensor_comparison(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    reference = reference.float().cpu()
    candidate = candidate.float().cpu()
    cosine = F.cosine_similarity(reference, candidate, dim=-1)
    difference = candidate - reference
    reference_rms = reference.square().mean().sqrt().clamp_min(1.0e-12)
    return {
        "mean_cosine_similarity": float(cosine.mean().item()),
        "minimum_cosine_similarity": float(cosine.min().item()),
        "normalized_rmse": float(
            (difference.square().mean().sqrt() / reference_rms).item()
        ),
        "maximum_absolute_difference": float(difference.abs().max().item()),
    }


def relative_difference(reference: float, candidate: float) -> float:
    return abs(candidate - reference) / max(abs(reference), 1.0e-12)


def plot_history(
    history: dict[str, list[dict[str, float | int]]],
    reference_metrics: dict[str, float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, rows in history.items():
        epochs = [int(row["epoch"]) + 1 for row in rows]
        axes[0].plot(
            epochs,
            [float(row["val_cross_entropy"]) for row in rows],
            marker="o",
            label=ARM_LABELS[name],
        )
        axes[1].plot(
            epochs,
            [float(row["val_nmad"]) for row in rows],
            marker="o",
            label=ARM_LABELS[name],
        )
    if reference_metrics:
        axes[0].axhline(
            float(reference_metrics["cross_entropy"]),
            color="black",
            linestyle="--",
            linewidth=1,
            label="existing corrected A",
        )
        axes[1].axhline(
            float(reference_metrics["nmad"]),
            color="black",
            linestyle="--",
            linewidth=1,
            label="existing corrected A",
        )
    axes[0].set(title="Validation cross-entropy", xlabel="Epoch", ylabel="CE")
    axes[1].set(title="Validation NMAD", xlabel="Epoch", ylabel="NMAD")
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# GLM Stage 1 matched frozen-head controls",
        "",
        "Decision metrics use only the fixed validation split; the test split was not evaluated.",
        "",
        "## Final validation metrics",
        "",
        "| Arm | Cross-entropy | NMAD | Catastrophic fraction | CRPS | Selected epoch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ARM_LABELS:
        metrics = report["final_validation_metrics"][name]
        lines.append(
            f"| {ARM_LABELS[name]} | {metrics['cross_entropy']:.6f} | "
            f"{metrics['nmad']:.6f} | "
            f"{metrics['catastrophic_outlier_fraction']:.6f} | "
            f"{metrics['mean_crps']:.6f} | "
            f"{report['selected_epoch'][name]} |"
        )
    reference = report["existing_corrected_A_reference"]
    reference_nmad_delta = 100.0 * (
        report["final_validation_metrics"]["A_cached_baseline"]["nmad"]
        / reference["nmad"]
        - 1.0
    )
    lines.extend(
        [
            "",
            "Existing corrected cached baseline reference: "
            f"CE {reference['cross_entropy']:.6f}, NMAD {reference['nmad']:.6f}.",
            "The existing-A NMAD check is diagnostic rather than gating; "
            f"the paired A arm changes it by {reference_nmad_delta:+.2f}%.",
            "",
            "## Checks",
            "",
        ]
    )
    for name, passed in report["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: {name}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            f"Validation curves: {report['artifacts']['validation_curves']}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def report_without_evaluations(report: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(report)


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.eval_batch_size < 1:
        raise ValueError("Epoch and evaluation batch sizes must be positive.")
    baseline_accumulation = accumulation_steps(
        args.baseline_effective_batch_size, args.micro_batch_size
    )
    direct_accumulation = accumulation_steps(
        args.direct_effective_batch_size, args.micro_batch_size
    )

    prepared = json.loads(args.prepared_json.read_text())
    frozen_result = torch.load(
        args.frozen_result, map_location="cpu", weights_only=False
    )
    cache_path = args.cache_path
    if cache_path is None:
        cache_path = Path(frozen_result["metadata"]["embedding_cache"])
    for required in (args.prepared_json, args.frozen_result, cache_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing Stage 1 input: {required}")

    pipeline_args = prepare_pipeline_args(prepared, args.seed)
    print("building the exact catalogue product", flush=True)
    product, preparation, bounds = glm_run.catalogue_pipeline.build_product(
        pipeline_args
    )
    labels = np.asarray(product["split_labels"], dtype=object)
    train_rows = np.flatnonzero(labels == "train")
    val_rows = np.flatnonzero(labels == "val")
    if args.train_limit is not None:
        train_rows = train_rows[: args.train_limit]
    if args.val_limit is not None:
        val_rows = val_rows[: args.val_limit]
    if len(train_rows) == 0 or len(val_rows) == 0:
        raise ValueError("Stage 1 requires non-empty training and validation splits.")

    print(f"memory-mapping corrected cache {cache_path}", flush=True)
    cache = torch.load(
        cache_path, map_location="cpu", weights_only=False, mmap=True
    )
    cache_metadata = cache.get("metadata", {})
    if not cache_metadata.get("kbit_prepared"):
        raise ValueError("Stage 1 requires a k-bit-prepared corrected cache.")
    if cache_metadata.get("last_token_pooling") != "padding_side_safe_v2":
        raise ValueError("Stage 1 requires the padding-safe-v2 cache.")
    product_ids = [str(value) for value in product["object_id"]]
    cache_ids = [str(value) for value in cache["object_id"]]
    if cache_ids != product_ids:
        raise RuntimeError("Cached and reconstructed catalogue row order differs.")

    cache_embeddings = torch.as_tensor(cache["embedding"])
    train_cached = cache_embeddings[
        torch.as_tensor(train_rows, dtype=torch.long)
    ].float().clone()
    val_cached = cache_embeddings[
        torch.as_tensor(val_rows, dtype=torch.long)
    ].float().clone()
    train_z = torch.as_tensor(product["z_spec"], dtype=torch.float32)[train_rows]
    val_z = torch.as_tensor(product["z_spec"], dtype=torch.float32)[val_rows]
    train_ids = [product_ids[int(row)] for row in train_rows]
    val_ids = [product_ids[int(row)] for row in val_rows]
    serialization = glm_run._serialization(pipeline_args)
    print("serializing Stage 1 train and validation rows", flush=True)
    train_texts = serialize_rows(product, train_rows, serialization)
    val_texts = serialize_rows(product, val_rows, serialization)

    model_config = InferenceOptimizedEmbeddingConfig(
        model_path=prepared["resolved_model_path"],
        device=args.device,
        torch_dtype="bfloat16",
        max_length=int(prepared["max_length"]),
        pooling="last",
        normalize=False,
        local_files_only=True,
        trust_remote_code=True,
        load_in_4bit=True,
        freeze_model=True,
    )
    print(f"loading and preparing frozen live model on {args.device}", flush=True)
    tokenizer, model, device = load_inference_optimized_transformer(model_config)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    set_random_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    initial_head = PhotoZHead(2048, n_z_bins=300, hidden_dim=256).to(device)
    initial_state = state_on_cpu(initial_head)
    heads = {}
    for name in ARM_LABELS:
        head = PhotoZHead(2048, n_z_bins=300, hidden_dim=256).to(device)
        head.load_state_dict(initial_state)
        heads[name] = head
    del initial_head

    optimizers = {
        name: torch.optim.AdamW(
            heads[name].parameters(),
            lr=args.head_learning_rate,
            weight_decay=(
                args.baseline_weight_decay
                if name in BASELINE_ARMS
                else args.direct_weight_decay
            ),
        )
        for name in ARM_LABELS
    }
    n_batches = (len(train_rows) + args.micro_batch_size - 1) // args.micro_batch_size
    baseline_updates_per_epoch = (
        n_batches + baseline_accumulation - 1
    ) // baseline_accumulation
    direct_updates_per_epoch = (
        n_batches + direct_accumulation - 1
    ) // direct_accumulation
    total_direct_updates = direct_updates_per_epoch * args.epochs
    direct_warmup_steps = int(
        total_direct_updates * args.direct_warmup_fraction
    )
    schedulers = {
        name: torch.optim.lr_scheduler.LambdaLR(
            optimizers[name],
            lambda step: _linear_warmup_decay_lambda(
                step,
                warmup_steps=direct_warmup_steps,
                total_steps=total_direct_updates,
            ),
        )
        for name in DIRECT_ARMS
    }
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint-latest.pt"
    run_config = {
        "prepared_json": str(args.prepared_json),
        "frozen_result": str(args.frozen_result),
        "cache_path": str(cache_path),
        "model": str(prepared["resolved_model_path"]),
        "epochs": args.epochs,
        "micro_batch_size": args.micro_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "baseline_effective_batch_size": args.baseline_effective_batch_size,
        "direct_effective_batch_size": args.direct_effective_batch_size,
        "head_learning_rate": args.head_learning_rate,
        "baseline_weight_decay": args.baseline_weight_decay,
        "direct_weight_decay": args.direct_weight_decay,
        "direct_warmup_fraction": args.direct_warmup_fraction,
        "direct_max_grad_norm": args.direct_max_grad_norm,
        "seed": args.seed,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
    }
    history: dict[str, list[dict[str, float | int]]] = {
        name: [] for name in ARM_LABELS
    }
    best_loss = {name: float("inf") for name in ARM_LABELS}
    best_state: dict[str, dict[str, torch.Tensor] | None] = {
        name: None for name in ARM_LABELS
    }
    selected_epoch = {name: -1 for name in ARM_LABELS}
    start_epoch = 0
    baseline_updates = 0
    direct_updates = 0

    if args.resume and checkpoint_path.exists():
        saved = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if saved["config"] != run_config:
            raise ValueError("Stage 1 checkpoint configuration does not match.")
        for name in ARM_LABELS:
            heads[name].load_state_dict(saved["heads"][name])
            optimizers[name].load_state_dict(saved["optimizers"][name])
            if name in DIRECT_ARMS:
                schedulers[name].load_state_dict(saved["schedulers"][name])
        history = saved["history"]
        best_loss = saved["best_loss"]
        best_state = saved["best_state"]
        selected_epoch = saved["selected_epoch"]
        start_epoch = int(saved["next_epoch"])
        baseline_updates = int(saved["baseline_updates"])
        direct_updates = int(saved["direct_updates"])
        print(
            f"resuming Stage 1 at epoch {start_epoch + 1}/{args.epochs}",
            flush=True,
        )

    edges, centers = make_redshift_grid(bounds[0], bounds[1], 300)
    edges_device = edges.to(device)

    for epoch in range(start_epoch, args.epochs):
        model.eval()
        for head in heads.values():
            head.train()
        torch.cuda.manual_seed_all(args.seed + 10_000 + epoch)
        generator = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(len(train_rows), generator=generator)
        loss_sum = {name: 0.0 for name in ARM_LABELS}
        seen = 0

        for batch_index, start in enumerate(
            range(0, len(train_rows), args.micro_batch_size)
        ):
            local = order[start : start + args.micro_batch_size]
            batch_texts = [train_texts[int(index)] for index in local]
            live = live_embeddings(
                batch_texts,
                tokenizer,
                model,
                device,
                max_length=int(prepared["max_length"]),
            )
            cached = train_cached[local].to(device)
            z_spec = train_z[local].to(device)

            baseline_cached_logits, baseline_live_logits = paired_logits(
                heads["A_cached_baseline"],
                heads["C_live_baseline"],
                cached,
                live,
                device=device,
                autocast_head=False,
            )
            direct_cached_logits, direct_live_logits = paired_logits(
                heads["B_cached_direct"],
                heads["D_live_direct"],
                cached,
                live,
                device=device,
                autocast_head=True,
            )
            logits = {
                "A_cached_baseline": baseline_cached_logits,
                "B_cached_direct": direct_cached_logits,
                "C_live_baseline": baseline_live_logits,
                "D_live_direct": direct_live_logits,
            }
            losses = {
                name: redshift_cross_entropy_loss(
                    value, z_spec, edges=edges_device
                )
                for name, value in logits.items()
            }
            baseline_scale = batch_loss_scale(
                batch_index,
                n_examples=len(train_rows),
                n_batches=n_batches,
                micro_batch_size=args.micro_batch_size,
                accumulation=baseline_accumulation,
            )
            direct_scale = batch_loss_scale(
                batch_index,
                n_examples=len(train_rows),
                n_batches=n_batches,
                micro_batch_size=args.micro_batch_size,
                accumulation=direct_accumulation,
            )
            for name in BASELINE_ARMS:
                (losses[name] * baseline_scale).backward()
            for name in DIRECT_ARMS:
                (losses[name] * direct_scale).backward()

            batch_count = int(len(local))
            for name in ARM_LABELS:
                loss_sum[name] += float(losses[name].detach()) * batch_count
            seen += batch_count

            if is_update_batch(
                batch_index,
                n_batches=n_batches,
                accumulation=baseline_accumulation,
            ):
                for name in BASELINE_ARMS:
                    optimizers[name].step()
                    optimizers[name].zero_grad(set_to_none=True)
                baseline_updates += 1

            if is_update_batch(
                batch_index,
                n_batches=n_batches,
                accumulation=direct_accumulation,
            ):
                for name in DIRECT_ARMS:
                    torch.nn.utils.clip_grad_norm_(
                        heads[name].parameters(), args.direct_max_grad_norm
                    )
                    optimizers[name].step()
                    schedulers[name].step()
                    optimizers[name].zero_grad(set_to_none=True)
                direct_updates += 1
                if (
                    direct_updates == 1
                    or direct_updates % args.progress_updates == 0
                    or direct_updates == total_direct_updates
                ):
                    print(
                        f"Stage 1 direct update {direct_updates:,}/"
                        f"{total_direct_updates:,}; epoch {epoch + 1}/"
                        f"{args.epochs}; cached loss "
                        f"{float(losses['B_cached_direct'].detach()):.4f}; live loss "
                        f"{float(losses['D_live_direct'].detach()):.4f}",
                        flush=True,
                    )

        _, val_metrics = evaluate_arms(
            heads=heads,
            cached_embeddings=val_cached,
            texts=val_texts,
            z_spec=val_z,
            object_ids=val_ids,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=int(prepared["max_length"]),
            batch_size=args.eval_batch_size,
            redshift_edges=edges,
            redshift_centers=centers,
            progress_label=f"epoch {epoch + 1}",
        )
        for name in ARM_LABELS:
            metrics = scalar_metrics(val_metrics[name])
            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": loss_sum[name] / max(seen, 1),
                **{f"val_{key}": value for key, value in metrics.items()},
            }
            if name in DIRECT_ARMS:
                row["learning_rate"] = float(
                    schedulers[name].get_last_lr()[0]
                )
            else:
                row["learning_rate"] = args.head_learning_rate
            history[name].append(row)
            if metrics["cross_entropy"] < best_loss[name]:
                best_loss[name] = metrics["cross_entropy"]
                best_state[name] = state_on_cpu(heads[name])
                selected_epoch[name] = epoch
        print(
            "Stage 1 epoch "
            f"{epoch + 1}/{args.epochs}: "
            + "; ".join(
                f"{name[0]} CE={val_metrics[name]['cross_entropy']:.4f} "
                f"NMAD={val_metrics[name]['nmad']:.4f}"
                for name in ARM_LABELS
            ),
            flush=True,
        )
        torch.save(
            {
                "config": run_config,
                "next_epoch": epoch + 1,
                "heads": {
                    name: state_on_cpu(head) for name, head in heads.items()
                },
                "optimizers": {
                    name: optimizer.state_dict()
                    for name, optimizer in optimizers.items()
                },
                "schedulers": {
                    name: scheduler.state_dict()
                    for name, scheduler in schedulers.items()
                },
                "history": history,
                "best_loss": best_loss,
                "best_state": best_state,
                "selected_epoch": selected_epoch,
                "baseline_updates": baseline_updates,
                "direct_updates": direct_updates,
            },
            checkpoint_path,
        )
        print(f"saved {checkpoint_path}", flush=True)

    for name, state in best_state.items():
        if state is None:
            raise RuntimeError(f"No best state was recorded for {name}.")
        heads[name].load_state_dict(state)

    final_evaluations, final_metrics_raw = evaluate_arms(
        heads=heads,
        cached_embeddings=val_cached,
        texts=val_texts,
        z_spec=val_z,
        object_ids=val_ids,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=int(prepared["max_length"]),
        batch_size=args.eval_batch_size,
        redshift_edges=edges,
        redshift_centers=centers,
        progress_label="selected checkpoints",
    )
    final_metrics = {
        name: scalar_metrics(metrics)
        for name, metrics in final_metrics_raw.items()
    }
    reference_metrics = scalar_metrics(
        frozen_result["final_metrics"]["val"]
    )
    comparisons = {
        "A_vs_C_baseline_logits": tensor_comparison(
            final_evaluations["A_cached_baseline"]["logits"],
            final_evaluations["C_live_baseline"]["logits"],
        ),
        "B_vs_D_direct_logits": tensor_comparison(
            final_evaluations["B_cached_direct"]["logits"],
            final_evaluations["D_live_direct"]["logits"],
        ),
    }
    checks = {
        "A_vs_C_validation_CE_within_1pct": relative_difference(
            final_metrics["A_cached_baseline"]["cross_entropy"],
            final_metrics["C_live_baseline"]["cross_entropy"],
        )
        <= 0.01,
        "A_vs_C_validation_NMAD_within_1pct": relative_difference(
            final_metrics["A_cached_baseline"]["nmad"],
            final_metrics["C_live_baseline"]["nmad"],
        )
        <= 0.01,
        "B_vs_D_validation_CE_within_1pct": relative_difference(
            final_metrics["B_cached_direct"]["cross_entropy"],
            final_metrics["D_live_direct"]["cross_entropy"],
        )
        <= 0.01,
        "B_vs_D_validation_NMAD_within_1pct": relative_difference(
            final_metrics["B_cached_direct"]["nmad"],
            final_metrics["D_live_direct"]["nmad"],
        )
        <= 0.01,
        "new_A_vs_existing_A_validation_CE_within_1pct": relative_difference(
            reference_metrics["cross_entropy"],
            final_metrics["A_cached_baseline"]["cross_entropy"],
        )
        <= 0.01,
        "new_A_vs_existing_A_validation_NMAD_within_1pct": relative_difference(
            reference_metrics["nmad"],
            final_metrics["A_cached_baseline"]["nmad"],
        )
        <= 0.01,
    }
    gate_passed = (
        checks["A_vs_C_validation_CE_within_1pct"]
        and checks["A_vs_C_validation_NMAD_within_1pct"]
    )
    checks["stage1_live_baseline_gate_passed"] = gate_passed

    direct_ce_delta = (
        final_metrics["D_live_direct"]["cross_entropy"]
        / final_metrics["C_live_baseline"]["cross_entropy"]
        - 1.0
    )
    if not gate_passed:
        interpretation = (
            "The live frozen GLM path does not reproduce the paired cached "
            "baseline within the Stage 1 gate. Investigate batch-dependent "
            "routing or live execution before adapter experiments."
        )
    elif not (
        checks["B_vs_D_validation_CE_within_1pct"]
        and checks["B_vs_D_validation_NMAD_within_1pct"]
    ):
        interpretation = (
            "The baseline live gate passes, but the direct cached/live pair "
            "diverges. The direct microbatch path remains sensitive to live "
            "execution and should be resolved before adapter experiments."
        )
    elif direct_ce_delta > 0.01:
        interpretation = (
            "The live representation path is valid, while the direct head "
            "protocol is more than 1% worse in validation CE. Head optimizer "
            "settings are the primary remaining control failure."
        )
    else:
        interpretation = (
            "Cached and live paths match under both protocols, and the direct "
            "head protocol is not more than 1% worse in validation CE. Stage 1 "
            "clears live frozen training; adapter optimization can be studied "
            "without invoking a representation or head-protocol failure."
        )

    curves_path = output_dir / "stage1_validation_curves.png"
    plot_history(history, reference_metrics, curves_path)
    result = {
        "stage": "glm52_stage1_head_controls",
        "config": run_config,
        "catalogue_preparation": preparation,
        "split": {
            "train_size": len(train_rows),
            "val_size": len(val_rows),
            "test_evaluated": False,
            "train_object_ids_sha256": sha256_lines(train_ids),
            "val_object_ids_sha256": sha256_lines(val_ids),
        },
        "protocols": {
            "baseline": {
                "effective_batch_size": args.baseline_effective_batch_size,
                "physical_batch_size": args.micro_batch_size,
                "gradient_accumulation_steps": baseline_accumulation,
                "learning_rate": args.head_learning_rate,
                "schedule": "constant",
                "weight_decay": args.baseline_weight_decay,
                "head_autocast": False,
            },
            "direct": {
                "effective_batch_size": args.direct_effective_batch_size,
                "physical_batch_size": args.micro_batch_size,
                "gradient_accumulation_steps": direct_accumulation,
                "learning_rate": args.head_learning_rate,
                "schedule": "linear_warmup_decay",
                "warmup_fraction": args.direct_warmup_fraction,
                "weight_decay": args.direct_weight_decay,
                "head_autocast": "bfloat16",
                "max_grad_norm": args.direct_max_grad_norm,
            },
        },
        "existing_corrected_A_reference": reference_metrics,
        "history": history,
        "selected_epoch": selected_epoch,
        "final_validation_metrics": final_metrics,
        "logit_comparisons": comparisons,
        "checks": checks,
        "interpretation": interpretation,
        "artifacts": {
            "validation_curves": str(curves_path),
            "checkpoint": str(checkpoint_path),
        },
    }
    scalar_report_path = output_dir / "stage1_report.json"
    scalar_report_path.write_text(
        json.dumps(report_without_evaluations(result), indent=2, sort_keys=True)
        + "\n"
    )
    markdown_path = output_dir / "stage1_report.md"
    write_markdown(markdown_path, result)
    result_path = output_dir / "stage1_result.pt"
    torch.save(
        {
            **result,
            "best_head_state": best_state,
            "validation_evaluations": final_evaluations,
            "redshift_edges": edges,
            "redshift_centers": centers,
        },
        result_path,
    )
    print(json.dumps({"checks": checks, "interpretation": interpretation}, indent=2))
    print(f"saved {scalar_report_path}", flush=True)
    print(f"saved {markdown_path}", flush=True)
    print(f"saved {result_path}", flush=True)

    del heads, optimizers, schedulers, model
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

