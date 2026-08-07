#!/usr/bin/env python3
"""Run IA3 or a residual cached-embedding adapter for Qwen photo-z."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import qwen_posttraining_comparison as comparison
from aion_magnitude.qwen_alternative_posttraining import (
    EmbeddingRedshiftDataset,
    ResidualEmbeddingAdapterConfig,
    train_ia3_photoz,
    train_residual_embedding_adapter,
)
from aion_magnitude.qwen_posttraining import (
    QwenPosttrainingConfig,
    TextRedshiftDataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = comparison.build_parser()
    parser.description = __doc__
    stage_action = next(
        action for action in parser._actions if action.dest == "stage"
    )
    stage_action.choices = ("prepare", "ia3", "embedding-adapter")

    parser.add_argument("--ia3-epochs", type=int, default=10)
    parser.add_argument("--ia3-batch-size", type=int, default=1)
    parser.add_argument("--ia3-gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--ia3-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--ia3-head-warmup-epochs", type=int, default=3)
    parser.add_argument("--ia3-max-grad-norm", type=float, default=0.1)
    parser.add_argument(
        "--ia3-target-modules",
        default="k_proj,v_proj,down_proj",
    )
    parser.add_argument("--ia3-feedforward-modules", default="down_proj")
    parser.add_argument("--ia3-checkpoint-dir", type=Path)
    parser.add_argument("--ia3-checkpoint-steps", type=int, default=100)
    parser.add_argument(
        "--ia3-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--embedding-adapter-epochs", type=int, default=10)
    parser.add_argument("--embedding-adapter-batch-size", type=int, default=16)
    parser.add_argument(
        "--embedding-adapter-eval-batch-size",
        type=int,
        default=8,
    )
    parser.add_argument("--embedding-adapter-bottleneck-dim", type=int, default=256)
    parser.add_argument("--embedding-adapter-learning-rate", type=float, default=2.0e-4)
    parser.add_argument(
        "--embedding-adapter-adapter-learning-rate", type=float, default=1.0e-5
    )
    parser.add_argument("--embedding-adapter-weight-decay", type=float, default=0.01)
    parser.add_argument("--embedding-adapter-head-warmup-epochs", type=int, default=3)
    parser.add_argument("--embedding-adapter-max-grad-norm", type=float, default=0.1)
    parser.add_argument("--embedding-adapter-dropout", type=float, default=0.05)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    comparison._validate_args(args)
    positive = {
        "ia3_epochs": args.ia3_epochs,
        "ia3_batch_size": args.ia3_batch_size,
        "ia3_gradient_accumulation_steps": args.ia3_gradient_accumulation_steps,
        "ia3_checkpoint_steps": args.ia3_checkpoint_steps,
        "embedding_adapter_epochs": args.embedding_adapter_epochs,
        "embedding_adapter_batch_size": args.embedding_adapter_batch_size,
        "embedding_adapter_eval_batch_size": args.embedding_adapter_eval_batch_size,
        "embedding_adapter_bottleneck_dim": args.embedding_adapter_bottleneck_dim,
    }
    invalid = [name for name, value in positive.items() if int(value) < 1]
    if invalid:
        raise ValueError(f"Positive settings required: {invalid}")
    if not 0 <= args.ia3_head_warmup_epochs < args.ia3_epochs:
        raise ValueError(
            "--ia3-head-warmup-epochs must be non-negative and smaller than "
            "--ia3-epochs."
        )
    if args.ia3_learning_rate <= 0 or args.ia3_max_grad_norm <= 0:
        raise ValueError("IA3 learning rate and gradient norm must be positive.")
    if (
        args.embedding_adapter_learning_rate <= 0
        or args.embedding_adapter_adapter_learning_rate <= 0
        or args.embedding_adapter_max_grad_norm <= 0
    ):
        raise ValueError(
            "Embedding-adapter learning rates and gradient norm must be positive."
        )
    if not 0 <= args.embedding_adapter_head_warmup_epochs < args.embedding_adapter_epochs:
        raise ValueError(
            "--embedding-adapter-head-warmup-epochs must be non-negative and "
            "smaller than --embedding-adapter-epochs."
        )
    if not 0 <= args.embedding_adapter_dropout < 1:
        raise ValueError("Embedding-adapter dropout must be in [0, 1).")


def _text_datasets(
    args: argparse.Namespace,
) -> tuple[
    TextRedshiftDataset,
    TextRedshiftDataset,
    TextRedshiftDataset,
    tuple[float, float],
]:
    product, _, redshift_bounds = comparison.build_product(args)
    _, serialization = comparison.qwen_settings(args)
    texts = comparison._serialize_product(product, serialization)
    redshifts = torch.as_tensor(product["z_spec"], dtype=torch.float32)
    object_ids = [str(value) for value in product["object_id"]]
    split_labels = np.asarray(product["split_labels"], dtype=object)

    def dataset(split: str) -> TextRedshiftDataset:
        rows = np.flatnonzero(split_labels == split)
        row_tensor = torch.as_tensor(rows, dtype=torch.long)
        return TextRedshiftDataset(
            [texts[index] for index in rows],
            redshifts[row_tensor],
            [object_ids[index] for index in rows],
        )

    return dataset("train"), dataset("val"), dataset("test"), redshift_bounds


def stage_ia3(args: argparse.Namespace) -> int:
    train_dataset, val_dataset, test_dataset, bounds = _text_datasets(args)
    config = QwenPosttrainingConfig(
        model_path=args.qwen_model,
        max_length=args.qwen_max_length,
        pooling="last",
        n_z_bins=args.n_z_bins,
        z_min=bounds[0],
        z_max=bounds[1],
        epochs=args.ia3_epochs,
        batch_size=args.ia3_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.ia3_gradient_accumulation_steps,
        learning_rate=args.head_learning_rate,
        lora_learning_rate=args.ia3_learning_rate,
        lora_max_grad_norm=args.ia3_max_grad_norm,
        head_warmup_epochs=args.ia3_head_warmup_epochs,
        seed=args.seed,
        device="cuda",
        local_files_only=not args.allow_qwen_download,
    ).normalized()
    output_dir = Path(args.output_dir).expanduser() / "ia3"
    result = train_ia3_photoz(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        output_dir=output_dir,
        config=config,
        checkpoint_dir=args.ia3_checkpoint_dir,
        checkpoint_interval=args.ia3_checkpoint_steps,
        resume=args.ia3_resume,
        target_modules=args.ia3_target_modules,
        feedforward_modules=args.ia3_feedforward_modules,
    )
    _write_method_summary(args, result, method="ia3")
    print(f"saved {output_dir / 'result.pt'}", flush=True)
    return 0


def _embedding_datasets(
    args: argparse.Namespace,
) -> tuple[
    EmbeddingRedshiftDataset,
    EmbeddingRedshiftDataset,
    EmbeddingRedshiftDataset,
    tuple[float, float],
    Path,
]:
    product, _, redshift_bounds = comparison.build_product(args)
    qwen_config, serialization = comparison.qwen_settings(args)
    cache_path = comparison.qwen_cache_path(args, product, qwen_config)
    embeddings = comparison.base.extract_or_load_qwen_embeddings(
        args,
        product,
        qwen_config,
        serialization,
        cache_path,
        torch.device("cuda"),
    )
    redshifts = torch.as_tensor(product["z_spec"], dtype=torch.float32)
    object_ids = [str(value) for value in product["object_id"]]
    split_labels = np.asarray(product["split_labels"], dtype=object)

    def dataset(split: str) -> EmbeddingRedshiftDataset:
        rows = np.flatnonzero(split_labels == split)
        row_tensor = torch.as_tensor(rows, dtype=torch.long)
        return EmbeddingRedshiftDataset(
            embeddings[row_tensor],
            redshifts[row_tensor],
            [object_ids[index] for index in rows],
        )

    return (
        dataset("train"),
        dataset("val"),
        dataset("test"),
        redshift_bounds,
        cache_path,
    )


def stage_embedding_adapter(args: argparse.Namespace) -> int:
    train_dataset, val_dataset, test_dataset, bounds, cache_path = (
        _embedding_datasets(args)
    )
    config = ResidualEmbeddingAdapterConfig(
        n_z_bins=args.n_z_bins,
        z_min=bounds[0],
        z_max=bounds[1],
        bottleneck_dim=args.embedding_adapter_bottleneck_dim,
        epochs=args.embedding_adapter_epochs,
        batch_size=args.embedding_adapter_batch_size,
        eval_batch_size=args.embedding_adapter_eval_batch_size,
        learning_rate=args.embedding_adapter_learning_rate,
        adapter_learning_rate=args.embedding_adapter_adapter_learning_rate,
        weight_decay=args.embedding_adapter_weight_decay,
        head_warmup_epochs=args.embedding_adapter_head_warmup_epochs,
        adapter_max_grad_norm=args.embedding_adapter_max_grad_norm,
        dropout=args.embedding_adapter_dropout,
        seed=args.seed,
        device="cuda",
    ).normalized()
    output_dir = Path(args.output_dir).expanduser() / "embedding_adapter"
    result = train_residual_embedding_adapter(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        output_dir=output_dir,
        config=config,
    )
    result["metadata"]["qwen_cache_path"] = str(cache_path)
    torch.save(result, output_dir / "result.pt")
    _write_method_summary(args, result, method="embedding_adapter")
    print(f"saved {output_dir / 'result.pt'}", flush=True)
    return 0


def _write_method_summary(
    args: argparse.Namespace,
    result: dict,
    *,
    method: str,
) -> None:
    output_dir = Path(args.output_dir).expanduser()
    prepared_path = output_dir / "prepared.json"
    prepared = json.loads(prepared_path.read_text()) if prepared_path.exists() else {}
    summary = {
        "method": method,
        "catalogue": prepared.get("catalogue"),
        "input_representation": prepared.get("input_representation"),
        "pooling": "last",
        "qwen_model": args.qwen_model,
        "result": str(
            output_dir
            / ("ia3" if method == "ia3" else "embedding_adapter")
            / "result.pt"
        ),
        "final_metrics": result["final_metrics"],
    }
    (output_dir / f"{method}_run.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    if args.stage == "prepare":
        return comparison.stage_prepare(args)
    if args.stage == "ia3":
        return stage_ia3(args)
    return stage_embedding_adapter(args)


if __name__ == "__main__":
    raise SystemExit(main())
