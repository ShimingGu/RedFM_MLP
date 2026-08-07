#!/usr/bin/env python3
"""Run controlled DoRA-SFT or QLoRA-SFT-to-RLVR photo-z post-training."""

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
from aion_magnitude.qwen_posttraining import (
    QwenPosttrainingConfig,
    TextRedshiftDataset,
    train_qlora_photoz,
)
from aion_magnitude.qwen_rlvr import RLVRConfig, train_qlora_rlvr


DEFAULT_QLORA_RESULT = Path(
    "/arc/home/gsm/aion_output/figures/"
    "qwen-qwen_posttraining_comparison-e10/qlora"
)


def build_parser() -> argparse.ArgumentParser:
    parser = comparison.build_parser()
    parser.description = __doc__
    stage_action = next(
        action for action in parser._actions if action.dest == "stage"
    )
    stage_action.choices = ("prepare", "dora", "rlvr")

    parser.add_argument("--dora-epochs", type=int, default=10)
    parser.add_argument("--dora-batch-size", type=int, default=1)
    parser.add_argument("--dora-gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--dora-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--dora-head-warmup-epochs", type=int, default=3)
    parser.add_argument("--dora-max-grad-norm", type=float, default=0.1)
    parser.add_argument("--dora-rank", type=int, default=8)
    parser.add_argument("--dora-alpha", type=int, default=16)
    parser.add_argument("--dora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--dora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
    )
    parser.add_argument("--dora-checkpoint-dir", type=Path)
    parser.add_argument("--dora-checkpoint-steps", type=int, default=100)
    parser.add_argument(
        "--dora-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--rlvr-epochs", type=int, default=1)
    parser.add_argument("--rlvr-batch-size", type=int, default=1)
    parser.add_argument("--rlvr-gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--rlvr-head-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--rlvr-adapter-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--rlvr-group-samples", type=int, default=8)
    parser.add_argument("--rlvr-reward-scale", type=float, default=0.05)
    parser.add_argument("--rlvr-outlier-threshold", type=float, default=0.15)
    parser.add_argument("--rlvr-outlier-penalty", type=float, default=0.5)
    parser.add_argument("--rlvr-kl-beta", type=float, default=0.02)
    parser.add_argument("--rlvr-entropy-coefficient", type=float, default=0.001)
    parser.add_argument(
        "--rlvr-source-adapter-dir",
        type=Path,
        default=DEFAULT_QLORA_RESULT / "adapter",
    )
    parser.add_argument(
        "--rlvr-source-head-path",
        type=Path,
        default=DEFAULT_QLORA_RESULT / "photoz_head.pt",
    )
    parser.add_argument("--rlvr-checkpoint-dir", type=Path)
    parser.add_argument("--rlvr-checkpoint-steps", type=int, default=100)
    parser.add_argument(
        "--rlvr-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    comparison._validate_args(args)
    positive = {
        "dora_epochs": args.dora_epochs,
        "dora_batch_size": args.dora_batch_size,
        "dora_gradient_accumulation_steps": args.dora_gradient_accumulation_steps,
        "dora_rank": args.dora_rank,
        "dora_alpha": args.dora_alpha,
        "dora_checkpoint_steps": args.dora_checkpoint_steps,
        "rlvr_epochs": args.rlvr_epochs,
        "rlvr_batch_size": args.rlvr_batch_size,
        "rlvr_gradient_accumulation_steps": args.rlvr_gradient_accumulation_steps,
        "rlvr_group_samples": args.rlvr_group_samples,
        "rlvr_checkpoint_steps": args.rlvr_checkpoint_steps,
    }
    invalid = [name for name, value in positive.items() if int(value) < 1]
    if invalid:
        raise ValueError(f"Positive settings required: {invalid}")
    if not 0 <= args.dora_head_warmup_epochs < args.dora_epochs:
        raise ValueError(
            "--dora-head-warmup-epochs must be non-negative and smaller than "
            "--dora-epochs."
        )
    if args.dora_learning_rate <= 0 or args.dora_max_grad_norm <= 0:
        raise ValueError("DoRA learning rate and gradient norm must be positive.")
    RLVRConfig(
        epochs=args.rlvr_epochs,
        batch_size=args.rlvr_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.rlvr_gradient_accumulation_steps,
        head_learning_rate=args.rlvr_head_learning_rate,
        adapter_learning_rate=args.rlvr_adapter_learning_rate,
        group_samples=args.rlvr_group_samples,
        reward_scale=args.rlvr_reward_scale,
        outlier_threshold=args.rlvr_outlier_threshold,
        outlier_penalty=args.rlvr_outlier_penalty,
        kl_beta=args.rlvr_kl_beta,
        entropy_coefficient=args.rlvr_entropy_coefficient,
        seed=args.seed,
    ).normalized()


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


def _base_config(
    args: argparse.Namespace,
    bounds: tuple[float, float],
    *,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    adapter_learning_rate: float,
    head_warmup_epochs: int,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: str = "q_proj,k_proj,v_proj,o_proj",
) -> QwenPosttrainingConfig:
    return QwenPosttrainingConfig(
        model_path=args.qwen_model,
        max_length=args.qwen_max_length,
        pooling="last",
        n_z_bins=args.n_z_bins,
        z_min=bounds[0],
        z_max=bounds[1],
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.head_learning_rate,
        lora_learning_rate=adapter_learning_rate,
        lora_max_grad_norm=args.dora_max_grad_norm,
        head_warmup_epochs=head_warmup_epochs,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        lora_target_modules=target_modules,
        seed=args.seed,
        device="cuda",
        local_files_only=not args.allow_qwen_download,
    ).normalized()


def stage_dora(args: argparse.Namespace) -> int:
    train_dataset, val_dataset, test_dataset, bounds = _text_datasets(args)
    config = _base_config(
        args,
        bounds,
        epochs=args.dora_epochs,
        batch_size=args.dora_batch_size,
        gradient_accumulation_steps=args.dora_gradient_accumulation_steps,
        adapter_learning_rate=args.dora_learning_rate,
        head_warmup_epochs=args.dora_head_warmup_epochs,
        rank=args.dora_rank,
        alpha=args.dora_alpha,
        dropout=args.dora_dropout,
        target_modules=args.dora_target_modules,
    )
    output_dir = Path(args.output_dir).expanduser() / "dora"
    result = train_qlora_photoz(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        output_dir=output_dir,
        config=config,
        checkpoint_dir=args.dora_checkpoint_dir,
        checkpoint_interval=args.dora_checkpoint_steps,
        resume=args.dora_resume,
        use_dora=True,
    )
    _write_summary(args, result, method="dora")
    print(f"saved {output_dir / 'result.pt'}", flush=True)
    return 0


def stage_rlvr(args: argparse.Namespace) -> int:
    train_dataset, val_dataset, test_dataset, bounds = _text_datasets(args)
    base_config = _base_config(
        args,
        bounds,
        epochs=10,
        batch_size=1,
        gradient_accumulation_steps=16,
        adapter_learning_rate=1.0e-5,
        head_warmup_epochs=3,
    )
    rlvr_config = RLVRConfig(
        epochs=args.rlvr_epochs,
        batch_size=args.rlvr_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.rlvr_gradient_accumulation_steps,
        head_learning_rate=args.rlvr_head_learning_rate,
        adapter_learning_rate=args.rlvr_adapter_learning_rate,
        group_samples=args.rlvr_group_samples,
        reward_scale=args.rlvr_reward_scale,
        outlier_threshold=args.rlvr_outlier_threshold,
        outlier_penalty=args.rlvr_outlier_penalty,
        kl_beta=args.rlvr_kl_beta,
        entropy_coefficient=args.rlvr_entropy_coefficient,
        seed=args.seed,
    ).normalized()
    output_dir = Path(args.output_dir).expanduser() / "rlvr"
    result = train_qlora_rlvr(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        output_dir=output_dir,
        base_config=base_config,
        rlvr_config=rlvr_config,
        source_adapter_dir=args.rlvr_source_adapter_dir,
        source_head_path=args.rlvr_source_head_path,
        checkpoint_dir=args.rlvr_checkpoint_dir,
        checkpoint_interval=args.rlvr_checkpoint_steps,
        resume=args.rlvr_resume,
    )
    _write_summary(args, result, method="rlvr")
    print(f"saved {output_dir / 'result.pt'}", flush=True)
    return 0


def _write_summary(args: argparse.Namespace, result: dict, *, method: str) -> None:
    output_dir = Path(args.output_dir).expanduser()
    prepared_path = output_dir / "prepared.json"
    prepared = json.loads(prepared_path.read_text()) if prepared_path.exists() else {}
    summary = {
        "method": method,
        "catalogue": prepared.get("catalogue"),
        "input_representation": prepared.get("input_representation"),
        "pooling": "last",
        "qwen_model": args.qwen_model,
        "result": str(output_dir / method / "result.pt"),
        "final_metrics": result["final_metrics"],
    }
    if "verifier_metrics" in result:
        summary["verifier_metrics"] = result["verifier_metrics"]
    (output_dir / f"{method}_run.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    if args.stage == "prepare":
        return comparison.stage_prepare(args)
    if args.stage == "dora":
        return stage_dora(args)
    return stage_rlvr(args)


if __name__ == "__main__":
    raise SystemExit(main())
