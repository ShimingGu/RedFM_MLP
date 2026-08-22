#!/usr/bin/env python3
"""Continue a completed encoder-level AION QLoRA policy with RLVR."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aion_magnitude.aion_qlora_rlvr import (
    load_aion_qlora_source_artifact,
    train_aion_qlora_rlvr,
)
from aion_magnitude.qwen_rlvr import RLVRConfig
from aion_magnitude.utils import set_random_seed
from notebooks import aion_posttraining as shared


def main(argv: list[str] | None = None) -> int:
    parser = shared.build_parser()
    supplied = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(["--stage", "rlvr", *supplied])
    shared._validate_args(args)
    if args.rlvr_source_result_path is None:
        parser.error("--rlvr-source-result-path is required for AION QLoRA RLVR")
    set_random_seed(args.seed)

    _, _, source_result, _ = load_aion_qlora_source_artifact(
        args.rlvr_source_result_path
    )
    source_mode = source_result.get("metadata", {}).get("aion_input_mode")
    if source_mode != args.input_mode:
        raise ValueError(
            "AION QLoRA source input mode does not match this run: "
            f"source={source_mode!r}, requested={args.input_mode!r}"
        )

    product, cache_path = shared._load_prepared_product(args)
    train, val, test = shared._token_datasets(product)
    checkpoint_dir = (
        args.rlvr_checkpoint_dir
        if args.rlvr_checkpoint_dir is not None
        else args.output_dir.expanduser() / "rlvr_checkpoints"
    )
    config = RLVRConfig(
        epochs=args.rlvr_epochs,
        batch_size=args.rlvr_batch_size,
        eval_batch_size=args.rlvr_eval_batch_size,
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
    output_dir = args.output_dir.expanduser() / "rlvr"
    result = train_aion_qlora_rlvr(
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        output_dir=output_dir,
        source_result_path=args.rlvr_source_result_path,
        config=config,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.rlvr_checkpoint_steps,
        resume=args.rlvr_resume,
    )
    result["metadata"].update(
        {
            "aion_input_mode": args.input_mode,
            "product_cache_path": str(cache_path),
            "aion_only": True,
            "uses_qwen": False,
            "comparison_role": "encoder_qlora_rlvr",
        }
    )
    torch.save(result, output_dir / "result.pt")
    shared._write_summary(
        args,
        method="rlvr",
        result=result,
        product_path=cache_path,
    )
    print(f"saved {output_dir / 'result.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
