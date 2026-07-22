#!/usr/bin/env python3
"""Compare a frozen Qwen photo-z probe against task-specific QLoRA post-training."""

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

import qwen_mlp_full_comparison as base
from aion_magnitude.FM_Qwen import (
    QwenEmbeddingConfig,
    QwenSerializationConfig,
    serialize_qwen_feature_row,
)
from aion_magnitude.clauds_bands import HSC_AION_BANDS
from aion_magnitude.morphology import (
    AIONMorphologyConfig,
    cache_aion_morphology_tokens,
    resolve_morphology_paths,
    save_morphology_comparison_artifacts,
    train_single_morphology_model,
)
from aion_magnitude.qwen_posttraining import (
    QwenPosttrainingConfig,
    TextRedshiftDataset,
    train_qlora_photoz,
)


COMPARISON_NAME = "qwen-qwen_posttraining_comparison"
DEFAULT_OUTPUT_DIR = Path(
    "/arc/home/gsm/aion_output/figures/qwen-qwen_posttraining_comparison"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "frozen", "qlora", "collect"), required=True
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=ROOT / "data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits",
    )
    parser.add_argument(
        "--morphology-dir",
        type=Path,
        default=ROOT / "data/clauds/images/tilesv5",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cache-root", type=Path, default=Path("/scratch/.tmp-gsm/aion_output/cache")
    )
    parser.add_argument("--max-rows", type=base.parse_max_rows, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-batch-size", type=int, default=64)
    parser.add_argument("--image-flux-scale", type=float, default=1.0)
    parser.add_argument("--min-cutout-weight-coverage", type=float, default=0.90)
    parser.add_argument("--n-z-bins", type=int, default=300)
    parser.add_argument("--tomographic-samples", type=int, default=100)
    parser.add_argument("--qwen-model", default="Qwen3.5-4B-Base")
    parser.add_argument("--qwen-max-length", type=int, default=2048)
    parser.add_argument("--qwen-pooling", choices=("last",), default="last")
    parser.add_argument("--qwen-embedding-batch-size", type=int, default=8)
    parser.add_argument("--allow-qwen-download", action="store_true")
    parser.add_argument("--force-rebuild-tokens", action="store_true")
    parser.add_argument("--force-rebuild-photometry", action="store_true")
    parser.add_argument("--force-recompute-qwen", action="store_true")
    parser.add_argument("--frozen-epochs", type=int, default=10)
    parser.add_argument("--frozen-train-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--head-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--qlora-epochs", type=int, default=3)
    parser.add_argument("--qlora-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--qlora-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "token_batch_size": args.token_batch_size,
        "n_z_bins": args.n_z_bins,
        "qwen_max_length": args.qwen_max_length,
        "qwen_embedding_batch_size": args.qwen_embedding_batch_size,
        "frozen_epochs": args.frozen_epochs,
        "frozen_train_batch_size": args.frozen_train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "qlora_epochs": args.qlora_epochs,
        "qlora_batch_size": args.qlora_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
    }
    invalid = [name for name, value in positive.items() if int(value) < 1]
    if invalid:
        raise ValueError(f"Positive settings required: {invalid}")
    if args.qwen_pooling != "last":
        raise ValueError("This comparison requires last-token pooling.")


def build_product(
    args: argparse.Namespace,
) -> tuple[dict, AIONMorphologyConfig, dict, tuple[float, float]]:
    catalogue = base.resolve_existing_path(args.catalogue)
    morphology_dir = base.resolve_existing_path(args.morphology_dir)
    cache_root = Path(args.cache_root).expanduser()
    selection_tag = "all" if args.max_rows is None else f"n{args.max_rows}"
    run_cache_dir = cache_root / "qwen_mlp_full_comparison" / selection_tag
    split_output_dir = run_cache_dir / "clauds_split"
    z_min, z_max = base.finite_catalogue_redshift_bounds(
        catalogue,
        split_output_dir,
        max_rows=args.max_rows,
    )
    config = AIONMorphologyConfig(
        catalogue_path=catalogue,
        morphology_dir=morphology_dir,
        cache_root=cache_root,
        split_output_dir=split_output_dir,
        photometry_cache_path=run_cache_dir / "photometry_no_magnitude_or_redshift_cut.pt",
        output_dir=Path(args.output_dir).expanduser(),
        max_rows=args.max_rows,
        sample_mode="random",
        sample_seed=args.seed,
        sample_require_valid_bands=(),
        force_rebuild_photometry=args.force_rebuild_photometry,
        force_rebuild_tokens=args.force_rebuild_tokens,
        preserve_photometry_splits=True,
        z_min=z_min,
        z_max=z_max,
        redshift_include_min=True,
        redshift_include_max=True,
        n_z_bins=args.n_z_bins,
        hsc_mag_faint_limits={band: None for band in HSC_AION_BANDS},
        extra_band_invalid_fill="median",
        extra_band_include_valid_flags=False,
        include_grizy_in_mlp=True,
        use_aion_magnitude_embedding=False,
        image_flux_scale=args.image_flux_scale,
        min_cutout_weight_coverage=args.min_cutout_weight_coverage,
        token_batch_size=args.token_batch_size,
        model_kinds=("qwen_morphology",),
        feature_scaling="none",
        epochs=args.frozen_epochs,
        learning_rate=args.head_learning_rate,
        train_batch_size=args.frozen_train_batch_size,
        eval_batch_size=max(args.eval_batch_size, 64),
        tomographic_samples=args.tomographic_samples,
        seed=args.seed,
        device_choice="cuda",
    ).normalized()
    paths = resolve_morphology_paths(config)
    product = cache_aion_morphology_tokens(config)
    return product, config, paths, (z_min, z_max)


def qwen_settings(args: argparse.Namespace):
    config = QwenEmbeddingConfig(
        model_path=args.qwen_model,
        device="cuda",
        load_in_4bit=True,
        torch_dtype="auto",
        max_length=args.qwen_max_length,
        pooling="last",
        normalize=False,
        local_files_only=not args.allow_qwen_download,
        trust_remote_code=True,
    )
    serialization = QwenSerializationConfig(
        schema_name="clauds_all_magnitude_v1",
        include_hsc_grizy=False,
        include_object_metadata=False,
        prefix="galaxy all_magnitudes_ab",
    )
    return config, serialization


def qwen_cache_path(args: argparse.Namespace, paths: dict, config) -> Path:
    return (
        Path(args.cache_root).expanduser()
        / "qwen_mlp_full_comparison"
        / str(paths["morphology_tag"])
        / f"{base.qwen_run_tag(config)}.pt"
    )


def stage_prepare(args: argparse.Namespace) -> int:
    product, _, paths, redshift_bounds = build_product(args)
    split_labels = np.asarray(product["split_labels"], dtype=object)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "comparison": COMPARISON_NAME,
        "morphology_product": str(paths["morphology_product_path"]),
        "pooling": "last",
        "max_rows": args.max_rows,
        "sample_mode": "random",
        "sample_seed": args.seed,
        "redshift_bounds": list(redshift_bounds),
        "rows": {
            split: int(np.sum(split_labels == split))
            for split in ("train", "val", "test")
        },
    }
    (output_dir / "prepared.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def stage_frozen(args: argparse.Namespace) -> int:
    product, config, paths, _ = build_product(args)
    qwen_config, serialization = qwen_settings(args)
    cache_path = qwen_cache_path(args, paths, qwen_config)
    embeddings = base.extract_or_load_qwen_embeddings(
        args,
        product,
        qwen_config,
        serialization,
        cache_path,
        torch.device("cuda"),
    )
    qwen_product = dict(product)
    qwen_product["aion_embedding"] = embeddings
    metadata = dict(qwen_product.get("metadata", {}))
    metadata.update(
        {
            "posttraining_method": "frozen_qwen_control",
            "qwen_pooling": "last",
            "qwen_cache_path": str(cache_path),
        }
    )
    qwen_product["metadata"] = metadata
    output_dir = Path(args.output_dir).expanduser() / "frozen"
    result = train_single_morphology_model(
        qwen_product,
        "aion",
        output_dir=output_dir,
        config=config,
        device="cuda",
    )
    result.pop("model", None)
    torch.save(result, output_dir / "result.pt")
    print(f"saved {output_dir / 'result.pt'}", flush=True)
    return 0


def _serialize_product(product: dict, serialization) -> list[str]:
    features = torch.as_tensor(product["extra_features"], dtype=torch.float32)
    names = [str(name) for name in product["feature_names"]]
    if features.ndim != 2 or features.shape[1] != len(names):
        raise ValueError("Magnitude matrix and feature names are inconsistent.")
    return [
        serialize_qwen_feature_row(
            {name: features[row, column] for column, name in enumerate(names)},
            serialization=serialization,
        )
        for row in range(len(features))
    ]


def stage_qlora(args: argparse.Namespace) -> int:
    product, _, _, redshift_bounds = build_product(args)
    _, serialization = qwen_settings(args)
    texts = _serialize_product(product, serialization)
    redshifts = torch.as_tensor(product["z_spec"], dtype=torch.float32)
    object_ids = [str(value) for value in product["object_id"]]
    split_labels = np.asarray(product["split_labels"], dtype=object)

    def dataset(split: str) -> TextRedshiftDataset:
        rows = np.flatnonzero(split_labels == split)
        return TextRedshiftDataset(
            [texts[index] for index in rows],
            redshifts[torch.as_tensor(rows, dtype=torch.long)],
            [object_ids[index] for index in rows],
        )

    config = QwenPosttrainingConfig(
        model_path=args.qwen_model,
        max_length=args.qwen_max_length,
        pooling="last",
        n_z_bins=args.n_z_bins,
        z_min=redshift_bounds[0],
        z_max=redshift_bounds[1],
        epochs=args.qlora_epochs,
        batch_size=args.qlora_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.qlora_learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        device="cuda",
        local_files_only=not args.allow_qwen_download,
    ).normalized()
    output_dir = Path(args.output_dir).expanduser() / "qlora"
    train_qlora_photoz(
        train_dataset=dataset("train"),
        val_dataset=dataset("val"),
        test_dataset=dataset("test"),
        output_dir=output_dir,
        config=config,
    )
    print(f"saved {output_dir / 'result.pt'}", flush=True)
    return 0


def stage_collect(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser()
    frozen_path = output_dir / "frozen/result.pt"
    qlora_path = output_dir / "qlora/result.pt"
    missing = [str(path) for path in (frozen_path, qlora_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Post-training result files are missing: {missing}")
    results = {
        "frozen": torch.load(frozen_path, map_location="cpu", weights_only=False),
        "qlora": torch.load(qlora_path, map_location="cpu", weights_only=False),
    }
    artifacts = save_morphology_comparison_artifacts(
        results,
        model_kinds=("frozen", "qlora"),
        output_dir=output_dir,
        tomographic_samples=args.tomographic_samples,
        comparison_labels=(
            "frozen-Qwen+photo-z-head",
            "QLoRA-Qwen+photo-z-head",
        ),
        comparison_prefix=output_dir / COMPARISON_NAME,
    )
    torch.save(results, output_dir / "results.pt")
    summary = {
        "comparison": COMPARISON_NAME,
        "pooling": "last",
        "qwen_model": args.qwen_model,
        "frozen_result": str(frozen_path),
        "qlora_result": str(qlora_path),
        "final_metrics": {
            name: result["final_metrics"] for name, result in results.items()
        },
        "artifacts": artifacts,
    }
    (output_dir / "run.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary["final_metrics"], indent=2, sort_keys=True), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    return {
        "prepare": stage_prepare,
        "frozen": stage_frozen,
        "qlora": stage_qlora,
        "collect": stage_collect,
    }[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
