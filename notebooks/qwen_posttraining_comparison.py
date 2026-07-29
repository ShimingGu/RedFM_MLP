#!/usr/bin/env python3
"""Compare a frozen Qwen photo-z probe against task-specific QLoRA post-training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import qwen_mlp_full_comparison as base
from aion_magnitude import table_models as tm
from aion_magnitude.dataset import make_random_split
from aion_magnitude.FM_Qwen import (
    QwenEmbeddingConfig,
    QwenSerializationConfig,
    serialize_qwen_feature_row,
)
from aion_magnitude.morphology import save_morphology_comparison_artifacts
from aion_magnitude.qwen_posttraining import (
    QwenPosttrainingConfig,
    TextRedshiftDataset,
    train_qlora_photoz,
)
from aion_magnitude.training import train_single_baseline
from aion_magnitude.utils import make_redshift_grid, set_random_seed


COMPARISON_NAME = "qwen-qwen_posttraining_comparison"
CATALOGUE_ROOT = Path(
    "/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs"
)
DEFAULT_UPDATED_CATALOGUE = (
    CATALOGUE_ROOT
    / "COSMOS-HSCpipe-Phosphoros_morphological_multiband_updated.fits"
)
FALLBACK_MULTIBAND_CATALOGUE = (
    CATALOGUE_ROOT
    / "COSMOS-HSCpipe-Phosphoros_morphological_multiband.fits"
)
DEFAULT_OUTPUT_DIR = Path(
    "/arc/home/gsm/aion_output/figures/qwen-qwen_posttraining_comparison"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "frozen", "qlora", "collect"), required=True
    )
    parser.add_argument(
        "--catalogue", type=Path, default=DEFAULT_UPDATED_CATALOGUE
    )
    parser.add_argument(
        "--use-morphology",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include the 42 measured u,g,r,i,z,y morphology fields. The cohort "
            "then requires morphology_available_* in all six bands."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cache-root", type=Path, default=Path("/scratch/.tmp-gsm/aion_output/cache")
    )
    parser.add_argument("--max-rows", type=base.parse_max_rows, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.75)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--n-z-bins", type=int, default=300)
    parser.add_argument("--tomographic-samples", type=int, default=100)
    parser.add_argument("--qwen-model", default="Qwen3.5-4B-Base")
    parser.add_argument("--qwen-max-length", type=int, default=2048)
    parser.add_argument("--qwen-pooling", choices=("last",), default="last")
    parser.add_argument("--qwen-embedding-batch-size", type=int, default=8)
    parser.add_argument("--allow-qwen-download", action="store_true")
    parser.add_argument("--force-recompute-qwen", action="store_true")
    parser.add_argument("--frozen-epochs", type=int, default=10)
    parser.add_argument("--frozen-train-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--head-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--qlora-epochs", type=int, default=10)
    parser.add_argument("--qlora-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument(
        "--qlora-learning-rate",
        type=float,
        default=1.0e-5,
        help="LoRA adapter learning rate; the photo-z head uses --head-learning-rate.",
    )
    parser.add_argument("--head-warmup-epochs", type=int, default=3)
    parser.add_argument("--lora-max-grad-norm", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--qlora-checkpoint-dir", type=Path)
    parser.add_argument("--qlora-checkpoint-steps", type=int, default=100)
    parser.add_argument("--qlora-resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
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
        "qlora_checkpoint_steps": args.qlora_checkpoint_steps,
    }
    invalid = [name for name, value in positive.items() if int(value) < 1]
    if invalid:
        raise ValueError(f"Positive settings required: {invalid}")
    if args.qwen_pooling != "last":
        raise ValueError("This comparison requires last-token pooling.")
    if args.head_warmup_epochs < 0 or args.head_warmup_epochs >= args.qlora_epochs:
        raise ValueError(
            "--head-warmup-epochs must be non-negative and smaller than --qlora-epochs."
        )
    if args.qlora_learning_rate <= 0 or args.lora_max_grad_norm <= 0:
        raise ValueError("QLoRA learning rate and gradient norm must be positive.")
    if not math.isclose(
        args.train_fraction + args.test_fraction + args.val_fraction, 1.0
    ):
        raise ValueError("Train, test, and validation fractions must sum to one.")


def resolve_catalogue_path(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    if requested.exists():
        return requested.resolve()
    if requested == DEFAULT_UPDATED_CATALOGUE and FALLBACK_MULTIBAND_CATALOGUE.exists():
        fallback = FALLBACK_MULTIBAND_CATALOGUE.resolve()
        print(
            f"revised multiband catalogue is not available yet: {requested}; "
            f"using verified fallback: {fallback}",
            flush=True,
        )
        return fallback
    raise FileNotFoundError(f"Catalogue not found: {requested}")


def build_product(
    args: argparse.Namespace,
) -> tuple[dict, dict, tuple[float, float]]:
    catalogue = resolve_catalogue_path(args.catalogue)
    data = tm.load_catalogue_data(
        catalogue,
        max_rows=args.max_rows,
        seed=args.seed,
        include_full121=False,
        include_morphology=args.use_morphology,
        z_min=-np.inf,
        z_max=np.inf,
    )
    split_labels = make_random_split(
        len(data.target),
        train_fraction=args.train_fraction,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    feature_frame = data.magnitude_features
    if args.use_morphology:
        if data.morphology_features is None:
            raise RuntimeError("Requested morphology fields were not loaded.")
        feature_frame = pd.concat(
            [feature_frame, data.morphology_features], axis=1
        )
    prepared_features, imputation = tm.impute_from_training(
        feature_frame, split_labels
    )
    feature_names = [str(name) for name in prepared_features.columns]
    feature_values = torch.from_numpy(
        prepared_features.to_numpy(dtype=np.float32, copy=True)
    )
    target = torch.from_numpy(np.asarray(data.target, dtype=np.float32))
    z_min = float(target.min().item())
    z_max = float(target.max().item())
    if z_max <= z_min:
        padding = max(abs(z_min) * 1.0e-6, 1.0e-6)
        z_min -= padding
        z_max += padding
    split_counts = {
        split: int(np.count_nonzero(split_labels == split))
        for split in ("train", "val", "test")
    }
    product = {
        "object_id": data.object_id.tolist(),
        "field": ["COSMOS"] * len(data.object_id),
        "aion_embedding": torch.empty(
            (len(data.object_id), 0), dtype=torch.float32
        ),
        "extra_features": feature_values,
        "feature_names": feature_names,
        "z_spec": target,
        "split_labels": split_labels.tolist(),
        "redshift_reference": {"zphot": target.clone()},
        "metadata": {
            "catalogue": str(catalogue),
            "input_representation": (
                "magnitudes_plus_catalogue_multiband_morphology"
                if args.use_morphology else "magnitudes_only"
            ),
            "use_morphology": bool(args.use_morphology),
            "morphology_bands": (
                list(tm.MORPHOLOGY_BANDS) if args.use_morphology else []
            ),
            "morphology_availability_rule": (
                "all" if args.use_morphology else None
            ),
            "image_cutouts_read": False,
            "image_tokens_read": False,
            "split_counts": split_counts,
            "imputation": imputation,
        },
    }
    forbidden = {"image_token_ids_path", "image_token_row_indices"} & set(product)
    if forbidden:
        raise RuntimeError(f"Catalogue product contains image-token inputs: {forbidden}")
    preparation = {
        "catalogue": str(catalogue),
        "use_morphology": bool(args.use_morphology),
        "n_rows": len(data.object_id),
        "split_counts": split_counts,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_magnitude_features": sum(name.endswith("_mag") for name in feature_names),
        "n_morphology_features": sum(
            name in tm.MORPHOLOGY_FEATURE_COLUMNS for name in feature_names
        ),
        "imputation": imputation,
    }
    return product, preparation, (z_min, z_max)


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
        schema_name=(
            "clauds_all_magnitude_multiband_morphology_v1"
            if args.use_morphology else "clauds_all_magnitude_v1"
        ),
        include_hsc_grizy=False,
        include_object_metadata=False,
        prefix=(
            "galaxy all_magnitudes_ab measured_multiband_morphology"
            if args.use_morphology else "galaxy all_magnitudes_ab"
        ),
    )
    return config, serialization


def qwen_cache_path(args: argparse.Namespace, product: dict, config) -> Path:
    catalogue = Path(product["metadata"]["catalogue"])
    stat = catalogue.stat()
    provenance = hashlib.sha256(
        f"{catalogue.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()[:12]
    input_tag = "magnitudes_morphology" if args.use_morphology else "magnitudes"
    selection_tag = "all" if args.max_rows is None else f"n{args.max_rows}"
    return (
        Path(args.cache_root).expanduser()
        / "qwen_posttraining_comparison"
        / catalogue.stem
        / f"{selection_tag}_seed{args.seed}_{input_tag}_{provenance}"
        / f"{base.qwen_run_tag(config)}.pt"
    )


def stage_prepare(args: argparse.Namespace) -> int:
    product, preparation, redshift_bounds = build_product(args)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "comparison": COMPARISON_NAME,
        "catalogue": preparation["catalogue"],
        "use_morphology": preparation["use_morphology"],
        "input_representation": product["metadata"]["input_representation"],
        "image_cutouts_read": False,
        "image_tokens_read": False,
        "pooling": "last",
        "max_rows": args.max_rows,
        "sample_mode": "random",
        "sample_seed": args.seed,
        "redshift_bounds": list(redshift_bounds),
        "n_features": preparation["n_features"],
        "n_magnitude_features": preparation["n_magnitude_features"],
        "n_morphology_features": preparation["n_morphology_features"],
        "rows": preparation["split_counts"],
    }
    (output_dir / "prepared.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def stage_frozen(args: argparse.Namespace) -> int:
    product, _, redshift_bounds = build_product(args)
    qwen_config, serialization = qwen_settings(args)
    cache_path = qwen_cache_path(args, product, qwen_config)
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
    redshift_edges, redshift_centers = make_redshift_grid(
        redshift_bounds[0], redshift_bounds[1], args.n_z_bins
    )
    set_random_seed(args.seed)
    result = train_single_baseline(
        qwen_product,
        "aion",
        output_dir=output_dir,
        n_z_bins=args.n_z_bins,
        redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
        epochs=args.frozen_epochs,
        learning_rate=args.head_learning_rate,
        weight_decay=1.0e-4,
        train_batch_size=args.frozen_train_batch_size,
        eval_batch_size=max(args.eval_batch_size, 64),
        device="cuda",
    )
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
    product, _, redshift_bounds = build_product(args)
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
        learning_rate=args.head_learning_rate,
        lora_learning_rate=args.qlora_learning_rate,
        lora_max_grad_norm=args.lora_max_grad_norm,
        head_warmup_epochs=args.head_warmup_epochs,
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
        checkpoint_dir=args.qlora_checkpoint_dir,
        checkpoint_interval=args.qlora_checkpoint_steps,
        resume=args.qlora_resume,
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
    prepared = json.loads((output_dir / "prepared.json").read_text())
    summary = {
        "comparison": COMPARISON_NAME,
        "catalogue": prepared["catalogue"],
        "use_morphology": prepared["use_morphology"],
        "input_representation": prepared["input_representation"],
        "image_cutouts_read": False,
        "image_tokens_read": False,
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
