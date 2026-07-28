#!/usr/bin/env python3
"""Compare frozen Qwen and an MLP on matched multiband catalogue morphology."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import qwen_mlp_full_comparison as qwen_base
from aion_magnitude import table_models as tm
from aion_magnitude.dataset import make_random_split
from aion_magnitude.FM_Qwen import (
    QWEN_POOLING_MODES,
    QwenEmbeddingConfig,
    load_frozen_qwen,
)
from aion_magnitude.FM_Qwen3 import (
    Qwen3SerializationConfig,
    qwen3_embedding_metadata,
    serialize_qwen3_batch,
)
from aion_magnitude.morphology import (
    FEATURE_SCALING_MODES,
    save_morphology_comparison_artifacts,
    scale_product_features_from_training_split,
)
from aion_magnitude.training import train_single_baseline
from aion_magnitude.utils import make_redshift_grid, set_random_seed


COMPARISON_NAME = "qwen_mlp_full_image_comparison"
DEFAULT_CATALOGUE = Path(
    "/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/"
    "COSMOS-HSCpipe-Phosphoros_morphological_multiband.fits"
)
DEFAULT_OUTPUT_DIR = Path(
    "/arc/home/gsm/aion_output/figures/qwen-mlp_full_image_comparison"
)
DEFAULT_CACHE_ROOT = Path("/scratch/.tmp-gsm/aion_output/cache")
DEFAULT_QWEN_MODEL = "Qwen3-8B-Base"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--max-rows", type=qwen_base.parse_max_rows, default=200_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--qwen-embedding-batch-size", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--train-fraction", type=float, default=0.63)
    parser.add_argument("--test-fraction", type=float, default=0.32)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--n-z-bins", type=int, default=300)
    parser.add_argument("--tomographic-samples", type=int, default=100)
    parser.add_argument(
        "--feature-scaling", choices=FEATURE_SCALING_MODES, default="minmax"
    )
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-cache-path", type=Path)
    parser.add_argument("--qwen-max-length", type=int, default=2048)
    parser.add_argument("--qwen-pooling", choices=QWEN_POOLING_MODES, default="last")
    parser.add_argument("--qwen-normalize", action="store_true")
    parser.add_argument("--no-qwen-4bit", action="store_true")
    parser.add_argument("--allow-qwen-download", action="store_true")
    parser.add_argument("--force-recompute-qwen", action="store_true")
    parser.add_argument("--no-qwen-physical-context", action="store_true")
    parser.add_argument("--allow-qwen-truncation", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if min(
        args.qwen_embedding_batch_size,
        args.train_batch_size,
        args.eval_batch_size,
        args.n_z_bins,
    ) <= 0:
        raise ValueError("Batch sizes and --n-z-bins must be positive.")
    if args.qwen_max_length <= 0:
        raise ValueError("--qwen-max-length must be positive.")
    if not math.isclose(
        args.train_fraction + args.test_fraction + args.val_fraction, 1.0
    ):
        raise ValueError("Train, test, and validation fractions must sum to one.")


def qwen_settings(
    args: argparse.Namespace,
) -> tuple[QwenEmbeddingConfig, Qwen3SerializationConfig]:
    config = QwenEmbeddingConfig(
        model_path=args.qwen_model,
        device=args.device,
        load_in_4bit=not args.no_qwen_4bit,
        torch_dtype="auto",
        max_length=args.qwen_max_length,
        pooling=args.qwen_pooling,
        normalize=args.qwen_normalize,
        local_files_only=not args.allow_qwen_download,
        trust_remote_code=True,
    )
    serialization = Qwen3SerializationConfig(
        schema_name="clauds_physical_magnitudes_multiband_morphology_v1",
        include_physical_context=not args.no_qwen_physical_context,
        include_image_context=False,
        include_unrecognized_features=True,
        prefix="Galaxy magnitudes and measured multiband morphology",
        final_marker="Combined galaxy representation:",
    )
    return config, serialization


def expected_qwen_metadata(
    config: QwenEmbeddingConfig,
    serialization: Qwen3SerializationConfig,
    feature_names: list[str],
) -> dict[str, Any]:
    return {
        **qwen3_embedding_metadata(config, serialization),
        "input_feature_names": feature_names,
        "input_scope": "available magnitudes plus 42 six-band morphology fields",
        "morphology_bands": list(tm.MORPHOLOGY_BANDS),
        "morphology_availability_rule": "all",
        "aion_image_embedding_used": False,
        "aion_image_tokens_read_by_qwen": False,
        "image_cutouts_read": False,
    }


def build_catalogue_product(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], tuple[float, float]]:
    data = tm.load_catalogue_data(
        args.catalogue,
        max_rows=args.max_rows,
        seed=args.seed,
        include_full121=False,
        include_morphology=True,
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
    raw_features = pd.concat(
        [data.magnitude_features, data.morphology_features], axis=1
    )
    prepared, imputation = tm.impute_from_training(raw_features, split_labels)
    feature_names = [str(name) for name in prepared.columns]
    feature_values = torch.from_numpy(
        prepared.to_numpy(dtype=np.float32, copy=True)
    )
    target = torch.from_numpy(np.asarray(data.target, dtype=np.float32))
    split_counts = {
        name: int(np.count_nonzero(split_labels == name))
        for name in ("train", "val", "test")
    }
    z_min = float(target.min().item())
    z_max = float(target.max().item())
    if z_max <= z_min:
        padding = max(abs(z_min) * 1.0e-6, 1.0e-6)
        z_min -= padding
        z_max += padding
    product = {
        "object_id": data.object_id.tolist(),
        "field": ["COSMOS"] * len(data.object_id),
        "aion_embedding": torch.empty((len(data.object_id), 0), dtype=torch.float32),
        "extra_features": feature_values,
        "feature_names": feature_names,
        "z_spec": target,
        "split_labels": split_labels.tolist(),
        "redshift_reference": {"zphot": target.clone()},
        "metadata": {
            "catalogue": str(Path(args.catalogue).expanduser().resolve()),
            "n_rows": len(data.object_id),
            "split_counts": split_counts,
            "input_representation": "catalogue_multiband_morphology",
            "morphology_bands": list(tm.MORPHOLOGY_BANDS),
            "morphology_feature_columns": list(tm.MORPHOLOGY_FEATURE_COLUMNS),
            "morphology_availability_columns": list(
                tm.MORPHOLOGY_AVAILABILITY_COLUMNS
            ),
            "morphology_availability_rule": "all",
            "image_cutouts_read": False,
            "image_tokens_read": False,
            "imputation": imputation,
        },
    }
    if "image_token_ids_path" in product or "image_token_row_indices" in product:
        raise RuntimeError("Catalogue product unexpectedly contains image-token inputs.")
    preparation = {
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


def prompt_length_preflight(
    args: argparse.Namespace,
    product: dict[str, Any],
    tokenizer: Any,
    config: QwenEmbeddingConfig,
    serialization: Qwen3SerializationConfig,
) -> dict[str, Any]:
    features = torch.as_tensor(product["extra_features"], dtype=torch.float32)
    feature_names = [str(name) for name in product["feature_names"]]
    sample_count = min(256, len(features))
    sample_rows = np.linspace(0, len(features) - 1, sample_count, dtype=np.int64)
    texts = serialize_qwen3_batch(
        features[sample_rows], feature_names, config=serialization
    )
    lengths = np.asarray(
        [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
    )
    stats = {
        "sample_count": int(sample_count),
        "minimum": int(lengths.min()),
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "maximum": int(lengths.max()),
        "max_length": int(config.max_length),
        "sampled_rows_exceeding_max_length": int((lengths > config.max_length).sum()),
    }
    print(
        "Qwen prompt token lengths: "
        f"min={stats['minimum']} median={stats['median']:.1f} "
        f"p95={stats['p95']:.1f} max={stats['maximum']} "
        f"limit={stats['max_length']}",
        flush=True,
    )
    if stats["sampled_rows_exceeding_max_length"] and not args.allow_qwen_truncation:
        raise RuntimeError(
            "Qwen multiband-morphology prompts exceed --qwen-max-length. "
            "Increase the limit or explicitly pass --allow-qwen-truncation."
        )
    return stats


def extract_or_load_qwen_embeddings(
    args: argparse.Namespace,
    product: dict[str, Any],
    config: QwenEmbeddingConfig,
    serialization: Qwen3SerializationConfig,
    cache_path: Path,
    device: torch.device,
) -> torch.Tensor:
    feature_names = [str(name) for name in product["feature_names"]]
    features = torch.as_tensor(product["extra_features"], dtype=torch.float32)
    expected = expected_qwen_metadata(config, serialization, feature_names)
    if cache_path.exists() and not args.force_recompute_qwen:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        return qwen_base.validate_qwen_cache(cached, product, expected, cache_path)

    model, tokenizer = load_frozen_qwen(
        config.model_path,
        device=device,
        load_in_4bit=config.load_in_4bit,
        torch_dtype=config.torch_dtype,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )
    prompt_stats = prompt_length_preflight(
        args, product, tokenizer, config, serialization
    )
    parts: list[torch.Tensor] = []
    try:
        for start in range(0, len(features), args.qwen_embedding_batch_size):
            stop = min(start + args.qwen_embedding_batch_size, len(features))
            texts = serialize_qwen3_batch(
                features[start:stop], feature_names, config=serialization
            )
            parts.append(
                qwen_base.extract_qwen_embeddings_from_texts(
                    texts,
                    model,
                    tokenizer,
                    device=device,
                    max_length=config.max_length,
                    batch_size=args.qwen_embedding_batch_size,
                    pooling=config.pooling,
                    normalize=config.normalize,
                )
            )
            if stop == len(features) or stop % max(
                1000, args.qwen_embedding_batch_size
            ) == 0:
                print(f"Qwen embeddings: {stop:,}/{len(features):,}", flush=True)
        if not parts:
            raise ValueError("Qwen extraction received zero catalogue rows.")
        embeddings = torch.cat(parts, dim=0)
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "object_id": list(product["object_id"]),
            "embedding": embeddings,
            "metadata": {
                **expected,
                "prompt_token_length_preflight": prompt_stats,
            },
        },
        cache_path,
    )
    print(f"saved {cache_path}")
    return embeddings


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    preparation: dict[str, Any],
    redshift_bounds: tuple[float, float],
    qwen_cache_path: Path,
    artifacts: dict[str, str],
    status: str,
) -> None:
    payload = {
        "comparison": COMPARISON_NAME,
        "status": status,
        "catalogue": str(Path(args.catalogue).expanduser().resolve()),
        "output_dir": str(Path(args.output_dir).expanduser()),
        "cache_root": str(Path(args.cache_root).expanduser()),
        "qwen_cache": str(qwen_cache_path),
        "image_cutouts_read": False,
        "image_tokens_read": False,
        "morphology_bands": list(tm.MORPHOLOGY_BANDS),
        "morphology_availability_rule": "all",
        "redshift_bounds": list(redshift_bounds),
        "feature_scaling": args.feature_scaling,
        "feature_scaling_fit_split": "train",
        "preparation": preparation,
        "artifacts": artifacts,
        "arguments": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    args.catalogue = qwen_base.resolve_existing_path(args.catalogue)
    args.output_dir = Path(args.output_dir).expanduser()
    args.cache_root = Path(args.cache_root).expanduser()
    if not args.catalogue.exists():
        raise FileNotFoundError(f"Catalogue not found: {args.catalogue}")

    product, preparation, redshift_bounds = build_catalogue_product(args)
    qwen_config, serialization = qwen_settings(args)
    selection_tag = "all" if args.max_rows is None else f"n{args.max_rows}"
    qwen_cache_path = (
        Path(args.qwen_cache_path).expanduser()
        if args.qwen_cache_path is not None
        else args.cache_root
        / COMPARISON_NAME
        / "catalogue_multiband_complete6"
        / f"{selection_tag}_{qwen_base.qwen_run_tag(qwen_config)}.pt"
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "qwen_mlp_full_run.json"
    report_path = output_dir / f"{COMPARISON_NAME}_out.log"
    report_path.write_text(
        "Catalogue multiband morphology comparison\n"
        f"rows: {preparation['n_rows']:,}\n"
        f"split counts: {preparation['split_counts']}\n"
        f"features: {preparation['n_features']} "
        f"({preparation['n_magnitude_features']} magnitudes + "
        f"{preparation['n_morphology_features']} morphology)\n"
        "required morphology bands: u,g,r,i,z,y (all six)\n"
        "image cutouts read: no\n"
        "image tokens read: no\n"
    )
    print(f"catalogue: {args.catalogue}")
    print(f"rows: {preparation['n_rows']:,}")
    print(
        f"features: {preparation['n_magnitude_features']} magnitudes + "
        f"{preparation['n_morphology_features']} morphology"
    )
    print("image cutouts/tokens: disabled")
    print(f"output directory: {output_dir}")

    if args.prepare_only:
        write_manifest(
            manifest_path,
            args=args,
            preparation=preparation,
            redshift_bounds=redshift_bounds,
            qwen_cache_path=qwen_cache_path,
            artifacts={"report": str(report_path)},
            status="prepared",
        )
        print("Preparation complete; Qwen extraction and training skipped.")
        return 0

    qwen_base.preflight_qwen_source(
        qwen_config,
        qwen_cache_path,
        force_recompute=args.force_recompute_qwen,
    )
    device = qwen_base.am.select_torch_device(args.device)
    qwen_embeddings = extract_or_load_qwen_embeddings(
        args,
        product,
        qwen_config,
        serialization,
        qwen_cache_path,
        device,
    )
    qwen_product = dict(product)
    qwen_product["aion_embedding"] = qwen_embeddings
    qwen_product["metadata"] = {
        **dict(product["metadata"]),
        **expected_qwen_metadata(qwen_config, serialization, product["feature_names"]),
    }
    qwen_product = scale_product_features_from_training_split(
        qwen_product, feature_key="aion_embedding", mode=args.feature_scaling
    )
    mlp_product = scale_product_features_from_training_split(
        product, feature_key="extra_features", mode=args.feature_scaling
    )
    redshift_edges, redshift_centers = make_redshift_grid(
        redshift_bounds[0], redshift_bounds[1], args.n_z_bins
    )
    set_random_seed(args.seed)
    qwen_result = train_single_baseline(
        qwen_product,
        "qwen",
        output_dir=output_dir,
        n_z_bins=args.n_z_bins,
        redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        device=device,
    )
    set_random_seed(args.seed)
    mlp_result = train_single_baseline(
        mlp_product,
        "tabular",
        output_dir=output_dir,
        n_z_bins=args.n_z_bins,
        redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        device=device,
    )
    results = {"qwen": qwen_result, "tabular": mlp_result}
    prefix = output_dir / COMPARISON_NAME
    artifacts = save_morphology_comparison_artifacts(
        results,
        model_kinds=("qwen", "tabular"),
        output_dir=output_dir,
        tomographic_samples=args.tomographic_samples,
        comparison_labels=(
            "physical-magnitude+six-band-morphology-Qwen",
            "magnitude+six-band-morphology-MLP",
        ),
        comparison_prefix=prefix,
    )
    artifacts["report"] = str(report_path)
    summary_path = output_dir / "qwen_mlp_full_results.pt"
    torch.save(results, summary_path)
    artifacts["summary"] = str(summary_path)
    write_manifest(
        manifest_path,
        args=args,
        preparation=preparation,
        redshift_bounds=redshift_bounds,
        qwen_cache_path=qwen_cache_path,
        artifacts=artifacts,
        status="complete",
    )
    print(f"summary: {summary_path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
