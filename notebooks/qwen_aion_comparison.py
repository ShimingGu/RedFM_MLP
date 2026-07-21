#!/usr/bin/env python3
"""Compare grizy-only frozen AION and Qwen representations."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REDFM_MLP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REDFM_MLP_ROOT.parent
if str(REDFM_MLP_ROOT) not in sys.path:
    sys.path.insert(0, str(REDFM_MLP_ROOT))

import aion_magnitude as am
from aion_magnitude.FM_Qwen import (
    QWEN_POOLING_MODES,
    QwenEmbeddingConfig,
    QwenSerializationConfig,
    extract_qwen_embeddings_to_memory,
    load_frozen_qwen,
    qwen_embedding_metadata,
    require_transformers,
    resolve_qwen_model_path,
)

DEFAULT_CATALOGUE = Path("data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits")
DEFAULT_OUTPUT_DIR = Path("/arc/home/gsm/aion_output/figures")
DEFAULT_CACHE_ROOT = Path("/scratch/.tmp-gsm/aion_output/cache")
DEFAULT_QWEN_MODEL = "Qwen3-8B-Base"


def parse_max_rows(value: str) -> int | None:
    if value.lower() in {"none", "all", "full"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--max-rows must be positive, or 'none'.")
    return parsed


def resolve_existing_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, REDFM_MLP_ROOT / path, WORKSPACE_ROOT / path):
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare grizy-only frozen AION and Qwen embeddings with the same photo-z head."
    )
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--max-rows", type=parse_max_rows, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--aion-embedding-batch-size", type=int, default=512)
    parser.add_argument("--qwen-embedding-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    parser.add_argument("--n-z-bins", type=int, default=300)
    parser.add_argument("--z-max", type=float, default=6.0)
    parser.add_argument("--tomographic-samples", type=int, default=100)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-cache-path", type=Path)
    parser.add_argument("--qwen-max-length", type=int, default=256)
    parser.add_argument("--qwen-pooling", choices=QWEN_POOLING_MODES, default="last")
    parser.add_argument("--qwen-normalize", action="store_true")
    parser.add_argument("--no-qwen-4bit", action="store_true")
    parser.add_argument("--allow-qwen-download", action="store_true")
    return parser


def save_figure(fig, path: Path, *, dpi: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"saved {path}")


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def build_aion_config(
    args: argparse.Namespace,
    catalogue_path: Path,
    cache_root: Path,
) -> am.AIONMagnitudeConfig:
    return am.AIONMagnitudeConfig(
        catalogue_path=catalogue_path,
        max_rows=args.max_rows,
        cache_root=cache_root,
        force_recompute_embeddings=args.force_recompute_embeddings,
        z_min=0.0,
        z_max=args.z_max,
        n_z_bins=args.n_z_bins,
        split_strategy="random",
        train_fraction=0.20,
        test_fraction=0.75,
        val_fraction=0.05,
        baseline_epochs=args.epochs,
        baseline_train_batch_size=args.train_batch_size,
        baseline_eval_batch_size=args.eval_batch_size,
        aion_embedding_batch_size=args.aion_embedding_batch_size,
        hsc_mag_faint_limits={"g": 24.5, "r": 24.5, "i": 24.0, "z": 24.5, "y": 24.5},
        extra_bands=(),
        extra_band_invalid_fill="median",
        extra_band_include_valid_flags=False,
        use_aion_embedding=True,
        use_mlp_features=False,
        include_grizy_in_mlp=False,
        aion_input_bands=("g", "r", "i", "z", "y"),
        aion_mag_adjustment_path=None,
        model_kinds=("aion",),
        device_choice=args.device,
    ).normalized()


def qwen_settings(args: argparse.Namespace) -> tuple[QwenEmbeddingConfig, QwenSerializationConfig]:
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
    serialization = QwenSerializationConfig(
        schema_name="clauds_grizy_v1",
        include_hsc_grizy=True,
        include_object_metadata=False,
        hsc_bands=("g", "r", "i", "z", "y"),
    )
    return config, serialization


def qwen_run_tag(config: QwenEmbeddingConfig) -> str:
    model_tag = Path(str(resolve_qwen_model_path(config.model_path))).name.replace("-", "_")
    quant_tag = "4bit" if config.load_in_4bit else "full_precision"
    norm_tag = "normalized" if config.normalize else "raw"
    return f"{model_tag}_{config.pooling}_len{config.max_length}_{quant_tag}_{norm_tag}"


def resolve_qwen_paths(
    args: argparse.Namespace,
    cache_root: Path,
    aion_paths: dict[str, Any],
    config: QwenEmbeddingConfig,
) -> tuple[Path, Path]:
    tag = qwen_run_tag(config)
    cache_path = (
        Path(args.qwen_cache_path).expanduser()
        if args.qwen_cache_path is not None
        else cache_root / "qwen_embeddings" / f"{aion_paths['run_tag']}_{tag}.pt"
    )
    baseline_dir = cache_root / "qwen_aion_comparison" / str(aion_paths["run_tag"]) / tag
    return cache_path, baseline_dir


def preflight_qwen_source(
    config: QwenEmbeddingConfig,
    cache_path: Path,
    *,
    force_recompute: bool,
) -> None:
    if cache_path.exists() and not force_recompute:
        return
    require_transformers()
    resolved = resolve_qwen_model_path(config.model_path)
    model_path = Path(str(resolved)).expanduser()
    if config.local_files_only and model_path.is_absolute() and not model_path.exists():
        raise FileNotFoundError(
            f"Qwen checkpoint not found: {model_path}. "
            "Set QWEN_MODEL to an existing local checkpoint, or pass --qwen-model "
            "with --allow-qwen-download."
        )
    if config.load_in_4bit and am.select_torch_device(str(config.device or "auto")).type != "cuda":
        raise RuntimeError("The default 4-bit Qwen run requires CUDA; pass --no-qwen-4bit for CPU.")


def validate_qwen_product(
    product: dict[str, Any],
    aion_product: dict[str, Any],
    expected_metadata: dict[str, Any],
    cache_path: Path,
) -> None:
    product_ids = [str(item) for item in product.get("object_id", [])]
    aion_ids = [str(item) for item in aion_product["object_id"]]
    if product_ids != aion_ids:
        raise RuntimeError(
            f"Qwen cache row order does not match AION: {cache_path}. "
            "Rebuild with --force-recompute-embeddings."
        )
    embedding = torch.as_tensor(product.get("aion_embedding"))
    if embedding.ndim != 2 or embedding.shape[0] != len(aion_ids) or embedding.shape[1] == 0:
        raise RuntimeError(f"Qwen cache has an invalid embedding tensor: {cache_path}.")
    metadata = product.get("metadata", {})
    keys = (
        "qwen_model_path",
        "qwen_load_in_4bit",
        "qwen_max_length",
        "qwen_pooling",
        "qwen_normalize",
        "qwen_serialization_schema",
        "qwen_serialization_include_hsc_grizy",
        "qwen_serialization_include_object_metadata",
        "qwen_serialization_hsc_bands",
    )
    mismatches = [
        key for key in keys
        if metadata.get(key) != expected_metadata.get(key)
    ]
    if mismatches:
        raise RuntimeError(
            f"Qwen cache settings differ for {mismatches}: {cache_path}. "
            "Use the matching settings or rebuild with --force-recompute-embeddings."
        )


def build_qwen_product(
    args: argparse.Namespace,
    aion_config: am.AIONMagnitudeConfig,
    aion_paths: dict[str, Any],
    aion_product: dict[str, Any],
    qwen_config: QwenEmbeddingConfig,
    serialization: QwenSerializationConfig,
    cache_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    expected_metadata = qwen_embedding_metadata(qwen_config, serialization)
    if cache_path.exists() and not args.force_recompute_embeddings:
        product = am.load_cached_product(cache_path)
        validate_qwen_product(product, aion_product, expected_metadata, cache_path)
        product["split_labels"] = list(aion_product["split_labels"])
        return product

    raw_dataset, feature_names, raw_metadata = am.build_raw_clauds_photoz_dataset(
        aion_config.catalogue_path,
        aion_paths["split_output_dir"],
        split_chunk_size=aion_config.split_chunk_size,
        overwrite_split_cache=aion_config.overwrite_split_cache,
        max_rows=aion_config.max_rows,
        sample_mode=aion_config.sample_mode,
        sample_row_start=aion_config.sample_row_start,
        sample_row_stop=aion_config.sample_row_stop,
        sample_seed=aion_config.sample_seed,
        sample_require_valid_bands=aion_config.sample_require_valid_bands,
        field_column=aion_config.field_column,
        target_redshift_column=aion_config.target_redshift_column,
        z_min=aion_config.z_min,
        z_max=aion_config.z_max,
        redshift_include_min=aion_config.redshift_include_min,
        redshift_include_max=aion_config.redshift_include_max,
        n_z_bins=aion_config.n_z_bins,
        mag_zero_point=aion_config.mag_zero_point,
        hsc_mag_faint_limits=aion_config.hsc_mag_faint_limits,
        extra_bands=(),
        extra_band_invalid_fill="median",
        extra_band_include_valid_flags=False,
        use_mlp_features=False,
        include_grizy_in_mlp=False,
        use_aion_embedding=True,
        aion_mag_adjustment_path=None,
    )
    if feature_names:
        raise RuntimeError(f"Expected grizy-only Qwen input, found extra features: {feature_names}.")
    if [str(item) for item in raw_dataset.object_ids] != [
        str(item) for item in aion_product["object_id"]
    ]:
        raise RuntimeError("Rebuilt Qwen rows do not match the AION cached rows.")

    model, tokenizer = load_frozen_qwen(
        qwen_config.model_path,
        device=device,
        load_in_4bit=qwen_config.load_in_4bit,
        torch_dtype=qwen_config.torch_dtype,
        local_files_only=qwen_config.local_files_only,
        trust_remote_code=qwen_config.trust_remote_code,
    )
    try:
        embeddings = extract_qwen_embeddings_to_memory(
            raw_dataset,
            model,
            tokenizer,
            feature_names=(),
            serialization=serialization,
            batch_size=args.qwen_embedding_batch_size,
            device=device,
            max_length=qwen_config.max_length,
            pooling=qwen_config.pooling,
            normalize=qwen_config.normalize,
        )
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = {
        **raw_metadata,
        **expected_metadata,
        "embedding_backend": "qwen",
        "input_feature_names": [f"{band}_mag" for band in ("g", "r", "i", "z", "y")],
        "cached_embedding_field": "aion_embedding",
        "embedding_batch_size": args.qwen_embedding_batch_size,
    }
    am.save_cached_product(
        cache_path,
        raw_dataset,
        embeddings,
        feature_names=(),
        split_labels=aion_product["split_labels"],
        metadata=metadata,
    )
    product = am.load_cached_product(cache_path)
    validate_qwen_product(product, aion_product, expected_metadata, cache_path)
    return product


def train_and_evaluate(
    product: dict[str, Any],
    *,
    model_kind: str,
    output_dir: Path,
    args: argparse.Namespace,
    config: am.AIONMagnitudeConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    edges, centers = am.make_redshift_grid(0.0, args.z_max, args.n_z_bins)
    am.set_random_seed(config.seed)
    trained = am.train_single_baseline(
        product,
        model_kind,
        output_dir=output_dir,
        n_z_bins=args.n_z_bins,
        redshift_edges=edges,
        redshift_centers=centers,
        epochs=args.epochs,
        learning_rate=config.baseline_learning_rate,
        weight_decay=config.baseline_weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        device=device,
    )
    evaluated = am.load_and_evaluate_baseline(
        product,
        model_kind=model_kind,
        split="test",
        checkpoint_path=trained["checkpoint_path"],
        config=config,
        batch_size=args.eval_batch_size,
        device=device,
    )
    return trained, evaluated


def save_comparison(
    aion_train: dict[str, Any],
    qwen_train: dict[str, Any],
    aion_eval: dict[str, Any],
    qwen_eval: dict[str, Any],
    output_dir: Path,
    tomographic_samples: int,
) -> None:
    labels = ("grizy-aion", "grizy-qwen")
    prefix = output_dir / "qwen_aion_comparison"

    fig, _, _ = am.compare_config_loss(aion_train, qwen_train, labels=labels)
    save_figure(fig, Path(f"{prefix}_loss.jpeg"))

    fig, _ = am.compare_zpred_vs_zphot(
        aion_eval,
        qwen_eval,
        labels=labels,
        pred_key="z_p50",
        pmax=5.0,
        show_metrics=True,
    )
    save_figure(fig, Path(f"{prefix}_scatter.jpeg"))

    fig, _ = am.compare_pit_histogram(aion_eval, qwen_eval, labels=labels)
    save_figure(fig, Path(f"{prefix}_pit.jpeg"))

    fig, _, _ = am.compare_redshift_probability_distribution(
        aion_eval,
        qwen_eval,
        labels=labels,
        gaussian_sigma_bins=1.0,
    )
    save_figure(fig, Path(f"{prefix}_nz.jpeg"))

    fig, _, _ = am.compare_nz_lensing_alike(
        aion_eval,
        qwen_eval,
        labels=labels,
        zphot_bin=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        gaussian_sigma_bins=1.0,
        inferred_bin_key="z_p50",
        n_samples_per_object=tomographic_samples,
    )
    save_figure(fig, Path(f"{prefix}_nztomo.jpeg"))

    metrics = {
        labels[0]: am.summarize_pdf_metrics(aion_eval),
        labels[1]: am.summarize_pdf_metrics(qwen_eval),
    }
    metrics_path = output_dir / "qwen_aion_comparison_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(jsonable(metrics), indent=2, sort_keys=True) + "\n")
    print(f"saved {metrics_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalogue_path = resolve_existing_path(args.catalogue)
    output_dir = Path(args.output_dir).expanduser()
    cache_root = Path(args.cache_root).expanduser()

    if not catalogue_path.exists():
        raise FileNotFoundError(f"Catalogue not found: {catalogue_path}")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.qwen_embedding_batch_size <= 0:
        raise ValueError("--qwen-embedding-batch-size must be positive.")

    device = am.select_torch_device(args.device)
    aion_config = build_aion_config(args, catalogue_path, cache_root)
    aion_paths = am.resolve_training_paths(aion_config)
    qwen_config, serialization = qwen_settings(args)
    qwen_cache, qwen_baseline_dir = resolve_qwen_paths(
        args, cache_root, aion_paths, qwen_config
    )
    preflight_qwen_source(
        qwen_config,
        qwen_cache,
        force_recompute=args.force_recompute_embeddings,
    )

    print(f"device: {device}")
    print(f"catalogue: {catalogue_path}")
    print(f"AION cache: {aion_paths['cache_path']}")
    print(f"Qwen cache: {qwen_cache}")
    print(f"Qwen model: {resolve_qwen_model_path(qwen_config.model_path)}")
    print("inputs: grizy only for both frozen encoders")

    aion_product = am.build_and_cache_aion_embeddings_from_config(aion_config)
    qwen_product = build_qwen_product(
        args,
        aion_config,
        aion_paths,
        aion_product,
        qwen_config,
        serialization,
        qwen_cache,
        device,
    )

    comparison_root = (
        cache_root / "qwen_aion_comparison" / str(aion_paths["run_tag"])
    )
    aion_train, aion_evaluated = train_and_evaluate(
        aion_product,
        model_kind="aion",
        output_dir=comparison_root / "aion",
        args=args,
        config=aion_config,
        device=device,
    )
    qwen_train, qwen_evaluated = train_and_evaluate(
        qwen_product,
        model_kind="qwen",
        output_dir=qwen_baseline_dir,
        args=args,
        config=aion_config,
        device=device,
    )
    save_comparison(
        aion_train,
        qwen_train,
        aion_evaluated["evaluation"],
        qwen_evaluated["evaluation"],
        output_dir,
        args.tomographic_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
