#!/usr/bin/env python3
"""Compare full-magnitude Qwen and MLP representations with AION u-image tokens."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REDFM_MLP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REDFM_MLP_ROOT.parent
if str(REDFM_MLP_ROOT) not in sys.path:
    sys.path.insert(0, str(REDFM_MLP_ROOT))

import aion_magnitude as am
from aion_magnitude.clauds_bands import HSC_AION_BANDS, REDSHIFT_COLUMNS
from aion_magnitude.dataset import load_clauds_catalogue_from_fits
from aion_magnitude.FM_Qwen import (
    QWEN_POOLING_MODES,
    QwenEmbeddingConfig,
    QwenSerializationConfig,
    extract_qwen_embeddings_from_texts,
    load_frozen_qwen,
    qwen_embedding_metadata,
    require_transformers,
    resolve_qwen_model_path,
    serialize_qwen_feature_row,
)
from aion_magnitude.morphology import (
    AIONMorphologyConfig,
    FEATURE_SCALING_MODES,
    cache_aion_morphology_tokens,
    format_morphology_population_report,
    resolve_morphology_paths,
    save_morphology_comparison_artifacts,
    train_single_morphology_model,
)

DEFAULT_CATALOGUE = Path("data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits")
DEFAULT_MORPHOLOGY_DIR = Path("data/clauds/images/tilesv5/")
DEFAULT_OUTPUT_DIR = Path("/arc/home/gsm/aion_output/figures/qwen-mlp_full_comparison")
DEFAULT_CACHE_ROOT = Path("/scratch/.tmp-gsm/aion_output/cache")
DEFAULT_QWEN_MODEL = "Qwen3-8B-Base"
COMPARISON_NAME = "qwen_mlp_full_comparison"


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--morphology-dir", type=Path, default=DEFAULT_MORPHOLOGY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--max-rows", type=parse_max_rows, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--token-batch-size", type=int, default=64)
    parser.add_argument("--qwen-embedding-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--force-rebuild-tokens", action="store_true")
    parser.add_argument("--force-rebuild-photometry", action="store_true")
    parser.add_argument("--force-recompute-qwen", action="store_true")
    parser.add_argument("--n-z-bins", type=int, default=300)
    parser.add_argument("--tomographic-samples", type=int, default=100)
    parser.add_argument("--image-flux-scale", type=float, default=1.0)
    parser.add_argument("--min-cutout-weight-coverage", type=float, default=0.90)
    parser.add_argument("--feature-scaling", choices=FEATURE_SCALING_MODES, default="none")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-cache-path", type=Path)
    parser.add_argument("--qwen-max-length", type=int, default=256)
    parser.add_argument("--qwen-pooling", choices=QWEN_POOLING_MODES, default="last")
    parser.add_argument("--qwen-normalize", action="store_true")
    parser.add_argument("--no-qwen-4bit", action="store_true")
    parser.add_argument("--allow-qwen-download", action="store_true")
    return parser


def qwen_settings(
    args: argparse.Namespace,
) -> tuple[QwenEmbeddingConfig, QwenSerializationConfig]:
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
        schema_name="clauds_all_magnitude_v1",
        include_hsc_grizy=False,
        include_object_metadata=False,
        prefix="galaxy all_magnitudes_ab",
    )
    return config, serialization


def qwen_run_tag(config: QwenEmbeddingConfig) -> str:
    model_tag = Path(str(resolve_qwen_model_path(config.model_path))).name.replace("-", "_")
    quant_tag = "4bit" if config.load_in_4bit else "full_precision"
    norm_tag = "normalized" if config.normalize else "raw"
    return f"{model_tag}_{config.pooling}_len{config.max_length}_{quant_tag}_{norm_tag}"


def finite_catalogue_redshift_bounds(
    catalogue_path: Path,
    split_output_dir: Path,
    *,
    max_rows: int | None,
) -> tuple[float, float]:
    table = load_clauds_catalogue_from_fits(
        catalogue_path,
        split_output_dir,
        max_rows=max_rows,
        sample_mode="random",
        sample_seed=42,
    )
    values = np.asarray(table[REDSHIFT_COLUMNS["zphot"]], dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("The selected catalogue contains no finite photometric-redshift targets.")
    z_min = float(finite.min())
    z_max = float(finite.max())
    if z_max <= z_min:
        padding = max(abs(z_min) * 1e-6, 1e-6)
        z_min -= padding
        z_max += padding
    return z_min, z_max


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
            f"Qwen checkpoint not found: {model_path}. Set QWEN_MODEL to an existing "
            "local checkpoint, or enable downloads explicitly."
        )
    if config.load_in_4bit and am.select_torch_device(str(config.device or "auto")).type != "cuda":
        raise RuntimeError("The default 4-bit Qwen run requires CUDA; pass --no-qwen-4bit for CPU.")


def expected_qwen_metadata(
    qwen_config: QwenEmbeddingConfig,
    serialization: QwenSerializationConfig,
    feature_names: list[str],
) -> dict[str, Any]:
    return {
        **qwen_embedding_metadata(qwen_config, serialization),
        "input_feature_names": feature_names,
        "input_scope": "all magnitudes",
        "aion_image_embedding_used": False,
    }


def validate_qwen_cache(
    cached: dict[str, Any],
    product: dict[str, Any],
    expected_metadata: dict[str, Any],
    cache_path: Path,
) -> torch.Tensor:
    cached_ids = [str(value) for value in cached.get("object_id", [])]
    product_ids = [str(value) for value in product["object_id"]]
    if cached_ids != product_ids:
        raise RuntimeError(
            f"Qwen cache row order does not match the morphology product: {cache_path}. "
            "Rebuild with --force-recompute-qwen."
        )
    embeddings = torch.as_tensor(cached.get("embedding"), dtype=torch.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(product_ids) or embeddings.shape[1] == 0:
        raise RuntimeError(f"Qwen cache has an invalid embedding tensor: {cache_path}.")
    metadata = dict(cached.get("metadata", {}))
    mismatches = [
        key for key, value in expected_metadata.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"Qwen cache settings differ for {mismatches}: {cache_path}. "
            "Use matching settings or rebuild with --force-recompute-qwen."
        )
    return embeddings


def extract_or_load_qwen_embeddings(
    args: argparse.Namespace,
    product: dict[str, Any],
    qwen_config: QwenEmbeddingConfig,
    serialization: QwenSerializationConfig,
    cache_path: Path,
    device: torch.device,
) -> torch.Tensor:
    feature_names = [str(name) for name in product.get("feature_names", [])]
    features = torch.as_tensor(product["extra_features"], dtype=torch.float32)
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError("Morphology product feature names do not match its magnitude matrix.")
    expected_metadata = expected_qwen_metadata(qwen_config, serialization, feature_names)

    if cache_path.exists() and not args.force_recompute_qwen:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        return validate_qwen_cache(cached, product, expected_metadata, cache_path)

    model, tokenizer = load_frozen_qwen(
        qwen_config.model_path,
        device=device,
        load_in_4bit=qwen_config.load_in_4bit,
        torch_dtype=qwen_config.torch_dtype,
        local_files_only=qwen_config.local_files_only,
        trust_remote_code=qwen_config.trust_remote_code,
    )
    parts: list[torch.Tensor] = []
    try:
        for start in range(0, len(features), args.qwen_embedding_batch_size):
            batch = features[start : start + args.qwen_embedding_batch_size]
            texts = [
                serialize_qwen_feature_row(
                    {
                        feature_names[column]: batch[row, column]
                        for column in range(batch.shape[1])
                    },
                    serialization=serialization,
                )
                for row in range(batch.shape[0])
            ]
            parts.append(
                extract_qwen_embeddings_from_texts(
                    texts,
                    model,
                    tokenizer,
                    device=device,
                    max_length=qwen_config.max_length,
                    batch_size=args.qwen_embedding_batch_size,
                    pooling=qwen_config.pooling,
                    normalize=qwen_config.normalize,
                )
            )
            completed = min(start + len(batch), len(features))
            if completed == len(features) or completed % max(1000, args.qwen_embedding_batch_size) == 0:
                print(f"Qwen embeddings: {completed:,}/{len(features):,}")
        if not parts:
            raise ValueError("Qwen embedding extraction received zero morphology-matched rows.")
        embeddings = torch.cat(parts, dim=0)
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "object_id": list(product["object_id"]),
            "embedding": embeddings,
            "metadata": expected_metadata,
        },
        cache_path,
    )
    print(f"saved {cache_path}")
    return embeddings


def save_run_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    morphology_paths: dict[str, Any],
    qwen_cache_path: Path,
    redshift_bounds: tuple[float, float],
    artifacts: dict[str, str],
) -> None:
    manifest = {
        "comparison": COMPARISON_NAME,
        "catalogue": str(resolve_existing_path(args.catalogue)),
        "morphology_dir": str(resolve_existing_path(args.morphology_dir)),
        "output_dir": str(Path(args.output_dir).expanduser()),
        "morphology_product": str(morphology_paths["morphology_product_path"]),
        "qwen_cache": str(qwen_cache_path),
        "feature_scaling": args.feature_scaling,
        "feature_scaling_fit_split": "train",
        "magnitude_faint_end_cuts": None,
        "redshift_range_cut": None,
        "finite_redshift_target_required": True,
        "redshift_bin_bounds_from_selected_catalogue": list(redshift_bounds),
        "artifacts": artifacts,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalogue_path = resolve_existing_path(args.catalogue)
    morphology_dir = resolve_existing_path(args.morphology_dir)
    output_dir = Path(args.output_dir).expanduser()
    cache_root = Path(args.cache_root).expanduser()

    if not catalogue_path.exists():
        raise FileNotFoundError(f"Catalogue not found: {catalogue_path}")
    if not morphology_dir.exists():
        raise FileNotFoundError(f"Morphology directory not found: {morphology_dir}")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.qwen_embedding_batch_size <= 0:
        raise ValueError("--qwen-embedding-batch-size must be positive.")
    if args.train_batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Training and evaluation batch sizes must be positive.")
    if args.qwen_max_length <= 0:
        raise ValueError("--qwen-max-length must be positive.")

    selection_tag = "all" if args.max_rows is None else f"n{args.max_rows}"
    run_cache_dir = cache_root / "qwen_mlp_full_comparison" / selection_tag
    split_output_dir = run_cache_dir / "clauds_split"
    z_min, z_max = finite_catalogue_redshift_bounds(
        catalogue_path,
        split_output_dir,
        max_rows=args.max_rows,
    )
    no_faint_limits = {band: None for band in HSC_AION_BANDS}
    config = AIONMorphologyConfig(
        catalogue_path=catalogue_path,
        morphology_dir=morphology_dir,
        cache_root=cache_root,
        split_output_dir=split_output_dir,
        photometry_cache_path=run_cache_dir / "photometry_no_magnitude_or_redshift_cut.pt",
        output_dir=output_dir,
        max_rows=args.max_rows,
        sample_mode="random",
        sample_seed=42,
        sample_require_valid_bands=(),
        force_rebuild_photometry=args.force_rebuild_photometry,
        force_rebuild_tokens=args.force_rebuild_tokens,
        preserve_photometry_splits=True,
        z_min=z_min,
        z_max=z_max,
        redshift_include_min=True,
        redshift_include_max=True,
        n_z_bins=args.n_z_bins,
        hsc_mag_faint_limits=no_faint_limits,
        extra_band_invalid_fill="median",
        extra_band_include_valid_flags=False,
        include_grizy_in_mlp=True,
        use_aion_magnitude_embedding=False,
        image_flux_scale=args.image_flux_scale,
        min_cutout_weight_coverage=args.min_cutout_weight_coverage,
        token_batch_size=args.token_batch_size,
        model_kinds=("qwen_morphology", "morphology"),
        feature_scaling=args.feature_scaling,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        tomographic_samples=args.tomographic_samples,
        device_choice=args.device,
    ).normalized()
    morphology_paths = resolve_morphology_paths(config)
    qwen_config, serialization = qwen_settings(args)
    qwen_cache_path = (
        Path(args.qwen_cache_path).expanduser()
        if args.qwen_cache_path is not None
        else cache_root
        / "qwen_mlp_full_comparison"
        / str(morphology_paths["morphology_tag"])
        / f"{qwen_run_tag(qwen_config)}.pt"
    )
    preflight_qwen_source(
        qwen_config,
        qwen_cache_path,
        force_recompute=args.force_recompute_qwen,
    )

    device = am.select_torch_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")
    print(f"catalogue: {catalogue_path}")
    print(f"morphology tiles: {morphology_dir}")
    print(f"output directory: {output_dir}")
    print("inputs: all magnitudes plus AION-tokenized CLAUDS u images")
    print("magnitude faint-end cuts: disabled")
    print(f"redshift range cut: disabled; finite target grid is [{z_min:g}, {z_max:g}]")
    print(f"pre-train feature scaling: {config.feature_scaling} (fit on train only)")

    morphology_product = cache_aion_morphology_tokens(config)
    prefix = output_dir / COMPARISON_NAME
    report = dict(morphology_product.get("metadata", {})).get("population_report")
    if report is None:
        raise RuntimeError("Morphology product is missing its galaxy population report.")
    report_path = Path(f"{prefix}_out.log")
    report_text = format_morphology_population_report(report)
    report_text += (
        "\\nSelection and preprocessing\\n"
        "magnitude faint-end cuts: disabled\\n"
        "redshift range cut: disabled (finite target required)\\n"
        f"redshift bin bounds: [{z_min:g}, {z_max:g}]\\n"
        f"feature scaling: {config.feature_scaling} (fit on train only)\\n"
        "AION image-to-redshift embedding: not used\\n"
    )
    report_path.write_text(report_text)
    print(f"saved {report_path}")

    qwen_embeddings = extract_or_load_qwen_embeddings(
        args,
        morphology_product,
        qwen_config,
        serialization,
        qwen_cache_path,
        device,
    )
    qwen_product = dict(morphology_product)
    qwen_product["aion_embedding"] = qwen_embeddings
    qwen_metadata = dict(morphology_product.get("metadata", {}))
    qwen_metadata.update(expected_qwen_metadata(
        qwen_config,
        serialization,
        [str(name) for name in morphology_product.get("feature_names", [])],
    ))
    qwen_product["metadata"] = qwen_metadata

    results = {
        "qwen_morphology": train_single_morphology_model(
            qwen_product,
            "qwen_morphology",
            output_dir=output_dir,
            config=config,
            device=device,
        ),
        "morphology": train_single_morphology_model(
            morphology_product,
            "morphology",
            output_dir=output_dir,
            config=config,
            device=device,
        ),
    }
    artifacts = save_morphology_comparison_artifacts(
        results,
        model_kinds=("qwen_morphology", "morphology"),
        output_dir=output_dir,
        tomographic_samples=args.tomographic_samples,
        comparison_labels=(
            "all-magnitude-Qwen+tokenized-u-image",
            "all-magnitude-MLP+tokenized-u-image",
        ),
        comparison_prefix=prefix,
    )
    summary_path = output_dir / "qwen_mlp_full_results.pt"
    torch.save(results, summary_path)
    manifest_path = output_dir / "qwen_mlp_full_run.json"
    save_run_manifest(
        manifest_path,
        args=args,
        morphology_paths=morphology_paths,
        qwen_cache_path=qwen_cache_path,
        redshift_bounds=(z_min, z_max),
        artifacts=artifacts,
    )
    print(f"summary: {summary_path}")
    print(f"manifest: {manifest_path}")
    for name, path in artifacts.items():
        print(f"comparison {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
