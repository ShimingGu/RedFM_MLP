#!/usr/bin/env python3
"""Train and compare head-only and post-trained AION grizy representations."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aion_magnitude.aion_embedding_methods import (
    AIONEmbeddingMethodConfig,
    AIONTokenRedshiftDataset,
    train_aion_embedding_method,
)
from aion_magnitude.aion_embedding_rlvr import train_embedding_adapter_rlvr
from aion_magnitude.aion_image_embeddings import extract_grizy_image_aion_embeddings
from aion_magnitude.caching import (
    build_and_cache_aion_embeddings_from_config,
    extract_aion_tokens_to_memory,
    save_cached_product,
)
from aion_magnitude.config import AIONMagnitudeConfig
from aion_magnitude.dataset import (
    CLAUDSPhotoZDataset,
    build_raw_clauds_photoz_dataset,
    make_split_labels,
    split_metadata,
)
from aion_magnitude.models import load_frozen_aion
from aion_magnitude.qwen_alternative_posttraining import (
    EmbeddingRedshiftDataset,
    ResidualEmbeddingAdapterConfig,
    train_residual_embedding_adapter,
)
from aion_magnitude.qwen_rlvr import RLVRConfig
from aion_magnitude.training import train_single_baseline
from aion_magnitude.utils import (
    load_cached_product,
    make_redshift_grid,
    resolve_torch_device,
    set_random_seed,
)


INPUT_MODES = ("photometry", "photometry-images")


def parse_max_rows(value: str) -> int | None:
    if value.lower() in {"all", "full", "none"}:
        return None
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--max-rows must be positive or 'all'.")
    return parsed


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "prepare", "head-only", "attentive-head-only", "ia3",
            "embedding-adapter", "dora", "qlora", "rlvr",
        ),
        required=True,
    )
    parser.add_argument("--input-mode", choices=INPUT_MODES, required=True)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/scratch/.tmp-gsm/aion_output/cache"),
    )
    parser.add_argument("--product-cache-path", type=Path)
    parser.add_argument(
        "--hsc-image-dir",
        type=Path,
        default=Path("/arc/projects/ots/pdr3_dud"),
    )
    parser.add_argument(
        "--image-assignment-cache-dir",
        type=Path,
        default=Path(
            "/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/cache/"
            "aion_multiband_morphology_catalogue_updated"
        ),
    )
    parser.add_argument(
        "--fits-backend",
        choices=("auto", "torchfits", "astropy"),
        default="auto",
    )
    parser.add_argument("--min-cutout-coverage", type=float, default=0.90)
    parser.add_argument("--max-rows", type=parse_max_rows, default=300_000)
    parser.add_argument("--sample-mode", choices=("head", "random"), default="head")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.75)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--n-z-bins", type=int, default=300)
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=6.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--image-embedding-batch-size", type=int, default=8)
    parser.add_argument("--force-recompute-product", action="store_true")

    parser.add_argument("--head-only-epochs", type=int, default=10)
    parser.add_argument("--head-only-batch-size", type=int, default=256)
    parser.add_argument("--head-only-eval-batch-size", type=int, default=512)
    parser.add_argument("--head-only-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--head-only-weight-decay", type=float, default=1.0e-4)

    parser.add_argument("--encoder-head-epochs", type=int, default=10)
    parser.add_argument("--encoder-head-batch-size", type=int, default=1)
    parser.add_argument("--encoder-head-eval-batch-size", type=int, default=2)
    parser.add_argument("--encoder-head-gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--encoder-head-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--encoder-head-weight-decay", type=float, default=0.01)

    parser.add_argument("--embedding-adapter-epochs", type=int, default=10)
    parser.add_argument("--embedding-adapter-batch-size", type=int, default=256)
    parser.add_argument("--embedding-adapter-eval-batch-size", type=int, default=512)
    parser.add_argument("--embedding-adapter-bottleneck-dim", type=int, default=256)
    parser.add_argument("--embedding-adapter-learning-rate", type=float, default=2.0e-4)
    parser.add_argument(
        "--embedding-adapter-adapter-learning-rate",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument("--embedding-adapter-weight-decay", type=float, default=0.01)
    parser.add_argument("--embedding-adapter-head-warmup-epochs", type=int, default=3)
    parser.add_argument("--embedding-adapter-max-grad-norm", type=float, default=0.1)
    parser.add_argument("--embedding-adapter-dropout", type=float, default=0.05)

    parser.add_argument("--ia3-epochs", type=int, default=10)
    parser.add_argument("--ia3-batch-size", type=int, default=1)
    parser.add_argument("--ia3-eval-batch-size", type=int, default=2)
    parser.add_argument("--ia3-gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--ia3-head-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--ia3-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--ia3-weight-decay", type=float, default=0.01)
    parser.add_argument("--ia3-head-warmup-epochs", type=int, default=1)
    parser.add_argument("--ia3-max-grad-norm", type=float, default=0.1)

    parser.add_argument("--dora-epochs", type=int, default=10)
    parser.add_argument("--dora-batch-size", type=int, default=1)
    parser.add_argument("--dora-eval-batch-size", type=int, default=2)
    parser.add_argument("--dora-gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--dora-head-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--dora-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--dora-weight-decay", type=float, default=0.01)
    parser.add_argument("--dora-head-warmup-epochs", type=int, default=1)
    parser.add_argument("--dora-max-grad-norm", type=float, default=0.1)
    parser.add_argument("--dora-rank", type=int, default=8)
    parser.add_argument("--dora-alpha", type=float, default=16.0)
    parser.add_argument("--dora-dropout", type=float, default=0.05)

    parser.add_argument("--qlora-epochs", type=int, default=10)
    parser.add_argument("--qlora-batch-size", type=int, default=1)
    parser.add_argument("--qlora-eval-batch-size", type=int, default=2)
    parser.add_argument("--qlora-gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--qlora-head-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--qlora-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--qlora-weight-decay", type=float, default=0.01)
    parser.add_argument("--qlora-head-warmup-epochs", type=int, default=1)
    parser.add_argument("--qlora-max-grad-norm", type=float, default=0.1)
    parser.add_argument("--qlora-rank", type=int, default=8)
    parser.add_argument("--qlora-alpha", type=float, default=16.0)
    parser.add_argument("--qlora-dropout", type=float, default=0.05)
    parser.add_argument("--qlora-quantization-block-size", type=int, default=64)
    parser.add_argument("--rlvr-source-result-path", type=Path)
    parser.add_argument("--rlvr-epochs", type=int, default=1)
    parser.add_argument("--rlvr-batch-size", type=int, default=64)
    parser.add_argument("--rlvr-eval-batch-size", type=int, default=512)
    parser.add_argument("--rlvr-gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--rlvr-head-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--rlvr-adapter-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--rlvr-group-samples", type=int, default=8)
    parser.add_argument("--rlvr-reward-scale", type=float, default=0.05)
    parser.add_argument("--rlvr-outlier-threshold", type=float, default=0.15)
    parser.add_argument("--rlvr-outlier-penalty", type=float, default=0.5)
    parser.add_argument("--rlvr-kl-beta", type=float, default=0.02)
    parser.add_argument("--rlvr-entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--rlvr-checkpoint-dir", type=Path)
    parser.add_argument("--rlvr-checkpoint-steps", type=int, default=100)
    parser.add_argument(
        "--rlvr-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.catalogue.is_file():
        raise FileNotFoundError(f"Catalogue not found: {args.catalogue}")
    if args.z_max <= args.z_min:
        raise ValueError("--z-max must exceed --z-min.")
    if args.n_z_bins < 1:
        raise ValueError("--n-z-bins must be positive.")
    if not 0.0 <= args.min_cutout_coverage <= 1.0:
        raise ValueError("--min-cutout-coverage must lie in [0, 1].")
    fractions = args.train_fraction + args.test_fraction + args.val_fraction
    if not np.isclose(fractions, 1.0):
        raise ValueError("Train, test, and validation fractions must sum to one.")


def _product_cache_path(args: argparse.Namespace) -> Path:
    if args.product_cache_path is not None:
        return args.product_cache_path.expanduser()
    rows = "all" if args.max_rows is None else f"n{args.max_rows}"
    tag = (
        f"{args.catalogue.stem}_{args.input_mode.replace('-', '_')}_{rows}_"
        f"{args.sample_mode}_sample{args.sample_seed}_split{args.seed}_"
        f"frac{args.train_fraction:g}_{args.test_fraction:g}_{args.val_fraction:g}_"
        f"z{args.z_min:g}_{args.z_max:g}_bins{args.n_z_bins}_"
        f"cov{args.min_cutout_coverage:g}"
    ).replace(".", "p")
    return args.cache_root.expanduser() / "aion_posttraining_products" / f"{tag}.pt"


def _split_output_dir(args: argparse.Namespace) -> Path:
    rows = "all" if args.max_rows is None else f"n{args.max_rows}"
    return (
        args.cache_root.expanduser()
        / "aion_posttraining_splits"
        / f"{args.catalogue.stem}_{rows}_{args.sample_mode}_s{args.sample_seed}"
    )


def _base_config(args: argparse.Namespace, cache_path: Path) -> AIONMagnitudeConfig:
    return AIONMagnitudeConfig(
        catalogue_path=args.catalogue,
        max_rows=args.max_rows,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
        cache_root=args.cache_root,
        split_output_dir=_split_output_dir(args),
        cache_path=cache_path,
        split_strategy="random",
        train_fraction=args.train_fraction,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        z_min=args.z_min,
        z_max=args.z_max,
        n_z_bins=args.n_z_bins,
        extra_bands=(),
        use_aion_embedding=True,
        use_mlp_features=False,
        include_grizy_in_mlp=False,
        aion_input_bands=("g", "r", "i", "z", "y"),
        aion_embedding_batch_size=args.embedding_batch_size,
        force_recompute_embeddings=args.force_recompute_product,
        model_kinds=("aion",),
        seed=args.seed,
        device_choice=args.device,
    ).normalized()


def _subset_raw_dataset(
    dataset: CLAUDSPhotoZDataset,
    rows: np.ndarray,
) -> CLAUDSPhotoZDataset:
    row_tensor = torch.as_tensor(rows, dtype=torch.long)
    return CLAUDSPhotoZDataset(
        object_ids=[dataset.object_ids[index] for index in rows],
        fields=[dataset.fields[index] for index in rows],
        hsc_features={
            key: values[row_tensor]
            for key, values in dataset.hsc_features.items()
        },
        extra_features=torch.empty((len(rows), 0), dtype=torch.float32),
        z_spec=None if dataset.z_spec is None else dataset.z_spec[row_tensor],
        redshift_reference={
            key: values[row_tensor]
            for key, values in dataset.redshift_reference.items()
        },
    )


def _prepare_photometry_product(
    args: argparse.Namespace,
    cache_path: Path,
) -> dict[str, Any]:
    config = _base_config(args, cache_path)
    product = build_and_cache_aion_embeddings_from_config(config)
    if not product.get("aion_tokens"):
        raw_dataset, _, _ = build_raw_clauds_photoz_dataset(
            args.catalogue, _split_output_dir(args), max_rows=args.max_rows,
            sample_mode=args.sample_mode, sample_seed=args.sample_seed,
            z_min=args.z_min, z_max=args.z_max, n_z_bins=args.n_z_bins,
            extra_bands=(), use_mlp_features=False, include_grizy_in_mlp=False,
            use_aion_embedding=True,
        )
        if [str(value) for value in raw_dataset.object_ids] != [
            str(value) for value in product["object_id"]
        ]:
            raise RuntimeError("Legacy product rows do not match rebuilt token rows.")
        aion, codec_manager = load_frozen_aion(device=resolve_torch_device(args.device))
        product["aion_tokens"] = extract_aion_tokens_to_memory(
            raw_dataset, codec_manager, batch_size=args.embedding_batch_size,
            device=resolve_torch_device(args.device),
        )
        del aion, codec_manager
    metadata = dict(product.get("metadata", {}))
    metadata.update(
        {
            "aion_input_mode": "photometry",
            "input_representation": "native_aion_hsc_grizy_magnitudes",
            "aion_only": True,
            "uses_qwen": False,
            "aion_token_cache": True,
            "aion_token_modalities": sorted(product["aion_tokens"]),
        }
    )
    if product.get("metadata") != metadata:
        product["metadata"] = metadata
        torch.save(product, cache_path)
    return product


def _prepare_image_product(
    args: argparse.Namespace,
    cache_path: Path,
) -> dict[str, Any]:
    if cache_path.is_file() and not args.force_recompute_product:
        product = load_cached_product(cache_path)
        metadata = product.get("metadata", {})
        if metadata.get("aion_input_mode") != "photometry-images":
            raise RuntimeError(f"Cached product has the wrong AION input mode: {cache_path}")
        expected = {
            "image_root": str(args.hsc_image_dir.resolve()),
            "assignment_cache_dir": str(args.image_assignment_cache_dir.resolve()),
            "min_cutout_coverage": float(args.min_cutout_coverage),
        }
        mismatched = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatched:
            raise RuntimeError(
                f"Cached native-image provenance differs ({mismatched}); "
                "use --force-recompute-product."
            )
        if product.get("aion_tokens"):
            return product
        print(f"Legacy image product has no codec tokens; rebuilding {cache_path}", flush=True)

    raw_dataset, _, raw_metadata = build_raw_clauds_photoz_dataset(
        args.catalogue,
        _split_output_dir(args),
        max_rows=args.max_rows,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
        z_min=args.z_min,
        z_max=args.z_max,
        n_z_bins=args.n_z_bins,
        extra_bands=(),
        use_mlp_features=False,
        include_grizy_in_mlp=False,
        use_aion_embedding=True,
    )
    device = resolve_torch_device(args.device)
    aion, codec_manager = load_frozen_aion(device=device)
    extracted = extract_grizy_image_aion_embeddings(
        raw_dataset,
        aion=aion,
        codec_manager=codec_manager,
        device=device,
        catalogue_path=args.catalogue,
        hsc_image_dir=args.hsc_image_dir,
        assignment_cache_dir=args.image_assignment_cache_dir,
        batch_size=args.image_embedding_batch_size,
        min_cutout_coverage=args.min_cutout_coverage,
        fits_backend=args.fits_backend,
    )
    retained_rows = np.asarray(extracted["retained_rows"], dtype=np.int64)
    dataset = _subset_raw_dataset(raw_dataset, retained_rows)
    split_labels = make_split_labels(
        dataset.fields,
        split_strategy="random",
        train_fraction=args.train_fraction,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    metadata = {
        **raw_metadata,
        **extracted["metadata"],
        **split_metadata(
            split_labels,
            "random",
            args.train_fraction,
            args.test_fraction,
            args.val_fraction,
        ),
        "aion_model": "polymathic-ai/aion-base",
        "aion_embedding_pooling": "mean_encoder_tokens",
        "aion_encoder_tokens": 581,
        "aion_input_mode": "photometry-images",
        "input_representation": "native_aion_hsc_grizy_magnitudes_plus_images",
        "aion_only": True,
        "uses_qwen": False,
        "aion_token_cache": True,
        "aion_token_modalities": sorted(extracted["tokens"]),
    }
    save_cached_product(
        cache_path,
        dataset,
        extracted["embeddings"],
        feature_names=(),
        split_labels=split_labels,
        metadata=metadata,
        aion_tokens=extracted["tokens"],
    )
    return load_cached_product(cache_path)


def prepare_product(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    cache_path = _product_cache_path(args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if args.input_mode == "photometry":
            product = _prepare_photometry_product(args, cache_path)
        else:
            product = _prepare_image_product(args, cache_path)
    return product, cache_path


def _load_prepared_product(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    cache_path = _product_cache_path(args)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Prepared AION product not found: {cache_path}; run --stage prepare first."
        )
    product = load_cached_product(cache_path)
    mode = product.get("metadata", {}).get("aion_input_mode")
    if mode != args.input_mode:
        raise RuntimeError(f"Prepared product mode is {mode!r}, expected {args.input_mode!r}.")
    return product, cache_path


def _embedding_datasets(
    product: dict[str, Any],
) -> tuple[EmbeddingRedshiftDataset, EmbeddingRedshiftDataset, EmbeddingRedshiftDataset]:
    embeddings = torch.as_tensor(product["aion_embedding"], dtype=torch.float32)
    redshifts = torch.as_tensor(product["z_spec"], dtype=torch.float32)
    object_ids = [str(value) for value in product["object_id"]]
    split_labels = np.asarray(product["split_labels"], dtype=object)

    def build(split: str) -> EmbeddingRedshiftDataset:
        rows = np.flatnonzero(split_labels == split)
        row_tensor = torch.as_tensor(rows, dtype=torch.long)
        return EmbeddingRedshiftDataset(
            embeddings[row_tensor],
            redshifts[row_tensor],
            [object_ids[index] for index in rows],
        )

    return build("train"), build("val"), build("test")


def _token_datasets(
    product: dict[str, Any],
) -> tuple[AIONTokenRedshiftDataset, AIONTokenRedshiftDataset, AIONTokenRedshiftDataset]:
    cached_tokens = product.get("aion_tokens")
    if not cached_tokens:
        raise RuntimeError("Prepared product has no AION codec tokens; rerun --stage prepare.")
    tokens = {key: torch.as_tensor(value) for key, value in cached_tokens.items()}
    redshifts = torch.as_tensor(product["z_spec"], dtype=torch.float32)
    object_ids = [str(value) for value in product["object_id"]]
    split_labels = np.asarray(product["split_labels"], dtype=object)

    def build(split: str) -> AIONTokenRedshiftDataset:
        rows = np.flatnonzero(split_labels == split)
        row_tensor = torch.as_tensor(rows, dtype=torch.long)
        return AIONTokenRedshiftDataset(
            {key: value[row_tensor] for key, value in tokens.items()},
            redshifts[row_tensor],
            [object_ids[index] for index in rows],
        )

    return build("train"), build("val"), build("test")


def _write_summary(
    args: argparse.Namespace,
    *,
    method: str,
    result: dict[str, Any] | None,
    product_path: Path,
) -> None:
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": method,
        "aion_input_mode": args.input_mode,
        "catalogue": str(args.catalogue),
        "product_cache_path": str(product_path),
        "aion_only": True,
        "uses_qwen": False,
    }
    if result is not None:
        summary["result"] = str(output_dir / method / "result.pt")
        summary["final_metrics"] = result.get("final_metrics")
    (output_dir / f"{method}_run.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n"
    )


def stage_prepare(args: argparse.Namespace) -> int:
    product, cache_path = prepare_product(args)
    prepared = {
        "catalogue": str(args.catalogue),
        "input_mode": args.input_mode,
        "product_cache_path": str(cache_path),
        "n_rows": len(product["object_id"]),
        "embedding_shape": list(torch.as_tensor(product["aion_embedding"]).shape),
        "token_shapes": {
            key: list(torch.as_tensor(value).shape)
            for key, value in (product.get("aion_tokens") or {}).items()
        },
        "split_counts": product.get("metadata", {}).get("split_counts"),
        "metadata": product.get("metadata", {}),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prepared.json").write_text(
        json.dumps(_jsonable(prepared), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(_jsonable(prepared), indent=2, sort_keys=True), flush=True)
    return 0


def stage_head_only(args: argparse.Namespace) -> int:
    product, cache_path = _load_prepared_product(args)
    redshift_edges, redshift_centers = make_redshift_grid(
        args.z_min,
        args.z_max,
        args.n_z_bins,
    )
    output_dir = args.output_dir.expanduser() / "head_only"
    result = train_single_baseline(
        product,
        "aion",
        output_dir=output_dir,
        n_z_bins=args.n_z_bins,
        redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
        epochs=args.head_only_epochs,
        learning_rate=args.head_only_learning_rate,
        weight_decay=args.head_only_weight_decay,
        train_batch_size=args.head_only_batch_size,
        eval_batch_size=args.head_only_eval_batch_size,
        device=resolve_torch_device(args.device),
    )
    result["metadata"] = {
        **dict(product.get("metadata", {})),
        "posttraining_method": "head_only_cross_entropy",
        "adaptation_scope": "cached_mean_vector_baseline",
        "pooling": "mean_encoder_tokens",
        "comparison_role": "post_encoder_vector_baseline",
        "product_cache_path": str(cache_path),
    }
    torch.save(result, output_dir / "result.pt")
    _write_summary(
        args,
        method="head_only",
        result=result,
        product_path=cache_path,
    )
    print(f"saved {output_dir / 'result.pt'}", flush=True)
    return 0


def stage_attentive_head_only(args: argparse.Namespace) -> int:
    product, cache_path = _load_prepared_product(args)
    train, val, test = _token_datasets(product)
    config = AIONEmbeddingMethodConfig(
        method="frozen", n_z_bins=args.n_z_bins, z_min=args.z_min, z_max=args.z_max,
        epochs=args.encoder_head_epochs, batch_size=args.encoder_head_batch_size,
        eval_batch_size=args.encoder_head_eval_batch_size,
        gradient_accumulation_steps=args.encoder_head_gradient_accumulation_steps,
        learning_rate=args.encoder_head_learning_rate,
        weight_decay=args.encoder_head_weight_decay, head_warmup_epochs=0,
        seed=args.seed, device=args.device,
    ).normalized()
    output_dir = args.output_dir.expanduser() / "attentive_head_only"
    result = train_aion_embedding_method(
        train_dataset=train, val_dataset=val, test_dataset=test,
        output_dir=output_dir, config=config,
    )
    result["metadata"].update({
        "aion_input_mode": args.input_mode, "product_cache_path": str(cache_path),
        "aion_only": True, "uses_qwen": False,
        "comparison_role": "matched_frozen_encoder_baseline",
    })
    torch.save(result, output_dir / "result.pt")
    _write_summary(args, method="attentive_head_only", result=result, product_path=cache_path)
    print(f"saved {output_dir / 'result.pt'}", flush=True)
    return 0


def stage_embedding_adapter(args: argparse.Namespace) -> int:
    product, cache_path = _load_prepared_product(args)
    train, val, test = _embedding_datasets(product)
    config = ResidualEmbeddingAdapterConfig(
        n_z_bins=args.n_z_bins,
        z_min=args.z_min,
        z_max=args.z_max,
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
        device=args.device,
    ).normalized()
    output_dir = args.output_dir.expanduser() / "embedding_adapter"
    result = train_residual_embedding_adapter(
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        output_dir=output_dir,
        config=config,
    )
    result["metadata"].update(
        {
            "pooling": "mean_encoder_tokens",
            "adaptation_scope": "cached_vector_control",
            "comparison_role": "post_encoder_vector_control",
            "aion_input_mode": args.input_mode,
            "product_cache_path": str(cache_path),
            "aion_only": True,
            "uses_qwen": False,
        }
    )
    torch.save(result, output_dir / "result.pt")
    _write_summary(
        args,
        method="embedding_adapter",
        result=result,
        product_path=cache_path,
    )
    print(f"saved {output_dir / 'result.pt'}", flush=True)
    return 0


def stage_embedding_method(
    args: argparse.Namespace,
    method: str,
) -> int:
    product, cache_path = _load_prepared_product(args)
    train, val, test = _token_datasets(product)
    prefix = method.replace("-", "_")
    config = AIONEmbeddingMethodConfig(
        method=method,
        n_z_bins=args.n_z_bins,
        z_min=args.z_min,
        z_max=args.z_max,
        epochs=getattr(args, f"{prefix}_epochs"),
        batch_size=getattr(args, f"{prefix}_batch_size"),
        eval_batch_size=getattr(args, f"{prefix}_eval_batch_size"),
        gradient_accumulation_steps=getattr(
            args, f"{prefix}_gradient_accumulation_steps"
        ),
        learning_rate=getattr(args, f"{prefix}_head_learning_rate"),
        adapter_learning_rate=getattr(args, f"{prefix}_learning_rate"),
        weight_decay=getattr(args, f"{prefix}_weight_decay"),
        head_warmup_epochs=getattr(args, f"{prefix}_head_warmup_epochs"),
        adapter_max_grad_norm=getattr(args, f"{prefix}_max_grad_norm"),
        rank=getattr(args, f"{prefix}_rank", 8),
        alpha=getattr(args, f"{prefix}_alpha", 16.0),
        dropout=getattr(args, f"{prefix}_dropout", 0.0),
        quantization_block_size=getattr(
            args, f"{prefix}_quantization_block_size", 64
        ),
        seed=args.seed,
        device=args.device,
    ).normalized()
    output_dir = args.output_dir.expanduser() / method
    result = train_aion_embedding_method(
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        output_dir=output_dir,
        config=config,
    )
    result["metadata"].update(
        {
            "aion_input_mode": args.input_mode,
            "product_cache_path": str(cache_path),
            "aion_only": True,
            "uses_qwen": False,
        }
    )
    torch.save(result, output_dir / "result.pt")
    _write_summary(
        args,
        method=method,
        result=result,
        product_path=cache_path,
    )
    print(f"saved {output_dir / 'result.pt'}", flush=True)
    return 0


def stage_rlvr(args: argparse.Namespace) -> int:
    product, cache_path = _load_prepared_product(args)
    train, val, test = _embedding_datasets(product)
    source = (
        args.rlvr_source_result_path
        if args.rlvr_source_result_path is not None
        else args.output_dir.expanduser() / "embedding_adapter" / "result.pt"
    )
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
    result = train_embedding_adapter_rlvr(
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        output_dir=output_dir,
        source_result_path=source,
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
            "adaptation_scope": "cached_vector_control",
            "comparison_role": "post_encoder_vector_rlvr_control",
        }
    )
    torch.save(result, output_dir / "result.pt")
    _write_summary(
        args,
        method="rlvr",
        result=result,
        product_path=cache_path,
    )
    print(f"saved {output_dir / 'result.pt'}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    set_random_seed(args.seed)
    if args.stage == "prepare":
        return stage_prepare(args)
    if args.stage == "head-only":
        return stage_head_only(args)
    if args.stage == "attentive-head-only":
        return stage_attentive_head_only(args)
    if args.stage == "embedding-adapter":
        return stage_embedding_adapter(args)
    if args.stage in {"ia3", "dora", "qlora"}:
        return stage_embedding_method(args, args.stage)
    return stage_rlvr(args)


if __name__ == "__main__":
    raise SystemExit(main())
