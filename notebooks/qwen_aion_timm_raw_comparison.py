#!/usr/bin/env python3
"""Compare downstream AION-token and timm image fusion with raw-magnitude Qwen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import qwen_mlp_full_comparison as base
from aion_magnitude.timm_morphology import (
    DEFAULT_TIMM_MORPHOLOGY_MODEL,
    TimmMorphologyConfig,
    extract_or_load_timm_embeddings,
)


base.COMPARISON_NAME = "qwen_aion_timm_raw_comparison"
base.__doc__ = __doc__
base.DEFAULT_OUTPUT_DIR = Path(
    "/arc/home/gsm/aion_output/figures/qwen_aion-timm_raw_comparison"
)

_context: dict[str, object] = {}
_base_build_parser = base.build_parser
_base_extract_qwen = base.extract_or_load_qwen_embeddings
_base_train = base.train_single_morphology_model
_base_artifacts = base.save_morphology_comparison_artifacts
_base_save_manifest = base.save_run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = _base_build_parser()
    parser.add_argument("--timm-model", default=DEFAULT_TIMM_MORPHOLOGY_MODEL)
    parser.add_argument("--timm-input-size", type=int, default=224)
    parser.add_argument("--timm-batch-size", type=int, default=64)
    parser.add_argument("--timm-percentile", type=float, default=99.0)
    parser.add_argument("--timm-cache-path", type=Path)
    parser.add_argument("--timm-device-id", default=os.environ.get("AION_TIMM_GPU_DEVICE", "1"))
    parser.add_argument("--force-recompute-timm", action="store_true")
    parser.add_argument("--no-timm-pretrained", action="store_true")
    return parser


def _timm_config(args: argparse.Namespace, *, device: str = "cuda") -> TimmMorphologyConfig:
    return TimmMorphologyConfig(
        model_name=args.timm_model,
        pretrained=not args.no_timm_pretrained,
        input_size=args.timm_input_size,
        in_chans=1,
        global_pool="avg",
        batch_size=args.timm_batch_size,
        normalization="asinh_percentile",
        percentile=args.timm_percentile,
        device=device,
    ).normalized()


def _timm_cache_path(
    args: argparse.Namespace,
    product: dict,
    config: TimmMorphologyConfig,
) -> Path:
    if args.timm_cache_path is not None:
        return Path(args.timm_cache_path).expanduser()
    morphology_tag = str(dict(product.get("metadata", {}))["morphology_tag"])
    return (
        Path(args.cache_root).expanduser()
        / "timm_morphology"
        / morphology_tag
        / f"{config.cache_tag()}.pt"
    )


def _morphology_product_path(product: dict) -> Path:
    token_path = Path(str(product["image_token_ids_path"]))
    product_path = token_path.with_name("morphology_token_product.pt")
    if not product_path.exists():
        raise FileNotFoundError(f"Morphology product not found for timm worker: {product_path}")
    return product_path


def _spawn_timm_worker(
    args: argparse.Namespace,
    product: dict,
    cache_path: Path,
    config: TimmMorphologyConfig,
) -> subprocess.Popen:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "timm-worker",
        "--product-path",
        str(_morphology_product_path(product)),
        "--morphology-dir",
        str(Path(args.morphology_dir).expanduser()),
        "--cache-path",
        str(cache_path),
        "--model",
        config.model_name,
        "--input-size",
        str(config.input_size),
        "--batch-size",
        str(config.batch_size),
        "--percentile",
        str(config.percentile),
        *([] if config.pretrained else ["--no-pretrained"]),
        *(["--force"] if args.force_recompute_timm else []),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.timm_device_id)
    env["PYTHONUNBUFFERED"] = "1"
    print(
        f"starting timm raw-image worker on device {args.timm_device_id}: "
        f"{config.model_name}",
        flush=True,
    )
    return subprocess.Popen(command, cwd=ROOT, env=env)


def _load_compatible_qwen_cache(
    args: argparse.Namespace,
    product: dict,
    qwen_config,
    serialization,
    requested_path: Path,
) -> tuple[torch.Tensor | None, Path | None]:
    """Reuse native raw or qwen-qwen terse caches with identical serialization."""
    if args.force_recompute_qwen:
        return None, None
    expected = base.qwen_embedding_metadata(qwen_config, serialization)
    expected["input_feature_names"] = [
        str(name) for name in product.get("feature_names", [])
    ]
    candidates = [requested_path]
    candidates.extend(
        sorted(
            requested_path.parent.glob("*_terse.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    product_ids = [str(value) for value in product["object_id"]]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        cached = torch.load(candidate, map_location="cpu", weights_only=False)
        metadata = dict(cached.get("metadata", {}))
        if any(metadata.get(key) != value for key, value in expected.items()):
            continue
        if [str(value) for value in cached.get("object_id", [])] != product_ids:
            continue
        embedding = torch.as_tensor(cached.get("embedding"), dtype=torch.float32)
        if embedding.ndim != 2 or embedding.shape[0] != len(product_ids):
            continue
        print(f"reusing compatible raw-magnitude Qwen cache: {candidate}", flush=True)
        return embedding, candidate
    return None, None


def extract_qwen_and_timm(
    args,
    product,
    qwen_config,
    serialization,
    qwen_cache_path,
    device,
):
    timm_config = _timm_config(args)
    timm_cache_path = _timm_cache_path(args, product, timm_config)
    timm_worker = None
    if args.force_recompute_timm or not timm_cache_path.exists():
        timm_worker = _spawn_timm_worker(
            args, product, timm_cache_path, timm_config
        )
    try:
        qwen_embedding, actual_qwen_cache_path = _load_compatible_qwen_cache(
            args,
            product,
            qwen_config,
            serialization,
            Path(qwen_cache_path),
        )
        if qwen_embedding is None:
            qwen_embedding = _base_extract_qwen(
                args,
                product,
                qwen_config,
                serialization,
                qwen_cache_path,
                device,
            )
            actual_qwen_cache_path = Path(qwen_cache_path)
    except BaseException:
        if timm_worker is not None and timm_worker.poll() is None:
            timm_worker.terminate()
            timm_worker.wait()
        raise
    if timm_worker is not None:
        return_code = timm_worker.wait()
        if return_code != 0:
            raise RuntimeError(f"timm raw-image worker failed with exit code {return_code}.")
    timm_embedding = extract_or_load_timm_embeddings(
        product,
        morphology_dir=args.morphology_dir,
        cache_path=timm_cache_path,
        config=timm_config,
        force_recompute=False,
    )
    if qwen_embedding.shape[0] != timm_embedding.shape[0]:
        raise RuntimeError("Qwen and timm embedding row counts differ.")
    _context.update(
        args=args,
        source_product=product,
        qwen_embedding=qwen_embedding,
        timm_embedding=timm_embedding,
        timm_cache_path=timm_cache_path,
        timm_config=timm_config,
        qwen_cache_path=actual_qwen_cache_path,
    )
    return qwen_embedding


def train_image_backend_comparison(product, model_kind, **kwargs):
    kwargs = dict(kwargs)
    base.am.set_random_seed(kwargs["config"].seed)
    if model_kind == "qwen_morphology":
        kwargs["output_dir"] = Path(kwargs["output_dir"]) / "aion_image_tokens"
        return _base_train(product, "qwen_morphology", **kwargs)
    timm_product = dict(product)
    timm_product["aion_embedding"] = torch.as_tensor(
        _context["qwen_embedding"], dtype=torch.float32
    )
    timm_product["extra_features"] = torch.as_tensor(
        _context["timm_embedding"], dtype=torch.float32
    )
    timm_metadata = dict(timm_product.get("metadata", {}))
    timm_metadata.update(
        {
            "image_embedding_backend": "timm",
            "timm_cache_path": str(_context["timm_cache_path"]),
            "timm_config": {
                key: str(value)
                for key, value in vars(_context["timm_config"]).items()
                if key != "device"
            },
        }
    )
    timm_product["metadata"] = timm_metadata
    timm_product["feature_names"] = [
        f"timm_{index}" for index in range(timm_product["extra_features"].shape[1])
    ]
    kwargs["output_dir"] = Path(kwargs["output_dir"]) / "timm_raw_image"
    return _base_train(timm_product, "frozen_image_morphology", **kwargs)


def save_artifacts(results, **kwargs):
    kwargs["comparison_labels"] = (
        "raw-magnitude-Qwen+AION-FSQ-image",
        "raw-magnitude-Qwen+timm-raw-u-image",
    )
    return _base_artifacts(results, **kwargs)


def save_manifest(path: Path, **kwargs) -> None:
    _base_save_manifest(path, **kwargs)
    manifest = json.loads(path.read_text())
    manifest.update(
        {
            "image_backend_primary": "timm",
            "image_backend_comparison": "aion_fsq_tokens",
            "timm_cache": str(_context.get("timm_cache_path", "")),
            "actual_qwen_cache": str(_context.get("qwen_cache_path", "")),
            "image_tokens_read_by_qwen": False,
            "timm_features_read_by_qwen": False,
        }
    )
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def timm_worker_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Internal frozen-timm extraction worker.")
    parser.add_argument("--product-path", type=Path, required=True)
    parser.add_argument("--morphology-dir", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--percentile", type=float, required=True)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    product = torch.load(args.product_path, map_location="cpu", weights_only=False)
    config = TimmMorphologyConfig(
        model_name=args.model,
        pretrained=not args.no_pretrained,
        input_size=args.input_size,
        batch_size=args.batch_size,
        percentile=args.percentile,
        device="cuda",
    )
    extract_or_load_timm_embeddings(
        product,
        morphology_dir=args.morphology_dir,
        cache_path=args.cache_path,
        config=config,
        force_recompute=args.force,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "timm-worker":
        return timm_worker_main(argv[1:])
    base.build_parser = build_parser
    base.extract_or_load_qwen_embeddings = extract_qwen_and_timm
    base.train_single_morphology_model = train_image_backend_comparison
    base.save_morphology_comparison_artifacts = save_artifacts
    base.save_run_manifest = save_manifest
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
