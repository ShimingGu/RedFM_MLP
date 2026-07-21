#!/usr/bin/env python3
"""Physical all-magnitude + image-token Qwen versus unchanged image-token MLP."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import qwen_mlp_full_comparison as base
from aion_magnitude.FM_Qwen import QwenEmbeddingConfig, load_frozen_qwen
from aion_magnitude.FM_Qwen3 import (
    QWEN_IMAGE_INPUT_MODES,
    Qwen3SerializationConfig,
    qwen3_embedding_metadata,
    serialize_qwen3_batch,
)

base.COMPARISON_NAME = "qwen_mlp_full_image_comparison"
base.DEFAULT_OUTPUT_DIR = Path("/arc/home/gsm/aion_output/figures/qwen-mlp_full_image_comparison")

_run_tag = "center16"
_physical_context = True
_original_build_parser = base.build_parser


def build_parser():
    parser = _original_build_parser()
    parser.add_argument(
        "--qwen-image-input-mode", choices=QWEN_IMAGE_INPUT_MODES, default="center_crop"
    )
    parser.add_argument("--qwen-image-crop-size", type=int, default=16)
    parser.add_argument("--no-qwen-physical-context", action="store_true")
    parser.add_argument("--allow-qwen-truncation", action="store_true")
    return parser


def physical_settings(args):
    global _run_tag, _physical_context
    _physical_context = not args.no_qwen_physical_context
    image_tag = (
        f"center{args.qwen_image_crop_size}"
        if args.qwen_image_input_mode == "center_crop" else "full24_raw"
    )
    _run_tag = image_tag if _physical_context else f"raw_{image_tag}"
    return (
        QwenEmbeddingConfig(
            model_path=args.qwen_model,
            device=args.device,
            load_in_4bit=not args.no_qwen_4bit,
            torch_dtype="auto",
            max_length=args.qwen_max_length,
            pooling=args.qwen_pooling,
            normalize=args.qwen_normalize,
            local_files_only=not args.allow_qwen_download,
            trust_remote_code=True,
        ),
        Qwen3SerializationConfig(
            schema_name="clauds_physical_magnitudes_aion_image_v2",
            image_input_mode=args.qwen_image_input_mode,
            image_crop_size=args.qwen_image_crop_size,
            include_physical_context=_physical_context,
            include_image_context=False,
            final_marker="Combined galaxy representation:",
        ),
    )


def expected_metadata(config, serialization, feature_names):
    physical = bool(serialization.include_physical_context)
    return {
        **qwen3_embedding_metadata(config, serialization),
        "input_feature_names": feature_names,
        "input_scope": (
            "physically described all magnitudes plus tokenized galaxy image"
            if physical else "raw all-magnitude columns plus tokenized galaxy image"
        ),
        "aion_image_embedding_used": False,
        "aion_image_tokens_read_by_qwen": True,
    }


def prompt_length_preflight(args, features, feature_names, token_store, token_rows, tokenizer, config, serialization):
    if not len(features):
        raise ValueError("Qwen prompt preflight received zero rows.")
    sample_count = min(256, len(features))
    sample_rows = np.linspace(0, len(features) - 1, sample_count, dtype=np.int64)
    texts = serialize_qwen3_batch(
        features[sample_rows],
        feature_names,
        np.asarray(token_store[token_rows[sample_rows]], dtype=np.int64),
        config=serialization,
    )
    lengths = np.asarray([
        len(ids) for ids in tokenizer(
            texts, truncation=False
        )["input_ids"]
    ])
    stats = {
        "sample_count": int(sample_count),
        "minimum": int(lengths.min()),
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "maximum": int(lengths.max()),
        "max_length": int(config.max_length),
        "sampled_rows_exceeding_max_length": int((lengths > config.max_length).sum()),
    }
    print(f"Qwen prompt token lengths (n={sample_count}): "
          f"min={stats['minimum']} median={stats['median']:.1f} "
          f"p95={stats['p95']:.1f} max={stats['maximum']} "
          f"limit={config.max_length}", flush=True)
    if stats["sampled_rows_exceeding_max_length"] and not args.allow_qwen_truncation:
        raise RuntimeError(
            "Qwen image prompts exceed --qwen-max-length; change compactification or "
            "max length, or explicitly pass --allow-qwen-truncation for a diagnostic run."
        )
    return stats



def extract_physical_image_embeddings(args, product, config, serialization, cache_path, device):
    feature_names = [str(name) for name in product.get("feature_names", [])]
    features = torch.as_tensor(product["extra_features"], dtype=torch.float32)
    expected = expected_metadata(config, serialization, feature_names)
    if cache_path.exists() and not args.force_recompute_qwen:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        return base.validate_qwen_cache(cached, product, expected, cache_path)

    token_store = np.load(product["image_token_ids_path"], mmap_mode="r")
    token_rows = np.asarray(product["image_token_row_indices"], dtype=np.int64)
    model, tokenizer = load_frozen_qwen(
        config.model_path,
        device=device,
        load_in_4bit=config.load_in_4bit,
        torch_dtype=config.torch_dtype,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )
    prompt_stats = prompt_length_preflight(
        args, features, feature_names, token_store, token_rows, tokenizer, config, serialization
    )
    parts = []
    try:
        for start in range(0, len(features), args.qwen_embedding_batch_size):
            stop = min(start + args.qwen_embedding_batch_size, len(features))
            tokens = np.asarray(token_store[token_rows[start:stop]], dtype=np.int64)
            texts = serialize_qwen3_batch(
                features[start:stop], feature_names, tokens, config=serialization
            )
            parts.append(base.extract_qwen_embeddings_from_texts(
                texts, model, tokenizer, device=device,
                max_length=config.max_length, batch_size=args.qwen_embedding_batch_size,
                pooling=config.pooling, normalize=config.normalize,
            ))
            if stop == len(features) or stop % max(1000, args.qwen_embedding_batch_size) == 0:
                label = ("physical magnitude+image" if serialization.include_physical_context
                         else "raw magnitude+image")
                print(f"{label} Qwen embeddings: {stop:,}/{len(features):,}", flush=True)
        if not parts:
            raise ValueError("Qwen3 extraction received zero morphology-matched rows.")
        embeddings = torch.cat(parts)
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"object_id": list(product["object_id"]), "embedding": embeddings, "metadata": {**expected, "prompt_token_length_preflight": prompt_stats}}, cache_path)
    return embeddings


_original_train = base.train_single_morphology_model
_original_artifacts = base.save_morphology_comparison_artifacts


def train_without_second_image_input(product, model_kind, **kwargs):
    # Qwen has already read the image tokens; its downstream head receives only
    # the frozen Qwen representation. The unchanged MLP branch still uses them.
    if model_kind == "qwen_morphology":
        return _original_train(product, "aion", **kwargs)
    return _original_train(product, model_kind, **kwargs)


def save_artifacts(results, **kwargs):
    kwargs["comparison_labels"] = (
        (
            "physical-all-magnitude-Qwen+tokenized-galaxy-image"
            if _physical_context else "raw-all-magnitude-Qwen+tokenized-galaxy-image"
        ),
        "all-magnitude-MLP+tokenized-galaxy-image",
    )
    return _original_artifacts(results, **kwargs)


def main(argv=None):
    original_run_tag = base.qwen_run_tag
    base.qwen_run_tag = lambda config: f"physical_image_{_run_tag}_" + original_run_tag(config)
    base.build_parser = build_parser
    base.qwen_settings = physical_settings
    base.expected_qwen_metadata = expected_metadata
    base.extract_or_load_qwen_embeddings = extract_physical_image_embeddings
    base.train_single_morphology_model = train_without_second_image_input
    base.save_morphology_comparison_artifacts = save_artifacts
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
