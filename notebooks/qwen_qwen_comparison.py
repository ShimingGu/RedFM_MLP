#!/usr/bin/env python3
"""Compare terse and physically described all-magnitude Qwen inputs, without images."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import qwen_mlp_full_comparison as base
from aion_magnitude.FM_Qwen import (
    QwenEmbeddingConfig, QwenSerializationConfig, load_frozen_qwen,
    qwen_embedding_metadata, serialize_qwen_feature_row,
)
from aion_magnitude.FM_Qwen3 import (
    QWEN_PHYSICAL_CONTEXT_MODES, Qwen3SerializationConfig,
    qwen3_embedding_metadata, serialize_qwen3_batch,
)

base.COMPARISON_NAME = "qwen_qwen_comparison"
base.DEFAULT_OUTPUT_DIR = Path("/arc/home/gsm/aion_output/figures/qwen-qwen_comparison")
_context = {}
_base_build_parser = base.build_parser
_base_qwen_run_tag = base.qwen_run_tag


def build_parser():
    parser = _base_build_parser()
    parser.add_argument(
        "--qwen-physical-context-mode",
        choices=QWEN_PHYSICAL_CONTEXT_MODES,
        default="full",
        help="Ablate the amount of physical meaning added to the magnitude prompt.",
    )
    parser.add_argument(
        "--qwen-summary-marker",
        action="store_true",
        help="End the physical prompt with an explicit representation marker.",
    )
    return parser


def settings(args):
    config = QwenEmbeddingConfig(
        model_path=args.qwen_model, device=args.device,
        load_in_4bit=not args.no_qwen_4bit, torch_dtype="auto",
        max_length=args.qwen_max_length, pooling=args.qwen_pooling,
        normalize=args.qwen_normalize, local_files_only=not args.allow_qwen_download,
        trust_remote_code=True,
    )
    mode = args.qwen_physical_context_mode
    _context["physical_context_mode"] = mode
    _context["summary_marker"] = bool(args.qwen_summary_marker)
    serialization = Qwen3SerializationConfig(
        include_physical_context=mode != "none",
        physical_context_mode=mode,
        final_marker=(
            "Combined galaxy representation:" if args.qwen_summary_marker else None
        ),
    )
    return config, serialization


def physical_metadata(config, serialization, names):
    return {
        **qwen3_embedding_metadata(config, serialization),
        "input_feature_names": names,
        "input_scope": (
            f"all magnitudes with {serialization.physical_context_mode} physical context"
        ),
        "aion_image_tokens_read_by_qwen": False,
    }


def extract_text_embeddings(args, product, config, serialization, cache_path, device, *, physical):
    names = [str(name) for name in product.get("feature_names", [])]
    values = torch.as_tensor(product["extra_features"], dtype=torch.float32)
    if physical:
        metadata = physical_metadata(config, serialization, names)
        schema = "physical"
    else:
        terse = QwenSerializationConfig(
            schema_name="clauds_all_magnitude_v1", include_hsc_grizy=False,
            include_object_metadata=False, prefix="galaxy all_magnitudes_ab",
        )
        metadata = {
            **qwen_embedding_metadata(config, terse), "input_feature_names": names,
            "input_scope": "terse all magnitudes", "aion_image_tokens_read_by_qwen": False,
        }
        serialization = terse
        schema = "terse"
    if cache_path.exists() and not args.force_recompute_qwen:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        return base.validate_qwen_cache(cached, product, metadata, cache_path)

    model, tokenizer = load_frozen_qwen(
        config.model_path, device=device, load_in_4bit=config.load_in_4bit,
        torch_dtype=config.torch_dtype, local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )
    parts = []
    try:
        for start in range(0, len(values), args.qwen_embedding_batch_size):
            batch = values[start:start + args.qwen_embedding_batch_size]
            if physical:
                texts = serialize_qwen3_batch(batch, names, config=serialization)
            else:
                texts = [serialize_qwen_feature_row(
                    {names[col]: batch[row, col] for col in range(batch.shape[1])},
                    serialization=serialization,
                ) for row in range(batch.shape[0])]
            parts.append(base.extract_qwen_embeddings_from_texts(
                texts, model, tokenizer, device=device, max_length=config.max_length,
                batch_size=args.qwen_embedding_batch_size, pooling=config.pooling,
                normalize=config.normalize,
            ))
            completed = min(start + len(batch), len(values))
            if completed == len(values) or completed % max(1000, args.qwen_embedding_batch_size) == 0:
                print(
                    f"{schema} Qwen embeddings: {completed:,}/{len(values):,}",
                    flush=True,
                )
        if not parts:
            raise ValueError(f"{schema} Qwen extraction received zero rows.")
        result = torch.cat(parts)
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"object_id": list(product["object_id"]), "embedding": result, "metadata": metadata}, cache_path)
    return result


def extract_physical(args, product, config, serialization, cache_path, device):
    _context.update(args=args, product=product, config=config, device=device, cache_path=cache_path)
    return extract_text_embeddings(args, product, config, serialization, cache_path, device, physical=True)


_train = base.train_single_morphology_model
_artifacts = base.save_morphology_comparison_artifacts


def train_two_qwens(product, model_kind, **kwargs):
    if model_kind == "qwen_morphology":
        kwargs = dict(kwargs)
        kwargs["output_dir"] = Path(kwargs["output_dir"]) / "physical_qwen"
        base.am.set_random_seed(kwargs["config"].seed)
        return _train(product, "aion", **kwargs)
    args, source, config, device = (_context[k] for k in ("args", "product", "config", "device"))
    terse_path = Path(_context["cache_path"]).with_name(Path(_context["cache_path"]).stem + "_terse.pt")
    terse_embedding = extract_text_embeddings(
        args, source, config, QwenSerializationConfig(), terse_path, device, physical=False
    )
    terse_product = dict(source)
    terse_product["aion_embedding"] = terse_embedding
    kwargs = dict(kwargs)
    kwargs["output_dir"] = Path(kwargs["output_dir"]) / "terse_qwen"
    base.am.set_random_seed(kwargs["config"].seed)
    return _train(terse_product, "aion", **kwargs)


def artifacts(results, **kwargs):
    kwargs["comparison_labels"] = ("physical-all-magnitude-Qwen", "terse-all-magnitude-Qwen")
    return _artifacts(results, **kwargs)


def main(argv=None):
    base.qwen_run_tag = lambda config: (
        "physical_magnitude_"
        f"{_context.get('physical_context_mode', 'full')}_"
        f"marker{int(_context.get('summary_marker', False))}_"
        + _base_qwen_run_tag(config)
    )
    base.build_parser = build_parser
    base.qwen_settings = settings
    base.expected_qwen_metadata = physical_metadata
    base.extract_or_load_qwen_embeddings = extract_physical
    base.train_single_morphology_model = train_two_qwens
    base.save_morphology_comparison_artifacts = artifacts
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
