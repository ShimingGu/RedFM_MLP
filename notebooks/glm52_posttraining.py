#!/usr/bin/env python3
"""Post-train the local GLM-5.2 0.8B frozen mapper for catalogue photo-z."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = Path(__file__).resolve().parent
for import_path in (ROOT, NOTEBOOK_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import qwen_posttraining_comparison as catalogue_pipeline
from aion_magnitude.Inference_Opt_TFM import (
    CatalogueSerializationConfig,
    InferenceOptimizedEmbeddingConfig,
    build_embedding_metadata,
    extract_text_embeddings,
    load_inference_optimized_transformer,
    resolve_model_path,
    serialize_catalogue_row,
)
from aion_magnitude.glm52_posttraining import (
    DEFAULT_GLM52_MODEL,
    GLM52_IA3_FEEDFORWARD_MODULES,
    GLM52_IA3_TARGET_MODULES,
    GLM52_LORA_TARGET_SETTING,
    comma_join,
    inspect_glm52_architecture,
)
from aion_magnitude.qwen_alternative_posttraining import (
    EmbeddingRedshiftDataset,
    ResidualEmbeddingAdapterConfig,
    train_ia3_photoz,
    train_residual_embedding_adapter,
)
from aion_magnitude.qwen_posttraining import (
    QwenPosttrainingConfig,
    TextRedshiftDataset,
    train_qlora_photoz,
)
from aion_magnitude.qwen_rlvr import RLVRConfig, train_qlora_rlvr
from aion_magnitude.training import train_single_baseline
from aion_magnitude.utils import make_redshift_grid, set_random_seed


COMPARISON_NAME = "glm52-posttraining"
DEFAULT_OUTPUT_DIR = Path(
    "/arc/home/gsm/aion_output/figures/glm52-posttraining-e10"
)


def _action(parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
    return next(action for action in parser._actions if action.dest == dest)


def _add_option_alias(
    parser: argparse.ArgumentParser, action: argparse.Action, alias: str
) -> None:
    action.option_strings.append(alias)
    parser._option_string_actions[alias] = action


def build_parser() -> argparse.ArgumentParser:
    parser = catalogue_pipeline.build_parser()
    parser.description = __doc__
    _action(parser, "stage").choices = (
        "prepare",
        "frozen",
        "qlora",
        "dora",
        "ia3",
        "embedding-adapter",
        "rlvr",
    )
    _action(parser, "output_dir").default = DEFAULT_OUTPUT_DIR

    model_action = _action(parser, "qwen_model")
    _add_option_alias(parser, model_action, "--glm-model")
    model_action.default = DEFAULT_GLM52_MODEL
    model_action.help = "Local GLM checkpoint, registered short name, or Hub ID."

    max_length_action = _action(parser, "qwen_max_length")
    _add_option_alias(parser, max_length_action, "--glm-max-length")
    max_length_action.default = 512
    _add_option_alias(
        parser,
        _action(parser, "qwen_embedding_batch_size"),
        "--glm-embedding-batch-size",
    )
    _add_option_alias(
        parser, _action(parser, "allow_qwen_download"), "--allow-glm-download"
    )
    _add_option_alias(
        parser,
        _action(parser, "force_recompute_qwen"),
        "--force-recompute-glm",
    )

    parser.add_argument(
        "--qlora-target-modules",
        default=GLM52_LORA_TARGET_SETTING,
    )

    parser.add_argument("--dora-epochs", type=int, default=10)
    parser.add_argument("--dora-batch-size", type=int, default=1)
    parser.add_argument("--dora-gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--dora-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--dora-head-warmup-epochs", type=int, default=3)
    parser.add_argument("--dora-max-grad-norm", type=float, default=0.1)
    parser.add_argument("--dora-rank", type=int, default=8)
    parser.add_argument("--dora-alpha", type=int, default=16)
    parser.add_argument("--dora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--dora-target-modules",
        default=GLM52_LORA_TARGET_SETTING,
    )
    parser.add_argument("--dora-checkpoint-dir", type=Path)
    parser.add_argument("--dora-checkpoint-steps", type=int, default=100)
    parser.add_argument(
        "--dora-resume", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--ia3-epochs", type=int, default=10)
    parser.add_argument("--ia3-batch-size", type=int, default=1)
    parser.add_argument("--ia3-gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--ia3-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--ia3-head-warmup-epochs", type=int, default=3)
    parser.add_argument("--ia3-max-grad-norm", type=float, default=0.1)
    parser.add_argument(
        "--ia3-target-modules",
        default=comma_join(GLM52_IA3_TARGET_MODULES),
    )
    parser.add_argument(
        "--ia3-feedforward-modules",
        default=comma_join(GLM52_IA3_FEEDFORWARD_MODULES),
    )
    parser.add_argument("--ia3-checkpoint-dir", type=Path)
    parser.add_argument("--ia3-checkpoint-steps", type=int, default=100)
    parser.add_argument(
        "--ia3-resume", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--embedding-adapter-epochs", type=int, default=10)
    parser.add_argument("--embedding-adapter-batch-size", type=int, default=16)
    parser.add_argument("--embedding-adapter-eval-batch-size", type=int, default=8)
    parser.add_argument("--embedding-adapter-bottleneck-dim", type=int, default=256)
    parser.add_argument("--embedding-adapter-learning-rate", type=float, default=2.0e-4)
    parser.add_argument(
        "--embedding-adapter-adapter-learning-rate", type=float, default=1.0e-5
    )
    parser.add_argument("--embedding-adapter-weight-decay", type=float, default=0.01)
    parser.add_argument("--embedding-adapter-head-warmup-epochs", type=int, default=3)
    parser.add_argument("--embedding-adapter-max-grad-norm", type=float, default=0.1)
    parser.add_argument("--embedding-adapter-dropout", type=float, default=0.05)

    parser.add_argument("--rlvr-epochs", type=int, default=1)
    parser.add_argument("--rlvr-batch-size", type=int, default=1)
    parser.add_argument("--rlvr-gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--rlvr-head-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--rlvr-adapter-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--rlvr-group-samples", type=int, default=8)
    parser.add_argument("--rlvr-reward-scale", type=float, default=0.05)
    parser.add_argument("--rlvr-outlier-threshold", type=float, default=0.15)
    parser.add_argument("--rlvr-outlier-penalty", type=float, default=0.5)
    parser.add_argument("--rlvr-kl-beta", type=float, default=0.02)
    parser.add_argument("--rlvr-entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--rlvr-source-adapter-dir", type=Path)
    parser.add_argument("--rlvr-source-head-path", type=Path)
    parser.add_argument("--rlvr-checkpoint-dir", type=Path)
    parser.add_argument("--rlvr-checkpoint-steps", type=int, default=100)
    parser.add_argument(
        "--rlvr-resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    catalogue_pipeline._validate_args(args)
    positive = {
        "dora_epochs": args.dora_epochs,
        "dora_batch_size": args.dora_batch_size,
        "dora_gradient_accumulation_steps": args.dora_gradient_accumulation_steps,
        "dora_rank": args.dora_rank,
        "dora_alpha": args.dora_alpha,
        "dora_checkpoint_steps": args.dora_checkpoint_steps,
        "ia3_epochs": args.ia3_epochs,
        "ia3_batch_size": args.ia3_batch_size,
        "ia3_gradient_accumulation_steps": args.ia3_gradient_accumulation_steps,
        "ia3_checkpoint_steps": args.ia3_checkpoint_steps,
        "embedding_adapter_epochs": args.embedding_adapter_epochs,
        "embedding_adapter_batch_size": args.embedding_adapter_batch_size,
        "embedding_adapter_eval_batch_size": args.embedding_adapter_eval_batch_size,
        "embedding_adapter_bottleneck_dim": args.embedding_adapter_bottleneck_dim,
        "rlvr_epochs": args.rlvr_epochs,
        "rlvr_batch_size": args.rlvr_batch_size,
        "rlvr_gradient_accumulation_steps": args.rlvr_gradient_accumulation_steps,
        "rlvr_group_samples": args.rlvr_group_samples,
        "rlvr_checkpoint_steps": args.rlvr_checkpoint_steps,
    }
    invalid = [name for name, value in positive.items() if int(value) < 1]
    if invalid:
        raise ValueError(f"Positive GLM post-training settings required: {invalid}")
    for method, warmup, epochs in (
        ("DoRA", args.dora_head_warmup_epochs, args.dora_epochs),
        ("IA3", args.ia3_head_warmup_epochs, args.ia3_epochs),
        (
            "embedding adapter",
            args.embedding_adapter_head_warmup_epochs,
            args.embedding_adapter_epochs,
        ),
    ):
        if not 0 <= int(warmup) < int(epochs):
            raise ValueError(f"{method} head warmup must be in [0, epochs).")
    if not 0 <= args.embedding_adapter_dropout < 1:
        raise ValueError("Embedding-adapter dropout must be in [0, 1).")
    for name in (
        "qlora_target_modules",
        "dora_target_modules",
        "ia3_target_modules",
        "ia3_feedforward_modules",
    ):
        if not [part.strip() for part in getattr(args, name).split(",") if part.strip()]:
            raise ValueError(f"--{name.replace('_', '-')} must not be empty.")
    if not set(args.ia3_feedforward_modules.split(",")) <= set(
        args.ia3_target_modules.split(",")
    ):
        raise ValueError("IA3 feed-forward modules must be a subset of targets.")


def _resolved_model(args: argparse.Namespace) -> str:
    return resolve_model_path(args.qwen_model)


def _serialization(args: argparse.Namespace) -> CatalogueSerializationConfig:
    return CatalogueSerializationConfig(
        schema_name=(
            "clauds_all_magnitude_multiband_morphology_glm52_v1"
            if args.use_morphology
            else "clauds_all_magnitude_glm52_v1"
        ),
        decimals=5,
        missing_token="NA",
        prefix=(
            "CLAUDS galaxy AB magnitudes and measured multiband morphology"
            if args.use_morphology
            else "CLAUDS galaxy AB magnitudes"
        ),
        include_feature_names=True,
    )


def _serialize_product(
    product: dict[str, Any], serialization: CatalogueSerializationConfig
) -> list[str]:
    values = torch.as_tensor(product["extra_features"], dtype=torch.float32)
    names = [str(name) for name in product["feature_names"]]
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("Catalogue feature matrix and names are inconsistent.")
    return [
        serialize_catalogue_row(
            {name: values[row, column] for column, name in enumerate(names)},
            config=serialization,
        )
        for row in range(len(values))
    ]


def _embedding_config(args: argparse.Namespace) -> InferenceOptimizedEmbeddingConfig:
    return InferenceOptimizedEmbeddingConfig(
        model_path=args.qwen_model,
        device="cuda",
        torch_dtype="bfloat16",
        max_length=args.qwen_max_length,
        pooling="last",
        normalize=False,
        local_files_only=not args.allow_qwen_download,
        trust_remote_code=True,
        load_in_4bit=True,
        freeze_model=True,
    )


def _cache_path(
    args: argparse.Namespace,
    product: dict[str, Any],
    config: InferenceOptimizedEmbeddingConfig,
) -> Path:
    catalogue = Path(product["metadata"]["catalogue"])
    stat = catalogue.stat()
    provenance = hashlib.sha256(
        f"{catalogue.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()[:12]
    input_tag = "magnitudes_morphology" if args.use_morphology else "magnitudes"
    selection_tag = "all" if args.max_rows is None else f"n{args.max_rows}"
    model_tag = Path(resolve_model_path(config.model_path)).name.replace("-", "_")
    return (
        Path(args.cache_root).expanduser()
        / "glm52_posttraining"
        / catalogue.stem
        / f"{selection_tag}_seed{args.seed}_{input_tag}_{provenance}"
        / f"{model_tag}_last_len{config.max_length}_nf4.pt"
    )


def _expected_cache_metadata(
    config: InferenceOptimizedEmbeddingConfig,
    serialization: CatalogueSerializationConfig,
    product: dict[str, Any],
) -> dict[str, Any]:
    return {
        **build_embedding_metadata(config, serialization),
        "input_feature_names": [str(name) for name in product["feature_names"]],
        "input_representation": product["metadata"]["input_representation"],
        "image_cutouts_read": False,
        "image_tokens_read": False,
    }


def _extract_or_load_embeddings(
    args: argparse.Namespace, product: dict[str, Any]
) -> tuple[torch.Tensor, Path]:
    config = _embedding_config(args)
    serialization = _serialization(args)
    cache_path = _cache_path(args, product, config)
    expected = _expected_cache_metadata(config, serialization, product)
    object_ids = [str(value) for value in product["object_id"]]

    if cache_path.exists() and not args.force_recompute_qwen:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if [str(value) for value in cached.get("object_id", [])] != object_ids:
            raise RuntimeError(f"GLM cache row order differs: {cache_path}")
        if cached.get("metadata") != expected:
            raise RuntimeError(
                f"GLM cache settings differ: {cache_path}; rebuild with "
                "--force-recompute-glm."
            )
        embeddings = torch.as_tensor(cached.get("embedding"), dtype=torch.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(object_ids):
            raise RuntimeError(f"Invalid GLM embedding cache: {cache_path}")
        return embeddings, cache_path

    texts = _serialize_product(product, serialization)
    tokenizer, model, device = load_inference_optimized_transformer(config)
    try:
        embeddings = extract_text_embeddings(
            texts,
            tokenizer,
            model,
            device,
            batch_size=args.qwen_embedding_batch_size,
            max_length=config.max_length,
            pooling="last",
            normalize=False,
        ).float()
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if embeddings.shape != (len(object_ids), 2048):
        raise RuntimeError(
            f"Expected GLM embeddings {(len(object_ids), 2048)}, got "
            f"{tuple(embeddings.shape)}."
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"object_id": object_ids, "embedding": embeddings, "metadata": expected},
        cache_path,
    )
    print(f"saved {cache_path}", flush=True)
    return embeddings, cache_path


def _prepared_architecture(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "prepared.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("architecture", {})


def stage_prepare(args: argparse.Namespace) -> int:
    product, preparation, bounds = catalogue_pipeline.build_product(args)
    architecture = inspect_glm52_architecture(args.qwen_model)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "comparison": COMPARISON_NAME,
        "catalogue": preparation["catalogue"],
        "model": args.qwen_model,
        "resolved_model_path": _resolved_model(args),
        "architecture": architecture,
        "use_morphology": bool(args.use_morphology),
        "input_representation": product["metadata"]["input_representation"],
        "image_cutouts_read": False,
        "image_tokens_read": False,
        "pooling": "last",
        "max_length": args.qwen_max_length,
        "max_rows": args.max_rows,
        "redshift_bounds": list(bounds),
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
    product, _, bounds = catalogue_pipeline.build_product(args)
    embeddings, cache_path = _extract_or_load_embeddings(args, product)
    frozen_product = dict(product)
    frozen_product["aion_embedding"] = embeddings
    frozen_product["metadata"] = {
        **product["metadata"],
        "model_family": "glm52_frozen_mapping",
        "posttraining_method": "glm52_head_only_control",
        "pooling": "last",
        "embedding_cache": str(cache_path),
        "architecture": _prepared_architecture(Path(args.output_dir).expanduser()),
    }
    output_dir = Path(args.output_dir).expanduser() / "head_only"
    edges, centers = make_redshift_grid(bounds[0], bounds[1], args.n_z_bins)
    set_random_seed(args.seed)
    result = train_single_baseline(
        frozen_product,
        "iotfm",
        output_dir=output_dir,
        n_z_bins=args.n_z_bins,
        redshift_edges=edges,
        redshift_centers=centers,
        epochs=args.frozen_epochs,
        learning_rate=args.head_learning_rate,
        weight_decay=1.0e-4,
        train_batch_size=args.frozen_train_batch_size,
        eval_batch_size=max(args.eval_batch_size, 64),
        device="cuda",
    )
    result.setdefault("metadata", {}).update(frozen_product["metadata"])
    torch.save(result, output_dir / "result.pt")
    print(f"saved {output_dir / 'result.pt'}", flush=True)
    return 0


def _text_datasets(
    args: argparse.Namespace,
) -> tuple[TextRedshiftDataset, TextRedshiftDataset, TextRedshiftDataset, tuple[float, float]]:
    product, _, bounds = catalogue_pipeline.build_product(args)
    texts = _serialize_product(product, _serialization(args))
    redshifts = torch.as_tensor(product["z_spec"], dtype=torch.float32)
    object_ids = [str(value) for value in product["object_id"]]
    labels = np.asarray(product["split_labels"], dtype=object)

    def dataset(split: str) -> TextRedshiftDataset:
        rows = np.flatnonzero(labels == split)
        indices = torch.as_tensor(rows, dtype=torch.long)
        return TextRedshiftDataset(
            [texts[index] for index in rows],
            redshifts[indices],
            [object_ids[index] for index in rows],
        )

    return dataset("train"), dataset("val"), dataset("test"), bounds


def _posttraining_config(
    args: argparse.Namespace,
    bounds: tuple[float, float],
    *,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    adapter_learning_rate: float,
    head_warmup_epochs: int,
    max_grad_norm: float,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: str,
) -> QwenPosttrainingConfig:
    return QwenPosttrainingConfig(
        model_path=_resolved_model(args),
        max_length=args.qwen_max_length,
        pooling="last",
        n_z_bins=args.n_z_bins,
        z_min=bounds[0],
        z_max=bounds[1],
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.head_learning_rate,
        lora_learning_rate=adapter_learning_rate,
        lora_max_grad_norm=max_grad_norm,
        head_warmup_epochs=head_warmup_epochs,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        lora_target_modules=target_modules,
        seed=args.seed,
        device="cuda",
        local_files_only=not args.allow_qwen_download,
    ).normalized()


def _annotate_result(
    args: argparse.Namespace,
    result: dict[str, Any],
    *,
    method: str,
    adapter_scope: str,
) -> None:
    output_dir = Path(args.output_dir).expanduser()
    method_dir = "embedding_adapter" if method == "embedding-adapter" else method
    metadata = result.setdefault("metadata", {})
    metadata.update(
        {
            "model_family": "glm52_frozen_mapping",
            "base_model": args.qwen_model,
            "resolved_model_path": _resolved_model(args),
            "checkpoint_role": "architecture_test_checkpoint",
            "adapter_scope": adapter_scope,
            "architecture": _prepared_architecture(output_dir),
            "image_cutouts_read": False,
            "image_tokens_read": False,
        }
    )
    torch.save(result, output_dir / method_dir / "result.pt")
    prepared_path = output_dir / "prepared.json"
    prepared = json.loads(prepared_path.read_text()) if prepared_path.exists() else {}
    summary = {
        "comparison": COMPARISON_NAME,
        "method": method,
        "model": args.qwen_model,
        "resolved_model_path": _resolved_model(args),
        "catalogue": prepared.get("catalogue"),
        "input_representation": prepared.get("input_representation"),
        "pooling": "last",
        "adapter_scope": adapter_scope,
        "result": str(output_dir / method_dir / "result.pt"),
        "final_metrics": result["final_metrics"],
    }
    if "verifier_metrics" in result:
        summary["verifier_metrics"] = result["verifier_metrics"]
    (output_dir / f"{method_dir}_run.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def stage_qlora(args: argparse.Namespace) -> int:
    train, val, test, bounds = _text_datasets(args)
    config = _posttraining_config(
        args,
        bounds,
        epochs=args.qlora_epochs,
        batch_size=args.qlora_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        adapter_learning_rate=args.qlora_learning_rate,
        head_warmup_epochs=args.head_warmup_epochs,
        max_grad_norm=args.lora_max_grad_norm,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.qlora_target_modules,
    )
    result = train_qlora_photoz(
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        output_dir=Path(args.output_dir).expanduser() / "qlora",
        config=config,
        checkpoint_dir=args.qlora_checkpoint_dir,
        checkpoint_interval=args.qlora_checkpoint_steps,
        resume=args.qlora_resume,
    )
    _annotate_result(
        args,
        result,
        method="qlora",
        adapter_scope="GLM MLA attention and DSA-indexer linears; all MLP and routed-expert tensors frozen",
    )
    return 0


def stage_dora(args: argparse.Namespace) -> int:
    train, val, test, bounds = _text_datasets(args)
    config = _posttraining_config(
        args,
        bounds,
        epochs=args.dora_epochs,
        batch_size=args.dora_batch_size,
        gradient_accumulation_steps=args.dora_gradient_accumulation_steps,
        adapter_learning_rate=args.dora_learning_rate,
        head_warmup_epochs=args.dora_head_warmup_epochs,
        max_grad_norm=args.dora_max_grad_norm,
        rank=args.dora_rank,
        alpha=args.dora_alpha,
        dropout=args.dora_dropout,
        target_modules=args.dora_target_modules,
    )
    result = train_qlora_photoz(
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        output_dir=Path(args.output_dir).expanduser() / "dora",
        config=config,
        checkpoint_dir=args.dora_checkpoint_dir,
        checkpoint_interval=args.dora_checkpoint_steps,
        resume=args.dora_resume,
        use_dora=True,
    )
    _annotate_result(
        args,
        result,
        method="dora",
        adapter_scope="GLM MLA attention and DSA-indexer linears; all MLP and routed-expert tensors frozen",
    )
    return 0


def stage_ia3(args: argparse.Namespace) -> int:
    train, val, test, bounds = _text_datasets(args)
    config = _posttraining_config(
        args,
        bounds,
        epochs=args.ia3_epochs,
        batch_size=args.ia3_batch_size,
        gradient_accumulation_steps=args.ia3_gradient_accumulation_steps,
        adapter_learning_rate=args.ia3_learning_rate,
        head_warmup_epochs=args.ia3_head_warmup_epochs,
        max_grad_norm=args.ia3_max_grad_norm,
        rank=1,
        alpha=1,
        dropout=0.0,
        target_modules=args.ia3_target_modules,
    )
    result = train_ia3_photoz(
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        output_dir=Path(args.output_dir).expanduser() / "ia3",
        config=config,
        checkpoint_dir=args.ia3_checkpoint_dir,
        checkpoint_interval=args.ia3_checkpoint_steps,
        resume=args.ia3_resume,
        target_modules=args.ia3_target_modules,
        feedforward_modules=args.ia3_feedforward_modules,
    )
    _annotate_result(
        args,
        result,
        method="ia3",
        adapter_scope="MLA kv_b_proj output scaling plus shared/dense down_proj feed-forward input scaling; routed experts frozen",
    )
    return 0


def _embedding_datasets(
    args: argparse.Namespace,
) -> tuple[
    EmbeddingRedshiftDataset,
    EmbeddingRedshiftDataset,
    EmbeddingRedshiftDataset,
    tuple[float, float],
    Path,
]:
    product, _, bounds = catalogue_pipeline.build_product(args)
    embeddings, cache_path = _extract_or_load_embeddings(args, product)
    redshifts = torch.as_tensor(product["z_spec"], dtype=torch.float32)
    object_ids = [str(value) for value in product["object_id"]]
    labels = np.asarray(product["split_labels"], dtype=object)

    def dataset(split: str) -> EmbeddingRedshiftDataset:
        rows = np.flatnonzero(labels == split)
        indices = torch.as_tensor(rows, dtype=torch.long)
        return EmbeddingRedshiftDataset(
            embeddings[indices],
            redshifts[indices],
            [object_ids[index] for index in rows],
        )

    return dataset("train"), dataset("val"), dataset("test"), bounds, cache_path


def stage_embedding_adapter(args: argparse.Namespace) -> int:
    train, val, test, bounds, cache_path = _embedding_datasets(args)
    config = ResidualEmbeddingAdapterConfig(
        n_z_bins=args.n_z_bins,
        z_min=bounds[0],
        z_max=bounds[1],
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
        device="cuda",
    ).normalized()
    result = train_residual_embedding_adapter(
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        output_dir=Path(args.output_dir).expanduser() / "embedding_adapter",
        config=config,
    )
    result.setdefault("metadata", {})["embedding_cache"] = str(cache_path)
    _annotate_result(
        args,
        result,
        method="embedding-adapter",
        adapter_scope="post-encoder residual bottleneck on cached frozen last-token embeddings",
    )
    return 0


def stage_rlvr(args: argparse.Namespace) -> int:
    train, val, test, bounds = _text_datasets(args)
    output_dir = Path(args.output_dir).expanduser()
    source_adapter = (
        args.rlvr_source_adapter_dir
        if args.rlvr_source_adapter_dir is not None
        else output_dir / "qlora/adapter"
    )
    source_head = (
        args.rlvr_source_head_path
        if args.rlvr_source_head_path is not None
        else output_dir / "qlora/photoz_head.pt"
    )
    base_config = _posttraining_config(
        args,
        bounds,
        epochs=args.qlora_epochs,
        batch_size=args.qlora_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        adapter_learning_rate=args.qlora_learning_rate,
        head_warmup_epochs=args.head_warmup_epochs,
        max_grad_norm=args.lora_max_grad_norm,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.qlora_target_modules,
    )
    rlvr_config = RLVRConfig(
        epochs=args.rlvr_epochs,
        batch_size=args.rlvr_batch_size,
        eval_batch_size=args.eval_batch_size,
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
    result = train_qlora_rlvr(
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        output_dir=output_dir / "rlvr",
        base_config=base_config,
        rlvr_config=rlvr_config,
        source_adapter_dir=source_adapter,
        source_head_path=source_head,
        checkpoint_dir=args.rlvr_checkpoint_dir,
        checkpoint_interval=args.rlvr_checkpoint_steps,
        resume=args.rlvr_resume,
    )
    _annotate_result(
        args,
        result,
        method="rlvr",
        adapter_scope="QLoRA policy continued with grouped verifiable photo-z reward; same GLM linear targets",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    stages = {
        "prepare": stage_prepare,
        "frozen": stage_frozen,
        "qlora": stage_qlora,
        "dora": stage_dora,
        "ia3": stage_ia3,
        "embedding-adapter": stage_embedding_adapter,
        "rlvr": stage_rlvr,
    }
    return stages[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())


