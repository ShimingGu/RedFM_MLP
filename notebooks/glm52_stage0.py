#!/usr/bin/env python3
"""Check cached/live/prepared/zero-adapter GLM representation invariants."""

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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = Path(__file__).resolve().parent
for import_path in (ROOT, NOTEBOOK_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import glm52_posttraining as glm_run
from aion_magnitude.Inference_Opt_TFM import (
    InferenceOptimizedEmbeddingConfig,
    load_inference_optimized_transformer,
    pool_hidden_states,
    serialize_catalogue_row,
)
from aion_magnitude.metrics import predict_photoz_from_logits, summarize_pdf_metrics
from aion_magnitude.models import PhotoZHead
from aion_magnitude.utils import make_redshift_grid, set_random_seed
from peft import PeftModel, prepare_model_for_kbit_training
from safetensors.torch import load_file as load_safetensors


DEFAULT_RUN_ROOT = Path(
    "/arc/home/gsm/aion_output/figures/glm52-posttraining-e10/photometry"
)
DEFAULT_OUTPUT_DIR = Path(
    "/arc/home/gsm/aion_output/figures/glm52-failure-analysis-0828/"
    "stage0_invariants"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-json", type=Path, default=DEFAULT_RUN_ROOT / "prepared.json"
    )
    parser.add_argument(
        "--frozen-checkpoint",
        type=Path,
        default=DEFAULT_RUN_ROOT / "head_only/iotfm_baseline.pt",
    )
    parser.add_argument(
        "--frozen-result",
        type=Path,
        default=DEFAULT_RUN_ROOT / "head_only/result.pt",
    )
    parser.add_argument(
        "--adapter-dir", type=Path, default=DEFAULT_RUN_ROOT / "qlora/adapter"
    )
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-every", type=int, default=32)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Cannot compare tensor shapes {tuple(reference.shape)} and "
            f"{tuple(candidate.shape)}."
        )
    difference = candidate - reference
    reference_rms = reference.square().mean().sqrt().clamp_min(1.0e-12)
    cosine = F.cosine_similarity(reference, candidate, dim=-1)
    return {
        "mean_cosine_similarity": float(cosine.mean().item()),
        "min_cosine_similarity": float(cosine.min().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "normalized_rmse": float(
            (difference.square().mean().sqrt() / reference_rms).item()
        ),
        "mean_absolute_difference": float(difference.abs().mean().item()),
        "maximum_absolute_difference": float(difference.abs().max().item()),
    }


def serialize_rows(product: dict[str, Any], rows: np.ndarray, serialization) -> list[str]:
    values = torch.as_tensor(product["extra_features"], dtype=torch.float32)
    names = [str(name) for name in product["feature_names"]]
    return [
        serialize_catalogue_row(
            {
                name: values[int(row), column]
                for column, name in enumerate(names)
            },
            config=serialization,
        )
        for row in rows
    ]


def tokenizer_diagnostics(tokenizer, texts: list[str], max_length: int) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    raw = tokenizer(texts, padding=False, truncation=False, add_special_tokens=True)
    truncated = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    raw_lengths = np.asarray([len(ids) for ids in raw["input_ids"]], dtype=np.int64)
    mask = truncated["attention_mask"]
    lengths = mask.sum(dim=1).long()
    last_indices = lengths.clamp_min(1) - 1
    last_ids = truncated["input_ids"][torch.arange(len(texts)), last_indices]
    digest = hashlib.sha256()
    digest.update(truncated["input_ids"].numpy().tobytes())
    digest.update(mask.numpy().tobytes())
    diagnostics = {
        "tokenizer_class": type(tokenizer).__name__,
        "padding_side": str(tokenizer.padding_side),
        "truncation_side": str(tokenizer.truncation_side),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_length": int(max_length),
        "truncated_rows": int(np.count_nonzero(raw_lengths > max_length)),
        "raw_length": {
            "minimum": int(raw_lengths.min()),
            "p50": float(np.quantile(raw_lengths, 0.50)),
            "p95": float(np.quantile(raw_lengths, 0.95)),
            "p99": float(np.quantile(raw_lengths, 0.99)),
            "maximum": int(raw_lengths.max()),
        },
        "encoded_length": {
            "minimum": int(lengths.min().item()),
            "maximum": int(lengths.max().item()),
        },
        "last_token_ids": {
            str(int(token_id)): int(count)
            for token_id, count in zip(*torch.unique(last_ids, return_counts=True))
        },
        "token_tensor_sha256": digest.hexdigest(),
    }
    return diagnostics, {
        key: value.detach().cpu() for key, value in truncated.items()
    }


@torch.no_grad()
def extract_embeddings(
    label: str,
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
    progress_every: int,
) -> tuple[torch.Tensor, str]:
    model.eval()
    batches: list[torch.Tensor] = []
    token_digest = hashlib.sha256()
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(texts), batch_size), start=1):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        token_digest.update(encoded["input_ids"].numpy().tobytes())
        token_digest.update(encoded["attention_mask"].numpy().tobytes())
        inputs = {key: value.to(device) for key, value in encoded.items()}
        output = model(**inputs, use_cache=False, return_dict=True)
        pooled = pool_hidden_states(
            output.last_hidden_state,
            inputs["attention_mask"],
            pooling="last",
        )
        batches.append(pooled.detach().float().cpu())
        if batch_number == 1 or batch_number == total_batches or (
            progress_every > 0 and batch_number % progress_every == 0
        ):
            print(
                f"{label}: batch {batch_number}/{total_batches}",
                flush=True,
            )
    return torch.cat(batches, dim=0), token_digest.hexdigest()


def load_head(checkpoint_path: Path) -> tuple[PhotoZHead, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    prefix = "photoz_head."
    head_state = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    input_dim = int(head_state["net.0.weight"].shape[1])
    hidden_dim = int(head_state["net.0.weight"].shape[0])
    n_z_bins = int(head_state["net.6.weight"].shape[0])
    head = PhotoZHead(input_dim, n_z_bins=n_z_bins, hidden_dim=hidden_dim)
    head.load_state_dict(head_state)
    head.eval()
    return head, checkpoint


@torch.no_grad()
def head_logits(head: PhotoZHead, embeddings: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    return torch.cat(
        [head(embeddings[start : start + batch_size].float()) for start in range(0, len(embeddings), batch_size)],
        dim=0,
    )


def evaluation_metrics(
    logits: torch.Tensor,
    z_spec: torch.Tensor,
    edges: torch.Tensor,
    centers: torch.Tensor,
) -> dict[str, float]:
    evaluation: dict[str, Any] = {
        "logits": logits.float().cpu(),
        "z_spec": z_spec.float().cpu(),
        "redshift_edges": edges.float().cpu(),
        "redshift_centers": centers.float().cpu(),
    }
    evaluation.update(
        predict_photoz_from_logits(evaluation["logits"], centers=evaluation["redshift_centers"])
    )
    return summarize_pdf_metrics(evaluation)


def adapter_statistics(adapter_dir: Path) -> dict[str, Any]:
    model_path = adapter_dir / "adapter_model.safetensors"
    tensors = load_safetensors(model_path, device="cpu")
    groups: dict[str, list[torch.Tensor]] = {"lora_A": [], "lora_B": [], "other": []}
    for name, value in tensors.items():
        if "lora_A" in name:
            groups["lora_A"].append(value.float())
        elif "lora_B" in name:
            groups["lora_B"].append(value.float())
        else:
            groups["other"].append(value.float())

    def summarize(values: list[torch.Tensor]) -> dict[str, float | int]:
        if not values:
            return {"tensor_count": 0, "parameter_count": 0, "maximum_absolute_value": 0.0, "l2_norm": 0.0}
        return {
            "tensor_count": len(values),
            "parameter_count": int(sum(value.numel() for value in values)),
            "maximum_absolute_value": float(max(value.abs().max().item() for value in values)),
            "l2_norm": float(sum(value.square().sum().item() for value in values) ** 0.5),
        }

    return {
        "path": str(model_path),
        "sha256": file_sha256(model_path),
        "groups": {name: summarize(values) for name, values in groups.items()},
    }


def prepare_pipeline_args(prepared: dict[str, Any]) -> argparse.Namespace:
    command = [
        "--stage",
        "prepare",
        "--catalogue",
        str(prepared["catalogue"]),
        "--output-dir",
        str(Path(prepared["catalogue"]).parent),
        "--max-rows",
        str(prepared["max_rows"]),
        "--seed",
        "42",
        "--glm-model",
        str(prepared["resolved_model_path"]),
        "--glm-max-length",
        str(prepared["max_length"]),
    ]
    if prepared.get("use_morphology"):
        command.append("--use-morphology")
    return glm_run.build_parser().parse_args(command)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    comparisons = report["comparisons"]
    metrics = report["sample_metrics"]
    lines = [
        "# GLM Stage 0 invariant report",
        "",
        f"Sample: first {report['sample']['size']} validation rows from the fixed seed-42 split.",
        "",
        "## Representation comparisons",
        "",
        "| Comparison | Mean cosine | Minimum cosine | Normalized RMSE | Maximum absolute difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in comparisons["embeddings"].items():
        lines.append(
            f"| {name} | {values['mean_cosine_similarity']:.9f} | "
            f"{values['min_cosine_similarity']:.9f} | {values['normalized_rmse']:.6g} | "
            f"{values['maximum_absolute_difference']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Frozen-head replay",
            "",
            "| Path | Cross-entropy | NMAD | Bias | Catastrophic fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, values in metrics.items():
        lines.append(
            f"| {name} | {values['cross_entropy']:.6f} | {values['nmad']:.6f} | "
            f"{values['bias']:.6f} | {values['catastrophic_outlier_fraction']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Automatic checks",
            "",
        ]
    )
    for name, passed in report["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    if args.sample_size < 1 or args.batch_size < 1:
        raise ValueError("Sample size and batch size must be positive.")
    prepared = json.loads(args.prepared_json.read_text())
    frozen_result = torch.load(args.frozen_result, map_location="cpu", weights_only=False)
    cache_path = args.cache_path
    if cache_path is None:
        cache_path = Path(frozen_result["metadata"]["embedding_cache"])
    required_paths = (
        args.prepared_json,
        args.frozen_checkpoint,
        args.frozen_result,
        args.adapter_dir / "adapter_config.json",
        args.adapter_dir / "adapter_model.safetensors",
        cache_path,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 0 inputs: {missing}")

    set_random_seed(42)
    pipeline_args = prepare_pipeline_args(prepared)
    print("building the exact catalogue product", flush=True)
    product, preparation, bounds = glm_run.catalogue_pipeline.build_product(pipeline_args)
    labels = np.asarray(product["split_labels"], dtype=object)
    validation_rows = np.flatnonzero(labels == "val")
    rows = validation_rows[: min(args.sample_size, len(validation_rows))]
    if len(rows) != args.sample_size:
        raise ValueError(
            f"Requested {args.sample_size} validation rows, but only {len(rows)} exist."
        )

    print(f"memory-mapping embedding cache {cache_path}", flush=True)
    cached = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    cache_ids = [str(value) for value in cached["object_id"]]
    product_ids = [str(value) for value in product["object_id"]]
    row_order_matches = cache_ids == product_ids
    if not row_order_matches:
        raise RuntimeError("Cached and reconstructed catalogue object-ID order differs.")
    cached_embeddings = torch.as_tensor(cached["embedding"])[rows].float().clone()
    sample_ids = [product_ids[int(row)] for row in rows]
    z_spec = torch.as_tensor(product["z_spec"], dtype=torch.float32)[rows]
    texts = serialize_rows(product, rows, glm_run._serialization(pipeline_args))

    model_config = InferenceOptimizedEmbeddingConfig(
        model_path=prepared["resolved_model_path"],
        device=args.device,
        torch_dtype="bfloat16",
        max_length=int(prepared["max_length"]),
        pooling="last",
        normalize=False,
        local_files_only=True,
        trust_remote_code=True,
        load_in_4bit=True,
        freeze_model=True,
    )
    print(f"loading live model on {args.device}", flush=True)
    tokenizer, base_model, device = load_inference_optimized_transformer(model_config)
    token_diagnostics, token_tensors = tokenizer_diagnostics(
        tokenizer, texts, int(prepared["max_length"])
    )

    live_embeddings, live_token_hash = extract_embeddings(
        "live",
        texts,
        tokenizer,
        base_model,
        device,
        batch_size=args.batch_size,
        max_length=int(prepared["max_length"]),
        progress_every=args.progress_every,
    )

    print("preparing the 4-bit model for k-bit training", flush=True)
    prepared_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True
    )
    prepared_model.eval()
    prepared_embeddings, prepared_token_hash = extract_embeddings(
        "prepared",
        texts,
        tokenizer,
        prepared_model,
        device,
        batch_size=args.batch_size,
        max_length=int(prepared["max_length"]),
        progress_every=args.progress_every,
    )

    print(f"loading saved epoch-0 adapter {args.adapter_dir}", flush=True)
    peft_model = PeftModel.from_pretrained(
        prepared_model,
        args.adapter_dir,
        is_trainable=False,
    )
    peft_model.eval()
    zero_adapter_embeddings, zero_adapter_token_hash = extract_embeddings(
        "zero-adapter",
        texts,
        tokenizer,
        peft_model,
        device,
        batch_size=args.batch_size,
        max_length=int(prepared["max_length"]),
        progress_every=args.progress_every,
    )

    head, frozen_checkpoint = load_head(args.frozen_checkpoint)
    embedding_paths = {
        "cached": cached_embeddings,
        "live": live_embeddings,
        "prepared": prepared_embeddings,
        "zero_adapter": zero_adapter_embeddings,
    }
    logits = {
        name: head_logits(head, embedding) for name, embedding in embedding_paths.items()
    }
    edges, centers = make_redshift_grid(bounds[0], bounds[1], logits["cached"].shape[-1])
    sample_metrics = {
        name: evaluation_metrics(value, z_spec, edges, centers)
        for name, value in logits.items()
    }

    embedding_comparisons = {
        "cached_vs_live": tensor_comparison(cached_embeddings, live_embeddings),
        "cached_vs_prepared": tensor_comparison(
            cached_embeddings, prepared_embeddings
        ),
        "live_vs_prepared": tensor_comparison(live_embeddings, prepared_embeddings),
        "prepared_vs_zero_adapter": tensor_comparison(
            prepared_embeddings, zero_adapter_embeddings
        ),
        "cached_vs_zero_adapter": tensor_comparison(
            cached_embeddings, zero_adapter_embeddings
        ),
    }
    logit_comparisons = {
        "cached_vs_live": tensor_comparison(logits["cached"], logits["live"]),
        "cached_vs_prepared": tensor_comparison(
            logits["cached"], logits["prepared"]
        ),
        "live_vs_prepared": tensor_comparison(logits["live"], logits["prepared"]),
        "prepared_vs_zero_adapter": tensor_comparison(
            logits["prepared"], logits["zero_adapter"]
        ),
        "cached_vs_zero_adapter": tensor_comparison(
            logits["cached"], logits["zero_adapter"]
        ),
    }
    adapter = adapter_statistics(args.adapter_dir)
    token_hashes = {
        "live": live_token_hash,
        "prepared": prepared_token_hash,
        "zero_adapter": zero_adapter_token_hash,
    }
    checks = {
        "cache_row_order_matches_product": row_order_matches,
        "no_rows_truncated": token_diagnostics["truncated_rows"] == 0,
        "tokenization_identical_across_live_paths": len(set(token_hashes.values())) == 1,
        "cached_vs_prepared_mean_cosine_at_least_0.999": embedding_comparisons[
            "cached_vs_prepared"
        ]["mean_cosine_similarity"] >= 0.999,
        "prepared_vs_zero_adapter_mean_cosine_at_least_0.999999": embedding_comparisons[
            "prepared_vs_zero_adapter"
        ]["mean_cosine_similarity"]
        >= 0.999999,
        "saved_lora_B_is_exactly_zero": adapter["groups"]["lora_B"][
            "maximum_absolute_value"
        ]
        == 0.0,
        "cached_vs_prepared_cross_entropy_within_0.01": abs(
            sample_metrics["cached"]["cross_entropy"]
            - sample_metrics["prepared"]["cross_entropy"]
        )
        <= 0.01,
    }
    if not checks["cached_vs_prepared_mean_cosine_at_least_0.999"]:
        interpretation = (
            "The k-bit-prepared NF4 representation does not reproduce the corrected cache. "
            "Investigate cache provenance and prepared-model inference before training."
        )
    elif not checks["prepared_vs_zero_adapter_mean_cosine_at_least_0.999999"]:
        interpretation = (
            "The corrected cache matches the prepared model, but loading the inactive adapter changes "
            "the representation. Inspect PEFT initialization and checkpoint loading."
        )
    else:
        interpretation = (
            "The corrected prepared-model cache, live prepared model, and saved zero-adapter "
            "representations agree. Stage 0 clears the matched PEFT representation path."
        )

    report: dict[str, Any] = {
        "stage": "glm52_stage0_invariants",
        "sample": {
            "split": "val",
            "selection": "first validation rows in reconstructed catalogue order",
            "size": len(rows),
            "row_indices_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
            "object_ids_sha256": hashlib.sha256(
                "\n".join(sample_ids).encode()
            ).hexdigest(),
        },
        "inputs": {
            "prepared_json": str(args.prepared_json),
            "frozen_checkpoint": str(args.frozen_checkpoint),
            "frozen_checkpoint_sha256": file_sha256(args.frozen_checkpoint),
            "frozen_result": str(args.frozen_result),
            "cache_path": str(cache_path),
            "cache_metadata": cached.get("metadata", {}),
            "adapter": adapter,
            "model": str(prepared["resolved_model_path"]),
            "device": str(device),
        },
        "catalogue_preparation": preparation,
        "redshift_bounds": [float(bounds[0]), float(bounds[1])],
        "tokenization": token_diagnostics,
        "inference_token_hashes": token_hashes,
        "comparisons": {
            "embeddings": embedding_comparisons,
            "head_logits": logit_comparisons,
        },
        "sample_metrics": sample_metrics,
        "reference_full_validation_metrics": frozen_checkpoint.get(
            "final_metrics", {}
        ).get("val", {}),
        "checks": checks,
        "interpretation": interpretation,
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "stage0_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    tensor_path = args.output_dir / "stage0_sample.pt"
    torch.save(
        {
            "sample_rows": torch.from_numpy(rows.copy()),
            "object_id": sample_ids,
            "z_spec": z_spec,
            "token_tensors": token_tensors,
            "embeddings": embedding_paths,
            "logits": logits,
            "redshift_edges": edges,
            "redshift_centers": centers,
        },
        tensor_path,
    )
    markdown_path = args.output_dir / "stage0_report.md"
    write_markdown(markdown_path, report)
    print(json.dumps({"checks": checks, "interpretation": interpretation}, indent=2), flush=True)
    print(f"saved {report_path}", flush=True)
    print(f"saved {markdown_path}", flush=True)
    print(f"saved {tensor_path}", flush=True)

    del peft_model, prepared_model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
