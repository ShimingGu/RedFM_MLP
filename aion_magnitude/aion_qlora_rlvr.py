from __future__ import annotations

"""Encoder-level AION QLoRA continuation with group-relative RLVR."""

from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .aion_embedding_methods import (
    AIONEmbeddingMethodConfig,
    AIONEncoderPhotoZModel,
    AIONTokenRedshiftDataset,
    _load_trainable_state,
    _trainable_state,
    build_aion_embedding_method_model,
    evaluate_aion_token_model,
)
from .metrics import summarize_pdf_metrics
from .models import load_frozen_aion
from .qwen_posttraining import _linear_warmup_decay_lambda, trainable_parameter_summary
from .qwen_rlvr import (
    RLVRConfig,
    _restore_rng_state,
    _rng_state,
    expected_verifier_reward,
    rlvr_group_policy_loss,
)
from .utils import make_redshift_grid, resolve_torch_device, set_random_seed


def _cohort_fingerprint(*datasets: AIONTokenRedshiftDataset) -> str:
    digest = hashlib.sha256()
    for dataset in datasets:
        for object_id in dataset.object_ids:
            digest.update(str(object_id).encode())
            digest.update(b"\0")
        digest.update(np.ascontiguousarray(dataset.redshifts.numpy()).view(np.uint8))
        digest.update(b"\xff")
    return digest.hexdigest()


def load_aion_qlora_source_artifact(
    source_result_path: str | Path,
) -> tuple[
    AIONEmbeddingMethodConfig,
    dict[str, torch.Tensor],
    dict[str, Any],
    Path,
]:
    """Validate and read a completed encoder-level AION QLoRA policy."""
    source_path = Path(source_result_path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"Completed AION QLoRA result not found: {source_path}")
    result = torch.load(source_path, map_location="cpu", weights_only=False)
    metadata = dict(result.get("metadata", {}))
    if result.get("model_kind") != "aion_qlora_encoder_photoz":
        raise ValueError(
            "AION RLVR requires model_kind='aion_qlora_encoder_photoz'; "
            "cached-vector residual adapters are not valid source policies."
        )
    if metadata.get("adaptation_scope") != "aion_encoder":
        raise ValueError("AION RLVR source must adapt the AION encoder itself.")
    raw_config = dict(metadata.get("config", {}))
    if raw_config.get("method") != "qlora":
        raise ValueError("AION RLVR source configuration must use method='qlora'.")
    config = AIONEmbeddingMethodConfig(**raw_config).normalized()

    checkpoint_path = Path(metadata.get("checkpoint", "")).expanduser()
    if not checkpoint_path.is_file():
        checkpoint_path = source_path.parent / "qlora_aion_encoder_adapter.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Completed AION QLoRA encoder checkpoint not found: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if dict(checkpoint.get("config", {})) != asdict(config):
        raise ValueError("AION QLoRA result and encoder checkpoint configurations differ.")
    state = checkpoint.get("trainable_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("AION QLoRA checkpoint has no trainable_state_dict.")
    state = {str(name): torch.as_tensor(value) for name, value in state.items()}
    required_scopes = ("aion.", "pooler.", "photoz_head.")
    missing = [scope for scope in required_scopes if not any(name.startswith(scope) for name in state)]
    if missing:
        raise ValueError(f"AION QLoRA checkpoint is incomplete; missing scopes: {missing}")
    return config, state, result, checkpoint_path


def create_aion_qlora_rlvr_model(
    source_result_path: str | Path,
    *,
    device: torch.device | str = "cuda",
) -> tuple[AIONEncoderPhotoZModel, AIONEmbeddingMethodConfig, dict[str, Any], Path]:
    """Restore the completed supervised QLoRA policy as a trainable model."""
    config, state, result, checkpoint_path = load_aion_qlora_source_artifact(
        source_result_path
    )
    resolved_device = resolve_torch_device(device)
    if resolved_device.type != "cuda":
        raise RuntimeError("AION QLoRA RLVR requires CUDA.")
    aion, _ = load_frozen_aion(device=resolved_device)
    model = build_aion_embedding_method_model(aion, config).to(resolved_device)
    expected = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise ValueError(
            "AION QLoRA source does not match the reconstructed policy: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    _load_trainable_state(model, state)
    model.freeze_adapter = False
    return model, config, result, checkpoint_path


@torch.no_grad()
def cache_aion_reference_log_probs(
    model: AIONEncoderPhotoZModel,
    dataset: AIONTokenRedshiftDataset,
    *,
    device: torch.device,
    batch_size: int,
    label: str,
) -> tuple[dict[str, int], torch.Tensor]:
    """Cache the fixed completed QLoRA policy used by the RLVR KL term."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    object_ids: list[str] = []
    parts: list[torch.Tensor] = []
    seen = 0
    for batch in loader:
        tokens = {key: value.to(device) for key, value in batch["tokens"].items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(tokens)
        object_ids.extend(str(value) for value in batch["object_id"])
        parts.append(torch.log_softmax(logits.float(), dim=-1).half().cpu())
        seen += int(logits.shape[0])
        if seen % 1000 == 0 or seen == len(dataset):
            print(
                f"AION QLoRA RLVR reference {label}: {seen:,}/{len(dataset):,}",
                flush=True,
            )
    if len(set(object_ids)) != len(object_ids):
        raise ValueError(f"AION RLVR {label} object IDs must be unique.")
    if not parts:
        raise ValueError(f"AION RLVR {label} cohort is empty.")
    return (
        {object_id: index for index, object_id in enumerate(object_ids)},
        torch.cat(parts),
    )


def _reference_rows(
    object_ids: list[str],
    index: dict[str, int],
    values: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    try:
        rows = [index[str(object_id)] for object_id in object_ids]
    except KeyError as error:
        raise ValueError(
            f"Object is absent from the AION QLoRA reference cache: {error}"
        ) from error
    return values[torch.as_tensor(rows, dtype=torch.long)].to(device).float()


def _checkpoint_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("aion-qlora-rlvr-checkpoint-update-*.pt"))


def _save_checkpoint(directory: Path, **state: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"aion-qlora-rlvr-checkpoint-update-{state['global_update']:07d}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"format_version": 1, **state}, temporary)
    temporary.replace(path)
    print(f"saved AION QLoRA RLVR checkpoint {path}", flush=True)
    return path


def train_aion_qlora_rlvr(
    *,
    train_dataset: AIONTokenRedshiftDataset,
    val_dataset: AIONTokenRedshiftDataset,
    test_dataset: AIONTokenRedshiftDataset,
    output_dir: str | Path,
    source_result_path: str | Path,
    config: RLVRConfig,
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 100,
    resume: bool = True,
) -> dict[str, Any]:
    """Continue a completed encoder-level AION QLoRA policy with RLVR."""
    config = config.normalized()
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive.")
    token_keys = set(train_dataset.tokens)
    if token_keys != set(val_dataset.tokens) or token_keys != set(test_dataset.tokens):
        raise ValueError("AION RLVR token modalities do not match across cohorts.")
    set_random_seed(config.seed)
    device = resolve_torch_device("cuda")
    model, supervised_config, _, source_checkpoint = create_aion_qlora_rlvr_model(
        source_result_path,
        device=device,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.eval_batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.eval_batch_size, shuffle=False
    )
    redshift_edges, redshift_centers = make_redshift_grid(
        supervised_config.z_min,
        supervised_config.z_max,
        supervised_config.n_z_bins,
    )
    redshift_edges = redshift_edges.to(device)
    redshift_centers = redshift_centers.to(device)

    reference_index, reference_values = cache_aion_reference_log_probs(
        model,
        train_dataset,
        device=device,
        batch_size=config.eval_batch_size,
        label="train",
    )

    head_parameters = list(model.pooler.parameters()) + list(model.photoz_head.parameters())
    adapter_parameters = model.adapter_parameters()
    if not adapter_parameters:
        raise RuntimeError("AION RLVR policy has no trainable QLoRA parameters.")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": head_parameters,
                "lr": config.head_learning_rate,
                "weight_decay": config.weight_decay,
            },
            {
                "params": adapter_parameters,
                "lr": config.adapter_learning_rate,
                "weight_decay": 0.0,
            },
        ]
    )
    batches_per_epoch = int(np.ceil(len(train_dataset) / config.batch_size))
    updates_per_epoch = int(
        np.ceil(batches_per_epoch / config.gradient_accumulation_steps)
    )
    total_updates = max(updates_per_epoch * config.epochs, 1)
    warmup_steps = int(total_updates * config.warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [
            lambda step: _linear_warmup_decay_lambda(
                step, warmup_steps=warmup_steps, total_steps=total_updates
            ),
            lambda step: _linear_warmup_decay_lambda(
                step, warmup_steps=warmup_steps, total_steps=total_updates
            ),
        ],
    )

    history: list[dict[str, float | int]] = []
    best_val_reward = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    global_update = 0
    start_epoch = 0
    start_batch_index = 0
    resumed_totals: dict[str, float] = {}
    resumed_count = 0
    cohort_fingerprint = _cohort_fingerprint(
        train_dataset, val_dataset, test_dataset
    )
    source_path = str(Path(source_result_path).expanduser().resolve())
    checkpoint_path = (
        None if checkpoint_dir is None else Path(checkpoint_dir).expanduser()
    )
    optimizer.zero_grad(set_to_none=True)
    if resume and checkpoint_path is not None:
        candidates = _checkpoint_files(checkpoint_path)
        if candidates:
            saved = torch.load(candidates[-1], map_location="cpu", weights_only=False)
            if saved.get("rlvr_config") != asdict(config):
                raise ValueError(f"RLVR configuration does not match: {candidates[-1]}")
            if saved.get("source_result_path") != source_path:
                raise ValueError(f"RLVR source policy does not match: {candidates[-1]}")
            if saved.get("cohort_fingerprint") != cohort_fingerprint:
                raise ValueError(f"RLVR cohort does not match: {candidates[-1]}")
            _load_trainable_state(model, saved["trainable_state"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            history = list(saved["history"])
            best_val_reward = float(saved["best_val_reward"])
            best_state = saved["best_state"]
            global_update = int(saved["global_update"])
            start_epoch = int(saved["epoch"])
            start_batch_index = int(saved["next_batch_index"])
            resumed_totals = dict(saved["epoch_totals"])
            resumed_count = int(saved["epoch_count"])
            _restore_rng_state(saved)
            print(
                f"resuming AION QLoRA RLVR from {candidates[-1]} at "
                f"{global_update:,}/{total_updates:,}",
                flush=True,
            )

    diagnostic_names = (
        "loss",
        "policy_loss",
        "kl",
        "entropy",
        "reward",
        "normalized_error",
        "sampled_outlier_fraction",
    )
    for epoch in range(start_epoch, config.epochs):
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed + epoch),
        )
        model.train()
        model.freeze_adapter = False
        totals = (
            {name: float(resumed_totals.get(name, 0.0)) for name in diagnostic_names}
            if epoch == start_epoch
            else {name: 0.0 for name in diagnostic_names}
        )
        count = resumed_count if epoch == start_epoch else 0
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < start_batch_index:
                continue
            tokens = {key: value.to(device) for key, value in batch["tokens"].items()}
            z_spec = batch["z_spec"].to(device)
            object_ids = [str(value) for value in batch["object_id"]]
            reference_log_probs = _reference_rows(
                object_ids,
                reference_index,
                reference_values,
                device=device,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(tokens)
            loss, diagnostics = rlvr_group_policy_loss(
                logits.float(),
                reference_log_probs,
                z_spec,
                redshift_centers,
                group_samples=config.group_samples,
                reward_scale=config.reward_scale,
                outlier_threshold=config.outlier_threshold,
                outlier_penalty=config.outlier_penalty,
                kl_beta=config.kl_beta,
                entropy_coefficient=config.entropy_coefficient,
            )
            (loss / config.gradient_accumulation_steps).backward()
            batch_count = int(z_spec.shape[0])
            totals["loss"] += float(loss.detach()) * batch_count
            for name, value in diagnostics.items():
                totals[name] += float(value) * batch_count
            count += batch_count
            is_update = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if not is_update:
                continue
            torch.nn.utils.clip_grad_norm_(
                head_parameters, config.head_max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                adapter_parameters, config.adapter_max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
            if global_update == 1 or global_update % 100 == 0:
                print(
                    f"AION QLoRA RLVR update {global_update:,}/{total_updates:,} "
                    f"epoch={epoch + 1}/{config.epochs} "
                    f"reward={float(diagnostics['reward']):.4f} "
                    f"kl={float(diagnostics['kl']):.4f}",
                    flush=True,
                )
            if checkpoint_path is not None and global_update % checkpoint_interval == 0:
                _save_checkpoint(
                    checkpoint_path,
                    rlvr_config=asdict(config),
                    source_result_path=source_path,
                    cohort_fingerprint=cohort_fingerprint,
                    epoch=epoch,
                    next_batch_index=batch_index + 1,
                    global_update=global_update,
                    history=history,
                    best_val_reward=best_val_reward,
                    best_state=best_state,
                    trainable_state=_trainable_state(model),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    epoch_totals=totals,
                    epoch_count=count,
                    **_rng_state(),
                )

        val_evaluation = evaluate_aion_token_model(
            model,
            val_loader,
            device=device,
            redshift_edges=redshift_edges,
            redshift_centers=redshift_centers,
        )
        val_metrics = summarize_pdf_metrics(val_evaluation)
        val_reward = expected_verifier_reward(
            val_evaluation,
            reward_scale=config.reward_scale,
            outlier_threshold=config.outlier_threshold,
            outlier_penalty=config.outlier_penalty,
        )
        row = {
            "epoch": epoch,
            **{name: value / max(count, 1) for name, value in totals.items()},
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "adapter_learning_rate": float(scheduler.get_last_lr()[1]),
            "val_expected_verifier_reward": val_reward,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"AION QLoRA RLVR epoch={epoch:03d} reward={row['reward']:.4f} "
            f"val_reward={val_reward:.4f} "
            f"val_loss={val_metrics['cross_entropy']:.4f} "
            f"val_nmad={val_metrics['nmad']:.4f}",
            flush=True,
        )
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            best_state = _trainable_state(model)
        start_batch_index = 0
        resumed_totals = {}
        resumed_count = 0

    if best_state is not None:
        _load_trainable_state(model, best_state)
    val_evaluation = evaluate_aion_token_model(
        model,
        val_loader,
        device=device,
        redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
    )
    test_evaluation = evaluate_aion_token_model(
        model,
        test_loader,
        device=device,
        redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
    )
    final_metrics = {
        "val": summarize_pdf_metrics(val_evaluation),
        "test": summarize_pdf_metrics(test_evaluation),
    }
    verifier_metrics = {
        "val_expected_reward": expected_verifier_reward(
            val_evaluation,
            reward_scale=config.reward_scale,
            outlier_threshold=config.outlier_threshold,
            outlier_penalty=config.outlier_penalty,
        ),
        "test_expected_reward": expected_verifier_reward(
            test_evaluation,
            reward_scale=config.reward_scale,
            outlier_threshold=config.outlier_threshold,
            outlier_penalty=config.outlier_penalty,
        ),
    }

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "qlora_aion_encoder_adapter_rlvr.pt"
    torch.save(
        {
            "trainable_state_dict": _trainable_state(model),
            "supervised_config": asdict(supervised_config),
            "rlvr_config": asdict(config),
        },
        model_path,
    )
    result = {
        "model_kind": "aion_qlora_rlvr_encoder_photoz",
        "history": history,
        "final_metrics": final_metrics,
        "verifier_metrics": verifier_metrics,
        "val_evaluation": val_evaluation,
        "test_evaluation": test_evaluation,
        "metadata": {
            "posttraining_method": "aion_encoder_qlora_sft_then_group_relative_rlvr",
            "adaptation_scope": "aion_encoder",
            "pooling": "single_query_cross_attention",
            "cached_representation": "aion_codec_tokens",
            "quantization": "bitsandbytes_nf4_double_quantization",
            "source_result_path": source_path,
            "source_checkpoint": str(source_checkpoint.resolve()),
            "reference_policy": "fixed_completed_aion_encoder_qlora_sft",
            "supervised_config": asdict(supervised_config),
            "rlvr_config": asdict(config),
            "cohort_fingerprint": cohort_fingerprint,
            **trainable_parameter_summary(model),
            "checkpoint": str(model_path),
        },
    }
    torch.save(result, output_path / "result.pt")
    return result
