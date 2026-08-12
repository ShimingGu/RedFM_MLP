from __future__ import annotations

"""RLVR continuation for a supervised residual adapter on cached AION vectors."""

import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import summarize_pdf_metrics
from .qwen_alternative_posttraining import (
    EmbeddingRedshiftDataset,
    ResidualEmbeddingAdapterConfig,
    ResidualEmbeddingPhotoZModel,
    evaluate_embedding_adapter,
)
from .qwen_posttraining import _linear_warmup_decay_lambda, trainable_parameter_summary
from .qwen_rlvr import (
    RLVRConfig,
    _restore_rng_state,
    _rng_state,
    expected_verifier_reward,
    rlvr_group_policy_loss,
)
from .utils import make_redshift_grid, resolve_torch_device, set_random_seed


def _cohort_fingerprint(*datasets: EmbeddingRedshiftDataset) -> str:
    digest = hashlib.sha256()
    for dataset in datasets:
        for object_id in dataset.object_ids:
            digest.update(object_id.encode())
            digest.update(b"\0")
        digest.update(np.ascontiguousarray(dataset.redshifts.numpy()).view(np.uint8))
    return digest.hexdigest()


def load_supervised_embedding_adapter(
    source_result_path: str | Path,
    *,
    device: torch.device | str,
) -> tuple[ResidualEmbeddingPhotoZModel, ResidualEmbeddingAdapterConfig, dict[str, Any]]:
    """Restore the completed supervised adapter policy used as the RLVR anchor."""
    source_path = Path(source_result_path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"Supervised AION adapter result not found: {source_path}")
    result = torch.load(source_path, map_location="cpu", weights_only=False)
    metadata = dict(result.get("metadata", {}))
    raw_config = dict(metadata.get("config", {}))
    if not raw_config:
        raise ValueError("Supervised adapter result does not contain its configuration.")
    config = ResidualEmbeddingAdapterConfig(**raw_config).normalized()
    embedding_dim = int(metadata["embedding_dim"])
    model = ResidualEmbeddingPhotoZModel(
        embedding_dim=embedding_dim,
        n_z_bins=config.n_z_bins,
        bottleneck_dim=config.bottleneck_dim,
        head_hidden_dim=config.head_hidden_dim,
        dropout=config.dropout,
    )
    checkpoint = Path(metadata.get("checkpoint", ""))
    if not checkpoint.is_file():
        checkpoint = source_path.parent / "residual_embedding_adapter.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Supervised AION adapter checkpoint not found: {checkpoint}")
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    return model.to(resolve_torch_device(device)), config, result


def _checkpoint_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("aion-rlvr-checkpoint-update-*.pt"))


def _save_checkpoint(directory: Path, **state: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"aion-rlvr-checkpoint-update-{state['global_update']:07d}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"format_version": 1, **state}, temporary)
    temporary.replace(path)
    print(f"saved AION RLVR checkpoint {path}", flush=True)
    return path


def train_embedding_adapter_rlvr(
    *,
    train_dataset: EmbeddingRedshiftDataset,
    val_dataset: EmbeddingRedshiftDataset,
    test_dataset: EmbeddingRedshiftDataset,
    output_dir: str | Path,
    source_result_path: str | Path,
    config: RLVRConfig,
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 100,
    resume: bool = True,
) -> dict[str, Any]:
    """Continue a supervised AION residual adapter with group-relative rewards."""
    config = config.normalized()
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive.")
    set_random_seed(config.seed)
    device = resolve_torch_device("cuda")
    if device.type != "cuda":
        raise RuntimeError("AION embedding RLVR requires CUDA.")
    model, supervised_config, _ = load_supervised_embedding_adapter(
        source_result_path,
        device=device,
    )
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False

    expected_dim = train_dataset.embeddings.shape[1]
    for name, dataset in (
        ("validation", val_dataset),
        ("test", test_dataset),
    ):
        if dataset.embeddings.shape[1] != expected_dim:
            raise ValueError(f"{name} embedding dimension does not match training.")
    if int(model.input_norm.normalized_shape[0]) != expected_dim:
        raise ValueError("Supervised adapter and dataset embedding dimensions do not match.")

    train_loader_length = int(np.ceil(len(train_dataset) / config.batch_size))
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
    )
    redshift_edges, redshift_centers = make_redshift_grid(
        supervised_config.z_min,
        supervised_config.z_max,
        supervised_config.n_z_bins,
    )
    redshift_edges = redshift_edges.to(device)
    redshift_centers = redshift_centers.to(device)

    head_parameters = list(model.photoz_head.parameters())
    adapter_parameters = list(model.input_norm.parameters()) + list(
        model.adapter.parameters()
    )
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
    updates_per_epoch = int(
        np.ceil(train_loader_length / config.gradient_accumulation_steps)
    )
    total_updates = max(updates_per_epoch * config.epochs, 1)
    warmup_steps = int(total_updates * config.warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [
            lambda step: _linear_warmup_decay_lambda(
                step,
                warmup_steps=warmup_steps,
                total_steps=total_updates,
            ),
            lambda step: _linear_warmup_decay_lambda(
                step,
                warmup_steps=warmup_steps,
                total_steps=total_updates,
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
        train_dataset,
        val_dataset,
        test_dataset,
    )
    checkpoint_path = (
        None if checkpoint_dir is None else Path(checkpoint_dir).expanduser()
    )
    source_path = str(Path(source_result_path).expanduser().resolve())
    if resume and checkpoint_path is not None:
        candidates = _checkpoint_files(checkpoint_path)
        if candidates:
            saved = torch.load(candidates[-1], map_location="cpu", weights_only=False)
            if saved.get("config") != asdict(config):
                raise ValueError(f"RLVR configuration does not match: {candidates[-1]}")
            if saved.get("source_result_path") != source_path:
                raise ValueError(f"RLVR source policy does not match: {candidates[-1]}")
            if saved.get("cohort_fingerprint") != cohort_fingerprint:
                raise ValueError(f"RLVR cohort does not match: {candidates[-1]}")
            model.load_state_dict(saved["model_state"])
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
                f"resuming AION RLVR from {candidates[-1]} at "
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
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, config.epochs):
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed + epoch),
        )
        model.train()
        totals = (
            {name: float(resumed_totals.get(name, 0.0)) for name in diagnostic_names}
            if epoch == start_epoch
            else {name: 0.0 for name in diagnostic_names}
        )
        count = resumed_count if epoch == start_epoch else 0
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < start_batch_index:
                continue
            embedding = batch["embedding"].to(device)
            z_spec = batch["z_spec"].to(device)
            with torch.no_grad(), torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                reference_logits = reference(embedding)
            reference_log_probs = torch.log_softmax(
                reference_logits.float(),
                dim=-1,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(embedding)
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
                head_parameters,
                config.head_max_grad_norm,
            )
            torch.nn.utils.clip_grad_norm_(
                adapter_parameters,
                config.adapter_max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
            if global_update == 1 or global_update % 100 == 0:
                print(
                    f"AION RLVR update {global_update:,}/{total_updates:,} "
                    f"epoch={epoch + 1}/{config.epochs} "
                    f"reward={float(diagnostics['reward']):.4f} "
                    f"kl={float(diagnostics['kl']):.4f}",
                    flush=True,
                )
            if checkpoint_path is not None and global_update % checkpoint_interval == 0:
                _save_checkpoint(
                    checkpoint_path,
                    config=asdict(config),
                    source_result_path=source_path,
                    cohort_fingerprint=cohort_fingerprint,
                    epoch=epoch,
                    next_batch_index=batch_index + 1,
                    global_update=global_update,
                    history=history,
                    best_val_reward=best_val_reward,
                    best_state=best_state,
                    model_state={
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    epoch_totals=totals,
                    epoch_count=count,
                    **_rng_state(),
                )

        val_evaluation = evaluate_embedding_adapter(
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
            f"AION RLVR epoch={epoch:03d} reward={row['reward']:.4f} "
            f"val_reward={val_reward:.4f} "
            f"val_loss={val_metrics['cross_entropy']:.4f} "
            f"val_nmad={val_metrics['nmad']:.4f}",
            flush=True,
        )
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        start_batch_index = 0
        resumed_totals = {}
        resumed_count = 0

    if best_state is not None:
        model.load_state_dict(best_state)
    val_evaluation = evaluate_embedding_adapter(
        model,
        val_loader,
        device=device,
        redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
    )
    test_evaluation = evaluate_embedding_adapter(
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
    model_path = output_path / "residual_embedding_adapter_rlvr.pt"
    torch.save(model.state_dict(), model_path)
    result = {
        "model_kind": "aion_residual_embedding_adapter_rlvr_photoz",
        "history": history,
        "final_metrics": final_metrics,
        "verifier_metrics": verifier_metrics,
        "val_evaluation": val_evaluation,
        "test_evaluation": test_evaluation,
        "metadata": {
            "posttraining_method": "aion_residual_adapter_sft_then_group_relative_rlvr",
            "pooling": "mean_encoder_tokens",
            "source_result_path": source_path,
            "reference_policy": "fixed_completed_aion_residual_adapter_sft",
            "embedding_dim": expected_dim,
            "supervised_config": asdict(supervised_config),
            "rlvr_config": asdict(config),
            "cohort_fingerprint": cohort_fingerprint,
            **trainable_parameter_summary(model),
            "checkpoint": str(model_path),
        },
    }
    torch.save(result, output_path / "result.pt")
    return result
