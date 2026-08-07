from __future__ import annotations

"""RLVR continuation for a supervised QLoRA photo-z policy."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .FM_Qwen import load_frozen_qwen
from .metrics import summarize_pdf_metrics
from .qwen_posttraining import (
    QwenPhotoZModel,
    QwenPosttrainingConfig,
    TextRedshiftDataset,
    _cpu_byte_rng_state,
    _linear_warmup_decay_lambda,
    evaluate_text_photoz_model,
    make_text_collator,
    qwen_hidden_size,
    trainable_parameter_summary,
)
from .utils import make_redshift_grid, resolve_torch_device, set_random_seed

try:
    from peft import PeftModel, prepare_model_for_kbit_training
except ImportError:
    PeftModel = None
    prepare_model_for_kbit_training = None


@dataclass(frozen=True)
class RLVRConfig:
    epochs: int = 1
    batch_size: int = 1
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 16
    head_learning_rate: float = 1.0e-5
    adapter_learning_rate: float = 1.0e-6
    weight_decay: float = 0.01
    warmup_fraction: float = 0.03
    head_max_grad_norm: float = 1.0
    adapter_max_grad_norm: float = 0.1
    group_samples: int = 8
    reward_scale: float = 0.05
    outlier_threshold: float = 0.15
    outlier_penalty: float = 0.5
    kl_beta: float = 0.02
    entropy_coefficient: float = 0.001
    seed: int = 42

    def normalized(self) -> "RLVRConfig":
        integers = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "group_samples": self.group_samples,
        }
        invalid = [name for name, value in integers.items() if int(value) < 1]
        if invalid:
            raise ValueError(f"Positive RLVR integer settings required: {invalid}")
        positive = {
            "head_learning_rate": self.head_learning_rate,
            "adapter_learning_rate": self.adapter_learning_rate,
            "head_max_grad_norm": self.head_max_grad_norm,
            "adapter_max_grad_norm": self.adapter_max_grad_norm,
            "reward_scale": self.reward_scale,
            "outlier_threshold": self.outlier_threshold,
        }
        invalid = [name for name, value in positive.items() if float(value) <= 0]
        if invalid:
            raise ValueError(f"Positive RLVR settings required: {invalid}")
        nonnegative = {
            "weight_decay": self.weight_decay,
            "outlier_penalty": self.outlier_penalty,
            "kl_beta": self.kl_beta,
            "entropy_coefficient": self.entropy_coefficient,
        }
        invalid = [name for name, value in nonnegative.items() if float(value) < 0]
        if invalid:
            raise ValueError(f"Non-negative RLVR settings required: {invalid}")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1).")
        return replace(
            self,
            epochs=int(self.epochs),
            batch_size=int(self.batch_size),
            eval_batch_size=int(self.eval_batch_size),
            gradient_accumulation_steps=int(self.gradient_accumulation_steps),
            head_learning_rate=float(self.head_learning_rate),
            adapter_learning_rate=float(self.adapter_learning_rate),
            weight_decay=float(self.weight_decay),
            warmup_fraction=float(self.warmup_fraction),
            head_max_grad_norm=float(self.head_max_grad_norm),
            adapter_max_grad_norm=float(self.adapter_max_grad_norm),
            group_samples=int(self.group_samples),
            reward_scale=float(self.reward_scale),
            outlier_threshold=float(self.outlier_threshold),
            outlier_penalty=float(self.outlier_penalty),
            kl_beta=float(self.kl_beta),
            entropy_coefficient=float(self.entropy_coefficient),
            seed=int(self.seed),
        )


def verifier_rewards(
    sampled_bins: torch.Tensor,
    z_spec: torch.Tensor,
    redshift_centers: torch.Tensor,
    *,
    reward_scale: float,
    outlier_threshold: float,
    outlier_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic photo-z rewards and normalized errors."""
    sampled_redshift = redshift_centers.to(sampled_bins.device)[sampled_bins]
    normalized_error = torch.abs(sampled_redshift - z_spec[:, None]) / (
        1.0 + z_spec[:, None]
    )
    reward = torch.exp(-normalized_error / float(reward_scale))
    reward = reward - float(outlier_penalty) * (
        normalized_error > float(outlier_threshold)
    ).to(reward.dtype)
    return reward, normalized_error


def rlvr_group_policy_loss(
    logits: torch.Tensor,
    reference_log_probs: torch.Tensor,
    z_spec: torch.Tensor,
    redshift_centers: torch.Tensor,
    *,
    group_samples: int,
    reward_scale: float,
    outlier_threshold: float,
    outlier_penalty: float,
    kl_beta: float,
    entropy_coefficient: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """GRPO-style categorical policy loss with a fixed SFT KL anchor."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probabilities = log_probs.exp()
    sampled_bins = torch.multinomial(
        probabilities,
        num_samples=int(group_samples),
        replacement=True,
    )
    sampled_log_probs = log_probs.gather(1, sampled_bins)
    rewards, normalized_error = verifier_rewards(
        sampled_bins,
        z_spec.float(),
        redshift_centers,
        reward_scale=reward_scale,
        outlier_threshold=outlier_threshold,
        outlier_penalty=outlier_penalty,
    )
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    scale = rewards.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    advantages = centered / scale
    policy_loss = -(advantages.detach() * sampled_log_probs).mean()

    reference_log_probs = reference_log_probs.to(
        logits.device, dtype=log_probs.dtype
    )
    kl = (
        probabilities
        * (log_probs - reference_log_probs)
    ).sum(dim=-1).mean()
    entropy = -(probabilities * log_probs).sum(dim=-1).mean()
    loss = (
        policy_loss
        + float(kl_beta) * kl
        - float(entropy_coefficient) * entropy
    )
    diagnostics = {
        "policy_loss": policy_loss.detach(),
        "kl": kl.detach(),
        "entropy": entropy.detach(),
        "reward": rewards.mean().detach(),
        "normalized_error": normalized_error.mean().detach(),
        "sampled_outlier_fraction": (
            normalized_error > float(outlier_threshold)
        ).float().mean().detach(),
    }
    return loss, diagnostics


def expected_verifier_reward(
    evaluation: dict[str, Any],
    *,
    reward_scale: float,
    outlier_threshold: float,
    outlier_penalty: float,
) -> float:
    pz = torch.as_tensor(evaluation["pz"], dtype=torch.float32)
    z_spec = torch.as_tensor(evaluation["z_spec"], dtype=torch.float32)
    centers = torch.as_tensor(
        evaluation["redshift_centers"], dtype=torch.float32
    )
    normalized_error = torch.abs(
        centers[None, :] - z_spec[:, None]
    ) / (1.0 + z_spec[:, None])
    rewards = torch.exp(-normalized_error / float(reward_scale))
    rewards = rewards - float(outlier_penalty) * (
        normalized_error > float(outlier_threshold)
    ).float()
    return float((pz * rewards).sum(dim=-1).mean())


def create_rlvr_model(
    base_config: QwenPosttrainingConfig,
    *,
    source_adapter_dir: str | Path,
    source_head_path: str | Path,
) -> tuple[QwenPhotoZModel, Any, torch.device]:
    """Restore the completed supervised QLoRA policy as a trainable policy."""
    if PeftModel is None or prepare_model_for_kbit_training is None:
        raise ImportError("RLVR continuation requires PEFT.")
    base_config = base_config.normalized()
    device = resolve_torch_device(base_config.device)
    if device.type != "cuda":
        raise RuntimeError("RLVR continuation requires a CUDA device.")
    adapter_dir = Path(source_adapter_dir).expanduser()
    head_path = Path(source_head_path).expanduser()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Source QLoRA adapter not found: {adapter_dir}")
    if not head_path.is_file():
        raise FileNotFoundError(f"Source QLoRA head not found: {head_path}")

    base_model, tokenizer = load_frozen_qwen(
        base_config.model_path,
        device=device,
        load_in_4bit=True,
        torch_dtype="bf16",
        local_files_only=base_config.local_files_only,
        trust_remote_code=True,
    )
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
    )
    qwen = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=True,
    )
    model = QwenPhotoZModel(
        qwen,
        hidden_size=qwen_hidden_size(qwen),
        n_z_bins=base_config.n_z_bins,
        head_hidden_dim=base_config.head_hidden_dim,
        pooling=base_config.pooling,
    ).to(device)
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    model.photoz_head.load_state_dict(state)
    return model, tokenizer, device


@torch.no_grad()
def cache_reference_log_probs(
    model: QwenPhotoZModel,
    dataset: TextRedshiftDataset,
    *,
    tokenizer: Any,
    device: torch.device,
    max_length: int,
    batch_size: int,
    label: str,
) -> tuple[dict[str, int], torch.Tensor]:
    """Cache the fixed SFT policy used by the RLVR KL term."""
    collate = make_text_collator(tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    model.eval()
    object_ids: list[str] = []
    parts: list[torch.Tensor] = []
    seen = 0
    for batch in loader:
        batch.pop("z_spec")
        object_ids.extend(batch.pop("object_id"))
        inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**inputs)
        parts.append(torch.log_softmax(logits.float(), dim=-1).half().cpu())
        seen += int(logits.shape[0])
        if seen % 1000 == 0 or seen == len(dataset):
            print(f"RLVR reference {label}: {seen:,}/{len(dataset):,}", flush=True)
    if len(set(object_ids)) != len(object_ids):
        raise ValueError(f"RLVR {label} object IDs must be unique.")
    return {object_id: index for index, object_id in enumerate(object_ids)}, torch.cat(parts)


def _trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _load_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    missing = sorted(set(state) - set(parameters))
    if missing:
        raise ValueError(f"RLVR checkpoint contains unknown parameters: {missing}")
    for name, value in state.items():
        parameter = parameters[name]
        parameter.data.copy_(value.to(parameter.device, dtype=parameter.dtype))


def _rng_state() -> dict[str, Any]:
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state().cpu(),
        "cuda_rng_state": [state.cpu() for state in torch.cuda.get_rng_state_all()],
    }


def _restore_rng_state(saved: dict[str, Any]) -> None:
    random.setstate(saved["python_rng_state"])
    np.random.set_state(saved["numpy_rng_state"])
    torch.set_rng_state(
        _cpu_byte_rng_state(saved["torch_rng_state"], name="torch_rng_state")
    )
    torch.cuda.set_rng_state_all(
        [
            _cpu_byte_rng_state(state, name=f"cuda_rng_state[{index}]")
            for index, state in enumerate(saved["cuda_rng_state"])
        ]
    )


def _checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    return sorted(checkpoint_dir.glob("rlvr-checkpoint-update-*.pt"))


def _save_checkpoint(checkpoint_dir: Path, **state: Any) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"rlvr-checkpoint-update-{state['global_update']:07d}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"format_version": 1, **state}, temporary)
    temporary.replace(path)
    print(f"saved RLVR checkpoint {path}", flush=True)
    return path


def _reference_rows(
    object_ids: list[str],
    index: dict[str, int],
    values: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    try:
        rows = [index[object_id] for object_id in object_ids]
    except KeyError as error:
        raise ValueError(f"Object is absent from the SFT reference cache: {error}") from error
    return values[torch.as_tensor(rows, dtype=torch.long)].to(device).float()


def train_qlora_rlvr(
    *,
    train_dataset: TextRedshiftDataset,
    val_dataset: TextRedshiftDataset,
    test_dataset: TextRedshiftDataset,
    output_dir: str | Path,
    base_config: QwenPosttrainingConfig,
    rlvr_config: RLVRConfig,
    source_adapter_dir: str | Path,
    source_head_path: str | Path,
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 100,
    resume: bool = True,
) -> dict[str, Any]:
    """Continue a supervised QLoRA policy with verifiable photo-z rewards."""
    base_config = base_config.normalized()
    rlvr_config = rlvr_config.normalized()
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive.")
    set_random_seed(rlvr_config.seed)
    model, tokenizer, device = create_rlvr_model(
        base_config,
        source_adapter_dir=source_adapter_dir,
        source_head_path=source_head_path,
    )
    collate = make_text_collator(tokenizer, max_length=base_config.max_length)
    val_loader = DataLoader(
        val_dataset,
        batch_size=rlvr_config.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=rlvr_config.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    redshift_edges, redshift_centers = make_redshift_grid(
        base_config.z_min, base_config.z_max, base_config.n_z_bins
    )
    redshift_edges = redshift_edges.to(device)
    redshift_centers = redshift_centers.to(device)

    reference_train_index, reference_train = cache_reference_log_probs(
        model,
        train_dataset,
        tokenizer=tokenizer,
        device=device,
        max_length=base_config.max_length,
        batch_size=rlvr_config.eval_batch_size,
        label="train",
    )

    head_parameters = [
        parameter for parameter in model.photoz_head.parameters()
        if parameter.requires_grad
    ]
    adapter_parameters = [
        parameter for parameter in model.qwen.parameters()
        if parameter.requires_grad
    ]
    if not adapter_parameters:
        raise RuntimeError("RLVR policy has no trainable QLoRA parameters.")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": head_parameters,
                "lr": rlvr_config.head_learning_rate,
                "weight_decay": rlvr_config.weight_decay,
            },
            {
                "params": adapter_parameters,
                "lr": rlvr_config.adapter_learning_rate,
                "weight_decay": 0.0,
            },
        ]
    )
    batches_per_epoch = int(np.ceil(len(train_dataset) / rlvr_config.batch_size))
    updates_per_epoch = int(
        np.ceil(batches_per_epoch / rlvr_config.gradient_accumulation_steps)
    )
    total_updates = max(updates_per_epoch * rlvr_config.epochs, 1)
    warmup_steps = int(total_updates * rlvr_config.warmup_fraction)
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
    optimizer.zero_grad(set_to_none=True)
    checkpoint_path = Path(checkpoint_dir).expanduser() if checkpoint_dir else None

    if resume and checkpoint_path is not None:
        candidates = _checkpoint_files(checkpoint_path)
        if candidates:
            saved = torch.load(candidates[-1], map_location="cpu", weights_only=False)
            if saved.get("base_config") != asdict(base_config):
                raise ValueError(f"RLVR base configuration does not match: {candidates[-1]}")
            if saved.get("rlvr_config") != asdict(rlvr_config):
                raise ValueError(f"RLVR configuration does not match: {candidates[-1]}")
            expected_sizes = (len(train_dataset), len(val_dataset), len(test_dataset))
            if tuple(saved.get("dataset_sizes", ())) != expected_sizes:
                raise ValueError(f"RLVR dataset sizes do not match: {candidates[-1]}")
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
                f"resuming RLVR from {candidates[-1]} at "
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
    for epoch in range(start_epoch, rlvr_config.epochs):
        generator = torch.Generator().manual_seed(rlvr_config.seed + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=rlvr_config.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate,
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
            z_spec = batch.pop("z_spec").to(device)
            object_ids = batch.pop("object_id")
            reference_log_probs = _reference_rows(
                object_ids,
                reference_train_index,
                reference_train,
                device=device,
            )
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**inputs)
            loss, diagnostics = rlvr_group_policy_loss(
                logits.float(),
                reference_log_probs,
                z_spec,
                redshift_centers,
                group_samples=rlvr_config.group_samples,
                reward_scale=rlvr_config.reward_scale,
                outlier_threshold=rlvr_config.outlier_threshold,
                outlier_penalty=rlvr_config.outlier_penalty,
                kl_beta=rlvr_config.kl_beta,
                entropy_coefficient=rlvr_config.entropy_coefficient,
            )
            (loss / rlvr_config.gradient_accumulation_steps).backward()
            batch_count = int(z_spec.shape[0])
            totals["loss"] += float(loss.detach()) * batch_count
            for name, value in diagnostics.items():
                totals[name] += float(value) * batch_count
            count += batch_count
            is_update = (
                (batch_index + 1) % rlvr_config.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if not is_update:
                continue
            torch.nn.utils.clip_grad_norm_(
                head_parameters, rlvr_config.head_max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                adapter_parameters, rlvr_config.adapter_max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
            if global_update == 1 or global_update % 100 == 0:
                print(
                    f"RLVR update {global_update:,}/{total_updates:,} "
                    f"epoch={epoch + 1}/{rlvr_config.epochs} "
                    f"reward={float(diagnostics['reward']):.4f} "
                    f"kl={float(diagnostics['kl']):.4f} "
                    f"head_lr={scheduler.get_last_lr()[0]:.3g} "
                    f"adapter_lr={scheduler.get_last_lr()[1]:.3g}",
                    flush=True,
                )
            if checkpoint_path is not None and global_update % checkpoint_interval == 0:
                _save_checkpoint(
                    checkpoint_path,
                    base_config=asdict(base_config),
                    rlvr_config=asdict(rlvr_config),
                    source_adapter_dir=str(Path(source_adapter_dir).expanduser()),
                    source_head_path=str(Path(source_head_path).expanduser()),
                    dataset_sizes=(len(train_dataset), len(val_dataset), len(test_dataset)),
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

        val_evaluation = evaluate_text_photoz_model(
            model,
            val_loader,
            device=device,
            redshift_edges=redshift_edges,
            redshift_centers=redshift_centers,
        )
        val_metrics = summarize_pdf_metrics(val_evaluation)
        val_reward = expected_verifier_reward(
            val_evaluation,
            reward_scale=rlvr_config.reward_scale,
            outlier_threshold=rlvr_config.outlier_threshold,
            outlier_penalty=rlvr_config.outlier_penalty,
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
            f"RLVR epoch={epoch:03d} reward={row['reward']:.4f} "
            f"val_reward={val_reward:.4f} val_loss={val_metrics['cross_entropy']:.4f} "
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
    val_evaluation = evaluate_text_photoz_model(
        model,
        val_loader,
        device=device,
        redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
    )
    test_evaluation = evaluate_text_photoz_model(
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
            reward_scale=rlvr_config.reward_scale,
            outlier_threshold=rlvr_config.outlier_threshold,
            outlier_penalty=rlvr_config.outlier_penalty,
        ),
        "test_expected_reward": expected_verifier_reward(
            test_evaluation,
            reward_scale=rlvr_config.reward_scale,
            outlier_threshold=rlvr_config.outlier_threshold,
            outlier_penalty=rlvr_config.outlier_penalty,
        ),
    }

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_path / "adapter"
    model.qwen.save_pretrained(adapter_dir)
    head_path = output_path / "photoz_head.pt"
    torch.save(model.photoz_head.state_dict(), head_path)
    result = {
        "model_kind": "qlora_rlvr_photoz",
        "history": history,
        "final_metrics": final_metrics,
        "verifier_metrics": verifier_metrics,
        "val_evaluation": val_evaluation,
        "test_evaluation": test_evaluation,
        "metadata": {
            "posttraining_method": "qlora_sft_then_group_relative_rlvr",
            "pooling": base_config.pooling,
            "base_config": asdict(base_config),
            "rlvr_config": asdict(rlvr_config),
            "source_adapter_dir": str(Path(source_adapter_dir).expanduser()),
            "source_head_path": str(Path(source_head_path).expanduser()),
            "reference_policy": "fixed_completed_qlora_sft",
            **trainable_parameter_summary(model),
            "adapter_dir": str(adapter_dir),
            "head_checkpoint": str(head_path),
        },
    }
    torch.save(result, output_path / "result.pt")
    return result
