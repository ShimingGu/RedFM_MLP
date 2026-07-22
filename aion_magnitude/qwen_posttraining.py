from __future__ import annotations

"""Task-specific QLoRA post-training for Qwen photo-z representations."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .FM_Qwen import load_frozen_qwen, pool_qwen_hidden_states
from .metrics import (
    predict_photoz_from_logits,
    redshift_cross_entropy_loss,
    summarize_pdf_metrics,
)
from .models import PhotoZHead
from .utils import make_redshift_grid, resolve_torch_device, set_random_seed

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except ImportError:  # Keep non-post-training workflows importable without PEFT.
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


@dataclass(frozen=True)
class QwenPosttrainingConfig:
    """Configuration shared by frozen-probe and QLoRA photo-z arms."""

    model_path: str | Path = "Qwen3.5-4B-Base"
    max_length: int = 512
    pooling: str = "last"
    n_z_bins: int = 300
    z_min: float = 0.0
    z_max: float = 6.0
    head_hidden_dim: int = 256
    epochs: int = 3
    batch_size: int = 1
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.03
    max_grad_norm: float = 1.0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = (
        "q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,gate_proj,up_proj,down_proj"
    )
    seed: int = 42
    device: str | torch.device = "cuda"
    local_files_only: bool = True

    def normalized(self) -> "QwenPosttrainingConfig":
        if self.pooling != "last":
            raise ValueError(
                "Post-training comparison requires pooling='last' so both arms use "
                "the same final non-padding token."
            )
        integer_fields = {
            "max_length": self.max_length,
            "n_z_bins": self.n_z_bins,
            "head_hidden_dim": self.head_hidden_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
        }
        invalid = [name for name, value in integer_fields.items() if int(value) < 1]
        if invalid:
            raise ValueError(f"Post-training integer settings must be positive: {invalid}")
        if float(self.z_max) <= float(self.z_min):
            raise ValueError("z_max must exceed z_min.")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if not (0.0 <= float(self.warmup_fraction) < 1.0):
            raise ValueError("warmup_fraction must be in [0, 1).")
        return replace(
            self,
            model_path=str(self.model_path),
            max_length=int(self.max_length),
            n_z_bins=int(self.n_z_bins),
            z_min=float(self.z_min),
            z_max=float(self.z_max),
            head_hidden_dim=int(self.head_hidden_dim),
            epochs=int(self.epochs),
            batch_size=int(self.batch_size),
            eval_batch_size=int(self.eval_batch_size),
            gradient_accumulation_steps=int(self.gradient_accumulation_steps),
            learning_rate=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
            warmup_fraction=float(self.warmup_fraction),
            max_grad_norm=float(self.max_grad_norm),
            lora_rank=int(self.lora_rank),
            lora_alpha=int(self.lora_alpha),
            lora_dropout=float(self.lora_dropout),
            lora_target_modules=str(self.lora_target_modules),
            seed=int(self.seed),
            local_files_only=bool(self.local_files_only),
        )


class TextRedshiftDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        redshifts: torch.Tensor | np.ndarray,
        object_ids: Sequence[str],
    ):
        if len(texts) != len(redshifts) or len(texts) != len(object_ids):
            raise ValueError("texts, redshifts, and object_ids must have equal length.")
        self.texts = list(texts)
        self.redshifts = torch.as_tensor(redshifts, dtype=torch.float32)
        self.object_ids = [str(value) for value in object_ids]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "text": self.texts[index],
            "z_spec": self.redshifts[index],
            "object_id": self.object_ids[index],
        }


def make_text_collator(tokenizer, *, max_length: int):
    def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [item["text"] for item in items],
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        return {
            **encoded,
            "z_spec": torch.stack([item["z_spec"] for item in items]).float(),
            "object_id": [item["object_id"] for item in items],
        }

    return collate


def qwen_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(getattr(config, "text_config", None), "hidden_size", None)
    if hidden_size is None:
        base_config = getattr(getattr(model, "base_model", None), "config", None)
        hidden_size = getattr(getattr(base_config, "text_config", None), "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Could not determine Qwen hidden size from its configuration.")
    return int(hidden_size)


class QwenPhotoZModel(nn.Module):
    """Qwen final-token representation followed by a matched photo-z PDF head."""

    def __init__(
        self,
        qwen: nn.Module,
        *,
        hidden_size: int,
        n_z_bins: int,
        head_hidden_dim: int,
        pooling: str = "last",
    ):
        super().__init__()
        if pooling != "last":
            raise ValueError("QwenPhotoZModel currently requires last-token pooling.")
        self.qwen = qwen
        self.pooling = pooling
        self.photoz_head = PhotoZHead(hidden_size, n_z_bins, head_hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **model_inputs: torch.Tensor,
    ) -> torch.Tensor:
        output = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            **model_inputs,
        )
        embedding = pool_qwen_hidden_states(
            output.last_hidden_state,
            attention_mask,
            pooling=self.pooling,
        ).float()
        return self.photoz_head(embedding)


def require_peft() -> None:
    if any(
        value is None
        for value in (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    ):
        raise ImportError(
            "Qwen QLoRA post-training requires PEFT. Use the repository Pixi "
            "environment or install the qwen-cuda extra."
        )


def create_qlora_photoz_model(
    config: QwenPosttrainingConfig,
) -> tuple[QwenPhotoZModel, Any, torch.device]:
    """Load a 4-bit frozen base, attach trainable LoRA adapters, then add the head."""
    require_peft()
    config = config.normalized()
    device = resolve_torch_device(config.device)
    if device.type != "cuda":
        raise RuntimeError("QLoRA post-training requires a CUDA device.")
    base_model, tokenizer = load_frozen_qwen(
        config.model_path,
        device=device,
        load_in_4bit=True,
        torch_dtype="bf16",
        local_files_only=config.local_files_only,
        trust_remote_code=True,
    )
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
    )
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            name.strip() for name in config.lora_target_modules.split(",")
            if name.strip()
        ],
        bias="none",
    )
    qwen = get_peft_model(base_model, lora_config)
    hidden_size = qwen_hidden_size(qwen)
    model = QwenPhotoZModel(
        qwen,
        hidden_size=hidden_size,
        n_z_bins=config.n_z_bins,
        head_hidden_dim=config.head_hidden_dim,
        pooling=config.pooling,
    ).to(device)
    return model, tokenizer, device


def trainable_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_fraction": float(trainable / max(total, 1)),
    }


@torch.no_grad()
def evaluate_text_photoz_model(
    model: QwenPhotoZModel,
    loader: DataLoader,
    *,
    device: torch.device,
    redshift_edges: torch.Tensor,
    redshift_centers: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    z_parts: list[torch.Tensor] = []
    object_ids: list[str] = []
    for batch in loader:
        z_spec = batch.pop("z_spec")
        object_ids.extend(batch.pop("object_id"))
        inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**inputs)
        logits_parts.append(logits.detach().float().cpu())
        z_parts.append(z_spec.cpu())
    if not logits_parts:
        raise ValueError("No batches were available for post-training evaluation.")
    logits = torch.cat(logits_parts)
    z_spec = torch.cat(z_parts)
    evaluation: dict[str, Any] = {
        "logits": logits,
        "z_spec": z_spec,
        "object_id": object_ids,
        "redshift_edges": redshift_edges.detach().cpu(),
        "redshift_centers": redshift_centers.detach().cpu(),
        "loss": redshift_cross_entropy_loss(
            logits, z_spec, edges=redshift_edges.cpu()
        ).item(),
    }
    evaluation.update(predict_photoz_from_logits(logits, centers=redshift_centers.cpu()))
    return evaluation


def _linear_warmup_decay_lambda(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    remaining = max(total_steps - step, 0)
    decay_steps = max(total_steps - warmup_steps, 1)
    return float(remaining) / float(decay_steps)


def train_qlora_photoz(
    *,
    train_dataset: TextRedshiftDataset,
    val_dataset: TextRedshiftDataset,
    test_dataset: TextRedshiftDataset,
    output_dir: str | Path,
    config: QwenPosttrainingConfig,
) -> dict[str, Any]:
    """Jointly train QLoRA adapters and the photo-z head with binned CE."""
    config = config.normalized()
    set_random_seed(config.seed)
    model, tokenizer, device = create_qlora_photoz_model(config)
    collate = make_text_collator(tokenizer, max_length=config.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    redshift_edges, redshift_centers = make_redshift_grid(
        config.z_min, config.z_max, config.n_z_bins
    )
    redshift_edges = redshift_edges.to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    updates_per_epoch = int(
        np.ceil(len(train_loader) / config.gradient_accumulation_steps)
    )
    total_updates = max(updates_per_epoch * config.epochs, 1)
    warmup_steps = int(total_updates * config.warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _linear_warmup_decay_lambda(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_updates,
        ),
    )

    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_trainable_state: dict[str, torch.Tensor] | None = None
    optimizer.zero_grad(set_to_none=True)
    global_update = 0
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_index, batch in enumerate(train_loader):
            z_spec = batch.pop("z_spec").to(device)
            batch.pop("object_id")
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**inputs)
                loss = redshift_cross_entropy_loss(
                    logits.float(), z_spec, edges=redshift_edges
                )
            (loss / config.gradient_accumulation_steps).backward()
            total_loss += float(loss.detach()) * int(z_spec.shape[0])
            total_count += int(z_spec.shape[0])
            is_update = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if is_update:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                if global_update == 1 or global_update % 100 == 0:
                    print(
                        f"QLoRA update {global_update:,}/{total_updates:,} "
                        f"epoch={epoch + 1}/{config.epochs} loss={float(loss):.4f}",
                        flush=True,
                    )
        val_evaluation = evaluate_text_photoz_model(
            model,
            val_loader,
            device=device,
            redshift_edges=redshift_edges,
            redshift_centers=redshift_centers,
        )
        val_metrics = summarize_pdf_metrics(val_evaluation)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"QLoRA epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"val_loss={val_metrics['cross_entropy']:.4f} "
            f"val_nmad={val_metrics['nmad']:.4f}",
            flush=True,
        )
        if val_metrics["cross_entropy"] < best_val_loss:
            best_val_loss = val_metrics["cross_entropy"]
            best_trainable_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }

    if best_trainable_state is not None:
        for name, parameter in model.named_parameters():
            if name in best_trainable_state:
                parameter.data.copy_(
                    best_trainable_state[name].to(
                        device=parameter.device, dtype=parameter.dtype
                    )
                )

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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"
    model.qwen.save_pretrained(adapter_dir)
    torch.save(model.photoz_head.state_dict(), output_dir / "photoz_head.pt")
    result = {
        "model_kind": "qlora_photoz",
        "history": history,
        "final_metrics": {
            "val": summarize_pdf_metrics(val_evaluation),
            "test": summarize_pdf_metrics(test_evaluation),
        },
        "val_evaluation": val_evaluation,
        "test_evaluation": test_evaluation,
        "metadata": {
            "posttraining_method": "qlora_direct_photoz_cross_entropy",
            "pooling": config.pooling,
            "config": {
                key: str(value) if isinstance(value, (Path, torch.device)) else value
                for key, value in asdict(config).items()
            },
            **trainable_parameter_summary(model),
            "adapter_dir": str(adapter_dir),
            "head_checkpoint": str(output_dir / "photoz_head.pt"),
        },
    }
    torch.save(result, output_dir / "result.pt")
    return result
