from __future__ import annotations

"""Alternative Qwen post-training methods for photo-z inference."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .FM_Qwen import load_frozen_qwen
from .metrics import (
    predict_photoz_from_logits,
    redshift_cross_entropy_loss,
    summarize_pdf_metrics,
)
from .models import PhotoZHead
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
    from peft import (
        IA3Config,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except ImportError:
    IA3Config = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


def _comma_names(value: str, *, setting: str) -> list[str]:
    names = [name.strip() for name in str(value).split(",") if name.strip()]
    if not names:
        raise ValueError(f"{setting} must contain at least one module name.")
    return names


def create_ia3_photoz_model(
    config: QwenPosttrainingConfig,
    *,
    target_modules: str = "k_proj,v_proj,down_proj",
    feedforward_modules: str = "down_proj",
) -> tuple[QwenPhotoZModel, Any, torch.device]:
    """Load 4-bit Qwen, attach IA3 vectors, and add the matched PDF head."""
    if any(
        value is None
        for value in (
            IA3Config,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    ):
        raise ImportError("IA3 post-training requires PEFT.")
    config = config.normalized()
    device = resolve_torch_device(config.device)
    if device.type != "cuda":
        raise RuntimeError("IA3 post-training requires a CUDA device.")

    targets = _comma_names(target_modules, setting="target_modules")
    feedforward = _comma_names(
        feedforward_modules, setting="feedforward_modules"
    )
    if not set(feedforward).issubset(targets):
        raise ValueError("feedforward_modules must be a subset of target_modules.")

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
    adapter_config = IA3Config(
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=targets,
        feedforward_modules=feedforward,
    )
    qwen = get_peft_model(base_model, adapter_config)
    model = QwenPhotoZModel(
        qwen,
        hidden_size=qwen_hidden_size(qwen),
        n_z_bins=config.n_z_bins,
        head_hidden_dim=config.head_hidden_dim,
        pooling=config.pooling,
    ).to(device)
    return model, tokenizer, device


def _trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _load_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    unknown = sorted(set(state) - set(parameters))
    if unknown:
        raise ValueError(f"Checkpoint contains unknown parameters: {unknown}")
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


def _ia3_checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    return sorted(checkpoint_dir.glob("ia3-checkpoint-update-*.pt"))


def _save_ia3_checkpoint(checkpoint_dir: Path, **state: Any) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"ia3-checkpoint-update-{state['global_update']:07d}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"format_version": 1, "adapter_method": "ia3", **state}, temporary)
    temporary.replace(path)
    print(f"saved IA3 checkpoint {path}", flush=True)
    return path


def train_ia3_photoz(
    *,
    train_dataset: TextRedshiftDataset,
    val_dataset: TextRedshiftDataset,
    test_dataset: TextRedshiftDataset,
    output_dir: str | Path,
    config: QwenPosttrainingConfig,
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 100,
    resume: bool = True,
    target_modules: str = "k_proj,v_proj,down_proj",
    feedforward_modules: str = "down_proj",
) -> dict[str, Any]:
    """Train IA3 vectors and the photo-z head with binned cross entropy."""
    config = config.normalized()
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive.")
    target_names = _comma_names(target_modules, setting="target_modules")
    feedforward_names = _comma_names(
        feedforward_modules, setting="feedforward_modules"
    )
    set_random_seed(config.seed)
    model, tokenizer, device = create_ia3_photoz_model(
        config,
        target_modules=",".join(target_names),
        feedforward_modules=",".join(feedforward_names),
    )
    collate = make_text_collator(tokenizer, max_length=config.max_length)
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

    head_parameters = [
        parameter for parameter in model.photoz_head.parameters()
        if parameter.requires_grad
    ]
    adapter_parameters = [
        parameter for parameter in model.qwen.parameters()
        if parameter.requires_grad
    ]
    if not adapter_parameters:
        raise RuntimeError("IA3 model has no trainable adapter parameters.")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": head_parameters,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
            {
                "params": adapter_parameters,
                "lr": config.lora_learning_rate,
                "weight_decay": 0.0,
            },
        ]
    )
    batches_per_epoch = int(np.ceil(len(train_dataset) / config.batch_size))
    updates_per_epoch = int(
        np.ceil(batches_per_epoch / config.gradient_accumulation_steps)
    )
    total_updates = max(updates_per_epoch * config.epochs, 1)
    head_warmup_steps = int(total_updates * config.warmup_fraction)
    adapter_start_update = updates_per_epoch * config.head_warmup_epochs
    adapter_total_updates = max(total_updates - adapter_start_update, 1)
    adapter_warmup_steps = int(adapter_total_updates * config.warmup_fraction)

    def adapter_schedule(step: int) -> float:
        if step < adapter_start_update:
            return 0.0
        return _linear_warmup_decay_lambda(
            step - adapter_start_update,
            warmup_steps=adapter_warmup_steps,
            total_steps=adapter_total_updates,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [
            lambda step: _linear_warmup_decay_lambda(
                step,
                warmup_steps=head_warmup_steps,
                total_steps=total_updates,
            ),
            adapter_schedule,
        ],
    )

    history: list[dict[str, float | int | str]] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    global_update = 0
    start_epoch = 0
    start_batch_index = 0
    resumed_total_loss = 0.0
    resumed_total_count = 0
    optimizer.zero_grad(set_to_none=True)

    checkpoint_path = Path(checkpoint_dir).expanduser() if checkpoint_dir else None
    if resume and checkpoint_path is not None:
        candidates = _ia3_checkpoint_files(checkpoint_path)
        if candidates:
            saved = torch.load(candidates[-1], map_location="cpu", weights_only=False)
            if saved.get("config") != asdict(config):
                raise ValueError(f"IA3 checkpoint configuration does not match: {candidates[-1]}")
            if saved.get("target_modules") != target_names:
                raise ValueError(f"IA3 target modules do not match: {candidates[-1]}")
            if saved.get("feedforward_modules") != feedforward_names:
                raise ValueError(f"IA3 feedforward modules do not match: {candidates[-1]}")
            expected_sizes = (len(train_dataset), len(val_dataset), len(test_dataset))
            if tuple(saved.get("dataset_sizes", ())) != expected_sizes:
                raise ValueError(f"IA3 checkpoint dataset sizes do not match: {candidates[-1]}")
            _load_trainable_state(model, saved["trainable_state"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            start_epoch = int(saved["epoch"])
            start_batch_index = int(saved["next_batch_index"])
            global_update = int(saved["global_update"])
            history = list(saved["history"])
            best_val_loss = float(saved["best_val_loss"])
            best_state = saved["best_state"]
            resumed_total_loss = float(saved["epoch_total_loss"])
            resumed_total_count = int(saved["epoch_total_count"])
            _restore_rng_state(saved)
            print(
                f"resuming IA3 from {candidates[-1]} at "
                f"{global_update:,}/{total_updates:,}",
                flush=True,
            )

    for epoch in range(start_epoch, config.epochs):
        generator = torch.Generator().manual_seed(config.seed + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate,
        )
        model.train()
        model.freeze_qwen_representation = epoch < config.head_warmup_epochs
        phase = "head_warmup" if model.freeze_qwen_representation else "joint_ia3"
        if epoch == start_epoch or epoch == config.head_warmup_epochs:
            print(f"IA3 training phase: {phase}", flush=True)
        total_loss = resumed_total_loss if epoch == start_epoch else 0.0
        total_count = resumed_total_count if epoch == start_epoch else 0

        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < start_batch_index:
                continue
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
            if not is_update:
                continue
            torch.nn.utils.clip_grad_norm_(head_parameters, config.max_grad_norm)
            if not model.freeze_qwen_representation:
                torch.nn.utils.clip_grad_norm_(
                    adapter_parameters, config.lora_max_grad_norm
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
            if global_update == 1 or global_update % 100 == 0:
                print(
                    f"IA3 update {global_update:,}/{total_updates:,} "
                    f"epoch={epoch + 1}/{config.epochs} phase={phase} "
                    f"loss={float(loss.detach()):.4f} "
                    f"head_lr={scheduler.get_last_lr()[0]:.3g} "
                    f"ia3_lr={scheduler.get_last_lr()[1]:.3g}",
                    flush=True,
                )
            if checkpoint_path is not None and global_update % checkpoint_interval == 0:
                _save_ia3_checkpoint(
                    checkpoint_path,
                    config=asdict(config),
                    target_modules=target_names,
                    feedforward_modules=feedforward_names,
                    dataset_sizes=(len(train_dataset), len(val_dataset), len(test_dataset)),
                    epoch=epoch,
                    next_batch_index=batch_index + 1,
                    global_update=global_update,
                    history=history,
                    best_val_loss=best_val_loss,
                    best_state=best_state,
                    trainable_state=_trainable_state(model),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    epoch_total_loss=total_loss,
                    epoch_total_count=total_count,
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
        row = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": total_loss / max(total_count, 1),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "ia3_learning_rate": float(scheduler.get_last_lr()[1]),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"IA3 epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"val_loss={val_metrics['cross_entropy']:.4f} "
            f"val_nmad={val_metrics['nmad']:.4f}",
            flush=True,
        )
        if val_metrics["cross_entropy"] < best_val_loss:
            best_val_loss = val_metrics["cross_entropy"]
            best_state = _trainable_state(model)
        start_batch_index = 0
        resumed_total_loss = 0.0
        resumed_total_count = 0

    model.freeze_qwen_representation = False
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

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_path / "adapter"
    model.qwen.save_pretrained(adapter_dir)
    head_path = output_path / "photoz_head.pt"
    torch.save(model.photoz_head.state_dict(), head_path)
    result = {
        "model_kind": "ia3_photoz",
        "history": history,
        "final_metrics": {
            "val": summarize_pdf_metrics(val_evaluation),
            "test": summarize_pdf_metrics(test_evaluation),
        },
        "val_evaluation": val_evaluation,
        "test_evaluation": test_evaluation,
        "metadata": {
            "posttraining_method": "ia3_direct_photoz_cross_entropy",
            "pooling": config.pooling,
            "target_modules": target_names,
            "feedforward_modules": feedforward_names,
            "config": asdict(config),
            **trainable_parameter_summary(model),
            "adapter_dir": str(adapter_dir),
            "head_checkpoint": str(head_path),
        },
    }
    torch.save(result, output_path / "result.pt")
    return result


@dataclass(frozen=True)
class ResidualEmbeddingAdapterConfig:
    n_z_bins: int = 300
    z_min: float = 0.0
    z_max: float = 6.0
    bottleneck_dim: int = 256
    head_hidden_dim: int = 256
    epochs: int = 10
    batch_size: int = 16
    eval_batch_size: int = 8
    learning_rate: float = 2.0e-4
    adapter_learning_rate: float = 1.0e-5
    weight_decay: float = 0.01
    dropout: float = 0.05
    warmup_fraction: float = 0.03
    max_grad_norm: float = 1.0
    adapter_max_grad_norm: float = 0.1
    head_warmup_epochs: int = 3
    seed: int = 42
    device: str | torch.device = "cuda"

    def normalized(self) -> "ResidualEmbeddingAdapterConfig":
        integer_fields = {
            "n_z_bins": self.n_z_bins,
            "bottleneck_dim": self.bottleneck_dim,
            "head_hidden_dim": self.head_hidden_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
        }
        invalid = [name for name, value in integer_fields.items() if int(value) < 1]
        if invalid:
            raise ValueError(f"Positive settings required: {invalid}")
        if self.z_max <= self.z_min:
            raise ValueError("z_max must exceed z_min.")
        if (
            self.learning_rate <= 0
            or self.adapter_learning_rate <= 0
            or self.max_grad_norm <= 0
            or self.adapter_max_grad_norm <= 0
        ):
            raise ValueError("Learning rates and gradient norms must be positive.")
        if not 0 <= self.head_warmup_epochs < self.epochs:
            raise ValueError(
                "head_warmup_epochs must be non-negative and smaller than epochs."
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1).")
        return replace(
            self,
            n_z_bins=int(self.n_z_bins),
            z_min=float(self.z_min),
            z_max=float(self.z_max),
            bottleneck_dim=int(self.bottleneck_dim),
            head_hidden_dim=int(self.head_hidden_dim),
            epochs=int(self.epochs),
            batch_size=int(self.batch_size),
            eval_batch_size=int(self.eval_batch_size),
            learning_rate=float(self.learning_rate),
            adapter_learning_rate=float(self.adapter_learning_rate),
            weight_decay=float(self.weight_decay),
            dropout=float(self.dropout),
            warmup_fraction=float(self.warmup_fraction),
            max_grad_norm=float(self.max_grad_norm),
            adapter_max_grad_norm=float(self.adapter_max_grad_norm),
            head_warmup_epochs=int(self.head_warmup_epochs),
            seed=int(self.seed),
        )


class ResidualEmbeddingPhotoZModel(nn.Module):
    """Layer-normalized cached embedding, residual bottleneck, and PDF head."""

    def __init__(
        self,
        embedding_dim: int,
        n_z_bins: int,
        bottleneck_dim: int = 256,
        head_hidden_dim: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, embedding_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.photoz_head = PhotoZHead(
            embedding_dim,
            n_z_bins=n_z_bins,
            hidden_dim=head_hidden_dim,
        )
        self.freeze_adapter = False

    def adapted_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(embedding.float())
        adapted = normalized + self.adapter(normalized)
        if self.freeze_adapter:
            adapted = adapted.detach()
        return adapted

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.photoz_head(self.adapted_embedding(embedding))


class EmbeddingRedshiftDataset(Dataset):
    def __init__(
        self,
        embeddings: torch.Tensor,
        redshifts: torch.Tensor | np.ndarray,
        object_ids: Sequence[str],
    ):
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        self.redshifts = torch.as_tensor(redshifts, dtype=torch.float32)
        self.object_ids = [str(value) for value in object_ids]
        if self.embeddings.ndim != 2 or self.embeddings.shape[1] < 1:
            raise ValueError("embeddings must be a non-empty two-dimensional tensor.")
        if len(self.embeddings) != len(self.redshifts) or len(self.embeddings) != len(self.object_ids):
            raise ValueError("embeddings, redshifts, and object_ids must have equal length.")

    def __len__(self) -> int:
        return len(self.redshifts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "embedding": self.embeddings[index],
            "z_spec": self.redshifts[index],
            "object_id": self.object_ids[index],
        }


@torch.no_grad()
def evaluate_embedding_adapter(
    model: ResidualEmbeddingPhotoZModel,
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
        embedding = batch["embedding"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(embedding)
        logits_parts.append(logits.float().cpu())
        z_parts.append(batch["z_spec"].float().cpu())
        object_ids.extend(batch["object_id"])
    if not logits_parts:
        raise ValueError("No batches were available for embedding-adapter evaluation.")
    logits = torch.cat(logits_parts)
    z_spec = torch.cat(z_parts)
    evaluation: dict[str, Any] = {
        "logits": logits,
        "z_spec": z_spec,
        "object_id": object_ids,
        "redshift_edges": redshift_edges.cpu(),
        "redshift_centers": redshift_centers.cpu(),
        "loss": redshift_cross_entropy_loss(
            logits, z_spec, edges=redshift_edges.cpu()
        ).item(),
    }
    evaluation.update(
        predict_photoz_from_logits(logits, centers=redshift_centers.cpu())
    )
    return evaluation


def train_residual_embedding_adapter(
    *,
    train_dataset: EmbeddingRedshiftDataset,
    val_dataset: EmbeddingRedshiftDataset,
    test_dataset: EmbeddingRedshiftDataset,
    output_dir: str | Path,
    config: ResidualEmbeddingAdapterConfig,
) -> dict[str, Any]:
    """Train a residual bottleneck and PDF head on cached frozen Qwen vectors."""
    config = config.normalized()
    if train_dataset.embeddings.shape[1] != val_dataset.embeddings.shape[1]:
        raise ValueError("Train and validation embedding dimensions do not match.")
    if train_dataset.embeddings.shape[1] != test_dataset.embeddings.shape[1]:
        raise ValueError("Train and test embedding dimensions do not match.")
    set_random_seed(config.seed)
    device = resolve_torch_device(config.device)
    if device.type != "cuda":
        raise RuntimeError("Residual embedding-adapter training requires CUDA.")

    model = ResidualEmbeddingPhotoZModel(
        train_dataset.embeddings.shape[1],
        config.n_z_bins,
        bottleneck_dim=config.bottleneck_dim,
        head_hidden_dim=config.head_hidden_dim,
        dropout=config.dropout,
    ).to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
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
        config.z_min, config.z_max, config.n_z_bins
    )
    redshift_edges = redshift_edges.to(device)
    head_parameters = list(model.photoz_head.parameters())
    adapter_parameters = list(model.input_norm.parameters()) + list(
        model.adapter.parameters()
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": head_parameters,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
            {
                "params": adapter_parameters,
                "lr": config.adapter_learning_rate,
                "weight_decay": 0.0,
            },
        ]
    )
    updates_per_epoch = len(train_loader)
    total_updates = max(updates_per_epoch * config.epochs, 1)
    head_warmup_steps = int(total_updates * config.warmup_fraction)
    adapter_start_update = updates_per_epoch * config.head_warmup_epochs
    adapter_total_updates = max(total_updates - adapter_start_update, 1)
    adapter_warmup_steps = int(adapter_total_updates * config.warmup_fraction)

    def adapter_schedule(step: int) -> float:
        if step < adapter_start_update:
            return 0.0
        return _linear_warmup_decay_lambda(
            step - adapter_start_update,
            warmup_steps=adapter_warmup_steps,
            total_steps=adapter_total_updates,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [
            lambda step: _linear_warmup_decay_lambda(
                step,
                warmup_steps=head_warmup_steps,
                total_steps=total_updates,
            ),
            adapter_schedule,
        ],
    )

    history: list[dict[str, float | int | str]] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(config.epochs):
        model.train()
        model.freeze_adapter = epoch < config.head_warmup_epochs
        phase = "head_warmup" if model.freeze_adapter else "joint_embedding_adapter"
        if epoch == 0 or epoch == config.head_warmup_epochs:
            print(f"embedding-adapter training phase: {phase}", flush=True)
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            embedding = batch["embedding"].to(device)
            z_spec = batch["z_spec"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(embedding)
                loss = redshift_cross_entropy_loss(
                    logits.float(), z_spec, edges=redshift_edges
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_parameters, config.max_grad_norm)
            if not model.freeze_adapter:
                torch.nn.utils.clip_grad_norm_(
                    adapter_parameters, config.adapter_max_grad_norm
                )
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach()) * int(z_spec.shape[0])
            total_count += int(z_spec.shape[0])

        val_evaluation = evaluate_embedding_adapter(
            model,
            val_loader,
            device=device,
            redshift_edges=redshift_edges,
            redshift_centers=redshift_centers,
        )
        val_metrics = summarize_pdf_metrics(val_evaluation)
        row = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": total_loss / max(total_count, 1),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "adapter_learning_rate": float(scheduler.get_last_lr()[1]),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"embedding-adapter epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.4f} "
            f"val_loss={val_metrics['cross_entropy']:.4f} "
            f"val_nmad={val_metrics['nmad']:.4f}",
            flush=True,
        )
        if val_metrics["cross_entropy"] < best_val_loss:
            best_val_loss = val_metrics["cross_entropy"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    model.freeze_adapter = False
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

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "residual_embedding_adapter.pt"
    torch.save(model.state_dict(), checkpoint_path)
    result = {
        "model_kind": "residual_embedding_adapter_photoz",
        "history": history,
        "final_metrics": {
            "val": summarize_pdf_metrics(val_evaluation),
            "test": summarize_pdf_metrics(test_evaluation),
        },
        "val_evaluation": val_evaluation,
        "test_evaluation": test_evaluation,
        "metadata": {
            "posttraining_method": "residual_embedding_adapter_cross_entropy",
            "pooling": "last",
            "embedding_dim": int(train_dataset.embeddings.shape[1]),
            "config": asdict(config),
            **trainable_parameter_summary(model),
            "checkpoint": str(checkpoint_path),
        },
    }
    torch.save(result, output_path / "result.pt")
    return result
