from __future__ import annotations

"""Encoder-level IA3, DoRA, and QLoRA adaptation for AION.

Unlike the older cached-vector controls, these methods modify the linear layers
inside AION's transformer encoder. Codec outputs are cached, while every train
step recomputes the encoder sequence and applies learned attentive pooling.
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
from torch.utils.data import DataLoader, Dataset

from .metrics import (
    predict_photoz_from_logits,
    redshift_cross_entropy_loss,
    summarize_pdf_metrics,
)
from .models import PhotoZHead, encode_aion_tokens, load_frozen_aion
from .qwen_posttraining import _linear_warmup_decay_lambda, trainable_parameter_summary
from .utils import make_redshift_grid, resolve_torch_device, set_random_seed


AION_ENCODER_METHODS = ("frozen", "ia3", "dora", "qlora")
AION_EMBEDDING_METHODS = AION_ENCODER_METHODS[1:]


@dataclass(frozen=True)
class AIONEmbeddingMethodConfig:
    method: Literal["frozen", "ia3", "dora", "qlora"]
    n_z_bins: int = 300
    z_min: float = 0.0
    z_max: float = 6.0
    head_hidden_dim: int = 256
    pooling_heads: int = 8
    epochs: int = 10
    batch_size: int = 1
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2.0e-4
    adapter_learning_rate: float = 1.0e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.03
    max_grad_norm: float = 1.0
    adapter_max_grad_norm: float = 0.1
    head_warmup_epochs: int = 1
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    quantization_block_size: int = 64
    seed: int = 42
    device: str | torch.device = "cuda"

    def normalized(self) -> "AIONEmbeddingMethodConfig":
        if self.method not in AION_ENCODER_METHODS:
            raise ValueError(f"Unknown AION encoder method: {self.method}")
        integers = {
            "n_z_bins": self.n_z_bins,
            "head_hidden_dim": self.head_hidden_dim,
            "pooling_heads": self.pooling_heads,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "rank": self.rank,
            "quantization_block_size": self.quantization_block_size,
        }
        invalid = [name for name, value in integers.items() if int(value) < 1]
        if invalid:
            raise ValueError(f"Positive settings required: {invalid}")
        if self.z_max <= self.z_min:
            raise ValueError("z_max must exceed z_min.")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("The head learning rate and gradient norm must be positive.")
        if self.method != "frozen" and (
            self.adapter_learning_rate <= 0
            or self.adapter_max_grad_norm <= 0
            or self.alpha <= 0
        ):
            raise ValueError("Adapter learning rate, gradient norm, and alpha must be positive.")
        if not 0 <= self.head_warmup_epochs < self.epochs:
            raise ValueError("head_warmup_epochs must be non-negative and smaller than epochs.")
        if self.method == "frozen" and self.head_warmup_epochs != 0:
            raise ValueError("The frozen baseline requires head_warmup_epochs=0.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1).")
        if self.method == "qlora" and self.quantization_block_size not in {
            32, 64, 128, 256, 512, 1024, 2048, 4096
        }:
            raise ValueError("Unsupported bitsandbytes NF4 quantization_block_size.")
        return replace(
            self,
            n_z_bins=int(self.n_z_bins), z_min=float(self.z_min), z_max=float(self.z_max),
            head_hidden_dim=int(self.head_hidden_dim), pooling_heads=int(self.pooling_heads),
            epochs=int(self.epochs), batch_size=int(self.batch_size),
            eval_batch_size=int(self.eval_batch_size),
            gradient_accumulation_steps=int(self.gradient_accumulation_steps),
            learning_rate=float(self.learning_rate),
            adapter_learning_rate=float(self.adapter_learning_rate),
            weight_decay=float(self.weight_decay), warmup_fraction=float(self.warmup_fraction),
            max_grad_norm=float(self.max_grad_norm),
            adapter_max_grad_norm=float(self.adapter_max_grad_norm),
            head_warmup_epochs=int(self.head_warmup_epochs), rank=int(self.rank),
            alpha=float(self.alpha), dropout=float(self.dropout),
            quantization_block_size=int(self.quantization_block_size), seed=int(self.seed),
        )


class AIONTokenRedshiftDataset(Dataset):
    """A same-modality collection of cached codec tokens and redshift targets."""

    def __init__(
        self,
        tokens: Mapping[str, torch.Tensor],
        redshifts: torch.Tensor,
        object_ids: Sequence[str],
    ):
        if not tokens:
            raise ValueError("AION token cache is empty.")
        self.tokens = {key: torch.as_tensor(value).cpu() for key, value in tokens.items()}
        self.redshifts = torch.as_tensor(redshifts, dtype=torch.float32).cpu()
        self.object_ids = list(object_ids)
        lengths = {len(value) for value in self.tokens.values()}
        lengths.update((len(self.redshifts), len(self.object_ids)))
        if len(lengths) != 1:
            raise ValueError(f"AION token-cache lengths differ: {sorted(lengths)}")

    def __len__(self) -> int:
        return len(self.redshifts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "tokens": {key: value[index] for key, value in self.tokens.items()},
            "z_spec": self.redshifts[index],
            "object_id": self.object_ids[index],
        }


class QLoRALinear(nn.Module):
    """bitsandbytes NF4/double-quantized real AION linear plus LoRA."""

    def __init__(self, linear: nn.Linear, *, rank: int, alpha: float,
                 dropout: float, block_size: int):
        super().__init__()
        if bnb is None:
            raise ImportError("AION QLoRA requires bitsandbytes.")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.base = bnb.nn.Linear4bit(
            linear.in_features, linear.out_features, bias=linear.bias is not None,
            compute_dtype=torch.bfloat16, compress_statistics=True,
            quant_type="nf4", quant_storage=torch.uint8, device="cpu",
        )
        self.base.weight = bnb.nn.Params4bit(
            linear.weight.detach().float().cpu(), requires_grad=False,
            blocksize=block_size, compress_statistics=True, quant_type="nf4",
            quant_storage=torch.uint8, module=self.base,
        )
        if linear.bias is not None:
            self.base.bias = nn.Parameter(
                linear.bias.detach().float().cpu(), requires_grad=False
            )
        self.base.requires_grad_(False)
        self.scale = float(alpha) / float(rank)
        self.lora_a = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, linear.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5.0))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.scale * self.lora_b(
            self.lora_a(self.dropout(inputs))
        )


class DoRALinear(nn.Module):
    """DoRA weight decomposition around a frozen real AION linear."""

    def __init__(self, linear: nn.Linear, *, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.scale = float(alpha) / float(rank)
        self.register_buffer("base_weight", linear.weight.detach().clone())
        self.register_buffer(
            "bias", None if linear.bias is None else linear.bias.detach().clone()
        )
        self.lora_a = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, linear.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.magnitude = nn.Parameter(linear.weight.detach().float().norm(dim=1))
        nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5.0))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.scale * (self.lora_b.weight @ self.lora_a.weight)
        direction = self.base_weight.to(update.dtype) + update
        row_norm = direction.float().norm(dim=1).clamp_min(1.0e-6).detach()
        base = F.linear(inputs, self.base_weight.to(inputs.dtype), None)
        low_rank = self.scale * self.lora_b(self.lora_a(self.dropout(inputs)))
        output = (base + low_rank) * (self.magnitude / row_norm).to(base.dtype)
        return output if self.bias is None else output + self.bias.to(output.dtype)


class IA3PackedQKVLinear(nn.Module):
    """Apply IA3 vectors to K and V outputs of AION's packed QKV linear."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        if linear.out_features % 3:
            raise ValueError("AION packed QKV output dimension must be divisible by three.")
        self.base = linear
        width = linear.out_features // 3
        self.key_gate = nn.Parameter(torch.ones(width))
        self.value_gate = nn.Parameter(torch.ones(width))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query, key, value = self.base(inputs).chunk(3, dim=-1)
        return torch.cat((
            query, key * self.key_gate.to(key.dtype),
            value * self.value_gate.to(value.dtype),
        ), dim=-1)


class IA3FeedForwardLinear(nn.Module):
    """Apply IA3 to the feed-forward activation entering AION's fc2."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.base = linear
        self.feedforward_gate = nn.Parameter(torch.ones(linear.in_features))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs * self.feedforward_gate.to(inputs.dtype))


def _aion_linear_targets(aion: nn.Module) -> list[tuple[str, nn.Module, str, nn.Linear]]:
    targets: list[tuple[str, nn.Module, str, nn.Linear]] = []
    for index, block in enumerate(aion.encoder):
        for parent_name, parent, attribute in (
            ("attn", block.attn, "qkv"), ("attn", block.attn, "proj"),
            ("mlp", block.mlp, "fc1"), ("mlp", block.mlp, "fc2"),
            ("mlp", block.mlp, "fc3"),
        ):
            linear = getattr(parent, attribute)
            if not isinstance(linear, nn.Linear):
                raise TypeError(f"encoder.{index}.{parent_name}.{attribute} is not Linear")
            targets.append((f"encoder.{index}.{parent_name}.{attribute}", parent, attribute, linear))
    projection = aion.decoder_proj_context
    if not isinstance(projection, nn.Linear):
        raise TypeError("AION decoder_proj_context is not Linear")
    targets.append(("decoder_proj_context", aion, "decoder_proj_context", projection))
    return targets


def inject_aion_encoder_adapters(aion: nn.Module, config: AIONEmbeddingMethodConfig) -> list[str]:
    """Replace real encoder linears in-place and return their canonical names."""
    config = config.normalized()
    for parameter in aion.parameters():
        parameter.requires_grad = False
    if config.method == "frozen":
        return []
    replaced: list[str] = []
    if config.method == "ia3":
        for index, block in enumerate(aion.encoder):
            qkv = block.attn.qkv
            fc2 = block.mlp.fc2
            if not isinstance(qkv, nn.Linear) or not isinstance(fc2, nn.Linear):
                raise TypeError("IA3 injection expects unwrapped AION encoder linears.")
            block.attn.qkv = IA3PackedQKVLinear(qkv)
            block.mlp.fc2 = IA3FeedForwardLinear(fc2)
            replaced.extend((f"encoder.{index}.attn.qkv", f"encoder.{index}.mlp.fc2"))
        return replaced
    for name, parent, attribute, linear in _aion_linear_targets(aion):
        if config.method == "dora":
            wrapper = DoRALinear(
                linear, rank=config.rank, alpha=config.alpha, dropout=config.dropout
            )
        else:
            wrapper = QLoRALinear(
                linear, rank=config.rank, alpha=config.alpha, dropout=config.dropout,
                block_size=config.quantization_block_size,
            )
        setattr(parent, attribute, wrapper)
        replaced.append(name)
    return replaced


class AIONAttentivePooler(nn.Module):
    """One learned cross-attention query over the full AION encoder sequence."""

    def __init__(self, embedding_dim: int, num_heads: int = 8):
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by pooling heads.")
        self.query = nn.Parameter(torch.empty(1, 1, embedding_dim))
        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embedding_dim)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(sequence.shape[0], -1, -1)
        pooled, _ = self.attention(query, sequence, sequence, need_weights=False)
        return self.norm(pooled[:, 0] + query[:, 0]).float()


class AIONEncoderPhotoZModel(nn.Module):
    def __init__(self, aion: nn.Module, config: AIONEmbeddingMethodConfig):
        super().__init__()
        config = config.normalized()
        self.aion = aion
        self.method = config.method
        embedding_dim = int(aion.decoder_proj_context.out_features)
        # Initialize the common pooler/head before method-specific adapters consume
        # RNG, so a shared seed gives the baseline and candidate identical heads.
        self.pooler = AIONAttentivePooler(embedding_dim, config.pooling_heads)
        self.photoz_head = PhotoZHead(
            embedding_dim, n_z_bins=config.n_z_bins, hidden_dim=config.head_hidden_dim
        )
        self.adapted_modules = inject_aion_encoder_adapters(aion, config)
        self.freeze_adapter = config.method != "frozen"

    def forward(self, tokens: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.freeze_adapter:
            with torch.no_grad():
                sequence = encode_aion_tokens(tokens, self.aion)
        else:
            sequence = encode_aion_tokens(tokens, self.aion)
        return self.photoz_head(self.pooler(sequence))

    def adapter_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.aion.parameters() if parameter.requires_grad]


def build_aion_embedding_method_model(
    aion: nn.Module,
    config: AIONEmbeddingMethodConfig,
) -> AIONEncoderPhotoZModel:
    return AIONEncoderPhotoZModel(aion, config)


@torch.no_grad()
def evaluate_aion_token_model(
    model: AIONEncoderPhotoZModel,
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
        tokens = {key: value.to(device) for key, value in batch["tokens"].items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(tokens)
        logits_parts.append(logits.float().cpu())
        z_parts.append(batch["z_spec"].float().cpu())
        object_ids.extend(batch["object_id"])
    if not logits_parts:
        raise ValueError("No batches were available for AION token evaluation.")
    logits = torch.cat(logits_parts)
    z_spec = torch.cat(z_parts)
    evaluation: dict[str, Any] = {
        "logits": logits, "z_spec": z_spec, "object_id": object_ids,
        "redshift_edges": redshift_edges.cpu(),
        "redshift_centers": redshift_centers.cpu(),
        "loss": redshift_cross_entropy_loss(
            logits, z_spec, edges=redshift_edges.cpu()
        ).item(),
    }
    evaluation.update(predict_photoz_from_logits(logits, centers=redshift_centers.cpu()))
    return evaluation


def _trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def _load_trainable_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        if name not in parameters:
            raise ValueError(f"Unknown trainable checkpoint parameter: {name}")
        parameters[name].data.copy_(value.to(parameters[name].device, parameters[name].dtype))


def train_aion_embedding_method(
    *,
    train_dataset: AIONTokenRedshiftDataset,
    val_dataset: AIONTokenRedshiftDataset,
    test_dataset: AIONTokenRedshiftDataset,
    output_dir: str | Path,
    config: AIONEmbeddingMethodConfig,
) -> dict[str, Any]:
    """Train a frozen or parameter-efficient AION encoder from cached tokens."""
    config = config.normalized()
    if set(train_dataset.tokens) != set(val_dataset.tokens) or set(train_dataset.tokens) != set(test_dataset.tokens):
        raise ValueError("Train, validation, and test token modalities do not match.")
    set_random_seed(config.seed)
    device = resolve_torch_device(config.device)
    if device.type != "cuda":
        raise RuntimeError("AION encoder post-training requires CUDA.")

    aion, _ = load_frozen_aion(device=device)
    model = build_aion_embedding_method_model(aion, config).to(device)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)
    redshift_edges, redshift_centers = make_redshift_grid(config.z_min, config.z_max, config.n_z_bins)
    redshift_edges = redshift_edges.to(device)

    head_parameters = list(model.pooler.parameters()) + list(model.photoz_head.parameters())
    adapter_parameters = model.adapter_parameters()
    groups = [{"params": head_parameters, "lr": config.learning_rate,
               "weight_decay": config.weight_decay}]
    if adapter_parameters:
        groups.append({"params": adapter_parameters, "lr": config.adapter_learning_rate,
                       "weight_decay": 0.0})
    optimizer = (
        bnb.optim.PagedAdamW(groups)
        if config.method == "qlora" and bnb is not None
        else torch.optim.AdamW(groups)
    )
    batches_per_epoch = len(train_loader)
    updates_per_epoch = max(int(np.ceil(batches_per_epoch / config.gradient_accumulation_steps)), 1)
    total_updates = max(updates_per_epoch * config.epochs, 1)
    head_warmup_steps = int(total_updates * config.warmup_fraction)
    schedules = [lambda step: _linear_warmup_decay_lambda(
        step, warmup_steps=head_warmup_steps, total_steps=total_updates
    )]
    if adapter_parameters:
        adapter_start = updates_per_epoch * config.head_warmup_epochs
        adapter_total = max(total_updates - adapter_start, 1)
        adapter_warmup = int(adapter_total * config.warmup_fraction)
        schedules.append(lambda step: 0.0 if step < adapter_start else _linear_warmup_decay_lambda(
            step - adapter_start, warmup_steps=adapter_warmup, total_steps=adapter_total
        ))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedules)

    history: list[dict[str, float | int | str]] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    global_update = 0
    for epoch in range(config.epochs):
        model.train()
        model.freeze_adapter = bool(adapter_parameters) and epoch < config.head_warmup_epochs
        phase = "head_warmup" if model.freeze_adapter else (
            "attentive_head" if config.method == "frozen" else f"joint_{config.method}"
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_count = 0
        for batch_index, batch in enumerate(train_loader):
            tokens = {key: value.to(device) for key, value in batch["tokens"].items()}
            z_spec = batch["z_spec"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(tokens)
                raw_loss = redshift_cross_entropy_loss(logits.float(), z_spec, edges=redshift_edges)
                loss = raw_loss / config.gradient_accumulation_steps
            loss.backward()
            total_loss += float(raw_loss.detach()) * int(z_spec.shape[0])
            total_count += int(z_spec.shape[0])
            should_step = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or batch_index + 1 == batches_per_epoch
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(head_parameters, config.max_grad_norm)
                if adapter_parameters and not model.freeze_adapter:
                    torch.nn.utils.clip_grad_norm_(adapter_parameters, config.adapter_max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

        val_evaluation = evaluate_aion_token_model(
            model, val_loader, device=device, redshift_edges=redshift_edges,
            redshift_centers=redshift_centers,
        )
        val_metrics = summarize_pdf_metrics(val_evaluation)
        lrs = scheduler.get_last_lr()
        row = {
            "epoch": epoch, "phase": phase,
            "train_loss": total_loss / max(total_count, 1),
            "learning_rate": float(lrs[0]),
            "adapter_learning_rate": float(lrs[1]) if len(lrs) > 1 else 0.0,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"AION {config.method} epoch={epoch:03d} phase={phase} "
            f"train_loss={row['train_loss']:.4f} val_loss={val_metrics['cross_entropy']:.4f} "
            f"val_nmad={val_metrics['nmad']:.4f}", flush=True,
        )
        if val_metrics["cross_entropy"] < best_val_loss:
            best_val_loss = val_metrics["cross_entropy"]
            best_state = _trainable_state(model)

    model.freeze_adapter = False
    if best_state is not None:
        _load_trainable_state(model, best_state)
    val_evaluation = evaluate_aion_token_model(
        model, val_loader, device=device, redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
    )
    test_evaluation = evaluate_aion_token_model(
        model, test_loader, device=device, redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
    )
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / f"{config.method}_aion_encoder_adapter.pt"
    torch.save({"trainable_state_dict": _trainable_state(model), "config": asdict(config)}, checkpoint_path)
    result = {
        "model_kind": f"aion_{config.method}_encoder_photoz",
        "history": history,
        "final_metrics": {"val": summarize_pdf_metrics(val_evaluation),
                          "test": summarize_pdf_metrics(test_evaluation)},
        "val_evaluation": val_evaluation, "test_evaluation": test_evaluation,
        "metadata": {
            "posttraining_method": (
                "aion_frozen_encoder_attentive_head_cross_entropy"
                if config.method == "frozen"
                else f"aion_encoder_{config.method}_cross_entropy"
            ),
            "adaptation_scope": "aion_encoder" if config.method != "frozen" else "frozen_aion_encoder",
            "pooling": "single_query_cross_attention",
            "cached_representation": "aion_codec_tokens",
            "quantization": (
                "bitsandbytes_nf4_double_quantization"
                if config.method == "qlora" else None
            ),
            "optimizer": ("bitsandbytes_paged_adamw" if config.method == "qlora" else "adamw"),
            "adapted_modules": model.adapted_modules,
            "config": asdict(config), **trainable_parameter_summary(model),
            "checkpoint": str(checkpoint_path),
        },
    }
    torch.save(result, output_path / "result.pt")
    return result
