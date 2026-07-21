"""IoTFM embeddings from small architecture-test transformer checkpoints.

``IoTFM`` means **inference-optimized transformer feature mapping**: a frozen
transformer maps each serialized catalogue record to a fixed feature vector for
a separately trained downstream task head.

This module deliberately treats models from ``inference-optimization`` as
generic transformer feature extractors.  It does not assume that a reduced
checkpoint retains the knowledge or capabilities of the flagship model whose
architecture it resembles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .utils import resolve_torch_device

try:
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
except ImportError:  # Keep serialization utilities usable without transformers.
    AutoModel = AutoTokenizer = BitsAndBytesConfig = None


INFERENCE_OPT_MODEL_ROOT = Path(
    os.environ.get("AION_INFERENCE_OPT_MODEL_ROOT", Path.home() / "hf_models")
).expanduser()


@dataclass(frozen=True)
class InferenceOptimizedModelSpec:
    """Registry information for one architecture-test checkpoint."""

    canonical_name: str
    hub_id: str
    local_dir_name: str
    architecture_family: str
    intended_use_note: str


_GLM_052_SPEC = InferenceOptimizedModelSpec(
    canonical_name="GLM-5.2-0.8B-A0.8B",
    hub_id="inference-optimization/GLM-5.2-0.8B-A0.8B",
    local_dir_name="GLM-5.2-0.8B-A0.8B",
    architecture_family="glm_moe_dsa",
    intended_use_note=(
        "Small testing/development checkpoint derived from the GLM-5.2 "
        "architecture; it does not reproduce the flagship model's capabilities."
    ),
)

INFERENCE_OPTIMIZED_MODELS: Mapping[str, InferenceOptimizedModelSpec] = MappingProxyType(
    {_GLM_052_SPEC.canonical_name: _GLM_052_SPEC}
)
INFERENCE_OPTIMIZED_MODEL_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "glm_5_2_0_8b_a0_8b": _GLM_052_SPEC.canonical_name,
        "glm-5.2-0.8b-a0.8b": _GLM_052_SPEC.canonical_name,
    }
)


@dataclass(frozen=True)
class CatalogueSerializationConfig:
    """Stable text representation of an ordered catalogue row."""

    schema_name: str = "catalogue_features_v1"
    decimals: int = 5
    missing_token: str = "NA"
    prefix: str = "catalogue observation"
    include_feature_names: bool = True


@dataclass(frozen=True)
class InferenceOptimizedEmbeddingConfig:
    """Loading and embedding options for a registered or arbitrary model."""

    model_path: str | Path = "GLM-5.2-0.8B-A0.8B"
    device: str | torch.device | None = None
    torch_dtype: str | torch.dtype | None = "auto"
    max_length: int = 2048
    pooling: str = "mean"
    normalize: bool = False
    local_files_only: bool = True
    trust_remote_code: bool = True
    load_in_4bit: bool = False
    freeze_model: bool = True


def _canonical_name(name: str) -> str | None:
    if name in INFERENCE_OPTIMIZED_MODELS:
        return name
    return INFERENCE_OPTIMIZED_MODEL_ALIASES.get(name.lower())


def get_model_spec(model_path: str | Path) -> InferenceOptimizedModelSpec | None:
    """Return registry metadata, or ``None`` for an arbitrary path/Hub ID."""

    canonical = _canonical_name(str(model_path))
    return INFERENCE_OPTIMIZED_MODELS.get(canonical) if canonical else None


def resolve_model_path(model_path: str | Path) -> str:
    """Resolve a registered short name to its persistent local checkpoint."""

    spec = get_model_spec(model_path)
    if spec is None:
        return str(Path(model_path).expanduser()) if Path(model_path).expanduser().exists() else str(model_path)
    local_path = INFERENCE_OPT_MODEL_ROOT / spec.local_dir_name
    return str(local_path if local_path.exists() else spec.hub_id)


def _format_catalogue_value(value: Any, config: CatalogueSerializationConfig) -> str:
    if value is None:
        return config.missing_token
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Each catalogue value must be scalar.")
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return config.missing_token
        return f"{value:.{config.decimals}f}"
    return str(value)


def serialize_catalogue_row(
    row: Mapping[str, Any] | Sequence[Any],
    feature_names: Sequence[str] | None = None,
    config: CatalogueSerializationConfig | None = None,
) -> str:
    """Serialize one row without inventing semantic or physical descriptions."""

    config = config or CatalogueSerializationConfig()
    if isinstance(row, Mapping):
        names = list(row)
        values = list(row.values())
    else:
        values = list(row)
        names = list(feature_names) if feature_names is not None else [f"feature_{i}" for i in range(len(values))]
    if len(names) != len(values):
        raise ValueError("feature_names and row must have the same length.")
    fields = []
    for name, value in zip(names, values):
        rendered = _format_catalogue_value(value, config)
        fields.append(f"{name}={rendered}" if config.include_feature_names else rendered)
    return f"{config.prefix}; " + "; ".join(fields)


def serialize_catalogue_batch(
    values: torch.Tensor | Sequence[Sequence[Any]],
    feature_names: Sequence[str],
    config: CatalogueSerializationConfig | None = None,
) -> list[str]:
    """Serialize a two-dimensional catalogue matrix row by row."""

    if isinstance(values, torch.Tensor):
        if values.ndim != 2:
            raise ValueError("values must be a two-dimensional matrix.")
        rows = values.detach().cpu().tolist()
    else:
        rows = list(values)
    if any(len(row) != len(feature_names) for row in rows):
        raise ValueError("Every row must match the number of feature_names.")
    return [serialize_catalogue_row(row, feature_names, config) for row in rows]


def _resolve_dtype(dtype: str | torch.dtype | None, device: torch.device) -> torch.dtype | None:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        return torch.bfloat16 if device.type == "cuda" else None
    aliases = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    try:
        return aliases[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch_dtype: {dtype!r}") from exc


def load_inference_optimized_transformer(
    config: InferenceOptimizedEmbeddingConfig | None = None,
) -> tuple[Any, Any, torch.device]:
    """Load the tokenizer and base transformer, intentionally without an LM head."""

    if AutoModel is None:
        raise ImportError("transformers is required to load inference-optimized models.")
    config = config or InferenceOptimizedEmbeddingConfig()
    device = resolve_torch_device(config.device)
    model_path = resolve_model_path(config.model_path)
    common = {
        "local_files_only": config.local_files_only,
        "trust_remote_code": config.trust_remote_code,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_path, **common)
    model_kwargs = dict(common)
    dtype = _resolve_dtype(config.torch_dtype, device)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if config.load_in_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError("transformers with bitsandbytes support is required for 4-bit loading.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        model_kwargs["device_map"] = "auto"
    model = AutoModel.from_pretrained(model_path, **model_kwargs)
    if not config.load_in_4bit:
        model.to(device)
    if config.freeze_model:
        model.requires_grad_(False)
    model.eval()
    return tokenizer, model, device


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str = "mean",
) -> torch.Tensor:
    """Pool token states using only non-padding positions."""

    if hidden_states.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Expected hidden_states [batch, tokens, hidden] and attention_mask [batch, tokens].")
    mask = attention_mask.to(hidden_states.device).bool()
    if pooling == "mean":
        weights = mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * weights).sum(1) / weights.sum(1).clamp_min(1)
    positions = mask.long().sum(1).sub(1).clamp_min(0)
    last = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), positions]
    if pooling == "last":
        return last
    if pooling == "mean_last":
        mean = pool_hidden_states(hidden_states, attention_mask, "mean")
        return torch.cat((mean, last), dim=-1)
    raise ValueError("pooling must be one of: mean, last, mean_last")


@torch.no_grad()
def extract_text_embeddings(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    device: str | torch.device,
    *,
    batch_size: int = 32,
    max_length: int = 2048,
    pooling: str = "mean",
    normalize: bool = False,
) -> torch.Tensor:
    """Extract CPU embeddings without materializing language-model logits."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    device = torch.device(device)
    batches = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = model(**encoded, use_cache=False, return_dict=True)
        pooled = pool_hidden_states(output.last_hidden_state, encoded["attention_mask"], pooling)
        if normalize:
            pooled = F.normalize(pooled.float(), dim=-1)
        batches.append(pooled.detach().cpu())
    if batches:
        return torch.cat(batches, dim=0)
    hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0))
    if pooling == "mean_last":
        hidden_size *= 2
    return torch.empty((0, hidden_size))


def build_embedding_metadata(
    embedding_config: InferenceOptimizedEmbeddingConfig,
    serialization_config: CatalogueSerializationConfig | None = None,
) -> dict[str, Any]:
    """Build metadata that keeps the checkpoint's experimental status explicit."""

    spec = get_model_spec(embedding_config.model_path)
    return {
        "backend": "inference_optimized_transformer",
        "model": asdict(spec) if spec else {"requested_model": str(embedding_config.model_path)},
        "resolved_model_path": resolve_model_path(embedding_config.model_path),
        "model_role": "architecture_test_checkpoint",
        "capability_warning": (
            spec.intended_use_note if spec else "Capabilities must be established empirically for this checkpoint."
        ),
        "embedding": {**asdict(embedding_config), "model_path": str(embedding_config.model_path)},
        "serialization": asdict(serialization_config or CatalogueSerializationConfig()),
    }


__all__ = [
    "CatalogueSerializationConfig",
    "INFERENCE_OPTIMIZED_MODELS",
    "InferenceOptimizedEmbeddingConfig",
    "InferenceOptimizedModelSpec",
    "build_embedding_metadata",
    "extract_text_embeddings",
    "get_model_spec",
    "load_inference_optimized_transformer",
    "pool_hidden_states",
    "resolve_model_path",
    "serialize_catalogue_batch",
    "serialize_catalogue_row",
]
