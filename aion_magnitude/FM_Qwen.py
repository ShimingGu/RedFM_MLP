from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .clauds_bands import HSC_AION_BANDS
from .dataset import CLAUDSPhotoZDataset, collate_clauds_photoz
from .utils import resolve_torch_device

try:
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
    TRANSFORMERS_IMPORT_ERROR = None
except ImportError as exc:
    AutoModel = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_IMPORT_ERROR = exc


QWEN_MODEL_ROOT = Path(
    os.environ.get(
        "AION_QWEN_MODEL_ROOT",
        Path.home() / "hf_models",
    )
).expanduser()
QWEN_DEFAULT_MODELS = {
    "Qwen3-4B-Base": QWEN_MODEL_ROOT / "Qwen3-4B-Base",
    "Qwen3-8B-Base": QWEN_MODEL_ROOT / "Qwen3-8B-Base",
    "Qwen3.5-0.8B-Base": QWEN_MODEL_ROOT / "Qwen3.5-0.8B-Base",
    "Qwen3.5-4B-Base": QWEN_MODEL_ROOT / "Qwen3.5-4B-Base",
    "Qwen3.5-4B": QWEN_MODEL_ROOT / "Qwen3.5-4B",
    # Backward-compatible aliases for existing commands and cached metadata.
    "qwen3_8b_base": QWEN_MODEL_ROOT / "Qwen3-8B-Base",
    "qwen3_4b_base": QWEN_MODEL_ROOT / "Qwen3-4B-Base",
    "qwen3_5_0_8b_base": QWEN_MODEL_ROOT / "Qwen3.5-0.8B-Base",
    "qwen3_5_4b_base": QWEN_MODEL_ROOT / "Qwen3.5-4B-Base",
    "qwen3_5_4b": QWEN_MODEL_ROOT / "Qwen3.5-4B",
    "qwen2_5_math_7b": QWEN_MODEL_ROOT / "Qwen2.5-Math-7B",
}
QWEN_POOLING_MODES = ("mean", "last", "mean_last")


@dataclass(frozen=True)
class QwenSerializationConfig:
    """Deterministic text schema for serializing one catalogue row for Qwen."""

    schema_name: str = "clauds_mag_morph_v1"
    decimals: int = 5
    missing_token: str = "NA"
    include_hsc_grizy: bool = True
    include_object_metadata: bool = False
    hsc_bands: Sequence[str] = field(default_factory=lambda: tuple(HSC_AION_BANDS))
    prefix: str = "galaxy"


@dataclass(frozen=True)
class QwenEmbeddingConfig:
    """Runtime options for frozen Qwen embedding extraction."""

    model_path: str | Path = QWEN_DEFAULT_MODELS["Qwen3-8B-Base"]
    device: str | torch.device | None = None
    load_in_4bit: bool = True
    torch_dtype: str | torch.dtype | None = "auto"
    max_length: int = 256
    pooling: str = "last"
    normalize: bool = False
    local_files_only: bool = True
    trust_remote_code: bool = True


def require_transformers() -> None:
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "Qwen embedding extraction requires the `transformers` package. "
            "Install it in the pixi environment before calling load_frozen_qwen()."
        ) from TRANSFORMERS_IMPORT_ERROR


def resolve_qwen_model_path(model_path: str | Path) -> Path | str:
    """Resolve a friendly model alias or local path for Qwen loading."""
    text = str(model_path)
    return QWEN_DEFAULT_MODELS.get(text, model_path)


def resolve_qwen_dtype(dtype: str | torch.dtype | None, device: torch.device) -> torch.dtype | str | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    dtype = str(dtype).lower()
    if dtype == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        return None
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown Qwen torch dtype: {dtype!r}.")


def load_frozen_qwen(
    model_path: str | Path = QWEN_DEFAULT_MODELS["Qwen3-8B-Base"],
    *,
    device: torch.device | str | None = None,
    load_in_4bit: bool = True,
    torch_dtype: str | torch.dtype | None = "auto",
    local_files_only: bool = True,
    trust_remote_code: bool = True,
):
    """Load a Qwen base transformer and tokenizer for frozen catalogue embeddings.

    The returned model is eval-only with gradients disabled. On the current
    11 GiB MIG slice, `load_in_4bit=True` is the practical default.
    """
    require_transformers()
    device = resolve_torch_device(device)
    model_path = resolve_qwen_model_path(model_path)
    dtype = resolve_qwen_dtype(torch_dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if dtype is not None:
        model_kwargs["dtype"] = dtype

    if load_in_4bit:
        if device.type != "cuda":
            raise RuntimeError("4-bit Qwen loading requires a CUDA device; set load_in_4bit=False for CPU runs.")
        if BitsAndBytesConfig is None:
            raise ImportError("4-bit Qwen loading requires transformers with BitsAndBytesConfig support.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": 0}
    else:
        model_kwargs["device_map"] = None

    model = AutoModel.from_pretrained(model_path, **model_kwargs)
    if not load_in_4bit:
        model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, tokenizer


def _format_scalar(value: Any, *, decimals: int, missing_token: str) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return missing_token
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text if text else missing_token
    if not np.isfinite(number):
        return missing_token
    return f"{number:.{decimals}f}"


def serialize_qwen_feature_row(
    features: Mapping[str, Any],
    *,
    serialization: QwenSerializationConfig | None = None,
) -> str:
    """Serialize a flat feature mapping into a stable Qwen input string."""
    serialization = serialization or QwenSerializationConfig()
    parts = [serialization.prefix, f"schema={serialization.schema_name}"]
    for name, raw_value in features.items():
        value = _format_scalar(
            raw_value,
            decimals=serialization.decimals,
            missing_token=serialization.missing_token,
        )
        parts.append(f"{name}={value}")
    return " ".join(parts)


def serialize_qwen_batch(
    hsc_batch: Mapping[str, torch.Tensor],
    extra_features: torch.Tensor | None = None,
    feature_names: Sequence[str] | None = None,
    *,
    object_ids: Sequence[Any] | None = None,
    fields: Sequence[Any] | None = None,
    serialization: QwenSerializationConfig | None = None,
) -> list[str]:
    """Serialize a CLAUDS batch into deterministic strings for Qwen.

    HSC grizy magnitudes are written first by band order; `extra_features` are
    then written in the provided `feature_names` order. Object metadata is off
    by default because IDs and field labels can leak survey-specific shortcuts.
    """
    serialization = serialization or QwenSerializationConfig()
    feature_names = list(feature_names or [])
    if extra_features is None:
        extra_features = torch.empty((len(next(iter(hsc_batch.values()))), 0), dtype=torch.float32)
    extra_features = torch.as_tensor(extra_features)
    if extra_features.ndim != 2:
        raise ValueError("extra_features must be a 2D tensor with shape (batch, n_features).")
    if len(feature_names) != int(extra_features.shape[1]):
        raise ValueError(
            "feature_names length must match extra_features.shape[1]: "
            f"{len(feature_names)} vs {extra_features.shape[1]}."
        )

    n_rows = int(extra_features.shape[0])
    texts: list[str] = []
    for row in range(n_rows):
        row_features: dict[str, Any] = {}
        if serialization.include_object_metadata:
            if object_ids is not None:
                row_features["object_id"] = object_ids[row]
            if fields is not None:
                row_features["field"] = fields[row]
        if serialization.include_hsc_grizy:
            for band in serialization.hsc_bands:
                key = f"{band}_mag"
                if key in hsc_batch:
                    row_features[key] = hsc_batch[key][row]
        for column, name in enumerate(feature_names):
            row_features[str(name)] = extra_features[row, column]
        texts.append(serialize_qwen_feature_row(row_features, serialization=serialization))
    return texts


def pool_qwen_hidden_states(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pooling: str = "last",
) -> torch.Tensor:
    """Pool Qwen hidden states into one vector per row."""
    if pooling not in QWEN_POOLING_MODES:
        raise ValueError(f"pooling must be one of {QWEN_POOLING_MODES}, got {pooling!r}.")
    mask = attention_mask.to(last_hidden_state.device).unsqueeze(-1).to(last_hidden_state.dtype)
    lengths = mask.sum(dim=1).clamp_min(1.0)
    mean_embedding = (last_hidden_state * mask).sum(dim=1) / lengths
    if pooling == "mean":
        return mean_embedding

    last_indices = attention_mask.to(last_hidden_state.device).sum(dim=1).clamp_min(1) - 1
    batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    last_embedding = last_hidden_state[batch_indices, last_indices]
    if pooling == "last":
        return last_embedding
    return torch.cat([mean_embedding, last_embedding], dim=-1)


@torch.no_grad()
def extract_qwen_embeddings_from_texts(
    texts: Sequence[str],
    model,
    tokenizer,
    *,
    device: torch.device | str | None = None,
    max_length: int = 256,
    batch_size: int = 8,
    pooling: str = "last",
    normalize: bool = False,
) -> torch.Tensor:
    """Extract frozen Qwen embeddings from already serialized catalogue rows."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    device = resolve_torch_device(device)
    embeddings: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = model(**encoded, use_cache=False, return_dict=True)
        embedding = pool_qwen_hidden_states(
            output.last_hidden_state,
            encoded["attention_mask"],
            pooling=pooling,
        )
        if normalize:
            embedding = F.normalize(embedding.float(), p=2, dim=-1)
        embeddings.append(embedding.detach().float().cpu())
    if not embeddings:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(embeddings, dim=0)


@torch.no_grad()
def extract_qwen_embeddings_to_memory(
    dataset: CLAUDSPhotoZDataset,
    model,
    tokenizer,
    *,
    feature_names: Sequence[str] | None = None,
    serialization: QwenSerializationConfig | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    device: torch.device | str | None = None,
    max_length: int = 256,
    pooling: str = "last",
    normalize: bool = False,
) -> torch.Tensor:
    """Extract Qwen embeddings for a CLAUDSPhotoZDataset.

    This mirrors extract_aion_embeddings_to_memory(), but Qwen receives a text
    serialization of each catalogue row instead of typed AION modalities.
    """
    device = resolve_torch_device(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_clauds_photoz,
    )
    parts: list[torch.Tensor] = []
    for batch in loader:
        texts = serialize_qwen_batch(
            batch.hsc_batch,
            batch.extra_features,
            feature_names=feature_names,
            object_ids=batch.object_id,
            fields=batch.field,
            serialization=serialization,
        )
        embedding = extract_qwen_embeddings_from_texts(
            texts,
            model,
            tokenizer,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            pooling=pooling,
            normalize=normalize,
        )
        parts.append(embedding.cpu())
    if not parts:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(parts, dim=0)


def qwen_embedding_metadata(
    config: QwenEmbeddingConfig,
    serialization: QwenSerializationConfig | None = None,
) -> dict[str, Any]:
    serialization = serialization or QwenSerializationConfig()
    model_path = resolve_qwen_model_path(config.model_path)
    return {
        "embedding_backend": "qwen",
        "qwen_model_path": str(model_path),
        "qwen_load_in_4bit": bool(config.load_in_4bit),
        "qwen_torch_dtype": None if config.torch_dtype is None else str(config.torch_dtype),
        "qwen_max_length": int(config.max_length),
        "qwen_pooling": config.pooling,
        "qwen_normalize": bool(config.normalize),
        "qwen_local_files_only": bool(config.local_files_only),
        "qwen_serialization_schema": serialization.schema_name,
        "qwen_serialization_decimals": int(serialization.decimals),
        "qwen_serialization_missing_token": serialization.missing_token,
        "qwen_serialization_include_hsc_grizy": bool(serialization.include_hsc_grizy),
        "qwen_serialization_include_object_metadata": bool(serialization.include_object_metadata),
        "qwen_serialization_hsc_bands": list(serialization.hsc_bands),
    }
