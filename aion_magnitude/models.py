from __future__ import annotations
import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .clauds_bands import HSC_AION_BANDS, ALL_BAND_FLUX_COLUMNS, ALL_FLAG_COLUMNS, OBJECT_ID_COLUMN
from .utils import flux_to_ab_mag, _path_tag, resolve_torch_device, table_length, table_column

# Try loading AION libraries
try:
    from aion.model import AION
    from aion.codecs import CodecManager
    from aion.modalities import HSCMagG, HSCMagR, HSCMagI, HSCMagZ, HSCMagY
    AION_AVAILABLE = True
    AION_IMPORT_ERROR = None
except ImportError as exc:
    AION_AVAILABLE = False
    AION_IMPORT_ERROR = exc

AION_FINETUNE_MODES = ("hsc_magnitude_token_embeddings", "last_encoder_block_and_norm")
HSC_MAG_TOKEN_EMBEDDING_NAMES = ("tok_mag_g", "tok_mag_r", "tok_mag_i", "tok_mag_z", "tok_mag_y")


class ExtraPhotometryEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhotoZHead(nn.Module):
    def __init__(self, input_dim: int, n_z_bins: int | None = None, hidden_dim: int = 256):
        super().__init__()
        n_z_bins = 300 if n_z_bins is None else n_z_bins

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_z_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TabularPhotoZModel(nn.Module):
    def __init__(
        self,
        extra_feature_dim: int,
        n_z_bins: int | None = None,
        extra_hidden_dim: int = 128,
        head_hidden_dim: int = 256,
    ):
        super().__init__()
        self.extra_encoder = ExtraPhotometryEncoder(extra_feature_dim, extra_hidden_dim)
        self.photoz_head = PhotoZHead(extra_hidden_dim, n_z_bins, head_hidden_dim)

    def forward(self, extra_features: torch.Tensor) -> torch.Tensor:
        extra_embedding = self.extra_encoder(extra_features)
        return self.photoz_head(extra_embedding)


class AIONOnlyPhotoZModel(nn.Module):
    def __init__(self, aion_dim: int, n_z_bins: int | None = None, head_hidden_dim: int = 256):
        super().__init__()
        self.photoz_head = PhotoZHead(aion_dim, n_z_bins, head_hidden_dim)

    def forward(self, aion_embedding: torch.Tensor) -> torch.Tensor:
        return self.photoz_head(aion_embedding)


class CLAUDSPhotoZModel(nn.Module):
    def __init__(
        self,
        aion_dim: int,
        extra_feature_dim: int,
        n_z_bins: int | None = None,
        extra_hidden_dim: int = 128,
        head_hidden_dim: int = 256,
    ):
        super().__init__()

        self.extra_encoder = ExtraPhotometryEncoder(
            n_features=extra_feature_dim,
            hidden_dim=extra_hidden_dim,
        )
        self.photoz_head = PhotoZHead(
            input_dim=aion_dim + extra_hidden_dim,
            n_z_bins=n_z_bins,
            hidden_dim=head_hidden_dim,
        )

    def forward(self, aion_embedding: torch.Tensor, extra_features: torch.Tensor) -> torch.Tensor:
        extra_embedding = self.extra_encoder(extra_features)
        fused = torch.cat([aion_embedding, extra_embedding], dim=-1)
        return self.photoz_head(fused)


def load_frozen_aion(
    model_name: str = "polymathic-ai/aion-base",
    device: torch.device | str | None = None,
):
    device = resolve_torch_device(device)
    if not AION_AVAILABLE:
        raise ImportError(
            "The `aion` package is not importable in this environment. "
            "Install it in `aion_env` before running AION embedding extraction."
        ) from AION_IMPORT_ERROR

    aion = AION.from_pretrained(model_name).to(device).eval()
    codec_manager = CodecManager(device=device)

    for parameter in aion.parameters():
        parameter.requires_grad = False

    return aion, codec_manager


def extract_hsc_aion_embedding(
    batch: dict[str, torch.Tensor],
    aion,
    codec_manager,
    device: torch.device | str | None = None,
    *,
    track_input_grad: bool = False,
) -> torch.Tensor:
    device = resolve_torch_device(device)
    modalities = [
        HSCMagG(value=batch["g_mag"].to(device)),
        HSCMagR(value=batch["r_mag"].to(device)),
        HSCMagI(value=batch["i_mag"].to(device)),
        HSCMagZ(value=batch["z_mag"].to(device)),
        HSCMagY(value=batch["y_mag"].to(device)),
    ]

    context = nullcontext() if track_input_grad else torch.no_grad()
    with context:
        tokens = codec_manager.encode(*modalities)
        n_tokens = sum(
            tensor.shape[1] if tensor.ndim > 1 else 1
            for tensor in tokens.values()
        )
        # AION provides the encoder sequence; mean pooling makes one vector per object.
        sequence = aion.encode(tokens, num_encoder_tokens=n_tokens)
        embedding = sequence.mean(dim=1)

    return embedding


def build_baseline_model(
    model_kind: str,
    aion_dim: int,
    extra_feature_dim: int,
    n_z_bins: int = 300,
) -> nn.Module:
    if model_kind == "tabular":
        return TabularPhotoZModel(extra_feature_dim=extra_feature_dim, n_z_bins=n_z_bins)
    if model_kind in {"aion", "qwen", "iotfm"}:
        return AIONOnlyPhotoZModel(aion_dim=aion_dim, n_z_bins=n_z_bins)
    if model_kind == "fusion":
        return CLAUDSPhotoZModel(aion_dim=aion_dim, extra_feature_dim=extra_feature_dim, n_z_bins=n_z_bins)
    raise ValueError(f"Unknown model_kind: {model_kind}")


def load_baseline_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_kind: str,
    aion_dim: int,
    extra_feature_dim: int,
    n_z_bins: int | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    device = resolve_torch_device(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    if n_z_bins is None:
        for key, value in checkpoint["state_dict"].items():
            if key.endswith("photoz_head.net.6.weight"):
                n_z_bins = int(value.shape[0])
                break
    model = build_baseline_model(model_kind, aion_dim, extra_feature_dim, n_z_bins=n_z_bins or 300).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_aion_mag_adjustment(adjustment_path: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load an optional linear magnitude adjustment for the frozen AION input."""
    if isinstance(adjustment_path, Mapping):
        return dict(adjustment_path)
    return torch.load(Path(adjustment_path), map_location="cpu", weights_only=False)


def aion_mag_adjustment_tag(
    adjustment_path: str | Path | None,
    explicit_tag: str | None = None,
) -> str | None:
    """Return a cache/path-safe tag for an optional AION magnitude adjustment."""
    if adjustment_path is None:
        return None
    tag = explicit_tag or Path(adjustment_path).stem
    return _path_tag(str(tag))


def aion_mag_adjustment_metadata(
    adjustment_path: str | Path | None,
    adjustment: Mapping[str, Any] | None = None,
    explicit_tag: str | None = None,
) -> dict[str, Any]:
    """Small metadata block used to keep adapted AION embedding caches honest."""
    if adjustment_path is None:
        return {}
    path = Path(adjustment_path)
    if adjustment is None:
        adjustment = load_aion_mag_adjustment(path)
    metadata = {
        "aion_mag_adjustment_path": str(path),
        "aion_mag_adjustment_tag": aion_mag_adjustment_tag(path, explicit_tag),
        "aion_mag_adjustment_mode": adjustment.get("mode"),
        "aion_mag_adjustment_source_bands": list(adjustment.get("source_bands", [])),
        "aion_mag_adjustment_target_bands": list(adjustment.get("target_bands", HSC_AION_BANDS)),
    }
    if path.exists():
        metadata["aion_mag_adjustment_mtime"] = path.stat().st_mtime
    return metadata


def validate_cached_aion_mag_adjustment(
    cached_metadata: Mapping[str, Any],
    adjustment_path: str | Path | None,
    explicit_tag: str | None = None,
) -> None:
    """Reject stale adapted embedding caches when the learned M file changed."""
    if adjustment_path is None:
        return
    current = aion_mag_adjustment_metadata(adjustment_path, explicit_tag=explicit_tag)
    expected_tag = current.get("aion_mag_adjustment_tag")
    cached_tag = cached_metadata.get("aion_mag_adjustment_tag")
    if cached_tag != expected_tag:
        raise RuntimeError(
            "Cached AION embeddings were built with a different magnitude adjustment tag. "
            "Use a separate cache path or rerun with force_recompute_embeddings=True."
        )
    current_mtime = current.get("aion_mag_adjustment_mtime")
    cached_mtime = cached_metadata.get("aion_mag_adjustment_mtime")
    if current_mtime is not None and cached_mtime is not None and not np.isclose(float(current_mtime), float(cached_mtime)):
        raise RuntimeError(
            "The learned magnitude-adjustment file has changed since this AION embedding cache was built. "
            "Rerun with force_recompute_embeddings=True to refresh adapted embeddings."
        )


def _adjustment_tensor(
    adjustment: Mapping[str, Any],
    key: str,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    value = adjustment[key]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def apply_aion_mag_adjustment_to_hsc_features(
    hsc_features: Mapping[str, torch.Tensor],
    source_features: torch.Tensor,
    adjustment_path: str | Path | Mapping[str, Any],
    *,
    inplace: bool = False,
) -> dict[str, torch.Tensor]:
    """Apply a learned linear adapter to the magnitudes seen by frozen AION."""
    adjustment = load_aion_mag_adjustment(adjustment_path)
    target_bands = list(adjustment.get("target_bands", HSC_AION_BANDS))
    source_bands = list(adjustment.get("source_bands", []))
    matrix = _adjustment_tensor(adjustment, "matrix")
    source_mean = _adjustment_tensor(adjustment, "source_mean").reshape(1, -1)
    source_std = _adjustment_tensor(adjustment, "source_std").reshape(1, -1).clamp_min(1e-6)
    active_mask = adjustment.get("active_mask")
    if active_mask is None:
        active_mask_tensor = torch.ones_like(matrix)
    else:
        active_mask_tensor = _adjustment_tensor(adjustment, "active_mask")

    source_features = torch.as_tensor(source_features, dtype=torch.float32)
    if source_features.ndim != 2:
        raise ValueError("source_features must be a 2D tensor with shape (n_rows, n_source_bands).")
    if source_features.shape[1] != matrix.shape[1]:
        raise ValueError(
            "source_features column count does not match the learned adjustment matrix: "
            f"{source_features.shape[1]} vs {matrix.shape[1]}."
        )
    if matrix.shape[0] != len(target_bands):
        raise ValueError("adjustment matrix row count must match target_bands.")
    if source_bands and len(source_bands) != source_features.shape[1]:
        raise ValueError("source_bands length must match source_features column count.")

    standardized = (source_features - source_mean) / source_std
    delta = standardized @ (matrix * active_mask_tensor).T
    delta_clip = adjustment.get("delta_clip_mag")
    if delta_clip is not None:
        delta = delta.clamp(min=-float(delta_clip), max=float(delta_clip))

    adjusted = dict(hsc_features) if inplace else {
        key: value.clone()
        for key, value in hsc_features.items()
    }
    for column, band in enumerate(target_bands):
        key = f"{band}_mag"
        if key not in adjusted:
            raise KeyError(f"hsc_features is missing required target band {key!r}.")
        adjusted[key] = adjusted[key].float() + delta[:, column].to(adjusted[key].device)
    return adjusted


def build_aion_mag_adjustment_source_matrix_from_table(
    table,
    adjustment_path: str | Path | Mapping[str, Any],
    *,
    rows=None,
    mag_zero_point: float = 23.0,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    """Build the source-band matrix expected by a learned AION magnitude adjustment."""
    adjustment = load_aion_mag_adjustment(adjustment_path)
    source_bands = list(adjustment.get("source_bands", []))
    if not source_bands:
        n_rows = table_length(table) if rows is None else len(table_column(table, OBJECT_ID_COLUMN, rows=rows))
        return torch.empty((n_rows, 0), dtype=torch.float32), [], {}

    hsc_features: dict[str, torch.Tensor] = {}
    hsc_source_bands = [band for band in source_bands if band in HSC_AION_BANDS]
    if hsc_source_bands:
        hsc_features, _ = build_hsc_aion_features_from_table(
            table,
            rows=rows,
            mag_zero_point=mag_zero_point,
        )

    extra_source_bands = [band for band in source_bands if band not in HSC_AION_BANDS]
    extra_feature_by_name: dict[str, torch.Tensor] = {}
    metadata: dict[str, Any] = {
        "source_hsc_bands": hsc_source_bands,
        "source_extra_bands": extra_source_bands,
    }
    if extra_source_bands:
        extra_features, extra_names, extra_metadata = build_extra_feature_matrix_from_table(
            table,
            rows=rows,
            extra_bands=extra_source_bands,
            mag_zero_point=mag_zero_point,
            invalid_fill=adjustment.get("source_invalid_fill", "median"),
            include_valid_flags=False,
            return_metadata=True,
        )
        extra_feature_by_name = {
            name: extra_features[:, index]
            for index, name in enumerate(extra_names)
        }
        metadata.update(extra_metadata)

    columns: list[torch.Tensor] = []
    source_names: list[str] = []
    for band in source_bands:
        feature_name = f"{band}_mag"
        if band in HSC_AION_BANDS:
            columns.append(torch.as_tensor(hsc_features[feature_name], dtype=torch.float32).reshape(-1))
        elif feature_name in extra_feature_by_name:
            columns.append(torch.as_tensor(extra_feature_by_name[feature_name], dtype=torch.float32).reshape(-1))
        else:
            raise KeyError(f"Could not build source feature {feature_name!r} for AION magnitude adjustment.")
        source_names.append(feature_name)
    source_features = torch.stack(columns, dim=1)

    expected_names = list(adjustment.get("source_feature_names", [f"{band}_mag" for band in source_bands]))
    if list(source_names) != expected_names:
        raise RuntimeError(
            "Source feature names do not match the learned AION magnitude adjustment. "
            f"Expected {expected_names}, got {source_names}."
        )
    return source_features, source_names, metadata
