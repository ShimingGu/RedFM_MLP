from __future__ import annotations

"""Frozen timm raw-image features aligned to the CLAUDS morphology cohort."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .morphology import _extract_cutout, load_morphology_tiles
from .utils import resolve_torch_device

try:
    import timm
except ImportError:  # Keep the rest of aion_magnitude importable without timm.
    timm = None


DEFAULT_TIMM_MORPHOLOGY_MODEL = "hf-hub:timm/convnext_tiny.dinov3_lvd1689m"
TIMM_CUTOUT_NORMALIZATIONS = ("asinh_percentile",)


@dataclass(frozen=True)
class TimmMorphologyConfig:
    """timm-style frozen feature-extractor configuration."""

    model_name: str = DEFAULT_TIMM_MORPHOLOGY_MODEL
    pretrained: bool = True
    input_size: int = 224
    in_chans: int = 1
    global_pool: str = "avg"
    batch_size: int = 128
    normalization: str = "asinh_percentile"
    percentile: float = 99.0
    device: str | torch.device | None = "auto"

    def normalized(self) -> "TimmMorphologyConfig":
        if int(self.input_size) < 16:
            raise ValueError("timm input_size must be at least 16.")
        if int(self.in_chans) != 1:
            raise ValueError("The CLAUDS u-band timm backend currently requires in_chans=1.")
        if int(self.batch_size) < 1:
            raise ValueError("timm batch_size must be positive.")
        if str(self.global_pool) not in {"avg", "max", "avgmax", "catavgmax"}:
            raise ValueError("Unsupported timm global_pool.")
        if str(self.normalization) not in TIMM_CUTOUT_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {TIMM_CUTOUT_NORMALIZATIONS}."
            )
        if not (0.0 < float(self.percentile) <= 100.0):
            raise ValueError("percentile must be in (0, 100].")
        return replace(
            self,
            model_name=str(self.model_name),
            pretrained=bool(self.pretrained),
            input_size=int(self.input_size),
            in_chans=int(self.in_chans),
            global_pool=str(self.global_pool),
            batch_size=int(self.batch_size),
            normalization=str(self.normalization),
            percentile=float(self.percentile),
        )

    def cache_tag(self) -> str:
        model_tag = self.model_name.rsplit("/", 1)[-1].replace(".", "_").replace("-", "_")
        return (
            f"{model_tag}_in{self.input_size}_{self.global_pool}_"
            f"{self.normalization}_p{self.percentile:g}"
        )


def require_timm() -> None:
    if timm is None:
        raise ImportError(
            "The timm morphology backend requires `timm`. Install the vision extra "
            "or use the repository Pixi environment."
        )


def create_timm_image_encoder(
    config: TimmMorphologyConfig,
) -> tuple[torch.nn.Module, torch.device, float, float]:
    """Create a classifier-free timm model using its native model factory."""
    require_timm()
    config = config.normalized()
    device = resolve_torch_device(config.device)
    model = timm.create_model(
        config.model_name,
        pretrained=config.pretrained,
        in_chans=config.in_chans,
        num_classes=0,
        global_pool=config.global_pool,
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    pretrained_cfg = dict(getattr(model, "pretrained_cfg", {}) or {})
    means = np.asarray(pretrained_cfg.get("mean", (0.5,)), dtype=np.float32)
    stds = np.asarray(pretrained_cfg.get("std", (0.5,)), dtype=np.float32)
    input_mean = float(np.mean(means))
    input_std = float(np.mean(stds))
    if not np.isfinite(input_std) or input_std <= 0.0:
        raise ValueError("timm pretrained input standard deviation must be positive.")
    return model, device, input_mean, input_std


def preprocess_timm_cutouts(
    cutouts: np.ndarray | torch.Tensor,
    *,
    input_size: int,
    percentile: float,
    input_mean: float,
    input_std: float,
) -> torch.Tensor:
    """Apply a deterministic signed-asinh stretch before timm normalization."""
    values = torch.as_tensor(cutouts, dtype=torch.float32)
    if values.ndim == 3:
        values = values[:, None, :, :]
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError("cutouts must have shape [batch, height, width] or [batch, 1, height, width].")
    flat_abs = values.abs().flatten(1)
    scale = torch.quantile(flat_abs, float(percentile) / 100.0, dim=1)
    fallback = flat_abs.amax(dim=1).clamp_min(1e-6)
    scale = torch.where(scale > 0.0, scale, fallback).reshape(-1, 1, 1, 1)
    stretched = torch.asinh(values / scale) / float(np.arcsinh(1.0))
    unit_interval = (stretched.clamp(-1.0, 1.0) + 1.0) * 0.5
    resized = F.interpolate(
        unit_interval,
        size=(int(input_size), int(input_size)),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return (resized - float(input_mean)) / float(input_std)


def _expected_metadata(
    product: Mapping[str, Any],
    config: TimmMorphologyConfig,
) -> dict[str, Any]:
    product_metadata = dict(product.get("metadata", {}))
    return {
        "image_embedding_backend": "timm",
        "timm_config": {
            key: str(value) if isinstance(value, torch.device) else value
            for key, value in asdict(config.normalized()).items()
            if key != "device"
        },
        "morphology_tag": product_metadata.get("morphology_tag"),
        "image_background_mode": product_metadata.get("image_background_mode"),
        "image_flux_scale": product_metadata.get("image_flux_scale"),
        "min_cutout_weight_coverage": product_metadata.get(
            "min_cutout_weight_coverage"
        ),
    }


def validate_timm_embedding_cache(
    cached: Mapping[str, Any],
    product: Mapping[str, Any],
    expected_metadata: Mapping[str, Any],
    cache_path: str | Path,
) -> torch.Tensor:
    cached_ids = [str(value) for value in cached.get("object_id", [])]
    product_ids = [str(value) for value in product["object_id"]]
    if cached_ids != product_ids:
        raise RuntimeError(f"timm cache row order does not match the morphology cohort: {cache_path}")
    embeddings = torch.as_tensor(cached.get("embedding"), dtype=torch.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(product_ids) or embeddings.shape[1] == 0:
        raise RuntimeError(f"timm cache contains an invalid embedding tensor: {cache_path}")
    metadata = dict(cached.get("metadata", {}))
    mismatches = [
        key for key, value in expected_metadata.items() if metadata.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"timm cache settings differ for {mismatches}: {cache_path}. Rebuild it."
        )
    return embeddings


@torch.no_grad()
def extract_or_load_timm_embeddings(
    product: Mapping[str, Any],
    *,
    morphology_dir: str | Path,
    cache_path: str | Path,
    config: TimmMorphologyConfig | None = None,
    force_recompute: bool = False,
) -> torch.Tensor:
    """Extract frozen timm features in morphology-product object order."""
    config = (config or TimmMorphologyConfig()).normalized()
    cache_path = Path(cache_path)
    expected_metadata = _expected_metadata(product, config)
    if cache_path.exists() and not force_recompute:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        return validate_timm_embedding_cache(
            cached, product, expected_metadata, cache_path
        )

    quality = np.asarray(product.get("image_quality"))
    if quality.dtype.names is None or not {"object_id", "assigned_tile", "x_image", "y_image"}.issubset(quality.dtype.names):
        raise ValueError("Morphology product lacks the image-quality coordinates required by timm.")
    if [str(value) for value in quality["object_id"]] != [
        str(value) for value in product["object_id"]
    ]:
        raise RuntimeError("Morphology image-quality rows do not match product object IDs.")

    metadata = dict(product.get("metadata", {}))
    tile_manifest = list(metadata.get("morphology_tile_manifest", []))
    if not tile_manifest:
        raise RuntimeError("Morphology product does not record its tile manifest.")
    morphology_dir = Path(morphology_dir)
    image_paths = [morphology_dir / relative_path for relative_path in tile_manifest]
    tiles = load_morphology_tiles(morphology_dir, image_paths=image_paths)
    model, device, input_mean, input_std = create_timm_image_encoder(config)
    expected_metadata = {
        **expected_metadata,
        "timm_version": str(getattr(timm, "__version__", "unknown")),
        "input_mean": input_mean,
        "input_std": input_std,
    }
    embeddings: torch.Tensor | None = None
    completed = 0
    next_report = 1_000
    try:
        for tile_index, tile in enumerate(tiles):
            try:
                rows = np.flatnonzero(quality["assigned_tile"] == tile_index)
                for start in range(0, len(rows), config.batch_size):
                    batch_rows = rows[start : start + config.batch_size]
                    cutouts = [
                        _extract_cutout(
                            tile,
                            float(quality["x_image"][row]),
                            float(quality["y_image"][row]),
                            cutout_size=int(metadata["aion_image_cutout_size"]),
                            background_mode=str(metadata["image_background_mode"]),
                            image_flux_scale=float(metadata["image_flux_scale"]),
                        )[0]
                        for row in batch_rows
                    ]
                    inputs = preprocess_timm_cutouts(
                        np.stack(cutouts),
                        input_size=config.input_size,
                        percentile=config.percentile,
                        input_mean=input_mean,
                        input_std=input_std,
                    ).to(device)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        output = model(inputs)
                    output = torch.as_tensor(output).detach().float().cpu()
                    if output.ndim != 2:
                        raise RuntimeError(
                            f"Expected pooled timm embeddings [batch, features], got {tuple(output.shape)}."
                        )
                    if embeddings is None:
                        embeddings = torch.empty(
                            (len(product["object_id"]), output.shape[1]),
                            dtype=torch.float32,
                        )
                    embeddings[torch.as_tensor(batch_rows, dtype=torch.long)] = output
                    completed += len(batch_rows)
                    if completed >= next_report or completed == len(product["object_id"]):
                        print(
                            f"timm raw-image embeddings: {completed:,}/{len(product['object_id']):,}",
                            flush=True,
                        )
                        next_report = ((completed // 1_000) + 1) * 1_000
            finally:
                tile.close()
    finally:
        for tile in tiles:
            tile.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if embeddings is None or completed != len(product["object_id"]):
        raise RuntimeError(
            f"timm extracted {completed:,} of {len(product['object_id']):,} morphology rows."
        )
    if not bool(torch.isfinite(embeddings).all()):
        raise RuntimeError("timm embeddings contain non-finite values.")
    expected_metadata["embedding_dim"] = int(embeddings.shape[1])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "object_id": list(product["object_id"]),
            "embedding": embeddings,
            "metadata": expected_metadata,
        },
        cache_path,
    )
    print(f"saved {cache_path}", flush=True)
    return embeddings
