from __future__ import annotations

"""Build a reusable CLAUDS catalogue of AION and pixel morphology features."""

import argparse
import json
import os
import shutil
import time
import types
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import FITSFixedWarning

from .morphology import (
    AIONGalaxy10MorphologyHead,
    AION_IMAGE_BAND_ALIAS,
    AION_IMAGE_INPUT_SIZE,
    DEFAULT_MORPHOLOGICAL_MISMATCH_THRESHOLD,
    GALAXY10_AION_CLASS_NAMES,
    MorphologyTile,
    _extract_cutout,
    collapse_galaxy10_morphology_probabilities,
    compute_pixel_morphology_batch,
    discover_morphology_image_paths,
    possible_morphological_mismatch,
)
from .utils import resolve_torch_device, set_random_seed


DEFAULT_BENCHMARK_REPO = "astronolan/galaxy10-aion"
DEFAULT_AION_MODEL = "polymathic-ai/aion-base"
MORPHOLOGY_COLUMN_NAMES = (
    "p_spiral",
    "p_bar",
    "p_elliptical_type",
    "axis_ellipticity",
    "concentration_C",
    "asymmetry_A",
    "possible_morphological_mismatch",
    "morphology_available",
)


@dataclass
class MorphologyCatalogueConfig:
    catalogue_path: Path = Path("data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits")
    output_catalogue_path: Path = Path(
        "data/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological.fits"
    )
    morphology_dir: Path = Path("data/clauds/images/tilesv5")
    cache_dir: Path = Path("cache/aion_morphology_catalogue")
    benchmark_repo: str = DEFAULT_BENCHMARK_REPO
    aion_model: str = DEFAULT_AION_MODEL
    aion_band_alias: str = AION_IMAGE_BAND_ALIAS
    device: str = "auto"
    benchmark_batch_size: int = 32
    target_batch_size: int = 32
    head_epochs: int = 100
    head_learning_rate: float = 1.0e-3
    head_weight_decay: float = 1.0e-4
    head_patience: int = 30
    min_cutout_weight_coverage: float = 0.90
    min_signal_to_noise: float = 5.0
    min_valid_hsc_bands: int = 3
    catalogue_flux_zeropoint: float = 23.0
    aion_hsc_image_zeropoint: float = 27.0
    mismatch_threshold: float = DEFAULT_MORPHOLOGICAL_MISMATCH_THRESHOLD
    max_target_rows: int | None = None
    stop_after_processed_rows: int | None = None
    seed: int = 42
    overwrite_output: bool = False
    force_benchmark_embeddings: bool = False
    force_head_training: bool = False
    force_target_features: bool = False

    def normalized(self) -> "MorphologyCatalogueConfig":
        values = asdict(self)
        for name in (
            "catalogue_path",
            "output_catalogue_path",
            "morphology_dir",
            "cache_dir",
        ):
            values[name] = Path(values[name])
        if values["benchmark_batch_size"] < 1 or values["target_batch_size"] < 1:
            raise ValueError("Batch sizes must be positive.")
        if not 0.0 <= values["min_cutout_weight_coverage"] <= 1.0:
            raise ValueError("min_cutout_weight_coverage must lie in [0, 1].")
        if values["min_signal_to_noise"] < 0.0:
            raise ValueError("min_signal_to_noise must be non-negative.")
        if not 1 <= values["min_valid_hsc_bands"] <= 5:
            raise ValueError("min_valid_hsc_bands must lie in [1, 5].")
        if values["max_target_rows"] is not None and values["max_target_rows"] < 1:
            raise ValueError("max_target_rows must be positive or None.")
        if (
            values["stop_after_processed_rows"] is not None
            and values["stop_after_processed_rows"] < 1
        ):
            raise ValueError("stop_after_processed_rows must be positive or None.")
        return MorphologyCatalogueConfig(**values)

    @property
    def embedding_cache_path(self) -> Path:
        return self.cache_dir / "galaxy10_aion_embeddings.npz"

    @property
    def head_path(self) -> Path:
        return self.cache_dir / "aion_galaxy10_morphology_head.pt"

    @property
    def assignment_path(self) -> Path:
        return self.cache_dir / "cosmos_tile_assignment.npz"

    @property
    def feature_dir(self) -> Path:
        return self.cache_dir / "cosmos_morphology_features"


def _enable_fused_aion_attention(model: torch.nn.Module) -> int:
    """Use PyTorch SDPA for AION self-attention during inference.

    AION 0.0.2 computes and materializes attention probabilities manually.
    SDPA is algebraically equivalent for the model's zero-dropout evaluation
    path and uses the fused CUDA kernel on supported GPUs.
    """
    from aion.fourm.fm_utils import Attention, NormAttention

    def attention_forward(module, x, mask=None):
        batch, n_tokens, channels = x.shape
        qkv = (
            module.qkv(x)
            .reshape(
                batch,
                n_tokens,
                3,
                module.num_heads,
                channels // module.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        if isinstance(module, NormAttention):
            query = module.q_norm(query)
            key = module.k_norm(key)
        attention_mask = None if mask is None else ~mask.unsqueeze(1)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=module.attn_drop.p if module.training else 0.0,
            scale=module.scale,
        )
        output = output.transpose(1, 2).reshape(batch, n_tokens, channels)
        return module.proj_drop(module.proj(output))

    if not hasattr(F, "scaled_dot_product_attention"):
        return 0
    patched = 0
    for module in model.modules():
        if isinstance(module, (Attention, NormAttention)) and not module.allow_zero_attn:
            module.forward = types.MethodType(attention_forward, module)
            patched += 1
    return patched


class FrozenAIONImageEncoder:
    """Own the frozen AION encoder and both survey-specific image codecs."""

    def __init__(self, model_name: str, device: torch.device):
        from aion import AION
        from aion.codecs import CodecManager
        from aion.modalities import HSCImage, LegacySurveyImage

        self.device = device
        self.model = AION.from_pretrained(model_name).to(device).eval()
        self.sdpa_attention_modules = _enable_fused_aion_attention(self.model)
        self.codec_manager = CodecManager(device=device)
        self.HSCImage = HSCImage
        self.LegacySurveyImage = LegacySurveyImage

    @torch.inference_mode()
    def encode_legacy(self, images: np.ndarray) -> torch.Tensor:
        flux = torch.from_numpy(np.asarray(images, dtype=np.float32)).to(self.device)
        modality = self.LegacySurveyImage(
            flux=flux,
            bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
        )
        tokens = self.codec_manager.encode(modality)
        key = self.LegacySurveyImage.token_key
        return self._mean_embedding({key: tokens[key]})

    @torch.inference_mode()
    def encode_hsc(self, images: np.ndarray) -> torch.Tensor:
        flux = torch.from_numpy(np.asarray(images, dtype=np.float32)).to(self.device)
        if flux.ndim != 4 or flux.shape[1] != 5:
            raise ValueError("HSC proxy images must have shape (batch, 5, height, width).")
        modality = self.HSCImage(
            flux=flux,
            bands=["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"],
        )
        tokens = self.codec_manager.encode(modality)
        key = self.HSCImage.token_key
        return self._mean_embedding({key: tokens[key]})

    def _mean_embedding(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        enabled = self.device.type == "cuda"
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=enabled,
        ):
            embeddings = self.model.encode(tokens, num_encoder_tokens=600)
        return embeddings.mean(dim=1).float()


def _nested_arrow_array_to_numpy(column: Any, n_rows: int) -> np.ndarray:
    array = column
    while hasattr(array, "values"):
        array = array.values
    values = np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float32)
    expected = n_rows * 4 * AION_IMAGE_INPUT_SIZE * AION_IMAGE_INPUT_SIZE
    if values.size != expected:
        raise RuntimeError(
            f"Unexpected image_bands size {values.size:,}; expected {expected:,}."
        )
    return values.reshape(n_rows, 4, AION_IMAGE_INPUT_SIZE, AION_IMAGE_INPUT_SIZE)


def download_galaxy10_aion_benchmark(config: MorphologyCatalogueConfig) -> Path:
    """Download the exact AION paper Galaxy10 split into the HF cache."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=config.benchmark_repo,
            repo_type="dataset",
            allow_patterns=("README.md", "data/*.parquet"),
        )
    )


def _iter_benchmark_batches(
    parquet_paths: Sequence[Path],
    *,
    batch_size: int,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    import pyarrow.parquet as pq

    for parquet_path in parquet_paths:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=("image_bands", "label"),
        ):
            images = _nested_arrow_array_to_numpy(batch.column(0), batch.num_rows)
            labels = np.asarray(batch.column(1).to_numpy(), dtype=np.int64)
            yield images, labels


def cache_galaxy10_aion_embeddings(
    config: MorphologyCatalogueConfig,
    *,
    encoder: FrozenAIONImageEncoder | None = None,
) -> dict[str, np.ndarray]:
    """Cache frozen AION mean embeddings for the exact benchmark split."""
    config = config.normalized()
    output_path = config.embedding_cache_path
    if output_path.exists() and not config.force_benchmark_embeddings:
        with np.load(output_path) as cached:
            return {name: np.asarray(cached[name]) for name in cached.files}

    snapshot = download_galaxy10_aion_benchmark(config)
    train_paths = sorted((snapshot / "data").glob("train-*.parquet"))
    test_paths = sorted((snapshot / "data").glob("test-*.parquet"))
    if not train_paths or not test_paths:
        raise FileNotFoundError(f"Benchmark parquet shards were not found under {snapshot}.")

    device = resolve_torch_device(config.device)
    encoder = encoder or FrozenAIONImageEncoder(config.aion_model, device)
    arrays: dict[str, np.ndarray] = {}
    for split, paths in (("train", train_paths), ("test", test_paths)):
        embeddings: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        completed = 0
        started = time.time()
        for images, batch_labels in _iter_benchmark_batches(
            paths,
            batch_size=config.benchmark_batch_size,
        ):
            embeddings.append(encoder.encode_legacy(images).cpu().numpy())
            labels.append(batch_labels)
            completed += len(batch_labels)
            if completed % 500 < len(batch_labels):
                rate = completed / max(time.time() - started, 1.0e-6)
                print(f"AION benchmark {split}: {completed:,} rows ({rate:.1f}/s)")
        arrays[f"{split}_embeddings"] = np.concatenate(embeddings).astype(np.float32)
        arrays[f"{split}_labels"] = np.concatenate(labels).astype(np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    return arrays


def _stratified_train_validation_indices(
    labels: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[np.ndarray] = []
    validation: list[np.ndarray] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_validation = max(1, int(round(len(indices) * validation_fraction)))
        validation.append(indices[:n_validation])
        train.append(indices[n_validation:])
    train_indices = np.concatenate(train)
    validation_indices = np.concatenate(validation)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def _classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    logits = logits.detach().clone()
    labels = labels.detach().clone()
    log_temperature = torch.nn.Parameter(torch.zeros((), device=logits.device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp().clamp_min(1.0e-6), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def train_aion_galaxy10_morphology_head(
    config: MorphologyCatalogueConfig,
    *,
    embedding_product: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Train and temperature-calibrate the paper's two-layer frozen-AION head."""
    config = config.normalized()
    if config.head_path.exists() and not config.force_head_training:
        return torch.load(config.head_path, map_location="cpu", weights_only=False)

    set_random_seed(config.seed)
    embedding_product = embedding_product or cache_galaxy10_aion_embeddings(config)
    train_embeddings = torch.from_numpy(embedding_product["train_embeddings"]).float()
    train_labels = torch.from_numpy(embedding_product["train_labels"]).long()
    test_embeddings = torch.from_numpy(embedding_product["test_embeddings"]).float()
    test_labels = torch.from_numpy(embedding_product["test_labels"]).long()
    train_indices, validation_indices = _stratified_train_validation_indices(
        train_labels.numpy(), validation_fraction=0.1, seed=config.seed
    )

    device = resolve_torch_device(config.device)
    head = AIONGalaxy10MorphologyHead(input_dim=train_embeddings.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.head_learning_rate,
        weight_decay=config.head_weight_decay,
    )
    generator = torch.Generator().manual_seed(config.seed)
    dataset = torch.utils.data.TensorDataset(
        train_embeddings[train_indices], train_labels[train_indices]
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        generator=generator,
    )
    validation_x = train_embeddings[validation_indices].to(device)
    validation_y = train_labels[validation_indices].to(device)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(config.head_epochs):
        head.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            validation_logits = head(validation_x)
            validation_loss = float(F.cross_entropy(validation_logits, validation_y).item())
            validation_accuracy = _classification_accuracy(validation_logits, validation_y)
        print(
            f"Galaxy10 head epoch {epoch + 1:03d}: "
            f"val_loss={validation_loss:.4f} val_accuracy={validation_accuracy:.4f}"
        )
        if validation_loss < best_loss - 1.0e-5:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu() for name, value in head.state_dict().items()}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= config.head_patience:
                break
    if best_state is None:
        raise RuntimeError("AION morphology-head training did not produce a checkpoint.")
    head.load_state_dict(best_state)
    head.to(device).eval()
    with torch.inference_mode():
        validation_logits = head(validation_x)
    temperature = _fit_temperature(validation_logits.detach(), validation_y)

    # Refit on every official training row for the validation-selected epoch count.
    set_random_seed(config.seed)
    head = AIONGalaxy10MorphologyHead(input_dim=train_embeddings.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.head_learning_rate,
        weight_decay=config.head_weight_decay,
    )
    full_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_embeddings, train_labels),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    for _ in range(best_epoch):
        head.train()
        for batch_x, batch_y in full_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    head.eval()
    head.temperature.fill_(temperature)
    with torch.inference_mode():
        test_logits = head(test_embeddings.to(device))
        test_accuracy = _classification_accuracy(test_logits, test_labels.to(device))
        test_nll = float(
            F.cross_entropy(test_logits / head.temperature, test_labels.to(device)).item()
        )
    checkpoint = {
        "state_dict": {name: value.detach().cpu() for name, value in head.state_dict().items()},
        "input_dim": int(train_embeddings.shape[1]),
        "hidden_dim": 256,
        "class_names": GALAXY10_AION_CLASS_NAMES,
        "benchmark_repo": config.benchmark_repo,
        "aion_model": config.aion_model,
        "temperature": temperature,
        "selected_epoch": best_epoch,
        "validation_loss": best_loss,
        "test_accuracy": test_accuracy,
        "test_nll": test_nll,
        "seed": config.seed,
    }
    config.head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, config.head_path)
    print(
        f"Saved AION morphology head: {config.head_path} "
        f"(test accuracy={test_accuracy:.4f}, temperature={temperature:.3f})"
    )
    if test_accuracy < 0.70:
        warnings.warn(
            f"AION morphology-head test accuracy is unexpectedly low ({test_accuracy:.3f}).",
            RuntimeWarning,
            stacklevel=2,
        )
    return checkpoint


def load_aion_galaxy10_morphology_head(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[AIONGalaxy10MorphologyHead, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    head = AIONGalaxy10MorphologyHead(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
        temperature=float(checkpoint.get("temperature", 1.0)),
    )
    head.load_state_dict(checkpoint["state_dict"])
    return head.to(device).eval(), checkpoint


def _tile_sky_bounds(tile: MorphologyTile) -> tuple[float, float, float, float]:
    height, width = tile.shape
    x = np.asarray([0.0, width - 1.0, width - 1.0, 0.0])
    y = np.asarray([0.0, 0.0, height - 1.0, height - 1.0])
    ra, dec = tile.wcs.all_pix2world(x, y, 0)
    return (
        float(np.nanmin(ra)),
        float(np.nanmax(ra)),
        float(np.nanmin(dec)),
        float(np.nanmax(dec)),
    )


def select_catalogue_tiles(
    morphology_dir: str | Path,
    ra: np.ndarray,
    dec: np.ndarray,
) -> tuple[list[MorphologyTile], np.ndarray]:
    """Load only tiles whose corner bounds overlap the catalogue footprint."""
    catalogue_bounds = (
        float(np.nanmin(ra)),
        float(np.nanmax(ra)),
        float(np.nanmin(dec)),
        float(np.nanmax(dec)),
    )
    selected: list[MorphologyTile] = []
    bounds: list[tuple[float, float, float, float]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        for path in discover_morphology_image_paths(morphology_dir):
            tile = MorphologyTile(path)
            tile_bounds = _tile_sky_bounds(tile)
            overlaps = not (
                tile_bounds[1] < catalogue_bounds[0]
                or tile_bounds[0] > catalogue_bounds[1]
                or tile_bounds[3] < catalogue_bounds[2]
                or tile_bounds[2] > catalogue_bounds[3]
            )
            if overlaps:
                selected.append(tile)
                bounds.append(tile_bounds)
            else:
                tile.close()
    return selected, np.asarray(bounds, dtype=np.float64)


def assign_tiles_to_positions_indexed(
    ra: np.ndarray,
    dec: np.ndarray,
    tiles: Sequence[MorphologyTile],
    tile_bounds: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign best-overlap tiles using an RA index before exact WCS transforms."""
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    n_rows = len(ra)
    eligible = np.ones(n_rows, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    order = np.argsort(ra, kind="stable")
    sorted_ra = ra[order]
    tile_index = np.full(n_rows, -1, dtype=np.int16)
    x_out = np.full(n_rows, np.nan, dtype=np.float32)
    y_out = np.full(n_rows, np.nan, dtype=np.float32)
    best_margin = np.full(n_rows, -np.inf, dtype=np.float32)
    for index, (tile, bounds) in enumerate(zip(tiles, tile_bounds)):
        left = int(np.searchsorted(sorted_ra, bounds[0], side="left"))
        right = int(np.searchsorted(sorted_ra, bounds[1], side="right"))
        candidates = order[left:right]
        candidates = candidates[
            eligible[candidates]
            & (dec[candidates] >= bounds[2])
            & (dec[candidates] <= bounds[3])
        ]
        if not len(candidates):
            continue
        try:
            x, y = tile.wcs.all_world2pix(ra[candidates], dec[candidates], 0)
        except Exception:
            continue
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        height, width = tile.shape
        inside = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= 0.0)
            & (x < width)
            & (y >= 0.0)
            & (y < height)
        )
        if not inside.any():
            continue
        rows = candidates[inside]
        x = x[inside]
        y = y[inside]
        margin = np.minimum.reduce([x, y, width - 1.0 - x, height - 1.0 - y])
        update = margin > best_margin[rows]
        rows = rows[update]
        tile_index[rows] = index
        x_out[rows] = x[update].astype(np.float32)
        y_out[rows] = y[update].astype(np.float32)
        best_margin[rows] = margin[update].astype(np.float32)
    return tile_index, x_out, y_out


def _read_catalogue_coordinates(
    config: MorphologyCatalogueConfig,
) -> dict[str, np.ndarray]:
    import fitsio

    hsc_flux_columns = tuple(f"FLUX_CMODEL_HSC-{band}" for band in "GRIZY")
    columns = fitsio.read(
        config.catalogue_path,
        ext=1,
        columns=("ID", "RA", "DEC", "isStar", *hsc_flux_columns),
    )
    stop = len(columns) if config.max_target_rows is None else config.max_target_rows
    return {
        "id": np.asarray(columns["ID"][:stop]),
        "ra": np.asarray(columns["RA"][:stop], dtype=np.float64),
        "dec": np.asarray(columns["DEC"][:stop], dtype=np.float64),
        "is_star": np.asarray(columns["isStar"][:stop], dtype=bool),
        "hsc_flux": np.stack(
            [np.asarray(columns[name][:stop], dtype=np.float32) for name in hsc_flux_columns],
            axis=1,
        ),
        "catalogue_n_rows": np.asarray([len(columns)], dtype=np.int64),
    }


def build_or_load_tile_assignment(
    config: MorphologyCatalogueConfig,
    catalogue: dict[str, np.ndarray],
) -> tuple[list[MorphologyTile], np.ndarray, np.ndarray, np.ndarray]:
    config = config.normalized()
    tiles, bounds = select_catalogue_tiles(
        config.morphology_dir, catalogue["ra"], catalogue["dec"]
    )
    manifest = np.asarray(
        [str(tile.image_path.relative_to(config.morphology_dir)) for tile in tiles]
    )
    if config.assignment_path.exists() and not config.force_target_features:
        with np.load(config.assignment_path) as cached:
            if (
                np.array_equal(cached["tile_manifest"], manifest)
                and len(cached["tile_index"]) == len(catalogue["ra"])
                and np.array_equal(cached["object_id"], catalogue["id"])
            ):
                return (
                    tiles,
                    np.asarray(cached["tile_index"]),
                    np.asarray(cached["x_image"]),
                    np.asarray(cached["y_image"]),
                )
    print(f"Assigning {len(catalogue['ra']):,} catalogue rows to {len(tiles):,} tiles")
    assigned = assign_tiles_to_positions_indexed(
        catalogue["ra"],
        catalogue["dec"],
        tiles,
        bounds,
        eligible=~catalogue["is_star"],
    )
    config.assignment_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        config.assignment_path,
        object_id=catalogue["id"],
        tile_manifest=manifest,
        tile_index=assigned[0],
        x_image=assigned[1],
        y_image=assigned[2],
    )
    return tiles, *assigned


def _open_feature_array(
    path: Path,
    *,
    dtype: np.dtype | type,
    n_rows: int,
    fill: float | int | bool,
    force: bool,
) -> np.memmap:
    if path.exists() and not force:
        values = np.load(path, mmap_mode="r+")
        if values.shape != (n_rows,) or values.dtype != np.dtype(dtype):
            raise RuntimeError(f"Incompatible cached feature array: {path}")
        return values
    values = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(n_rows,))
    values[:] = fill
    values.flush()
    return values


def open_target_feature_arrays(
    config: MorphologyCatalogueConfig,
    n_rows: int,
) -> dict[str, np.memmap]:
    config.feature_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: _open_feature_array(
            config.feature_dir / f"{name}.npy",
            dtype=np.float32,
            n_rows=n_rows,
            fill=np.nan,
            force=config.force_target_features,
        )
        for name in MORPHOLOGY_COLUMN_NAMES[:6]
    }
    arrays["possible_morphological_mismatch"] = _open_feature_array(
        config.feature_dir / "possible_morphological_mismatch.npy",
        dtype=np.bool_,
        n_rows=n_rows,
        fill=False,
        force=config.force_target_features,
    )
    arrays["morphology_available"] = _open_feature_array(
        config.feature_dir / "morphology_available.npy",
        dtype=np.bool_,
        n_rows=n_rows,
        fill=False,
        force=config.force_target_features,
    )
    arrays["status"] = _open_feature_array(
        config.feature_dir / "status.npy",
        dtype=np.uint8,
        n_rows=n_rows,
        fill=0,
        force=config.force_target_features,
    )
    return arrays


def _flush_feature_arrays(arrays: dict[str, np.memmap]) -> None:
    for values in arrays.values():
        values.flush()


def compute_cosmos_morphology_features(
    config: MorphologyCatalogueConfig,
    *,
    encoder: FrozenAIONImageEncoder | None = None,
) -> dict[str, Any]:
    """Compute the requested eight catalogue columns with resumable memmaps."""
    config = config.normalized()
    catalogue = _read_catalogue_coordinates(config)
    n_rows = len(catalogue["id"])
    tiles, tile_index, x_image, y_image = build_or_load_tile_assignment(config, catalogue)
    arrays = open_target_feature_arrays(config, n_rows)
    device = resolve_torch_device(config.device)
    encoder = encoder or FrozenAIONImageEncoder(config.aion_model, device)
    head, checkpoint = load_aion_galaxy10_morphology_head(config.head_path, device=device)
    started = time.time()
    completed_before = int(np.count_nonzero(arrays["status"]))
    completed = completed_before

    @torch.inference_mode()
    def flush(rows: list[int], cutouts: list[np.ndarray]) -> None:
        nonlocal completed
        if not rows:
            return
        row_array = np.asarray(rows, dtype=np.int64)
        batch = np.stack(cutouts).astype(np.float32, copy=False)
        measured = compute_pixel_morphology_batch(
            batch,
            min_signal_to_noise=config.min_signal_to_noise,
        )
        pixel_valid = measured["morphology_pixel_valid"]
        invalid_rows = row_array[~pixel_valid]
        arrays["status"][invalid_rows] = 1
        if pixel_valid.any():
            pixel_rows = row_array[pixel_valid]
            for name in ("axis_ellipticity", "concentration_C", "asymmetry_A"):
                arrays[name][pixel_rows] = measured[name][pixel_valid]

            hsc_flux = catalogue["hsc_flux"][pixel_rows].astype(np.float32, copy=True)
            finite_positive = np.isfinite(hsc_flux) & (hsc_flux > 0.0)
            aion_valid = finite_positive.sum(axis=1) >= config.min_valid_hsc_bands
            no_aion_rows = pixel_rows[~aion_valid]
            arrays["status"][no_aion_rows] = 1
            if aion_valid.any():
                valid_rows = pixel_rows[aion_valid]
                valid_cutouts = batch[pixel_valid][aion_valid]
                valid_flux = hsc_flux[aion_valid]
                valid_mask = finite_positive[aion_valid]
                median_flux = np.nanmedian(
                    np.where(valid_mask, valid_flux, np.nan), axis=1
                )
                valid_flux = np.where(valid_mask, valid_flux, median_flux[:, None])
                aperture_flux = measured["aperture_flux"][pixel_valid][aion_valid]
                zeropoint_scale = 10.0 ** (
                    (config.aion_hsc_image_zeropoint - config.catalogue_flux_zeropoint)
                    / 2.5
                )
                band_scales = (
                    valid_flux * float(zeropoint_scale) / aperture_flux[:, None]
                )
                proxy_images = (
                    valid_cutouts[:, None, :, :]
                    * band_scales[:, :, None, None]
                )
                embeddings = encoder.encode_hsc(proxy_images)
                probabilities = head.predict_proba(embeddings)
                collapsed = collapse_galaxy10_morphology_probabilities(probabilities)
                for name in ("p_spiral", "p_bar", "p_elliptical_type"):
                    arrays[name][valid_rows] = collapsed[name].float().cpu().numpy()
                arrays["possible_morphological_mismatch"][valid_rows] = (
                    possible_morphological_mismatch(
                        arrays["p_elliptical_type"][valid_rows],
                        arrays["axis_ellipticity"][valid_rows],
                        threshold=config.mismatch_threshold,
                    )
                )
                arrays["morphology_available"][valid_rows] = True
                arrays["status"][valid_rows] = 2
        completed += len(row_array)
        rows.clear()
        cutouts.clear()

    for current_tile, tile in enumerate(tiles):
        if (
            config.stop_after_processed_rows is not None
            and completed - completed_before >= config.stop_after_processed_rows
        ):
            break
        rows = np.flatnonzero((tile_index == current_tile) & (arrays["status"] == 0))
        if config.stop_after_processed_rows is not None:
            remaining = config.stop_after_processed_rows - (completed - completed_before)
            rows = rows[:remaining]
        if not len(rows):
            tile.close()
            continue
        batch_rows: list[int] = []
        batch_cutouts: list[np.ndarray] = []
        try:
            for row in rows:
                cutout, stats = _extract_cutout(
                    tile,
                    float(x_image[row]),
                    float(y_image[row]),
                    cutout_size=AION_IMAGE_INPUT_SIZE,
                    background_mode="median",
                    image_flux_scale=1.0,
                )
                if stats["weight_coverage"] < config.min_cutout_weight_coverage:
                    arrays["status"][row] = 1
                    completed += 1
                    continue
                batch_rows.append(int(row))
                batch_cutouts.append(cutout)
                if len(batch_rows) >= config.target_batch_size:
                    flush(batch_rows, batch_cutouts)
            flush(batch_rows, batch_cutouts)
        finally:
            tile.close()
        _flush_feature_arrays(arrays)
        elapsed = max(time.time() - started, 1.0e-6)
        new_completed = completed - completed_before
        rate = new_completed / elapsed
        print(
            f"COSMOS morphology tile {current_tile + 1:,}/{len(tiles):,}: "
            f"processed={completed:,}/{n_rows:,}, available="
            f"{int(np.count_nonzero(arrays['morphology_available'])):,}, rate={rate:.1f}/s"
        )
    _flush_feature_arrays(arrays)
    metadata = {
        "n_target_rows": n_rows,
        "catalogue_n_rows": int(catalogue["catalogue_n_rows"][0]),
        "n_assigned": int(np.count_nonzero(tile_index >= 0)),
        "n_available": int(np.count_nonzero(arrays["morphology_available"])),
        "n_possible_mismatch": int(
            np.count_nonzero(arrays["possible_morphological_mismatch"])
        ),
        "n_tiles": len(tiles),
        "n_processed": int(np.count_nonzero(arrays["status"])),
        "processing_complete": bool(
            np.all(arrays["status"][tile_index >= 0] != 0)
        ),
        "aion_head_test_accuracy": float(checkpoint["test_accuracy"]),
        "aion_head_temperature": float(checkpoint["temperature"]),
        "aion_model": config.aion_model,
        "aion_fused_sdpa": bool(encoder.sdpa_attention_modules),
        "aion_sdpa_attention_modules": int(encoder.sdpa_attention_modules),
        "benchmark_repo": config.benchmark_repo,
        "aion_target_input": (
            "five-band HSC proxy: CLAUDS u/uS spatial template scaled by each "
            "object's HSC grizy cmodel flux ratios"
        ),
        "out_of_domain_warning": (
            "The AION morphology head was trained on true four-band Legacy Survey "
            "images; target inference uses a five-band proxy with shared u/uS morphology."
        ),
        "catalogue_flux_zeropoint": config.catalogue_flux_zeropoint,
        "aion_hsc_image_zeropoint": config.aion_hsc_image_zeropoint,
        "min_valid_hsc_bands": config.min_valid_hsc_bands,
        "min_signal_to_noise": config.min_signal_to_noise,
        "min_cutout_weight_coverage": config.min_cutout_weight_coverage,
        "mismatch_definition": (
            "abs(p_elliptical_type - (1 - axis_ellipticity)) >= "
            f"{config.mismatch_threshold:g}"
        ),
    }
    (config.feature_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"arrays": arrays, "metadata": metadata}


def write_morphological_catalogue(
    config: MorphologyCatalogueConfig,
    feature_product: dict[str, Any] | None = None,
) -> Path:
    """Copy the source FITS and append morphology columns to table extension 1."""
    import fitsio

    config = config.normalized()
    feature_product = feature_product or {
        "arrays": open_target_feature_arrays(
            config,
            int(_read_catalogue_coordinates(config)["catalogue_n_rows"][0]),
        )
    }
    arrays = feature_product["arrays"]
    if feature_product.get("metadata", {}).get("processing_complete") is False:
        raise RuntimeError("Refusing to write a catalogue from an incomplete feature pass.")
    if not config.assignment_path.exists():
        raise FileNotFoundError(f"Missing target assignment cache: {config.assignment_path}")
    with np.load(config.assignment_path) as assignment:
        assigned = np.asarray(assignment["tile_index"]) >= 0
    if np.any(np.asarray(arrays["status"])[assigned] == 0):
        raise RuntimeError("Refusing to write while assigned morphology rows remain unprocessed.")
    with fitsio.FITS(config.catalogue_path) as source_fits:
        source_rows = int(source_fits[1].get_nrows())
    if len(arrays["p_spiral"]) != source_rows:
        raise RuntimeError(
            "A full catalogue can only be written from full-length features; "
            f"features have {len(arrays['p_spiral']):,} rows and source has {source_rows:,}."
        )
    output = config.output_catalogue_path
    if output.exists() and not config.overwrite_output:
        with fitsio.FITS(output) as fits_file:
            names = set(fits_file[1].get_colnames())
            if set(MORPHOLOGY_COLUMN_NAMES).issubset(names):
                return output
        raise FileExistsError(f"Output exists without all morphology columns: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(config.catalogue_path, temporary)
    try:
        with fitsio.FITS(temporary, mode="rw") as fits_file:
            table = fits_file[1]
            existing = set(table.get_colnames())
            overlap = existing.intersection(MORPHOLOGY_COLUMN_NAMES)
            if overlap:
                raise RuntimeError(f"Source already contains morphology columns: {sorted(overlap)}")
            for name in MORPHOLOGY_COLUMN_NAMES:
                print(f"Inserting FITS column: {name}")
                table.insert_column(name, np.asarray(arrays[name]))
            table.write_key("MORPHMOD", config.aion_model, comment="AION morphology encoder")
            table.write_key("MORPHDS", config.benchmark_repo, comment="Morphology probe dataset")
            table.write_key("MORPHSNR", config.min_signal_to_noise, comment="Minimum pixel S/N")
            table.write_key("MORPHMIS", config.mismatch_threshold, comment="Mismatch threshold")
            table.write_history(
                "AION probabilities transfer a Legacy-trained head to five-band HSC "
                "proxies using CLAUDS u/uS morphology and HSC grizy flux ratios."
            )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return output


def verify_morphological_catalogue(path: str | Path) -> dict[str, Any]:
    import fitsio

    with fitsio.FITS(path) as fits_file:
        table = fits_file[1]
        names = set(table.get_colnames())
        missing = set(MORPHOLOGY_COLUMN_NAMES).difference(names)
        if missing:
            raise RuntimeError(f"Morphology catalogue is missing columns: {sorted(missing)}")
        sample = table.read(
            rows=np.asarray([0, table.get_nrows() - 1]),
            columns=MORPHOLOGY_COLUMN_NAMES,
        )
        return {
            "path": str(path),
            "n_rows": int(table.get_nrows()),
            "n_columns": len(table.get_colnames()),
            "size_bytes": Path(path).stat().st_size,
            "sample_dtype": sample.dtype.descr,
        }


def build_morphological_catalogue(
    config: MorphologyCatalogueConfig | None = None,
) -> dict[str, Any]:
    config = (config or MorphologyCatalogueConfig()).normalized()
    if config.max_target_rows is not None:
        raise ValueError("The final catalogue build requires max_target_rows=None.")
    embeddings = cache_galaxy10_aion_embeddings(config)
    train_aion_galaxy10_morphology_head(config, embedding_product=embeddings)
    features = compute_cosmos_morphology_features(config)
    output = write_morphological_catalogue(config, features)
    verification = verify_morphological_catalogue(output)
    return {
        "output_catalogue_path": output,
        "head_path": config.head_path,
        "feature_metadata": features["metadata"],
        "verification": verification,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = MorphologyCatalogueConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("train-head", "features", "catalogue", "all"), default="all", nargs="?"
    )
    parser.add_argument("--catalogue-path", type=Path, default=defaults.catalogue_path)
    parser.add_argument("--output-catalogue-path", type=Path, default=defaults.output_catalogue_path)
    parser.add_argument("--morphology-dir", type=Path, default=defaults.morphology_dir)
    parser.add_argument("--cache-dir", type=Path, default=defaults.cache_dir)
    parser.add_argument("--benchmark-repo", default=defaults.benchmark_repo)
    parser.add_argument("--aion-model", default=defaults.aion_model)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--benchmark-batch-size", type=int, default=defaults.benchmark_batch_size)
    parser.add_argument("--target-batch-size", type=int, default=defaults.target_batch_size)
    parser.add_argument("--head-epochs", type=int, default=defaults.head_epochs)
    parser.add_argument("--min-signal-to-noise", type=float, default=defaults.min_signal_to_noise)
    parser.add_argument(
        "--min-cutout-weight-coverage", type=float, default=defaults.min_cutout_weight_coverage
    )
    parser.add_argument("--mismatch-threshold", type=float, default=defaults.mismatch_threshold)
    parser.add_argument("--max-target-rows", type=int, default=None)
    parser.add_argument("--stop-after-processed-rows", type=int, default=None)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--force-benchmark-embeddings", action="store_true")
    parser.add_argument("--force-head-training", action="store_true")
    parser.add_argument("--force-target-features", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = MorphologyCatalogueConfig(
        catalogue_path=args.catalogue_path,
        output_catalogue_path=args.output_catalogue_path,
        morphology_dir=args.morphology_dir,
        cache_dir=args.cache_dir,
        benchmark_repo=args.benchmark_repo,
        aion_model=args.aion_model,
        device=args.device,
        benchmark_batch_size=args.benchmark_batch_size,
        target_batch_size=args.target_batch_size,
        head_epochs=args.head_epochs,
        min_signal_to_noise=args.min_signal_to_noise,
        min_cutout_weight_coverage=args.min_cutout_weight_coverage,
        mismatch_threshold=args.mismatch_threshold,
        max_target_rows=args.max_target_rows,
        stop_after_processed_rows=args.stop_after_processed_rows,
        overwrite_output=args.overwrite_output,
        force_benchmark_embeddings=args.force_benchmark_embeddings,
        force_head_training=args.force_head_training,
        force_target_features=args.force_target_features,
    ).normalized()
    if args.command == "train-head":
        product = cache_galaxy10_aion_embeddings(config)
        checkpoint = train_aion_galaxy10_morphology_head(config, embedding_product=product)
        print(json.dumps({key: value for key, value in checkpoint.items() if key != "state_dict"}, indent=2, default=str))
        return
    if args.command == "features":
        train_aion_galaxy10_morphology_head(config)
        product = compute_cosmos_morphology_features(config)
        print(json.dumps(product["metadata"], indent=2))
        return
    if args.command == "catalogue":
        output = write_morphological_catalogue(config)
        print(json.dumps(verify_morphological_catalogue(output), indent=2))
        return
    result = build_morphological_catalogue(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
