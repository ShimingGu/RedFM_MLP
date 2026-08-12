from __future__ import annotations

"""Native HSC grizy image extraction for AION post-training experiments."""

import fcntl
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .clauds_bands import HSC_AION_BANDS
from .dataset import CLAUDSPhotoZDataset
from .models import encode_aion_tokens, encode_hsc_aion_image_tokens
from .multiband_morphology_catalogue import (
    BandImageTile,
    MultibandMorphologyConfig,
    _read_catalogue,
    build_or_load_band_assignment,
    build_or_load_image_manifest,
)


GRIZY = tuple(HSC_AION_BANDS)


def _source_rows_for_object_ids(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
) -> np.ndarray:
    """Locate unique target IDs in an arbitrary source-ID ordering."""
    source = np.asarray(source_ids)
    target = np.asarray(target_ids, dtype=source.dtype)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("Object-ID arrays must be one-dimensional.")
    if len(np.unique(target)) != len(target):
        raise ValueError("Target object IDs must be unique.")
    sorter = np.argsort(source, kind="stable")
    insertion = np.searchsorted(source, target, sorter=sorter)
    in_range = insertion < len(source)
    rows = np.full(len(target), -1, dtype=np.int64)
    rows[in_range] = sorter[insertion[in_range]]
    matched = in_range & (source[rows.clip(min=0)] == target)
    if not matched.all():
        missing = target[~matched][:5].tolist()
        raise KeyError(f"Object IDs are absent from image assignments: {missing}")
    return rows


def _image_config(
    *,
    catalogue_path: str | Path,
    hsc_image_dir: str | Path,
    assignment_cache_dir: str | Path,
    fits_backend: str,
) -> MultibandMorphologyConfig:
    return MultibandMorphologyConfig(
        catalogue_path=Path(catalogue_path),
        hsc_image_dir=Path(hsc_image_dir),
        cache_dir=Path(assignment_cache_dir),
        bands=GRIZY,
        fits_backend=fits_backend,
        max_target_rows=None,
    ).normalized()


def ensure_grizy_assignments(
    *,
    catalogue_path: str | Path,
    hsc_image_dir: str | Path,
    assignment_cache_dir: str | Path,
    fits_backend: str = "auto",
) -> tuple[Any, dict[str, Path]]:
    """Build the missing lightweight WCS assignment caches, including HSC-Y."""
    config = _image_config(
        catalogue_path=catalogue_path,
        hsc_image_dir=hsc_image_dir,
        assignment_cache_dir=assignment_cache_dir,
        fits_backend=fits_backend,
    )
    manifest = build_or_load_image_manifest(config)
    assignment_dir = config.cache_dir / "assignments"
    assignment_dir.mkdir(parents=True, exist_ok=True)
    lock_path = assignment_dir / ".grizy_assignment.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        catalogue = _read_catalogue(config)
        for band in GRIZY:
            build_or_load_band_assignment(config, manifest, band, catalogue)
    paths = {band: config.assignment_path(band) for band in GRIZY}
    return manifest, paths


def load_grizy_assignments(
    object_ids: list[Any] | np.ndarray,
    *,
    catalogue_path: str | Path,
    hsc_image_dir: str | Path,
    assignment_cache_dir: str | Path,
    fits_backend: str = "auto",
) -> tuple[Any, dict[str, dict[str, np.ndarray]], np.ndarray]:
    """Return selected-row tile and pixel assignments in catalogue-product order."""
    manifest, paths = ensure_grizy_assignments(
        catalogue_path=catalogue_path,
        hsc_image_dir=hsc_image_dir,
        assignment_cache_dir=assignment_cache_dir,
        fits_backend=fits_backend,
    )
    target_ids = np.asarray(object_ids)
    first_path = paths[GRIZY[0]]
    with np.load(first_path) as cached:
        source_ids = np.asarray(cached["object_id"])
    source_rows = _source_rows_for_object_ids(source_ids, target_ids)

    assignments: dict[str, dict[str, np.ndarray]] = {}
    for band in GRIZY:
        with np.load(paths[band]) as cached:
            if not np.array_equal(np.asarray(cached["object_id"]), source_ids):
                raise RuntimeError(f"{band}-band assignment object IDs do not match.")
            assignments[band] = {
                "tile_index": np.asarray(cached["tile_index"], dtype=np.int32)[source_rows],
                "x_image": np.asarray(cached["x_image"], dtype=np.float32)[source_rows],
                "y_image": np.asarray(cached["y_image"], dtype=np.float32)[source_rows],
            }
    return manifest, assignments, source_rows


@torch.inference_mode()
def extract_grizy_image_aion_embeddings(
    dataset: CLAUDSPhotoZDataset,
    *,
    aion: Any,
    codec_manager: Any,
    device: torch.device,
    catalogue_path: str | Path,
    hsc_image_dir: str | Path,
    assignment_cache_dir: str | Path,
    batch_size: int = 8,
    min_cutout_coverage: float = 0.90,
    fits_backend: str = "auto",
) -> dict[str, Any]:
    """Extract joint magnitude+real-image AION embeddings for image-covered rows."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if not 0.0 <= min_cutout_coverage <= 1.0:
        raise ValueError("min_cutout_coverage must lie in [0, 1].")
    manifest, assignments, _ = load_grizy_assignments(
        dataset.object_ids,
        catalogue_path=catalogue_path,
        hsc_image_dir=hsc_image_dir,
        assignment_cache_dir=assignment_cache_dir,
        fits_backend=fits_backend,
    )
    tile_matrix = np.stack(
        [assignments[band]["tile_index"] for band in GRIZY],
        axis=1,
    )
    assigned = np.all(tile_matrix >= 0, axis=1)
    candidate_rows = np.flatnonzero(assigned)
    if not len(candidate_rows):
        raise RuntimeError("No selected catalogue rows have complete grizy image assignments.")

    tile_keys, inverse = np.unique(
        tile_matrix[candidate_rows],
        axis=0,
        return_inverse=True,
    )
    grouped_order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=len(tile_keys))
    boundaries = np.concatenate(([0], np.cumsum(counts)))

    embedding_parts: list[torch.Tensor] = []
    token_parts: dict[str, list[torch.Tensor]] = {}
    retained_parts: list[np.ndarray] = []
    processed = 0
    for group_index, tile_indices in enumerate(tile_keys):
        group_rows = candidate_rows[
            grouped_order[boundaries[group_index] : boundaries[group_index + 1]]
        ]
        tiles = [
            BandImageTile(
                manifest.path[int(tile_index)],
                manifest.kind[int(tile_index)],
                fits_backend=fits_backend,
            )
            for tile_index in tile_indices
        ]
        try:
            for start in range(0, len(group_rows), batch_size):
                local_rows = group_rows[start : start + batch_size]
                images = np.zeros(
                    (len(local_rows), len(GRIZY), 96, 96),
                    dtype=np.float32,
                )
                coverage = np.zeros((len(local_rows), len(GRIZY)), dtype=np.float32)
                for band_index, (band, tile) in enumerate(zip(GRIZY, tiles)):
                    x = assignments[band]["x_image"][local_rows]
                    y = assignments[band]["y_image"][local_rows]
                    for item_index, (x_value, y_value) in enumerate(zip(x, y)):
                        cutout = tile.extract(float(x_value), float(y_value))
                        images[item_index, band_index] = cutout["background_subtracted"]
                        coverage[item_index, band_index] = cutout["coverage"]
                usable = np.all(coverage >= min_cutout_coverage, axis=1)
                if usable.any():
                    kept = local_rows[usable]
                    hsc_batch = {
                        f"{band}_mag": dataset.hsc_features[f"{band}_mag"][kept]
                        for band in GRIZY
                    }
                    tokens = encode_hsc_aion_image_tokens(
                        hsc_batch, images[usable], codec_manager, device=device
                    )
                    embedding = encode_aion_tokens(
                        tokens, aion, device=device
                    ).mean(dim=1)
                    retained_parts.append(kept)
                    embedding_parts.append(embedding.float().cpu())
                    for key, value in tokens.items():
                        token_parts.setdefault(key, []).append(value.cpu())
                processed += len(local_rows)
        finally:
            for tile in tiles:
                tile.close()
        if (
            group_index == 0
            or (group_index + 1) % 50 == 0
            or group_index + 1 == len(tile_keys)
        ):
            retained = sum(len(rows) for rows in retained_parts)
            print(
                f"AION grizy images: groups={group_index + 1:,}/{len(tile_keys):,} "
                f"processed={processed:,}/{len(candidate_rows):,} retained={retained:,}",
                flush=True,
            )

    if not embedding_parts:
        raise RuntimeError("No grizy cutouts passed the coverage requirement.")
    retained_rows = np.concatenate(retained_parts)
    embeddings = torch.cat(embedding_parts)
    order = np.argsort(retained_rows, kind="stable")
    retained_rows = retained_rows[order]
    order_tensor = torch.as_tensor(order, dtype=torch.long)
    embeddings = embeddings[order_tensor]
    tokens = {
        key: torch.cat(parts)[order_tensor]
        for key, parts in token_parts.items()
    }
    metadata = {
        "representation": "aion_hsc_grizy_magnitudes_plus_native_grizy_images",
        "bands": list(GRIZY),
        "image_shape": [5, 96, 96],
        "image_root": str(Path(hsc_image_dir).resolve()),
        "assignment_cache_dir": str(Path(assignment_cache_dir).resolve()),
        "manifest_fingerprint": manifest.fingerprint,
        "fits_backend": fits_backend,
        "min_cutout_coverage": float(min_cutout_coverage),
        "n_input_rows": len(dataset),
        "n_complete_assignments": int(assigned.sum()),
        "n_retained_rows": int(len(retained_rows)),
        "n_rejected_by_coverage": int(len(candidate_rows) - len(retained_rows)),
    }
    return {
        "embeddings": embeddings,
        "tokens": tokens,
        "retained_rows": retained_rows,
        "metadata": metadata,
    }
