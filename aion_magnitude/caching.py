from __future__ import annotations
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from torch.utils.data import DataLoader

from .clauds_bands import HSC_AION_BANDS, REDSHIFT_COLUMNS, OBJECT_ID_COLUMN
from .utils import (
    select_torch_device, resolve_torch_device, set_random_seed, make_redshift_grid,
    apply_numpy_mask_to_tensor_dict, load_cached_product,
)
from .dataset import (
    CLAUDSPhotoZDataset,
    collate_clauds_photoz,
    build_raw_clauds_photoz_dataset,
    resolve_include_grizy_in_mlp,
    make_split_labels,
    split_metadata,
)
from .models import (
    load_frozen_aion,
    extract_hsc_aion_embedding,
    validate_cached_aion_mag_adjustment,
    aion_mag_adjustment_metadata,
)
from .config import AIONMagnitudeConfig, make_magnitude_config, resolve_training_paths


def extract_aion_embeddings_to_memory(
    dataset: CLAUDSPhotoZDataset,
    aion,
    codec_manager,
    batch_size: int = 512,
    num_workers: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    device = resolve_torch_device(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_clauds_photoz,
    )

    embeddings = []
    for batch in loader:
        embedding = extract_hsc_aion_embedding(batch.hsc_batch, aion, codec_manager, device=device)
        embeddings.append(embedding.cpu())

    return torch.cat(embeddings, dim=0)


def save_cached_product(
    path: str | Path,
    dataset: CLAUDSPhotoZDataset,
    aion_embeddings: torch.Tensor,
    feature_names: Sequence[str],
    split_labels: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    product = {
        "object_id": dataset.object_ids,
        "field": dataset.fields,
        "z_spec": dataset.z_spec,
        "redshift_reference": dataset.redshift_reference,
        "aion_embedding": aion_embeddings.cpu(),
        "extra_features": dataset.extra_features.cpu(),
        "feature_names": list(feature_names),
        "split_labels": None if split_labels is None else list(split_labels),
        "metadata": dict(metadata or {}),
    }
    torch.save(product, path)





def ensure_cached_product_redshift_reference(
    product: dict[str, Any],
    *,
    catalogue_path: str | Path,
    split_output_dir: str | Path,
    split_chunk_size: int = 250_000,
    overwrite_split_cache: bool = False,
    max_rows: int | None = 20_000,
    sample_mode: str = "head",
    sample_row_start: int | None = None,
    sample_row_stop: int | None = None,
    sample_seed: int = 42,
    sample_require_valid_bands: Sequence[str] = (),
    field_column: str | None = None,
    target_redshift_column: str = REDSHIFT_COLUMNS["zphot"],
    z_min: float = 0.0,
    z_max: float = 6.0,
    redshift_include_min: bool = True,
    redshift_include_max: bool = True,
    n_z_bins: int = 300,
    mag_zero_point: float = 23.0,
    hsc_mag_faint_limits: Mapping[str, float | None] | None = None,
    extra_bands: Sequence[str] | None = None,
    extra_band_invalid_fill: str | float = "median",
    extra_band_include_valid_flags: bool = False,
    aion_mag_adjustment_path: str | Path | None = None,
) -> dict[str, Any]:
    if product.get("redshift_reference"):
        return product

    raw_dataset, _, metadata = build_raw_clauds_photoz_dataset(
        catalogue_path,
        split_output_dir,
        split_chunk_size=split_chunk_size,
        overwrite_split_cache=overwrite_split_cache,
        max_rows=max_rows,
        sample_mode=sample_mode,
        sample_row_start=sample_row_start,
        sample_row_stop=sample_row_stop,
        sample_seed=sample_seed,
        sample_require_valid_bands=sample_require_valid_bands,
        field_column=field_column,
        target_redshift_column=target_redshift_column,
        z_min=z_min,
        z_max=z_max,
        redshift_include_min=redshift_include_min,
        redshift_include_max=redshift_include_max,
        n_z_bins=n_z_bins,
        mag_zero_point=mag_zero_point,
        hsc_mag_faint_limits=hsc_mag_faint_limits,
        extra_bands=extra_bands,
        extra_band_invalid_fill=extra_band_invalid_fill,
        extra_band_include_valid_flags=extra_band_include_valid_flags,
        aion_mag_adjustment_path=aion_mag_adjustment_path,
    )
    product_ids = [str(object_id) for object_id in product["object_id"]]
    raw_ids = [str(object_id) for object_id in raw_dataset.object_ids]
    if product_ids != raw_ids:
        raise RuntimeError(
            "Cached product object order does not match the rebuilt catalogue rows. "
            "Rerun with force_recompute_embeddings=True to rebuild the cache."
        )
    product["redshift_reference"] = raw_dataset.redshift_reference
    product_metadata = dict(product.get("metadata", {}))
    product_metadata["redshift_reference_keys"] = metadata.get("redshift_reference_keys", [])
    product["metadata"] = product_metadata
    return product


def refresh_cached_product_catalogue_features(
    product: dict[str, Any],
    *,
    catalogue_path: str | Path,
    split_output_dir: str | Path,
    split_chunk_size: int = 250_000,
    overwrite_split_cache: bool = False,
    max_rows: int | None = 20_000,
    sample_mode: str = "head",
    sample_row_start: int | None = None,
    sample_row_stop: int | None = None,
    sample_seed: int = 42,
    sample_require_valid_bands: Sequence[str] = (),
    field_column: str | None = None,
    target_redshift_column: str = REDSHIFT_COLUMNS["zphot"],
    z_min: float = 0.0,
    z_max: float = 6.0,
    redshift_include_min: bool = True,
    redshift_include_max: bool = True,
    n_z_bins: int = 300,
    mag_zero_point: float = 23.0,
    hsc_mag_faint_limits: Mapping[str, float | None] | None = None,
    extra_bands: Sequence[str] | None = None,
    extra_band_invalid_fill: str | float = "median",
    extra_band_include_valid_flags: bool = False,
    use_mlp_features: bool = True,
    include_grizy_in_mlp: bool | None = None,
    use_aion_embedding: bool = True,
    aion_mag_adjustment_path: str | Path | None = None,
) -> dict[str, Any]:
    """Refresh non-AION catalogue-side tensors on an existing cached product."""
    raw_dataset, feature_names, metadata = build_raw_clauds_photoz_dataset(
        catalogue_path,
        split_output_dir,
        split_chunk_size=split_chunk_size,
        overwrite_split_cache=overwrite_split_cache,
        max_rows=max_rows,
        sample_mode=sample_mode,
        sample_row_start=sample_row_start,
        sample_row_stop=sample_row_stop,
        sample_seed=sample_seed,
        sample_require_valid_bands=sample_require_valid_bands,
        field_column=field_column,
        target_redshift_column=target_redshift_column,
        z_min=z_min,
        z_max=z_max,
        redshift_include_min=redshift_include_min,
        redshift_include_max=redshift_include_max,
        n_z_bins=n_z_bins,
        mag_zero_point=mag_zero_point,
        hsc_mag_faint_limits=hsc_mag_faint_limits,
        extra_bands=extra_bands,
        extra_band_invalid_fill=extra_band_invalid_fill,
        extra_band_include_valid_flags=extra_band_include_valid_flags,
        use_mlp_features=use_mlp_features,
        include_grizy_in_mlp=include_grizy_in_mlp,
        use_aion_embedding=use_aion_embedding,
        aion_mag_adjustment_path=aion_mag_adjustment_path,
    )
    product_ids = [str(object_id) for object_id in product["object_id"]]
    raw_ids = [str(object_id) for object_id in raw_dataset.object_ids]
    if product_ids != raw_ids:
        raise RuntimeError(
            "Cached product object order does not match the rebuilt catalogue rows. "
            "Rerun with force_recompute_embeddings=True to rebuild the cache."
        )

    n_rows = len(raw_dataset)
    if use_aion_embedding:
        if "aion_embedding" not in product:
            raise RuntimeError(
                "Cached product does not contain AION embeddings. "
                "Rerun with force_recompute_embeddings=True to rebuild the cache."
            )
        aion_embedding = torch.as_tensor(product["aion_embedding"], dtype=torch.float32)
        if aion_embedding.ndim != 2 or aion_embedding.shape[0] != n_rows or aion_embedding.shape[1] == 0:
            raise RuntimeError(
                "Cached product has no usable AION embeddings for use_aion_embedding=True. "
                "Rerun with force_recompute_embeddings=True, or use a cache path built with AION enabled."
            )
        product["aion_embedding"] = aion_embedding.cpu()
    else:
        product["aion_embedding"] = torch.empty((n_rows, 0), dtype=torch.float32)

    product["extra_features"] = raw_dataset.extra_features.cpu()
    product["feature_names"] = list(feature_names)
    product["z_spec"] = raw_dataset.z_spec
    product["redshift_reference"] = raw_dataset.redshift_reference
    product_metadata = dict(product.get("metadata", {}))
    product_metadata.update(metadata)
    if not use_aion_embedding:
        product_metadata.update({
            "aion_model": None,
            "aion_embedding_pooling": None,
            "embedding_batch_size": None,
            "aion_embedding_note": "AION embedding extraction was skipped; use model_kind='tabular'.",
        })
    product["metadata"] = product_metadata
    return product


def make_cache_run_tag(
    catalogue_path: str | Path,
    max_rows: int | None,
    mag_zero_point: float,
    *,
    sample_mode: str = "head",
    sample_row_start: int | None = None,
    sample_row_stop: int | None = None,
    sample_seed: int = 42,
    sample_require_valid_bands: Sequence[str] = (),
) -> str:
    stem = Path(catalogue_path).stem.replace("-", "_")
    n_tag = "all" if max_rows is None else f"n{max_rows}"
    zp_tag = f"zp{mag_zero_point:.1f}".replace(".", "p")
    run_tag = f"{stem}_{zp_tag}_{n_tag}"
    if sample_mode != "head":
        run_tag = f"{run_tag}_{sample_mode}_s{sample_seed}"
    if sample_row_start is not None or sample_row_stop is not None:
        start_tag = 0 if sample_row_start is None else sample_row_start
        stop_tag = "end" if sample_row_stop is None else sample_row_stop
        run_tag = f"{run_tag}_rows{start_tag}_{stop_tag}"
    if sample_require_valid_bands:
        required_tag = "_".join(str(band).replace("*", "star") for band in sample_require_valid_bands)
        run_tag = f"{run_tag}_req_{required_tag}"
    return run_tag


def build_grizy_mlp_feature_matrix(
    hsc_features: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, list[str]]:
    """Return grizy magnitudes as ordinary MLP/tabular features."""
    columns = [
        torch.as_tensor(hsc_features[f"{band}_mag"], dtype=torch.float32).reshape(-1)
        for band in HSC_AION_BANDS
    ]
    feature_names = [f"{band}_mag" for band in HSC_AION_BANDS]
    return torch.stack(columns, dim=1), feature_names


def build_and_cache_aion_embeddings(
    *,
    catalogue_path: str | Path,
    split_output_dir: str | Path,
    cache_path: str | Path,
    split_chunk_size: int = 250_000,
    overwrite_split_cache: bool = False,
    max_rows: int | None = 20_000,
    sample_mode: str = "head",
    sample_row_start: int | None = None,
    sample_row_stop: int | None = None,
    sample_seed: int = 42,
    sample_require_valid_bands: Sequence[str] = (),
    field_column: str | None = None,
    target_redshift_column: str = REDSHIFT_COLUMNS["zphot"],
    z_min: float = 0.0,
    z_max: float = 6.0,
    redshift_include_min: bool = True,
    redshift_include_max: bool = True,
    n_z_bins: int = 300,
    mag_zero_point: float = 23.0,
    hsc_mag_faint_limits: Mapping[str, float | None] | None = None,
    extra_bands: Sequence[str] | None = None,
    extra_band_invalid_fill: str | float = "median",
    extra_band_include_valid_flags: bool = False,
    use_mlp_features: bool = True,
    include_grizy_in_mlp: bool | None = None,
    split_strategy: str = "random",
    train_fraction: float = 0.20,
    test_fraction: float = 0.75,
    val_fraction: float = 0.05,
    test_fields: Sequence[str] = (),
    batch_size: int = 512,
    force_recompute_embeddings: bool = False,
    use_aion_embedding: bool = True,
    aion_input_bands: Sequence[str] | None = None,
    aion_mag_adjustment_path: str | Path | None = None,
    device: torch.device | str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    device = resolve_torch_device(device)
    include_grizy_in_mlp = resolve_include_grizy_in_mlp(
        include_grizy_in_mlp,
        use_aion_embedding=use_aion_embedding,
        use_mlp_features=use_mlp_features,
        extra_bands=extra_bands,
    )
    if not use_mlp_features and not use_aion_embedding:
        raise ValueError("use_mlp_features=False requires use_aion_embedding=True for AION-only mode.")
    if not use_mlp_features:
        include_grizy_in_mlp = False
    if aion_input_bands is not None and tuple(aion_input_bands) != tuple(HSC_AION_BANDS):
        warnings.warn(
            "Disabling individual HSC grizy bands is not currently supported because "
            "the frozen AION embedding expects the full grizy input; the requested "
            "AION band selection will be ignored for now.",
            RuntimeWarning,
            stacklevel=2,
        )
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_recompute_embeddings:
        product = load_cached_product(cache_path)
        validate_cached_aion_mag_adjustment(
            product.get("metadata", {}),
            aion_mag_adjustment_path,
        )
        product = refresh_cached_product_catalogue_features(
            product,
            catalogue_path=catalogue_path,
            split_output_dir=split_output_dir,
            split_chunk_size=split_chunk_size,
            overwrite_split_cache=overwrite_split_cache,
            max_rows=max_rows,
            sample_mode=sample_mode,
            sample_row_start=sample_row_start,
            sample_row_stop=sample_row_stop,
            sample_seed=sample_seed,
            sample_require_valid_bands=sample_require_valid_bands,
            field_column=field_column,
            target_redshift_column=target_redshift_column,
            z_min=z_min,
            z_max=z_max,
            redshift_include_min=redshift_include_min,
            redshift_include_max=redshift_include_max,
            n_z_bins=n_z_bins,
            mag_zero_point=mag_zero_point,
            hsc_mag_faint_limits=hsc_mag_faint_limits,
            extra_bands=extra_bands,
            extra_band_invalid_fill=extra_band_invalid_fill,
            extra_band_include_valid_flags=extra_band_include_valid_flags,
            use_mlp_features=use_mlp_features,
            include_grizy_in_mlp=include_grizy_in_mlp,
            use_aion_embedding=use_aion_embedding,
            aion_mag_adjustment_path=aion_mag_adjustment_path,
        )
        split_labels = make_split_labels(
            product["field"],
            split_strategy=split_strategy,
            train_fraction=train_fraction,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
            test_fields=test_fields,
            seed=seed,
        )
        product["split_labels"] = list(split_labels)
        metadata = dict(product.get("metadata", {}))
        metadata.update(split_metadata(split_labels, split_strategy, train_fraction, test_fraction, val_fraction))
        metadata["use_aion_embedding"] = bool(use_aion_embedding)
        metadata["use_mlp_features"] = bool(use_mlp_features)
        metadata["include_grizy_in_mlp"] = bool(include_grizy_in_mlp)
        metadata["aion_input_bands"] = list(HSC_AION_BANDS)
        metadata["n_z_bins"] = int(n_z_bins)
        metadata["redshift_include_min"] = bool(redshift_include_min)
        metadata["redshift_include_max"] = bool(redshift_include_max)
        metadata["redshift_edges"] = make_redshift_grid(z_min, z_max, n_z_bins)[0]
        metadata["redshift_centers"] = make_redshift_grid(z_min, z_max, n_z_bins)[1]
        metadata.update(aion_mag_adjustment_metadata(aion_mag_adjustment_path))
        product["metadata"] = metadata
        torch.save(product, cache_path)
        return product

    raw_dataset, feature_names, metadata = build_raw_clauds_photoz_dataset(
        catalogue_path,
        split_output_dir,
        split_chunk_size=split_chunk_size,
        overwrite_split_cache=overwrite_split_cache,
        max_rows=max_rows,
        sample_mode=sample_mode,
        sample_row_start=sample_row_start,
        sample_row_stop=sample_row_stop,
        sample_seed=sample_seed,
        sample_require_valid_bands=sample_require_valid_bands,
        field_column=field_column,
        target_redshift_column=target_redshift_column,
        z_min=z_min,
        z_max=z_max,
        redshift_include_min=redshift_include_min,
        redshift_include_max=redshift_include_max,
        n_z_bins=n_z_bins,
        mag_zero_point=mag_zero_point,
        hsc_mag_faint_limits=hsc_mag_faint_limits,
        extra_bands=extra_bands,
        extra_band_invalid_fill=extra_band_invalid_fill,
        extra_band_include_valid_flags=extra_band_include_valid_flags,
        use_mlp_features=use_mlp_features,
        include_grizy_in_mlp=include_grizy_in_mlp,
        use_aion_embedding=use_aion_embedding,
        aion_mag_adjustment_path=aion_mag_adjustment_path,
    )
    split_labels = make_split_labels(
        raw_dataset.fields,
        split_strategy=split_strategy,
        train_fraction=train_fraction,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        test_fields=test_fields,
        seed=seed,
    )

    if use_aion_embedding:
        aion, codec_manager = load_frozen_aion(device=device)
        aion_embeddings = extract_aion_embeddings_to_memory(
            raw_dataset,
            aion=aion,
            codec_manager=codec_manager,
            batch_size=batch_size,
            device=device,
        )
        aion_metadata = {
            "aion_model": "polymathic-ai/aion-base",
            "aion_embedding_pooling": "mean_encoder_tokens",
            "embedding_batch_size": batch_size,
        }
    else:
        aion_embeddings = torch.empty((len(raw_dataset), 0), dtype=torch.float32)
        aion_metadata = {
            "aion_model": None,
            "aion_embedding_pooling": None,
            "embedding_batch_size": None,
            "aion_embedding_note": "AION embedding extraction was skipped; use model_kind='tabular'.",
        }
    metadata.update({
        **aion_metadata,
        "use_aion_embedding": bool(use_aion_embedding),
        "use_mlp_features": bool(use_mlp_features),
        "include_grizy_in_mlp": bool(include_grizy_in_mlp),
        "aion_input_bands": list(HSC_AION_BANDS),
        "n_z_bins": n_z_bins,
        **split_metadata(split_labels, split_strategy, train_fraction, test_fraction, val_fraction),
    })
    save_cached_product(
        cache_path,
        raw_dataset,
        aion_embeddings,
        feature_names,
        split_labels=split_labels,
        metadata=metadata,
    )
    return load_cached_product(cache_path)


def build_and_cache_aion_embeddings_from_config(
    config: AIONMagnitudeConfig | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build or load the cached AION embeddings using only config defaults/overrides."""
    config = make_magnitude_config(config, **overrides)
    set_random_seed(config.seed)
    run_device = select_torch_device(config.device_choice)
    paths = resolve_training_paths(config)
    return build_and_cache_aion_embeddings(
        catalogue_path=config.catalogue_path,
        split_output_dir=paths["split_output_dir"],
        cache_path=paths["cache_path"],
        split_chunk_size=config.split_chunk_size,
        overwrite_split_cache=config.overwrite_split_cache,
        max_rows=config.max_rows,
        sample_mode=config.sample_mode,
        sample_row_start=config.sample_row_start,
        sample_row_stop=config.sample_row_stop,
        sample_seed=config.sample_seed,
        sample_require_valid_bands=config.sample_require_valid_bands,
        field_column=config.field_column,
        target_redshift_column=config.target_redshift_column,
        z_min=config.z_min,
        z_max=config.z_max,
        redshift_include_min=config.redshift_include_min,
        redshift_include_max=config.redshift_include_max,
        n_z_bins=config.n_z_bins,
        mag_zero_point=config.mag_zero_point,
        hsc_mag_faint_limits=config.hsc_mag_faint_limits,
        extra_bands=config.extra_bands,
        extra_band_invalid_fill=config.extra_band_invalid_fill,
        extra_band_include_valid_flags=config.extra_band_include_valid_flags,
        use_mlp_features=config.use_mlp_features,
        include_grizy_in_mlp=resolve_include_grizy_in_mlp(
            config.include_grizy_in_mlp,
            use_aion_embedding=config.use_aion_embedding,
            use_mlp_features=config.use_mlp_features,
            extra_bands=config.extra_bands,
        ),
        split_strategy=config.split_strategy,
        train_fraction=config.train_fraction,
        test_fraction=config.test_fraction,
        val_fraction=config.val_fraction,
        test_fields=config.test_fields,
        batch_size=config.aion_embedding_batch_size,
        force_recompute_embeddings=config.force_recompute_embeddings,
        use_aion_embedding=config.use_aion_embedding,
        aion_input_bands=config.aion_input_bands,
        aion_mag_adjustment_path=config.aion_mag_adjustment_path,
        device=run_device,
        seed=config.seed,
    )
