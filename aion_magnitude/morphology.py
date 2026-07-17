from __future__ import annotations

"""CLAUDS u-image morphology experiments through the AION image tokenizer.

This module intentionally does not use the AION redshift model or AION encoder
embeddings for images.  It only uses the pretrained AION image codec to turn
CLAUDS u-band cutouts into image token IDs, decodes those IDs into their FSQ
factors, and trains a CLAUDS-supervised photo-z MLP.  The optional AION
magnitude mode uses the frozen grizy AION embedding as the photometric input;
it never uses AION's internal image-to-redshift representation.
"""

import argparse
import math
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from astropy.io import fits
from astropy.wcs import WCS
from torch.utils.data import DataLoader, Dataset

from .caching import build_and_cache_aion_embeddings_from_config
from .clauds_bands import HSC_AION_BANDS, REDSHIFT_COLUMNS, default_hsc_mag_faint_limits
from .config import (
    AIONMagnitudeConfig,
    make_magnitude_config as make_base_magnitude_config,
    resolve_training_paths,
)
from .dataset import clauds_redshift_filter_mask, make_split_labels, split_metadata
from .extra_bands import (
    DEFAULT_EXTRA_BANDS,
    EXTRA_BAND_LABELS,
    extract_extra_band_magnitudes_from_split_arrays,
    resolve_extra_band_names,
)
from .metrics import (
    evaluate_conformal_hpd,
    predict_photoz_from_logits,
    redshift_cross_entropy_loss,
    summarize_pdf_metrics,
)
from .models import AIONOnlyPhotoZModel, ExtraPhotometryEncoder, PhotoZHead
from .plotting import (
    compare_config_loss,
    compare_nz_lensing_alike,
    compare_pit_histogram,
    compare_redshift_probability_distribution,
    compare_zpred_vs_zphot,
)
from .training import save_calibration_artifacts
from .utils import (
    flux_to_ab_mag,
    make_redshift_grid,
    resolve_torch_device,
    select_torch_device,
    set_random_seed,
    validate_split_fractions,
)


AION_IMAGE_TOKEN_KEY = "tok_image_hsc"
AION_IMAGE_BAND_ALIAS = "HSC-G"
AION_IMAGE_INPUT_SIZE = 96
AION_IMAGE_GRID_SIZE = 24
DEFAULT_AION_IMAGE_QUANTIZER_LEVELS = (7, 5, 5, 5, 5)
MORPHOLOGY_REPORT_BANDS = ("u", "u_star", "Y", "J", "Ks")


def _as_path_or_none(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value)


def _safe_path_tag(value: str) -> str:
    return str(value).replace("*", "star").replace("/", "_").replace(" ", "_").replace("-", "_")


def _float_path_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _optional_int(value: str | int | None) -> int | None:
    if value is None or isinstance(value, int):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"all", "none", "null"}:
        return None
    return int(value)


@dataclass
class AIONMorphologyConfig:
    """Configuration for AION-tokenized CLAUDS u-image morphology training."""

    catalogue_path: str | Path = Path("data/clauds/COSMOS-HSCpipe-Phosphoros.fits")
    morphology_dir: str | Path = Path("data/clauds/images/tilesv5/")
    cache_root: str | Path = Path("cache")
    split_output_dir: str | Path | None = None
    photometry_cache_path: str | Path | None = None
    token_cache_dir: str | Path | None = None
    output_dir: str | Path | None = None

    # Conservative default; full COSMOS image tokenization is a multi-GB cache.
    max_rows: int | None = 20_000
    sample_mode: str = "random"
    sample_row_start: int | None = None
    sample_row_stop: int | None = None
    sample_seed: int = 42
    sample_require_valid_bands: Sequence[str] = field(default_factory=tuple)

    split_chunk_size: int = 250_000
    overwrite_split_cache: bool = False
    force_rebuild_photometry: bool = False
    force_rebuild_tokens: bool = False
    preserve_photometry_splits: bool = False
    field_column: str | None = None
    split_strategy: str = "random"
    train_fraction: float = 0.20
    test_fraction: float = 0.75
    val_fraction: float = 0.05
    test_fields: Sequence[str] = field(default_factory=list)
    target_redshift_column: str = REDSHIFT_COLUMNS["zphot"]

    # Mirrors the recent low-z comparison controls in token_embedding.ipynb.
    z_min: float = 0.0
    z_max: float = 2.5
    redshift_include_min: bool = True
    redshift_include_max: bool = False
    n_z_bins: int = 100

    mag_zero_point: float = 23.0
    hsc_mag_faint_limits: Mapping[str, float | None] = field(
        default_factory=default_hsc_mag_faint_limits
    )
    extra_bands: Sequence[str] = field(default_factory=lambda: list(DEFAULT_EXTRA_BANDS))
    extra_band_invalid_fill: str | float = "median"
    extra_band_include_valid_flags: bool = False
    include_grizy_in_mlp: bool = True
    use_aion_magnitude_embedding: bool = False
    aion_embedding_batch_size: int = 512

    cutout_size: int = AION_IMAGE_INPUT_SIZE
    aion_image_band_alias: str = AION_IMAGE_BAND_ALIAS
    image_background_mode: str = "median"
    image_flux_scale: float = 1.0
    min_cutout_weight_coverage: float = 0.90
    tile_assignment_chunk_size: int = 100_000
    token_batch_size: int = 64

    model_kinds: Sequence[str] = ("photometry", "morphology", "shuffled_morphology")
    image_hidden_dim: int = 512
    image_embedding_dim: int = 128
    photometry_hidden_dim: int = 128
    head_hidden_dim: int = 256
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_batch_size: int = 256
    eval_batch_size: int = 512
    early_stopping_patience: int | None = None
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    tomographic_samples: int = 100

    seed: int = 42
    device_choice: str = "auto"

    def normalized(self) -> "AIONMorphologyConfig":
        sample_mode = str(self.sample_mode)
        if sample_mode not in {"head", "random"}:
            raise ValueError("sample_mode must be 'head' or 'random'.")
        if self.split_strategy not in {"random", "field"}:
            raise ValueError("split_strategy must be 'random' or 'field'.")
        validate_split_fractions(self.train_fraction, self.test_fraction, self.val_fraction)
        if int(self.cutout_size) != AION_IMAGE_INPUT_SIZE:
            raise ValueError("AION image codec expects 96x96 cutouts.")
        if int(self.token_batch_size) < 1:
            raise ValueError("token_batch_size must be >= 1.")
        if int(self.aion_embedding_batch_size) < 1:
            raise ValueError("aion_embedding_batch_size must be >= 1.")
        if int(self.tomographic_samples) < 1:
            raise ValueError("tomographic_samples must be >= 1.")
        if not (0.0 <= float(self.min_cutout_weight_coverage) <= 1.0):
            raise ValueError("min_cutout_weight_coverage must be in [0, 1].")
        if not np.isfinite(self.image_flux_scale) or float(self.image_flux_scale) <= 0.0:
            raise ValueError("image_flux_scale must be finite and > 0.")
        background_mode = str(self.image_background_mode).lower()
        if background_mode not in {"none", "median"}:
            raise ValueError("image_background_mode must be 'none' or 'median'.")
        model_kinds = tuple(str(kind) for kind in self.model_kinds)
        unknown = set(model_kinds).difference(
            {
                "photometry",
                "morphology",
                "shuffled_morphology",
                "aion",
                "aion_morphology",
                "shuffled_aion_morphology",
            }
        )
        if unknown:
            raise ValueError(f"Unknown model_kinds: {sorted(unknown)}")
        aion_model_kinds = {
            "aion",
            "aion_morphology",
            "shuffled_aion_morphology",
        }
        selected_aion_kinds = set(model_kinds) & aion_model_kinds
        if self.use_aion_magnitude_embedding and set(model_kinds) != selected_aion_kinds:
            raise ValueError(
                "use_aion_magnitude_embedding=True requires only AION model kinds: "
                "'aion', 'aion_morphology', or 'shuffled_aion_morphology'."
            )
        if not self.use_aion_magnitude_embedding and selected_aion_kinds:
            raise ValueError("AION model kinds require use_aion_magnitude_embedding=True.")

        return replace(
            self,
            catalogue_path=Path(self.catalogue_path),
            morphology_dir=Path(self.morphology_dir),
            cache_root=Path(self.cache_root),
            split_output_dir=_as_path_or_none(self.split_output_dir),
            photometry_cache_path=_as_path_or_none(self.photometry_cache_path),
            token_cache_dir=_as_path_or_none(self.token_cache_dir),
            output_dir=_as_path_or_none(self.output_dir),
            max_rows=None if self.max_rows is None else int(self.max_rows),
            sample_mode=sample_mode,
            sample_row_start=None if self.sample_row_start is None else int(self.sample_row_start),
            sample_row_stop=None if self.sample_row_stop is None else int(self.sample_row_stop),
            sample_seed=int(self.sample_seed),
            sample_require_valid_bands=tuple(dict.fromkeys(str(band) for band in self.sample_require_valid_bands)),
            preserve_photometry_splits=bool(self.preserve_photometry_splits),
            hsc_mag_faint_limits=dict(self.hsc_mag_faint_limits),
            extra_bands=resolve_extra_band_names(self.extra_bands),
            extra_band_include_valid_flags=bool(self.extra_band_include_valid_flags),
            include_grizy_in_mlp=bool(self.include_grizy_in_mlp),
            use_aion_magnitude_embedding=bool(self.use_aion_magnitude_embedding),
            image_background_mode=background_mode,
            test_fields=list(self.test_fields),
            model_kinds=model_kinds,
        )


def make_magnitude_config(config: AIONMorphologyConfig) -> AIONMagnitudeConfig:
    config = config.normalized()
    return make_base_magnitude_config(
        catalogue_path=config.catalogue_path,
        max_rows=config.max_rows,
        sample_mode=config.sample_mode,
        sample_row_start=config.sample_row_start,
        sample_row_stop=config.sample_row_stop,
        sample_seed=config.sample_seed,
        sample_require_valid_bands=config.sample_require_valid_bands,
        cache_root=config.cache_root,
        split_output_dir=config.split_output_dir,
        cache_path=config.photometry_cache_path,
        split_chunk_size=config.split_chunk_size,
        overwrite_split_cache=config.overwrite_split_cache,
        force_recompute_embeddings=config.force_rebuild_photometry,
        field_column=config.field_column,
        split_strategy=config.split_strategy,
        train_fraction=config.train_fraction,
        test_fraction=config.test_fraction,
        val_fraction=config.val_fraction,
        test_fields=config.test_fields,
        target_redshift_column=config.target_redshift_column,
        z_min=config.z_min,
        z_max=config.z_max,
        redshift_include_min=config.redshift_include_min,
        redshift_include_max=config.redshift_include_max,
        n_z_bins=config.n_z_bins,
        mag_zero_point=config.mag_zero_point,
        hsc_mag_faint_limits=config.hsc_mag_faint_limits,
        extra_bands=() if config.use_aion_magnitude_embedding else config.extra_bands,
        extra_band_invalid_fill=config.extra_band_invalid_fill,
        extra_band_include_valid_flags=config.extra_band_include_valid_flags,
        use_aion_embedding=config.use_aion_magnitude_embedding,
        use_mlp_features=not config.use_aion_magnitude_embedding,
        include_grizy_in_mlp=(
            False if config.use_aion_magnitude_embedding else config.include_grizy_in_mlp
        ),
        aion_embedding_batch_size=config.aion_embedding_batch_size,
        model_kinds=("aion",) if config.use_aion_magnitude_embedding else ("tabular",),
        seed=config.seed,
        device_choice=config.device_choice,
    )


def resolve_morphology_paths(config: AIONMorphologyConfig) -> dict[str, Path | str]:
    config = config.normalized()
    mag_config = make_magnitude_config(config)
    magnitude_paths = resolve_training_paths(mag_config)
    alias_tag = _safe_path_tag(config.aion_image_band_alias)
    preprocessing_tag = (
        f"bg_{_safe_path_tag(config.image_background_mode)}"
        f"_scale_{_float_path_tag(config.image_flux_scale)}"
        f"_cov_{_float_path_tag(config.min_cutout_weight_coverage)}"
    )
    morph_tag = (
        f"{magnitude_paths['experiment_tag']}_u_as_{alias_tag}"
        f"_{preprocessing_tag}_aion_image_tokens"
    )
    cache_root = Path(config.cache_root)
    token_cache_dir = (
        Path(config.token_cache_dir)
        if config.token_cache_dir is not None
        else cache_root / f"morphology_tokens_{morph_tag}"
    )
    output_dir = (
        Path(config.output_dir)
        if config.output_dir is not None
        else cache_root / f"morphology_aion_{morph_tag}"
    )
    return {
        **magnitude_paths,
        "morphology_tag": morph_tag,
        "token_cache_dir": token_cache_dir,
        "token_ids_path": token_cache_dir / "aion_u_as_hsc_g_token_ids.npy",
        "token_quality_path": token_cache_dir / "aion_u_as_hsc_g_token_quality.npy",
        "token_histogram_path": token_cache_dir / "aion_u_as_hsc_g_token_histogram.npy",
        "morphology_product_path": token_cache_dir / "morphology_token_product.pt",
        "morphology_output_dir": output_dir,
    }


def build_or_load_photometry_product(config: AIONMorphologyConfig) -> dict[str, Any]:
    config = config.normalized()
    set_random_seed(config.seed)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Disabling individual HSC grizy bands is not currently supported.*",
            category=RuntimeWarning,
        )
        product = build_and_cache_aion_embeddings_from_config(make_magnitude_config(config))
    if config.use_aion_magnitude_embedding and product["aion_embedding"].shape[1] == 0:
        raise RuntimeError("AION magnitude mode produced no grizy AION embeddings.")
    return product


def _split_redshift_name(target_redshift_column: str) -> str:
    names = {
        "ZPHOT": "zphot",
        "Z_LOW68": "z_low68",
        "Z_HIGH68": "z_high68",
        "Z_CHI": "z_chi",
        "Z_PEAK": "z_peak",
        "Posterior-Log": "posterior_log",
        "Likelihood-Log": "likelihood_log",
    }
    if target_redshift_column in names:
        return names[target_redshift_column]
    lowered = target_redshift_column.lower()
    if lowered in names.values():
        return lowered
    raise KeyError(f"Unsupported split redshift column: {target_redshift_column!r}")


def build_photometry_usable_mask_from_split_arrays(
    bands: np.ndarray,
    flags: np.ndarray,
    redshifts: np.ndarray,
    *,
    mag_zero_point: float,
    hsc_mag_faint_limits: Mapping[str, float | None] | None,
    target_redshift_column: str,
    z_min: float,
    z_max: float,
    redshift_include_min: bool,
    redshift_include_max: bool,
) -> np.ndarray:
    mask = np.ones(len(bands), dtype=bool)
    flag_names = set(flags.dtype.names or ())

    for band in HSC_AION_BANDS:
        mag, valid = flux_to_ab_mag(
            bands[f"flux_cmodel_{band}"],
            mag_zero_point=mag_zero_point,
        )
        mask &= valid
        if hsc_mag_faint_limits is not None:
            limit = hsc_mag_faint_limits.get(band)
            if limit is not None:
                mask &= np.isfinite(mag) & (mag <= float(limit))

        for prefix in ("is_no_data", "not_observed", "has_bad_photometry"):
            flag_name = f"{prefix}_{band}"
            if flag_name in flag_names:
                mask &= ~flags[flag_name].astype(bool)

    z_name = _split_redshift_name(target_redshift_column)
    z_values = np.asarray(redshifts[z_name], dtype=np.float32)
    mask &= clauds_redshift_filter_mask(
        z_values,
        z_min=z_min,
        z_max=z_max,
        include_min=redshift_include_min,
        include_max=redshift_include_max,
    )
    return mask


def load_product_sky_positions(product: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    metadata = dict(product.get("metadata", {}))
    split_dir = Path(metadata["split_output_dir"])
    bands = np.load(split_dir / "clauds_bands.npy", mmap_mode="r")
    flags = np.load(split_dir / "clauds_flags.npy", mmap_mode="r")
    redshifts = np.load(split_dir / "clauds_redshifts.npy", mmap_mode="r")

    usable_mask = build_photometry_usable_mask_from_split_arrays(
        bands,
        flags,
        redshifts,
        mag_zero_point=float(metadata["mag_zero_point"]),
        hsc_mag_faint_limits=metadata.get("hsc_mag_faint_limits"),
        target_redshift_column=metadata.get("target_redshift_column", "ZPHOT"),
        z_min=float(metadata.get("z_min", 0.0)),
        z_max=float(metadata.get("z_max", 6.0)),
        redshift_include_min=bool(metadata.get("redshift_include_min", True)),
        redshift_include_max=bool(metadata.get("redshift_include_max", True)),
    )
    product_ids = np.asarray(product["object_id"], dtype=np.int64)
    split_ids = np.asarray(bands["id"][usable_mask], dtype=np.int64)
    if product_ids.shape == split_ids.shape and np.array_equal(product_ids, split_ids):
        return (
            np.asarray(bands["ra"][usable_mask], dtype=np.float64),
            np.asarray(bands["dec"][usable_mask], dtype=np.float64),
        )

    warnings.warn(
        "Photometry product row order did not match the rebuilt split-cache mask; "
        "falling back to object-id lookup for RA/Dec alignment.",
        RuntimeWarning,
        stacklevel=2,
    )
    id_to_index = {int(object_id): index for index, object_id in enumerate(np.asarray(bands["id"], dtype=np.int64))}
    indices = np.asarray([id_to_index[int(object_id)] for object_id in product_ids], dtype=np.int64)
    return (
        np.asarray(bands["ra"][indices], dtype=np.float64),
        np.asarray(bands["dec"][indices], dtype=np.float64),
    )


def load_product_extra_band_validity(
    product: Mapping[str, Any],
    *,
    extra_bands: Sequence[str] = MORPHOLOGY_REPORT_BANDS,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load per-object extra-band validity aligned to a photometry product."""
    metadata = dict(product.get("metadata", {}))
    split_dir = Path(metadata["split_output_dir"])
    bands = np.load(split_dir / "clauds_bands.npy", mmap_mode="r")
    flags = np.load(split_dir / "clauds_flags.npy", mmap_mode="r")
    redshifts = np.load(split_dir / "clauds_redshifts.npy", mmap_mode="r")
    usable_mask = build_photometry_usable_mask_from_split_arrays(
        bands,
        flags,
        redshifts,
        mag_zero_point=float(metadata["mag_zero_point"]),
        hsc_mag_faint_limits=metadata.get("hsc_mag_faint_limits"),
        target_redshift_column=metadata.get("target_redshift_column", "ZPHOT"),
        z_min=float(metadata.get("z_min", 0.0)),
        z_max=float(metadata.get("z_max", 6.0)),
        redshift_include_min=bool(metadata.get("redshift_include_min", True)),
        redshift_include_max=bool(metadata.get("redshift_include_max", True)),
    )
    usable_indices = np.flatnonzero(usable_mask)
    product_ids = np.asarray(product["object_id"], dtype=np.int64)
    usable_ids = np.asarray(bands["id"][usable_indices], dtype=np.int64)
    if product_ids.shape == usable_ids.shape and np.array_equal(product_ids, usable_ids):
        source_indices = usable_indices
    else:
        id_to_index = {
            int(object_id): index
            for index, object_id in enumerate(np.asarray(bands["id"], dtype=np.int64))
        }
        try:
            source_indices = np.asarray(
                [id_to_index[int(object_id)] for object_id in product_ids],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                f"Photometry product object_id {exc.args[0]} is absent from the split cache."
            ) from exc

    _, valid, selected_bands = extract_extra_band_magnitudes_from_split_arrays(
        bands[source_indices],
        flags[source_indices],
        extra_bands=extra_bands,
        mag_zero_point=float(metadata["mag_zero_point"]),
        warn_missing=False,
    )
    return np.asarray(valid, dtype=bool), selected_bands


def _population_count(n_gal: int, denominator_n_gal: int) -> dict[str, float | int]:
    n_gal = int(n_gal)
    denominator_n_gal = int(denominator_n_gal)
    percent = 100.0 * n_gal / denominator_n_gal if denominator_n_gal else 0.0
    return {
        "n_gal": n_gal,
        "denominator_n_gal": denominator_n_gal,
        "percent": percent,
    }


def build_morphology_population_report(
    product: Mapping[str, Any],
    *,
    morphology_available: Sequence[bool] | np.ndarray,
    extra_band_valid: torch.Tensor | np.ndarray,
    extra_bands: Sequence[str],
) -> dict[str, Any]:
    """Count independent morphology and band availability in original splits."""
    split_labels = np.asarray(product["split_labels"], dtype=object)
    morphology_available = np.asarray(morphology_available, dtype=bool)
    if isinstance(extra_band_valid, torch.Tensor):
        extra_band_valid = extra_band_valid.detach().cpu().numpy()
    extra_band_valid = np.asarray(extra_band_valid, dtype=bool)
    selected_bands = tuple(extra_bands)
    n_rows = len(split_labels)
    if morphology_available.shape != (n_rows,):
        raise ValueError("morphology_available must have one value per product row.")
    if extra_band_valid.shape != (n_rows, len(selected_bands)):
        raise ValueError(
            "extra_band_valid must have shape "
            f"({n_rows}, {len(selected_bands)}), got {extra_band_valid.shape}."
        )

    split_counts: dict[str, dict[str, float | int]] = {}
    morphology_counts: dict[str, dict[str, float | int]] = {}
    valid_band_counts: dict[str, dict[str, dict[str, float | int]]] = {}
    for split_name in ("train", "val", "test"):
        split_mask = split_labels == split_name
        split_n = int(split_mask.sum())
        split_counts[split_name] = _population_count(split_n, n_rows)
        if split_name not in {"train", "val"}:
            continue
        morphology_counts[split_name] = _population_count(
            int(np.sum(split_mask & morphology_available)),
            split_n,
        )
        valid_band_counts[split_name] = {
            band: _population_count(
                int(np.sum(split_mask & extra_band_valid[:, band_idx])),
                split_n,
            )
            for band_idx, band in enumerate(selected_bands)
        }

    return {
        "total_selected_n_gal": int(n_rows),
        "split_counts": split_counts,
        "morphology_counts": morphology_counts,
        "valid_band_counts": valid_band_counts,
        "reported_extra_bands": list(selected_bands),
        "band_counts_are_independent_of_morphology": True,
    }


def _format_population_count(record: Mapping[str, float | int]) -> str:
    return f"n_gal={int(record['n_gal']):,} ({float(record['percent']):.2f}%)"


def format_morphology_population_report(report: Mapping[str, Any]) -> str:
    """Format the requested figure-prefix population report."""
    split_counts = report["split_counts"]
    morphology_counts = report["morphology_counts"]
    valid_band_counts = report["valid_band_counts"]
    lines = [
        "Galaxy population report",
        (
            "selected catalogue: "
            f"n_gal={int(report['total_selected_n_gal']):,} (100.00%)"
        ),
        "",
        f"train: {_format_population_count(split_counts['train'])}",
        f"validation: {_format_population_count(split_counts['val'])}",
        f"test: {_format_population_count(split_counts['test'])}",
        "",
        (
            "train usable matched morphology: "
            f"{_format_population_count(morphology_counts['train'])}"
        ),
        (
            "validation usable matched morphology: "
            f"{_format_population_count(morphology_counts['val'])}"
        ),
        "",
    ]
    for band in report["reported_extra_bands"]:
        label = EXTRA_BAND_LABELS.get(band, band)
        lines.extend(
            [
                (
                    f"train valid {label}: "
                    f"{_format_population_count(valid_band_counts['train'][band])}"
                ),
                (
                    f"validation valid {label}: "
                    f"{_format_population_count(valid_band_counts['val'][band])}"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Split percentages use the full selected catalogue.",
            "Morphology and band percentages use their original train/validation split.",
            "Band-validity counts are independent and are not intersected with morphology.",
        ]
    )
    return "\n".join(lines) + "\n"


class MorphologyTile:
    def __init__(self, image_path: Path):
        self.image_path = Path(image_path)
        self.weight_path = self.image_path.with_name(f"{self.image_path.stem}.weight.fits")
        if not self.weight_path.exists():
            raise FileNotFoundError(f"Missing weight map for {self.image_path}: {self.weight_path}")

        with fits.open(self.image_path, memmap=True) as hdul:
            self.header = hdul[0].header.copy()
            self.shape = tuple(int(v) for v in hdul[0].data.shape)
        self.wcs = WCS(self.header)
        self.name = self.image_path.name
        self._image_hdul = None
        self._weight_hdul = None

    @property
    def image_data(self) -> np.ndarray:
        if self._image_hdul is None:
            self._image_hdul = fits.open(self.image_path, memmap=True)
        return self._image_hdul[0].data

    @property
    def weight_data(self) -> np.ndarray:
        if self._weight_hdul is None:
            self._weight_hdul = fits.open(self.weight_path, memmap=True)
        return self._weight_hdul[0].data

    def close(self) -> None:
        if self._image_hdul is not None:
            self._image_hdul.close()
            self._image_hdul = None
        if self._weight_hdul is not None:
            self._weight_hdul.close()
            self._weight_hdul = None


def discover_morphology_image_paths(morphology_dir: str | Path) -> list[Path]:
    """Return every science FITS tile with a matching weight map."""
    morphology_dir = Path(morphology_dir)
    image_paths = [
        path
        for path in sorted(morphology_dir.rglob("*.fits"))
        if not path.name.endswith(".weight.fits")
    ]
    if not image_paths:
        raise FileNotFoundError(f"No science FITS files found in {morphology_dir}")
    missing_weights = [
        path.with_name(f"{path.stem}.weight.fits")
        for path in image_paths
        if not path.with_name(f"{path.stem}.weight.fits").exists()
    ]
    if missing_weights:
        preview = ", ".join(str(path) for path in missing_weights[:5])
        raise FileNotFoundError(
            f"Missing weight maps for {len(missing_weights)} science tiles; first missing: {preview}"
        )
    return image_paths


def load_morphology_tiles(
    morphology_dir: str | Path,
    *,
    image_paths: Sequence[Path] | None = None,
) -> list[MorphologyTile]:
    paths = discover_morphology_image_paths(morphology_dir) if image_paths is None else image_paths
    return [MorphologyTile(path) for path in paths]


def assign_tiles_to_positions(
    ra: np.ndarray,
    dec: np.ndarray,
    tiles: Sequence[MorphologyTile],
    *,
    chunk_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = len(ra)
    tile_index = np.full(n_rows, -1, dtype=np.int16)
    x_out = np.full(n_rows, np.nan, dtype=np.float32)
    y_out = np.full(n_rows, np.nan, dtype=np.float32)
    best_margin = np.full(n_rows, -np.inf, dtype=np.float32)

    for start in range(0, n_rows, chunk_size):
        stop = min(start + chunk_size, n_rows)
        ra_chunk = np.asarray(ra[start:stop], dtype=np.float64)
        dec_chunk = np.asarray(dec[start:stop], dtype=np.float64)
        local_best = best_margin[start:stop]
        for idx, tile in enumerate(tiles):
            height, width = tile.shape
            try:
                x, y = tile.wcs.all_world2pix(ra_chunk, dec_chunk, 0)
            except Exception:
                continue
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            finite = np.isfinite(x) & np.isfinite(y)
            inside = finite & (x >= 0.0) & (x < width) & (y >= 0.0) & (y < height)
            if not inside.any():
                continue
            margin = np.minimum.reduce([x, y, width - 1.0 - x, height - 1.0 - y]).astype(np.float32)
            update = inside & (margin > local_best)
            if update.any():
                update_rows = np.arange(start, stop, dtype=np.int64)[update]
                tile_index[update_rows] = idx
                x_out[update_rows] = x[update].astype(np.float32)
                y_out[update_rows] = y[update].astype(np.float32)
                local_best[update] = margin[update]
        best_margin[start:stop] = local_best

    return tile_index, x_out, y_out


def _extract_cutout(
    tile: MorphologyTile,
    x: float,
    y: float,
    *,
    cutout_size: int,
    background_mode: str,
    image_flux_scale: float,
) -> tuple[np.ndarray, dict[str, float]]:
    image = tile.image_data
    weight = tile.weight_data
    height, width = image.shape
    half = cutout_size // 2
    cx = int(np.rint(float(x)))
    cy = int(np.rint(float(y)))
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + cutout_size
    y1 = y0 + cutout_size

    cutout = np.zeros((cutout_size, cutout_size), dtype=np.float32)
    weight_cutout = np.zeros((cutout_size, cutout_size), dtype=np.float32)

    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(width, x1)
    sy1 = min(height, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return cutout, {"weight_coverage": 0.0, "background": 0.0, "finite_fraction": 0.0}

    dx0 = sx0 - x0
    dy0 = sy0 - y0
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)

    src = np.asarray(image[sy0:sy1, sx0:sx1], dtype=np.float32)
    wsrc = np.asarray(weight[sy0:sy1, sx0:sx1], dtype=np.float32)
    valid = np.isfinite(src) & np.isfinite(wsrc) & (wsrc > 0.0)
    dest = cutout[dy0:dy1, dx0:dx1]
    wdest = weight_cutout[dy0:dy1, dx0:dx1]
    dest[valid] = src[valid]
    wdest[valid] = wsrc[valid]

    full_valid = np.isfinite(cutout) & np.isfinite(weight_cutout) & (weight_cutout > 0.0)
    coverage = float(full_valid.mean())
    finite_fraction = float(np.isfinite(cutout).mean())
    background = 0.0
    if background_mode == "median" and full_valid.any():
        background = float(np.nanmedian(cutout[full_valid]))
        cutout[full_valid] -= background
    cutout[~full_valid] = 0.0
    cutout *= float(image_flux_scale)

    return cutout.astype(np.float32, copy=False), {
        "weight_coverage": coverage,
        "background": background,
        "finite_fraction": finite_fraction,
    }


def load_aion_image_codec(device: torch.device | str):
    from aion.codecs import CodecManager
    from aion.modalities import HSCImage

    codec_manager = CodecManager(device=device)
    codec = codec_manager._load_codec(HSCImage).to(device).eval()
    levels = tuple(int(v) for v in codec.quantizer.levels.detach().cpu().tolist())
    return codec_manager, HSCImage, levels


@torch.no_grad()
def tokenize_cutout_batch(
    cutouts: np.ndarray,
    *,
    codec_manager,
    image_modality_type,
    device: torch.device | str,
    band_alias: str,
) -> np.ndarray:
    flux = torch.from_numpy(cutouts[:, None, :, :].astype(np.float32, copy=False)).to(device)
    modality = image_modality_type(flux=flux, bands=[band_alias])
    tokens = codec_manager.encode(modality)[image_modality_type.token_key]
    return tokens.detach().cpu().numpy()


def token_quality_dtype() -> np.dtype:
    return np.dtype(
        [
            ("object_id", np.int64),
            ("assigned_tile", np.int16),
            ("x_image", np.float32),
            ("y_image", np.float32),
            ("weight_coverage", np.float32),
            ("background", np.float32),
            ("finite_fraction", np.float32),
            ("token_available", np.bool_),
        ]
    )


def _subset_first_axis(value: Any, indices: np.ndarray, n_rows: int) -> Any:
    if isinstance(value, torch.Tensor) and value.shape[:1] == (n_rows,):
        return value[torch.as_tensor(indices, dtype=torch.long)]
    if isinstance(value, np.ndarray) and value.shape[:1] == (n_rows,):
        return value[indices]
    if isinstance(value, list) and len(value) == n_rows:
        return [value[int(idx)] for idx in indices]
    if isinstance(value, dict):
        return {key: _subset_first_axis(item, indices, n_rows) for key, item in value.items()}
    return value


def subset_product_rows(product: Mapping[str, Any], indices: np.ndarray) -> dict[str, Any]:
    indices = np.asarray(indices, dtype=np.int64)
    n_rows = len(product["object_id"])
    output = {
        key: _subset_first_axis(value, indices, n_rows)
        for key, value in product.items()
    }
    metadata = dict(product.get("metadata", {}))
    metadata["source_n_rows_before_morphology_filter"] = int(n_rows)
    metadata["n_usable_rows"] = int(len(indices))
    output["metadata"] = metadata
    return output


def refresh_morphology_split_labels(
    product: Mapping[str, Any],
    config: AIONMorphologyConfig,
) -> dict[str, Any]:
    """Assign post-morphology splits, optionally preserving photometry labels."""
    config = config.normalized()
    output = dict(product)
    old_metadata = dict(output.get("metadata", {}))
    old_split_counts = old_metadata.get("split_counts")
    if config.preserve_photometry_splits:
        source_labels = output.get("photometry_split_labels")
        if source_labels is None:
            raise RuntimeError(
                "The cached morphology product predates preserved photometry splits. "
                "Rebuild it with --force-rebuild-tokens."
            )
        split_labels = np.asarray(source_labels, dtype=object)
        if split_labels.shape != (len(output["object_id"]),):
            raise ValueError("photometry_split_labels do not match morphology product rows.")
        split_count_scope = "preserved_photometry_assignments_after_morphology_filter"
    else:
        split_labels = make_split_labels(
            output["field"],
            split_strategy=config.split_strategy,
            train_fraction=config.train_fraction,
            test_fraction=config.test_fraction,
            val_fraction=config.val_fraction,
            test_fields=config.test_fields,
            seed=config.seed,
        )
        split_count_scope = "post_morphology_filter"

    output["split_labels"] = list(split_labels)
    metadata = dict(old_metadata)
    population_report = metadata.get("population_report", {})
    original_split_counts = population_report.get("split_counts", {})
    if original_split_counts:
        metadata["pre_morphology_filter_split_counts"] = {
            split_name: int(record["n_gal"])
            for split_name, record in original_split_counts.items()
        }
    elif old_split_counts is not None and int(sum(old_split_counts.values())) != len(split_labels):
        metadata.setdefault("pre_morphology_filter_split_counts", old_split_counts)
    metadata.update(
        split_metadata(
            split_labels,
            config.split_strategy,
            config.train_fraction,
            config.test_fraction,
            config.val_fraction,
        )
    )
    metadata["split_count_scope"] = split_count_scope
    metadata["n_usable_rows"] = int(len(split_labels))
    output["metadata"] = metadata
    return output


def cache_aion_morphology_tokens(config: AIONMorphologyConfig | None = None, **overrides: Any) -> dict[str, Any]:
    config = AIONMorphologyConfig() if config is None else config
    if overrides:
        config = replace(config, **overrides)
    config = config.normalized()
    paths = resolve_morphology_paths(config)
    image_paths = discover_morphology_image_paths(config.morphology_dir)
    tile_manifest = tuple(
        str(path.relative_to(config.morphology_dir))
        for path in image_paths
    )
    product_path = Path(paths["morphology_product_path"])
    token_ids_path = Path(paths["token_ids_path"])
    if (
        product_path.exists()
        and token_ids_path.exists()
        and not config.force_rebuild_tokens
        and not config.force_rebuild_photometry
    ):
        product = torch.load(product_path, map_location="cpu", weights_only=False)
        metadata = dict(product.get("metadata", {}))
        cached_preprocessing = (
            metadata.get("image_background_mode"),
            metadata.get("image_flux_scale"),
            metadata.get("min_cutout_weight_coverage"),
            metadata.get("aion_image_band_alias"),
            tuple(metadata.get("morphology_tile_manifest", ())),
        )
        requested_preprocessing = (
            config.image_background_mode,
            float(config.image_flux_scale),
            float(config.min_cutout_weight_coverage),
            config.aion_image_band_alias,
            tile_manifest,
        )
        reporting_available = (
            "population_report" in metadata
            and "photometry_split_labels" in product
        )
        if (
            cached_preprocessing == requested_preprocessing
            and (not config.preserve_photometry_splits or reporting_available)
        ):
            product = refresh_morphology_split_labels(product, config)
            torch.save(product, product_path)
            return product
        if cached_preprocessing == requested_preprocessing:
            warning_message = (
                "Cached morphology tokens predate population reporting and preserved "
                "photometry splits; rebuilding them."
            )
        else:
            warning_message = (
                "Cached morphology preprocessing does not match the requested config; "
                "rebuilding AION image tokens."
            )
        warnings.warn(
            warning_message,
            RuntimeWarning,
            stacklevel=2,
        )

    product = build_or_load_photometry_product(config)
    extra_band_valid, report_bands = load_product_extra_band_validity(product)
    ra, dec = load_product_sky_positions(product)
    tiles = load_morphology_tiles(config.morphology_dir, image_paths=image_paths)
    print(f"Loaded {len(tiles):,} morphology science tiles from {config.morphology_dir}")
    try:
        tile_index, x_image, y_image = assign_tiles_to_positions(
            ra,
            dec,
            tiles,
            chunk_size=config.tile_assignment_chunk_size,
        )
        assigned_tile_counts = np.bincount(
            tile_index[tile_index >= 0],
            minlength=len(tiles),
        )

        device = select_torch_device(config.device_choice)
        codec_manager, image_modality_type, quantizer_levels = load_aion_image_codec(device)
        vocab_size = int(math.prod(quantizer_levels))

        token_cache_dir = Path(paths["token_cache_dir"])
        token_cache_dir.mkdir(parents=True, exist_ok=True)
        n_rows = len(product["object_id"])
        token_ids = np.lib.format.open_memmap(
            paths["token_ids_path"],
            mode="w+",
            dtype=np.uint16,
            shape=(n_rows, AION_IMAGE_GRID_SIZE * AION_IMAGE_GRID_SIZE),
        )
        token_ids[:] = 0
        quality = np.lib.format.open_memmap(
            paths["token_quality_path"],
            mode="w+",
            dtype=token_quality_dtype(),
            shape=(n_rows,),
        )
        quality["object_id"] = np.asarray(product["object_id"], dtype=np.int64)
        quality["assigned_tile"] = tile_index
        quality["x_image"] = x_image
        quality["y_image"] = y_image
        quality["weight_coverage"] = 0.0
        quality["background"] = 0.0
        quality["finite_fraction"] = 0.0
        quality["token_available"] = False
        token_histogram = np.zeros(vocab_size, dtype=np.int64)

        batch_cutouts: list[np.ndarray] = []
        batch_rows: list[int] = []

        def flush_batch() -> None:
            if not batch_cutouts:
                return
            encoded = tokenize_cutout_batch(
                np.stack(batch_cutouts, axis=0),
                codec_manager=codec_manager,
                image_modality_type=image_modality_type,
                device=device,
                band_alias=config.aion_image_band_alias,
            )
            if encoded.shape[1] != AION_IMAGE_GRID_SIZE * AION_IMAGE_GRID_SIZE:
                raise RuntimeError(f"Unexpected AION image token shape: {encoded.shape}")
            if int(encoded.max()) > np.iinfo(np.uint16).max:
                raise RuntimeError("Image token IDs do not fit in uint16.")
            rows = np.asarray(batch_rows, dtype=np.int64)
            token_ids[rows] = encoded.astype(np.uint16, copy=False)
            quality["token_available"][rows] = True
            token_histogram[:] += np.bincount(encoded.reshape(-1), minlength=vocab_size)[:vocab_size]
            batch_cutouts.clear()
            batch_rows.clear()

        for tile_idx, tile in enumerate(tiles):
            try:
                rows = np.flatnonzero(tile_index == tile_idx)
                for row in rows:
                    cutout, stats = _extract_cutout(
                        tile,
                        float(x_image[row]),
                        float(y_image[row]),
                        cutout_size=config.cutout_size,
                        background_mode=config.image_background_mode,
                        image_flux_scale=config.image_flux_scale,
                    )
                    quality["weight_coverage"][row] = stats["weight_coverage"]
                    quality["background"][row] = stats["background"]
                    quality["finite_fraction"][row] = stats["finite_fraction"]
                    if stats["weight_coverage"] < config.min_cutout_weight_coverage:
                        continue
                    batch_cutouts.append(cutout)
                    batch_rows.append(int(row))
                    if len(batch_rows) >= config.token_batch_size:
                        flush_batch()
                flush_batch()
            finally:
                tile.close()

        token_ids.flush()
        quality.flush()
        np.save(paths["token_histogram_path"], token_histogram)
    finally:
        for tile in tiles:
            tile.close()

    quality_array = np.load(paths["token_quality_path"], mmap_mode="r")
    kept_indices = np.flatnonzero(np.asarray(quality_array["token_available"], dtype=bool))
    if len(kept_indices) == 0:
        raise RuntimeError(
            "No CLAUDS rows produced usable morphology tokens. "
            "Check morphology_dir, sky coverage, and min_cutout_weight_coverage."
        )
    population_report = build_morphology_population_report(
        product,
        morphology_available=np.asarray(quality_array["token_available"], dtype=bool),
        extra_band_valid=extra_band_valid,
        extra_bands=report_bands,
    )
    morphology_product = subset_product_rows(product, kept_indices)
    morphology_product["photometry_split_labels"] = list(
        np.asarray(product["split_labels"], dtype=object)[kept_indices]
    )
    morphology_product["ra"] = torch.from_numpy(ra[kept_indices].astype(np.float64))
    morphology_product["dec"] = torch.from_numpy(dec[kept_indices].astype(np.float64))
    morphology_product["image_token_ids_path"] = str(Path(paths["token_ids_path"]))
    morphology_product["image_token_row_indices"] = torch.from_numpy(kept_indices.astype(np.int64))
    morphology_product["image_quality"] = np.asarray(quality_array[kept_indices])

    metadata = dict(morphology_product.get("metadata", {}))
    metadata.update(
        {
            "morphology_tag": paths["morphology_tag"],
            "morphology_dir": str(config.morphology_dir),
            "morphology_tile_manifest": list(tile_manifest),
            "n_morphology_tiles_loaded": int(len(tiles)),
            "n_morphology_tiles_with_assigned_rows": int(np.sum(assigned_tile_counts > 0)),
            "morphology_tile_assigned_row_counts": assigned_tile_counts.tolist(),
            "population_report": population_report,
            "aion_image_tokenizer_only": True,
            "aion_image_band_alias": config.aion_image_band_alias,
            "aion_image_token_key": AION_IMAGE_TOKEN_KEY,
            "aion_image_quantizer_levels": list(quantizer_levels),
            "aion_image_vocab_size": int(math.prod(quantizer_levels)),
            "aion_image_cutout_size": int(config.cutout_size),
            "aion_image_grid_size": AION_IMAGE_GRID_SIZE,
            "image_background_mode": config.image_background_mode,
            "image_flux_scale": float(config.image_flux_scale),
            "min_cutout_weight_coverage": float(config.min_cutout_weight_coverage),
            "n_rows_before_morphology_filter": int(len(product["object_id"])),
            "n_rows_with_tile_assignment": int(np.sum(tile_index >= 0)),
            "n_rows_with_aion_image_tokens": int(len(kept_indices)),
            "token_ids_path": str(Path(paths["token_ids_path"])),
            "token_quality_path": str(Path(paths["token_quality_path"])),
            "token_histogram_path": str(Path(paths["token_histogram_path"])),
        }
    )
    morphology_product["metadata"] = metadata
    morphology_product = refresh_morphology_split_labels(morphology_product, config)
    product_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(morphology_product, product_path)
    return morphology_product


class FSQTokenDecoder(nn.Module):
    """Decode AION FSQ token IDs into normalized scalar quantizer factors."""

    def __init__(
        self,
        levels: Sequence[int] = DEFAULT_AION_IMAGE_QUANTIZER_LEVELS,
        *,
        grid_size: int = AION_IMAGE_GRID_SIZE,
    ):
        super().__init__()
        levels_tensor = torch.as_tensor(tuple(int(v) for v in levels), dtype=torch.long)
        basis = torch.cumprod(
            torch.as_tensor([1, *levels_tensor[:-1].tolist()], dtype=torch.long),
            dim=0,
        )
        self.register_buffer("levels", levels_tensor)
        self.register_buffer("basis", basis)
        self.grid_size = int(grid_size)

    @property
    def embedding_dim(self) -> int:
        return int(self.levels.numel())

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = token_ids.long()
        components = (ids.unsqueeze(-1) // self.basis) % self.levels
        half_width = (self.levels // 2).to(dtype=torch.float32)
        decoded = (components.to(dtype=torch.float32) - half_width) / half_width
        decoded = decoded.movedim(-1, 1)
        return decoded.reshape(
            ids.shape[0],
            self.embedding_dim,
            self.grid_size,
            self.grid_size,
        )


class ImageTokenFactorEncoder(nn.Module):
    def __init__(
        self,
        *,
        levels: Sequence[int],
        hidden_dim: int = 512,
        output_dim: int = 128,
        grid_size: int = AION_IMAGE_GRID_SIZE,
    ):
        super().__init__()
        self.decoder = FSQTokenDecoder(levels, grid_size=grid_size)
        input_dim = self.decoder.embedding_dim * grid_size * grid_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        factors = self.decoder(token_ids)
        return self.net(factors)


class PhotometryOnlyPhotoZModel(nn.Module):
    def __init__(
        self,
        *,
        extra_feature_dim: int,
        n_z_bins: int,
        photometry_hidden_dim: int = 128,
        head_hidden_dim: int = 256,
    ):
        super().__init__()
        self.photometry_encoder = ExtraPhotometryEncoder(extra_feature_dim, photometry_hidden_dim)
        self.photoz_head = PhotoZHead(photometry_hidden_dim, n_z_bins, head_hidden_dim)

    def forward(self, extra_features: torch.Tensor, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        del token_ids
        h_photo = self.photometry_encoder(extra_features)
        return self.photoz_head(h_photo)


class MorphologyResidualPhotoZModel(nn.Module):
    def __init__(
        self,
        *,
        extra_feature_dim: int,
        n_z_bins: int,
        quantizer_levels: Sequence[int],
        photometry_hidden_dim: int = 128,
        image_hidden_dim: int = 512,
        image_embedding_dim: int = 128,
        head_hidden_dim: int = 256,
    ):
        super().__init__()
        self.photometry_encoder = ExtraPhotometryEncoder(extra_feature_dim, photometry_hidden_dim)
        self.photometry_head = PhotoZHead(photometry_hidden_dim, n_z_bins, head_hidden_dim)
        self.image_encoder = ImageTokenFactorEncoder(
            levels=quantizer_levels,
            hidden_dim=image_hidden_dim,
            output_dim=image_embedding_dim,
        )
        self.image_delta_head = nn.Sequential(
            nn.Linear(photometry_hidden_dim + image_embedding_dim, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, n_z_bins),
        )

    def forward(self, extra_features: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        h_photo = self.photometry_encoder(extra_features)
        base_logits = self.photometry_head(h_photo)
        h_image = self.image_encoder(token_ids)
        delta_logits = self.image_delta_head(torch.cat([h_photo, h_image], dim=-1))
        return base_logits + delta_logits


class AIONMagnitudeMorphologyResidualPhotoZModel(nn.Module):
    """Add an image-token MLP residual to the standard grizy-AION head."""

    def __init__(
        self,
        *,
        aion_dim: int,
        n_z_bins: int,
        quantizer_levels: Sequence[int],
        image_hidden_dim: int = 512,
        image_embedding_dim: int = 128,
        head_hidden_dim: int = 256,
    ):
        super().__init__()
        self.photometry_head = PhotoZHead(aion_dim, n_z_bins, head_hidden_dim)
        self.image_encoder = ImageTokenFactorEncoder(
            levels=quantizer_levels,
            hidden_dim=image_hidden_dim,
            output_dim=image_embedding_dim,
        )
        self.image_delta_head = nn.Sequential(
            nn.Linear(aion_dim + image_embedding_dim, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, n_z_bins),
        )

    def forward(self, aion_embedding: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        base_logits = self.photometry_head(aion_embedding)
        h_image = self.image_encoder(token_ids)
        delta_logits = self.image_delta_head(torch.cat([aion_embedding, h_image], dim=-1))
        return base_logits + delta_logits


@dataclass
class MorphologyTokenBatch:
    object_id: list[Any]
    field: list[Any]
    aion_embedding: torch.Tensor
    extra_features: torch.Tensor
    token_ids: torch.Tensor
    z_spec: torch.Tensor | None = None
    redshift_reference: dict[str, torch.Tensor] | None = None


class MorphologyTokenDataset(Dataset):
    def __init__(
        self,
        product: Mapping[str, Any],
        *,
        shuffle_tokens: bool = False,
        seed: int = 42,
    ):
        self.object_ids = list(product["object_id"])
        self.fields = list(product["field"])
        self.aion_embedding = torch.as_tensor(product["aion_embedding"], dtype=torch.float32)
        self.extra_features = torch.as_tensor(product["extra_features"], dtype=torch.float32)
        self.z_spec = None if product.get("z_spec") is None else torch.as_tensor(product["z_spec"], dtype=torch.float32)
        self.redshift_reference = {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in dict(product.get("redshift_reference") or {}).items()
        }
        self.token_ids = np.load(product["image_token_ids_path"], mmap_mode="r")
        self.token_row_indices = np.asarray(product["image_token_row_indices"], dtype=np.int64)
        if shuffle_tokens:
            rng = np.random.default_rng(seed)
            self.token_source_order = rng.permutation(len(self.token_row_indices))
        else:
            self.token_source_order = np.arange(len(self.token_row_indices), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        token_source = int(self.token_source_order[idx])
        token_row = int(self.token_row_indices[token_source])
        item = {
            "object_id": self.object_ids[idx],
            "field": self.fields[idx],
            "aion_embedding": self.aion_embedding[idx],
            "extra_features": self.extra_features[idx],
            "token_ids": torch.from_numpy(np.asarray(self.token_ids[token_row], dtype=np.int64)),
        }
        if self.z_spec is not None:
            item["z_spec"] = self.z_spec[idx]
        if self.redshift_reference:
            item["redshift_reference"] = {
                key: value[idx]
                for key, value in self.redshift_reference.items()
            }
        return item


def collate_morphology_token_batch(items: list[dict[str, Any]]) -> MorphologyTokenBatch:
    z_spec = None
    if "z_spec" in items[0]:
        z_spec = torch.stack([item["z_spec"] for item in items]).float()
    redshift_reference = None
    if "redshift_reference" in items[0]:
        reference_keys = items[0]["redshift_reference"].keys()
        redshift_reference = {
            key: torch.stack([item["redshift_reference"][key] for item in items]).float()
            for key in reference_keys
        }
    return MorphologyTokenBatch(
        object_id=[item["object_id"] for item in items],
        field=[item["field"] for item in items],
        aion_embedding=torch.stack([item["aion_embedding"] for item in items]).float(),
        extra_features=torch.stack([item["extra_features"] for item in items]).float(),
        token_ids=torch.stack([item["token_ids"] for item in items]).long(),
        z_spec=z_spec,
        redshift_reference=redshift_reference,
    )


def make_morphology_loader(
    dataset: MorphologyTokenDataset,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_morphology_token_batch,
    )


def morphology_product_to_dataset(
    product: Mapping[str, Any],
    *,
    shuffle_tokens: bool = False,
    seed: int = 42,
) -> MorphologyTokenDataset:
    return MorphologyTokenDataset(product, shuffle_tokens=shuffle_tokens, seed=seed)


def split_morphology_product(product: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    split_labels = np.asarray(product["split_labels"], dtype=object)
    return (
        subset_product_rows(product, np.flatnonzero(split_labels == "train")),
        subset_product_rows(product, np.flatnonzero(split_labels == "val")),
        subset_product_rows(product, np.flatnonzero(split_labels == "test")),
    )


def _build_morphology_model(
    *,
    model_kind: str,
    aion_dim: int,
    extra_feature_dim: int,
    n_z_bins: int,
    quantizer_levels: Sequence[int],
    config: AIONMorphologyConfig,
) -> nn.Module:
    if model_kind == "photometry":
        return PhotometryOnlyPhotoZModel(
            extra_feature_dim=extra_feature_dim,
            n_z_bins=n_z_bins,
            photometry_hidden_dim=config.photometry_hidden_dim,
            head_hidden_dim=config.head_hidden_dim,
        )
    if model_kind in {"morphology", "shuffled_morphology"}:
        return MorphologyResidualPhotoZModel(
            extra_feature_dim=extra_feature_dim,
            n_z_bins=n_z_bins,
            quantizer_levels=quantizer_levels,
            photometry_hidden_dim=config.photometry_hidden_dim,
            image_hidden_dim=config.image_hidden_dim,
            image_embedding_dim=config.image_embedding_dim,
            head_hidden_dim=config.head_hidden_dim,
        )
    if model_kind == "aion":
        return AIONOnlyPhotoZModel(
            aion_dim=aion_dim,
            n_z_bins=n_z_bins,
            head_hidden_dim=config.head_hidden_dim,
        )
    if model_kind in {"aion_morphology", "shuffled_aion_morphology"}:
        return AIONMagnitudeMorphologyResidualPhotoZModel(
            aion_dim=aion_dim,
            n_z_bins=n_z_bins,
            quantizer_levels=quantizer_levels,
            image_hidden_dim=config.image_hidden_dim,
            image_embedding_dim=config.image_embedding_dim,
            head_hidden_dim=config.head_hidden_dim,
        )
    raise ValueError(f"Unknown model_kind: {model_kind}")


def _logits_from_morphology_batch(
    model: nn.Module,
    batch: MorphologyTokenBatch,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    if isinstance(model, AIONOnlyPhotoZModel):
        return model(batch.aion_embedding.to(device))
    token_ids = batch.token_ids.to(device)
    if isinstance(model, AIONMagnitudeMorphologyResidualPhotoZModel):
        return model(batch.aion_embedding.to(device), token_ids)
    return model(batch.extra_features.to(device), token_ids)


def train_morphology_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    redshift_edges: torch.Tensor,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if batch.z_spec is None:
            raise ValueError("Training requires z_spec labels.")
        optimizer.zero_grad(set_to_none=True)
        logits = _logits_from_morphology_batch(model, batch, device=device)
        z_spec = batch.z_spec.to(device)
        loss = redshift_cross_entropy_loss(logits, z_spec, edges=redshift_edges)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(z_spec.shape[0])
        total_count += int(z_spec.shape[0])
    if total_count == 0:
        raise ValueError("No batches were processed during morphology training.")
    return total_loss / total_count


@torch.no_grad()
def evaluate_morphology_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | str,
    redshift_edges: torch.Tensor,
    redshift_centers: torch.Tensor,
    max_batches: int | None = None,
) -> dict[str, torch.Tensor | float]:
    model.eval()
    logits_parts = []
    z_parts = []
    redshift_reference_parts: dict[str, list[torch.Tensor]] = {}
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        logits = _logits_from_morphology_batch(model, batch, device=device)
        logits_parts.append(logits.detach().cpu())
        if batch.z_spec is not None:
            z_parts.append(batch.z_spec.cpu())
        if batch.redshift_reference:
            for key, values in batch.redshift_reference.items():
                redshift_reference_parts.setdefault(key, []).append(values.cpu())
    if not logits_parts:
        raise ValueError("No batches were processed during morphology evaluation.")
    logits = torch.cat(logits_parts, dim=0)
    output: dict[str, torch.Tensor | float] = {
        "logits": logits,
        "redshift_edges": redshift_edges.detach().cpu(),
        "redshift_centers": redshift_centers.detach().cpu(),
    }
    output.update(predict_photoz_from_logits(logits, centers=redshift_centers))
    if z_parts:
        z_spec = torch.cat(z_parts, dim=0)
        output["z_spec"] = z_spec
        output["loss"] = redshift_cross_entropy_loss(logits, z_spec, edges=redshift_edges).item()
    if redshift_reference_parts:
        output["redshift_reference"] = {
            key: torch.cat(parts, dim=0)
            for key, parts in redshift_reference_parts.items()
        }
    return output


def train_single_morphology_model(
    product: Mapping[str, Any],
    model_kind: str,
    *,
    output_dir: str | Path,
    config: AIONMorphologyConfig,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    config = config.normalized()
    device = resolve_torch_device(device)
    redshift_edges, redshift_centers = make_redshift_grid(config.z_min, config.z_max, config.n_z_bins)
    train_product, val_product, test_product = split_morphology_product(product)
    if len(train_product["object_id"]) == 0:
        raise ValueError("No training rows are available after morphology filtering.")
    if len(val_product["object_id"]) == 0:
        raise ValueError("No validation rows are available after morphology filtering.")

    shuffle_tokens = model_kind in {
        "shuffled_morphology",
        "shuffled_aion_morphology",
    }
    train_dataset = morphology_product_to_dataset(train_product, shuffle_tokens=shuffle_tokens, seed=config.seed)
    val_dataset = morphology_product_to_dataset(val_product, shuffle_tokens=shuffle_tokens, seed=config.seed + 1)
    test_dataset = morphology_product_to_dataset(test_product, shuffle_tokens=shuffle_tokens, seed=config.seed + 2)
    train_loader = make_morphology_loader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
    )
    val_loader = make_morphology_loader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
    )

    metadata = dict(product.get("metadata", {}))
    quantizer_levels = metadata.get("aion_image_quantizer_levels", DEFAULT_AION_IMAGE_QUANTIZER_LEVELS)
    aion_dim = int(torch.as_tensor(product["aion_embedding"]).shape[1])
    extra_feature_dim = int(torch.as_tensor(product["extra_features"]).shape[1])
    if model_kind.startswith("aion") or model_kind == "shuffled_aion_morphology":
        if aion_dim == 0:
            raise ValueError(f"model_kind={model_kind!r} requires grizy AION embeddings.")
    elif extra_feature_dim == 0:
        raise ValueError(f"model_kind={model_kind!r} requires MLP photometry features.")
    model = _build_morphology_model(
        model_kind=model_kind,
        aion_dim=aion_dim,
        extra_feature_dim=extra_feature_dim,
        n_z_bins=config.n_z_bins,
        quantizer_levels=quantizer_levels,
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history = []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    set_random_seed(config.seed)
    for epoch in range(config.epochs):
        train_loss = train_morphology_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            redshift_edges=redshift_edges,
            max_batches=config.max_train_batches,
        )
        val_eval = evaluate_morphology_model(
            model,
            val_loader,
            device=device,
            redshift_edges=redshift_edges,
            redshift_centers=redshift_centers,
            max_batches=config.max_eval_batches,
        )
        val_metrics = summarize_pdf_metrics(val_eval)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"{model_kind:20s} epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['cross_entropy']:.4f} "
            f"val_nmad={val_metrics['nmad']:.4f}"
        )
        if val_metrics["cross_entropy"] < best_val_loss:
            best_val_loss = val_metrics["cross_entropy"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                config.early_stopping_patience is not None
                and epochs_without_improvement >= config.early_stopping_patience
            ):
                print(f"{model_kind:20s} early stopping after {epochs_without_improvement} non-improving epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_eval = evaluate_morphology_model(
        model,
        val_loader,
        device=device,
        redshift_edges=redshift_edges,
        redshift_centers=redshift_centers,
        max_batches=config.max_eval_batches,
    )
    final_metrics: dict[str, Any] = {"val": summarize_pdf_metrics(val_eval)}
    calibration: dict[str, Any] = {
        "val": save_calibration_artifacts(val_eval, output_dir, f"{model_kind}_val")
    }
    test_eval = None
    if len(test_product["object_id"]) > 0:
        test_loader = make_morphology_loader(
            test_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
        )
        test_eval = evaluate_morphology_model(
            model,
            test_loader,
            device=device,
            redshift_edges=redshift_edges,
            redshift_centers=redshift_centers,
            max_batches=config.max_eval_batches,
        )
        final_metrics["test"] = summarize_pdf_metrics(test_eval)
        calibration["test"] = save_calibration_artifacts(test_eval, output_dir, f"{model_kind}_test")
        calibration["conformal_hpd"] = evaluate_conformal_hpd(val_eval, test_eval)
    else:
        calibration["conformal_hpd"] = {
            "status": "skipped_no_test_split",
            "message": "No test split rows are available after morphology filtering.",
        }

    checkpoint_path = output_dir / f"{model_kind}.pt"
    torch.save(
        {
            "model_kind": model_kind,
            "state_dict": model.state_dict(),
            "history": history,
            "final_metrics": final_metrics,
            "calibration": calibration,
            "metadata": {
                **metadata,
                "feature_names": list(product.get("feature_names", [])),
                "n_z_bins": config.n_z_bins,
                "redshift_edges": redshift_edges.detach().cpu(),
                "redshift_centers": redshift_centers.detach().cpu(),
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "train_batch_size": config.train_batch_size,
                "eval_batch_size": config.eval_batch_size,
                "epochs": config.epochs,
                "model_kind": model_kind,
                "shuffle_tokens": bool(shuffle_tokens),
            },
        },
        checkpoint_path,
    )
    return {
        "model_kind": model_kind,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
        "final_metrics": final_metrics,
        "calibration": calibration,
        "model": model,
        "val_evaluation": val_eval,
        "test_evaluation": test_eval,
    }


_MORPHOLOGY_COMPARISON_LABELS = {
    "photometry": "grizy-MLP-only",
    "morphology": "grizy-MLP+tokenized-u-image",
    "aion": "grizy-aion-only",
    "aion_morphology": "grizy-aion+tokenized-u-image",
}



def morphology_comparison_prefix(
    output_dir: str | Path,
    model_kinds: Sequence[str],
) -> Path | None:
    if len(model_kinds) != 2:
        return None
    model_kind_1, model_kind_2 = tuple(model_kinds)
    return Path(output_dir) / f"{model_kind_1}_{model_kind_2}_comparison"


def save_morphology_population_report(
    product: Mapping[str, Any],
    *,
    model_kinds: Sequence[str],
    output_dir: str | Path,
) -> str | None:
    report = dict(product.get("metadata", {})).get("population_report")
    prefix = morphology_comparison_prefix(output_dir, model_kinds)
    if report is None or prefix is None:
        return None
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(f"{prefix}_out.log")
    report_path.write_text(format_morphology_population_report(report))
    return str(report_path)


def morphology_comparison_labels(config: AIONMorphologyConfig) -> tuple[str, str] | None:
    if tuple(config.model_kinds) == ("photometry", "morphology") and config.extra_bands:
        return (
            "all-magnitude-MLP-only",
            "all-magnitude-MLP+tokenized-u-image",
        )
    return None


def save_morphology_comparison_artifacts(
    results: Mapping[str, Mapping[str, Any]],
    *,
    model_kinds: Sequence[str],
    output_dir: str | Path,
    tomographic_samples: int,
    comparison_labels: tuple[str, str] | None = None,
) -> dict[str, str]:
    """Save the same paired diagnostics produced by standard_comparison.sh."""
    if len(model_kinds) != 2:
        return {}
    model_kind_1, model_kind_2 = tuple(model_kinds)
    result_1 = results[model_kind_1]
    result_2 = results[model_kind_2]
    evaluation_1 = result_1.get("test_evaluation") or result_1["val_evaluation"]
    evaluation_2 = result_2.get("test_evaluation") or result_2["val_evaluation"]
    labels = comparison_labels or (
        _MORPHOLOGY_COMPARISON_LABELS.get(model_kind_1, model_kind_1),
        _MORPHOLOGY_COMPARISON_LABELS.get(model_kind_2, model_kind_2),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = morphology_comparison_prefix(output_dir, model_kinds)
    assert prefix is not None
    artifacts = {
        "loss": str(Path(f"{prefix}_loss.jpeg")),
        "scatter": str(Path(f"{prefix}_scatter.jpeg")),
        "pit": str(Path(f"{prefix}_pit.jpeg")),
        "nz": str(Path(f"{prefix}_nz.jpeg")),
        "nztomo": str(Path(f"{prefix}_nztomo.jpeg")),
    }

    fig, _, _ = compare_config_loss(
        result_1,
        result_2,
        output_path=artifacts["loss"],
        labels=labels,
    )
    plt.close(fig)
    fig, _ = compare_zpred_vs_zphot(
        evaluation_1,
        evaluation_2,
        output_path=artifacts["scatter"],
        labels=labels,
        pred_key="z_p50",
        pmax=5.0,
        show_metrics=True,
    )
    plt.close(fig)
    fig, _ = compare_pit_histogram(
        evaluation_1,
        evaluation_2,
        output_path=artifacts["pit"],
        labels=labels,
    )
    plt.close(fig)
    fig, _, _ = compare_redshift_probability_distribution(
        evaluation_1,
        evaluation_2,
        output_path=artifacts["nz"],
        labels=labels,
        gaussian_sigma_bins=1.0,
    )
    plt.close(fig)
    fig, _, _ = compare_nz_lensing_alike(
        evaluation_1,
        evaluation_2,
        output_path=artifacts["nztomo"],
        labels=labels,
        zphot_bin=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        gaussian_sigma_bins=1.0,
        inferred_bin_key="z_p50",
        n_samples_per_object=tomographic_samples,
    )
    plt.close(fig)
    return artifacts


def run_morphology_experiment(config: AIONMorphologyConfig | None = None, **overrides: Any) -> dict[str, Any]:
    config = AIONMorphologyConfig() if config is None else config
    if overrides:
        config = replace(config, **overrides)
    config = config.normalized()
    product = cache_aion_morphology_tokens(config)
    paths = resolve_morphology_paths(config)
    population_report_path = save_morphology_population_report(
        product,
        model_kinds=config.model_kinds,
        output_dir=paths["morphology_output_dir"],
    )
    device = select_torch_device(config.device_choice)
    results = {}
    for model_kind in config.model_kinds:
        results[model_kind] = train_single_morphology_model(
            product,
            model_kind,
            output_dir=paths["morphology_output_dir"],
            config=config,
            device=device,
        )
    comparison_artifacts = save_morphology_comparison_artifacts(
        results,
        model_kinds=config.model_kinds,
        output_dir=paths["morphology_output_dir"],
        tomographic_samples=config.tomographic_samples,
        comparison_labels=morphology_comparison_labels(config),
    )
    summary_path = Path(paths["morphology_output_dir"]) / "morphology_results.pt"
    torch.save(results, summary_path)
    return {
        "config": config,
        "paths": paths,
        "product": product,
        "results": results,
        "comparison_artifacts": comparison_artifacts,
        "population_report_path": population_report_path,
        "summary_path": str(summary_path),
    }


def _parse_model_kinds(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("cache", "train"), default="train")
    parser.add_argument("--catalogue-path", type=Path, default=AIONMorphologyConfig.catalogue_path)
    parser.add_argument("--morphology-dir", type=Path, default=AIONMorphologyConfig.morphology_dir)
    parser.add_argument("--cache-root", type=Path, default=AIONMorphologyConfig.cache_root)
    parser.add_argument("--output-dir", type=Path, default=AIONMorphologyConfig.output_dir)
    parser.add_argument("--max-rows", type=_optional_int, default=AIONMorphologyConfig.max_rows)
    parser.add_argument("--sample-mode", choices=("head", "random"), default=AIONMorphologyConfig.sample_mode)
    parser.add_argument("--sample-row-start", type=int, default=None)
    parser.add_argument("--sample-row-stop", type=int, default=None)
    parser.add_argument("--sample-require-valid-bands", default="")
    parser.add_argument("--z-max", type=float, default=AIONMorphologyConfig.z_max)
    parser.add_argument("--n-z-bins", type=int, default=AIONMorphologyConfig.n_z_bins)
    parser.add_argument("--include-z-max", action="store_true")
    parser.add_argument("--extra-valid-flags", action="store_true")
    parser.add_argument(
        "--grizy-only",
        action="store_true",
        help="Exclude u*, Y, J, H, and Ks from MLP features.",
    )
    parser.add_argument("--use-aion-magnitude-embedding", action="store_true")
    parser.add_argument(
        "--aion-embedding-batch-size",
        type=int,
        default=AIONMorphologyConfig.aion_embedding_batch_size,
    )
    parser.add_argument("--min-cutout-weight-coverage", type=float, default=AIONMorphologyConfig.min_cutout_weight_coverage)
    parser.add_argument("--image-flux-scale", type=float, default=AIONMorphologyConfig.image_flux_scale)
    parser.add_argument("--token-batch-size", type=int, default=AIONMorphologyConfig.token_batch_size)
    parser.add_argument("--force-rebuild-tokens", action="store_true")
    parser.add_argument("--force-rebuild-photometry", action="store_true")
    parser.add_argument(
        "--preserve-photometry-splits",
        action="store_true",
        help="Keep each selected galaxy original photometry train/validation/test assignment.",
    )
    parser.add_argument("--model-kinds", type=_parse_model_kinds, default=AIONMorphologyConfig.model_kinds)
    parser.add_argument("--epochs", type=int, default=AIONMorphologyConfig.epochs)
    parser.add_argument("--train-batch-size", type=int, default=AIONMorphologyConfig.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=AIONMorphologyConfig.eval_batch_size)
    parser.add_argument(
        "--tomographic-samples",
        type=int,
        default=AIONMorphologyConfig.tomographic_samples,
    )
    parser.add_argument("--device", default=AIONMorphologyConfig.device_choice)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    require_valid = tuple(
        part.strip()
        for part in args.sample_require_valid_bands.split(",")
        if part.strip()
    )
    config = AIONMorphologyConfig(
        catalogue_path=args.catalogue_path,
        morphology_dir=args.morphology_dir,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        sample_mode=args.sample_mode,
        sample_row_start=args.sample_row_start,
        sample_row_stop=args.sample_row_stop,
        sample_require_valid_bands=require_valid,
        z_max=args.z_max,
        redshift_include_max=bool(args.include_z_max),
        n_z_bins=args.n_z_bins,
        extra_bands=() if args.grizy_only else AIONMorphologyConfig().extra_bands,
        extra_band_include_valid_flags=bool(args.extra_valid_flags),
        use_aion_magnitude_embedding=bool(args.use_aion_magnitude_embedding),
        aion_embedding_batch_size=args.aion_embedding_batch_size,
        min_cutout_weight_coverage=args.min_cutout_weight_coverage,
        image_flux_scale=args.image_flux_scale,
        token_batch_size=args.token_batch_size,
        force_rebuild_tokens=bool(args.force_rebuild_tokens),
        force_rebuild_photometry=bool(args.force_rebuild_photometry),
        preserve_photometry_splits=bool(args.preserve_photometry_splits),
        model_kinds=args.model_kinds,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        tomographic_samples=args.tomographic_samples,
        device_choice=args.device,
    ).normalized()

    if args.command == "cache":
        product = cache_aion_morphology_tokens(config)
        print(f"Cached morphology product rows: {len(product['object_id']):,}")
        print(f"Product path: {resolve_morphology_paths(config)['morphology_product_path']}")
        return

    run = run_morphology_experiment(config)
    print(f"Morphology experiment summary: {run['summary_path']}")
    if run["population_report_path"] is not None:
        print(f"Population report: {run['population_report_path']}")
    for name, artifact_path in run["comparison_artifacts"].items():
        print(f"Comparison {name}: {artifact_path}")


if __name__ == "__main__":
    main()
