"""Physically informed CLAUDS and AION-image serialization for Qwen3.

This module intentionally leaves :mod:`aion_magnitude.FM_Qwen` unchanged as
the terse catalogue-only baseline.  Here Qwen receives CLAUDS magnitudes with
their survey, instrument, wavelength, and magnitude-system context, followed
by the ordered AION image-token grid under the neutral name ``tokenized galaxy
image``.  No fixed morphological interpretation is assigned to token IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .FM_Qwen import (
    QWEN_DEFAULT_MODELS,
    QWEN_POOLING_MODES,
    QwenEmbeddingConfig,
    extract_qwen_embeddings_from_texts,
    load_frozen_qwen,
    pool_qwen_hidden_states,
    qwen_embedding_metadata,
    require_transformers,
    resolve_qwen_dtype,
    resolve_qwen_model_path,
)
from .morphology import AION_IMAGE_GRID_SIZE


@dataclass(frozen=True)
class CLAUDBandDescription:
    canonical_name: str
    aliases: tuple[str, ...]
    facility_instrument: str
    wavelength_text: str
    spectral_region: str
    note: str = ""


# CLAUDS u/u* values are central wavelength and bandwidth from Sawicki et al.
# (2019). HSC values are measured effective wavelengths. VIRCAM values are
# representative response-weighted wavelengths from the COSMOS filter set.
CLAUDS_BAND_DESCRIPTIONS: tuple[CLAUDBandDescription, ...] = (
    CLAUDBandDescription(
        "u", ("u", "u_mag", "mag_u"), "CFHT/MegaCam",
        "central wavelength 3538 Angstrom; bandwidth 868 Angstrom",
        "near-ultraviolet",
        "This is the newer, bluer CLAUDS u filter.",
    ),
    CLAUDBandDescription(
        "u_star", ("u_star", "u*", "ustar", "uS", "u_star_mag", "mag_u_star"),
        "CFHT/MegaCam", "central wavelength 3743 Angstrom; bandwidth 758 Angstrom",
        "near-ultraviolet",
        "This is the older CLAUDS u-star filter; it is distinct from u and has a weak red leak near 5000 Angstrom.",
    ),
    CLAUDBandDescription("g", ("g", "g_mag", "mag_g"), "Subaru/Hyper Suprime-Cam", "effective wavelength about 4740 Angstrom", "blue optical"),
    CLAUDBandDescription("r", ("r", "r_mag", "mag_r"), "Subaru/Hyper Suprime-Cam", "effective wavelength about 6170 Angstrom", "optical"),
    CLAUDBandDescription("i", ("i", "i_mag", "mag_i"), "Subaru/Hyper Suprime-Cam", "effective wavelength about 7650 Angstrom", "red optical"),
    CLAUDBandDescription("z", ("z", "z_mag", "mag_z"), "Subaru/Hyper Suprime-Cam", "effective wavelength about 8890 Angstrom", "very-red optical"),
    CLAUDBandDescription(
        "y", ("y", "y_mag", "mag_y"), "Subaru/Hyper Suprime-Cam",
        "effective wavelength about 9760 Angstrom", "very-red optical",
        "Lowercase HSC y is a different passband from uppercase VISTA Y.",
    ),
    CLAUDBandDescription(
        "Y", ("Y", "Y_mag", "mag_Y"), "VISTA/VIRCAM",
        "representative wavelength about 1.0214 micrometre", "near-infrared",
        "Uppercase VISTA Y is a different passband from lowercase HSC y.",
    ),
    CLAUDBandDescription("J", ("J", "J_mag", "mag_J"), "VISTA/VIRCAM", "representative wavelength about 1.2535 micrometre", "near-infrared"),
    CLAUDBandDescription("H", ("H", "H_mag", "mag_H"), "VISTA/VIRCAM", "representative wavelength about 1.6453 micrometre", "near-infrared"),
    CLAUDBandDescription("Ks", ("Ks", "Ks_mag", "mag_Ks", "K_s", "K_s_mag"), "VISTA/VIRCAM", "representative wavelength about 2.1540 micrometre", "near-infrared"),
)


PHYSICAL_CONTEXT = (
    "These measurements sample one galaxy's observed-frame spectral energy distribution "
    "from near-ultraviolet to near-infrared. Magnitudes are AB magnitudes: a smaller "
    "magnitude means greater observed flux density, and magnitude differences are colours. "
    "A missing value means no usable measurement, not zero flux; it can result from survey "
    "coverage, imaging depth, masking, or measurement failure."
)

TOKEN_IMAGE_CONTEXT = (
    "An image tokenizer converted the observed galaxy cutout into an ordered grid of "
    "discrete tokens. The grid retains information from the image and may provide visual "
    "context complementary to the photometric magnitudes. No predefined interpretation "
    "of individual tokens is supplied."
)

QWEN_IMAGE_INPUT_MODES = ("full_grid", "center_crop")
QWEN_PHYSICAL_CONTEXT_MODES = ("none", "global", "compact", "full")


@dataclass(frozen=True)
class Qwen3SerializationConfig:
    schema_name: str = "clauds_physical_magnitudes_aion_image_v1"
    decimals: int = 5
    missing_token: str = "NA"
    image_grid_size: int = AION_IMAGE_GRID_SIZE
    image_input_mode: str = "full_grid"
    image_crop_size: int = 16
    include_physical_context: bool = True
    physical_context_mode: str = "full"
    include_image_context: bool = True
    include_unrecognized_features: bool = True
    image_label: str = "tokenized galaxy image"
    prefix: str = "Galaxy observation"
    final_marker: str | None = None
    band_descriptions: Sequence[CLAUDBandDescription] = field(
        default_factory=lambda: CLAUDS_BAND_DESCRIPTIONS
    )


@dataclass(frozen=True)
class Qwen3EmbeddingConfig(QwenEmbeddingConfig):
    """Qwen3 defaults sized for magnitude descriptions plus a 24x24 token grid."""

    model_path: str = "Qwen3-8B-Base"
    max_length: int = 2048


def _physical_context_mode(config: Qwen3SerializationConfig) -> str:
    """Resolve the context ablation while preserving the old boolean switch."""
    if not config.include_physical_context:
        return "none"
    mode = str(config.physical_context_mode).strip().lower()
    if mode not in QWEN_PHYSICAL_CONTEXT_MODES:
        raise ValueError(
            "physical_context_mode must be one of "
            f"{QWEN_PHYSICAL_CONTEXT_MODES}; received {config.physical_context_mode!r}."
        )
    return mode


def _format_value(value: Any, config: Qwen3SerializationConfig) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return config.missing_token
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or config.missing_token
    if not np.isfinite(number):
        return config.missing_token
    return f"{number:.{config.decimals}f}"


def _pop_alias(features: dict[str, Any], aliases: Sequence[str]) -> tuple[bool, str, Any]:
    for alias in aliases:
        if alias in features:
            return True, alias, features.pop(alias)
    return False, "", None


def serialize_tokenized_galaxy_image(
    token_ids: torch.Tensor | np.ndarray | Sequence[int],
    *,
    config: Qwen3SerializationConfig | None = None,
) -> str:
    """Serialize an ordered AION token grid without imposing morphology labels."""
    config = config or Qwen3SerializationConfig()
    tokens = torch.as_tensor(token_ids).detach().cpu()
    grid_size = int(config.image_grid_size)
    expected = grid_size * grid_size
    if tokens.ndim == 2 and tuple(tokens.shape) == (grid_size, grid_size):
        grid = tokens
    elif tokens.ndim == 1 and tokens.numel() == expected:
        grid = tokens.reshape(grid_size, grid_size)
    else:
        raise ValueError(
            "token_ids must be a flat grid or a square grid with "
            f"{expected} values; received shape {tuple(tokens.shape)}."
        )
    if tokens.is_floating_point() and not bool(torch.equal(tokens, tokens.round())):
        raise ValueError("AION image token IDs must be integers.")
    if config.image_input_mode not in QWEN_IMAGE_INPUT_MODES:
        raise ValueError(f"image_input_mode must be one of {QWEN_IMAGE_INPUT_MODES}.")
    serialized_grid_size = grid_size
    if config.image_input_mode == "center_crop":
        crop_size = int(config.image_crop_size)
        if crop_size <= 0 or crop_size > grid_size:
            raise ValueError("image_crop_size must be positive and no larger than image_grid_size.")
        if (grid_size - crop_size) % 2:
            raise ValueError(
                "A centered crop requires image_grid_size - image_crop_size to be even."
            )
        offset = (grid_size - crop_size) // 2
        grid = grid[offset : offset + crop_size, offset : offset + crop_size]
        serialized_grid_size = crop_size
    rows = [",".join(str(int(value)) for value in row.tolist()) for row in grid]
    context = f" {TOKEN_IMAGE_CONTEXT}" if config.include_image_context else ""
    return (
        f"{config.image_label}: source_grid_shape={grid_size}x{grid_size};"
        f" serialized_grid_shape={serialized_grid_size}x{serialized_grid_size};"
        f" image_input_mode={config.image_input_mode};"
        f"{context} ordered_token_rows=[" + ";".join(rows) + "]"
    )


def serialize_qwen3_observation(
    magnitude_features: Mapping[str, Any],
    image_token_ids: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    *,
    config: Qwen3SerializationConfig | None = None,
) -> str:
    """Serialize physical CLAUDS magnitudes first and optional image tokens second."""
    config = config or Qwen3SerializationConfig()
    physical_mode = _physical_context_mode(config)
    remaining = {str(name): value for name, value in magnitude_features.items()}
    sections = [f"{config.prefix}; schema={config.schema_name}."]
    if physical_mode != "none":
        sections.append(PHYSICAL_CONTEXT)

    magnitude_lines: list[str] = []
    for band in config.band_descriptions:
        found, input_name, raw_value = _pop_alias(remaining, band.aliases)
        if not found:
            continue
        value = _format_value(raw_value, config)
        if physical_mode == "full":
            line = (
                f"{band.canonical_name} AB magnitude={value}; instrument={band.facility_instrument}; "
                f"passband={band.wavelength_text}; region={band.spectral_region}."
            )
            if band.note:
                line += f" {band.note}"
        elif physical_mode == "compact":
            line = (
                f"{band.canonical_name} AB magnitude={value}; "
                f"passband={band.wavelength_text}; region={band.spectral_region}."
            )
        elif physical_mode == "global":
            line = f"{band.canonical_name} AB magnitude={value}."
        else:
            line = f"{input_name}={value}"
        magnitude_lines.append(line)
    if config.include_unrecognized_features:
        for name, raw_value in remaining.items():
            magnitude_lines.append(f"additional measured feature {name}={_format_value(raw_value, config)}.")
    magnitude_label = (
        "Photometric magnitudes, ordered by wavelength: "
        if physical_mode != "none" else "Magnitude columns: "
    )
    sections.append(magnitude_label + " ".join(magnitude_lines))

    if image_token_ids is not None:
        sections.append(serialize_tokenized_galaxy_image(image_token_ids, config=config))
    if config.final_marker:
        sections.append(config.final_marker)
    return "\n".join(sections)


def serialize_qwen3_batch(
    magnitude_features: torch.Tensor | np.ndarray,
    feature_names: Sequence[str],
    image_token_ids: torch.Tensor | np.ndarray | None = None,
    *,
    config: Qwen3SerializationConfig | None = None,
) -> list[str]:
    """Serialize a batch with magnitudes first and matched image-token grids second."""
    values = torch.as_tensor(magnitude_features)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("magnitude_features must have shape (batch, len(feature_names)).")
    tokens = None if image_token_ids is None else torch.as_tensor(image_token_ids)
    if tokens is not None and tokens.shape[0] != values.shape[0]:
        raise ValueError("image_token_ids and magnitude_features must have the same batch size.")
    return [
        serialize_qwen3_observation(
            {str(name): values[row, column] for column, name in enumerate(feature_names)},
            None if tokens is None else tokens[row],
            config=config,
        )
        for row in range(values.shape[0])
    ]


def qwen3_embedding_metadata(
    embedding_config: Qwen3EmbeddingConfig,
    serialization_config: Qwen3SerializationConfig | None = None,
) -> dict[str, Any]:
    serialization_config = serialization_config or Qwen3SerializationConfig()
    physical_mode = _physical_context_mode(serialization_config)
    return {
        **qwen_embedding_metadata(embedding_config),
        "qwen_serialization_schema": serialization_config.schema_name,
        "qwen_physical_band_context": physical_mode != "none",
        "qwen_physical_context_mode": physical_mode,
        "qwen_image_input": serialization_config.image_label,
        "qwen_image_context": bool(serialization_config.include_image_context),
        "qwen_image_grid_size": int(serialization_config.image_grid_size),
        "qwen_image_input_mode": serialization_config.image_input_mode,
        "qwen_image_crop_size": (
            int(serialization_config.image_crop_size)
            if serialization_config.image_input_mode == "center_crop" else None
        ),
        "qwen_serialized_image_grid_size": (
            int(serialization_config.image_crop_size)
            if serialization_config.image_input_mode == "center_crop"
            else int(serialization_config.image_grid_size)
        ),
        "qwen_final_marker": serialization_config.final_marker,
        "qwen_image_token_interpretation": "not predefined",
        "qwen_band_order": [band.canonical_name for band in serialization_config.band_descriptions],
    }


__all__ = [
    "CLAUDBandDescription",
    "CLAUDS_BAND_DESCRIPTIONS",
    "QWEN_IMAGE_INPUT_MODES",
    "QWEN_PHYSICAL_CONTEXT_MODES",
    "PHYSICAL_CONTEXT",
    "TOKEN_IMAGE_CONTEXT",
    "Qwen3SerializationConfig",
    "Qwen3EmbeddingConfig",
    "serialize_tokenized_galaxy_image",
    "serialize_qwen3_observation",
    "serialize_qwen3_batch",
    "qwen3_embedding_metadata",
    "QWEN_DEFAULT_MODELS",
    "QWEN_POOLING_MODES",
    "extract_qwen_embeddings_from_texts",
    "load_frozen_qwen",
    "pool_qwen_hidden_states",
    "require_transformers",
    "resolve_qwen_dtype",
    "resolve_qwen_model_path",
]
