from __future__ import annotations

"""Create a resumable, band-resolved CLAUDS/HSC morphology catalogue."""

import argparse
import hashlib
import json
import os
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from astropy.wcs import FITSFixedWarning, WCS
from astropy.wcs.utils import proj_plane_pixel_area

from .morphology import (
    AIONGalaxy10MorphologyHead,
    AION_IMAGE_INPUT_SIZE,
    DEFAULT_MORPHOLOGICAL_MISMATCH_THRESHOLD,
    GALAXY10_AION_CLASS_NAMES,
    collapse_galaxy10_morphology_probabilities,
    compute_pixel_morphology_batch,
    possible_morphological_mismatch,
)
from .morphology_catalogue import (
    DEFAULT_AION_MODEL,
    DEFAULT_BENCHMARK_REPO,
    FrozenAIONImageEncoder,
    _classification_accuracy,
    _fit_temperature,
    _iter_benchmark_batches,
    _open_feature_array,
    _stratified_train_validation_indices,
    download_galaxy10_aion_benchmark,
)
from .utils import resolve_torch_device, set_random_seed


MULTIBAND_MORPHOLOGY_BANDS = ("u", "g", "r", "i", "z", "y")
AION_HSC_BAND_BY_TARGET = {
    "u": "HSC-G",
    "g": "HSC-G",
    "r": "HSC-R",
    "i": "HSC-I",
    "z": "HSC-Z",
    "y": "HSC-Y",
}
GALAXY10_DES_BANDS = ("DES-G", "DES-R", "DES-I", "DES-Z")
MULTIBAND_FEATURE_STEMS = (
    "p_spiral",
    "p_bar",
    "p_elliptical_type",
    "axis_ellipticity",
    "concentration_C",
    "asymmetry_A",
    "possible_morphological_mismatch",
    "surface_brightness_24",
    "surface_brightness_96",
    "mean_per_sqarcsec_12",
    "mean_per_sqarcsec_24",
    "morphology_available",
)
FLOAT_FEATURE_STEMS = tuple(
    name
    for name in MULTIBAND_FEATURE_STEMS
    if name not in {"possible_morphological_mismatch", "morphology_available"}
)
FITS_BACKENDS = ("auto", "torchfits", "astropy")
STATUS_PENDING = np.uint8(0)
STATUS_SUCCESS = np.uint8(1)
STATUS_QUALITY_REJECTED = np.uint8(2)
UPDATED_CATALOGUE_ROOT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/clauds")

DEFAULT_INVALID_HSC_MASK_PLANES = (
    "BAD",
    "SAT",
    "INTRP",
    "CR",
    "EDGE",
    "SUSPECT",
    "NO_DATA",
    "BRIGHT_OBJECT",
    "CROSSTALK",
    "NOT_DEBLENDED",
    "UNMASKEDNAN",
    "REJECTED",
    "CLIPPED",
    "SENSOR_EDGE",
)


def multiband_column_names(
    bands: Sequence[str] = MULTIBAND_MORPHOLOGY_BANDS,
) -> tuple[str, ...]:
    return tuple(f"{stem}_{band}" for band in bands for stem in MULTIBAND_FEATURE_STEMS)


def parse_bands(value: str | Sequence[str]) -> tuple[str, ...]:
    raw = value if not isinstance(value, str) else value.replace(",", " ").split()
    bands = tuple(str(band).strip().lower() for band in raw if str(band).strip())
    if not bands:
        raise ValueError("At least one morphology band is required.")
    unknown = sorted(set(bands).difference(MULTIBAND_MORPHOLOGY_BANDS))
    if unknown:
        raise ValueError(f"Unsupported morphology bands: {unknown}")
    if len(set(bands)) != len(bands):
        raise ValueError("Morphology bands must not be repeated.")
    return tuple(band for band in MULTIBAND_MORPHOLOGY_BANDS if band in bands)


@dataclass
class MultibandMorphologyConfig:
    catalogue_path: Path = UPDATED_CATALOGUE_ROOT / "catalogs/COSMOS-HSCpipe-Phosphoros.fits"
    output_catalogue_path: Path = UPDATED_CATALOGUE_ROOT / (
        "catalogs/COSMOS-HSCpipe-Phosphoros_morphological_multiband_updated.fits"
    )
    u_image_dir: Path = UPDATED_CATALOGUE_ROOT / "images/tilesv5"
    hsc_image_dir: Path = Path("/arc/projects/ots/pdr3_dud")
    cache_dir: Path = Path("cache/aion_multiband_morphology_catalogue_updated")
    bands: tuple[str, ...] = MULTIBAND_MORPHOLOGY_BANDS
    benchmark_repo: str = DEFAULT_BENCHMARK_REPO
    aion_model: str = DEFAULT_AION_MODEL
    device: str = "auto"
    fits_backend: str = "auto"
    benchmark_batch_size: int = 32
    target_batch_size: int = 256
    head_epochs: int = 100
    head_learning_rate: float = 1.0e-3
    head_weight_decay: float = 1.0e-4
    head_patience: int = 30
    min_cutout_coverage: float = 0.90
    min_aperture_coverage: float = 0.90
    min_morphology_coverage: float = 0.98
    min_signal_to_noise: float = 5.0
    mismatch_threshold: float = DEFAULT_MORPHOLOGICAL_MISMATCH_THRESHOLD
    u_flux_scale: float = 1.0
    g_flux_scale: float = 1.0
    r_flux_scale: float = 1.0
    i_flux_scale: float = 1.0
    z_flux_scale: float = 1.0
    y_flux_scale: float = 1.0
    max_target_rows: int | None = None
    stop_after_processed_rows: int | None = None
    seed: int = 42
    overwrite_output: bool = False
    force_manifest: bool = False
    force_assignments: bool = False
    force_benchmark_embeddings: bool = False
    force_head_training: bool = False
    force_target_features: bool = False

    def normalized(self) -> "MultibandMorphologyConfig":
        values = asdict(self)
        for key in (
            "catalogue_path",
            "output_catalogue_path",
            "u_image_dir",
            "hsc_image_dir",
            "cache_dir",
        ):
            values[key] = Path(values[key])
        values["bands"] = parse_bands(values["bands"])
        values["fits_backend"] = str(values["fits_backend"]).lower()
        if values["fits_backend"] not in FITS_BACKENDS:
            raise ValueError(f"fits_backend must be one of {FITS_BACKENDS}.")
        if min(values["benchmark_batch_size"], values["target_batch_size"]) < 1:
            raise ValueError("Batch sizes must be positive.")
        if min(values["head_epochs"], values["head_patience"]) < 1:
            raise ValueError("Head epochs and patience must be positive.")
        for key in (
            "min_cutout_coverage",
            "min_aperture_coverage",
            "min_morphology_coverage",
        ):
            if not 0.0 <= float(values[key]) <= 1.0:
                raise ValueError(f"{key} must lie in [0, 1].")
        if float(values["min_signal_to_noise"]) < 0.0:
            raise ValueError("min_signal_to_noise must be non-negative.")
        for band in MULTIBAND_MORPHOLOGY_BANDS:
            scale = float(values[f"{band}_flux_scale"])
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"{band}_flux_scale must be finite and positive.")
        for key in ("max_target_rows", "stop_after_processed_rows"):
            if values[key] is not None and int(values[key]) < 1:
                raise ValueError(f"{key} must be positive or None.")
        return MultibandMorphologyConfig(**values)

    def flux_scale(self, band: str) -> float:
        if band not in MULTIBAND_MORPHOLOGY_BANDS:
            raise ValueError(f"Unsupported band: {band}")
        return float(getattr(self, f"{band}_flux_scale"))

    @property
    def manifest_path(self) -> Path:
        return self.cache_dir / "image_manifest.npz"

    @property
    def embedding_cache_path(self) -> Path:
        return self.cache_dir / "galaxy10_single_band_embeddings.npz"

    @property
    def head_path(self) -> Path:
        return self.cache_dir / "aion_galaxy10_single_band_head.pt"

    @property
    def feature_dir(self) -> Path:
        return self.cache_dir / "features"

    def assignment_path(self, band: str) -> Path:
        return self.cache_dir / "assignments" / f"{band}_assignment.npz"


@dataclass(frozen=True)
class ImageManifest:
    path: np.ndarray
    band: np.ndarray
    kind: np.ndarray
    bounds: np.ndarray
    height: np.ndarray
    width: np.ndarray
    pixel_area_arcsec2: np.ndarray
    fingerprint: str

    def select(self, band: str) -> np.ndarray:
        return np.flatnonzero(self.band == band)


def _json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _path_signature(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    for source in (Path(__file__), Path(__file__).with_name("morphology.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _manifest_fingerprint(
    paths: Sequence[str], bands: Sequence[str], kinds: Sequence[str]
) -> str:
    digest = hashlib.sha256()
    for image_path, band, kind in zip(paths, bands, kinds):
        signature = _path_signature(image_path)
        digest.update(
            json.dumps(
                {"band": band, "kind": kind, **signature},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    return digest.hexdigest()


def _science_config_payload(config: "MultibandMorphologyConfig") -> dict[str, Any]:
    return {
        "catalogue": _path_signature(config.catalogue_path),
        "u_image_dir": str(config.u_image_dir.resolve()),
        "hsc_image_dir": str(config.hsc_image_dir.resolve()),
        "bands": list(config.bands),
        "benchmark_repo": config.benchmark_repo,
        "aion_model": config.aion_model,
        "min_cutout_coverage": config.min_cutout_coverage,
        "min_aperture_coverage": config.min_aperture_coverage,
        "min_morphology_coverage": config.min_morphology_coverage,
        "min_signal_to_noise": config.min_signal_to_noise,
        "mismatch_threshold": config.mismatch_threshold,
        "flux_scales": {
            band: config.flux_scale(band) for band in MULTIBAND_MORPHOLOGY_BANDS
        },
        "code_fingerprint": _code_fingerprint(),
    }


def _wcs_bounds(wcs: WCS, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    height, width = shape
    x = np.asarray([0.0, width - 1.0, width - 1.0, 0.0])
    y = np.asarray([0.0, 0.0, height - 1.0, height - 1.0])
    ra, dec = wcs.all_pix2world(x, y, 0)
    return (
        float(np.nanmin(ra)),
        float(np.nanmax(ra)),
        float(np.nanmin(dec)),
        float(np.nanmax(dec)),
    )


def _pixel_area_arcsec2(wcs: WCS) -> float:
    area = proj_plane_pixel_area(wcs)
    value_deg2 = area.value if hasattr(area, "value") else area
    return float(abs(value_deg2) * 3600.0**2)


def _read_image_geometry(path: Path, kind: str) -> tuple[WCS, tuple[int, int]]:
    extension = 0 if kind == "clauds_u" else 1
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
        header = hdul[extension].header.copy()
        shape = tuple(int(value) for value in hdul[extension].shape)
    return WCS(header), shape


def _discover_image_records(
    config: MultibandMorphologyConfig,
) -> list[tuple[Path, str, str]]:
    records: list[tuple[Path, str, str]] = []
    if "u" in config.bands:
        u_paths = [
            path
            for path in sorted(config.u_image_dir.rglob("*.fits"))
            if not path.name.endswith(".weight.fits")
        ]
        records.extend((path, "u", "clauds_u") for path in u_paths)
    for band in config.bands:
        if band == "u":
            continue
        pattern = f"calexp-HSC-{band.upper()}-*.fits"
        records.extend(
            (path, band, "hsc_pdr3")
            for path in sorted(config.hsc_image_dir.glob(pattern))
        )
    missing = [band for band in config.bands if not any(row[1] == band for row in records)]
    if missing:
        raise FileNotFoundError(f"No image patches found for bands: {missing}")
    return records


def build_or_load_image_manifest(
    config: MultibandMorphologyConfig,
) -> ImageManifest:
    config = config.normalized()
    if config.manifest_path.exists() and not config.force_manifest:
        with np.load(config.manifest_path) as cached:
            required_provenance = {"u_image_dir", "hsc_image_dir"}
            missing_provenance = required_provenance.difference(cached.files)
            if missing_provenance:
                missing_text = ", ".join(sorted(missing_provenance))
                raise RuntimeError(
                    "Cached image manifest predates updated provenance "
                    f"({missing_text} missing); rerun with --force-manifest "
                    "and --force-assignments --force-target-features."
                )
            cached_manifest = ImageManifest(
                path=np.asarray(cached["path"]).astype(str),
                band=np.asarray(cached["band"]).astype(str),
                kind=np.asarray(cached["kind"]).astype(str),
                bounds=np.asarray(cached["bounds"], dtype=np.float64),
                height=np.asarray(cached["height"], dtype=np.int32),
                width=np.asarray(cached["width"], dtype=np.int32),
                pixel_area_arcsec2=np.asarray(
                    cached["pixel_area_arcsec2"], dtype=np.float64
                ),
                fingerprint=str(np.asarray(cached["fingerprint"]).item()),
            )
            cached_u_root = str(np.asarray(cached["u_image_dir"]).item())
            cached_hsc_root = str(np.asarray(cached["hsc_image_dir"]).item())
        roots_match = (
            cached_u_root == str(config.u_image_dir.resolve())
            and cached_hsc_root == str(config.hsc_image_dir.resolve())
        )
        current_fingerprint = _manifest_fingerprint(
            cached_manifest.path, cached_manifest.band, cached_manifest.kind
        )
        if (
            roots_match
            and current_fingerprint == cached_manifest.fingerprint
            and set(config.bands).issubset(set(cached_manifest.band))
        ):
            return cached_manifest
        raise RuntimeError(
            "Cached image manifest provenance changed; rerun with --force-manifest "
            "and --force-assignments --force-target-features."
        )

    records = _discover_image_records(config)
    paths: list[str] = []
    bands: list[str] = []
    kinds: list[str] = []
    bounds: list[tuple[float, float, float, float]] = []
    heights: list[int] = []
    widths: list[int] = []
    pixel_areas: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        for index, (path, band, kind) in enumerate(records):
            wcs, shape = _read_image_geometry(path, kind)
            paths.append(str(path.resolve()))
            bands.append(band)
            kinds.append(kind)
            bounds.append(_wcs_bounds(wcs, shape))
            heights.append(shape[0])
            widths.append(shape[1])
            pixel_areas.append(_pixel_area_arcsec2(wcs))
            if (index + 1) % 500 == 0 or index + 1 == len(records):
                print(f"Indexed image headers: {index + 1:,}/{len(records):,}")
    fingerprint = _manifest_fingerprint(paths, bands, kinds)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        config.manifest_path,
        path=np.asarray(paths),
        band=np.asarray(bands),
        kind=np.asarray(kinds),
        bounds=np.asarray(bounds, dtype=np.float64),
        height=np.asarray(heights, dtype=np.int32),
        width=np.asarray(widths, dtype=np.int32),
        pixel_area_arcsec2=np.asarray(pixel_areas, dtype=np.float64),
        fingerprint=np.asarray(fingerprint),
        u_image_dir=np.asarray(str(config.u_image_dir.resolve())),
        hsc_image_dir=np.asarray(str(config.hsc_image_dir.resolve())),
    )
    return build_or_load_image_manifest(
        MultibandMorphologyConfig(**{**asdict(config), "force_manifest": False})
    )


def _sigma_clipped_background(values: np.ndarray) -> tuple[float, float]:
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return np.nan, np.nan
    for _ in range(3):
        centre = float(np.median(sample))
        sigma = float(1.4826 * np.median(np.abs(sample - centre)))
        if not np.isfinite(sigma) or sigma <= 0.0:
            return centre, 0.0
        kept = np.abs(sample - centre) <= 3.0 * sigma
        if kept.all() or not kept.any():
            return centre, sigma
        sample = sample[kept]
    centre = float(np.median(sample))
    sigma = float(1.4826 * np.median(np.abs(sample - centre)))
    return centre, sigma


def _central_mask(size: int, aperture: int) -> np.ndarray:
    if aperture > size or aperture < 1:
        raise ValueError("Aperture must be positive and no larger than the cutout.")
    start = (size - aperture) // 2
    mask = np.zeros((size, size), dtype=bool)
    mask[start : start + aperture, start : start + aperture] = True
    return mask


def compute_brightness_features(
    raw_cutouts: np.ndarray,
    valid_masks: np.ndarray,
    backgrounds: np.ndarray,
    pixel_area_arcsec2: np.ndarray,
    *,
    min_aperture_coverage: float = 0.90,
) -> dict[str, np.ndarray]:
    """Calculate the four requested brightness measurements."""
    raw = np.asarray(raw_cutouts, dtype=np.float32)
    valid = np.asarray(valid_masks, dtype=bool)
    background = np.asarray(backgrounds, dtype=np.float64)
    area = np.asarray(pixel_area_arcsec2, dtype=np.float64)
    if raw.ndim != 3 or valid.shape != raw.shape:
        raise ValueError("raw_cutouts and valid_masks must share shape [batch, size, size].")
    if background.shape != (len(raw),) or area.shape != (len(raw),):
        raise ValueError("background and pixel area must have one value per cutout.")
    output = {
        name: np.full(len(raw), np.nan, dtype=np.float32)
        for name in (
            "surface_brightness_24",
            "surface_brightness_96",
            "mean_per_sqarcsec_12",
            "mean_per_sqarcsec_24",
        )
    }
    valid_all = np.isfinite(background) & np.isfinite(area) & (area > 0.0)
    for aperture in (12, 24, raw.shape[1]):
        aperture_mask = _central_mask(raw.shape[1], aperture)
        aperture_valid = valid & aperture_mask[None, :, :]
        counts = aperture_valid.sum(axis=(1, 2))
        coverage = counts / float(aperture * aperture)
        usable = valid_all & (coverage >= float(min_aperture_coverage)) & (counts > 0)
        raw_sum = np.where(aperture_valid, raw, 0.0).sum(
            axis=(1, 2), dtype=np.float64
        )
        raw_mean = np.divide(
            raw_sum,
            counts,
            out=np.full(len(raw), np.nan, dtype=np.float64),
            where=counts > 0,
        )
        if aperture in (24, raw.shape[1]):
            name = f"surface_brightness_{aperture}"
            values = raw_sum - background * counts
            output[name][usable] = values[usable].astype(np.float32)
        if aperture in (12, 24):
            name = f"mean_per_sqarcsec_{aperture}"
            values = raw_mean / area
            output[name][usable] = values[usable].astype(np.float32)
    output["brightness_valid"] = np.logical_and.reduce(
        [np.isfinite(output[name]) for name in output]
    )
    return output


def hsc_invalid_mask_bits(
    header: fits.Header,
    names: Sequence[str] = DEFAULT_INVALID_HSC_MASK_PLANES,
) -> int:
    bits = 0
    for name in names:
        key = f"MP_{name}"
        if key in header:
            bits |= 1 << int(header[key])
    return bits


class BandImageTile:
    """Per-patch CLAUDS/HSC science and quality access with subset I/O."""

    def __init__(
        self,
        path: str | Path,
        kind: str,
        *,
        flux_scale: float = 1.0,
        fits_backend: str = "auto",
    ):
        self.path = Path(path)
        self.kind = str(kind)
        self.flux_scale = float(flux_scale)
        self.requested_fits_backend = str(fits_backend).lower()
        self.fits_backend = self._resolve_backend(self.requested_fits_backend)
        self._science_hdul = None
        self._weight_hdul = None
        self._subset_managers: list[Any] = []
        self._subset_readers: list[Any] = []
        if self.kind == "clauds_u":
            self.weight_path = self.path.with_name(f"{self.path.stem}.weight.fits")
            if not self.weight_path.exists():
                raise FileNotFoundError(f"Missing CLAUDS weight map: {self.weight_path}")
            with fits.open(self.path, memmap=True, lazy_load_hdus=True) as hdul:
                self.header = hdul[0].header.copy()
                self.shape = tuple(int(value) for value in hdul[0].shape)
            self.invalid_mask_bits = 0
        elif self.kind == "hsc_pdr3":
            self.weight_path = None
            with fits.open(self.path, memmap=True, lazy_load_hdus=True) as hdul:
                self.header = hdul[1].header.copy()
                self.shape = tuple(int(value) for value in hdul[1].shape)
                self.invalid_mask_bits = hsc_invalid_mask_bits(hdul[2].header)
        else:
            raise ValueError(f"Unknown image tile kind: {self.kind}")
        self.wcs = WCS(self.header)
        self.pixel_area_arcsec2 = _pixel_area_arcsec2(self.wcs)

    @staticmethod
    def _resolve_backend(requested: str) -> str:
        if requested == "astropy":
            return "astropy"
        try:
            import torchfits  # noqa: F401
        except (ImportError, OSError) as error:
            if requested == "torchfits":
                raise RuntimeError(
                    "--fits-backend torchfits was requested but torchfits could not load."
                ) from error
            return "astropy"
        return "torchfits"

    def _open(self) -> None:
        if self.fits_backend == "torchfits":
            if self._subset_readers:
                return
            import torchfits

            targets = (
                ((self.path, 0), (self.weight_path, 0))
                if self.kind == "clauds_u"
                else ((self.path, 1), (self.path, 2), (self.path, 3))
            )
            try:
                for image_path, hdu in targets:
                    manager = torchfits.open_subset_reader(
                        str(image_path), hdu=hdu, device="cpu"
                    )
                    self._subset_managers.append(manager)
                    self._subset_readers.append(manager.__enter__())
            except BaseException:
                self.close()
                raise
            return
        if self._science_hdul is None:
            self._science_hdul = fits.open(self.path, memmap=True)
        if self.kind == "clauds_u" and self._weight_hdul is None:
            self._weight_hdul = fits.open(self.weight_path, memmap=True)

    def _read_subset(
        self, plane: int, x0: int, y0: int, x1: int, y1: int
    ) -> np.ndarray:
        self._open()
        if self.fits_backend == "torchfits":
            tensor = self._subset_readers[plane](x0, y0, x1, y1)
            return tensor.detach().cpu().numpy()
        if self.kind == "clauds_u":
            data = (
                self._science_hdul[0].data
                if plane == 0
                else self._weight_hdul[0].data
            )
        else:
            data = self._science_hdul[plane + 1].data
        return np.asarray(data[y0:y1, x0:x1])

    def extract(
        self, x: float, y: float, size: int = AION_IMAGE_INPUT_SIZE
    ) -> dict[str, Any]:
        height, width = self.shape
        half = size // 2
        cx, cy = int(np.rint(x)), int(np.rint(y))
        x0, y0 = cx - half, cy - half
        x1, y1 = x0 + size, y0 + size
        raw = np.zeros((size, size), dtype=np.float32)
        valid = np.zeros((size, size), dtype=bool)
        noise_variance = np.full((size, size), np.nan, dtype=np.float64)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(width, x1), min(height, y1)
        if sx1 > sx0 and sy1 > sy0:
            dx0, dy0 = sx0 - x0, sy0 - y0
            dx1, dy1 = dx0 + sx1 - sx0, dy0 + sy1 - sy0
            source = np.asarray(
                self._read_subset(0, sx0, sy0, sx1, sy1), dtype=np.float32
            )
            if self.kind == "clauds_u":
                weight = np.asarray(
                    self._read_subset(1, sx0, sy0, sx1, sy1), dtype=np.float64
                )
                source_valid = np.isfinite(source) & np.isfinite(weight) & (weight > 0.0)
                source_variance = np.divide(
                    self.flux_scale**2,
                    weight,
                    out=np.full_like(weight, np.nan, dtype=np.float64),
                    where=source_valid,
                )
            else:
                mask = np.asarray(
                    self._read_subset(1, sx0, sy0, sx1, sy1), dtype=np.int64
                )
                variance_cutout = np.asarray(
                    self._read_subset(2, sx0, sy0, sx1, sy1), dtype=np.float64
                )
                source_valid = (
                    np.isfinite(source)
                    & np.isfinite(variance_cutout)
                    & (variance_cutout > 0.0)
                    & ((mask & self.invalid_mask_bits) == 0)
                )
                source_variance = variance_cutout * self.flux_scale**2
            raw_view = raw[dy0:dy1, dx0:dx1]
            raw_view[source_valid] = source[source_valid] * self.flux_scale
            valid[dy0:dy1, dx0:dx1] = source_valid
            variance_view = noise_variance[dy0:dy1, dx0:dx1]
            variance_view[source_valid] = source_variance[source_valid]
        yy, xx = np.mgrid[:size, :size]
        radius = np.hypot(xx - (size - 1) / 2.0, yy - (size - 1) / 2.0)
        border = valid & (radius >= size / 2.0 - 6.0)
        background, sigma = _sigma_clipped_background(raw[border])
        background_subtracted = np.zeros_like(raw)
        if np.isfinite(background):
            background_subtracted[valid] = raw[valid] - background
        return {
            "raw": raw,
            "valid": valid,
            "variance": noise_variance,
            "background": background,
            "background_sigma": sigma,
            "background_subtracted": background_subtracted,
            "coverage": float(valid.mean()),
            "pixel_area_arcsec2": self.pixel_area_arcsec2,
            "fits_backend": self.fits_backend,
        }

    def close(self) -> None:
        while self._subset_managers:
            manager = self._subset_managers.pop()
            manager.__exit__(None, None, None)
        self._subset_readers.clear()
        if self._science_hdul is not None:
            self._science_hdul.close()
            self._science_hdul = None
        if self._weight_hdul is not None:
            self._weight_hdul.close()
            self._weight_hdul = None



def _embedding_fingerprint(config: MultibandMorphologyConfig) -> str:
    return _json_digest(
        {
            "benchmark_repo": config.benchmark_repo,
            "aion_model": config.aion_model,
            "training_bands": GALAXY10_DES_BANDS,
            "code_fingerprint": _code_fingerprint(),
        }
    )


def _head_training_fingerprint(config: MultibandMorphologyConfig) -> str:
    return _json_digest(
        {
            "embedding_fingerprint": _embedding_fingerprint(config),
            "epochs": config.head_epochs,
            "learning_rate": config.head_learning_rate,
            "weight_decay": config.head_weight_decay,
            "patience": config.head_patience,
            "seed": config.seed,
            "split": "grouped-fit-80-selection-10-calibration-10",
        }
    )


class SingleBandAIONEncoder(FrozenAIONImageEncoder):
    """Encode one observed band through the corresponding AION image channel."""

    @torch.inference_mode()
    def encode_legacy_band(self, images: np.ndarray, band: str) -> torch.Tensor:
        flux = torch.from_numpy(np.asarray(images, dtype=np.float32)).to(self.device)
        if flux.ndim == 3:
            flux = flux[:, None, :, :]
        if flux.ndim != 4 or flux.shape[1] != 1:
            raise ValueError("Single-band Legacy images must have shape [batch, 1, 96, 96].")
        modality = self.LegacySurveyImage(flux=flux, bands=[band])
        tokens = self.codec_manager.encode(modality)
        key = self.LegacySurveyImage.token_key
        return self._mean_embedding({key: tokens[key]})

    @torch.inference_mode()
    def encode_hsc_band(self, images: np.ndarray, band: str) -> torch.Tensor:
        flux = torch.from_numpy(np.asarray(images, dtype=np.float32)).to(self.device)
        if flux.ndim == 3:
            flux = flux[:, None, :, :]
        if flux.ndim != 4 or flux.shape[1] != 1:
            raise ValueError("Single-band HSC images must have shape [batch, 1, 96, 96].")
        modality = self.HSCImage(flux=flux, bands=[band])
        tokens = self.codec_manager.encode(modality)
        key = self.HSCImage.token_key
        return self._mean_embedding({key: tokens[key]})


def cache_single_band_benchmark_embeddings(
    config: MultibandMorphologyConfig,
    *,
    encoder: SingleBandAIONEncoder | None = None,
) -> dict[str, np.ndarray]:
    """Cache Galaxy10 embeddings with exactly one DES channel present at a time."""
    config = config.normalized()
    expected_fingerprint = _embedding_fingerprint(config)
    if config.embedding_cache_path.exists() and not config.force_benchmark_embeddings:
        with np.load(config.embedding_cache_path) as cached:
            stored_fingerprint = str(np.asarray(cached["_provenance"]).item())
            if stored_fingerprint != expected_fingerprint:
                raise RuntimeError(
                    "Cached benchmark embeddings have stale provenance; rerun with "
                    "--force-benchmark-embeddings --force-head-training "
                    "--force-target-features."
                )
            return {
                name: np.asarray(cached[name])
                for name in cached.files
                if not name.startswith("_")
            }

    snapshot = download_galaxy10_aion_benchmark(config)  # type: ignore[arg-type]
    split_paths = {
        split: sorted((snapshot / "data").glob(f"{split}-*.parquet"))
        for split in ("train", "test")
    }
    if any(not paths for paths in split_paths.values()):
        raise FileNotFoundError(f"Galaxy10 benchmark shards were not found under {snapshot}.")
    device = resolve_torch_device(config.device)
    encoder = encoder or SingleBandAIONEncoder(config.aion_model, device)
    product: dict[str, np.ndarray] = {}
    for split, paths in split_paths.items():
        embeddings: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        groups: list[np.ndarray] = []
        band_indices: list[np.ndarray] = []
        source_offset = 0
        started = time.time()
        for images, batch_labels in _iter_benchmark_batches(
            paths, batch_size=config.benchmark_batch_size
        ):
            n_batch = len(batch_labels)
            source_groups = np.arange(source_offset, source_offset + n_batch, dtype=np.int64)
            for band_index, band in enumerate(GALAXY10_DES_BANDS):
                embeddings.append(
                    encoder.encode_legacy_band(images[:, band_index], band).cpu().numpy()
                )
                labels.append(batch_labels.copy())
                groups.append(source_groups.copy())
                band_indices.append(np.full(n_batch, band_index, dtype=np.uint8))
            source_offset += n_batch
            if source_offset % 500 < n_batch:
                rate = source_offset / max(time.time() - started, 1.0e-6)
                print(f"Single-band AION benchmark {split}: {source_offset:,} galaxies ({rate:.1f}/s)")
        product[f"{split}_embeddings"] = np.concatenate(embeddings).astype(np.float32)
        product[f"{split}_labels"] = np.concatenate(labels).astype(np.int64)
        product[f"{split}_group"] = np.concatenate(groups).astype(np.int64)
        product[f"{split}_band_index"] = np.concatenate(band_indices).astype(np.uint8)

    config.embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        config.embedding_cache_path,
        **product,
        _provenance=np.asarray(expected_fingerprint),
        _snapshot_revision=np.asarray(snapshot.name),
    )
    return product


def _group_stratified_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split at source-galaxy level so its four band views cannot leak."""
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    unique_groups, first = np.unique(groups, return_index=True)
    group_train, group_validation = _stratified_train_validation_indices(
        labels[first], validation_fraction=validation_fraction, seed=seed
    )
    train_groups = unique_groups[group_train]
    validation_groups = unique_groups[group_validation]
    return np.flatnonzero(np.isin(groups, train_groups)), np.flatnonzero(
        np.isin(groups, validation_groups)
    )


def _fit_head_epochs(
    head: AIONGalaxy10MorphologyHead,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    config: MultibandMorphologyConfig,
    device: torch.device,
) -> None:
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.head_learning_rate,
        weight_decay=config.head_weight_decay,
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(embeddings, labels),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    for _ in range(epochs):
        head.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(batch_x), batch_y)
            loss.backward()
            optimizer.step()


def train_single_band_morphology_head(
    config: MultibandMorphologyConfig,
    *,
    embedding_product: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Train and calibrate a Galaxy10 head on band-isolated AION embeddings."""
    config = config.normalized()
    expected_fingerprint = _head_training_fingerprint(config)
    if config.head_path.exists() and not config.force_head_training:
        checkpoint = torch.load(config.head_path, map_location="cpu", weights_only=False)
        if checkpoint.get("training_fingerprint") != expected_fingerprint:
            raise RuntimeError(
                "Cached morphology head has stale provenance; rerun with "
                "--force-head-training --force-target-features."
            )
        return checkpoint
    set_random_seed(config.seed)
    product = embedding_product or cache_single_band_benchmark_embeddings(config)
    train_x = torch.from_numpy(product["train_embeddings"]).float()
    train_y = torch.from_numpy(product["train_labels"]).long()
    test_x = torch.from_numpy(product["test_embeddings"]).float()
    test_y = torch.from_numpy(product["test_labels"]).long()
    fit_indices, holdout_indices = _group_stratified_indices(
        product["train_labels"],
        product["train_group"],
        validation_fraction=0.2,
        seed=config.seed,
    )
    selection_local, calibration_local = _group_stratified_indices(
        product["train_labels"][holdout_indices],
        product["train_group"][holdout_indices],
        validation_fraction=0.5,
        seed=config.seed + 1,
    )
    selection_indices = holdout_indices[selection_local]
    calibration_indices = holdout_indices[calibration_local]
    device = resolve_torch_device(config.device)
    head = AIONGalaxy10MorphologyHead(input_dim=train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.head_learning_rate,
        weight_decay=config.head_weight_decay,
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x[fit_indices], train_y[fit_indices]),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    validation_x = train_x[selection_indices].to(device)
    validation_y = train_y[selection_indices].to(device)
    calibration_x = train_x[calibration_indices].to(device)
    calibration_y = train_y[calibration_indices].to(device)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(config.head_epochs):
        head.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
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
            f"Single-band Galaxy10 head epoch {epoch + 1:03d}: "
            f"val_loss={validation_loss:.4f} val_accuracy={validation_accuracy:.4f}"
        )
        if validation_loss < best_loss - 1.0e-5:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= config.head_patience:
                break
    if best_state is None:
        raise RuntimeError("Single-band morphology-head training produced no checkpoint.")
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        calibration_logits = head(calibration_x)
    temperature = _fit_temperature(calibration_logits, calibration_y)
    head.temperature.fill_(temperature)
    with torch.inference_mode():
        test_logits = head(test_x.to(device))
        test_accuracy = _classification_accuracy(test_logits, test_y.to(device))
        test_nll = float(
            F.cross_entropy(test_logits / head.temperature, test_y.to(device)).item()
        )
        test_band_accuracy = {
            band: _classification_accuracy(
                test_logits[torch.from_numpy(product["test_band_index"] == index).to(device)],
                test_y[torch.from_numpy(product["test_band_index"] == index)].to(device),
            )
            for index, band in enumerate(GALAXY10_DES_BANDS)
        }
    checkpoint = {
        "state_dict": {
            name: value.detach().cpu() for name, value in head.state_dict().items()
        },
        "input_dim": int(train_x.shape[1]),
        "hidden_dim": 256,
        "class_names": GALAXY10_AION_CLASS_NAMES,
        "benchmark_repo": config.benchmark_repo,
        "aion_model": config.aion_model,
        "single_band_training": True,
        "training_bands": GALAXY10_DES_BANDS,
        "temperature": temperature,
        "selected_epoch": best_epoch,
        "validation_loss": best_loss,
        "test_accuracy": test_accuracy,
        "test_nll": test_nll,
        "test_band_accuracy": test_band_accuracy,
        "seed": config.seed,
        "training_fingerprint": expected_fingerprint,
        "calibration_scope": "held-out DES Galaxy10 only; target CLAUDS/HSC uncalibrated",
        "n_fit": int(len(fit_indices)),
        "n_selection": int(len(selection_indices)),
        "n_calibration": int(len(calibration_indices)),
    }
    config.head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, config.head_path)
    print(
        f"Saved single-band AION morphology head: {config.head_path} "
        f"(test accuracy={test_accuracy:.4f}, temperature={temperature:.3f})"
    )
    return checkpoint


def load_single_band_morphology_head(
    path: str | Path,
    *,
    device: torch.device,
    expected_fingerprint: str | None = None,
) -> tuple[AIONGalaxy10MorphologyHead, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not checkpoint.get("single_band_training", False):
        raise RuntimeError("Checkpoint was not trained on band-isolated AION inputs.")
    if (
        expected_fingerprint is not None
        and checkpoint.get("training_fingerprint") != expected_fingerprint
    ):
        raise RuntimeError("Morphology-head provenance does not match this run.")
    head = AIONGalaxy10MorphologyHead(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
        temperature=float(checkpoint.get("temperature", 1.0)),
    )
    head.load_state_dict(checkpoint["state_dict"])
    return head.to(device).eval(), checkpoint


def _read_catalogue(config: MultibandMorphologyConfig) -> dict[str, np.ndarray]:
    import fitsio

    values = fitsio.read(
        config.catalogue_path, ext=1, columns=("ID", "RA", "DEC", "isStar")
    )
    stop = len(values) if config.max_target_rows is None else config.max_target_rows
    catalogue = {
        "id": np.asarray(values["ID"][:stop]),
        "ra": np.asarray(values["RA"][:stop], dtype=np.float64),
        "dec": np.asarray(values["DEC"][:stop], dtype=np.float64),
        "is_star": np.asarray(values["isStar"][:stop], dtype=bool),
        "catalogue_n_rows": np.asarray([len(values)], dtype=np.int64),
    }
    digest = hashlib.sha256(
        json.dumps(_path_signature(config.catalogue_path), sort_keys=True).encode()
    )
    for name in ("id", "ra", "dec", "is_star"):
        array = np.ascontiguousarray(catalogue[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.view(np.uint8))
    catalogue["fingerprint"] = np.asarray(digest.hexdigest())
    return catalogue


def _assign_band_to_catalogue(
    manifest: ImageManifest,
    band: str,
    catalogue: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra = np.asarray(catalogue["ra"], dtype=np.float64)
    dec = np.asarray(catalogue["dec"], dtype=np.float64)
    eligible = ~np.asarray(catalogue["is_star"], dtype=bool)
    order = np.argsort(ra, kind="stable")
    sorted_ra = ra[order]
    tile_index = np.full(len(ra), -1, dtype=np.int32)
    x_out = np.full(len(ra), np.nan, dtype=np.float32)
    y_out = np.full(len(ra), np.nan, dtype=np.float32)
    best_margin = np.full(len(ra), -np.inf, dtype=np.float32)
    for manifest_index in manifest.select(band):
        bounds = manifest.bounds[manifest_index]
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
            wcs, shape = _read_image_geometry(
                Path(manifest.path[manifest_index]), str(manifest.kind[manifest_index])
            )
            x, y = wcs.all_world2pix(ra[candidates], dec[candidates], 0)
        except Exception as error:
            warnings.warn(f"Skipping WCS assignment for {manifest.path[manifest_index]}: {error}")
            continue
        x, y = np.asarray(x), np.asarray(y)
        height, width = shape
        inside = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= 0.0)
            & (x < width)
            & (y >= 0.0)
            & (y < height)
        )
        rows = candidates[inside]
        if not len(rows):
            continue
        x, y = x[inside], y[inside]
        margin = np.minimum.reduce((x, y, width - 1.0 - x, height - 1.0 - y))
        update = margin > best_margin[rows]
        rows = rows[update]
        tile_index[rows] = manifest_index
        x_out[rows] = x[update].astype(np.float32)
        y_out[rows] = y[update].astype(np.float32)
        best_margin[rows] = margin[update].astype(np.float32)
    return tile_index, x_out, y_out


def build_or_load_band_assignment(
    config: MultibandMorphologyConfig,
    manifest: ImageManifest,
    band: str,
    catalogue: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = config.assignment_path(band)
    if path.exists() and not config.force_assignments:
        with np.load(path) as cached:
            compatible = (
                str(np.asarray(cached["manifest_fingerprint"]).item()) == manifest.fingerprint
                and str(np.asarray(cached["catalogue_fingerprint"]).item())
                == str(np.asarray(catalogue["fingerprint"]).item())
                and np.array_equal(cached["object_id"], catalogue["id"])
            )
            if compatible:
                return (
                    np.asarray(cached["tile_index"], dtype=np.int32),
                    np.asarray(cached["x_image"], dtype=np.float32),
                    np.asarray(cached["y_image"], dtype=np.float32),
                )
    print(f"Assigning {len(catalogue['id']):,} catalogue rows in {band} band")
    assigned = _assign_band_to_catalogue(manifest, band, catalogue)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        object_id=np.asarray(catalogue["id"]),
        manifest_fingerprint=np.asarray(manifest.fingerprint),
        catalogue_fingerprint=np.asarray(catalogue["fingerprint"]),
        tile_index=assigned[0],
        x_image=assigned[1],
        y_image=assigned[2],
    )
    return assigned



def _prepare_feature_cache(
    config: MultibandMorphologyConfig,
    *,
    catalogue_fingerprint: str,
    manifest_fingerprint: str,
) -> str:
    payload = {
        **_science_config_payload(config),
        "catalogue_fingerprint": catalogue_fingerprint,
        "manifest_fingerprint": manifest_fingerprint,
        "head_fingerprint": _file_sha256(config.head_path),
        "head_training_fingerprint": _head_training_fingerprint(config),
        "fits_backend": BandImageTile._resolve_backend(config.fits_backend),
        "status_schema": {
            "pending": int(STATUS_PENDING),
            "success": int(STATUS_SUCCESS),
            "quality_rejected": int(STATUS_QUALITY_REJECTED),
        },
    }
    fingerprint = _json_digest(payload)
    provenance_path = config.feature_dir / "provenance.json"
    existing_arrays = list(config.feature_dir.glob("*/*.npy"))
    if provenance_path.exists() and not config.force_target_features:
        stored = json.loads(provenance_path.read_text())
        if stored.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "Feature cache provenance changed; use a new cache directory or "
                "rerun with --force-target-features."
            )
    elif existing_arrays and not config.force_target_features:
        raise RuntimeError(
            "Feature arrays exist without provenance; use a new cache directory or "
            "rerun with --force-target-features."
        )
    config.feature_dir.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps({"fingerprint": fingerprint, "payload": payload}, indent=2) + "\n"
    )
    return fingerprint


def open_band_feature_arrays(
    config: MultibandMorphologyConfig, band: str, n_rows: int
) -> dict[str, np.memmap]:
    directory = config.feature_dir / band
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        stem: _open_feature_array(
            directory / f"{stem}.npy",
            dtype=np.float32,
            n_rows=n_rows,
            fill=np.nan,
            force=config.force_target_features,
        )
        for stem in FLOAT_FEATURE_STEMS
    }
    for stem in ("possible_morphological_mismatch", "morphology_available"):
        arrays[stem] = _open_feature_array(
            directory / f"{stem}.npy",
            dtype=np.bool_,
            n_rows=n_rows,
            fill=False,
            force=config.force_target_features,
        )
    for diagnostic in (
        "cutout_coverage",
        "signal_coverage",
        "asymmetry_pair_coverage",
    ):
        arrays[diagnostic] = _open_feature_array(
            directory / f"{diagnostic}.npy",
            dtype=np.float32,
            n_rows=n_rows,
            fill=np.nan,
            force=config.force_target_features,
        )
    for diagnostic in ("pixel_valid", "brightness_valid", "probability_valid"):
        arrays[diagnostic] = _open_feature_array(
            directory / f"{diagnostic}.npy",
            dtype=np.bool_,
            n_rows=n_rows,
            fill=False,
            force=config.force_target_features,
        )
    arrays["status"] = _open_feature_array(
        directory / "status.npy",
        dtype=np.uint8,
        n_rows=n_rows,
        fill=STATUS_PENDING,
        force=config.force_target_features,
    )
    return arrays


def _flush_feature_arrays(arrays: Mapping[str, np.memmap]) -> None:
    for values in arrays.values():
        values.flush()


@torch.inference_mode()
def _store_cutout_batch(
    *,
    rows: list[int],
    cutouts: list[dict[str, Any]],
    arrays: Mapping[str, np.memmap],
    band: str,
    config: MultibandMorphologyConfig,
    encoder: SingleBandAIONEncoder,
    head: AIONGalaxy10MorphologyHead,
) -> int:
    if not rows:
        return 0
    row_array = np.asarray(rows, dtype=np.int64)
    for stem in FLOAT_FEATURE_STEMS:
        arrays[stem][row_array] = np.nan
    arrays["possible_morphological_mismatch"][row_array] = False
    arrays["morphology_available"][row_array] = False
    arrays["probability_valid"][row_array] = False
    arrays["status"][row_array] = STATUS_PENDING

    raw = np.stack([item["raw"] for item in cutouts]).astype(np.float32, copy=False)
    valid = np.stack([item["valid"] for item in cutouts]).astype(bool, copy=False)
    variance = np.stack([item["variance"] for item in cutouts]).astype(
        np.float64, copy=False
    )
    background = np.asarray([item["background"] for item in cutouts], dtype=np.float64)
    pixel_area = np.asarray(
        [item["pixel_area_arcsec2"] for item in cutouts], dtype=np.float64
    )
    background_subtracted = np.stack(
        [item["background_subtracted"] for item in cutouts]
    ).astype(np.float32, copy=False)
    coverage = np.asarray([item["coverage"] for item in cutouts], dtype=np.float32)

    brightness = compute_brightness_features(
        raw,
        valid,
        background,
        pixel_area,
        min_aperture_coverage=config.min_aperture_coverage,
    )
    measured = compute_pixel_morphology_batch(
        background_subtracted,
        valid_masks=valid,
        variance_cutouts=variance,
        min_signal_to_noise=config.min_signal_to_noise,
        min_valid_fraction=config.min_morphology_coverage,
    )
    pixel_valid = measured["morphology_pixel_valid"]
    output_valid = (
        (coverage >= config.min_cutout_coverage)
        & brightness["brightness_valid"]
        & pixel_valid
    )
    arrays["cutout_coverage"][row_array] = coverage
    arrays["signal_coverage"][row_array] = measured["signal_coverage"]
    arrays["asymmetry_pair_coverage"][row_array] = measured[
        "asymmetry_pair_coverage"
    ]
    arrays["pixel_valid"][row_array] = pixel_valid
    arrays["brightness_valid"][row_array] = brightness["brightness_valid"]

    success_local = np.zeros(len(row_array), dtype=bool)
    collapsed_numpy: dict[str, np.ndarray] = {}
    if output_valid.any():
        candidate_local = np.flatnonzero(output_valid)
        embeddings = encoder.encode_hsc_band(
            background_subtracted[output_valid], AION_HSC_BAND_BY_TARGET[band]
        )
        probabilities = head.predict_proba(embeddings).float().cpu().numpy()
        probability_sums = probabilities.sum(axis=1, dtype=np.float64)
        probability_valid = (
            np.isfinite(probabilities).all(axis=1)
            & (probabilities >= -1.0e-7).all(axis=1)
            & (probabilities <= 1.0 + 1.0e-7).all(axis=1)
            & np.isclose(probability_sums, 1.0, rtol=1.0e-5, atol=1.0e-6)
        )
        collapsed = collapse_galaxy10_morphology_probabilities(probabilities)
        collapsed_numpy = {
            stem: np.asarray(collapsed[stem], dtype=np.float32)
            for stem in ("p_spiral", "p_bar", "p_elliptical_type")
        }
        probability_valid &= np.logical_and.reduce(
            [
                np.isfinite(collapsed_numpy[stem])
                & (collapsed_numpy[stem] >= 0.0)
                & (collapsed_numpy[stem] <= 1.0)
                for stem in collapsed_numpy
            ]
        )
        success_local[candidate_local[probability_valid]] = True
        arrays["probability_valid"][row_array[candidate_local]] = probability_valid

        successful_candidates = np.flatnonzero(probability_valid)
        successful_local = candidate_local[successful_candidates]
        successful_rows = row_array[successful_local]
        for stem in (
            "surface_brightness_24",
            "surface_brightness_96",
            "mean_per_sqarcsec_12",
            "mean_per_sqarcsec_24",
        ):
            arrays[stem][successful_rows] = brightness[stem][successful_local]
        for stem in ("axis_ellipticity", "concentration_C", "asymmetry_A"):
            arrays[stem][successful_rows] = measured[stem][successful_local]
        for stem in ("p_spiral", "p_bar", "p_elliptical_type"):
            arrays[stem][successful_rows] = collapsed_numpy[stem][successful_candidates]
        arrays["possible_morphological_mismatch"][successful_rows] = (
            possible_morphological_mismatch(
                arrays["p_elliptical_type"][successful_rows],
                arrays["axis_ellipticity"][successful_rows],
                threshold=config.mismatch_threshold,
            )
        )
        arrays["morphology_available"][successful_rows] = True

    arrays["status"][row_array] = STATUS_QUALITY_REJECTED
    arrays["status"][row_array[success_local]] = STATUS_SUCCESS
    rows.clear()
    cutouts.clear()
    return len(row_array)


def compute_multiband_morphology_features(
    config: MultibandMorphologyConfig,
    *,
    encoder: SingleBandAIONEncoder | None = None,
) -> dict[str, Any]:
    """Compute selected band-resolved features into resumable per-band memmaps."""
    config = config.normalized()
    catalogue = _read_catalogue(config)
    manifest = build_or_load_image_manifest(config)
    device = resolve_torch_device(config.device)
    encoder = encoder or SingleBandAIONEncoder(config.aion_model, device)
    head, checkpoint = load_single_band_morphology_head(
        config.head_path,
        device=device,
        expected_fingerprint=_head_training_fingerprint(config),
    )
    resolved_fits_backend = BandImageTile._resolve_backend(config.fits_backend)
    feature_fingerprint = _prepare_feature_cache(
        config,
        catalogue_fingerprint=str(np.asarray(catalogue["fingerprint"]).item()),
        manifest_fingerprint=manifest.fingerprint,
    )
    arrays_by_band: dict[str, dict[str, np.memmap]] = {}
    band_metadata: dict[str, dict[str, Any]] = {}
    remaining_budget = config.stop_after_processed_rows
    started = time.time()

    for band in config.bands:
        arrays = open_band_feature_arrays(config, band, len(catalogue["id"]))
        arrays_by_band[band] = arrays
        tile_index, x_image, y_image = build_or_load_band_assignment(
            config, manifest, band, catalogue
        )
        completed_before = int(np.count_nonzero(arrays["status"]))
        processed_now = 0
        for current_tile in manifest.select(band):
            if remaining_budget is not None and remaining_budget <= 0:
                break
            rows = np.flatnonzero(
                (tile_index == current_tile)
                & (np.asarray(arrays["status"]) == STATUS_PENDING)
            )
            if remaining_budget is not None:
                rows = rows[:remaining_budget]
            if not len(rows):
                continue
            flux_scale = config.flux_scale(band)
            tile = BandImageTile(
                manifest.path[current_tile],
                manifest.kind[current_tile],
                flux_scale=flux_scale,
                fits_backend=resolved_fits_backend,
            )
            batch_rows: list[int] = []
            batch_cutouts: list[dict[str, Any]] = []
            try:
                for row in rows:
                    batch_rows.append(int(row))
                    batch_cutouts.append(
                        tile.extract(float(x_image[row]), float(y_image[row]))
                    )
                    if len(batch_rows) >= config.target_batch_size:
                        n_done = _store_cutout_batch(
                            rows=batch_rows,
                            cutouts=batch_cutouts,
                            arrays=arrays,
                            band=band,
                            config=config,
                            encoder=encoder,
                            head=head,
                        )
                        processed_now += n_done
                        if remaining_budget is not None:
                            remaining_budget -= n_done
                n_done = _store_cutout_batch(
                    rows=batch_rows,
                    cutouts=batch_cutouts,
                    arrays=arrays,
                    band=band,
                    config=config,
                    encoder=encoder,
                    head=head,
                )
                processed_now += n_done
                if remaining_budget is not None:
                    remaining_budget -= n_done
            finally:
                tile.close()
            _flush_feature_arrays(arrays)
            elapsed = max(time.time() - started, 1.0e-6)
            print(
                f"{band}-band tile {current_tile}: processed_now={processed_now:,}, "
                f"available={int(np.count_nonzero(arrays['morphology_available'])):,}, "
                f"rate={processed_now / elapsed:.1f}/s"
            )
        _flush_feature_arrays(arrays)
        assigned = tile_index >= 0
        available = np.asarray(arrays["morphology_available"], dtype=bool)
        band_metadata[band] = {
            "aion_channel": AION_HSC_BAND_BY_TARGET[band],
            "n_assigned": int(np.count_nonzero(assigned)),
            "n_processed": int(np.count_nonzero(arrays["status"])),
            "n_processed_this_run": processed_now,
            "n_valid_96_coverage": int(
                np.count_nonzero(
                    np.asarray(arrays["cutout_coverage"]) >= config.min_cutout_coverage
                )
            ),
            "n_rejected_mask_or_variance": int(
                np.count_nonzero(
                    (np.asarray(arrays["status"]) != 0)
                    & (
                        np.asarray(arrays["cutout_coverage"])
                        < config.min_cutout_coverage
                    )
                )
            ),
            "n_valid_pixel_morphology": int(
                np.count_nonzero(arrays["pixel_valid"])
            ),
            "n_valid_aion_probabilities": int(
                np.count_nonzero(arrays["probability_valid"])
            ),
            "n_complete_output_features": int(np.count_nonzero(available)),
            "n_available": int(np.count_nonzero(available)),
            "n_possible_mismatch": int(
                np.count_nonzero(arrays["possible_morphological_mismatch"])
            ),
            "processing_complete": bool(
                np.all(arrays["status"][assigned] != STATUS_PENDING)
            ),
            "n_quality_rejected": int(
                np.count_nonzero(arrays["status"] == STATUS_QUALITY_REJECTED)
            ),
            "fits_backend": resolved_fits_backend,
            "flux_scale": config.flux_scale(band),
            "n_previously_processed": completed_before,
        }

    metadata = {
        "n_target_rows": len(catalogue["id"]),
        "catalogue_n_rows": int(catalogue["catalogue_n_rows"][0]),
        "bands": list(config.bands),
        "n_columns": len(multiband_column_names(config.bands)),
        "processing_complete": all(
            band_metadata[band]["processing_complete"] for band in config.bands
        ),
        "band_metadata": band_metadata,
        "aion_model": config.aion_model,
        "benchmark_repo": config.benchmark_repo,
        "aion_head_test_accuracy": float(checkpoint["test_accuracy"]),
        "aion_head_test_band_accuracy": checkpoint.get("test_band_accuracy", {}),
        "aion_head_temperature": float(checkpoint["temperature"]),
        "aion_training_input": "one Galaxy10 DES band per AION input",
        "aion_calibration_scope": (
            "temperature calibrated on held-out DES Galaxy10 only; "
            "CLAUDS/HSC target scores are uncalibrated cross-survey transfers"
        ),
        "aion_target_channels": AION_HSC_BAND_BY_TARGET,
        "domain_warning": (
            "The single-band head is trained on DES Galaxy10 images and transferred "
            "to HSC/CLAUDS images; u uses HSC-G and HSC-Y lacks a matching training band."
        ),
        "psf_matching": False,
        "fits_backend": resolved_fits_backend,
        "clauds_weight_interpretation": "inverse variance; empirically verified on real tiles",
        "feature_fingerprint": feature_fingerprint,
        "code_fingerprint": _code_fingerprint(),
        "catalogue_fingerprint": str(np.asarray(catalogue["fingerprint"]).item()),
        "surface_brightness_definition": (
            "signed sum(raw - sigma-clipped local-border background) over valid pixels"
        ),
        "mean_per_sqarcsec_definition": (
            "raw valid-pixel mean without background subtraction divided by WCS pixel area"
        ),
        "min_cutout_coverage": config.min_cutout_coverage,
        "min_aperture_coverage": config.min_aperture_coverage,
        "min_morphology_coverage": config.min_morphology_coverage,
        "min_signal_to_noise": config.min_signal_to_noise,
        "mismatch_definition": (
            "abs(p_elliptical_type_x - (1 - axis_ellipticity_x)) >= "
            f"{config.mismatch_threshold:g}"
        ),
        "manifest_fingerprint": manifest.fingerprint,
    }
    config.feature_dir.mkdir(parents=True, exist_ok=True)
    (config.feature_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    return {"arrays": arrays_by_band, "metadata": metadata}



def _open_all_band_features(
    config: MultibandMorphologyConfig, n_rows: int
) -> dict[str, dict[str, np.memmap]]:
    return {
        band: open_band_feature_arrays(config, band, n_rows)
        for band in config.bands
    }


def write_multiband_morphological_catalogue(
    config: MultibandMorphologyConfig,
    feature_product: dict[str, Any] | None = None,
) -> Path:
    """Atomically append all 72 band-resolved columns to a copy of the source FITS."""
    import fitsio

    config = config.normalized()
    if config.max_target_rows is not None:
        raise ValueError("A final catalogue requires max_target_rows=None.")
    if config.bands != MULTIBAND_MORPHOLOGY_BANDS:
        raise ValueError("A final catalogue requires all ugrizy bands.")
    catalogue = _read_catalogue(config)
    source_rows = int(catalogue["catalogue_n_rows"][0])
    manifest = build_or_load_image_manifest(config)
    feature_fingerprint = _prepare_feature_cache(
        config,
        catalogue_fingerprint=str(np.asarray(catalogue["fingerprint"]).item()),
        manifest_fingerprint=manifest.fingerprint,
    )
    arrays_by_band = (
        feature_product["arrays"]
        if feature_product is not None
        else _open_all_band_features(config, source_rows)
    )
    for band in config.bands:
        arrays = arrays_by_band[band]
        if len(arrays["p_spiral"]) != source_rows:
            raise RuntimeError(f"{band}-band features are not full catalogue length.")
        assignment_path = config.assignment_path(band)
        if not assignment_path.exists():
            raise FileNotFoundError(f"Missing assignment cache: {assignment_path}")
        with np.load(assignment_path) as assignment:
            if (
                str(np.asarray(assignment["catalogue_fingerprint"]).item())
                != str(np.asarray(catalogue["fingerprint"]).item())
                or str(np.asarray(assignment["manifest_fingerprint"]).item())
                != manifest.fingerprint
            ):
                raise RuntimeError(f"Stale {band}-band assignment provenance.")
            assigned = np.asarray(assignment["tile_index"]) >= 0
        if np.any(
            np.asarray(arrays["status"])[assigned] == STATUS_PENDING
        ):
            raise RuntimeError(
                f"Refusing to write while assigned {band}-band rows remain unprocessed."
            )

    output = config.output_catalogue_path
    required = set(multiband_column_names())
    if output.exists() and not config.overwrite_output:
        with fitsio.FITS(output) as output_fits:
            header = output_fits[1].read_header()
            if (
                required.issubset(output_fits[1].get_colnames())
                and header.get("MORPHCFG") == feature_fingerprint
            ):
                return output
        raise FileExistsError(
            f"Output exists with missing columns or different provenance: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(config.catalogue_path, temporary)
    try:
        with fitsio.FITS(temporary, mode="rw") as output_fits:
            table = output_fits[1]
            overlap = required.intersection(table.get_colnames())
            if overlap:
                raise RuntimeError(
                    f"Source already contains multiband morphology columns: {sorted(overlap)}"
                )
            for band in config.bands:
                for stem in MULTIBAND_FEATURE_STEMS:
                    name = f"{stem}_{band}"
                    print(f"Inserting FITS column: {name}")
                    table.insert_column(name, np.asarray(arrays_by_band[band][stem]))
            table.write_key("MORPHVER", 2, comment="Updated morphology pipeline version")
            table.write_key("MORPHCFG", feature_fingerprint, comment="Feature provenance hash")
            table.write_key("MORPHCOD", _code_fingerprint(), comment="Measurement code hash")
            table.write_key("MORPHIMG", manifest.fingerprint, comment="Image manifest hash")
            table.write_key(
                "MORPHCAT",
                str(np.asarray(catalogue["fingerprint"]).item()),
                comment="Input catalogue hash",
            )
            table.write_key("MORPHMOD", config.aion_model, comment="AION morphology encoder")
            table.write_key("MORPHDS", config.benchmark_repo, comment="Probe dataset")
            table.write_key("MORPHCAL", "DES-only", comment="Temperature calibration domain")
            table.write_key("MORPHTGT", False, comment="Target-domain calibrated")
            table.write_key("MORPHIO", config.fits_backend, comment="Requested FITS backend")
            table.write_key("MORPHSNR", config.min_signal_to_noise, comment="Minimum pixel S/N")
            table.write_key("MORPHMIS", config.mismatch_threshold, comment="Mismatch threshold")
            table.write_key("MORPHNBD", 6, comment="Number of morphology bands")
            table.write_key(
                "MORPHUNT", "scaled-native", comment="Brightness-column flux unit"
            )
            for band in config.bands:
                table.write_key(
                    f"MFSCAL{band.upper()}",
                    config.flux_scale(band),
                    comment=f"Applied {band}-band image flux scale",
                )
            table.write_history(
                "Band-resolved morphology uses CLAUDS u and HSC PDR3 grizy images; "
                "u is encoded through AION HSC-G and grizy through matching HSC channels."
            )
            table.write_history(
                "AION scores are temperature-calibrated on held-out DES Galaxy10 only; "
                "CLAUDS/HSC values are uncalibrated cross-survey transfers."
            )
            table.write_history(
                "Bands were not PSF-matched; u uses the HSC-G codec proxy and HSC-Y "
                "has no matching Galaxy10 training band."
            )
            table.write_history(
                "surface_brightness_* is a background-subtracted signed aperture sum; "
                "mean_per_sqarcsec_* is an unsubtracted mean divided by WCS pixel area."
            )
        os.replace(temporary, output)
        output_provenance = output.with_suffix(output.suffix + ".provenance.json")
        provenance_payload = json.loads(
            (config.feature_dir / "provenance.json").read_text()
        )
        provenance_payload["output"] = _path_signature(output)
        provenance_payload["feature_metadata"] = (
            feature_product.get("metadata") if feature_product is not None else None
        )
        output_provenance.write_text(
            json.dumps(provenance_payload, indent=2, default=str) + "\n"
        )
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return output


def verify_multiband_morphological_catalogue(
    path: str | Path,
    *,
    source_path: str | Path | None = None,
    expected_feature_fingerprint: str | None = None,
) -> dict[str, Any]:
    import fitsio

    path = Path(path)
    inconsistent_rows = 0
    unavailable_finite_rows = 0
    probability_range_errors = 0
    probability_subset_errors = 0
    unavailable_mismatch_errors = 0
    constant_columns: list[str] = []
    availability: dict[str, int] = {}
    quantiles: dict[str, dict[str, list[float]]] = {}
    with fitsio.FITS(path) as fits_file:
        table = fits_file[1]
        header = table.read_header()
        if int(header.get("MORPHVER", 0)) < 2:
            raise RuntimeError("Updated morphology provenance header is missing.")
        feature_fingerprint = header.get("MORPHCFG")
        if expected_feature_fingerprint is not None and feature_fingerprint != expected_feature_fingerprint:
            raise RuntimeError("Output feature provenance does not match the requested run.")
        names = set(table.get_colnames())
        missing = set(multiband_column_names()).difference(names)
        if missing:
            raise RuntimeError(f"Multiband morphology columns are missing: {sorted(missing)}")
        n_rows = int(table.get_nrows())
        if n_rows < 1:
            raise RuntimeError("Multiband morphology catalogue is empty.")
        sample_rows = np.asarray(sorted(set((0, n_rows - 1))), dtype=np.int64)
        sample = table.read(rows=sample_rows, columns=multiband_column_names())
        output_ids = np.asarray(table.read(columns=["ID"])["ID"])
        for band in MULTIBAND_MORPHOLOGY_BANDS:
            columns = tuple(f"{stem}_{band}" for stem in MULTIBAND_FEATURE_STEMS)
            values = table.read(columns=columns)
            finite_by_stem = {
                stem: np.isfinite(values[f"{stem}_{band}"])
                for stem in FLOAT_FEATURE_STEMS
            }
            complete = np.logical_and.reduce(list(finite_by_stem.values()))
            any_finite = np.logical_or.reduce(list(finite_by_stem.values()))
            available = np.asarray(values[f"morphology_available_{band}"], dtype=bool)
            mismatch = np.asarray(
                values[f"possible_morphological_mismatch_{band}"], dtype=bool
            )
            availability[band] = int(np.count_nonzero(available))
            if not availability[band]:
                raise RuntimeError(f"No valid updated morphology rows exist in {band} band.")
            inconsistent_rows += int(np.count_nonzero(available != complete))
            unavailable_finite_rows += int(np.count_nonzero((~available) & any_finite))
            unavailable_mismatch_errors += int(np.count_nonzero((~available) & mismatch))
            quantiles[band] = {}
            for stem in FLOAT_FEATURE_STEMS:
                column = np.asarray(values[f"{stem}_{band}"], dtype=np.float64)
                selected = column[available]
                if selected.size > 1 and float(np.nanmax(selected) - np.nanmin(selected)) <= 1.0e-12:
                    constant_columns.append(f"{stem}_{band}")
                quantiles[band][stem] = [
                    float(value)
                    for value in np.nanquantile(selected, (0.01, 0.5, 0.99))
                ]
            for stem in ("p_spiral", "p_bar", "p_elliptical_type"):
                probability = np.asarray(values[f"{stem}_{band}"], dtype=np.float64)
                probability_range_errors += int(
                    np.count_nonzero(
                        np.isfinite(probability)
                        & ((probability < 0.0) | (probability > 1.0))
                    )
                )
            p_spiral = np.asarray(values[f"p_spiral_{band}"], dtype=np.float64)
            p_bar = np.asarray(values[f"p_bar_{band}"], dtype=np.float64)
            probability_subset_errors += int(
                np.count_nonzero(available & (p_bar > p_spiral + 1.0e-6))
            )
            ellipticity = np.asarray(
                values[f"axis_ellipticity_{band}"], dtype=np.float64
            )
            probability_range_errors += int(
                np.count_nonzero(
                    available & ((ellipticity < 0.0) | (ellipticity > 1.0))
                )
            )
            for stem in ("concentration_C", "asymmetry_A"):
                measurement = np.asarray(values[f"{stem}_{band}"], dtype=np.float64)
                probability_range_errors += int(
                    np.count_nonzero(available & (measurement < 0.0))
                )
        n_columns = len(table.get_colnames())
    ids_match_source = None
    if source_path is not None:
        source_ids = np.asarray(fitsio.read(source_path, ext=1, columns=["ID"])["ID"])
        ids_match_source = bool(np.array_equal(source_ids, output_ids))
        if not ids_match_source:
            raise RuntimeError("Output object IDs or row order differ from the source catalogue.")
    errors = {
        "availability_consistency_errors": inconsistent_rows,
        "unavailable_finite_rows": unavailable_finite_rows,
        "probability_or_morphology_range_errors": probability_range_errors,
        "p_bar_not_subset_errors": probability_subset_errors,
        "unavailable_mismatch_errors": unavailable_mismatch_errors,
    }
    nonzero_errors = {name: value for name, value in errors.items() if value}
    if nonzero_errors:
        raise RuntimeError(f"Updated morphology verification failed: {nonzero_errors}")
    if constant_columns:
        raise RuntimeError(f"Constant morphology outputs detected: {constant_columns}")
    provenance_path = path.with_suffix(path.suffix + ".provenance.json")
    if not provenance_path.exists():
        raise RuntimeError(f"Output provenance sidecar is missing: {provenance_path}")
    return {
        "path": str(path),
        "n_rows": n_rows,
        "n_columns": n_columns,
        "n_morphology_columns": len(multiband_column_names()),
        "size_bytes": path.stat().st_size,
        "ids_match_source": ids_match_source,
        "feature_fingerprint": feature_fingerprint,
        "availability": availability,
        "quantiles_01_50_99": quantiles,
        "constant_columns": constant_columns,
        **errors,
        "sample_dtype": sample.dtype.descr,
    }


def build_multiband_morphological_catalogue(
    config: MultibandMorphologyConfig | None = None,
) -> dict[str, Any]:
    config = (config or MultibandMorphologyConfig()).normalized()
    if config.max_target_rows is not None:
        raise ValueError("The complete build requires max_target_rows=None.")
    build_or_load_image_manifest(config)
    embeddings = cache_single_band_benchmark_embeddings(config)
    checkpoint = train_single_band_morphology_head(
        config, embedding_product=embeddings
    )
    features = compute_multiband_morphology_features(config)
    output = write_multiband_morphological_catalogue(config, features)
    return {
        "output_catalogue_path": output,
        "head_path": config.head_path,
        "head_test_accuracy": checkpoint["test_accuracy"],
        "feature_metadata": features["metadata"],
        "verification": verify_multiband_morphological_catalogue(
            output,
            source_path=config.catalogue_path,
            expected_feature_fingerprint=features["metadata"]["feature_fingerprint"],
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = MultibandMorphologyConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("manifest", "train-head", "features", "catalogue", "all"),
        default="all",
        nargs="?",
    )
    parser.add_argument("--catalogue-path", type=Path, default=defaults.catalogue_path)
    parser.add_argument(
        "--output-catalogue-path", type=Path, default=defaults.output_catalogue_path
    )
    parser.add_argument("--u-image-dir", type=Path, default=defaults.u_image_dir)
    parser.add_argument("--hsc-image-dir", type=Path, default=defaults.hsc_image_dir)
    parser.add_argument("--cache-dir", type=Path, default=defaults.cache_dir)
    parser.add_argument("--bands", default=",".join(defaults.bands))
    parser.add_argument("--benchmark-repo", default=defaults.benchmark_repo)
    parser.add_argument("--aion-model", default=defaults.aion_model)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--fits-backend", choices=FITS_BACKENDS, default=defaults.fits_backend)
    parser.add_argument(
        "--benchmark-batch-size", type=int, default=defaults.benchmark_batch_size
    )
    parser.add_argument("--target-batch-size", type=int, default=defaults.target_batch_size)
    parser.add_argument("--head-epochs", type=int, default=defaults.head_epochs)
    parser.add_argument(
        "--head-learning-rate", type=float, default=defaults.head_learning_rate
    )
    parser.add_argument("--head-weight-decay", type=float, default=defaults.head_weight_decay)
    parser.add_argument("--head-patience", type=int, default=defaults.head_patience)
    parser.add_argument(
        "--min-cutout-coverage", type=float, default=defaults.min_cutout_coverage
    )
    parser.add_argument(
        "--min-aperture-coverage", type=float, default=defaults.min_aperture_coverage
    )
    parser.add_argument(
        "--min-morphology-coverage", type=float, default=defaults.min_morphology_coverage
    )
    parser.add_argument("--min-signal-to-noise", type=float, default=defaults.min_signal_to_noise)
    parser.add_argument("--mismatch-threshold", type=float, default=defaults.mismatch_threshold)
    for band in MULTIBAND_MORPHOLOGY_BANDS:
        parser.add_argument(
            f"--{band}-flux-scale",
            type=float,
            default=defaults.flux_scale(band),
            help=f"Multiply {band}-band pixels by this calibration factor.",
        )
    parser.add_argument("--max-target-rows", type=int, default=None)
    parser.add_argument("--stop-after-processed-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--force-manifest", action="store_true")
    parser.add_argument("--force-assignments", action="store_true")
    parser.add_argument("--force-benchmark-embeddings", action="store_true")
    parser.add_argument("--force-head-training", action="store_true")
    parser.add_argument("--force-target-features", action="store_true")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> MultibandMorphologyConfig:
    return MultibandMorphologyConfig(
        catalogue_path=args.catalogue_path,
        output_catalogue_path=args.output_catalogue_path,
        u_image_dir=args.u_image_dir,
        hsc_image_dir=args.hsc_image_dir,
        cache_dir=args.cache_dir,
        bands=parse_bands(args.bands),
        benchmark_repo=args.benchmark_repo,
        aion_model=args.aion_model,
        device=args.device,
        fits_backend=args.fits_backend,
        benchmark_batch_size=args.benchmark_batch_size,
        target_batch_size=args.target_batch_size,
        head_epochs=args.head_epochs,
        head_learning_rate=args.head_learning_rate,
        head_weight_decay=args.head_weight_decay,
        head_patience=args.head_patience,
        min_cutout_coverage=args.min_cutout_coverage,
        min_aperture_coverage=args.min_aperture_coverage,
        min_morphology_coverage=args.min_morphology_coverage,
        min_signal_to_noise=args.min_signal_to_noise,
        mismatch_threshold=args.mismatch_threshold,
        u_flux_scale=args.u_flux_scale,
        g_flux_scale=args.g_flux_scale,
        r_flux_scale=args.r_flux_scale,
        i_flux_scale=args.i_flux_scale,
        z_flux_scale=args.z_flux_scale,
        y_flux_scale=args.y_flux_scale,
        max_target_rows=args.max_target_rows,
        stop_after_processed_rows=args.stop_after_processed_rows,
        seed=args.seed,
        overwrite_output=args.overwrite_output,
        force_manifest=args.force_manifest,
        force_assignments=args.force_assignments,
        force_benchmark_embeddings=args.force_benchmark_embeddings,
        force_head_training=args.force_head_training,
        force_target_features=args.force_target_features,
    ).normalized()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = _config_from_args(args)
    if args.command == "manifest":
        manifest = build_or_load_image_manifest(config)
        print(
            json.dumps(
                {
                    "path": str(config.manifest_path),
                    "n_images": len(manifest.path),
                    "bands": {
                        band: int(np.count_nonzero(manifest.band == band))
                        for band in config.bands
                    },
                    "fingerprint": manifest.fingerprint,
                },
                indent=2,
            )
        )
        return
    if args.command == "train-head":
        embeddings = cache_single_band_benchmark_embeddings(config)
        checkpoint = train_single_band_morphology_head(
            config, embedding_product=embeddings
        )
        print(
            json.dumps(
                {key: value for key, value in checkpoint.items() if key != "state_dict"},
                indent=2,
                default=str,
            )
        )
        return
    if args.command == "features":
        train_single_band_morphology_head(config)
        product = compute_multiband_morphology_features(config)
        print(json.dumps(product["metadata"], indent=2, default=str))
        return
    if args.command == "catalogue":
        output = write_multiband_morphological_catalogue(config)
        print(
            json.dumps(
                verify_multiband_morphological_catalogue(
                    output, source_path=config.catalogue_path
                ),
                indent=2,
            )
        )
        return
    print(json.dumps(build_multiband_morphological_catalogue(config), indent=2, default=str))


if __name__ == "__main__":
    main()
