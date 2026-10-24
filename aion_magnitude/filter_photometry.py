"""Versioned native-band serializers; no labels, fitting, or network access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .FM_Qwen import QwenSerializationConfig, serialize_qwen_feature_row

SERIALIZER_VERSION = "filter_photometry_v1"
DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "configs/filter_registry_v1.json"
CLAUDS_BANDS = ("u", "u_star", "g", "r", "i", "z", "y", "Y", "J", "H", "Ks")
SURVEY_BANDS = {"desi-ls": ("g", "r", "z", "W1", "W2"),
                "galaxiesml-spec": ("g", "r", "i", "z", "y"), "clauds": CLAUDS_BANDS}


def load_registry(path=DEFAULT_REGISTRY, *, verify_curves=False):
    path = Path(path)
    registry = json.loads(path.read_text())
    for key, entry in registry["filters"].items():
        if (entry["wavelength_unit"] != "nm" or entry["pivot_nm"] <= 0
                or entry["width_nm"] <= 0):
            raise ValueError(f"Invalid wavelength/width for {key}")
        if verify_curves:
            for prefix in ("curve", "original"):
                actual = hashlib.sha256((path.parent / entry[f"{prefix}_path"]).read_bytes()).hexdigest()
                if actual != entry[f"{prefix}_sha256"]:
                    raise ValueError(f"Response checksum mismatch: {key}")
    return registry


def filter_key(kind, band, legacy_release=None):
    if kind == "desi-ls":
        if band in ("W1", "W2"):
            return f"WISE.{band}"
        if legacy_release in (9010, 9012):
            return f"DECam.{band}"
        if legacy_release == 9011:
            return f"{'MzLS' if band == 'z' else 'BASS'}.{band}"
        raise ValueError("Legacy optical filters require a known DR9 RELEASE (9010/9011/9012); no footprint guess is made")
    if kind == "galaxiesml-spec" or (kind == "clauds" and band in ("g", "r", "i", "z", "y")):
        return f"HSC.{band}"
    if kind == "clauds":
        return f"{'MegaCam' if band in ('u', 'u_star') else 'VIRCAM'}.{band}"
    raise ValueError(f"Unsupported catalogue: {kind}")


def legacy_release_from_group(group):
    fields = str(group).split(":")
    if len(fields) != 4 or fields[0] != "ls":
        raise ValueError("Expected cached LS group ls:release:brick:object")
    return int(fields[1])


def feature_columns(kind):
    if kind == "clauds":
        return [(b, f"{b}_mag", None) for b in CLAUDS_BANDS]
    prefix = "LS_DR9_tractor" if kind == "desi-ls" else "HSC_PDR2_cmodel"
    return [(b, f"{prefix}_{b}_flux", f"{prefix}_{b}_flux_error") for b in SURVEY_BANDS[kind]]


def legacy_config(kind):
    if kind == "clauds":
        return QwenSerializationConfig(schema_name="clauds_all_magnitude_v1", prefix="galaxy all_magnitudes_ab", include_hsc_grizy=False, include_object_metadata=False)
    return QwenSerializationConfig(schema_name=f"{kind}_native_photometry_specz_v1", prefix="galaxy native_observed_photometry flux_unit=AB_nanomaggy errors=one_sigma extinction=as_catalogued", include_hsc_grizy=False, include_object_metadata=False)


def _number(value):
    return f"{float(value):.5f}" if np.isfinite(value) else "NA"


def magnitude_to_nanomaggy(magnitude, sigma_mag=None):
    """AB conversion; first-order sigma valid for small magnitude errors."""
    m = np.asarray(magnitude, float)
    f = np.full(m.shape, np.nan)
    valid = np.isfinite(m) & (m != -99)
    f[valid] = 10 ** ((22.5 - m[valid]) / 2.5)
    sigma = np.full(m.shape, np.nan)
    if sigma_mag is not None:
        e = np.asarray(sigma_mag, float)
        good = valid & np.isfinite(e) & (e > 0)
        sigma[good] = f[good] * np.log(10) / 2.5 * e[good]
    return f, sigma


def clauds_flux_to_nanomaggy(flux, sigma):
    """CLAUDS catalogue ZP=23 linear units -> AB nanomaggies (ZP=22.5).

    Valid negative/zero fluxes remain valid; missing sentinel -99 is not flux.
    Flux and uncertainty validity are retained independently.
    """
    f, e = np.asarray(flux, float), np.asarray(sigma, float)
    scale = 10 ** ((22.5 - 23.0) / 2.5)
    valid = np.isfinite(f) & (f != -99)
    good_e = valid & np.isfinite(e) & (e > 0)
    return np.where(valid, f * scale, np.nan), np.where(good_e, e * scale, np.nan), valid


def serialize_photometry(features, *, catalogue, level=0, numerics="native", registry=None,
                         legacy_release=None, clauds_flux=None):
    """Serialize named photometry only, rejecting extra fields such as redshift.

    Native CLAUDS values are the archived, training-imputed magnitudes. Their
    original masks belong to the dataset sidecar, not to a changed Level-0 text.
    Optional CLAUDS fluxes must come from raw catalogue flux/error columns, never
    by exponentiating imputed magnitudes (which would lose negative fluxes).
    """
    if catalogue not in SURVEY_BANDS or level not in (0, 1, 2):
        raise ValueError("Known catalogue and representation level 0/1/2 required")
    if numerics not in ("native", "flux"):
        raise ValueError("numerics must be native or flux")
    columns = feature_columns(catalogue)
    allowed = {name for _, value, error in columns for name in (value, error) if name}
    if catalogue != "clauds":
        allowed.add("MW_EBV")
    if set(features) - allowed:
        raise ValueError(f"Unexpected input fields (possible label leakage): {sorted(set(features) - allowed)}")
    if level == 0:
        if numerics != "native":
            raise ValueError("Level 0 is exact legacy/native only; use Level 1/2 for flux harmonisation")
        return serialize_qwen_feature_row(features, serialization=legacy_config(catalogue))
    if registry is None:
        registry = load_registry()
    use_mag = catalogue == "clauds" and numerics == "native"
    if catalogue == "clauds" and numerics == "flux" and clauds_flux is None:
        raise ValueError("Raw CLAUDS flux/error pairs are required for flux harmonisation")
    unit = "AB_mag" if use_mag else "AB_nanomaggy"
    measurement = {"clauds": "cmodel", "galaxiesml-spec": "cmodel", "desi-ls": "Tractor"}[catalogue]
    lines = [f"galaxy survey={catalogue} measurement={measurement} unit={unit} sigma_unit={unit} extinction=as_loaded"]
    for index, (band, value_name, error_name) in enumerate(columns):
        if value_name not in features:
            raise ValueError(f"Missing expected feature: {value_name}")
        key = filter_key(catalogue, band, legacy_release)
        entry = registry["filters"][key]
        if use_mag:
            value, error = features[value_name], np.nan  # original pipeline has no magnitude errors
        elif catalogue == "clauds":
            value, error = clauds_flux[index]
        else:
            value, error = features[value_name], features[error_name]
        physical = f" pivot_nm={entry['pivot_nm']:.3f} width_nm={entry['width_nm']:.3f}" if level == 2 else ""
        lines.append(f"filter={key}{physical} {'mag' if use_mag else 'flux'}={_number(value)} sigma={_number(error)}")
    lines.append(f"EBV={_number(features.get('MW_EBV', np.nan))}")
    return "\n".join(lines)


def serialize_rows(data, metadata, *, catalogue, level, numerics="native", registry=None):
    names = metadata["feature_names"]
    if len(names) != len(set(names)) or data["features"].shape[1] != len(names):
        raise ValueError("Unique feature names aligned with matrix columns required")
    texts = []
    for i, row in enumerate(data["features"]):
        release = None
        if catalogue == "desi-ls" and level:
            release = legacy_release_from_group(data["group"][i])
        extra = data["common_flux"][i] if "common_flux" in data else None
        texts.append(serialize_photometry(dict(zip(names, row)), catalogue=catalogue,
                     level=level, numerics=numerics, registry=registry,
                     legacy_release=release, clauds_flux=extra))
    return texts


def audit_tokens(tokenizer, texts, max_length, *, batch_size=256):
    if not texts:
        raise ValueError("Cannot audit an empty cohort")
    lengths = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], truncation=False, padding=False)
        lengths.extend(map(len, encoded["input_ids"]))
    a = np.asarray(lengths)
    if a.max() > max_length:
        raise ValueError(f"Complete records need {a.max()} tokens; max_seq_length={max_length}")
    return dict(n=len(a), min=int(a.min()), median=float(np.median(a)), p95=float(np.quantile(a, .95)),
                max=int(a.max()), max_seq_length=max_length, truncated=0, total_tokens=int(a.sum()))
