"""Photometry-only, spec-z-labelled cohorts for DESI–LS and DESI x HSC.

The first experiment uses native observed fluxes in AB nanomaggies, including
signed LS measurements, with errors and explicit missing values. It does not
read images or spectral pixels. A uniform DELTACHI2 cut is a training-sample
quality control, not the DESI tracer-specific LSS/OII success definition.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

DATA_ROOT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time")
DEFAULT_DESI_ROOT = DATA_ROOT / "DESI-LS"
DEFAULT_HSC_FILE = DATA_ROOT / "GalaxiesMLSpec/DESI_HSC_64x64_v2.hdf5"
ZCAT_RELATIVE = Path("catalogues/desi_dr1/iron/zcatalog/v1/zall-pix-iron.fits")
TRACTOR_RELATIVE = Path("catalogues/ls_dr9/iron/v1.1/tractorphot")
SCHEMA_VERSION = "native_photometry_specz_v1"
BANDS = {
    "desi-ls": ("g", "r", "z", "W1", "W2"),
    "galaxiesml-spec": ("g", "r", "i", "z", "y"),
}


def source_paths(kind, desi_root, hsc_file):
    root = Path(desi_root)
    paths = [root / ZCAT_RELATIVE]
    if kind == "galaxiesml-spec":
        paths.append(Path(hsc_file))
    if kind == "desi-ls":
        paths.extend(sorted((root / TRACTOR_RELATIVE).glob("tractorphot-*-iron.fits")))
        if len(paths) == 1:
            raise FileNotFoundError(
                f"No LS DR9 photometry shards in {root / TRACTOR_RELATIVE}"
            )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def fingerprints(paths):
    return [
        {
            "path": str(p.resolve()),
            "bytes": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
        }
        for p in paths
    ]


def exact_ids(values):
    """Preserve large catalogue identifiers; never pass them through float."""
    a = np.asarray(values)
    if a.dtype.kind not in "iuSUO":
        raise ValueError(
            "Catalogue IDs must be integer or decimal strings, never floating point"
        )
    return np.asarray([int(x) for x in a], dtype=np.int64)


def _ls_keys(release, brick, obj):
    return [
        f"ls:{int(r)}:{int(b)}:{int(o)}" if r > 0 and b >= 0 and o >= 0 else ""
        for r, b, o in zip(release, brick, obj)
    ]


def match_indices(sorted_ids, query):
    positions = np.searchsorted(sorted_ids, query)
    valid = positions < len(sorted_ids)
    valid[valid] &= sorted_ids[positions[valid]] == query[valid]
    return valid, positions


def read_redshifts(
    path, *, kind, z_min, z_max, min_deltachi2, wanted=None, chunk_rows=200000
):
    import fitsio

    columns = [
        "TARGETID",
        "Z",
        "ZERR",
        "ZCAT_PRIMARY",
        "SPECTYPE",
        "OBJTYPE",
        "ZWARN",
        "COADD_FIBERSTATUS",
        "DELTACHI2",
        "SURVEY",
        "PROGRAM",
    ]
    parts = []
    seen = 0
    with fitsio.FITS(str(path)) as f:
        table = f[1]
        for start in range(0, table.get_nrows(), chunk_rows):
            rows = table.read(
                rows=np.arange(start, min(start + chunk_rows, table.get_nrows())),
                columns=columns,
            )
            seen += len(rows)
            if wanted is not None:
                selected, _ = match_indices(wanted, rows["TARGETID"])
                rows = rows[selected]
            text = lambda name: np.char.strip(rows[name].astype("U"))
            good = (
                (rows["ZCAT_PRIMARY"] != 0)
                & (text("SPECTYPE") == "GALAXY")
                & (text("OBJTYPE") == "TGT")
                & (rows["ZWARN"] == 0)
                & (rows["COADD_FIBERSTATUS"] == 0)
                & np.isfinite(rows["DELTACHI2"])
                & (rows["DELTACHI2"] > min_deltachi2)
                & np.isfinite(rows["Z"])
                & (rows["Z"] >= z_min)
                & (rows["Z"] <= z_max)
                & np.isfinite(rows["ZERR"])
                & (rows["ZERR"] >= 0)
            )
            if kind == "desi-ls":
                good &= (text("SURVEY") == "main") & np.isin(
                    text("PROGRAM"), ["bright", "dark"]
                )
            if good.any():
                parts.append(rows[good][["TARGETID", "Z", "ZERR"]].copy())
    if not parts:
        raise ValueError(
            "No galaxies pass the explicit spectroscopic quality selection"
        )
    rows = np.concatenate(parts)
    rows = rows[np.argsort(rows["TARGETID"], kind="stable")]
    if np.any(np.diff(rows["TARGETID"]) == 0):
        raise ValueError("Multiple primary redshifts for one DESI TARGETID")
    return rows, {
        "redshift_rows_scanned": seen,
        "quality_selected_redshifts": len(rows),
    }


def flux_features(flux, ivar, nobs=None):
    """Retain valid negative fluxes, and mark unavailable measurements explicitly."""
    flux, ivar = np.asarray(flux, float), np.asarray(ivar, float)
    valid = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
    if nobs is not None:
        valid &= np.asarray(nobs) > 0
    sigma = np.full(flux.shape, np.nan)
    sigma[valid] = 1 / np.sqrt(ivar[valid])
    return np.where(valid, flux, np.nan), sigma, valid


def magnitude_features(mag, error):
    mag, error = np.asarray(mag, float), np.asarray(error, float)
    valid = (
        np.isfinite(mag)
        & (mag > 0)
        & (mag < 40)
        & np.isfinite(error)
        & (error > 0)
        & (error < 10)
    )
    flux = np.full(mag.shape, np.nan)
    flux[valid] = 10 ** ((22.5 - mag[valid]) / 2.5)
    sigma = flux * (np.log(10) / 2.5) * error
    return flux, sigma, valid


class _Sample:
    def __init__(self, limit, seed):
        self.limit = limit
        self.rng = np.random.default_rng(seed)
        self.parts = []
        self.count = 0
        self.eligible = 0

    def add(self, data):
        n = len(data["targetid"])
        if not n:
            return
        data["priority"] = self.rng.random(n)
        self.parts.append(data)
        self.count += n
        self.eligible += n
        if self.limit and self.count > 2 * self.limit:
            self.compact()

    def compact(self):
        data = {k: np.concatenate([p[k] for p in self.parts]) for k in self.parts[0]}
        if self.limit and len(data["targetid"]) > self.limit:
            keep = np.argpartition(data["priority"], self.limit - 1)[: self.limit]
            data = {k: v[keep] for k, v in data.items()}
        self.parts = [data]
        self.count = len(data["targetid"])
        return data


def build_cohort(
    *,
    kind,
    desi_root=DEFAULT_DESI_ROOT,
    hsc_file=DEFAULT_HSC_FILE,
    max_rows=300000,
    seed=42,
    train_fraction=0.20,
    val_fraction=0.05,
    z_min=0.0,
    z_max=4.5,
    min_deltachi2=40.0,
    min_valid_bands=3,
):
    if kind not in BANDS:
        raise ValueError(f"Unknown catalogue: {kind}")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive or None")
    if not (
        0 < train_fraction < 1
        and 0 < val_fraction < 1
        and train_fraction + val_fraction < 1
    ):
        raise ValueError(
            "Train/validation fractions must leave a nonempty test fraction"
        )
    if not (np.isfinite(z_min) and np.isfinite(z_max) and z_max > z_min):
        raise ValueError("Finite increasing redshift bounds required")
    if not 1 <= min_valid_bands <= 5:
        raise ValueError("min_valid_bands must be between 1 and 5")
    paths = source_paths(kind, desi_root, hsc_file)
    sample = _Sample(max_rows, seed)
    bands = BANDS[kind]
    if kind == "galaxiesml-spec":
        import h5py

        with h5py.File(hsc_file, "r") as f:
            wanted = np.sort(exact_ids(f["targetid"][:]))
    else:
        wanted = None
    redshifts, audit = read_redshifts(
        Path(desi_root) / ZCAT_RELATIVE,
        kind=kind,
        z_min=z_min,
        z_max=z_max,
        min_deltachi2=min_deltachi2,
        wanted=wanted,
    )
    disagreements = 0
    if kind == "desi-ls":
        import fitsio

        columns = ["TARGETID", "RELEASE", "BRICKID", "OBJID", "RA", "DEC", "EBV"]
        columns += [
            f"{prefix}_{b.upper()}"
            for b in bands
            for prefix in ("FLUX", "FLUX_IVAR", "NOBS")
        ]
        for path in paths[1:]:
            with fitsio.FITS(str(path)) as f:
                t = f[1]
                for start in range(0, t.get_nrows(), 100000):
                    rows = t.read(
                        rows=np.arange(start, min(start + 100000, t.get_nrows())),
                        columns=columns,
                    )
                    valid, index = match_indices(
                        redshifts["TARGETID"], rows["TARGETID"]
                    )
                    rows, index = rows[valid], index[valid]
                    if not len(rows):
                        continue
                    pairs = [
                        flux_features(
                            rows[f"FLUX_{b.upper()}"],
                            rows[f"FLUX_IVAR_{b.upper()}"],
                            rows[f"NOBS_{b.upper()}"],
                        )
                        for b in bands
                    ]
                    good = np.sum([p[2] for p in pairs], axis=0) >= min_valid_bands
                    keys = _ls_keys(rows["RELEASE"], rows["BRICKID"], rows["OBJID"])
                    groups = [
                        k if k else f"desi:{int(tid)}"
                        for k, tid in zip(keys, rows["TARGETID"])
                    ]
                    values = np.column_stack(
                        [a for pair in pairs for a in pair[:2]] + [rows["EBV"]]
                    )
                    sample.add(
                        {
                            "targetid": rows["TARGETID"][good].astype(np.int64),
                            "group": np.asarray(groups, dtype="U80")[good],
                            "features": values[good].astype(np.float32),
                            "z": redshifts["Z"][index][good].astype(np.float32),
                            "zerr": redshifts["ZERR"][index][good].astype(np.float32),
                        }
                    )
            print(
                f"photometry shard: {path.name}; eligible={sample.eligible:,}",
                flush=True,
            )
    else:
        import h5py

        with h5py.File(hsc_file, "r") as f:
            targetids = exact_ids(f["targetid"][:])
            valid, index = match_indices(redshifts["TARGETID"], targetids)
            # Check labels against current DR1 quality catalogue; retain native HDF5 labels.
            z = np.asarray(f["DESI_fibermap/DESI_redshift"][:], float)
            zerr = np.asarray(f["DESI_fibermap/DESI_redshift_err"][:], float)
            valid &= (
                np.isfinite(z)
                & (z >= z_min)
                & (z <= z_max)
                & np.isfinite(zerr)
                & (zerr >= 0)
            )
            good_indices = np.flatnonzero(valid)
            consistent = np.abs(
                z[good_indices] - redshifts["Z"][index[good_indices]]
            ) <= 0.001 * (1 + z[good_indices])
            disagreements = int((~consistent).sum())
            valid[good_indices[~consistent]] = False
            pairs = [
                magnitude_features(
                    f[f"HSC_metadata/{b}_cmodel_mag"][:],
                    f[f"HSC_metadata/{b}_cmodel_magsigma"][:],
                )
                for b in bands
            ]
            valid &= np.sum([p[2] for p in pairs], axis=0) >= min_valid_bands
            groups = [f"hsc:{oid}" for oid in exact_ids(f["object_id"][:])]
            values = np.column_stack(
                [a for pair in pairs for a in pair[:2]] + [f["DESI_fibermap/EBV"][:]]
            )
            sample.add(
                {
                    "targetid": targetids[valid],
                    "group": np.asarray(groups, dtype="U80")[valid],
                    "features": values[valid].astype(np.float32),
                    "z": z[valid].astype(np.float32),
                    "zerr": zerr[valid].astype(np.float32),
                }
            )
    if not sample.parts:
        raise ValueError("No matched galaxies with adequate photometry")
    data = sample.compact()
    # One row per physical imaging group: prefer smallest spec-z uncertainty, then TARGETID.
    order = np.lexsort((data["targetid"], data["zerr"], data["group"]))
    _, first = np.unique(data["group"][order], return_index=True)
    keep = order[first]
    removed = len(data["z"]) - len(keep)
    keep = keep[np.argsort(data["targetid"][keep])]
    data = {k: v[keep] for k, v in data.items() if k != "priority"}
    data["features"][~np.isfinite(data["features"])] = np.nan
    u = np.asarray(
        [
            int.from_bytes(hashlib.sha256(f"{seed}:{g}".encode()).digest()[:8], "big")
            / 2**64
            for g in data["group"]
        ]
    )
    data["split"] = np.where(
        u < train_fraction,
        "train",
        np.where(u < train_fraction + val_fraction, "val", "test"),
    )
    counts = {
        split: int(np.sum(data["split"] == split)) for split in ("train", "val", "test")
    }
    if not all(counts.values()):
        raise ValueError(f"Every split must have rows; increase max_rows: {counts}")
    prefix = "LS_DR9_tractor" if kind == "desi-ls" else "HSC_PDR2_cmodel"
    names = [
        f"{prefix}_{b}_{measure}" for b in bands for measure in ("flux", "flux_error")
    ] + ["MW_EBV"]
    metadata = {
        "schema": SCHEMA_VERSION,
        "catalogue_kind": kind,
        "sources": fingerprints(paths),
        "feature_names": names,
        "bands": list(bands),
        "flux_unit": "AB nanomaggy",
        "extinction": "as_catalogued; EBV supplied; no additional extinction correction",
        "target": "Z" if kind == "desi-ls" else "DESI_fibermap/DESI_redshift",
        "quality": {
            "spectype": "GALAXY",
            "zcat_primary": True,
            "zwarn": 0,
            "fiberstatus": 0,
            "deltachi2_greater_than": min_deltachi2,
            "minimum_valid_bands": min_valid_bands,
            "survey": (
                "main bright/dark"
                if kind == "desi-ls"
                else "all DR1 surveys matched to HDF5"
            ),
            "note": "uniform training quality cut; not tracer-specific LSS/OII selection",
        },
        "z_bounds": [z_min, z_max],
        "seed": seed,
        "max_rows": max_rows,
        "split_rule": "SHA256(seed:imaging_group); shared across methods within this catalogue",
        "split_fractions": {
            "train": train_fraction,
            "val": val_fraction,
            "test": 1 - train_fraction - val_fraction,
        },
        "split_counts": counts,
        "n_rows": len(data["z"]),
        "eligible_photometry_rows": sample.eligible,
        "duplicate_imaging_rows_removed_after_sampling": removed,
        "native_reference_redshift_disagreements_removed": disagreements,
        "image_pixels_read": False,
        "spectral_pixels_read": False,
        **audit,
    }
    return data, metadata
