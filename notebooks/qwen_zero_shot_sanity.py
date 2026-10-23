#!/usr/bin/env python3
"""Small, fixed-weight HSC QLoRA transfer checks on Legacy and CLAUDS validation."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

import qwen_posttraining_comparison as clauds_pipeline
import qwen_survey_posttraining as survey_pipeline
from aion_magnitude.FM_Qwen import load_frozen_qwen, serialize_qwen_feature_row
from aion_magnitude import table_models as tm
from aion_magnitude.dataset import make_random_split
from aion_magnitude.metrics import summarize_pdf_metrics
from aion_magnitude.qwen_posttraining import (
    QwenPhotoZModel,
    TextRedshiftDataset,
    evaluate_text_photoz_model,
    make_text_collator,
    qwen_hidden_size,
)
from aion_magnitude.utils import make_redshift_grid, set_random_seed


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_cohort(run):
    meta = json.loads((run / "prepared.json").read_text())
    folder = Path(meta["cohort_cache"])
    with np.load(folder / "cohort.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    return data, meta


def select_queries(data, excluded_ids, n, seed, bounds=(0, 4.5)):
    """Choose validation objects without consulting their redshift within bounds."""
    allowed = (
        (data["split"] == "val")
        & np.isfinite(data["z"])
        & (data["z"] >= bounds[0])
        & (data["z"] <= bounds[1])
        & ~np.isin(data["targetid"], excluded_ids)
    )
    eligible = np.flatnonzero(allowed)
    # At most one record per physical catalogue group.
    _, first = np.unique(data["group"][eligible], return_index=True)
    eligible = eligible[first]
    if len(eligible) < n:
        raise ValueError(f"Only {len(eligible)} eligible validation objects for n={n}")
    return np.random.default_rng(seed).permutation(eligible)[:n]


def take(data, indices):
    return {key: value[indices] for key, value in data.items()}


def native_texts(kind, data, meta):
    if kind != "clauds":
        return survey_pipeline.texts_for(
            argparse.Namespace(catalogue_kind=kind), data, meta
        )
    args = clauds_pipeline.build_parser().parse_args(["--stage", "prepare"])
    _, serializer = clauds_pipeline.qwen_settings(args)
    return [
        serialize_qwen_feature_row(
            dict(zip(meta["feature_names"], row)), serialization=serializer
        )
        for row in data["features"]
    ]


def source_positions(source, source_meta):
    import h5py

    path = next(
        Path(s["path"]) for s in source_meta["sources"] if s["path"].endswith(".hdf5")
    )
    with h5py.File(path, "r") as f:
        ids = np.asarray(f["targetid"][:], dtype=np.int64)
        exposed = np.isin(ids, source["targetid"])
        ra = np.asarray(f["DESI_fibermap/TARGET_RA"][:], float)[exposed]
        dec = np.asarray(f["DESI_fibermap/TARGET_DEC"][:], float)[exposed]
        object_ids = np.asarray(f["object_id"][:], dtype=np.int64)[exposed]
    return ra, dec, object_ids


def legacy_positions(ids, meta):
    import fitsio

    positions = {}
    for source in meta["sources"]:
        path = Path(source["path"])
        if not path.name.startswith("tractorphot-"):
            continue
        with fitsio.FITS(str(path)) as f:
            tids = f[1].read(columns=["TARGETID"])["TARGETID"]
            rows = np.flatnonzero(np.isin(tids, ids))
            if len(rows):
                found = f[1].read(rows=rows, columns=["TARGETID", "RA", "DEC"])
                positions.update(
                    {
                        int(r["TARGETID"]): (float(r["RA"]), float(r["DEC"]))
                        for r in found
                    }
                )
        if len(positions) == len(ids):
            break
    if len(positions) != len(ids):
        raise ValueError("Missing Legacy positions for selected validation candidates")
    return np.asarray([positions[int(tid)] for tid in ids])


def clauds_cohort(catalogue=None):
    """Use the existing seeded CLAUDS selection, split, magnitudes, and imputation."""
    import fitsio

    args = clauds_pipeline.build_parser().parse_args(["--stage", "prepare"])
    if catalogue is not None:
        args.catalogue = Path(catalogue)
    path = clauds_pipeline.resolve_catalogue_path(args.catalogue)
    cat = tm.load_catalogue_data(
        path,
        max_rows=args.max_rows,
        seed=args.seed,
        include_full121=False,
        include_morphology=False,
        z_min=-np.inf,
        z_max=np.inf,
    )
    splits = make_random_split(
        len(cat.target),
        train_fraction=args.train_fraction,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    features, imputation = tm.impute_from_training(cat.magnitude_features, splits)
    with fitsio.FITS(str(path)) as f:
        names = f[1].get_colnames()
        pair = next(
            (
                (ra, dec)
                for ra, dec in (("RA", "DEC"), ("ra", "dec"))
                if ra in names and dec in names
            ),
            None,
        )
        if pair is None:
            raise ValueError("Cannot find CLAUDS sky-coordinate columns")
        pos = f[1].read(rows=cat.source_row, columns=list(pair))
    return dict(
        targetid=cat.object_id,
        group=cat.object_id.astype(str),
        features=features.to_numpy(dtype=np.float32),
        z=cat.target,
        split=np.asarray(splits),
        ra=pos[pair[0]],
        dec=pos[pair[1]],
    ), dict(
        catalogue=str(path),
        feature_names=list(features.columns),
        target="ZPHOT",
        input_representation="existing CLAUDS AB magnitudes with training-only imputation",
        imputation=imputation,
        catalogue_fingerprint={
            "path": str(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        },
    )


def positional_filter(data, indices, source_ra, source_dec, n, threshold=1.0):
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    queries = SkyCoord(data["ra"][indices] * u.deg, data["dec"][indices] * u.deg)
    catalog = SkyCoord(source_ra * u.deg, source_dec * u.deg)
    _, separation, _ = queries.match_to_catalog_sky(catalog)
    distances = separation.arcsec
    kept = indices[distances > threshold][:n]
    if len(kept) != n:
        raise ValueError("Too few queries remain after positional overlap exclusion")
    return kept, dict(
        radius_arcsec=threshold,
        candidates_checked=len(indices),
        candidates_excluded=int((distances <= threshold).sum()),
        minimum_selected_separation_arcsec=float(
            distances[distances > threshold][:n].min()
        ),
    )


def audit_lengths(tokenizer, texts, max_length):
    lengths = [len(tokenizer(text, truncation=False)["input_ids"]) for text in texts]
    if max(lengths) > max_length:
        raise ValueError(f"Input needs {max(lengths)} tokens; limit is {max_length}")
    return dict(
        n=len(texts),
        min=min(lengths),
        max=max(lengths),
        max_length=max_length,
        truncated=0,
    )


@contextmanager
def reserve_one_gpu():
    spec = importlib.util.spec_from_file_location(
        "gpu_guard", ROOT / "scripts/qwen_survey_gpu_guard.py"
    )
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    devices = guard.driver_uuids()
    if len(devices) != 1:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to one unused MIG UUID")
    folder = Path("/scratch/.tmp-gsm/aion_output/gpu_locks") / str(os.getuid())
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{devices[0]}.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = guard.memory_info(devices[0])
        if info["free"] < 8:
            raise RuntimeError(f"Selected slice is busy: {info}")
        yield dict(uuid=devices[0], memory_before_gib=info)


def evaluate(model, tokenizer, data, texts, config):
    dataset = TextRedshiftDataset(texts, data["z"], [str(x) for x in data["targetid"]])
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=make_text_collator(tokenizer, max_length=config["max_length"]),
    )
    edges, centers = make_redshift_grid(
        config["z_min"], config["z_max"], config["n_z_bins"]
    )
    return evaluate_text_photoz_model(
        model,
        loader,
        device=torch.device("cuda"),
        redshift_edges=edges,
        redshift_centers=centers,
    )


def save_predictions(path, data, evaluation):
    np.savez_compressed(
        path,
        targetid=data["targetid"],
        features=data["features"],
        **{
            k: evaluation[k].numpy()
            for k in (
                "logits",
                "pz",
                "z_spec",
                "z_p16",
                "z_p50",
                "z_p84",
                "redshift_edges",
                "redshift_centers",
            )
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-run",
        type=Path,
        default=Path(
            "/arc/home/gsm/aion_output/figures/qwen-galaxiesml-spec-qlora-specz-e10"
        ),
    )
    p.add_argument(
        "--legacy-run",
        type=Path,
        default=Path("/arc/home/gsm/aion_output/figures/qwen-desi-ls-qlora-specz-e10"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-queries", type=int, default=256)
    p.add_argument("--seed", type=int, default=1023)
    args = p.parse_args()
    if args.output_dir.exists():
        raise ValueError("Use a new isolated output directory")
    args.output_dir.mkdir(parents=True)
    started = time.time()
    source, source_meta = read_cohort(args.source_run)
    legacy, legacy_meta = read_cohort(args.legacy_run)
    source_result = torch.load(
        args.source_run / "qlora/result.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = source_result["metadata"]["config"]
    source_ra, source_dec, source_object_ids = source_positions(source, source_meta)
    output = dict(
        source_run=str(args.source_run),
        weights_updated=False,
        calibration_examples=0,
        seed=args.seed,
        n_queries=args.n_queries,
        source_config=config,
        source_exclusion="Entire source cohort, including train/validation/test; integer IDs and 1-arcsec positions",
        targets={},
    )
    targets = {}
    for kind in ("desi-ls", "clauds"):
        if kind == "desi-ls":
            data, meta = legacy, legacy_meta
            excluded = source["targetid"]
        else:
            data, meta = clauds_cohort()
            excluded = source_object_ids
        candidates = select_queries(data, excluded, args.n_queries * 2, args.seed)
        if kind == "desi-ls":
            coords = legacy_positions(data["targetid"][candidates], meta)
            data = dict(
                data,
                ra=np.full(len(data["z"]), np.nan),
                dec=np.full(len(data["z"]), np.nan),
            )
            data["ra"][candidates], data["dec"][candidates] = coords.T
        selected, overlap = positional_filter(
            data, candidates, source_ra, source_dec, args.n_queries
        )
        small = take(data, selected)
        texts = native_texts(kind, small, meta)
        targets[kind] = (small, texts)
        output["targets"][kind] = dict(
            label=meta["target"],
            feature_names=meta["feature_names"],
            selected_indices=selected.tolist(),
            overlap_audit=overlap,
            split="validation",
            redshift_range=[float(small["z"].min()), float(small["z"].max())],
            target_preprocessing=meta.get(
                "input_representation", "native signed fluxes, sigma, and EBV"
            ),
        )
        (args.output_dir / f"{kind}_inputs.json").write_text(
            json.dumps(
                dict(object_id=[str(x) for x in small["targetid"]], texts=texts),
                indent=2,
            )
        )
        print(
            f"Prepared {kind}: {len(texts)} queries; positional audit {overlap}",
            flush=True,
        )
    # Snapshot the completed inference artifacts, never a live training checkpoint.
    snapshot = args.output_dir / "source_snapshot"
    shutil.copytree(args.source_run / "qlora/adapter", snapshot / "adapter")
    shutil.copy2(args.source_run / "qlora/photoz_head.pt", snapshot / "photoz_head.pt")
    hashes = {
        str(f.relative_to(snapshot)): sha256(f)
        for f in snapshot.rglob("*")
        if f.is_file()
    }
    output["source_snapshot_sha256"] = hashes
    with reserve_one_gpu() as device:
        output["gpu"] = device
        set_random_seed(args.seed)
        from peft import PeftModel, prepare_model_for_kbit_training

        base, tokenizer = load_frozen_qwen(
            config["model_path"],
            device="cuda",
            load_in_4bit=True,
            torch_dtype="bf16",
            local_files_only=True,
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        qwen = PeftModel.from_pretrained(base, snapshot / "adapter", is_trainable=False)
        model = QwenPhotoZModel(
            qwen,
            hidden_size=qwen_hidden_size(qwen),
            n_z_bins=config["n_z_bins"],
            head_hidden_dim=config["head_hidden_dim"],
            pooling=config["pooling"],
        ).to("cuda")
        model.photoz_head.load_state_dict(
            torch.load(
                snapshot / "photoz_head.pt", map_location="cpu", weights_only=True
            )
        )
        model.requires_grad_(False)
        model.eval()
        assert not any(parameter.requires_grad for parameter in model.parameters())
        rows = np.flatnonzero(source["split"] == "val")[:32]
        small_source = take(source, rows)
        source_texts = native_texts("galaxiesml-spec", small_source, source_meta)
        audit_lengths(tokenizer, source_texts, config["max_length"])
        replay = evaluate(model, tokenizer, small_source, source_texts, config)
        saved = source_result["val_evaluation"]
        saved_rows = {str(tid): i for i, tid in enumerate(saved["object_id"])}
        indices = [saved_rows[str(tid)] for tid in small_source["targetid"]]
        reference = saved["logits"][indices]
        torch.testing.assert_close(replay["logits"], reference, rtol=1e-3, atol=0.02)
        max_pdf_error = float((replay["pz"] - saved["pz"][indices]).abs().max())
        if max_pdf_error > 0.005:
            raise ValueError(f"Source replay PDF mismatch: {max_pdf_error}")
        output["source_replay"] = dict(
            n=32,
            max_abs_logit_error=float((replay["logits"] - reference).abs().max()),
            max_abs_pdf_error=max_pdf_error,
            passed=True,
        )
        save_predictions(args.output_dir / "source_replay.npz", small_source, replay)
        print(f"Source checkpoint replay passed: {output['source_replay']}", flush=True)
        for kind, (data, texts) in targets.items():
            output["targets"][kind]["token_audit"] = audit_lengths(
                tokenizer, texts, config["max_length"]
            )
            before = time.time()
            evaluation = evaluate(model, tokenizer, data, texts, config)
            save_predictions(
                args.output_dir / f"{kind}_predictions.npz", data, evaluation
            )
            output["targets"][kind]["metrics"] = summarize_pdf_metrics(evaluation)
            output["targets"][kind]["inference_seconds"] = time.time() - before
            print(kind, json.dumps(output["targets"][kind]["metrics"]), flush=True)
            survey_pipeline.write_json(args.output_dir / "summary.json", output)
    if hashes != {
        str(f.relative_to(snapshot)): sha256(f)
        for f in snapshot.rglob("*")
        if f.is_file()
    }:
        raise ValueError("Source inference artifacts changed during the pilot")
    output["elapsed_seconds"] = time.time() - started
    survey_pipeline.write_json(args.output_dir / "summary.json", output)


if __name__ == "__main__":
    main()
