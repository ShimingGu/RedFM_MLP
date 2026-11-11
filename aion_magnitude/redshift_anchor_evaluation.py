"""Frozen evaluation primitives: identities, chunked inference, metrics and bootstrap."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .redshift_anchor_calibration import CENTERS, EDGES, apply_calibration, z_bin

BANDS = ("g", "r", "i", "z", "y")
FIXED_BINS = np.array([0.0, 0.6, 0.8, 1.0, 1.5, 2.0, 4.5])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def lower_median(a):
    a = np.asarray(a)
    return float(np.partition(a, (len(a) - 1) // 2)[(len(a) - 1) // 2])


def quantiles_from_pz(pz, probs=(0.025, 0.16, 0.50, 0.84, 0.975)):
    pz = np.asarray(pz, dtype=np.float64)
    cdf = np.cumsum(pz, axis=1)
    return {q: CENTERS[np.argmax(cdf >= q, axis=1)] for q in probs}


def serialize_e_membership(path, repo_root):
    """Serialize the sealed E HSC photometry with the identical Level-1 serializer."""
    import sys

    root = str(Path(repo_root))
    if root not in sys.path:
        sys.path.insert(0, root)
    from aion_magnitude.filter_photometry import load_registry, serialize_photometry

    e = read(path)
    names = [f"HSC_PDR2_cmodel_{b}_{s}" for b in BANDS for s in ("flux", "flux_error")] + ["MW_EBV"]
    registry = load_registry(Path(repo_root) / "configs/filter_registry_v1.json", verify_curves=True)
    texts = []
    for f in e["features"]:
        texts.append(serialize_photometry(dict(zip(names, f.astype(float))), catalogue="galaxiesml-spec", level=1, registry=registry))
    assert len(set(int(x) for x in e["targetid"])) == len(e["targetid"])
    return dict(texts=texts, targetid=e["targetid"], z=e["z"], group=e["group"], features_sha256=hashlib.sha256(np.ascontiguousarray(e["features"]).tobytes()).hexdigest())


def load_p_texts(h1_root):
    h1_root = Path(h1_root)
    manifest = json.loads((h1_root / "manifest.json").read_text())
    locked = h1_root / "evaluation/locked_P.npz"
    assert sha(locked) == manifest["P_sha256"][str(locked)]
    d = read(locked)
    records = [json.loads(x) for x in (h1_root / "evaluation/P_inputs.jsonl").read_text().splitlines() if x.strip()]
    assert len(records) == len(d["targetid"]) == 61696
    assert np.array_equal(np.array([int(r["targetid"]) for r in records]), d["targetid"])
    return dict(texts=[r["text"] for r in records], targetid=d["targetid"], z=d["z"], group=np.array(["P"] * len(d["targetid"])))


def load_h1_predictions(h1_root, arm, cohort):
    h1_root = Path(h1_root)
    folder = h1_root / "evaluation" / arm
    complete = json.loads((folder / "complete.json").read_text())
    for name, digest in complete["prediction_sha256"].items():
        assert sha(folder / name) == digest, (arm, name)
    chunks = [read(p) for p in sorted(folder.glob("??????_??????.npz"))]
    assert chunks
    pred = {k: (chunks[0][k] if k.startswith("redshift_") else np.concatenate([c[k] for c in chunks])) for k in chunks[0]}
    np.testing.assert_array_equal(pred["targetid"], cohort["targetid"])
    np.testing.assert_array_equal(pred["z_spec"], cohort["z"])
    for c in chunks:
        for k in ("redshift_edges", "redshift_centers"):
            np.testing.assert_array_equal(c[k], pred[k])
    np.testing.assert_allclose(pred["pz"].sum(1), 1.0, atol=2e-6, rtol=1e-5)
    assert np.isfinite(pred["logits"]).all() and np.isfinite(pred["pz"]).all() and (pred["pz"] >= 0).all()
    return pred


def load_prediction_chunks(folder, cohort):
    folder = Path(folder)
    complete = json.loads((folder / "complete.json").read_text())
    for name, digest in complete["prediction_sha256"].items():
        assert sha(folder / name) == digest, name
    chunks = [read(p) for p in sorted(folder.glob("??????_??????.npz"))]
    assert chunks
    pred = {k: (chunks[0][k] if k.startswith("redshift_") else np.concatenate([c[k] for c in chunks])) for k in chunks[0]}
    np.testing.assert_array_equal(pred["targetid"], cohort["targetid"])
    np.testing.assert_array_equal(pred["z_spec"], cohort["z"])
    np.testing.assert_allclose(pred["pz"].sum(1), 1.0, atol=2e-6, rtol=1e-5)
    assert np.isfinite(pred["logits"]).all() and np.isfinite(pred["pz"]).all()
    return pred


def predict_texts(model, tokenizer, device, texts, *, batch_size=8, max_length=512):
    from torch.utils.data import DataLoader

    from .qwen_posttraining import TextRedshiftDataset, evaluate_text_photoz_model, make_text_collator
    from .utils import make_redshift_grid

    loader = DataLoader(
        TextRedshiftDataset(list(texts), np.zeros(len(texts), dtype=np.float32), [str(i) for i in range(len(texts))]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_text_collator(tokenizer, max_length=max_length),
    )
    edges, centers = make_redshift_grid(0.0, 4.5, 300)
    out = evaluate_text_photoz_model(model, loader, device=device, redshift_edges=edges, redshift_centers=centers)
    return {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in out.items() if k in ("logits", "pz", "z_spec", "redshift_edges", "redshift_centers", "z_p16", "z_p50", "z_p84")}


def export_predictions(folder, cohort, predict_fn, *, chunk_size=256, extra=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(cohort["targetid"])
    z = np.asarray(cohort["z"], dtype=np.float32)
    hashes = {}
    for start in range(0, len(ids), chunk_size):
        end = min(start + chunk_size, len(ids))
        path = folder / f"{start:06d}_{end:06d}.npz"
        if path.exists():
            old = read(path)
            np.testing.assert_array_equal(old["targetid"], ids[start:end])
            np.testing.assert_array_equal(old["z_spec"], z[start:end])
            np.testing.assert_allclose(old["pz"].sum(1), 1.0, atol=2e-6, rtol=1e-5)
            assert np.isfinite(old["pz"]).all() and np.isfinite(old["logits"]).all()
            hashes[path.name] = sha(path)
            continue
        pred = predict_fn([str(t) for t in cohort["texts"][start:end]])
        payload = dict(targetid=ids[start:end], z_spec=z[start:end], logits=pred["logits"], pz=pred["pz"], z_p16=pred["z_p16"], z_p50=pred["z_p50"], z_p84=pred["z_p84"], redshift_edges=pred["redshift_edges"], redshift_centers=pred["redshift_centers"])
        if extra:
            payload.update({k: v[start:end] for k, v in extra.items()})
        tmp = path.with_name(path.name + ".partial.npz")
        np.savez_compressed(tmp, **payload)
        tmp.replace(path)
        hashes[path.name] = sha(path)
        write_json(folder / "progress.json", dict(completed=end, total=len(ids), sha256=hashes))
    complete = dict(n=int(len(ids)), chunk_size=int(chunk_size), prediction_sha256=hashes, targetid_sha256=hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest())
    write_json(folder / "complete.json", complete)
    return complete


def object_metrics(pred):
    z = np.asarray(pred["z_spec"], dtype=np.float64)
    pz = np.asarray(pred["pz"], dtype=np.float64)
    p16, p50, p84 = (np.asarray(pred[k], dtype=np.float64) for k in ("z_p16", "z_p50", "z_p84"))
    delta = (p50 - z) / (1.0 + z)
    target = z_bin(z)
    ce = -np.log(np.clip(pz[np.arange(len(z)), target], 1e-300, None))
    cdf = np.cumsum(pz, axis=1)
    observed = (CENTERS[None, :] >= z[:, None]).astype(np.float64)
    crps = np.sum((cdf - observed) ** 2, axis=1) * np.mean(np.diff(CENTERS))
    q025 = CENTERS[np.argmax(cdf >= 0.025, axis=1)]
    q975 = CENTERS[np.argmax(cdf >= 0.975, axis=1)]
    mass_above = np.sum(pz * np.clip((EDGES[1:] - np.maximum(EDGES[:-1], 0.6)) / np.diff(EDGES), 0, 1)[None, :], axis=1)
    entropy = -np.sum(pz * np.log(np.clip(pz, 1e-30, None)), axis=1)
    return dict(delta=delta, ce=ce, crps=crps, coverage68=((z >= p16) & (z <= p84)).astype(float), coverage95=((z >= q025) & (z <= q975)).astype(float), pit=cdf[np.arange(len(z)), target], width68=p84 - p16, width95=q975 - q025, mass_above=mass_above, entropy=entropy, p50=p50)


def metrics(pred, rows):
    o = object_metrics(pred) if "delta" not in pred else pred
    r = np.asarray(rows)
    delta = o["delta"][r]
    med = lower_median(delta)
    return dict(
        n=int(len(r)),
        mae=float(np.abs(delta).mean()),
        bias=float(delta.mean()),
        median_bias=med,
        nmad=float(1.48 * lower_median(np.abs(delta - med))),
        outlier=float((np.abs(delta) > 0.15).mean()),
        ce=float(o["ce"][r].mean()),
        crps=float(o["crps"][r].mean()),
        coverage68=float(o["coverage68"][r].mean()),
        coverage95=float(o["coverage95"][r].mean()),
        pit_mean=float(o["pit"][r].mean()),
        mean_width68=float(o["width68"][r].mean()),
        mean_width95=float(o["width95"][r].mean()),
        mean_entropy=float(o["entropy"][r].mean()),
        mean_mass_above_cutoff=float(o["mass_above"][r].mean()),
        median_predicted_z=lower_median(o["p50"][r]),
    )


def fixed_strata(z):
    out = {}
    for lo, hi in zip(FIXED_BINS[:-1], FIXED_BINS[1:]):
        out[f"[{lo:g},{hi:g})"] = np.flatnonzero((z >= lo) & (z < hi))
    return out


def stratified_bootstrap_indices(z, *, draws=1000, seed=1110, bins=FIXED_BINS):
    z = np.asarray(z)
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero((z >= lo) & (z < hi)) for lo, hi in zip(bins[:-1], bins[1:])]
    out = np.empty((draws, len(z)), dtype=np.int32)
    for d in range(draws):
        parts = [rng.choice(rows, size=len(rows), replace=True) for rows in strata if len(rows)]
        assert sum(len(p) for p in parts) == len(z)
        out[d] = np.concatenate(parts)
    return out


def paired_contrast(pred_a, pred_b, rows, draws, *, metrics_keys=("mae", "outlier", "nmad", "ce", "crps", "bias")):
    r = np.asarray(rows)
    oa, ob = object_metrics(pred_a), object_metrics(pred_b)
    point_a, point_b = metrics(oa, r), metrics(ob, r)
    point = {k: point_b[k] - point_a[k] for k in metrics_keys}
    boot = {k: [] for k in metrics_keys}
    mask = np.zeros(len(pred_a["z_spec"]), dtype=bool)
    mask[r] = True
    for draw in draws:
        # H1 convention: take the bootstrap values that land in the target
        # stratum, not the draw entries at the target positions.
        selected = np.asarray(draw)[mask[np.asarray(draw)]]
        ma, mb = metrics(oa, selected), metrics(ob, selected)
        for k in metrics_keys:
            boot[k].append(mb[k] - ma[k])
    ci = {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in boot.items()}
    return dict(point=point, ci95=ci, n=int(len(r)), draws=int(len(draws)), model_a_name=None, model_b_name=None)


def write_metrics_csv(path, rows, metrics_dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(metrics_dict)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "population", "bin"] + keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def calibrate_prediction(pred, temperature, prior_coeffs):
    pz = apply_calibration(pred["pz"], temperature, prior_coeffs)
    q = quantiles_from_pz(pz)
    out = dict(pred)
    out["pz"] = pz
    out["logits"] = np.log(np.clip(pz, 1e-30, None)).astype(np.float32)
    out["z_p16"] = q[0.16]
    out["z_p50"] = q[0.50]
    out["z_p84"] = q[0.84]
    return out
