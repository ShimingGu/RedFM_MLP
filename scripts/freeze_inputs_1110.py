#!/usr/bin/env python3
"""Freeze low-z source, anchor roles and ledgers before any new spectrum fit.

All identities are fixed from the archived H1 split and the T catalogue.  Anchor
selection uses only T spectroscopic redshifts for the predeclared strata; P, E
and high-z V are never consulted here.  If the campaign manifest already exists
this script verifies its frozen memberships instead of overwriting them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
ARC = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints")
H1 = ARC / "1102/highz-hsc-pilot-zcut0p6"
PAIRED = ARC / "1028/matched-training-design/paired_split/paired_cohort.npz"
OUT = ARC / "1110/redshift-anchor-recovery"
SEED = 1110
LOWZ_N = 2048
STRATA = [(0.6, 0.8, 48, 16), (0.8, 1.0, 80, 16), (1.0, 1.5, 64, 32)]
NAMES = [f"HSC_PDR2_cmodel_{b}_{s}" for b in "grizy" for s in ("flux", "flux_error")] + ["MW_EBV"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def kmeans(x, k, seed, iterations=40):
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    d2 = ((x - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = np.maximum(d2, 0)
        if p.sum() == 0:
            p = np.ones_like(p)
        centers.append(x[rng.choice(len(x), p=p / p.sum())])
        d2 = np.minimum(d2, ((x - centers[-1]) ** 2).sum(1))
    centers = np.asarray(centers)
    labels = np.zeros(len(x), dtype=np.int64)
    for _ in range(iterations):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new = dist.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = labels == j
            if members.any():
                centers[j] = x[members].mean(0)
    return labels, centers


def allocate(sizes, total):
    n = sizes.sum()
    raw = total * sizes / n
    quota = np.floor(raw).astype(int)
    quota = np.minimum(quota, sizes)
    remaining = total - quota.sum()
    order = np.argsort(-(raw - quota), kind="stable")
    while remaining > 0:
        progressed = False
        for j in order:
            if remaining == 0:
                break
            if quota[j] < sizes[j]:
                quota[j] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("Cannot allocate low-z quotas")
    return quota


def lowz_selection(paired, lowz):
    rows = np.flatnonzero(paired["category"] == "T")
    row_of = {int(t): int(r) for t, r in zip(paired["targetid"][rows], rows)}
    idx = np.array([row_of[int(t)] for t in lowz["targetid"]], dtype=np.int64)
    f = paired["hsc_features"][idx].astype(np.float64)
    flux = f[:, 0::2]
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(flux > 0, 22.5 - 2.5 * np.log10(flux), np.nan)
    colors = mag[:, :-1] - mag[:, 1:]
    feature = np.column_stack([colors, mag[:, 2]])
    good = np.isfinite(feature).all(1)
    if not good.all():
        med = np.nanmedian(feature[good], axis=0)
        feature[~good] = med
    scale = np.percentile(feature, 75, axis=0) - np.percentile(feature, 25, axis=0)
    scale[scale <= 0] = 1.0
    zscore = (feature - np.median(feature, axis=0)) / scale
    labels, _ = kmeans(zscore, 64, SEED + 1)
    sizes = np.bincount(labels, minlength=64)
    quota = allocate(sizes, LOWZ_N)
    parts = []
    for j in range(64):
        members = np.flatnonzero(labels == j)
        # Deterministic within-cluster order by targetid hash, not photometric residual.
        keys = np.array([int.from_bytes(hashlib.sha256(f"1110-lowz:{int(lowz['targetid'][i])}".encode()).digest()[:8], "big") for i in members], dtype=np.uint64)
        parts.append(members[np.argsort(keys, kind="stable")[: quota[j]]])
    selected = np.sort(np.concatenate(parts))
    prob = np.zeros(len(lowz["targetid"]))
    for j in range(64):
        members = np.flatnonzero(labels == j)
        if len(members):
            prob[members] = quota[j] / len(members)
    return selected, labels, prob, sizes, quota, idx, feature


def main():
    frozen_manifest = OUT / "manifest.json"
    if frozen_manifest.exists():
        manifest = json.loads(frozen_manifest.read_text())
        assert manifest["seed"] == SEED and manifest["anchor_budget"] == 256
        for path, digest in manifest["frozen_sha256"].items():
            assert sha(path) == digest, f"Frozen file changed: {path}"
        print("FROZEN_INPUTS_ALREADY_PRESENT", json.dumps(manifest["frozen_sha256"], indent=1))
        return

    h1_manifest = json.loads((H1 / "manifest.json").read_text())
    for name in ("lowz_only_train.npz", "lowz_only_membership.npz", "lowz_only_validation.npz"):
        path = H1 / "splits" / name
        assert sha(path) == h1_manifest["assets_sha256"][str(path)], path
    paired = read(PAIRED)
    lowz = read(H1 / "splits/lowz_only_train.npz")
    membership = read(H1 / "splits/lowz_only_membership.npz")
    assert len(lowz["targetid"]) == len(np.unique(lowz["targetid"])) == 10489
    assert np.array_equal(lowz["targetid"], membership["train_targetid"])
    assert sha_array(lowz["targetid"]) == sha_array(membership["train_targetid"])

    trows = np.flatnonzero(paired["category"] == "T")
    tids = paired["targetid"][trows]
    tz = paired["z"][trows]
    rng = np.random.default_rng(SEED)
    selected_rows = []
    role = []
    stratum = []
    inclusion = []
    fit_counts = []
    dev_counts = []
    for lo, hi, nfit, ndev in STRATA:
        pool = np.flatnonzero((tz >= lo) & (tz < hi))
        n = nfit + ndev
        assert len(pool) >= n, (lo, hi, len(pool), n)
        take = rng.choice(pool, size=n, replace=False)
        take = take[np.argsort(tz[take], kind="stable")]
        role.extend(["fitting"] * nfit + ["development"] * ndev)
        selected_rows.extend(trows[take])
        stratum.extend([f"[{lo},{hi})"] * n)
        inclusion.extend([n / len(pool)] * n)
        fit_counts.append(nfit)
        dev_counts.append(ndev)
    selected_rows = np.asarray(selected_rows)
    anchors = dict(
        targetid=paired["targetid"][selected_rows],
        z=paired["z"][selected_rows],
        features=paired["hsc_features"][selected_rows],
        hsc_valid=paired["hsc_valid"][selected_rows],
        row=selected_rows,
        role=np.asarray(role),
        stratum=np.asarray(stratum),
        inclusion_probability=np.asarray(inclusion, dtype=np.float64),
    )
    assert len(np.unique(anchors["targetid"])) == 256
    assert (anchors["z"] >= 0.6).all()
    assert not np.intersect1d(anchors["targetid"], lowz["targetid"]).size

    lowz_sel, labels, prob, sizes, quota, idx, feature = lowz_selection(paired, lowz)
    lowz_subset = dict(
        targetid=lowz["targetid"][lowz_sel],
        z=lowz["z"][lowz_sel],
        features=paired["hsc_features"][idx[lowz_sel]],
        cluster=labels[lowz_sel],
        inclusion_probability=prob[lowz_sel],
    )
    assert len(lowz_subset["targetid"]) == LOWZ_N == len(np.unique(lowz_subset["targetid"]))
    assert not np.intersect1d(lowz_subset["targetid"], anchors["targetid"]).size

    data = OUT / "data/frozen"
    data.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data / "anchors.npz", **anchors)
    np.savez_compressed(data / "lowz_fit_subset.npz", **lowz_subset)
    np.savez_compressed(
        data / "lowz_source.npz",
        targetid=lowz["targetid"],
        z=lowz["z"],
        source_path=str(H1 / "splits/lowz_only_train.npz"),
        membership_path=str(H1 / "splits/lowz_only_membership.npz"),
    )

    parent_ledger = dict(
        paired_catalogue=str(PAIRED),
        paired_catalogue_sha256=sha(PAIRED),
        counts={c: int((paired["category"] == c).sum()) for c in "TVP"},
        t_strata={f"[{lo},{hi})": int(((tz >= lo) & (tz < hi)).sum()) for lo, hi, _, _ in STRATA}
        | {"[1.5,4.5)": int(((tz >= 1.5) & (tz < 4.5)).sum()), ">=4.5": int((tz >= 4.5).sum())},
        label_use="only T spectroscopic redshifts for fixed stratification; P, E and high-z V excluded from selection and development",
        E_membership=str(OUT / "data/frozen/E_membership.npz"),
        E_counts=json.loads((OUT / "protocol/ledgers/E_membership.json").read_text())["counts"],
    )
    oracle = dict(
        seed=SEED,
        rule="uniform without replacement within each predeclared T redshift stratum; roles assigned before any fitted spectrum is read",
        strata=[dict(z_low=lo, z_high=hi, available=int(((tz >= lo) & (tz < hi)).sum()), fitting=nfit, development=ndev, inclusion_probability=(nfit + ndev) / int(((tz >= lo) & (tz < hi)).sum())) for lo, hi, nfit, ndev in STRATA],
        fitting=int((anchors["role"] == "fitting").sum()),
        development=int((anchors["role"] == "development").sum()),
        selected_total=256,
        selected_targetid_sha256=sha_array(anchors["targetid"]),
        per_stratum_counts={s: dict(fitting=int(((anchors["stratum"] == s) & (anchors["role"] == "fitting")).sum()), development=int(((anchors["stratum"] == s) & (anchors["role"] == "development")).sum())) for s in sorted(set(anchors["stratum"]))},
        weights="development population weights under uniform selection = parent stratum count / development count; fitting weights likewise = parent stratum count / fitting count; report effective sample sizes",
        unselected_redshifts="only aggregate stratum counts inspected; no individual unselected redshift used",
    )
    fitting = dict(
        budget=256,
        fitting=192,
        development=64,
        cumulative_across_restarts=256,
        roles=[dict(stratum=s, fitting=int((anchors["role"] == "fitting").sum()), development=int((anchors["role"] == "development").sum()), targetid_sha256=sha_array(anchors["targetid"][anchors["stratum"] == s])) for s in sorted(set(anchors["stratum"]))],
        lowz_source_total=10489,
        lowz_spectra_planned=LOWZ_N,
        note="development objects count toward the 256-object budget even when only used for checkpoint choice or calibration",
    )
    lowz_ledger = dict(
        source_train_path=str(H1 / "splits/lowz_only_train.npz"),
        source_train_sha256=sha(H1 / "splits/lowz_only_train.npz"),
        membership_path=str(H1 / "splits/lowz_only_membership.npz"),
        membership_sha256=sha(H1 / "splits/lowz_only_membership.npz"),
        lowz_validation_path=str(H1 / "splits/lowz_only_validation.npz"),
        lowz_validation_sha256=sha(H1 / "splits/lowz_only_validation.npz"),
        n_source=10489,
        n_fit_subset=LOWZ_N,
        selection="k-means (64 clusters, seed 1111) on grizy colors and i magnitude from permitted T photometry; proportional quota 2048, deterministic targetid-hash order within cluster",
        cluster_sizes=sizes.tolist(),
        cluster_quota=quota.tolist(),
        fit_targetid_sha256=sha_array(lowz_subset["targetid"]),
        spectra_consumed=0,
        note="source-training spectra only; low-z V remains development data and is never read for the generator",
    )
    aux = dict(
        h1_manifest=str(H1 / "manifest.json"),
        h1_manifest_sha256=sha(H1 / "manifest.json"),
        h1_execution_commit=h1_manifest["execution_commit"],
        h1_protocol_config=str(H1 / "protocol/configs/H1-A.json"),
        h1_protocol_config_sha256=sha(H1 / "protocol/configs/H1-A.json"),
        h1_frozen_checkpoints=str(H1 / "evaluation/final_checkpoints_frozen.json"),
        h1_frozen_checkpoints_sha256=sha(H1 / "evaluation/final_checkpoints_frozen.json"),
        h1_results=str(H1 / "evaluation/results.json"),
        h1_results_sha256=sha(H1 / "evaluation/results.json"),
        parent_cohort=str(PAIRED),
        parent_cohort_sha256=sha(PAIRED),
        hsc_parent=str(ARC / "1102/highz-feasibility/raw/hsc_parent/cohort.npz"),
        hsc_parent_sha256=sha(ARC / "1102/highz-feasibility/raw/hsc_parent/cohort.npz"),
        hsc_flags=str(ARC / "1102/highz-feasibility/flags/hsc_parent.npz"),
        hsc_flags_sha256=sha(ARC / "1102/highz-feasibility/flags/hsc_parent.npz"),
        spectra=str(ARC.parent / "GalaxiesMLSpec/DESI_HSC_64x64_v2.hdf5"),
        spectra_sha256="26a0c4be519d21207c0976e2b8e664b29a625aada7b1e003ac14faeada8c9f33",
        spectra_manifest_sha256=sha(ARC.parent / "GalaxiesMLSpec/download_manifest.json"),
        e_membership_sha256=sha(OUT / "data/frozen/E_membership.npz"),
        e_counts=parent_ledger["E_counts"],
        exposure_audit="H1-A initialisation contains no high-z post-training labels; H1-B/C/E/F and zcut=.8 are excluded from initialisation, teaching, pseudo-labelling and calibration",
        unlabelled_data="T photometry only informs noise/depth/missingness, coverage and declared population corrections; no P/E photometry fit",
    )
    led = OUT / "protocol/ledgers"
    write_json(led / "parent_catalogue.json", parent_ledger)
    write_json(led / "oracle_screening.json", oracle)
    write_json(led / "fitting_development.json", fitting)
    write_json(led / "lowz_source.json", lowz_ledger)
    write_json(led / "auxiliary.json", aux)
    write_json(led / "spectrum_access.json", dict(reads=[], n_lowz=0, n_anchor_fitting=0, n_anchor_development=0, n_unselected=0, note="append one record per permitted spectrum read"))

    frozen_sha = {
        str(data / "anchors.npz"): sha(data / "anchors.npz"),
        str(data / "lowz_fit_subset.npz"): sha(data / "lowz_fit_subset.npz"),
        str(data / "lowz_source.npz"): sha(data / "lowz_source.npz"),
        str(data / "E_membership.npz"): sha(data / "E_membership.npz"),
    }
    manifest = dict(
        version="redshift-anchor-1110-v1",
        created_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        worktree=str(ROOT),
        branch="redshift/anchors-1110",
        base_commit="0fb1e01360759aa840fe52cec176e68467462c78",
        output_root=str(OUT),
        seed=SEED,
        anchor_budget=256,
        anchor_fitting=192,
        anchor_development=64,
        lowz_source=10489,
        lowz_fit_subset=LOWZ_N,
        E_objects=int(json.loads((OUT / "protocol/ledgers/E_membership.json").read_text())["n"]),
        frozen_sha256=frozen_sha,
        ledger_sha256={str(p): sha(p) for p in sorted(led.glob("*.json"))},
    )
    write_json(frozen_manifest, manifest)
    print("FROZEN", json.dumps(dict(anchors={s: oracle["per_stratum_counts"][s] for s in oracle["per_stratum_counts"]}, lowz=LOWZ_N, E=manifest["E_objects"]), indent=2))


if __name__ == "__main__":
    main()
