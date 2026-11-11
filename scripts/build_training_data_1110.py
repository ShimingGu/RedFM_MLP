#!/usr/bin/env python3
"""Assemble the single-pass mixed training pool, fixed sequence and validation set."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
H1 = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1102/highz-hsc-pilot-zcut0p6")
sys.path.insert(0, str(ROOT))

from aion_magnitude.redshift_anchor_physics import serialize_features  # noqa: E402

SEED = 1110
HEAD_PRESENTATIONS = 10240
JOINT_PRESENTATIONS = 89760
COUNTS = {
    "head": (5120, 3072, 2048),
    "joint": (44880, 26928, 17952),
}
WEIGHTS = np.array([0.5, 0.3, 0.2], dtype=np.float64)


def read(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    synthetic = read(OUT / "data/synthetic/synthetic.npz")
    lowz = read(OUT / "data/frozen/lowz_source.npz")
    h1low = read(H1 / "splits/lowz_only_train.npz")
    anchors = read(OUT / "data/frozen/anchors.npz")
    fit_anchor = anchors["role"] == "fitting"
    dev_anchor = anchors["role"] == "development"
    assert np.array_equal(lowz["targetid"], h1low["targetid"])
    assert len(h1low["text"]) == 10489
    n_syn = len(synthetic["z"])
    n_low = len(lowz["targetid"])
    n_anchor = int(fit_anchor.sum())
    assert n_syn >= 50000 and n_low == 10489 and n_anchor == 192
    assert not np.intersect1d(lowz["targetid"], anchors["targetid"][fit_anchor]).size

    pool_text, pool_z, pool_id, pool_component, pool_parent = [], [], [], [], []
    for i in range(n_syn):
        pool_text.append(str(synthetic["text"][i]))
        pool_z.append(float(synthetic["z"][i]))
        pool_id.append(f"syn-{-int(synthetic['synthetic_id'][i]):06d}")
        pool_component.append(0)
        pool_parent.append(int(synthetic["targetid"][i]))
    for i in range(n_low):
        pool_text.append(str(h1low["text"][i]))
        pool_z.append(float(h1low["z"][i]))
        pool_id.append(str(int(lowz["targetid"][i])))
        pool_component.append(1)
        pool_parent.append(int(lowz["targetid"][i]))
    for i in np.flatnonzero(fit_anchor):
        pool_text.append(serialize_features(anchors["features"][i], ROOT))
        pool_z.append(float(anchors["z"][i]))
        pool_id.append(str(int(anchors["targetid"][i])))
        pool_component.append(2)
        pool_parent.append(int(anchors["targetid"][i]))
    pool_text = np.array(pool_text, dtype=str)
    pool_z = np.array(pool_z, dtype=np.float32)
    pool_id = np.array(pool_id, dtype=str)
    pool_component = np.array(pool_component, dtype=np.int8)
    pool_parent = np.array(pool_parent, dtype=np.int64)

    rng = np.random.default_rng(SEED + 50)
    seq_parts, phase_parts = [], []
    for phase, start in (("head", 0), ("joint", n_syn + n_low)):
        ns, nl, na = COUNTS[phase]
        syn_idx = rng.choice(n_syn, size=ns, replace=False)
        low_idx = rng.choice(n_low, size=nl, replace=True) + n_syn
        anchor_idx = rng.choice(n_anchor, size=na, replace=True) + n_syn + n_low
        part = np.concatenate([syn_idx, low_idx, anchor_idx]).astype(np.int64)
        rng.shuffle(part)
        seq_parts.append(part)
        phase_parts.append(np.full(len(part), 0 if phase == "head" else 1, dtype=np.int8))
    order = np.concatenate(seq_parts)
    phase = np.concatenate(phase_parts)
    assert len(order) == 100000 == HEAD_PRESENTATIONS + JOINT_PRESENTATIONS
    comp = pool_component[order]
    counts = {phase_name: np.bincount(comp[phase == (0 if phase_name == "head" else 1)], minlength=3).tolist() for phase_name in ("head", "joint")}

    # Validation: real low-z V (development) plus all 64 budgeted development anchors.
    lowv = read(H1 / "splits/lowz_only_validation.npz")
    val_text = [str(x) for x in lowv["text"]]
    val_z = [float(x) for x in lowv["z"]]
    val_id = [str(int(x)) for x in lowv["targetid"]]
    val_group = ["lowz_V"] * len(lowv["text"])
    val_stratum = ["[0,0.6)"] * len(lowv["text"])
    for i in np.flatnonzero(dev_anchor):
        val_text.append(serialize_features(anchors["features"][i], ROOT))
        val_z.append(float(anchors["z"][i]))
        val_id.append(str(int(anchors["targetid"][i])))
        val_group.append("dev_anchor")
        val_stratum.append(str(anchors["stratum"][i]))

    data_dir = OUT / "data/training"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_dir / "pool.npz",
        text=pool_text,
        z=pool_z,
        object_id=pool_id,
        component=pool_component,
        parent_targetid=pool_parent,
    )
    np.savez_compressed(data_dir / "sequence.npz", pool_index=order, phase=phase, component=comp)
    np.savez_compressed(
        data_dir / "validation.npz",
        text=np.array(val_text, dtype=str),
        z=np.array(val_z, dtype=np.float32),
        object_id=np.array(val_id, dtype=str),
        group=np.array(val_group, dtype=str),
        stratum=np.array(val_stratum, dtype=str),
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/arc/home/gsm/hf_models/Qwen3.5-4B-Base", local_files_only=True)
    sample = list(range(0, len(pool_text), max(1, len(pool_text) // 2000)))
    token_lengths = [len(tokenizer([pool_text[i]], truncation=False)["input_ids"][0]) for i in sample]
    assert max(token_lengths) <= 512, max(token_lengths)
    meta = dict(
        token_audit=dict(sampled=len(token_lengths), max_tokens=int(max(token_lengths)), min_tokens=int(min(token_lengths)), truncated=0),
        created_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        seed=SEED,
        presentations=100000,
        head_presentations=HEAD_PRESENTATIONS,
        joint_presentations=JOINT_PRESENTATIONS,
        component_weights=dict(synthetic=0.5, lowz_source=0.3, anchor_fitting=0.2),
        component_order=["synthetic", "lowz_source", "anchor_fitting"],
        phase_component_counts=counts,
        pool_sizes=dict(synthetic=n_syn, lowz_source=n_low, anchor_fitting=n_anchor),
        sequence_counts={k: int((comp == i).sum()) for i, k in enumerate(["synthetic", "lowz_source", "anchor_fitting"])},
        unique_synthetic_used=int(len(np.unique(order[comp == 0]))),
        validation=dict(lowz_V=len(lowv["text"]), dev_anchor=int(dev_anchor.sum())),
        hashes={name: sha(data_dir / name) for name in ("pool.npz", "sequence.npz", "validation.npz")},
        firewall="pool excludes P, E, high-z V and all 64 development anchors; validation is development-only",
    )
    (data_dir / "training_data.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
