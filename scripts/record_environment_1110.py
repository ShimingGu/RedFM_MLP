#!/usr/bin/env python3
"""Record environment, provenance and code snapshot for the 1110 campaign."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
SHARED = Path("/scratch/RedFM_MLP")
CODE = [
    "aion_magnitude/redshift_anchor_physics.py",
    "aion_magnitude/redshift_anchor_fit.py",
    "aion_magnitude/redshift_anchor_training.py",
    "scripts/e_membership_1110.py",
    "scripts/freeze_inputs_1110.py",
    "scripts/build_fsps_basis_1110.py",
    "scripts/fit_spectra_1110.py",
    "scripts/generate_1110.py",
    "scripts/build_training_data_1110.py",
    "scripts/train_anchor_1110.py",
    "scripts/supervise_1110.py",
    "scripts/checks_1110.py",
    "scripts/build_comparator_1110.py",
    "scripts/comparator_dev_1110.py",
    "scripts/check_synthetic_colors_1110.py",
    "scripts/preflight_training_1110.py",
    "scripts/finalize_phase1_1110.py",
    "aion_magnitude/redshift_anchor_comparator.py",
    "aion_magnitude/redshift_anchor_calibration.py",
    "aion_magnitude/redshift_anchor_evaluation.py",
    "scripts/calibrate_comparator_1110.py",
    "scripts/select_checkpoint_1110.py",
    "scripts/calibrate_main_1110.py",
    "scripts/build_direct_comparator_data_1110.py",
    "scripts/train_direct_comparator_1110.py",
    "scripts/evaluate_frozen_1110.py",
    "scripts/supervise_post_1110.py",
    "scripts/supervise_report_1110.py",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    env = dict(
        created_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        worktree=str(ROOT),
        branch=git("branch", "--show-current"),
        commit=git("rev-parse", "HEAD"),
        base_commit="0fb1e01360759aa840fe52cec176e68467462c78",
        main_checkout=dict(path=str(SHARED), branch=git("branch", "--show-current", cwd=SHARED), commit=git("rev-parse", "HEAD", cwd=SHARED), dirty=bool(git("status", "--porcelain", cwd=SHARED))),
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        packages={name: version(name) for name in ("torch", "transformers", "peft", "bitsandbytes", "numpy", "scipy", "h5py", "astropy", "fsps", "dust_extinction", "accelerate", "tokenizers", "safetensors")},
        cuda=dict(available=__import__("torch").cuda.is_available(), device_count=__import__("torch").cuda.device_count()),
        gpu_device=os.environ.get("CUDA_VISIBLE_DEVICES"),
        training_env=str(ROOT / ".training-env"),
        physics_env=str(ROOT / ".physics-env"),
        base_python="/arc/home/gsm/.astroai/lab/saves/RedFM_MLP/.pixi/envs/default/bin/python",
        fsps=dict(sps_home=str(OUT / "physics/fsps"), commit=(OUT / "physics/fsps_commit.txt").read_text().strip(), version=version("fsps")),
        models=dict(base="/arc/home/gsm/hf_models/Qwen3.5-4B-Base", h1a_final=str(OUT.parent.parent / "1102/highz-hsc-pilot-zcut0p6/training/H1-A/final")),
        isolation=dict(main_checkout_preserved=True, other_worktrees_preserved=True, gpu_only="MIG-ed1476fa-775b-52cc-85d6-6b7ad1ef2323"),
    )
    (OUT / "protocol").mkdir(parents=True, exist_ok=True)
    (OUT / "protocol/environment.json").write_text(json.dumps(env, indent=2) + "\n")
    snapshot = OUT / "protocol/code_snapshot"
    hashes = {}
    for rel in CODE:
        src = ROOT / rel
        dest = snapshot / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        hashes[rel] = sha(src)
    frozen = json.loads((OUT / "manifest.json").read_text())
    provenance = dict(
        environment=str(OUT / "protocol/environment.json"),
        environment_sha256=sha(OUT / "protocol/environment.json"),
        code_sha256=hashes,
        frozen_inputs=str(OUT / "manifest.json"),
        frozen_inputs_sha256=sha(OUT / "manifest.json"),
        frozen_assets=frozen["frozen_sha256"],
        E_membership=str(OUT / "protocol/ledgers/E_membership.json"),
        anchors=str(OUT / "data/frozen/anchors.npz"),
        lowz_subset=str(OUT / "data/frozen/lowz_fit_subset.npz"),
        spectra_sha256="26a0c4be519d21207c0976e2b8e664b29a625aada7b1e003ac14faeada8c9f33",
        h1=dict(manifest=str(OUT.parent.parent / "1102/highz-hsc-pilot-zcut0p6/manifest.json"), commit="0fb1e01360759aa840fe52cec176e68467462c78"),
        policy=dict(
            highz_budget=256,
            fitting=192,
            development=64,
            lowz_spectra=2048,
            lowz_source=10489,
            P_E_sealed=True,
            development_spectra_never_read=True,
            oracle_redshifts="only T stratum aggregates and selected identities; no P/E/V-highz",
        ),
    )
    (OUT / "protocol/provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
