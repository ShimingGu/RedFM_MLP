#!/usr/bin/env python3
"""Preflight: H1-A replay, head reinitialisation, finite gradients, state reload."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/scratch/RedFM_MLP/redshift-anchor-1110-worktree")
OUT = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1110/redshift-anchor-recovery")
H1 = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1102/highz-hsc-pilot-zcut0p6")
sys.path.insert(0, str(ROOT))

from aion_magnitude.matched_training import state_fingerprint  # noqa: E402
from aion_magnitude.metrics import discrete_crps, redshift_cross_entropy_loss  # noqa: E402
from aion_magnitude.qwen_posttraining import TextRedshiftDataset, make_text_collator  # noqa: E402
from aion_magnitude.redshift_anchor_training import build_model, reinit_final_head, replay_check, restore_h1a  # noqa: E402
from aion_magnitude.utils import make_redshift_grid  # noqa: E402


def main():
    started = time.time()
    cfg = json.loads((H1 / "protocol/configs/H1-A.json").read_text())
    training = dict(cfg["training"], z_min=0.0, z_max=4.5, n_z_bins=300, seed=1110, device="cuda")
    model, tokenizer, device, qcfg = build_model(training)
    gate = json.loads((H1 / "evaluation/final_checkpoints_frozen.json").read_text())
    state_sha = gate["models"]["H1-A"]["primary_trainable_state_sha256"]
    saved = restore_h1a(model, H1 / "training/H1-A/final/trainable_state.pt", state_sha)
    replay = replay_check(model, tokenizer, device, H1 / "splits/lowz_only_validation.npz", H1 / "training/H1-A/final/validation_predictions.pt", qcfg)
    head_sha = reinit_final_head(model, 1117)

    data = np.load(H1 / "splits/lowz_only_validation.npz", allow_pickle=False)
    texts = [str(x) for x in data["text"][:8]]
    z = torch.as_tensor(data["z"][:8], dtype=torch.float32, device=device)
    ids = [str(x) for x in data["targetid"][:8]]
    collate = make_text_collator(tokenizer, max_length=qcfg.max_length)
    edges, centers = make_redshift_grid(0.0, 4.5, 300)
    edges, centers = edges.to(device), centers.to(device)
    model.train()
    for p in model.qwen.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
    model.freeze_qwen_representation = True
    model.qwen.eval()
    loss = torch.zeros((), device=device)
    ce = torch.zeros((), device=device)
    crps = torch.zeros((), device=device)
    dataset = TextRedshiftDataset(texts, data["z"][:8], ids)
    for i in range(4):
        batch = collate([dataset[i]])
        batch.pop("object_id")
        z = batch.pop("z_spec").to(device)
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**inputs)
        ce_i = redshift_cross_entropy_loss(logits.float(), z, edges)
        pz = torch.softmax(logits.float(), -1)
        crps_i = discrete_crps(pz, z, centers).mean()
        loss = loss + (ce_i + 0.5 * crps_i) / 4.0
        ce = ce + ce_i / 4.0
        crps = crps + crps_i / 4.0
    loss.backward()
    head_norm = torch.nn.utils.clip_grad_norm_(list(model.photoz_head.parameters()), 1.0)
    finite = bool(torch.isfinite(loss) and torch.isfinite(head_norm))
    assert finite
    assert float(head_norm) > 0
    before = {k: p.detach().cpu().clone() for k, p in model.named_parameters() if "photoz_head" in k}
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        batch = collate([dataset[0]])
        batch.pop("object_id")
        batch.pop("z_spec")
        inputs = {k: v.to(device) for k, v in batch.items()}
        logits_before = model(**inputs).float().cpu()
    # Round-trip all trainable tensors through a state dict.
    state = {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        current = dict(model.named_parameters())
        for k, v in state.items():
            current[k].copy_(v)
    with torch.no_grad():
        logits_after = model(**inputs).float().cpu()
    reload_diff = float((logits_before - logits_after).abs().max())
    assert reload_diff < 1e-6, reload_diff
    result = dict(
        h1a_replay_max_abs_diff=replay,
        head_reinit_state_sha256=head_sha,
        loss=float(loss), ce=float(ce), crps=float(crps),
        head_grad_norm=float(head_norm), finite_gradients=finite,
        state_reload_logit_diff=reload_diff,
        gpu=torch.cuda.get_device_name(0),
        memory_allocated_gib=torch.cuda.memory_allocated() / 2**30,
        memory_reserved_gib=torch.cuda.memory_reserved() / 2**30,
        elapsed_seconds=time.time() - started,
    )
    path = OUT / "protocol/checks_training_preflight.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
