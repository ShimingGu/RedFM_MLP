"""Resumable mixed real/synthetic QLoRA loop for the 1110 campaign."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .matched_training import export_model, rng_state, restore_rng, save_torch, state_fingerprint
from .metrics import discrete_crps, predict_photoz_from_logits, redshift_cross_entropy_loss
from .qwen_posttraining import (
    QwenPosttrainingConfig,
    TextRedshiftDataset,
    _linear_warmup_decay_lambda,
    create_qlora_photoz_model,
    evaluate_text_photoz_model,
    make_text_collator,
)
from .utils import make_redshift_grid, set_random_seed


def now():
    return datetime.now(timezone.utc).isoformat()


def _sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def model_state(model):
    return {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}


def build_model(training_cfg):
    config = QwenPosttrainingConfig(**training_cfg).normalized()
    set_random_seed(config.seed + 1110)
    model, tokenizer, device = create_qlora_photoz_model(config, use_dora=False)
    return model, tokenizer, device, config


def restore_h1a(model, path, expected_file_sha256=None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    current = dict(model.named_parameters())
    assert set(saved) == {k for k, p in model.named_parameters() if p.requires_grad}, (
        len(saved), len([k for k, p in model.named_parameters() if p.requires_grad])
    )
    with torch.no_grad():
        for k, v in saved.items():
            assert current[k].shape == v.shape, (k, current[k].shape, v.shape)
            current[k].copy_(v.to(current[k].device))
    if expected_file_sha256 is not None:
        actual_file = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert actual_file == expected_file_sha256, (actual_file, expected_file_sha256)
    return saved


def reinit_final_head(model, seed):
    layer = model.photoz_head.net[6]
    gen = torch.Generator(device=layer.weight.device).manual_seed(int(seed))
    with torch.no_grad():
        layer.weight.normal_(0.0, 0.02, generator=gen)
        layer.bias.zero_()
    return state_fingerprint({k: p.detach().cpu() for k, p in model.named_parameters() if p.requires_grad and "photoz_head" in k})


def replay_check(model, tokenizer, device, validation_path, predictions_path, config):
    data = np.load(validation_path, allow_pickle=False)
    texts = [str(x) for x in data["text"][:64]]
    z = torch.as_tensor(data["z"][:64], dtype=torch.float32)
    ids = [str(x) for x in data["targetid"][:64]]
    loader = DataLoader(
        TextRedshiftDataset(texts, z, ids),
        batch_size=8,
        shuffle=False,
        collate_fn=make_text_collator(tokenizer, max_length=config.max_length),
    )
    edges, centers = make_redshift_grid(0.0, 4.5, 300)
    model.eval()
    out = evaluate_text_photoz_model(model, loader, device=device, redshift_edges=edges, redshift_centers=centers)
    ref = torch.load(predictions_path, map_location="cpu", weights_only=False)
    diffs = {}
    for key in ("logits", "pz", "z_p50"):
        diffs[key] = float((out[key] - ref[key][:64]).abs().max())
    assert max(diffs.values()) < 1e-4, diffs
    return diffs


def _metrics(z_spec, z_p50, logits, pz, edges, centers):
    z_spec = np.asarray(z_spec, float)
    z_p50 = np.asarray(z_p50, float)
    delta = (z_p50 - z_spec) / (1.0 + z_spec)
    return dict(
        n=int(len(z_spec)),
        mae=float(np.mean(np.abs(delta))),
        bias=float(np.mean(delta)),
        median_bias=float(np.median(delta)),
        outlier=float(np.mean(np.abs(delta) > 0.15)),
        nmad=float(1.4826 * np.median(np.abs(delta - np.median(delta)))),
        ce=float(redshift_cross_entropy_loss(torch.as_tensor(logits), torch.as_tensor(z_spec, dtype=torch.float32), edges=edges).item()),
        crps=float(discrete_crps(torch.as_tensor(pz), torch.as_tensor(z_spec, dtype=torch.float32), centers=centers).mean().item()),
    )


def evaluate_development(model, tokenizer, device, data, config, batch_size=8):
    edges, centers = make_redshift_grid(0.0, 4.5, 300)
    loader = DataLoader(
        TextRedshiftDataset([str(x) for x in data["text"]], data["z"], [str(x) for x in data["object_id"]]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_text_collator(tokenizer, max_length=config.max_length),
    )
    model.eval()
    with torch.no_grad():
        out = evaluate_text_photoz_model(model, loader, device=device, redshift_edges=edges, redshift_centers=centers)
    pz = out["pz"].float()
    z50 = out["z_p50"].float()
    z = out["z_spec"].float()
    p16, p84 = out["z_p16"].float(), out["z_p84"].float()
    coverage = ((z >= p16) & (z <= p84)).float().mean().item()
    groups = np.asarray(data["group"]).astype(str)
    strata = np.asarray(data["stratum"]).astype(str)
    result = dict(
        overall=_metrics(z, z50, out["logits"], pz, edges, centers) | dict(coverage68=coverage),
        by_group={},
        by_stratum={},
    )
    for name in sorted(set(groups)):
        m = groups == name
        result["by_group"][name] = _metrics(z[m], z50[m], out["logits"][m], pz[m], edges, centers)
    for name in sorted(set(strata)):
        m = strata == name
        result["by_stratum"][name] = _metrics(z[m], z50[m], out["logits"][m], pz[m], edges, centers)
    macro = []
    for name in ("[0.8,1.0)", "[1.0,1.5)"):
        if name in result["by_stratum"] and result["by_stratum"][name]["n"] >= 8:
            macro.append(result["by_stratum"][name]["mae"])
    result["macro_mae_primary"] = float(np.mean(macro)) if macro else float("inf")
    low = result["by_group"].get("lowz_V", {})
    penalty = 0.0
    if low:
        penalty += 2.0 * max(0.0, low["mae"] - 0.035)
        penalty += 2.0 * max(0.0, abs(low["bias"]) - 0.01)
        penalty += 1.0 * max(0.0, low["nmad"] - 0.040)
    result["selection_score"] = float(result["macro_mae_primary"] + penalty)
    return result


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def train(config, output_dir, data_dir, *, resume=False):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)
    pool = np.load(data_dir / "pool.npz", allow_pickle=False)
    seq = np.load(data_dir / "sequence.npz", allow_pickle=False)
    val = np.load(data_dir / "validation.npz", allow_pickle=False)
    sequence_sha = hashlib.sha256(np.ascontiguousarray(seq["pool_index"]).tobytes()).hexdigest()
    order = seq["pool_index"].astype(np.int64)
    component = seq["component"].astype(np.int64)
    phase = seq["phase"].astype(np.int64)
    total = int(config["presentations"])
    head_only = int(config["head_only_presentations"])
    accum = int(config.get("accumulation", 16))
    weights = np.asarray(config["component_weights"], dtype=np.float64)
    assert len(order) == total and len(component) == total
    assert head_only % accum == 0 and total % accum == 0
    assert np.allclose(weights, [0.5, 0.3, 0.2])

    pool_text = pool["text"][:]
    pool_z = pool["z"][:]
    pool_id = pool["object_id"][:]
    texts = [str(pool_text[i]) for i in order]
    redshifts = pool_z[order]
    ids = [str(pool_id[i]) for i in order]
    dataset = TextRedshiftDataset(texts, redshifts, ids)

    model, tokenizer, device, qcfg = build_model(config["training"])
    collate = make_text_collator(tokenizer, max_length=qcfg.max_length)
    edges, centers = make_redshift_grid(0.0, 4.5, 300)
    edges = edges.to(device)
    centers = centers.to(device)

    head_params = list(model.photoz_head.parameters())
    adapter_params = [p for p in model.qwen.parameters() if p.requires_grad]
    assert adapter_params and head_params
    names = {k for k, p in model.named_parameters() if p.requires_grad}

    def state_all():
        return {k: p.detach().cpu().clone() for k, p in model.named_parameters() if k in names}

    optimizer = torch.optim.AdamW(
        [dict(params=head_params, lr=float(config["head_lr"]), weight_decay=0.01),
         dict(params=adapter_params, lr=float(config["adapter_lr"]), weight_decay=0.0)]
    )
    total_updates = total // accum
    head_warmup = int(config.get("head_warmup_updates", 100))
    adapter_start = head_only // accum
    adapter_budget = total_updates - adapter_start
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [lambda step: _linear_warmup_decay_lambda(step, warmup_steps=head_warmup, total_steps=total_updates),
         lambda step: 0.0 if step < adapter_start else _linear_warmup_decay_lambda(step - adapter_start, warmup_steps=max(1, int(0.03 * adapter_budget)), total_steps=adapter_budget)],
    )

    position = 0
    history = []
    best = dict(score=float("inf"), position=-1, state=None, metrics=None)
    rng = None
    if resume:
        slots = sorted((out / "checkpoints").glob("resume-slot*.pt"))
        candidates = []
        for path in slots:
            proof = path.with_suffix(".sha256")
            if not path.exists() or not proof.exists():
                continue
            if hashlib.sha256(path.read_bytes()).hexdigest() != proof.read_text().strip():
                continue
            value = torch.load(path, map_location="cpu", weights_only=False)
            if value.get("sequence_sha256") != sequence_sha or value.get("presentations") != total:
                continue
            candidates.append(value)
        if not candidates:
            raise FileNotFoundError("No verified resumable checkpoint for this sequence")
        saved = max(candidates, key=lambda v: v["position"])
        current = dict(model.named_parameters())
        assert set(saved["trainable_state"]) == names
        with torch.no_grad():
            for k, v in saved["trainable_state"].items():
                current[k].copy_(v.to(current[k].device))
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        position = int(saved["position"])
        history = saved["history"]
        best = saved["best"]
        restore_rng(saved["rng"])
        assert position % accum == 0
        print(f"RESUMED position={position}", flush=True)
    else:
        saved = restore_h1a(model, config["initial_trainable_state"], config["initial_trainable_state_sha256"])
        set_random_seed(int(config["seed"]))
        replay = replay_check(model, tokenizer, device, config["replay_validation_path"], config["replay_predictions_path"], qcfg)
        print("H1A_REPLAY", json.dumps(replay), flush=True)
        head_sha = reinit_final_head(model, int(config["seed"]) + 7)
        print("HEAD_REINIT", head_sha, flush=True)
        _atomic_json(out / "initialisation.json", dict(
            source_trainable_state=config["initial_trainable_state"],
            source_trainable_state_sha256=config["initial_trainable_state_sha256"],
            source_replay=replay,
            reinitialised_layer="photoz_head.net.6 (final 300-bin linear)",
            reinitialised_head_state_sha256=head_sha,
            retained="H1-A adapter and head hidden trunk",
            utc=now(),
        ))

    def checkpoint(force=False):
        value = dict(
            format="redshift_anchor_resume_v1",
            presentations=total,
            sequence_sha256=sequence_sha,
            position=position,
            history=history,
            best=best,
            trainable_state=state_all(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            rng=rng_state(),
        )
        path = out / "checkpoints" / f"resume-slot{((position // max(1, interval)) % 2)}.pt"
        save_torch(path, value)
        proof = path.with_suffix(".sha256")
        proof.write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")

    started = time.time()
    interval = int(config.get("checkpoint_interval_presentations", 1024))
    val_interval = int(config.get("validation_interval_presentations", 10240))
    optimizer.zero_grad(set_to_none=True)
    update_step = position // accum
    while position < total:
        window_start = position
        window_end = min(position + accum, total)
        counts = np.bincount(component[window_start:window_end], minlength=3)
        coefs = np.zeros(3)
        for c in range(3):
            if counts[c] > 0:
                coefs[c] = weights[c] / counts[c]
        assert abs(coefs @ counts - 1.0) < 1e-12 or (coefs @ counts) < 1.0 + 1e-12
        head_phase = window_start < head_only
        model.train()
        if head_phase:
            model.freeze_qwen_representation = True
            for p in adapter_params:
                p.requires_grad_(False)
            model.qwen.eval()
        else:
            model.freeze_qwen_representation = False
            for p in adapter_params:
                p.requires_grad_(True)
        loss_sum = 0.0
        for j in range(window_start, window_end):
            item = dataset[j]
            batch = collate([item])
            z = batch.pop("z_spec").to(device)
            batch.pop("object_id")
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**inputs)
            ce = redshift_cross_entropy_loss(logits.float(), z, edges)
            pz = torch.softmax(logits.float(), dim=-1)
            crps = discrete_crps(pz, z, centers).mean()
            loss = (ce + 0.5 * crps) * float(coefs[component[j]])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at presentation {j}")
            loss.backward()
            loss_sum += float(loss.detach())
        head_norm = torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        adapter_norm = torch.nn.utils.clip_grad_norm_(adapter_params, 0.1)
        if not (torch.isfinite(head_norm) and torch.isfinite(adapter_norm)):
            raise RuntimeError(f"Non-finite gradient norm at update {update_step}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1
        position = window_end
        history.append(dict(position=position, update=update_step, phase="head" if head_phase else "joint", loss=loss_sum, head_grad=float(head_norm), adapter_grad=float(adapter_norm)))
        if position % 100 == 0:
            elapsed = time.time() - started
            rate = position / max(elapsed, 1e-9)
            _atomic_json(out / "status.json", dict(
                utc=now(), stage="training", position=position, total=total, update=update_step,
                fraction=position / total, elapsed_seconds=elapsed, presentations_per_second=rate,
                eta_seconds=(total - position) / rate if rate > 0 else None,
                latest_loss=loss_sum, phase="head" if head_phase else "joint",
                lr_head=optimizer.param_groups[0]["lr"], lr_adapter=optimizer.param_groups[1]["lr"],
                gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), pid=os.getpid(),
            ))
        if position % interval == 0 or position == total:
            checkpoint()
        if position % val_interval == 0 or position == total:
            metrics = evaluate_development(model, tokenizer, device, val, qcfg)
            metrics["position"] = position
            metrics["utc"] = now()
            history.append(dict(position=position, validation=metrics))
            _atomic_json(out / f"validation_{position:06d}.json", metrics)
            print("VALIDATION", json.dumps(metrics), flush=True)
            if metrics["selection_score"] < best["score"]:
                best = dict(score=metrics["selection_score"], position=position, state=state_all(), metrics=metrics)
                print(f"BEST position={position} score={metrics['selection_score']}", flush=True)
            checkpoint()
    checkpoint()
    final_state = state_all()
    save_torch(out / "final_trainable_state.pt", final_state)
    export_model(model, out / "final", dict(
        format="redshift_anchor_1110_v1", presentations=total, position=position,
        initialisation="H1-A final adapter and head trunk; final 300-bin layer reinitialised",
        component_weights=dict(synthetic=0.5, lowz_source=0.3, anchor_fitting=0.2),
        sequence_sha256=sequence_sha, utc=now(),
    ))
    if best["state"] is not None:
        with torch.no_grad():
            current = dict(model.named_parameters())
            for k, v in best["state"].items():
                current[k].copy_(v.to(current[k].device))
        export_model(model, out / "selected", dict(
            format="redshift_anchor_1110_v1", selected_position=best["position"], selection_score=best["score"],
            selected_metrics=best["metrics"], presentations=total, sequence_sha256=sequence_sha, utc=now(),
        ))
        save_torch(out / "selected_trainable_state.pt", best["state"])
    summary = dict(stage="complete", position=position, total=total, best=best, elapsed_seconds=time.time() - started)
    _atomic_json(out / "status.json", summary | dict(utc=now(), gpu=os.environ.get("CUDA_VISIBLE_DEVICES")))
    print("TRAIN_COMPLETE", json.dumps(dict(position=position, best_score=best["score"], best_position=best["position"])), flush=True)
    return summary
