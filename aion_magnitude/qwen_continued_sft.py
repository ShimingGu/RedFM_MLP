"""Continued photo-z SFT: trained-state initialization and update-zero selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import redshift_cross_entropy_loss, summarize_pdf_metrics
from .qwen_posttraining import (
    _load_trainable_state, _restore_rng_state, _trainable_state,
    evaluate_text_photoz_model, make_text_collator,
)


@dataclass(frozen=True)
class ContinuedSFTProtocol:
    version: str = "hsc_continued_sft_1023_v1"
    max_epochs: int = 2
    head_warmup_epochs: int = 0
    adapter_lr: float = 3e-6
    head_lr: float = 1e-5
    microbatch_size: int = 1
    gradient_accumulation_steps: int = 16
    seed: int = 1023
    eval_every_updates: int = 250
    checkpoint_every_updates: int = 50
    eval_batch_size: int = 8
    head_weight_decay: float = .01
    adapter_weight_decay: float = 0.0
    head_max_grad_norm: float = 1.0
    adapter_max_grad_norm: float = .1
    lr_schedule: str = "constant; no learning-rate warmup"
    optimizer_initialization: str = "new AdamW state; trained adapter and PDF head restored"
    incomplete_batch: str = "epoch-final partial effective batch uses its actual sample count"

    def __post_init__(self):
        if not 1 <= self.max_epochs <= 2 or self.head_warmup_epochs != 0:
            raise ValueError("Continued SFT allows at most two epochs and no head-only warmup")
        for field in ("microbatch_size", "gradient_accumulation_steps", "eval_every_updates",
                      "checkpoint_every_updates", "eval_batch_size"):
            if getattr(self, field) < 1:
                raise ValueError(f"Positive {field} required")
        if self.adapter_lr <= 0 or self.head_lr <= 0:
            raise ValueError("Positive learning rates required")


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def atomic_torch_save(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def is_better(candidate_ce, best_ce):
    """Strict improvement preserves the initial checkpoint on a tie."""
    return math.isfinite(candidate_ce) and candidate_ce < best_ce


def group_sample_count(batch_index, n_samples, protocol):
    batch_group = batch_index // protocol.gradient_accumulation_steps
    effective = protocol.microbatch_size * protocol.gradient_accumulation_steps
    return min(effective, n_samples - batch_group * effective)


def parameter_changes(initial, current):
    if initial.keys() != current.keys():
        raise ValueError("Trainable parameter names changed")
    output = {}
    for group in ("adapter", "pdf_head"):
        keys = [key for key in initial if (key.startswith("photoz_head.")) == (group == "pdf_head")]
        delta2, norm2, maximum, changed, count = 0., 0., 0., 0, 0
        for key in keys:
            old, new = initial[key].double(), current[key].double()
            delta = new - old
            delta2 += float(delta.square().sum())
            norm2 += float(old.square().sum())
            maximum = max(maximum, float(delta.abs().max()))
            changed += int(torch.count_nonzero(delta))
            count += delta.numel()
        output[group] = dict(delta_l2=math.sqrt(delta2), relative_delta_l2=math.sqrt(delta2) / max(math.sqrt(norm2), 1e-30),
                             max_abs_change=maximum, changed_values=changed, n_values=count)
    return output


def verify_replay(actual, reference):
    if actual["object_id"] != reference["object_id"]:
        raise ValueError("Validation replay object membership/order differs")
    for key in ("z_spec", "redshift_edges", "redshift_centers"):
        torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
    torch.testing.assert_close(actual["logits"], reference["logits"], rtol=1e-3, atol=.02)
    pdf_error = float((actual["pz"] - reference["pz"]).abs().max())
    if pdf_error > .005:
        raise ValueError(f"Validation replay PDF error {pdf_error}")
    metrics, expected = summarize_pdf_metrics(actual), summarize_pdf_metrics(reference)
    if abs(metrics["cross_entropy"] - expected["cross_entropy"]) > 1e-4:
        raise ValueError("Validation replay cross-entropy differs")
    if abs(metrics["nmad"] - expected["nmad"]) > 1e-4:
        raise ValueError("Validation replay NMAD differs")
    return dict(passed=True, n=len(actual["object_id"]),
                max_abs_logit_error=float((actual["logits"] - reference["logits"]).abs().max()),
                max_abs_pdf_error=pdf_error, metrics=metrics, archived_metrics=expected)


def continue_sft(*, model, tokenizer, train_dataset, val_dataset, initial_evaluation,
                 output_dir, protocol, max_length, device, resume=False,
                 on_progress=None, evaluator=evaluate_text_photoz_model):
    """Optimize restored weights only; zero-update evaluation must already pass.

    No test dataset or target-query data enter this function. Validation CE is
    the only checkpoint-selection criterion. The optimizer is deliberately new.
    """
    out = Path(output_dir)
    device = torch.device(device)
    edges = initial_evaluation["redshift_edges"].to(device)
    centers = initial_evaluation["redshift_centers"]
    collate = make_text_collator(tokenizer, max_length=max_length)
    val_loader = DataLoader(val_dataset, batch_size=protocol.eval_batch_size, shuffle=False, collate_fn=collate)
    head = [p for p in model.photoz_head.parameters() if p.requires_grad]
    adapter = [p for p in model.qwen.parameters() if p.requires_grad]
    if not head or not adapter or model.freeze_qwen_representation:
        raise ValueError("Restored adapter and PDF head must both train from update one")
    optimizer = torch.optim.AdamW([
        dict(params=head, lr=protocol.head_lr, weight_decay=protocol.head_weight_decay),
        dict(params=adapter, lr=protocol.adapter_lr, weight_decay=protocol.adapter_weight_decay),
    ])
    initial_state = _trainable_state(model)
    initial_metrics = summarize_pdf_metrics(initial_evaluation)
    state = dict(global_update=0, epoch=0, next_batch=0, best_update=0,
                 best_ce=initial_metrics["cross_entropy"], best_metrics=initial_metrics,
                 best_state=initial_state,
                 validation_history=[dict(update=0, epoch=0., **initial_metrics)],
                 training_history=[], elapsed_seconds=0., epoch_loss_sum=0., epoch_samples=0)
    if resume:
        saved = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        if saved["protocol"] != asdict(protocol):
            raise ValueError("Resume protocol differs")
        state = saved["state"]
        _load_trainable_state(model, saved["trainable_state"])
        optimizer.load_state_dict(saved["optimizer"])
        _restore_rng_state(saved["rng"])
    else:
        atomic_torch_save(initial_state, out / "initial_trainable_state.pt")
        atomic_torch_save(initial_evaluation, out / "best_validation.pt")
    started = time.monotonic()
    elapsed_before = state["elapsed_seconds"]
    max_updates = math.ceil(math.ceil(len(train_dataset) / protocol.microbatch_size) / protocol.gradient_accumulation_steps) * protocol.max_epochs

    def progress(phase):
        state["elapsed_seconds"] = elapsed_before + time.monotonic() - started
        status = dict(phase=phase, global_update=state["global_update"], max_updates=max_updates,
                      best_update=state["best_update"], best_metrics=state["best_metrics"],
                      zero_update_metrics=initial_metrics, elapsed_seconds=state["elapsed_seconds"],
                      validation_history=state["validation_history"], training_history=state["training_history"],
                      test_evaluated=False, target_labels_fitted=False,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.)
        write_json(out / "status.json", status)
        if on_progress:
            on_progress(status)
        return status

    def checkpoint():
        progress("training")
        atomic_torch_save(dict(protocol=asdict(protocol), state=state,
                              trainable_state=_trainable_state(model), optimizer=optimizer.state_dict(),
                              rng=dict(python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state(),
                                       torch_rng_state=torch.get_rng_state(),
                                       cuda_rng_state=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])), out / "latest.pt")

    def validate(epoch_fraction):
        evaluation = evaluator(model, val_loader, device=device, redshift_edges=edges, redshift_centers=centers)
        metrics = summarize_pdf_metrics(evaluation)
        row = dict(update=state["global_update"], epoch=epoch_fraction, **metrics)
        state["validation_history"].append(row)
        if is_better(metrics["cross_entropy"], state["best_ce"]):
            state.update(best_update=state["global_update"], best_ce=metrics["cross_entropy"], best_metrics=metrics,
                         best_state=_trainable_state(model))
            atomic_torch_save(evaluation, out / "best_validation.pt")
        print(f"validation update={state['global_update']}/{max_updates} CE={metrics['cross_entropy']:.6f} NMAD={metrics['nmad']:.6f} best_update={state['best_update']}", flush=True)
        model.train()

    progress("training")
    start_epoch, start_batch = state["epoch"], state["next_batch"]
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, protocol.max_epochs):
        generator = torch.Generator().manual_seed(protocol.seed + epoch)
        train_loader = DataLoader(train_dataset, batch_size=protocol.microbatch_size, shuffle=True, generator=generator, collate_fn=collate)
        model.train()
        model.freeze_qwen_representation = False
        group_loss, group_n = 0., 0
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            labels = batch.pop("z_spec").to(device)
            batch.pop("object_id")
            inputs = {k:v.to(device) for k,v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**inputs)
                loss = redshift_cross_entropy_loss(logits.float(), labels, edges=edges)
            batch_n = len(labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite source training loss")
            (loss * batch_n / group_sample_count(batch_index, len(train_dataset), protocol)).backward()
            group_loss += float(loss.detach()) * batch_n
            group_n += batch_n
            state["epoch_loss_sum"] += float(loss.detach()) * batch_n
            state["epoch_samples"] += batch_n
            update = (batch_index + 1) % protocol.gradient_accumulation_steps == 0 or batch_index + 1 == len(train_loader)
            if not update:
                continue
            torch.nn.utils.clip_grad_norm_(head, protocol.head_max_grad_norm)
            torch.nn.utils.clip_grad_norm_(adapter, protocol.adapter_max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state["global_update"] += 1
            state["epoch"], state["next_batch"] = epoch, batch_index + 1
            state["training_history"].append(dict(update=state["global_update"], epoch=epoch + (batch_index+1)/len(train_loader),
                        loss=group_loss/group_n, samples=group_n, adapter_lr=protocol.adapter_lr, head_lr=protocol.head_lr))
            group_loss, group_n = 0., 0
            if state["global_update"] == 1 or state["global_update"] % 10 == 0:
                print(f"train update={state['global_update']}/{max_updates} epoch={epoch+1} loss={state['training_history'][-1]['loss']:.6f}", flush=True)
                progress("training")
            if state["global_update"] % protocol.eval_every_updates == 0:
                validate(epoch + (batch_index+1)/len(train_loader))
                checkpoint()
            elif state["global_update"] % protocol.checkpoint_every_updates == 0:
                checkpoint()
        if state["validation_history"][-1]["update"] != state["global_update"]:
            validate(epoch + 1.)
        state.update(epoch=epoch+1, next_batch=0, epoch_loss_sum=0., epoch_samples=0)
        checkpoint()
        start_batch = 0
    last_state = _trainable_state(model)
    changes = dict(last_vs_initial=parameter_changes(initial_state, last_state),
                   selected_vs_initial=parameter_changes(initial_state, state["best_state"]))
    _load_trainable_state(model, state["best_state"])
    model.eval()
    model.qwen.save_pretrained(out / "selected" / "adapter")
    atomic_torch_save(model.photoz_head.state_dict(), out / "selected" / "photoz_head.pt")
    atomic_torch_save(state["best_state"], out / "selected" / "trainable_state.pt")
    write_json(out / "parameter_changes.json", changes)
    status = progress("source_selection_complete")
    status["parameter_changes"] = changes
    write_json(out / "source_summary.json", status)
    return status
