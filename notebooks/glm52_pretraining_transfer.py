#!/usr/bin/env python3
"""Budgeted GalaxiesML-Spec transfer test: toy-pretrained GLM versus random GLM."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import glm52_posttraining as glm
import glm52_capacity_checks as capacity
from aion_magnitude.FM_Qwen import load_frozen_qwen, pool_qwen_hidden_states
from aion_magnitude.metrics import predict_photoz_from_logits, redshift_cross_entropy_loss, summarize_pdf_metrics
from aion_magnitude.models import PhotoZHead
from aion_magnitude.qwen_posttraining import (
    QwenPhotoZModel, TextRedshiftDataset, make_text_collator, qwen_hidden_size,
    _linear_warmup_decay_lambda,
)
from aion_magnitude.utils import make_redshift_grid, set_random_seed


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--survey-prepared-json', type=Path, default=capacity.DEFAULT_SURVEY_PREPARED)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--train-sizes', type=int, nargs='+', default=[128, 512])
    p.add_argument('--val-objects', type=int, default=256)
    p.add_argument('--data-seed', type=int, default=42)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--repeat-seeds', type=int, nargs='*', default=[43, 44])
    p.add_argument('--promising-relative-difference', type=float, default=.05,
                   help='Repeat both initializations if validation CE or NMAD differs by this fraction at any screen size/method.')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--microbatch-size', type=int, default=4)
    p.add_argument('--effective-batch-size', type=int, default=16)
    p.add_argument('--head-learning-rate', type=float, default=2e-4)
    p.add_argument('--adapter-learning-rate', type=float, default=1e-5)
    p.add_argument('--broad-backbone', choices=('auto', 'off'), default='auto')
    p.add_argument('--broad-train-objects', type=int, default=128)
    p.add_argument('--broad-epochs', type=int, default=3)
    p.add_argument('--broad-learning-rate', type=float, default=1e-4)
    p.add_argument('--random-cache-root', type=Path, default=Path('/scratch/.tmp-gsm/glm52-transfer-random'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--prepare-only', action='store_true', help='CPU dataset/tokenizer audit only; no model creation or training.')
    return p


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def fixed_subsets(product, sizes, val_objects, seed):
    labels = np.asarray(product['split_labels'])
    rng = np.random.default_rng(seed)
    train = rng.permutation(np.flatnonzero(labels == 'train'))
    val = rng.permutation(np.flatnonzero(labels == 'val'))
    if min(sizes) < 1 or max(sizes) > len(train) or not 1 <= val_objects <= len(val):
        raise ValueError('Requested sample sizes exceed the original train/validation splits.')
    return {size: np.sort(train[:size]) for size in sizes}, np.sort(val[:val_objects])


def prepare_data(args):
    """Reuse the native spec-z cohort; never load a model's test predictions."""
    _, product, preparation, bounds, reference, _ = capacity.load_galaxiesml_product(args.survey_prepared_json)
    config = reference['metadata']['config']
    sizes = sorted(set(args.train_sizes + [args.broad_train_objects]))
    subsets, val_rows = fixed_subsets(product, sizes, args.val_objects, args.data_seed)
    # Only selected training/validation objects enter serialized datasets or loaders.
    selected = np.unique(np.concatenate([subsets[max(sizes)], val_rows]))
    small = dict(product)
    small['extra_features'] = torch.as_tensor(product['extra_features'])[selected]
    texts = glm._serialize_product(small, capacity.galaxiesml_serialization())
    lookup = dict(zip(selected.tolist(), texts))

    def dataset(rows):
        return TextRedshiftDataset([lookup[int(row)] for row in rows], product['z_spec'][rows],
                                   [str(product['object_id'][row]) for row in rows])

    datasets = {size: dataset(rows) for size, rows in subsets.items()}
    val = dataset(val_rows)
    manifest = {
        'catalogue_kind': 'galaxiesml-spec', 'target': preparation['target'],
        'source_prepared_json': str(args.survey_prepared_json), 'cohort_cache_key': preparation['cache_key'],
        'feature_names': product['feature_names'],
        'data_seed': args.data_seed, 'redshift_bounds': list(bounds), 'n_z_bins': 300,
        'split_counts': preparation['split_counts'], 'missing_values': 'explicit NA; no numeric imputation in text inputs',
        'training_object_ids': {size: data.object_ids for size, data in datasets.items()},
        'validation_object_ids': val.object_ids, 'test_evaluated': False,
        'model_path': config['model_path'], 'max_length': config['max_length'],
        'serialization': vars(capacity.galaxiesml_serialization()),
    }
    del product, small
    return datasets, val, manifest


def head_state(hidden_size, seed):
    set_random_seed(seed)
    head = PhotoZHead(hidden_size, n_z_bins=300, hidden_dim=256)
    return {key: value.detach().clone() for key, value in head.state_dict().items()}


def save_random_backbone(model_path, destination, seed):
    """No pretrained tensors are copied: instantiate the identical AutoModel config."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    set_random_seed(seed)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    tokenizer.save_pretrained(destination)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    del model; gc.collect()
    return parameters


def broad_trainable(name):
    # Train embeddings, every attention block, dense/shared MLP, routers, and norms.
    # Keep routed expert tensors fixed to limit optimizer memory, symmetrically.
    return '.experts.' not in name


def load_arm(path, mode, seed, args):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    base, tokenizer = load_frozen_qwen(path, device=args.device, load_in_4bit=mode != 'broad',
                                      torch_dtype='bf16', local_files_only=True, trust_remote_code=True)
    if mode != 'broad':
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=mode == 'qlora')
    if mode == 'qlora':
        suffixes = glm.GLM52_LORA_TARGET_SETTING.removeprefix('linear-suffixes:').split(',')
        targets = [name for name, _ in base.named_modules() if name and any(name.endswith('.'+suffix) or name == suffix for suffix in suffixes)]
        set_random_seed(seed + 1000)
        base = get_peft_model(base, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                    r=8, lora_alpha=16, lora_dropout=.05, target_modules=targets, bias='none'))
    elif mode == 'broad':
        for name, parameter in base.named_parameters():
            parameter.requires_grad_(broad_trainable(name))
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        base.requires_grad_(False)
    model = QwenPhotoZModel(base, hidden_size=qwen_hidden_size(base), n_z_bins=300, head_hidden_dim=256)
    model.photoz_head.load_state_dict(head_state(qwen_hidden_size(base), seed))
    return model.to(args.device), tokenizer


def autocast(device):
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16,
                          enabled=torch.device(device).type == 'cuda')


@torch.no_grad()
def frozen_dataset(model, tokenizer, dataset, args, max_length):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.microbatch_size,
                        collate_fn=make_text_collator(tokenizer, max_length=max_length))
    parts = []
    for batch in loader:
        batch.pop('z_spec'); batch.pop('object_id')
        inputs = {key:value.to(args.device) for key,value in batch.items()}
        with autocast(args.device):
            output = model.qwen(**inputs, use_cache=False, return_dict=True)
            parts.append(pool_qwen_hidden_states(output.last_hidden_state, inputs['attention_mask'], pooling='last').float().cpu())
    return TensorDataset(torch.cat(parts), dataset.redshifts)


def batch_logits(model, batch, mode, device):
    if mode == 'head':
        return model.photoz_head(batch[0].to(device)), batch[1].to(device)
    z = batch.pop('z_spec').to(device); batch.pop('object_id')
    return model(**{key:value.to(device) for key,value in batch.items()}), z


@torch.no_grad()
def evaluate(model, dataset, mode, collate, edges, centers, args):
    model.eval(); logits_parts, targets = [], []
    loader = DataLoader(dataset, batch_size=args.microbatch_size, collate_fn=collate)
    for batch in loader:
        with autocast(args.device):
            logits, z = batch_logits(model, batch, mode, args.device)
        logits_parts.append(logits.float().cpu()); targets.append(z.cpu())
    logits, z = torch.cat(logits_parts), torch.cat(targets)
    evaluation = {'logits': logits, 'z_spec': z, 'redshift_edges': edges.cpu(), 'redshift_centers': centers.cpu()}
    evaluation.update(predict_photoz_from_logits(logits, centers=centers.cpu()))
    return summarize_pdf_metrics(evaluation)


def train_arm(model, tokenizer, train, val, mode, seed, epochs, bounds, max_length, args, folder):
    start = time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    if mode == 'head':
        train = frozen_dataset(model, tokenizer, train, args, max_length)
        val = frozen_dataset(model, tokenizer, val, args, max_length)
        collate = None
    else:
        collate = make_text_collator(tokenizer, max_length=max_length)
    head_parameters = list(model.photoz_head.parameters())
    backbone_parameters = [p for p in model.qwen.parameters() if p.requires_grad]
    groups = [{'params': head_parameters, 'lr': args.head_learning_rate, 'weight_decay': .01}]
    if backbone_parameters:
        groups.append({'params': backbone_parameters, 'lr': args.broad_learning_rate if mode == 'broad' else args.adapter_learning_rate,
                       'weight_decay': .01 if mode == 'broad' else 0.})
    if mode == 'broad':
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(groups)
    else:
        optimizer = torch.optim.AdamW(groups)
    updates_per_epoch = int(np.ceil(len(train)/args.effective_batch_size))
    total_updates = epochs * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _linear_warmup_decay_lambda(
        step, warmup_steps=int(.03*total_updates), total_steps=total_updates))
    edges, centers = make_redshift_grid(*bounds, 300)
    edges = edges.to(args.device)
    history = []; updates = 0
    if torch.device(args.device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    set_random_seed(seed + 2000)
    history.append({'epoch': 0, 'examples_seen': 0, 'updates': 0,
                    'train': evaluate(model, train, mode, collate, edges, centers, args),
                    'val': evaluate(model, val, mode, collate, edges, centers, args)})
    best = history[0]; best_state = None
    accumulation = args.effective_batch_size // args.microbatch_size
    for epoch in range(1, epochs+1):
        set_random_seed(seed + 2000 + epoch)
        loader = DataLoader(train, batch_size=args.microbatch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed+epoch), collate_fn=collate)
        model.train()
        if mode == 'head':
            model.qwen.eval()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            group_start = (batch_index // accumulation) * args.effective_batch_size
            group_count = min(args.effective_batch_size, len(train)-group_start)
            with autocast(args.device):
                logits, z = batch_logits(model, batch, mode, args.device)
                loss = redshift_cross_entropy_loss(logits.float(), z, edges=edges)
            (loss * len(z) / group_count).backward()
            if (batch_index+1) % accumulation == 0 or batch_index+1 == len(loader):
                torch.nn.utils.clip_grad_norm_(head_parameters, 1.)
                if backbone_parameters:
                    torch.nn.utils.clip_grad_norm_(backbone_parameters, .1 if mode == 'qlora' else 1.)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); updates += 1
        row = {'epoch': epoch, 'examples_seen': epoch*len(train), 'updates': updates,
               'train': evaluate(model, train, mode, collate, edges, centers, args),
               'val': evaluate(model, val, mode, collate, edges, centers, args)}
        history.append(row); print(f'{folder.name} '+json.dumps(row), flush=True)
        if row['val']['cross_entropy'] < best['val']['cross_entropy']:
            best = row
            if mode != 'broad':
                best_state = {name: parameter.detach().cpu().clone()
                              for name, parameter in model.named_parameters() if parameter.requires_grad}
        write_json(folder / 'history.json', history)
    result = {'mode': mode, 'seed': seed, 'train_objects': len(train), 'val_objects': len(val),
              'epochs': epochs, 'updates': updates, 'history': history, 'selected_epoch': best['epoch'],
              'train_metrics': best['train'], 'val_metrics': best['val'],
              'seconds': time.monotonic()-start, 'test_evaluated': False,
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'total_parameters': sum(p.numel() for p in model.parameters()),
              'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30 if torch.device(args.device).type == 'cuda' else None}
    # Avoid large broad-backbone snapshots; the experiment's outputs are curves and metrics.
    if mode != 'broad' and best_state is not None:
        torch.save(best_state, folder / 'selected_trainable_state.pt')
    write_json(folder / 'result.json', result)
    del optimizer, best_state
    return result


def relative_difference(toy, random, metric):
    return (random['val_metrics'][metric]-toy['val_metrics'][metric])/max(abs(random['val_metrics'][metric]), 1e-12)


def promising(results, threshold):
    pairs = {}
    for row in results:
        pairs.setdefault((row['mode'], row['train_objects'], row['seed']), {})[row['initialization']] = row
    return any(any(abs(relative_difference(pair['toy'], pair['random'], metric)) >= threshold
                   for metric in ('cross_entropy', 'nmad')) for pair in pairs.values() if set(pair) == {'toy','random'})


def paired_summary(results, threshold):
    groups = {}
    for row in results:
        groups.setdefault((row['mode'],row['train_objects']), {}).setdefault(row['seed'], {})[row['initialization']] = row
    summary = []
    for (mode, size), seeds in groups.items():
        differences = [relative_difference(pair['toy'], pair['random'], 'cross_entropy')
                       for pair in seeds.values() if set(pair) == {'toy','random'}]
        if not differences:
            continue
        mean = float(np.mean(differences))
        learned = all(any(min(row['train']['cross_entropy'] for row in arm['history'])
                          < arm['history'][0]['train']['cross_entropy'] - 1e-3
                          for arm in pair.values()) for pair in seeds.values())
        if not learned:
            interpretation = 'inconclusive: neither arm reduced training CE in at least one paired seed'
        elif len(differences) >= 3 and min(differences) > 0 and mean >= threshold:
            interpretation = 'positive transfer under this budget'
        elif len(differences) >= 3 and max(differences) < 0 and mean <= -threshold:
            interpretation = 'negative transfer under this budget'
        elif len(differences) < 3:
            interpretation = 'screen only; no replicated transfer conclusion'
        else:
            interpretation = 'no consistent measurable CE advantage under this budget; inconclusive'
        summary.append({'mode':mode, 'train_objects':size, 'paired_seeds':len(differences),
                        'toy_relative_ce_advantage':differences, 'mean_relative_ce_advantage':mean,
                        'std_relative_ce_advantage':float(np.std(differences, ddof=1)) if len(differences)>1 else None,
                        'interpretation':interpretation})
    return summary


def make_report(args, results, manifest, broad_status, repeated):
    summary = paired_summary(results, args.promising_relative_difference)
    write_json(args.output_dir / 'summary.json', {'comparisons':summary, 'broad_status':broad_status,
               'repeated_seeds':repeated, 'test_evaluated':False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    largest = max(args.train_sizes)
    for column, mode in enumerate(('head','qlora','broad')):
        for result in results:
            if result['mode'] != mode or (mode != 'broad' and result['train_objects'] != largest):
                continue
            label = f"{result['initialization']} seed {result['seed']}"
            for axis, split in zip(axes[:,column], ('train','val')):
                axis.plot([r['examples_seen'] for r in result['history']],
                          [r[split]['cross_entropy'] for r in result['history']], label=label)
                axis.set_xlabel('Training examples seen (includes repeats)'); axis.set_ylabel(f'{split} CE')
                axis.set_title(mode)
        if axes[0,column].lines:
            axes[0,column].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(args.output_dir / 'learning_curves.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    for axis, metric in zip(axes, ('cross_entropy','nmad')):
        for mode in ('head','qlora'):
            for initialization in ('toy','random'):
                sizes = sorted(set(r['train_objects'] for r in results if r['mode']==mode))
                values = [np.mean([r['val_metrics'][metric] for r in results if r['mode']==mode and r['initialization']==initialization and r['train_objects']==size]) for size in sizes]
                axis.plot(sizes, values, 'o-', label=f'{mode}: {initialization}')
        axis.set_xlabel('Unique training objects'); axis.set_ylabel('Selected validation '+metric); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(args.output_dir / 'sample_efficiency.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13,7))
    for column,mode in enumerate(('head','qlora','broad')):
        for initialization in ('toy','random'):
            selected=[r for r in results if r['mode']==mode and r['initialization']==initialization
                      and (mode=='broad' or r['train_objects']==largest)]
            if not selected:
                continue
            for axis,metric in zip(axes[:,column],('nmad','catastrophic_outlier_fraction')):
                for split,style in (('train','--'),('val','-')):
                    values=np.asarray([[row[split][metric] for row in r['history']] for r in selected])
                    x=[row['examples_seen'] for row in selected[0]['history']]
                    mean=values.mean(0); std=values.std(0)
                    line,=axis.plot(x,mean,style,label=f'{initialization} {split}')
                    if len(selected)>1:
                        axis.fill_between(x,mean-std,mean+std,color=line.get_color(),alpha=.1)
                axis.set_title(mode); axis.set_xlabel('Training examples seen'); axis.set_ylabel(metric)
        if axes[0,column].lines:
            axes[0,column].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(args.output_dir/'nmad_outlier_curves.png',dpi=150); plt.close(fig)
    lines = ['# Toy-pretraining transfer: matched GalaxiesML-Spec photo-z screen', '',
             'True z: DESI_fibermap/DESI_redshift (spectroscopic redshift). The quality-selected native HSC grizy fluxes/errors plus EBV, missing-value handling, numerical formatting, tokenizer, fixed validation subset, and nested training subsets are shared across all arms. No held-out test predictions or metrics were produced.', '',
             '## Matched experiment', '',
             f"Train sizes: {args.train_sizes}; validation objects: {args.val_objects}; screen seed: {args.seed}; repeats actually run: {repeated}.", '',
             'Toy weights come from the existing base checkpoint, not a downstream-trained adapter. Random weights are created using AutoModel.from_config with the identical configuration; no pretrained tensors are copied. The same tokenizer is used. Frozen-head and QLoRA arms both load through the same NF4 double-quantization/k-bit-preparation path. Paired arms receive identical initial heads, LoRA initialization, object order, learning rates, epochs, accumulation, and evaluation cadence.', '',
             f"Head/QLoRA epochs: {args.epochs}; effective batch: {args.effective_batch_size}; head LR: {args.head_learning_rate}; adapter LR: {args.adapter_learning_rate}; rank 8, alpha 16, dropout .05. Adapters train from the start; there is no head-only warmup phase. Each arm completes its budget; validation CE selects its reported checkpoint, including epoch 0 if training fails to improve.", '',
             'Frozen random features and low-rank updates may unfairly restrict random initialization. The broader pair trains embeddings, all attention blocks, dense/shared MLPs, routers, and norms in BF16 with an 8-bit optimizer; routed experts remain fixed in both arms. This permits much broader learning from scratch but is not a full-training capacity ceiling.', '',
             f"Broader feasibility: {broad_status}. Budget when run: {args.broad_train_objects} training objects, {args.broad_epochs} epochs, backbone LR {args.broad_learning_rate}. If either member runs out of memory, both are excluded from the comparison.", '',
             '## Selected validation and training metrics', '',
             '| Mode | N train | Seed | Initial weights | Epoch | Train CE | Val CE | Val NMAD | Val outliers | Seconds |',
             '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['mode']} | {r['train_objects']} | {r['seed']} | {r['initialization']} | {r['selected_epoch']} | {r['train_metrics']['cross_entropy']:.5f} | {r['val_metrics']['cross_entropy']:.5f} | {r['val_metrics']['nmad']:.5f} | {100*r['val_metrics']['catastrophic_outlier_fraction']:.2f}% | {r['seconds']:.1f} |")
    lines += ['', 'Point estimates use PDF medians. Outliers satisfy abs(z_pred-spec_z)/(1+spec_z)>0.15. Complete per-epoch train/validation CE, NMAD, outlier fractions, update counts, and examples seen are saved in each history.json.', '',
              '## Empirical transfer assessment', '']
    for row in summary:
        lines.append(f"- {row['mode']}, N={row['train_objects']}, {row['paired_seeds']} paired seed(s): mean relative toy CE advantage {100*row['mean_relative_ce_advantage']:+.2f}%; {row['interpretation']}.")
    lines += ['', f"Repeat rule: an absolute relative validation CE or NMAD difference of at least {100*args.promising_relative_difference:g}% in any screened pair triggers the requested repeat seeds for both initializations and all feasible screened settings. Transfer labels require at least three paired seeds, consistent CE direction, and mean effect at least that threshold; this is a descriptive decision rule, not a significance test.", '',
              'Mixed directions, weak effects, or failure of both arms to reduce training error are inconclusive. Lack of a clear advantage does not prove equivalence. Conclusions apply only to this small dataset and optimization budget. Sample-efficiency curves compare unique training objects at the same epoch budget; examples seen and optimizer updates are also recorded.', '',
              '## Mechanisms and limits', '',
              'These comparisons measure the consequence of the two initializations under a specified adaptation regime. They do not establish whether any effect is caused by copypasta content, numerical tokenization, expert routing, or another pretraining property. That would require further interventions. The broader arm is included to reduce the frozen-random disadvantage, but its remaining frozen experts and short budget still restrict inference.', '',
              '![Train and validation learning curves](learning_curves.png)', '',
              '![NMAD and outlier learning curves; shading is seed standard deviation](nmad_outlier_curves.png)', '',
              '![Sample efficiency](sample_efficiency.png)', '',
              'Provenance and fixed object IDs: [manifest.json](manifest.json). Aggregate comparisons: [summary.json](summary.json). Per-arm results and all learning-curve metrics are in seed*/ directories.', '']
    (args.output_dir / 'report.md').write_text('\n'.join(lines))


def main(argv=None):
    p=parser(); args=p.parse_args(argv)
    if any(n < 1 for n in args.train_sizes+[args.val_objects,args.epochs,args.microbatch_size,args.effective_batch_size,args.broad_train_objects,args.broad_epochs]):
        p.error('Sample sizes, batch sizes, and epoch budgets must be positive.')
    if args.effective_batch_size % args.microbatch_size:
        p.error('Effective batch size must be divisible by microbatch size.')
    if any(lr<=0 for lr in (args.head_learning_rate,args.adapter_learning_rate,args.broad_learning_rate)) or args.promising_relative_difference<=0:
        p.error('Learning rates and the repeat threshold must be positive.')
    args.train_sizes=sorted(set(args.train_sizes))
    if not args.train_sizes:
        p.error('At least one training sample size is required.')
    datasets, val, manifest=prepare_data(args)
    from transformers import AutoConfig, AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(manifest['model_path'],local_files_only=True,trust_remote_code=True)
    longest=max(len(tokenizer(text,truncation=False)['input_ids']) for text in datasets[max(datasets)].texts+val.texts)
    if longest>manifest['max_length']:
        raise ValueError('Selected inputs would be truncated under the original format.')
    manifest.update(arguments=vars(args),token_audit={'longest_input_tokens':longest,'passed':True})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.output_dir/'manifest.json',manifest)
    if args.prepare_only:
        print(f'CPU audit passed: GalaxiesML-Spec/DESI spec-z train sizes {args.train_sizes}, validation {len(val)}, max tokens {longest}.'); return 0
    if torch.device(args.device).type!='cuda':
        p.error('NF4 and broad 8-bit-optimizer comparison requires CUDA; use --prepare-only for CPU audit.')
    args.random_cache_root.mkdir(parents=True,exist_ok=True)
    config=AutoConfig.from_pretrained(manifest['model_path'],local_files_only=True,trust_remote_code=True)
    config_dict=config.to_dict()
    manifest['architecture_config_sha256']=hashlib.sha256(json.dumps(config_dict,sort_keys=True,default=str).encode()).hexdigest()
    free,total=torch.cuda.mem_get_info(torch.device(args.device))
    broad_enabled=args.broad_backbone=='auto' and free>=8*2**30
    broad_status='enabled with >=8 GiB initially free' if broad_enabled else 'skipped (disabled or <8 GiB initially free)'
    results=[]; repeat_triggered=False; repeated=[]
    experiment_start=time.monotonic()
    seeds=[args.seed]+[s for s in dict.fromkeys(args.repeat_seeds) if s!=args.seed]
    for seed_index,seed in enumerate(seeds):
        if seed_index and not repeat_triggered:
            break
        if seed_index:
            repeated.append(seed)
        with tempfile.TemporaryDirectory(prefix=f'seed{seed}-',dir=args.random_cache_root) as temporary:
            random_path=Path(temporary)
            count=save_random_backbone(manifest['model_path'],random_path,seed)
            manifest['backbone_parameter_count']=count
            plan=[(mode,size,args.epochs) for mode in ('head','qlora') for size in args.train_sizes]
            if broad_enabled:
                plan.append(('broad',args.broad_train_objects,args.broad_epochs))
            for mode,size,epochs in plan:
                pair=[]
                try:
                    for initialization,path in (('toy',manifest['model_path']),('random',random_path)):
                        folder=args.output_dir/f'seed{seed}'/f'{mode}-n{size}-{initialization}'
                        model,tokenizer=load_arm(path,mode,seed,args)
                        result=train_arm(model,tokenizer,datasets[size],val,mode,seed,epochs,
                                         manifest['redshift_bounds'],manifest['max_length'],args,folder)
                        result['initialization']=initialization
                        write_json(folder/'result.json',result)
                        pair.append(result)
                        del model,tokenizer; gc.collect(); torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    if mode!='broad':
                        raise
                    # Discard partial pairs and earlier broad comparisons for matched reporting.
                    if 'model' in locals():
                        del model
                    gc.collect(); torch.cuda.empty_cache()
                    broad_enabled=False; broad_status='skipped: OOM in a broader arm; both initializations excluded'
                    results=[r for r in results if r['mode']!='broad']; pair=[]
                results.extend(pair)
                write_json(args.output_dir/'results.json',results)
            if seed_index==0:
                repeat_triggered=promising(results,args.promising_relative_difference)
                print(f'Promising difference triggers repeats: {repeat_triggered}',flush=True)
        make_report(args,results,manifest,broad_status,repeated)
    manifest.update(broad_status=broad_status,repeat_triggered=repeat_triggered,repeated_seeds=repeated,
                    experiment_wall_seconds=time.monotonic()-experiment_start)
    write_json(args.output_dir/'manifest.json',manifest)
    print(f"Report: {args.output_dir/'report.md'}",flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
