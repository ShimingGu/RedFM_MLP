#!/usr/bin/env python3
"""Inference-only numerical sensitivity on a fixed GalaxiesML-Spec validation subset."""

from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import gc
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import glm52_capacity_checks as capacity
import glm52_posttraining as glm
import qwen_survey_posttraining as survey
from aion_magnitude.FM_Qwen import load_frozen_qwen, pool_qwen_hidden_states
from aion_magnitude.metrics import predict_photoz_from_logits
from aion_magnitude.models import build_baseline_model
from aion_magnitude.qwen_posttraining import QwenPhotoZModel, qwen_hidden_size
from aion_magnitude.utils import make_redshift_grid, set_random_seed


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--glm-run-dir', type=Path, required=True,
                   help='Completed GalaxiesML-Spec GLM capacity-check directory with adapter, head, and cohort.json.')
    p.add_argument('--mlp-run-dir', type=Path, required=True)
    p.add_argument('--qwen-run-dir', type=Path,
                   help='Optional survey run; uses completed adapter if present, otherwise saved frozen control.')
    p.add_argument('--survey-prepared-json', type=Path, default=capacity.DEFAULT_SURVEY_PREPARED)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--sample-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mag-steps', type=float, nargs='+', default=[.0001, .001, .01, .05, .1])
    p.add_argument('--jump-threshold', type=float, default=.02,
                   help='Descriptive flag for abs(delta PDF median)/(1+spec-z), not a significance test.')
    p.add_argument('--device', default='cuda')
    p.add_argument('--prepare-only', action='store_true', help='CPU cohort, perturbation, and token-length audit; no model inference.')
    return p


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


def select_validation(product, size, seed):
    rows = np.flatnonzero(np.asarray(product['split_labels']) == 'val')
    if not 1 <= size <= len(rows):
        raise ValueError(f'Choose 1–{len(rows)} validation objects.')
    return np.sort(np.random.default_rng(seed).choice(rows, size, replace=False))


def perturb_magnitude(values, band_index, delta):
    """Hold native magnitude error fixed: both converted flux and sigma scale together."""
    changed = values.clone()
    flux, sigma = values[2 * band_index:2 * band_index + 2]
    if not torch.isfinite(flux) or not torch.isfinite(sigma) or flux <= 0 or sigma <= 0:
        return None
    magnitude = 22.5 - 2.5 * np.log10(float(flux)) + delta
    if not 0 < magnitude < 40:
        return None
    changed[2 * band_index:2 * band_index + 2] *= 10 ** (-.4 * delta)
    return changed


def glm_text(values, names):
    # Exactly the serialization used by glm52_capacity_checks.run_overfit.
    serialization = capacity.galaxiesml_serialization()
    return glm._serialize_product({'extra_features': values[None], 'feature_names': names}, serialization)[0]


def numeric_input(values, saved):
    finite = torch.isfinite(values)
    imputation = saved['imputation_values']
    if imputation is None:
        raise ValueError('GalaxiesML-Spec MLP must contain its saved training-only imputation.')
    encoded = torch.cat((torch.where(finite, values, imputation), (~finite).float()))
    return (encoded - saved['feature_mean']) / saved['feature_scale']


def verify_cohort(run_dir, product, name):
    saved = json.loads((run_dir / 'cohort.json').read_text())
    if saved.get('catalogue_kind') != 'galaxiesml-spec':
        raise ValueError(f'{name} is not a GalaxiesML-Spec run.')
    for key in ('object_id', 'split_labels'):
        if saved[key] != product[key]:
            raise ValueError(f'{name} {key} differ from the diagnostic cohort.')
    if saved['preparation']['feature_names'] != product['feature_names']:
        raise ValueError(f'{name} feature order differs from the diagnostic cohort.')
    return saved


def load_policy(run_dir, config, device, *, frozen_checkpoint=None):
    """Restore inference weights with the same NF4/k-bit preparation used for training."""
    from peft import PeftModel, prepare_model_for_kbit_training

    base, tokenizer = load_frozen_qwen(
        config['model_path'], device=device, load_in_4bit=True,
        torch_dtype='bf16', local_files_only=True, trust_remote_code=True,
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    if frozen_checkpoint is None:
        base = PeftModel.from_pretrained(base, run_dir / 'adapter', is_trainable=False)
    model = QwenPhotoZModel(base, hidden_size=qwen_hidden_size(base),
                           n_z_bins=config['n_z_bins'], head_hidden_dim=config['head_hidden_dim'])
    if frozen_checkpoint is None:
        model.photoz_head.load_state_dict(torch.load(run_dir / 'photoz_head.pt', map_location='cpu', weights_only=True))
    else:
        # The frozen control's checkpoint includes only its PDF-head wrapper.
        model.photoz_head.load_state_dict({
            key.removeprefix('photoz_head.'): value
            for key, value in frozen_checkpoint['state_dict'].items()
        })
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, tokenizer


@torch.inference_mode()
def snapshot(model, tokenizer, text, config, device, routes=False):
    tokens = tokenizer(text, return_tensors='pt', truncation=False)
    if tokens['input_ids'].shape[1] > config['max_length']:
        raise ValueError('Perturbed input exceeds saved max_length; refusing silent truncation.')
    tokens = {key: value.to(device) for key, value in tokens.items()}
    captured, handles = {}, []

    def hook(name):
        def capture(module, inputs, output):
            logits, weights, ids = output
            captured[name] = {'ids': ids.detach().cpu(), 'logits': logits.detach().float().cpu()}
        return capture

    if routes:
        for name, module in model.qwen.named_modules():
            if module.__class__.__name__ == 'GlmMoeDsaTopkRouter':
                handles.append(module.register_forward_hook(hook(name)))
        if not handles:
            raise RuntimeError('GLM MoE routers were not found; routing diagnostic cannot be measured.')
    try:
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16,
                            enabled=torch.device(device).type == 'cuda'):
            output = model.qwen(**tokens, use_cache=False, return_dict=True, output_hidden_states=True)
            hidden = torch.stack([
                pool_qwen_hidden_states(layer, tokens['attention_mask'], pooling='last')[0].float()
                for layer in output.hidden_states
            ])
            logits = model.photoz_head(hidden[-1:])
    finally:
        for handle in handles:
            handle.remove()
    _, centers = make_redshift_grid(config['z_min'], config['z_max'], config['n_z_bins'])
    prediction = predict_photoz_from_logits(logits.float().cpu(), centers=centers)
    return {'median': float(prediction['z_p50'][0]), 'mean': float(prediction['z_mean'][0]),
            'pz': prediction['pz'][0], 'hidden': hidden.cpu(),
            'tokens': tokens['input_ids'][0].cpu().tolist(), 'routes': captured}


def token_and_route_changes(before, after):
    matcher = SequenceMatcher(a=before['tokens'], b=after['tokens'], autojunk=False)
    matches = [(i + offset, j + offset) for i, j, length in matcher.get_matching_blocks()
               for offset in range(length)]
    common = sum(block.size for block in matcher.get_matching_blocks())
    token_edit = max(len(before['tokens']), len(after['tokens'])) - common
    changes, aligned_changes, logit_changes = [], [], []
    for name, route in before['routes'].items():
        other = after['routes'][name]
        a, b = set(route['ids'][-1].tolist()), set(other['ids'][-1].tolist())
        changes.append(1 - len(a & b) / len(a | b))
        logit_changes.append(float((other['logits'][-1] - route['logits'][-1]).norm()
                                   / route['logits'][-1].norm().clamp_min(1e-12)))
        for i, j in matches:
            a, b = set(route['ids'][i].tolist()), set(other['ids'][j].tolist())
            aligned_changes.append(1 - len(a & b) / len(a | b))
    return {'token_changed': before['tokens'] != after['tokens'],
            'token_count_delta': len(after['tokens']) - len(before['tokens']),
            'unmatched_token_fraction': token_edit / max(len(before['tokens']), len(after['tokens'])),
            'routing_last_jaccard_distance': float(np.mean(changes)) if changes else None,
            'routing_aligned_jaccard_distance': float(np.mean(aligned_changes)) if aligned_changes else None,
            'router_logits_relative_l2': float(np.mean(logit_changes)) if logit_changes else None}


def compare_snapshots(before, after):
    relative = (after['hidden'] - before['hidden']).norm(dim=1) / before['hidden'].norm(dim=1).clamp_min(1e-12)
    result = token_and_route_changes(before, after)
    result.update(
        delta_z=after['median'] - before['median'], delta_z_mean=after['mean'] - before['mean'],
        pdf_total_variation=float(.5 * (after['pz'] - before['pz']).abs().sum()),
        hidden_relative_l2=float(relative[-1]), hidden_layer_relative_l2=relative.tolist(),
        hidden_cosine_distance=float((1 - F.cosine_similarity(before['hidden'][-1:], after['hidden'][-1:])).clamp_min(0)),
    )
    return result


@torch.inference_mode()
def mlp_prediction(values, saved, model, centers, device):
    logits = model(numeric_input(values, saved)[None].to(device)).float().cpu()
    predictions = predict_photoz_from_logits(logits, centers=centers)
    return {'median': float(predictions['z_p50'][0]), 'mean': float(predictions['z_mean'][0]),
            'pz': predictions['pz'][0]}


def cases_for(product, rows, steps):
    cases = []
    for row in rows:
        for band_index, band in enumerate(('g', 'r', 'i', 'z', 'y')):
            for step in steps:
                for sign in (-1, 1):
                    delta = sign * step
                    changed = perturb_magnitude(product['extra_features'][row], band_index, delta)
                    if changed is not None:
                        flux, error = product['extra_features'][row, 2 * band_index:2 * band_index + 2]
                        mag_error = 2.5 / np.log(10) * float(error / flux)
                        cases.append({'row': int(row), 'object_id': product['object_id'][row],
                                      'band': band, 'delta_mag': delta,
                                      'delta_mag_in_sigma': delta / mag_error,
                                      'true_z': float(product['z_spec'][row]), 'values': changed})
    return cases


def group_statistics(records, model_names, steps, threshold):
    table = []
    for name in model_names:
        for step in steps:
            selected = [r for r in records if r['model'] == name and abs(r['delta_mag']) == step]
            if not selected:
                continue
            dz = np.asarray([abs(r['delta_z']) for r in selected])
            normalized = np.asarray([abs(r['delta_z']) / (1 + r['true_z']) for r in selected])
            table.append({'model': name, 'step_mag': step, 'n': len(selected),
                          'median_abs_delta_z': float(np.median(dz)), 'p95_abs_delta_z': float(np.quantile(dz, .95)),
                          'max_abs_delta_z': float(dz.max()), 'flagged_fraction': float(np.mean(normalized > threshold)),
                          'median_pdf_tv': float(np.median([r['pdf_total_variation'] for r in selected]))})
    return table


def make_plots(folder, records, statistics, steps):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in dict.fromkeys(row['model'] for row in statistics):
        rows = [row for row in statistics if row['model'] == name]
        axes[0].plot([r['step_mag'] for r in rows], [r['p95_abs_delta_z'] for r in rows], 'o-', label=name)
        axes[1].plot([r['step_mag'] for r in rows], [r['median_pdf_tv'] for r in rows], 'o-', label=name)
    for axis in axes:
        axis.set_xscale('log'); axis.set_xlabel('|magnitude perturbation|'); axis.legend()
    axes[0].set_ylabel('95th percentile |delta PDF median z|')
    axes[1].set_ylabel('Median PDF total variation')
    fig.tight_layout(); fig.savefig(folder / 'prediction_sensitivity.png', dpi=160); plt.close(fig)
    glm_records = [r for r in records if r['model'] == 'GLM QLoRA']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    scatter = axes[0].scatter([r['hidden_relative_l2'] for r in glm_records],
                             [abs(r['delta_z']) for r in glm_records],
                             c=[r['routing_last_jaccard_distance'] for r in glm_records], s=8, alpha=.5)
    axes[0].set_xlabel('Relative last hidden-state L2 change'); axes[0].set_ylabel('|delta median z|')
    fig.colorbar(scatter, ax=axes[0], label='Last-token routing Jaccard distance')
    axes[1].scatter([r['unmatched_token_fraction'] for r in glm_records],
                    [r['pdf_total_variation'] for r in glm_records], s=8, alpha=.5)
    axes[1].set_xlabel('Unmatched token fraction'); axes[1].set_ylabel('PDF total variation')
    fig.tight_layout(); fig.savefig(folder / 'token_representation_routing.png', dpi=160); plt.close(fig)
    matrix = np.asarray([np.mean([r['hidden_layer_relative_l2'] for r in glm_records
                                 if abs(r['delta_mag']) == step], axis=0) for step in steps])
    fig, axis = plt.subplots(figsize=(8, 4))
    im = axis.imshow(matrix, aspect='auto', origin='lower')
    axis.set_yticks(range(len(steps)), [str(step) for step in steps])
    axis.set_xlabel('Hidden-state index (0 = input embeddings)'); axis.set_ylabel('|delta magnitude|')
    fig.colorbar(im, ax=axis, label='Mean relative last-token L2 change')
    fig.tight_layout(); fig.savefig(folder / 'hidden_layers.png', dpi=160); plt.close(fig)
    worst = max(glm_records, key=lambda r: abs(r['delta_z']) / (1 + r['true_z']))
    fig, axis = plt.subplots(figsize=(8, 4))
    for name in dict.fromkeys(r['model'] for r in records):
        rows = sorted([r for r in records if r['model'] == name and r['object_id'] == worst['object_id']
                       and r['band'] == worst['band']], key=lambda r: r['delta_mag'])
        axis.plot([r['delta_mag'] for r in rows], [r['delta_z'] for r in rows], 'o-', label=name)
    axis.axhline(0, color='gray', lw=.7); axis.set_xlabel('Signed magnitude perturbation')
    axis.set_ylabel('Delta PDF median z'); axis.set_title(f"Largest GLM response: {worst['object_id']}, band {worst['band']}")
    axis.legend(); fig.tight_layout(); fig.savefig(folder / 'largest_response.png', dpi=160); plt.close(fig)


def report(folder, summary, records):
    table = summary['statistics']
    lines = ['# GLM numerical sensitivity: fixed validation subset', '',
             f"Objects: {summary['sample_size']}; seed: {summary['seed']}; target: DESI spectroscopic redshift.", '',
             'Inference only. No model was trained and no held-out test object was evaluated.', '',
             '## Protocol', '',
             'Each grizy magnitude is perturbed individually in both directions. Its native magnitude error is held fixed, so converted flux and flux error scale by the same factor. EBV, missingness, saved imputation/scaling, text formatting, and model weights remain fixed. Nonfinite bands and perturbations outside the native valid magnitude range are skipped.', '',
             'GLM uses the saved corrected-implementation **512-object training-fit checkpoint**, evaluated here on validation objects. This is not a full-cohort GLM generalization run. MLP and optional Qwen comparisons are sensitivity controls with different training budgets.', '',
             'Token sequences are compared before/after perturbation. Hidden changes use the last non-padding token at every hidden-state output. MoE routing compares actual selected expert sets at the last token and at sequence-aligned unchanged tokens; expert order is ignored. Router logits are also compared. These observations establish associations, not causal mediation.', '',
             '## Prediction comparisons', '',
             f"Flag threshold: |delta median z|/(1+spec-z) > {summary['jump_threshold']}; a descriptive threshold, not a significance test.", '',
             '| Model | Step (mag) | Pairs | Median abs delta z | P95 abs delta z | Maximum | Flagged | Median PDF TV |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append(f"| {row['model']} | {row['step_mag']:g} | {row['n']} | {row['median_abs_delta_z']:.5g} | {row['p95_abs_delta_z']:.5g} | {row['max_abs_delta_z']:.5g} | {100*row['flagged_fraction']:.2f}% | {row['median_pdf_tv']:.5g} |")
    glm_rows = [r for r in records if r['model'] == 'GLM QLoRA']
    changed_tokens = sum(r['token_changed'] for r in glm_rows)
    changed_routes = sum(r['routing_last_jaccard_distance'] > 0 for r in glm_rows)
    lines += ['', '## Tokenisation, representations, and routing', '',
              f"Token sequences changed in {changed_tokens}/{len(glm_rows)} perturbations; last-token MoE expert sets changed in {changed_routes}/{len(glm_rows)}.", '',
              '| GLM condition | Pairs | P95 abs delta z | Median hidden relative L2 | Median PDF TV |',
              '|---|---:|---:|---:|---:|']
    for token_changed in (False, True):
        for routing_changed in (False, True):
            rows = [r for r in glm_rows if r['token_changed'] == token_changed
                    and (r['routing_last_jaccard_distance'] > 0) == routing_changed]
            if rows:
                lines.append(f"| Tokens {'changed' if token_changed else 'same'}, routing {'changed' if routing_changed else 'same'} | {len(rows)} | {np.quantile([abs(r['delta_z']) for r in rows],.95):.5g} | {np.median([r['hidden_relative_l2'] for r in rows]):.5g} | {np.median([r['pdf_total_variation'] for r in rows]):.5g} |")
    flags = [r for r in glm_rows if abs(r['delta_z']) / (1 + r['true_z']) > summary['jump_threshold']]
    lines += ['', '## Interpretation and limits', '',
              f"{len(flags)}/{len(glm_rows)} GLM perturbations exceeded the stated response threshold. "
              + ('Large finite-step responses occur in this subset; inspect the matched controls and individual cases before attributing them to numerical boundaries.' if flags else 'No response exceeded that threshold in this subset; this does not prove global continuity.'), '',
              f"The maximum repeated-identical-input GLM median-z difference was {summary['repeat_noise']['max_abs_delta_z']:.5g}; maximum PDF TV was {summary['repeat_noise']['max_pdf_tv']:.5g}.", '',
              'A changed text token is an input boundary; hidden-state movement measures its propagated representation effect. Expert-set changes indicate routing boundaries, but occur downstream of the same changed input. Correlation with prediction changes cannot by itself distinguish routing causation from representation changes. No routing-freezing counterfactual is performed.', '',
              'PDF medians are quantized to the saved redshift-bin centers. A median-bin jump alone is not evidence of a discontinuous model function: compare continuous PDF TV and mean-redshift changes in diagnostics.csv. The sweep detects finite differences; it does not prove mathematical discontinuities or explain generalization causally.', '',
              'Results are limited to a fixed validation subset, these perturbation sizes, and available checkpoints. Changes relative to catalogue magnitude errors are recorded as delta_mag_in_sigma; no universal physical threshold is assumed.', '',
              f"Optional Qwen: {summary['qwen_status']}", '',
              '## Plots and artifacts', '',
              '![Prediction response](prediction_sensitivity.png)', '',
              '![Token, representation, and routing associations](token_representation_routing.png)', '',
              '![Hidden layer response](hidden_layers.png)', '',
              '![Largest GLM response](largest_response.png)', '',
              'Per-pair measurements: [diagnostics.csv](diagnostics.csv). Largest GLM cases and token sequences: [largest_cases.json](largest_cases.json). Fixed subset/provenance: [manifest.json](manifest.json). Aggregate values: [summary.json](summary.json).', '']
    (folder / 'report.md').write_text('\n'.join(lines))


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if args.sample_size < 1 or not args.mag_steps or any(step <= 0 or not np.isfinite(step) for step in args.mag_steps) or args.jump_threshold <= 0:
        p.error('Sample size, finite magnitude steps, and response threshold must be positive.')
    steps = sorted(set(args.mag_steps))
    _, product, preparation, bounds, _, _ = capacity.load_galaxiesml_product(args.survey_prepared_json)
    glm_cohort = verify_cohort(args.glm_run_dir, product, 'GLM')
    verify_cohort(args.mlp_run_dir, product, 'MLP')
    rows = select_validation(product, args.sample_size, args.seed)
    cases = cases_for(product, rows, steps)
    if not cases:
        raise ValueError('No valid perturbations for the selected validation subset.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'arguments': vars(args), 'validation_object_ids': [product['object_id'][row] for row in rows],
                'validation_rows': rows.tolist(), 'target': preparation['target'], 'cohort_cache_key': preparation['cache_key'],
                'feature_names': product['feature_names'], 'test_evaluated': False, 'models_retrained': False,
                'glm_training_objects': glm_cohort['arguments']['train_objects'],
                'glm_training_scope': 'small-sample training-fit checkpoint', 'perturbation_pairs': len(cases)}
    # CPU audit is available even while the preceding GLM training is finishing.
    from transformers import AutoTokenizer
    config = dict(glm_cohort['reference_config'], epochs=glm_cohort['epochs'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True, trust_remote_code=True)
    longest = max(len(tokenizer(glm_text(case['values'], product['feature_names']), truncation=False)['input_ids']) for case in cases)
    if longest > config['max_length']:
        raise ValueError('Perturbed text would be truncated under the saved numerical format.')
    manifest['token_audit'] = {'longest_input_tokens': longest, 'max_length': config['max_length'], 'passed': True}
    write_json(args.output_dir / 'manifest.json', manifest)
    if args.prepare_only:
        print(f'CPU audit passed: {len(rows)} validation objects, {len(cases)} pairs, longest {longest} tokens.'); return 0
    required = [args.glm_run_dir / name for name in ('result.pt', 'adapter/adapter_config.json', 'photoz_head.pt')]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f'Wait for the preceding GLM run to finish: missing {path}')
    saved_glm = torch.load(args.glm_run_dir / 'result.pt', map_location='cpu', weights_only=False)
    if saved_glm['metadata']['config'] != dict(config, device=saved_glm['metadata']['config']['device']):
        raise ValueError('GLM saved config differs from its cohort training contract.')
    set_random_seed(args.seed)
    records, largest, repeat_noise = [], [], []
    glm_model, tokenizer = load_policy(args.glm_run_dir, config, args.device)
    baseline = {}
    for row in rows:
        text = glm_text(product['extra_features'][row], product['feature_names'])
        before = snapshot(glm_model, tokenizer, text, config, args.device, routes=True)
        repeated = snapshot(glm_model, tokenizer, text, config, args.device, routes=True)
        repeat_noise.append(compare_snapshots(before, repeated)); baseline[int(row)] = before
    for index, case in enumerate(cases):
        before = baseline[case['row']]
        text = glm_text(case['values'], product['feature_names'])
        after = snapshot(glm_model, tokenizer, text, config, args.device, routes=True)
        record = {key: value for key, value in case.items() if key != 'values'}
        record.update(model='GLM QLoRA', **compare_snapshots(before, after)); records.append(record)
        largest.append(dict(record, baseline_tokens=before['tokens'], perturbed_tokens=after['tokens'],
                            baseline_text=glm_text(product['extra_features'][case['row']], product['feature_names']), perturbed_text=text))
        largest = sorted(largest, key=lambda r: abs(r['delta_z']) / (1 + r['true_z']), reverse=True)[:10]
        if (index + 1) % 50 == 0:
            print(f'GLM inference pairs {index+1}/{len(cases)}', flush=True)
    del glm_model, baseline; gc.collect(); torch.cuda.empty_cache()
    saved_mlp = torch.load(args.mlp_run_dir / 'result.pt', map_location='cpu', weights_only=False)
    if saved_mlp['raw_feature_names'] != product['feature_names'] or saved_mlp['catalogue_kind'] != 'galaxiesml-spec':
        raise ValueError('MLP saved preprocessing does not match the survey.')
    _, centers = make_redshift_grid(*bounds, config['n_z_bins'])
    mlp = build_baseline_model('tabular', 0, len(saved_mlp['feature_names']), config['n_z_bins']).to(args.device).eval()
    mlp.load_state_dict(saved_mlp['state_dict']); mlp.requires_grad_(False)
    baseline = {int(row): mlp_prediction(product['extra_features'][row], saved_mlp, mlp, centers, args.device) for row in rows}
    for case in cases:
        before = baseline[case['row']]; after = mlp_prediction(case['values'], saved_mlp, mlp, centers, args.device)
        records.append(dict({key: value for key, value in case.items() if key != 'values'}, model='Numeric MLP',
                            delta_z=after['median']-before['median'], delta_z_mean=after['mean']-before['mean'],
                            pdf_total_variation=float(.5*(after['pz']-before['pz']).abs().sum())))
    del mlp; gc.collect(); torch.cuda.empty_cache()
    qwen_status = 'Not requested.'
    if args.qwen_run_dir:
        prepared = json.loads((args.qwen_run_dir / 'prepared.json').read_text())
        if prepared['cache_key'] != preparation['cache_key']:
            raise ValueError('Qwen belongs to another catalogue cohort.')
        candidate = args.qwen_run_dir / prepared['method']
        frozen_path = args.qwen_run_dir / 'frozen/aion_baseline.pt'
        frozen = None
        adapter_ready = all((candidate / name).is_file() for name in ('result.pt', 'adapter/adapter_config.json', 'photoz_head.pt'))
        if adapter_ready:
            qwen_status = f"Saved {prepared['method']} adapter and head: {candidate}"
        elif frozen_path.is_file():
            frozen = torch.load(frozen_path, map_location='cpu', weights_only=False)
            if frozen['metadata']['cache_key'] != preparation['cache_key']:
                raise ValueError('Qwen frozen-control checkpoint belongs to another cohort.')
            qwen_status = f'Saved frozen-Qwen + trained PDF-head control: {frozen_path}'
        else:
            qwen_status = 'Unavailable: no completed adapter/head or frozen-control checkpoint.'
        if frozen is not None or adapter_ready:
            model, tokenizer = load_policy(candidate, prepared['training'], args.device, frozen_checkpoint=frozen)
            qargs = survey.parser().parse_args(['--stage','prepare','--catalogue-kind','galaxiesml-spec',
                                               '--method',prepared['method'],'--output-dir',str(args.qwen_run_dir)])
            def text_for(values):
                return survey.texts_for(qargs, {'features': values[None].numpy()}, preparation)[0]
            baseline = {int(row): snapshot(model, tokenizer, text_for(product['extra_features'][row]), prepared['training'], args.device) for row in rows}
            for index, case in enumerate(cases):
                after = snapshot(model, tokenizer, text_for(case['values']), prepared['training'], args.device)
                records.append(dict({key:value for key,value in case.items() if key!='values'}, model='Qwen control' if frozen is not None else 'Qwen adapter',
                                    **compare_snapshots(baseline[case['row']], after)))
                if (index+1) % 50 == 0:
                    print(f'Qwen inference pairs {index+1}/{len(cases)}', flush=True)
            del model; gc.collect(); torch.cuda.empty_cache()
    fieldnames = list(dict.fromkeys(key for record in records for key in record))
    with (args.output_dir / 'diagnostics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader(); writer.writerows(records)
    statistics = group_statistics(records, list(dict.fromkeys(r['model'] for r in records)), steps, args.jump_threshold)
    summary = {'sample_size': len(rows), 'seed': args.seed, 'jump_threshold': args.jump_threshold,
               'statistics': statistics, 'qwen_status': qwen_status,
               'repeat_noise': {'max_abs_delta_z': max(abs(r['delta_z']) for r in repeat_noise),
                                'max_pdf_tv': max(r['pdf_total_variation'] for r in repeat_noise)},
               'test_evaluated': False, 'models_retrained': False}
    write_json(args.output_dir / 'summary.json', summary); write_json(args.output_dir / 'largest_cases.json', largest)
    make_plots(args.output_dir, records, statistics, steps); report(args.output_dir, summary, records)
    print(f"Report: {args.output_dir / 'report.md'}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
