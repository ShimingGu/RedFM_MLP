#!/usr/bin/env python3
"""GLM training-fit sanity check and numeric MLP on a shared catalogue cohort."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import glm52_posttraining as glm
from aion_magnitude import survey_catalogues as surveys
from aion_magnitude.Inference_Opt_TFM import CatalogueSerializationConfig
from aion_magnitude.metrics import summarize_pdf_metrics
from aion_magnitude.models import build_baseline_model
from aion_magnitude.qwen_posttraining import (
    QwenPosttrainingConfig, TextRedshiftDataset, train_qlora_photoz,
)
from aion_magnitude.training import (
    evaluate_model_on_dataset, make_cached_loader, split_cached_product,
    train_pdf_model_one_epoch,
)
from aion_magnitude.utils import make_redshift_grid, set_random_seed


DEFAULT_REFERENCE = Path(
    '/arc/home/gsm/aion_output/figures/glm52-posttraining-paddingfix-0905/photometry'
)
DEFAULT_SURVEY_PREPARED = Path(
    '/arc/home/gsm/aion_output/figures/qwen-galaxiesml-spec-qlora-specz-e10/prepared.json'
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', choices=('overfit', 'numeric-mlp'), required=True)
    parser.add_argument('--catalogue-kind', choices=('clauds', 'galaxiesml-spec'), default='clauds')
    parser.add_argument('--survey-prepared-json', type=Path, default=DEFAULT_SURVEY_PREPARED)
    parser.add_argument('--reference-dir', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--train-objects', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--mlp-batch-size', type=int, default=256)
    parser.add_argument('--mlp-learning-rate', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--prepare-only', action='store_true')
    return parser


def validate_match(product, prepared, reference, bounds):
    labels = np.asarray(product['split_labels'], dtype=object)
    counts = {split: int(np.sum(labels == split)) for split in ('train', 'val', 'test')}
    if counts != prepared['rows']:
        raise ValueError('Reconstructed GLM split counts differ from the reference.')
    if not np.allclose(bounds, prepared['redshift_bounds'], rtol=0, atol=1e-7):
        raise ValueError('Reconstructed GLM redshift bounds differ from the reference.')
    # Check saved row identities and labels; no test predictions are calculated.
    for split in ('val', 'test'):
        rows = np.flatnonzero(labels == split)
        saved = reference[f'{split}_evaluation']
        ids = [str(product['object_id'][row]) for row in rows]
        if ids != [str(value) for value in saved['object_id']]:
            raise ValueError(f'Reconstructed {split} object IDs differ from GLM.')
        z = torch.as_tensor(product['z_spec'])[rows]
        if not torch.equal(z.float(), saved['z_spec'].float()):
            raise ValueError(f'Reconstructed {split} labels differ from GLM.')
    config = reference['metadata']['config']
    if not np.allclose(bounds, [config['z_min'], config['z_max']], rtol=0, atol=1e-7):
        raise ValueError('Reference QLoRA redshift bounds differ from the cohort.')
    return labels


def load_matched_product(reference_dir):
    print(f'Rebuilding and checking the corrected GLM cohort from {reference_dir}', flush=True)
    prepared = json.loads((reference_dir / 'prepared.json').read_text())
    reference = torch.load(
        reference_dir / 'qlora/result.pt', map_location='cpu', weights_only=False,
    )
    config = reference['metadata']['config']
    counts = prepared['rows']
    total = sum(counts.values())
    command = [
        '--stage', 'prepare', '--catalogue', prepared['catalogue'],
        '--max-rows', str(prepared['max_rows']), '--seed', str(config['seed']),
        '--glm-model', prepared['resolved_model_path'],
        '--glm-max-length', str(config['max_length']),
        '--n-z-bins', str(config['n_z_bins']),
        '--train-fraction', str(counts['train'] / total),
        '--val-fraction', str(counts['val'] / total),
        '--test-fraction', str(counts['test'] / total),
    ]
    if prepared['use_morphology']:
        command.append('--use-morphology')
    pipeline_args = glm.build_parser().parse_args(command)
    product, preparation, bounds = glm.catalogue_pipeline.build_product(pipeline_args)
    labels = validate_match(product, prepared, reference, bounds)
    return pipeline_args, product, preparation, bounds, reference, labels


def load_galaxiesml_product(prepared_path):
    """Reuse the quality-selected native spec-z cohort, without CLAUDS artifacts."""
    prepared = json.loads(prepared_path.read_text())
    if (prepared['catalogue_kind'] != 'galaxiesml-spec'
            or prepared['target'] != 'DESI_fibermap/DESI_redshift'):
        raise ValueError('GalaxiesML-Spec requires native DESI spectroscopic redshift labels.')
    folder = Path(prepared['cohort_cache'])
    metadata = json.loads((folder / 'cohort.json').read_text())
    for key in ('cache_key', 'catalogue_kind', 'target', 'feature_names', 'split_counts', 'z_bounds', 'seed', 'sources'):
        if metadata[key] != prepared[key]:
            raise ValueError(f'Survey cohort metadata differs from prepared reference: {key}')
    if surveys.fingerprints([Path(source['path']) for source in metadata['sources']]) != metadata['sources']:
        raise ValueError('Survey source files changed; rebuild the prepared cohort.')
    with np.load(folder / 'cohort.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    labels = np.asarray(data['split'], dtype=object)
    counts = {split: int(np.sum(labels == split)) for split in ('train', 'val', 'test')}
    if counts != metadata['split_counts'] or sum(counts.values()) != len(labels):
        raise ValueError('Survey split assignments differ from prepared counts.')
    if data['features'].shape != (len(labels), len(metadata['feature_names'])):
        raise ValueError('Survey features differ from the declared schema.')
    bounds = tuple(metadata['z_bounds'])
    if not np.all(np.isfinite(data['z']) & (data['z'] >= bounds[0]) & (data['z'] <= bounds[1])):
        raise ValueError('Survey spectroscopic redshifts are invalid or outside the cohort bounds.')
    if len(np.unique(data['group'])) != len(labels) or len(np.unique(data['targetid'])) != len(labels):
        raise ValueError('Survey cohort contains duplicate imaging groups or target IDs.')
    product = {
        'object_id': [str(value) for value in data['targetid']],
        'field': ['galaxiesml-spec'] * len(labels),
        'aion_embedding': torch.empty(len(labels), 0),
        'extra_features': torch.from_numpy(data['features']).float(),
        'feature_names': metadata['feature_names'],
        'z_spec': torch.from_numpy(data['z']).float(),
        'split_labels': labels.tolist(), 'metadata': metadata,
    }
    config = QwenPosttrainingConfig(
        model_path=glm.resolve_model_path(glm.DEFAULT_GLM52_MODEL),
        max_length=512, n_z_bins=300, z_min=bounds[0], z_max=bounds[1],
        batch_size=8, gradient_accumulation_steps=2,
        lora_target_modules=glm.GLM52_LORA_TARGET_SETTING, seed=metadata['seed'],
    ).normalized()
    # No full-cohort GLM validation reference exists for these two checks yet.
    reference = {'metadata': {'config': asdict(config)}, 'final_metrics': {}}
    pipeline_args = SimpleNamespace(catalogue_kind='galaxiesml-spec')
    return pipeline_args, product, metadata, bounds, reference, labels


def data_source(args):
    if getattr(args, 'catalogue_kind', 'clauds') == 'galaxiesml-spec':
        return str(args.survey_prepared_json)
    return str(args.reference_dir)


def galaxiesml_serialization():
    return CatalogueSerializationConfig(
        schema_name='galaxiesml_spec_glm52_native_photometry_specz_v1',
        prefix='galaxy native_observed_photometry flux_unit=AB_nanomaggy errors=one_sigma extinction=as_catalogued',
    )


def run_overfit(args, pipeline_args, product, reference, labels):
    rows = np.flatnonzero(labels == 'train')
    if not 512 <= args.train_objects <= 2048 or args.train_objects > len(rows):
        raise ValueError('Choose 512–2,048 objects from the existing training split.')
    seed = reference['metadata']['config']['seed']
    rows = np.random.default_rng(seed).choice(rows, args.train_objects, replace=False)
    rows.sort()
    small = dict(product)
    small['extra_features'] = torch.as_tensor(product['extra_features'])[rows]
    if getattr(pipeline_args, 'catalogue_kind', None) == 'galaxiesml-spec':
        serialization = galaxiesml_serialization()
    else:
        serialization = glm._serialization(pipeline_args)
    texts = glm._serialize_product(small, serialization)
    dataset = TextRedshiftDataset(
        texts, torch.as_tensor(product['z_spec'])[rows],
        [str(product['object_id'][row]) for row in rows],
    )
    (args.output_dir / 'training_sample.json').write_text(
        json.dumps({'object_id': dataset.object_ids, 'evaluation_split': 'training_sample'}, indent=2) + '\n'
    )
    config = replace(
        QwenPosttrainingConfig(**reference['metadata']['config']),
        epochs=args.epochs, device=args.device,
    ).normalized()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True, trust_remote_code=True)
    longest = max(len(tokenizer(text, truncation=False)['input_ids']) for text in texts)
    if longest > config.max_length:
        raise ValueError(f'Sanity-check inputs need {longest} tokens; max_length={config.max_length}.')
    audit = {'rows_checked': len(texts), 'longest_input_tokens': longest,
             'max_length': config.max_length, 'passed': True}
    (args.output_dir / 'token_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if args.prepare_only:
        return {'train_object_ids': dataset.object_ids, 'evaluation_split': 'training_sample',
                'token_audit': audit}
    # The trainer selects on its "val" argument and evaluates its "test" argument.
    # Both deliberately receive the SAME training sample for this diagnostic.
    result = train_qlora_photoz(
        train_dataset=dataset, val_dataset=dataset, test_dataset=dataset,
        output_dir=args.output_dir, config=config,
        checkpoint_dir=args.output_dir / 'checkpoints', resume=False,
    )
    result['train_evaluation'] = result.pop('val_evaluation')
    result.pop('test_evaluation')
    result['final_metrics'] = {'train': result['final_metrics']['val']}
    result['history'] = [
        {key.replace('val_', 'train_eval_', 1) if key.startswith('val_') else key: value
         for key, value in row.items()} for row in result['history']
    ]
    result['metadata'].update(
        evaluation_split='training_sample', measures_generalization=False,
        train_object_ids=dataset.object_ids, data_source=data_source(args),
        catalogue_kind=getattr(args, 'catalogue_kind', 'clauds'),
    )
    torch.save(result, args.output_dir / 'result.pt')
    return {'train_object_ids': dataset.object_ids, 'final_metrics': result['final_metrics'],
            'evaluation_split': 'training_sample'}


def run_mlp(args, product, bounds, reference, labels):
    config = reference['metadata']['config']
    features = torch.as_tensor(product['extra_features'], dtype=torch.float32)
    train_rows = np.flatnonzero(labels == 'train')
    feature_names = list(product['feature_names'])
    imputation = None
    if product.get('metadata', {}).get('catalogue_kind') == 'galaxiesml-spec':
        # GLM receives explicit NA values. Give the MLP the same missingness information.
        finite = torch.isfinite(features)
        fills = []
        for column in range(features.shape[1]):
            values = features[train_rows, column]
            values = values[torch.isfinite(values)]
            fills.append(values.median() if len(values) else torch.tensor(0.))
        imputation = torch.stack(fills)
        features = torch.cat((torch.where(finite, features, imputation), (~finite).float()), dim=1)
        feature_names += [name + '_missing' for name in product['feature_names']]
    mean = features[train_rows].mean(0)
    scale = features[train_rows].std(0, unbiased=False).clamp_min(1e-6)
    numeric = dict(product)
    numeric['extra_features'] = (features - mean) / scale
    numeric['feature_names'] = feature_names
    # The original test rows are retained in the manifest but excluded from loaders.
    numeric['split_labels'] = ['unused' if label == 'test' else label for label in labels]
    train, val, _ = split_cached_product(numeric)
    if args.prepare_only:
        return {'train_rows': len(train), 'val_rows': len(val), 'test_evaluated': False}
    set_random_seed(config['seed'])
    model = build_baseline_model('tabular', 0, features.shape[1], config['n_z_bins'])
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.mlp_learning_rate, weight_decay=1e-4)
    edges, centers = make_redshift_grid(*bounds, config['n_z_bins'])
    loader = make_cached_loader(train, args.mlp_batch_size, shuffle=True)
    history, best_loss, best_state = [], float('inf'), None
    for epoch in range(args.epochs):
        loss = train_pdf_model_one_epoch(
            model, loader, optimizer, model_kind='tabular', device=args.device, redshift_edges=edges,
        )
        with torch.no_grad():
            evaluation = evaluate_model_on_dataset(
                model, val, 'tabular', batch_size=512, device=args.device,
                redshift_edges=edges, redshift_centers=centers,
            )
        metrics = summarize_pdf_metrics(evaluation)
        history.append({'epoch': epoch + 1, 'train_loss': loss, 'val': metrics})
        print(json.dumps(history[-1]), flush=True)
        if metrics['cross_entropy'] < best_loss:
            best_loss = metrics['cross_entropy']
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_evaluation, best_metrics, best_epoch = evaluation, metrics, epoch + 1
    best_evaluation['object_id'] = [str(value) for value in val.object_ids]
    result = {
        'model_kind': 'matched_numeric_mlp', 'state_dict': best_state,
        'feature_mean': mean, 'feature_scale': scale, 'feature_names': feature_names,
        'imputation_values': imputation, 'raw_feature_names': product['feature_names'],
        'history': history, 'selected_epoch': best_epoch, 'val_evaluation': best_evaluation,
        'final_metrics': {'val': best_metrics}, 'test_evaluated': False,
        'data_source': data_source(args),
        'catalogue_kind': getattr(args, 'catalogue_kind', 'clauds'),
    }
    torch.save(result, args.output_dir / 'result.pt')
    summary = {'final_metrics': result['final_metrics'], 'selected_epoch': best_epoch, 'test_evaluated': False}
    if 'val' in reference['final_metrics']:
        summary['glm_qlora_val_metrics'] = reference['final_metrics']['val']
    return summary


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.mlp_batch_size < 1 or args.mlp_learning_rate <= 0:
        parser.error('Epochs, batch size, and learning rate must be positive.')
    if args.catalogue_kind == 'galaxiesml-spec':
        pipeline_args, product, preparation, bounds, reference, labels = load_galaxiesml_product(args.survey_prepared_json)
    else:
        pipeline_args, product, preparation, bounds, reference, labels = load_matched_product(args.reference_dir)
    previous_cohort = args.output_dir / 'cohort.json'
    if previous_cohort.exists():
        previous_kind = json.loads(previous_cohort.read_text()).get('catalogue_kind', 'clauds')
        if previous_kind != args.catalogue_kind:
            raise ValueError('Output directory belongs to another catalogue; choose a separate directory.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'check': args.check, 'data_source': data_source(args), 'catalogue_kind': args.catalogue_kind,
        'preparation': preparation, 'redshift_bounds': list(bounds),
        'seed': reference['metadata']['config']['seed'],
        'object_id': [str(value) for value in product['object_id']],
        'split_labels': product['split_labels'], 'test_evaluated': False,
        'epochs': args.epochs, 'prepare_only': args.prepare_only,
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        'reference_config': reference['metadata']['config'],
    }
    (args.output_dir / 'cohort.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if args.check == 'overfit':
        summary = run_overfit(args, pipeline_args, product, reference, labels)
    else:
        summary = run_mlp(args, product, bounds, reference, labels)
    summary.update(check=args.check, catalogue_kind=args.catalogue_kind,
                   target=preparation.get('target', 'ZPHOT'),
                   prepare_only=args.prepare_only, test_evaluated=False)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    display = {key: value for key, value in summary.items() if key != 'train_object_ids'}
    if 'train_object_ids' in summary:
        display['train_objects'] = len(summary['train_object_ids'])
    print(json.dumps(display, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
