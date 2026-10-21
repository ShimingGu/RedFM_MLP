import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch


spec = importlib.util.spec_from_file_location(
    'glm52_capacity_checks', Path(__file__).resolve().parents[1] / 'notebooks/glm52_capacity_checks.py'
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def fixture_product(n_train=512):
    count = n_train + 2
    return {
        'object_id': [str(index) for index in range(count)],
        'field': ['COSMOS'] * count,
        'extra_features': torch.arange(count * 2, dtype=torch.float32).reshape(count, 2),
        'feature_names': ['g_mag', 'r_mag'],
        'aion_embedding': torch.empty(count, 0),
        'z_spec': torch.linspace(.01, 1.9, count),
        'split_labels': ['train'] * n_train + ['val', 'test'],
        'metadata': {},
    }


def fixture_reference(product):
    result = {'metadata': {'config': {'seed': 42, 'z_min': 0., 'z_max': 2., 'n_z_bins': 4}},
              'final_metrics': {'val': {'cross_entropy': 2.}}}
    for split in ('val', 'test'):
        rows = [index for index, label in enumerate(product['split_labels']) if label == split]
        result[f'{split}_evaluation'] = {
            'object_id': [product['object_id'][index] for index in rows],
            'z_spec': product['z_spec'][rows],
        }
    return result


def test_match_rejects_different_objects_and_labels():
    product = fixture_product()
    reference = fixture_reference(product)
    prepared = {'rows': {'train': 512, 'val': 1, 'test': 1}, 'redshift_bounds': [0., 2.]}
    driver.validate_match(product, prepared, reference, (0., 2.))
    reference['val_evaluation']['object_id'] = ['different']
    with pytest.raises(ValueError, match='object IDs'):
        driver.validate_match(product, prepared, reference, (0., 2.))
    reference = fixture_reference(product)
    reference['val_evaluation']['z_spec'] += 1
    with pytest.raises(ValueError, match='labels'):
        driver.validate_match(product, prepared, reference, (0., 2.))


def test_overfit_evaluates_only_the_selected_training_objects(tmp_path):
    product = fixture_product(600)
    reference = fixture_reference(product)
    args = SimpleNamespace(train_objects=512, epochs=100, device='cpu',
                           prepare_only=False, output_dir=tmp_path, reference_dir=tmp_path)
    result = {'val_evaluation': {}, 'test_evaluation': {},
              'final_metrics': {'val': {'cross_entropy': .1}, 'test': {}},
              'history': [{'val_cross_entropy': .1}], 'metadata': {}}
    with patch.object(driver.glm, '_serialization'), patch.object(
        driver.glm, '_serialize_product', return_value=['galaxy'] * 512
    ), patch('transformers.AutoTokenizer.from_pretrained',
             return_value=lambda *a, **kw: {'input_ids': [1]}), patch.object(
        driver, 'train_qlora_photoz', return_value=result
    ) as trainer:
        driver.run_overfit(args, None, product, reference, np.asarray(product['split_labels']))
    kwargs = trainer.call_args.kwargs
    assert kwargs['train_dataset'] is kwargs['val_dataset'] is kwargs['test_dataset']
    assert len(kwargs['train_dataset']) == 512
    assert set(kwargs['train_dataset'].object_ids) <= set(product['object_id'][:600])
    saved = torch.load(tmp_path / 'result.pt', weights_only=False)
    assert set(saved['final_metrics']) == {'train'}
    assert 'test_evaluation' not in saved
    assert 'train_eval_cross_entropy' in saved['history'][0]


def test_mlp_uses_training_scaling_and_never_evaluates_test(tmp_path):
    torch.set_num_threads(1)
    product = fixture_product(8)
    reference = fixture_reference(product)
    args = SimpleNamespace(prepare_only=False, epochs=2, mlp_batch_size=4,
                           mlp_learning_rate=.001, device='cpu', output_dir=tmp_path,
                           reference_dir=tmp_path)
    original_evaluate = driver.evaluate_model_on_dataset
    evaluated_lengths = []

    def evaluate(*args, **kwargs):
        evaluated_lengths.append(len(args[1]))
        assert args[1].object_ids == [product['object_id'][8]]
        return original_evaluate(*args, **kwargs)

    with patch.object(driver, 'evaluate_model_on_dataset', side_effect=evaluate):
        summary = driver.run_mlp(args, product, (0., 2.), reference,
                                 np.asarray(product['split_labels']))
    saved = torch.load(tmp_path / 'result.pt', weights_only=False)
    assert torch.equal(saved['feature_mean'], product['extra_features'][:8].mean(0))
    assert evaluated_lengths == [1, 1]
    assert not summary['test_evaluated']
    assert set(saved['final_metrics']) == {'val'}
    assert saved['val_evaluation']['object_id'] == [product['object_id'][8]]


def test_galaxiesml_loader_uses_native_spectroscopic_labels_and_grouped_splits(tmp_path):
    source = tmp_path / 'source.hdf5'
    source.touch()
    metadata = {
        'catalogue_kind': 'galaxiesml-spec', 'target': 'DESI_fibermap/DESI_redshift',
        'cache_key': 'test-cohort', 'feature_names': ['g_flux', 'g_flux_error'],
        'split_counts': {'train': 2, 'val': 1, 'test': 1}, 'z_bounds': [0., 4.5],
        'seed': 42, 'sources': driver.surveys.fingerprints([source]),
    }
    (tmp_path / 'cohort.json').write_text(json.dumps(metadata))
    prepared = dict(metadata, cohort_cache=str(tmp_path))
    prepared_path = tmp_path / 'prepared.json'
    prepared_path.write_text(json.dumps(prepared))
    np.savez(tmp_path / 'cohort.npz',
             targetid=np.array([39627762704586359 + i for i in range(4)]),
             group=np.array(['hsc:1', 'hsc:2', 'hsc:3', 'hsc:4']),
             features=np.array([[1., .1], [np.nan, np.nan], [2., .2], [3., .3]], dtype=np.float32),
             z=np.array([.1, .2, .3, .4], dtype=np.float32),
             ZPHOT=np.full(4, 99.), split=np.array(['train', 'train', 'val', 'test']))
    with patch.object(driver, 'load_matched_product', side_effect=AssertionError('CLAUDS must not load')):
        pipeline, product, preparation, bounds, reference, labels = driver.load_galaxiesml_product(prepared_path)
    assert pipeline.catalogue_kind == 'galaxiesml-spec'
    assert preparation['target'] == 'DESI_fibermap/DESI_redshift'
    torch.testing.assert_close(product['z_spec'], torch.tensor([.1, .2, .3, .4]))
    assert product['object_id'][0] == '39627762704586359'
    assert labels.tolist() == ['train', 'train', 'val', 'test']
    assert bounds == (0., 4.5)
    assert reference['metadata']['config']['lora_target_modules'] == driver.glm.GLM52_LORA_TARGET_SETTING
    assert reference['final_metrics'] == {}
    prepared['target'] = 'ZPHOT'
    prepared_path.write_text(json.dumps(prepared))
    with pytest.raises(ValueError, match='spectroscopic'):
        driver.load_galaxiesml_product(prepared_path)


def test_galaxiesml_mlp_imputation_uses_only_training_and_has_no_clauds_comparison(tmp_path):
    torch.set_num_threads(1)
    product = fixture_product(8)
    product['metadata']['catalogue_kind'] = 'galaxiesml-spec'
    product['extra_features'][0, 0] = float('nan')
    product['extra_features'][-1, 0] = 1e9
    reference = fixture_reference(product)
    reference['final_metrics'] = {}
    args = SimpleNamespace(prepare_only=False, epochs=1, mlp_batch_size=4,
                           mlp_learning_rate=.001, device='cpu', output_dir=tmp_path,
                           catalogue_kind='galaxiesml-spec', survey_prepared_json=tmp_path / 'prepared.json')
    summary = driver.run_mlp(args, product, (0., 2.), reference,
                             np.asarray(product['split_labels']))
    saved = torch.load(tmp_path / 'result.pt', weights_only=False)
    assert saved['imputation_values'][0] == product['extra_features'][1:8, 0].median()
    assert len(saved['feature_names']) == 4
    assert torch.isfinite(saved['feature_mean']).all()
    assert saved['feature_mean'][0] < 20
    assert 'glm_qlora_val_metrics' not in summary
    assert saved['catalogue_kind'] == 'galaxiesml-spec'
