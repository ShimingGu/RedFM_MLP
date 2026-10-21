import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


spec = importlib.util.spec_from_file_location(
    'glm52_numerical_sensitivity', Path(__file__).resolve().parents[1] / 'notebooks/glm52_numerical_sensitivity.py'
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_magnitude_perturbation_preserves_native_magnitude_error_and_other_bands():
    values = torch.tensor([10., 1., 20., 2., .01])
    changed = driver.perturb_magnitude(values, 0, .1)
    assert changed[0] == pytest.approx(10 * 10**(-.04))
    assert changed[1] / changed[0] == pytest.approx(.1)
    assert torch.equal(values[2:], changed[2:])
    assert values[0] == 10
    assert driver.perturb_magnitude(torch.tensor([float('nan'), 1.]), 0, .1) is None


def test_subset_contains_only_validation_and_is_reproducible():
    product = {'split_labels': ['train', 'val', 'test', 'val', 'val']}
    assert driver.select_validation(product, 2, 42).tolist() == driver.select_validation(product, 2, 42).tolist()
    assert set(driver.select_validation(product, 3, 42)) == {1, 3, 4}
    with pytest.raises(ValueError):
        driver.select_validation(product, 4, 42)


def test_routing_comparison_ignores_expert_order_and_aligns_tokens():
    before = {'tokens': [1, 2], 'routes': {'layer': {'ids': torch.tensor([[0, 1], [2, 3]]),
              'logits': torch.ones(2, 4)}}}
    after = {'tokens': [1, 9, 2], 'routes': {'layer': {'ids': torch.tensor([[1, 0], [0, 3], [3, 2]]),
             'logits': torch.ones(3, 4)}}}
    changes = driver.token_and_route_changes(before, after)
    assert changes['token_changed']
    assert changes['token_count_delta'] == 1
    assert changes['routing_last_jaccard_distance'] == 0
    assert changes['routing_aligned_jaccard_distance'] == 0
    after['routes']['layer']['ids'][-1] = torch.tensor([0, 2])
    assert driver.token_and_route_changes(before, after)['routing_last_jaccard_distance'] == pytest.approx(2/3)


def test_numeric_inference_uses_saved_scaling_and_missingness():
    saved = {'imputation_values': torch.tensor([5., 6.]),
             'feature_mean': torch.tensor([1., 2., 0., 0.]),
             'feature_scale': torch.tensor([2., 2., 1., 1.])}
    actual = driver.numeric_input(torch.tensor([float('nan'), 10.]), saved)
    torch.testing.assert_close(actual, torch.tensor([2., 4., 1., 0.]))


def test_glm_serialization_preserves_five_decimals_and_na():
    text = driver.glm_text(torch.tensor([1.234567, float('nan')]), ['flux', 'flux_error'])
    assert 'flux=1.23457' in text
    assert 'flux_error=NA' in text
    assert 'flux_unit=AB_nanomaggy' in text


def test_snapshot_records_real_glm_moe_routes_and_hidden_layers():
    from transformers import GlmMoeDsaConfig, GlmMoeDsaModel
    from aion_magnitude.qwen_posttraining import QwenPhotoZModel

    torch.set_num_threads(1)
    config = GlmMoeDsaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, moe_intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        first_k_dense_replace=1, num_local_experts=4, num_experts_per_tok=2,
        n_shared_experts=1, n_group=1, topk_group=1, q_lora_rank=8, kv_lora_rank=8,
        qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8,
        index_n_heads=2, index_head_dim=8, index_topk=8, indexer_types=['full', 'full'],
        max_position_embeddings=32,
    )
    model = QwenPhotoZModel(GlmMoeDsaModel(config), hidden_size=32, n_z_bins=4, head_hidden_dim=8).eval()
    model.requires_grad_(False)

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.ones(1, 3, dtype=torch.long)}

    settings = {'max_length': 32, 'z_min': 0., 'z_max': 4.5, 'n_z_bins': 4}
    before = driver.snapshot(model, Tokenizer(), 'text', settings, 'cpu', routes=True)
    after = driver.snapshot(model, Tokenizer(), 'text', settings, 'cpu', routes=True)
    assert before['hidden'].shape == (3, 32)
    assert len(before['routes']) == 1
    route = next(iter(before['routes'].values()))
    assert route['ids'].shape == (3, 2)
    changes = driver.compare_snapshots(before, after)
    assert changes['delta_z'] == 0
    assert changes['pdf_total_variation'] == 0
    assert changes['routing_last_jaccard_distance'] == 0


def test_statistics_do_not_assume_a_prediction_jump():
    records = [{'model': 'GLM', 'delta_mag': .001, 'delta_z': 0.,
                'true_z': .3, 'pdf_total_variation': .0002}]
    stats = driver.group_statistics(records, ['GLM'], [.001], .02)[0]
    assert stats['flagged_fraction'] == 0
    assert stats['median_pdf_tv'] == .0002


def test_plots_and_report_handle_no_prediction_jumps(tmp_path):
    records = []
    for delta in (-.001, .001):
        records.append({'model': 'GLM QLoRA', 'delta_mag': delta, 'delta_z': 0.,
                        'true_z': .3, 'pdf_total_variation': .0002,
                        'hidden_relative_l2': .001, 'hidden_layer_relative_l2': [.001, .002],
                        'routing_last_jaccard_distance': 0., 'unmatched_token_fraction': .1,
                        'token_changed': True, 'object_id': 'large-id', 'band': 'g'})
        records.append({'model': 'Numeric MLP', 'delta_mag': delta, 'delta_z': 0.,
                        'true_z': .3, 'pdf_total_variation': .0001, 'object_id': 'large-id', 'band': 'g'})
    statistics = driver.group_statistics(records, ['GLM QLoRA', 'Numeric MLP'], [.001], .02)
    summary = {'sample_size': 1, 'seed': 42, 'statistics': statistics, 'jump_threshold': .02,
               'repeat_noise': {'max_abs_delta_z': 0., 'max_pdf_tv': 0.}, 'qwen_status': 'Unavailable.'}
    driver.make_plots(tmp_path, records, statistics, [.001])
    driver.report(tmp_path, summary, records)
    assert len(list(tmp_path.glob('*.png'))) == 4
    text = (tmp_path / 'report.md').read_text()
    assert 'No response exceeded that threshold' in text
    assert 'not causal mediation' in text
    assert 'quantized' in text
