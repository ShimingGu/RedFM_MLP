import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset


spec = importlib.util.spec_from_file_location(
    'glm52_pretraining_transfer', Path(__file__).resolve().parents[1] / 'notebooks/glm52_pretraining_transfer.py'
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_subsets_are_nested_fixed_and_never_include_test():
    product = {'split_labels': ['train']*8+['val']*4+['test']*4}
    train, val = driver.fixed_subsets(product, [2, 6], 3, 42)
    other, other_val = driver.fixed_subsets(product, [2, 6], 3, 42)
    assert set(train[2]) <= set(train[6]) <= set(range(8))
    assert set(val) <= set(range(8,12))
    assert np.array_equal(train[6], other[6]) and np.array_equal(val, other_val)


def test_random_backbone_has_identical_configuration_and_fresh_weights(tmp_path):
    from transformers import GlmMoeDsaConfig, GlmMoeDsaModel
    torch.set_num_threads(1)
    config = GlmMoeDsaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        moe_intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, first_k_dense_replace=1, num_local_experts=4,
        num_experts_per_tok=2, n_shared_experts=1, n_group=1, topk_group=1,
        q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=8,
        v_head_dim=8, index_n_heads=2, index_head_dim=8, index_topk=8,
        indexer_types=['full','full'], max_position_embeddings=32)
    source = tmp_path/'toy'; destination=tmp_path/'random'
    torch.manual_seed(1)
    original=GlmMoeDsaModel(config)
    original.save_pretrained(source)
    tokenizer=SimpleNamespace(save_pretrained=lambda path: None)
    with patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer):
        count=driver.save_random_backbone(source,destination,42)
    loaded=GlmMoeDsaModel.from_pretrained(destination)
    assert count==sum(p.numel() for p in original.parameters())
    assert list(original.state_dict())==list(loaded.state_dict())
    assert original.config.hidden_size==loaded.config.hidden_size
    assert original.config.num_local_experts==loaded.config.num_local_experts
    assert not torch.equal(original.embed_tokens.weight,loaded.embed_tokens.weight.float())
    assert torch.equal(loaded.embed_tokens.weight[config.pad_token_id],torch.zeros(config.hidden_size)) if config.pad_token_id is not None else True


def result(initialization, seed, ce, nmad=.1, learned=True):
    return {'mode':'qlora','train_objects':512,'seed':seed,'initialization':initialization,
            'val_metrics':{'cross_entropy':ce,'nmad':nmad},
            'history':[{'train':{'cross_entropy':6.}}, {'train':{'cross_entropy':4. if learned else 6.}}]}


def test_promising_screen_repeats_and_requires_replication_for_transfer_label():
    rows=[result('toy',42,4.),result('random',42,5.)]
    assert driver.promising(rows,.05)
    assert 'screen only' in driver.paired_summary(rows,.05)[0]['interpretation']
    for seed in (43,44):
        rows += [result('toy',seed,4.),result('random',seed,5.)]
    assert driver.paired_summary(rows,.05)[0]['interpretation'].startswith('positive transfer')
    swapped=[dict(row,initialization='random' if row['initialization']=='toy' else 'toy') for row in rows]
    assert driver.paired_summary(swapped,.05)[0]['interpretation'].startswith('negative transfer')
    for row in rows:
        row['history'][1]['train']['cross_entropy']=6.
    assert 'inconclusive' in driver.paired_summary(rows,.05)[0]['interpretation']


def test_broad_policy_trains_embeddings_and_all_blocks_but_keeps_experts_fixed():
    for name in ('embed_tokens.weight','layers.0.self_attn.q_a_proj.weight',
                 'layers.1.mlp.down_proj.weight','layers.3.mlp.shared_experts.up_proj.weight',
                 'layers.3.mlp.gate.weight','norm.weight'):
        assert driver.broad_trainable(name)
    assert not driver.broad_trainable('layers.3.mlp.experts.gate_up_proj')


def test_head_training_has_complete_curves_and_matched_update_budget(tmp_path):
    from aion_magnitude.models import PhotoZHead
    torch.set_num_threads(1)
    model=torch.nn.Module()
    model.qwen=torch.nn.Identity()
    model.photoz_head=PhotoZHead(4,n_z_bins=300,hidden_dim=8)
    dataset=TensorDataset(torch.randn(7,4),torch.linspace(.1,1.,7))
    args=SimpleNamespace(device='cpu',microbatch_size=2,effective_batch_size=4,
                         head_learning_rate=.001,adapter_learning_rate=.0001,broad_learning_rate=.001)
    with patch.object(driver,'frozen_dataset',side_effect=lambda model,tokenizer,data,*rest: data):
        saved=driver.train_arm(model,None,dataset,dataset,'head',42,2,(0.,4.5),512,args,tmp_path)
    assert saved['updates']==4
    assert len(saved['history'])==3
    assert saved['history'][-1]['examples_seen']==14
    for row in saved['history']:
        assert {'cross_entropy','nmad','catastrophic_outlier_fraction'} <= row['val'].keys()
        assert {'cross_entropy','nmad','catastrophic_outlier_fraction'} <= row['train'].keys()
    assert not saved['test_evaluated']


def test_prepare_uses_native_spectroscopic_targets_and_no_test_objects():
    features=torch.arange(11*2,dtype=torch.float32).reshape(11,2)
    product={'split_labels':['train']*6+['val']*3+['test']*2,
             'object_id':[str(i) for i in range(11)], 'extra_features':features,
             'feature_names':['g_flux','g_flux_error'], 'z_spec':torch.linspace(.1,1.,11)}
    preparation={'target':'DESI_fibermap/DESI_redshift','cache_key':'matched',
                 'split_counts':{'train':6,'val':3,'test':2}}
    reference={'metadata':{'config':{'model_path':'local-glm','max_length':512}}}
    args=SimpleNamespace(survey_prepared_json=Path('prepared.json'),train_sizes=[2,4],
                         broad_train_objects=2,val_objects=2,data_seed=42)
    with patch.object(driver.capacity,'load_galaxiesml_product',return_value=(None,product,preparation,(0.,4.5),reference,None)):
        datasets,val,manifest=driver.prepare_data(args)
    assert manifest['target']=='DESI_fibermap/DESI_redshift'
    assert manifest['catalogue_kind']=='galaxiesml-spec'
    assert set(datasets[2].object_ids)<=set(datasets[4].object_ids)<=set(product['object_id'][:6])
    assert set(val.object_ids)<=set(product['object_id'][6:9])
    for dataset in list(datasets.values())+[val]:
        for index,object_id in enumerate(dataset.object_ids):
            assert dataset.redshifts[index]==product['z_spec'][int(object_id)]
        assert all('CLAUDS' not in text for text in dataset.texts)


def test_report_and_all_requested_learning_curve_plots(tmp_path):
    args=driver.parser().parse_args(['--output-dir',str(tmp_path),'--train-sizes','128','512'])
    metrics={'cross_entropy':4.,'nmad':.1,'catastrophic_outlier_fraction':.2}
    rows=[]
    for initialization in ('toy','random'):
        history=[{'epoch':0,'examples_seen':0,'train':dict(metrics,cross_entropy=6.),'val':dict(metrics)},
                 {'epoch':1,'examples_seen':512,'train':dict(metrics),'val':dict(metrics)}]
        rows.append({'mode':'qlora','train_objects':512,'seed':42,'initialization':initialization,
                     'selected_epoch':1,'train_metrics':dict(metrics),'val_metrics':dict(metrics),
                     'seconds':1.,'history':history})
    driver.make_report(args,rows,{},'skipped in test',[])
    assert len(list(tmp_path.glob('*.png')))==3
    text=(tmp_path/'report.md').read_text()
    assert 'spectroscopic redshift' in text
    assert 'screen only; no replicated transfer conclusion' in text
    assert 'mechanisms' in text.lower()
    assert 'not a significance test' in text
