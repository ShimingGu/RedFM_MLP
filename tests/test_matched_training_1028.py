"""Critical invariants for future matched training; only tiny CPU toy optimization."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'notebooks'))
from aion_magnitude.matched_training import initial_head,state_fingerprint,train_order,accumulation_window_size,train_loop
from aion_magnitude.qwen_posttraining import QwenPosttrainingConfig
from aion_magnitude import survey_catalogues as surveys
from task_1028 import physical_overlap


def test_bookkeeping_does_not_change_historical_sampling():
    first=surveys._Sample(50,42);second=surveys._Sample(50,42);rng=np.random.default_rng(1)
    for i,n in enumerate([67,81,39,100]):
        data=dict(targetid=np.arange(i*1000,i*1000+n),features=rng.normal(size=(n,11)),z=rng.random(n))
        first.add({k:v.copy() for k,v in data.items()})
        second.add(dict(**{k:v.copy() for k,v in data.items()},ra=rng.uniform(0,360,n),source_row=np.arange(n)))
    a,b=first.compact(),second.compact()
    for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_common_heads_and_orders_are_identical_without_source_dependence():
    h=initial_head(8,42);l=initial_head(8,42)
    assert state_fingerprint(h)==state_fingerprint(l)
    assert state_fingerprint(h)!=state_fingerprint(initial_head(8,43))
    orders=train_order(16600,42,10)
    for order in orders:np.testing.assert_array_equal(np.sort(order),np.arange(16600))
    np.testing.assert_array_equal(orders,train_order(16600,42,10))


@pytest.mark.parametrize('key',['initial_adapter','source_adapter'])
def test_source_trained_adapter_cannot_initialize_a_matched_branch(key):
    from aion_magnitude.matched_training import build_fresh_model
    with pytest.raises(ValueError,match='pre-adaptation'):
        build_fresh_model({key:'/some/historical/adapter'})


def test_peft_export_cannot_be_used_as_a_pretrained_base(tmp_path):
    from aion_magnitude.matched_training import build_fresh_model
    (tmp_path/'adapter_config.json').write_text('{}')
    with pytest.raises(ValueError,match='pre-adaptation'):
        build_fresh_model(dict(training=dict(model_path=str(tmp_path))))


def test_partial_effective_batch_averages_its_actual_examples():
    assert accumulation_window_size(16600,16592,16)==8
    assert accumulation_window_size(16600,16599,16)==8
    assert accumulation_window_size(16600,16591,16)==16
    x=torch.tensor(1.,requires_grad=True)
    for i in range(7):
        (x/accumulation_window_size(7,i,3)).backward()
        if i in [2,5,6]:
            assert x.grad.item()==pytest.approx(1.)
            x.grad=None


def test_cross_representation_positions_and_ra_wrap_catch_leakage():
    data=dict(z=np.array([.1,.2]),targetid=np.array([1,2]),galaxy_id=np.array(['hsc:1','hsc:2']),
              legacy_group=np.array(['ls:1:2:3','ls:1:2:4']),category=np.array(['T','P']),
              ra=np.array([0.,.0004]),dec=np.zeros(2),legacy_ra=np.array([.0001,.0003]),legacy_dec=np.zeros(2))
    assert physical_overlap(data)['cross_category_pairs']==[(0,1)]
    data['ra']=np.array([359.99995,.00005]);data['legacy_ra']=data['ra'].copy()
    assert physical_overlap(data)['cross_category_pairs']==[(0,1)]


class Tokenizer:
    def __call__(self,texts,**kwargs):
        return dict(input_ids=torch.tensor([[int(t)] for t in texts]),attention_mask=torch.ones(len(texts),1,dtype=torch.long))


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.qwen=torch.nn.Linear(1,4);self.photoz_head=torch.nn.Sequential(torch.nn.Dropout(.1),torch.nn.Linear(4,300));self.freeze_qwen_representation=False
    def forward(self,input_ids,attention_mask):
        x=self.qwen(input_ids.float())
        if self.freeze_qwen_representation:x=x.detach()
        return self.photoz_head(x)


def cpu_export(model,folder,metadata):
    folder.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),folder/'toy_model.pt')


def test_exact_update_budget_and_interrupted_resume_match_final_state(tmp_path):
    config=dict(seed=42,checkpoint_interval_updates=1,training=asdict(QwenPosttrainingConfig(model_path='synthetic-only',device='cpu',epochs=3,head_warmup_epochs=1,gradient_accumulation_steps=3)))
    texts=[str(i) for i in range(7)];z=np.linspace(.1,1.,7,dtype=np.float32);ids=np.arange(7)
    def run(folder,resume=False,hook=None):
        torch.manual_seed(999);model=Tiny()
        return train_loop(model,Tokenizer(),torch.device('cpu'),texts,z,ids,texts[:3],z[:3],ids[:3],config,folder,
                          resume=resume,export=cpu_export,checkpoint_hook=hook)
    full=run(tmp_path/'full')
    def stop(value):
        if value['global_update']==4:raise InterruptedError('Synthetic interruption')
    with pytest.raises(InterruptedError):run(tmp_path/'resumed',hook=stop)
    resumed=run(tmp_path/'resumed',resume=True)
    assert full['global_update']==resumed['global_update']==9
    assert full['examples_seen']==resumed['examples_seen']==21
    assert not full['test_evaluated'] and not resumed['test_evaluated']
    a=torch.load(tmp_path/'full/final/trainable_state.pt',weights_only=True);b=torch.load(tmp_path/'resumed/final/trainable_state.pt',weights_only=True)
    assert state_fingerprint(a)==state_fingerprint(b)
    assert full['history']==resumed['history']
    assert (tmp_path/'full/selected/validation_predictions.pt').exists()
