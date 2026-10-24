"""Selection, full replay validation, accumulation and resume for continued SFT."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from aion_magnitude.metrics import predict_photoz_from_logits
from aion_magnitude.qwen_continued_sft import (
    ContinuedSFTProtocol, continue_sft, group_sample_count, is_better,
    parameter_changes, verify_replay,
)
from aion_magnitude.qwen_posttraining import QwenPhotoZModel, TextRedshiftDataset, _trainable_state
from aion_magnitude.utils import make_redshift_grid


def evaluation(good=True):
    edges, centers = make_redshift_grid(0, 1, 5)
    logits = torch.tensor([[9.,0.,0.,0.,0.] if good else [0.,0.,0.,0.,9.]])
    result = dict(logits=logits, z_spec=torch.tensor([.1]), object_id=["validation"],
                  redshift_edges=edges, redshift_centers=centers)
    result.update(predict_photoz_from_logits(logits, centers=centers))
    return result


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Embedding(8, 8)
    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.adapter(input_ids))
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True)
        torch.save(self.state_dict(), Path(path) / "toy_adapter.pt")


def tokenizer(texts, **kwargs):
    return dict(input_ids=torch.tensor([[1,2]]*len(texts)),
                attention_mask=torch.ones((len(texts),2),dtype=torch.long))


def model_and_data():
    torch.manual_seed(1023)
    model = QwenPhotoZModel(Backbone(), hidden_size=8, n_z_bins=5, head_hidden_dim=8)
    train = TextRedshiftDataset(["one","two","three"], [.2,.4,.6], ["train1","train2","train3"])
    val = TextRedshiftDataset(["val"], [.1], ["validation"])
    return model, train, val


def test_matched_protocol_is_exact_and_disallows_head_warmup():
    p = ContinuedSFTProtocol()
    assert (p.max_epochs,p.head_warmup_epochs,p.adapter_lr,p.head_lr,p.seed,p.eval_every_updates) == (2,0,3e-6,1e-5,1023,250)
    assert p.microbatch_size*p.gradient_accumulation_steps == 16
    with pytest.raises(ValueError):ContinuedSFTProtocol(head_warmup_epochs=1)
    with pytest.raises(ValueError):ContinuedSFTProtocol(max_epochs=3)


def test_partial_effective_batch_has_correct_loss_denominator():
    p=ContinuedSFTProtocol()
    assert group_sample_count(0,22444,p)==16
    assert group_sample_count(22431,22444,p)==16
    assert group_sample_count(22432,22444,p)==12
    assert group_sample_count(22443,22444,p)==12


def test_zero_checkpoint_wins_ties_and_failed_candidates():
    assert not is_better(2.453,2.453)
    assert not is_better(float('nan'),2.453)
    assert not is_better(2.454,2.453)
    assert is_better(2.452,2.453)


def test_replay_verifies_all_objects_labels_grid_and_predictions():
    initial=evaluation()
    assert verify_replay(initial,initial)['passed']
    changed=dict(initial,object_id=["different"])
    with pytest.raises(ValueError,match="membership/order"):verify_replay(changed,initial)
    with pytest.raises(AssertionError):verify_replay(evaluation(False),initial)
    with pytest.raises(AssertionError):verify_replay(dict(initial,z_spec=torch.tensor([.2])),initial)


def test_parameter_change_reports_adapter_and_head_separately():
    initial={"qwen.lora":torch.tensor([1.,0.]),"photoz_head.weight":torch.tensor([2.])}
    selected={"qwen.lora":torch.tensor([1.,.1]),"photoz_head.weight":torch.tensor([2.])}
    change=parameter_changes(initial,selected)
    assert change['adapter']['changed_values']==1
    assert change['adapter']['delta_l2']==pytest.approx(.1)
    assert change['pdf_head']['changed_values']==0


def test_real_loop_keeps_zero_if_continued_sft_worsens_validation(tmp_path):
    model,train,val=model_and_data()
    initial=_trainable_state(model)
    p=ContinuedSFTProtocol(max_epochs=1,gradient_accumulation_steps=2,eval_every_updates=1,checkpoint_every_updates=1)
    result=continue_sft(model=model,tokenizer=tokenizer,train_dataset=train,val_dataset=val,
             initial_evaluation=evaluation(),output_dir=tmp_path,protocol=p,max_length=16,device='cpu',
             evaluator=lambda *a,**kw:evaluation(False))
    assert result['global_update']==2 and result['best_update']==0
    assert [r['samples'] for r in result['training_history']]==[2,1]
    assert all(r['adapter_lr']==3e-6 and r['head_lr']==1e-5 for r in result['training_history'])
    for key,value in _trainable_state(model).items():torch.testing.assert_close(value,initial[key],rtol=0,atol=0)
    assert result['parameter_changes']['selected_vs_initial']['adapter']['changed_values']==0
    assert result['parameter_changes']['last_vs_initial']['adapter']['changed_values']>0
    assert result['parameter_changes']['last_vs_initial']['pdf_head']['changed_values']>0
    assert not result['test_evaluated']


def test_resume_matches_uninterrupted_dropout_and_batch_order(tmp_path):
    uninterrupted=tmp_path/'uninterrupted';interrupted=tmp_path/'interrupted'
    uninterrupted.mkdir();interrupted.mkdir()
    protocol=ContinuedSFTProtocol(gradient_accumulation_steps=2,eval_every_updates=1,checkpoint_every_updates=1)
    def execute(folder,model,train,val,evaluator,resume=False):
        return continue_sft(model=model,tokenizer=tokenizer,train_dataset=train,val_dataset=val,
                  initial_evaluation=evaluation(),output_dir=folder,protocol=protocol,max_length=16,
                  device='cpu',evaluator=evaluator,resume=resume)
    model,train,val=model_and_data()
    execute(uninterrupted,model,train,val,lambda *a,**kw:evaluation(False))
    reference=torch.load(uninterrupted/'latest.pt',weights_only=False)['trainable_state']
    model,train,val=model_and_data()
    calls=0
    def fail_second(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('simulated interruption')
        return evaluation(False)
    with pytest.raises(RuntimeError,match='interruption'):execute(interrupted,model,train,val,fail_second)
    model,train,val=model_and_data()
    result=execute(interrupted,model,train,val,lambda *a,**kw:evaluation(False),resume=True)
    resumed=torch.load(interrupted/'latest.pt',weights_only=False)['trainable_state']
    for key,value in reference.items():torch.testing.assert_close(value,resumed[key],rtol=0,atol=0)
    assert result['global_update']==4
