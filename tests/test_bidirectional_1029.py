"""Execution-overlay equivalence, interruption recovery and locked-P isolation."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'notebooks'),str(ROOT/'tests')]
from test_matched_training_1028 import Tiny,Tokenizer,cpu_export
from aion_magnitude.matched_training import train_loop as approved,train_order,state_fingerprint
from aion_magnitude.bidirectional_1029_training import train_loop as isolated
from aion_magnitude.qwen_posttraining import QwenPosttrainingConfig

def test_source_only_overlay_matches_approved_weights_and_resumes_exactly(tmp_path):
    torch.manual_seed(999);canonical=Tiny()
    init=tmp_path/'initial';(init/'qlora-seed42').mkdir(parents=True)
    torch.save({k:v.detach().clone() for k,v in canonical.named_parameters()},init/'qlora-seed42/canonical_initial_trainable_state.pt')
    config=dict(seed=42,method='qlora',initial_head=str(init/'head_seed42.pt'),checkpoint_interval_updates=1,
        training=asdict(QwenPosttrainingConfig(model_path='synthetic-only',device='cpu',epochs=3,head_warmup_epochs=1,gradient_accumulation_steps=3)))
    texts=[str(i) for i in range(7)];z=np.linspace(.1,1.,7,dtype=np.float32);ids=np.arange(7)
    def run(fn,name,resume=False,hook=None):
        torch.manual_seed(999);model=Tiny()
        extra=dict(saved_orders=train_order(7,42,3)) if fn is isolated else {}
        return fn(model,Tokenizer(),torch.device('cpu'),texts,z,ids,texts[:3],z[:3],ids[:3],config,tmp_path/name,
            resume=resume,export=cpu_export,checkpoint_hook=hook,**extra)
    a=run(approved,'approved');b=run(isolated,'isolated')
    def stop(s):
        if s['global_update']==4:raise InterruptedError('Synthetic interruption after entering joint phase')
    with pytest.raises(InterruptedError):run(isolated,'resumed',hook=stop)
    c=run(isolated,'resumed',resume=True)
    for label in ('selected','final'):
        states=[torch.load(tmp_path/n/label/'trainable_state.pt',weights_only=True) for n in ('approved','isolated','resumed')]
        assert len(set(state_fingerprint(s) for s in states))==1
    assert a['global_update']==b['global_update']==c['global_update']==9
    assert a['examples_seen']==b['examples_seen']==c['examples_seen']==21
    assert [h['val'] for h in a['history']]==[h['val'] for h in b['history']]==[h['val'] for h in c['history']]
    assert b['history']==c['history']
    assert (tmp_path/'isolated/selected/training_predictions.pt').exists()
    assert (tmp_path/'isolated/final/optimizer_scheduler_rng.pt').exists()
    # A checkpoint saved at the end of the last epoch must also finish exports after resume.
    run(isolated,'resumed',resume=True)

def test_worker_guard_denies_full_cohort_but_allows_source_only_assets(tmp_path):
    package=tmp_path/'package';execution=tmp_path/'execution'
    (package/'paired_split').mkdir(parents=True);(execution/'source_only').mkdir(parents=True)
    forbidden=package/'paired_split/paired_cohort.npz';allowed=execution/'source_only/hsc_train.npz'
    np.savez(forbidden,z=np.arange(5));np.savez(allowed,z=np.arange(3))
    script='''
import sys,numpy as np
from aion_magnitude.bidirectional_1029_training import install_P_guard
install_P_guard(sys.argv[1],sys.argv[2])
try:np.load(sys.argv[3])
except PermissionError:pass
else:raise AssertionError('P opened')
assert len(np.load(sys.argv[4])['z'])==3
'''
    subprocess.run([sys.executable,'-c',script,str(package),str(execution),str(forbidden),str(allowed)],cwd=ROOT,check=True)


def test_historical_analysis_runs_all_cells_and_paired_contrasts(tmp_path,monkeypatch):
    import bidirectional_1029_report as report
    from aion_magnitude.utils import make_redshift_grid
    from aion_magnitude.metrics import predict_photoz_from_logits
    out=tmp_path/'historical';out.mkdir();n=200
    z=np.tile(np.array([.1,.3,.5,.8,1.1],np.float32),40)
    np.savez(out/'cohort.npz',targetid=np.arange(n),z=z)
    edges,centers=make_redshift_grid(0,4.5,300)
    np.save(out/'bootstrap_indices.npy',np.random.default_rng(1029).integers(0,n,(1000,n),dtype=np.uint32))
    manifest=dict(protected_sha256={},redshift_edges=edges.tolist(),redshift_centers=centers.tolist())
    monkeypatch.setattr(report,'OUT',out);monkeypatch.setattr(report,'load_manifest',lambda:manifest)
    monkeypatch.setattr(report,'plots',lambda *args:None)
    def predictions(name,domain,d):
        folder=out/'cells'/name/domain;folder.mkdir(parents=True,exist_ok=True)
        shift=(report.NAMES.index(name)*2+(domain=='legacy'))*.002
        logits=-((centers[None,:]-torch.from_numpy(z[:,None])-shift)/.08)**2
        p=dict(targetid=d['targetid'],z_spec=z,redshift_edges=edges.numpy(),redshift_centers=centers.numpy(),logits=logits.numpy())
        p.update({k:v.numpy() for k,v in predict_photoz_from_logits(logits,centers).items()});return p
    monkeypatch.setattr(report,'cell_predictions',predictions)
    report.historical()
    import json
    r=json.loads((out/'results.json').read_text());assert len(r['cells'])==8
    c=r['cells'];expected=c['hsc-qlora/legacy']['metrics']['ce']-c['hsc-qlora/hsc']['metrics']['ce']
    assert r['contrasts']['qlora_H_to_L_penalty']['full']['ce']['value']==pytest.approx(expected)
    assert len(r['redshift_bins'])==5
    assert np.load(out/'bootstrap_metrics.npz')['full'].shape[0]==1000
    assert json.loads((out/'stage1_complete.json').read_text())['complete']
