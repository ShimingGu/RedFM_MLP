"""Numerical, evaluator, exact-resume, gating and isolation tests; synthetic only."""
from dataclasses import asdict
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import ndtr,logsumexp

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'notebooks')]
from aion_magnitude.highz_h1_head import (TruncatedGaussianMixtureHead,mixture_log_bin_masses,parameters_from_raw,replace_head,MIN_SCALE,MAX_SCALE)
from aion_magnitude.models import PhotoZHead
from aion_magnitude.utils import make_redshift_grid
from aion_magnitude.metrics import (redshift_cross_entropy_loss,predict_photoz_from_logits,discrete_crps,summarize_pdf_metrics)
from aion_magnitude.matched_training import train_loop as canonical,train_order,state_fingerprint
from aion_magnitude.highz_h1_training import train_loop
from aion_magnitude.qwen_posttraining import QwenPosttrainingConfig


def test_probability_normalization_extreme_narrow_broad_and_finite_gradients():
    edges,_=make_redshift_grid(0,4.5,300)
    raw=torch.tensor([[0.,0.,0.,0.,0.,0.,-1000.,-1000.,-1000.],
        [1000.,-1000.,0.,-1000.,1000.,0.,1000.,1000.,1000.],
        [-30.,30.,-20.,-50.,50.,3.,-20.,20.,0.]],dtype=torch.float64,requires_grad=True)
    logp=mixture_log_bin_masses(raw,edges);p=logp.exp()
    assert torch.isfinite(logp).all() and torch.isfinite(p).all() and (p>=0).all()
    torch.testing.assert_close(p.sum(-1),torch.ones(3,dtype=p.dtype),rtol=1e-12,atol=1e-12)
    (-logp[:,[2,40,220]].mean()).backward();assert torch.isfinite(raw.grad).all()
    _,_,sigma=parameters_from_raw(raw);assert (sigma>=MIN_SCALE).all() and (sigma<=MAX_SCALE).all()


def test_integrals_agree_with_independent_scipy_quadrature_and_global_truncation():
    edges,_=make_redshift_grid(0,4.5,300)
    raw=torch.tensor([[.3,-.5,.7,-3.,.2,3.,-2.,-.5,1.]],dtype=torch.float64)
    lw,mu,sigma=parameters_from_raw(raw);w=lw.exp().numpy()[0];mu=mu.numpy()[0];sigma=sigma.numpy()[0]
    def density(z):return np.sum(w*np.exp(-.5*((z-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi)))
    norm=quad(density,0,4.5,epsabs=1e-12,epsrel=1e-12)[0]
    p=mixture_log_bin_masses(raw,edges).exp().detach().numpy()[0]
    reference=np.array([quad(density,float(a),float(b),epsabs=1e-14,epsrel=1e-12)[0]/norm for a,b in zip(edges[:-1],edges[1:])])
    np.testing.assert_allclose(p,reference,rtol=2e-10,atol=3e-15)
    individual=ndtr((edges[1:].numpy()[None,:]-mu[:,None])/sigma[:,None])-ndtr((edges[:-1].numpy()[None,:]-mu[:,None])/sigma[:,None])
    individually_truncated=(w[:,None]*individual/individual.sum(1,keepdims=True)).sum(0)
    assert np.max(np.abs(p-individually_truncated))>1e-4


def test_autodiff_matches_finite_differences_for_all_nine_outputs():
    edges,_=make_redshift_grid(0,4.5,300)
    raw=torch.tensor([[.2,-.3,.1,-1.,.2,1.3,-2.,-1.,.3]],dtype=torch.float64,requires_grad=True)
    def fn(x):return -mixture_log_bin_masses(x,edges)[:,[3,43,99,165,245]].mean()
    fn(raw).backward();autodiff=raw.grad.numpy().copy();finite=np.empty((1,9));eps=1e-5
    for j in range(9):
        a=raw.detach().clone();b=a.clone();a[0,j]+=eps;b[0,j]-=eps;finite[0,j]=float((fn(a)-fn(b))/(2*eps))
    np.testing.assert_allclose(autodiff,finite,rtol=2e-5,atol=2e-7)
    assert np.count_nonzero(np.abs(autodiff)>1e-8)==9


def test_native_grid_edge_targets_ce_crps_quantiles_and_save_reload(tmp_path):
    torch.manual_seed(42);head=TruncatedGaussianMixtureHead(4,8).eval();x=torch.randn(4,4)
    logits=head(x);edges,centers=make_redshift_grid(0,4.5,300)
    torch.testing.assert_close(head.redshift_edges,edges,rtol=0,atol=0)
    labels=torch.tensor([0.,float(edges[40]),float(edges[41])+1e-5,4.5]);bins=(torch.bucketize(labels,edges)-1).clamp(0,299)
    assert bins.tolist()==[0,39,41,299]
    loss=redshift_cross_entropy_loss(logits,labels,edges)
    expected=-torch.log_softmax(logits,dim=-1)[torch.arange(4),bins].mean();torch.testing.assert_close(loss,expected)
    p=predict_photoz_from_logits(logits,centers);cdf=p['pz'].cumsum(1)
    manual=((cdf-(centers[None,:]>=labels[:,None]).float())**2).sum(1)*(centers[1]-centers[0])
    torch.testing.assert_close(discrete_crps(p['pz'],labels,centers),manual)
    evaluation=dict(logits=logits,z_spec=labels,redshift_edges=edges,redshift_centers=centers,**p)
    assert all(np.isfinite(v) for v in summarize_pdf_metrics(evaluation).values())
    for key,q in [('z_p16',.16),('z_p50',.5),('z_p84',.84)]:
        idx=(cdf>=q).long().argmax(1);torch.testing.assert_close(p[key],centers[idx])
    path=tmp_path/'head.pt';torch.save(head.state_dict(),path);restored=TruncatedGaussianMixtureHead(4,8).eval();restored.load_state_dict(torch.load(path,weights_only=True))
    torch.testing.assert_close(restored(x),logits,rtol=0,atol=0)
    # The archived softmax head consumes exactly the same native edges/loss/evaluator.
    soft=PhotoZHead(4,300,8).eval();v=soft(x);assert v.shape==logits.shape
    assert torch.isfinite(redshift_cross_entropy_loss(v,labels,edges))


class Tokenizer:
    def __call__(self,texts,**kw):return dict(input_ids=torch.tensor([[int(t)] for t in texts]),attention_mask=torch.ones(len(texts),1,dtype=torch.long))
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.qwen=torch.nn.Linear(1,4);self.photoz_head=PhotoZHead(4,300,8);self.freeze_qwen_representation=False
    def forward(self,input_ids,attention_mask):
        x=self.qwen(input_ids.float())
        if self.freeze_qwen_representation:x=x.detach()
        return self.photoz_head(x)
def cpu_export(model,path,metadata):
    path.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),path/'toy_model.pt')


@pytest.mark.parametrize('head',['softmax','mixture'])
def test_output_updates_and_exact_interrupted_resume_preserve_budget(tmp_path,head):
    def model():
        torch.manual_seed(999);m=Tiny()
        if head=='mixture':replace_head(m)
        return m
    first=model();initial={k:v.detach().clone() for k,v in first.named_parameters()};torch.save(initial,tmp_path/'initial.pt')
    cfg=dict(seed=42,method='qlora',head=head,initial_trainable_state=str(tmp_path/'initial.pt'),checkpoint_interval_updates=1,
        training=asdict(QwenPosttrainingConfig(model_path='synthetic',device='cpu',epochs=3,head_warmup_epochs=1,gradient_accumulation_steps=3,z_max=4.5)))
    texts=list(map(str,range(7)));z=np.linspace(.1,1.,7,dtype=np.float32);ids=np.arange(7)
    def run(folder,resume=False,hook=None,fn=train_loop):
        extra={'saved_orders':train_order(7,42,3)} if fn is train_loop else {}
        return fn(model(),Tokenizer(),torch.device('cpu'),texts,z,ids,texts[:3],z[:3],ids[:3],cfg,tmp_path/folder,resume=resume,export=cpu_export,checkpoint_hook=hook,**extra)
    a=run('full')
    def stop(s):
        if s['global_update']==4:raise InterruptedError('Synthetic interruption')
    with pytest.raises(InterruptedError):run('resumed',hook=stop)
    # Incomplete/corrupted newer slot must be ignored in favor of the verified own slot.
    slots=list((tmp_path/'resumed/checkpoints').glob('resume-slot*.pt'))
    newest=max(slots,key=lambda p:torch.load(p,map_location='cpu',weights_only=False)['global_update'])
    newest.write_bytes(b'simulated interrupted checkpoint bytes')
    b=run('resumed',resume=True)
    for label in ['selected','final']:
        aa=torch.load(tmp_path/'full'/label/'trainable_state.pt',weights_only=True);bb=torch.load(tmp_path/'resumed'/label/'trainable_state.pt',weights_only=True)
        assert state_fingerprint(aa)==state_fingerprint(bb)
    assert a['global_update']==b['global_update']==9 and a['examples_seen']==b['examples_seen']==21
    assert [h['val'] for h in a['history']]==[h['val'] for h in b['history']]
    assert all(h['gradient_diagnostics']['head_output_grad_max']>0 for h in a['history'])
    assert all(h['epoch_wall_seconds_this_process']>0 and 'peak_reserved_gib' in h['gpu_memory'] for h in a['history'])
    assert any(not torch.equal(initial[k],aa[k]) for k in initial if k.startswith('photoz_head.net.6.'))
    if head=='softmax':
        c=run('canonical',fn=canonical)
        cc=torch.load(tmp_path/'canonical/final/trainable_state.pt',weights_only=True)
        assert state_fingerprint(cc)==state_fingerprint(torch.load(tmp_path/'full/final/trainable_state.pt',weights_only=True))
        assert [h['val'] for h in a['history']]==[h['val'] for h in c['history']]


def test_shared_hidden_trunk_and_different_parameter_capacity():
    torch.manual_seed(42);model=Tiny();old={k:v.clone() for k,v in model.photoz_head.state_dict().items()};n0=sum(p.numel() for p in model.photoz_head.parameters());replace_head(model)
    n1=sum(p.numel() for p in model.photoz_head.parameters());assert n0-n1==(300-9)*(8+1)
    for k,v in old.items():
        if not k.startswith('net.6.'):torch.testing.assert_close(model.photoz_head.state_dict()[k],v,rtol=0,atol=0)


def test_six_final_gate_blocks_missing_arm_and_never_uses_best_as_primary(tmp_path,monkeypatch):
    import highz_h1_1102 as h
    monkeypatch.setattr(h,'OUT',tmp_path);monkeypatch.setattr(h,'validate',lambda:{})
    (tmp_path/'protocol').mkdir();(tmp_path/'protocol/evaluation_policy.json').write_text('{}');(tmp_path/'evaluation').mkdir()
    for a in h.ARMS:
        r=tmp_path/'training'/a;(r/'final').mkdir(parents=True);(r/'selected').mkdir()
        for label in ['final','selected']:(r/label/'trainable_state.pt').write_bytes(b'synthetic placeholder')
        for name,value in {
            'complete.json':dict(global_update=6560,examples_seen=104890,best_update=1312,history=[dict(gradient_diagnostics=dict(head_output_grad_max=1))]*10),
            'completion_checks.json':dict(passed=True),
            'source_validation_replay.json':dict(selected=dict(exact=True),final=dict(exact=True)),
            'final/metadata.json':dict(global_update=6560,checkpoint_sha256={}),
            'selected/metadata.json':dict(global_update=1312,checkpoint_sha256={})}.items():(r/name).write_text(json.dumps(value))
    last=tmp_path/'training/H1-F/complete.json';value=last.read_text();last.unlink()
    with pytest.raises(FileNotFoundError):h.freeze_final()
    assert not (tmp_path/'evaluation/final_checkpoints_frozen.json').exists()
    last.write_text(value);gate=h.freeze_final();assert len(gate['models'])==6
    assert all(m['primary_update']==6560 and m['secondary_update']==1312 for m in gate['models'].values())


def test_worker_guard_denies_P_foreign_artifacts_and_all_foreign_writes(tmp_path):
    import highz_h1_1102 as h
    # Audit hooks cannot be uninstalled; test only in a disposable subprocess.
    code="""
import pathlib
import highz_h1_1102 as h
h.install_worker_guard()
paths=[h.OUT/'evaluation/locked_P.npz',h.PKG/'paired_split/paired_cohort.npz',h.ARC/'1102/morphology-three-feature-pilot/status.json',h.FEAS/'design/protocol.json']
for p in paths:
 try:p.open('rb')
 except PermissionError:pass
 else:raise AssertionError(p)
try:pathlib.Path('/tmp/highz-foreign-artifact-that-must-never-exist').open('w')
except PermissionError:pass
else:raise AssertionError('Foreign write permitted')
print('GUARD PASS')
"""
    p=subprocess.run([sys.executable,'-B','-c',code],env=h.environment(),text=True,capture_output=True,cwd=ROOT)
    assert p.returncode==0,p.stderr;assert 'GUARD PASS' in p.stdout
    assert not Path('/tmp/highz-foreign-artifact-that-must-never-exist').exists()
    assert not set(h.SLICES)&set(h.OTHER_SLICES)
    for key in ['TMPDIR','XDG_CACHE_HOME','HF_HOME','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR']:assert Path(h.environment()[key]).is_relative_to(h.OUT)


def test_paired_analysis_shared_draws_and_point_vs_broadening_metrics():
    import highz_h1_1102_report as h
    import highz_h1_1102 as controller
    from aion_magnitude.metrics import predict_photoz_from_logits
    n=24;z=np.tile(np.array([.1,.3,.5,.7,.9,1.2,1.6,2.1],np.float32),3);d=dict(z=z,targetid=np.arange(n))
    edge,center=make_redshift_grid(0,4.5,300);objects={}
    for i,a in enumerate(controller.ARMS):
        logits=-.5*((center[None,:]-torch.from_numpy(z[:,None])-i*.02)/(.08+i*.02))**2
        p={k:v.numpy() for k,v in predict_photoz_from_logits(logits,center).items()};p.update(logits=logits.numpy(),z_spec=z,redshift_edges=edge.numpy(),redshift_centers=center.numpy());objects[a]=h.additional_objects(p)
    for condition in ['lowz_only','natural_tail','tail_enriched']:objects['prior_'+condition]=objects['H1-A']
    objects['boundary']=objects['H1-A']
    master=np.random.default_rng(42).integers(0,n,(1000,n),dtype=np.uint32)
    r,draws=h.analysis(d,objects,dict(redshift_bins=[0,.6,.8,1,1.5,2,4.5]),master)
    high=r['strata']['above_cutoff'];assert high['n']==15
    a=high['cells']['H1-A'];dd=high['cells']['H1-D'];c=high['contrasts']['H1-D_minus_H1-A']
    assert c['ce']['value']==pytest.approx(dd['ce']['value']-a['ce']['value'])
    expected=draws['above_cutoff'][:,3,0]-draws['above_cutoff'][:,0,0]
    np.testing.assert_allclose(c['ce']['conditional_galaxy_ci95'],np.quantile(expected,[.025,.975]))
    assert dd['mean_width68']['value']>a['mean_width68']['value']
    assert 'fraction_predicted_below_cutoff' in a and len(high['diagnostics']['H1-A']['pit_counts'])==20
