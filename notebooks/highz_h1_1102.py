#!/usr/bin/env python3
"""Isolated six-arm HSC high-z H1 execution; no other experiment write API."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'notebooks')]
import numpy as np

SHARED = Path('/scratch/RedFM_MLP')
ARC = Path('/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints')
OUT = ARC/'1102/highz-hsc-pilot-zcut0p6'
FEAS = ARC/'1102/highz-feasibility'
PKG = ARC/'1028/matched-training-design'
TV = ARC/'1028/original-bidirectional-transfer/source_only'
PYTHON = SHARED/'.pixi/envs/default/bin/python'
BRANCH = 'redshift/h1-zcut0p6-1102'
SLICES = ['MIG-63d7886c-4484-57b3-acba-280cad46132a', 'MIG-07c57191-de6e-5338-8811-f0cfb7498db5']
OTHER_SLICES = ['MIG-1bba1f26-8b2b-5b97-9bb1-c886ab9bb8b7', 'MIG-0a2e21a6-8961-56b1-9eb4-997ae914386e', 'MIG-6c85a6c1-35aa-5ff1-ae36-a468e3e230b6', 'MIG-5bd31563-e820-50d1-a416-e14721d5a628', 'MIG-9c43b8e3-f3eb-566b-bdf8-74b3fa005f4d']
ARMS = {f'H1-{a}':dict(condition=c, head=h) for a,c,h in zip('ABCDEF', ['lowz_only','natural_tail','tail_enriched']*2, ['softmax']*3+['mixture']*3)}
# Pair each condition/head first so both architectures start promptly.
QUEUE = ['H1-A','H1-D','H1-B','H1-E','H1-C','H1-F']
CODE = ['aion_magnitude/highz_h1_head.py','aion_magnitude/highz_h1_training.py','notebooks/highz_h1_1102.py','notebooks/highz_h1_1102_report.py','tests/test_highz_h1_1102.py']


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''): h.update(b)
    return h.hexdigest()
def read(path):
    with np.load(path, allow_pickle=False) as f: return {k:f[k] for k in f.files}
def write(path, value):
    path = Path(path); assert path.is_relative_to(OUT), path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)
def fingerprint(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def git(*args): return subprocess.check_output(['git',*args], cwd=ROOT, text=True).strip()
def verify(mapping):
    for p,h in mapping.items(): assert sha(p)==h, f'Frozen file changed: {p}'


def protected_check():
    protected = json.loads((OUT/'isolation_audit/protected_files_before.json').read_text())
    verify(protected)
    assert subprocess.check_output(['git','branch','--show-current'],cwd=SHARED,text=True).strip()=='main'
    return dict(utc=now(),unchanged_files=len(protected),shared_branch='main',protected_slices=OTHER_SLICES)


def environment(gpu=''):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(ROOT)+':'+str(ROOT/'notebooks'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        TOKENIZERS_PARALLELISM='false', TMPDIR=str(OUT/'tmp'), XDG_CACHE_HOME=str(OUT/'cache'),
        HF_HOME=str(OUT/'cache/hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TORCHINDUCTOR_CACHE_DIR=str(OUT/'cache/torch'), TRITON_CACHE_DIR=str(OUT/'cache/triton'),
        MPLCONFIGDIR=str(OUT/'cache/matplotlib'))


def independent_rows(d):
    t = np.flatnonzero(d['category']=='T')
    low = t[d['z'][t]<.6]; high = t[d['z'][t]>=.6]
    def ordered(rows):
        hashes = [hashlib.sha256(f'1102:{int(x)}'.encode()).digest()[:8] for x in d['targetid'][rows]]
        return rows[np.argsort(np.array([int.from_bytes(x,'big') for x in hashes], dtype=np.uint64),kind='stable')]
    assert len(low)==14414 and len(high)==2186
    return dict(lowz_only=np.sort(ordered(low)[:10489]), natural_tail=np.sort(ordered(t)[:10489]),
        tail_enriched=np.sort(np.concatenate([high,ordered(low)[:10489-len(high)]])))


def prepare():
    import torch
    from transformers import AutoTokenizer
    from aion_magnitude.highz_h1_head import TruncatedGaussianMixtureHead, MIN_SCALE, MAX_SCALE
    from aion_magnitude.matched_training import state_fingerprint
    assert git('branch','--show-current')==BRANCH
    assert not (OUT/'manifest.json').exists(), 'Never overwrite a frozen H1 execution'
    tests = json.loads((OUT/'head_tests/verification.json').read_text()); assert tests['passed']
    protected_check()
    old = json.loads((FEAS/'design/manifest.json').read_text())
    verify(old['files_sha256']); assert sha(FEAS/'design/protocol.json')==old['protocol_sha256']
    frozen = {str(FEAS/'design/manifest.json'):sha(FEAS/'design/manifest.json'), **old['files_sha256']}
    write(OUT/'isolation_audit/feasibility_hashes.json', frozen)
    d = read(PKG/'paired_split/paired_cohort.npz')
    assert {c:int((d['category']==c).sum()) for c in 'TVP'}==dict(T=16600,V=4257,P=61696)
    assert len(np.unique(d['targetid']))==len(d['z'])
    rows = independent_rows(d); v=np.flatnonzero((d['category']=='V')&(d['z']<.6)); p=np.flatnonzero(d['category']=='P')
    assert len(v)==3745 and int((d['z'][p]>=.6).sum())==7910
    # This controller handles membership; training workers receive only source files.
    original_t = read(TV/'hsc_train.npz'); original_v=read(TV/'hsc_validation.npz')
    t=np.flatnonzero(d['category']=='T'); allv=np.flatnonzero(d['category']=='V')
    for x,ix in [(original_t,t),(original_v,allv)]:
        np.testing.assert_array_equal(x['targetid'],d['targetid'][ix]);np.testing.assert_array_equal(x['z'],d['z'][ix])
    tl={int(r):i for i,r in enumerate(t)};vl={int(r):i for i,r in enumerate(allv)}
    composition={}; token_audit={};tok=AutoTokenizer.from_pretrained('/arc/home/gsm/hf_models/Qwen3.5-4B-Base',local_files_only=True)
    for condition,tr in rows.items():
        old_membership=read(FEAS/'design/membership'/f'zcut0.6_{condition}.npz')
        for k,value in [('train_rows',tr),('validation_rows',v),('train_targetid',d['targetid'][tr]),('train_z',d['z'][tr])]:np.testing.assert_array_equal(old_membership[k],value)
        assert len(tr)==10489 and len(np.unique(d['targetid'][tr]))==10489
        assert not np.intersect1d(tr,p).size and not np.intersect1d(tr,v).size
        path=OUT/'splits'/f'{condition}_membership.npz'
        np.savez_compressed(path,train_rows=tr,validation_rows=v,train_targetid=d['targetid'][tr],validation_targetid=d['targetid'][v],train_z=d['z'][tr],validation_z=d['z'][v])
        for split,source,indices in [('train',original_t,[tl[int(x)] for x in tr]),('validation',original_v,[vl[int(x)] for x in v])]:
            path=OUT/'splits'/f'{condition}_{split}.npz';np.savez_compressed(path,**{k:a[indices] for k,a in source.items()})
            existing=read(FEAS/'design/inputs'/f'zcut0.6_{condition}_hsc_{split}.npz');actual=read(path)
            for k in actual:np.testing.assert_array_equal(actual[k],existing[k])
            lens=np.array(list(map(len,tok(actual['text'].tolist(),truncation=False)['input_ids'])));assert lens.max()<=512
            token_audit[f'{condition}/{split}']=dict(n=len(lens),min=int(lens.min()),max=int(lens.max()),truncated=0,source_text_bytes_unchanged=True)
        composition[condition]=dict(total=len(tr),below_cutoff=int((d['z'][tr]<.6).sum()),at_or_above_cutoff=int((d['z'][tr]>=.6).sum()),id_sha256=fingerprint(d['targetid'][tr]),validation=3745)
    write(OUT/'splits/membership_audit.json',dict(independently_recomputed=True,agrees_with_frozen_design=True,counts=composition,duplicates=0,train_validation_P_disjoint=True))
    write(OUT/'protocol/input_audit.json',token_audit)
    from scipy.stats import ks_2samp
    def phot_summary(ix):
        f=d['hsc_features'][ix,4].astype(float);err=d['hsc_features'][ix,5].astype(float)
        snr=f/err
        with np.errstate(divide='ignore',invalid='ignore'):mag=np.where(f>0,22.5-2.5*np.log10(f),np.nan)
        def summary(x):
            good=x[np.isfinite(x)];return dict(n_finite=len(good),quantiles_05_25_50_75_95=np.quantile(good,[.05,.25,.5,.75,.95]).tolist())
        return dict(n=len(ix),i_mag=summary(mag),i_SNR=summary(snr))
    groups={**rows,'validation_low':v,'P_low':p[d['z'][p]<.6],'P_high':p[d['z'][p]>=.6]}
    for lo,hi in zip([0,.2,.4,.6,.8,1,1.5,2],[.2,.4,.6,.8,1,1.5,2,4.5]):
        ix=p[(d['z'][p]>=lo)&(d['z'][p]<hi)]
        if len(ix):groups[f'P_{lo:g}_{hi:g}']=ix
    write(OUT/'protocol/selection_diagnostics.json',dict(groups={k:phot_summary(ix) for k,ix in groups.items()},interpretation='Supervised redshift support holdout also changes galaxy-property and selection distributions; observational properties are not held fixed. No reweighting/tuning from P.'))
    shutil.copy2(FEAS/'design/presentation_order_seed42.npy',OUT/'splits/presentation_order_seed42.npy')
    orders=np.load(OUT/'splits/presentation_order_seed42.npy');assert orders.shape==(10,10489)
    for e,o in enumerate(orders):np.testing.assert_array_equal(o,np.random.default_rng(42+20000+e).permutation(10489))
    # Locked benchmark information lives separately; the worker guard refuses this directory.
    np.savez_compressed(OUT/'evaluation/locked_P.npz',**{k:a[p] for k,a in d.items()})
    records={int(x['targetid']):x['text'] for x in map(json.loads,(PKG/'representation_audit/hsc_inputs.jsonl').open())}
    with (OUT/'evaluation/P_inputs.jsonl').open('w') as f:
        for tid in d['targetid'][p]:f.write(json.dumps(dict(targetid=str(tid),text=records[int(tid)]))+'\n')
    # Read the original master draws only after all six fixed-final checkpoints freeze.
    master=PKG/'evaluation_policy/bootstrap_indices.npy'
    policy=dict(version='highz_H1_v1',created_utc=now(),cutoff=.6,arms=ARMS,primary_checkpoint='fixed_final_epoch10',
        secondary_checkpoint='minimum_CE_on_the_same_3745_original_V_below_cutoff',nP=61696,high_P=7910,low_P=53786,
        benchmark='Previously analysed, locked paired-P benchmark; not a new prospective blind test',
        gating='All six own final checkpoint hashes and source-validation replays frozen before any P predictions',
        edges='Exact canonical float32 torch.linspace(0,4.5,301), integrated in float64 for mixture',
        target_bin='Unchanged torch.bucketize(z,edges,right=False)-1, clamped0..299. Exact interior-edge labels go to lower bin.',
        loss='Unchanged RedFM binned categorical cross entropy for both heads; normalized log bin masses are mixture logits',
        mixture=dict(components=3,mean='4.5*sigmoid(raw)',scale='min+(max-min)*sigmoid(raw)',minimum_scale=MIN_SCALE,maximum_scale=MAX_SCALE,
            truncation='One global mixture normalization over[0,4.5], not componentwise truncation',arithmetic='float64 stable log-CDF intervals, float32 returned log-mass logits',
            output_seed=421102,initial_means=[.75,2.25,3.75],initial_scale=.3,output_weights_std=.01),
        redshift_bins=[0.,.2,.4,.6,.8,1.,1.5,2.,4.5],exploratory_count_threshold=100,
        above_cut_mass='Piecewise-uniform within-bin overlap with[0.6,4.5]; same convention for both heads',
        bootstrap=dict(master_file=str(master),master_sha256=sha(master),draws=1000,policy='Original master paired-galaxy draws conditioned on each fixed spec-z stratum; shared across all arms and baselines',interval='percentile95%, conditional on one training seed'),
        boundary_baseline='Native-bin-centre Gaussian: last centre below0.6, sigma one bin0.015; floor mass1e-12 then normalize',
        prior_baseline='Each condition train-label histogram with additive1e-3 pseudocount per bin, normalized; no P fit',
        selection_diagnostics='HSC observed i magnitude and SNR distributions in training and locked-P strata; no feature reweighting',
        failure_rules=json.loads((FEAS/'design/protocol.json').read_text())['failure_rules'],
        scientific_contrasts=['D-A','B-A','C-A','E-D','F-D','C-B','F-E'],
        prohibitions=['No Legacy','No DoRA','No new seeds','No new models','No MLP/TabICL','No zcut1.5','No z>2 experiment','No reading other-branch P predictions'])
    write(OUT/'protocol/evaluation_policy.json',policy)
    # Own copies of every initial trainable state, with common canonical hidden trunk.
    initial=OUT/'protocol/initialisation';initial.mkdir()
    canonical=torch.load(PKG/'model_initialisation/qlora-seed42/canonical_initial_trainable_state.pt',map_location='cpu',weights_only=True)
    head=torch.load(PKG/'model_initialisation/head_seed42.pt',map_location='cpu',weights_only=True)
    torch.save(canonical,initial/'softmax_trainable.pt');torch.save(head,initial/'softmax_head.pt')
    h=TruncatedGaussianMixtureHead(head['net.0.weight'].shape[1],head['net.0.weight'].shape[0]);hs=h.state_dict()
    for k,val in head.items():
        if k in hs and hs[k].shape==val.shape:hs[k]=val.clone()
    h.load_state_dict(hs)
    mixed={k:v for k,v in canonical.items() if not k.startswith('photoz_head.')};mixed.update({'photoz_head.'+k:v for k,v in hs.items()})
    torch.save(mixed,initial/'mixture_trainable.pt');torch.save(hs,initial/'mixture_head.pt')
    params=dict(softmax=sum(v.numel() for v in head.values()),mixture=sum(p.numel() for p in h.parameters()),shared_trunk=sum(v.numel() for k,v in head.items() if not k.startswith('net.6.')))
    assert params['softmax']-params['mixture']==(300-9)*(256+1)
    write(OUT/'protocol/head_audit.json',dict(parameters=params,shared_trunk_exact=True,identical_head_capacity=False,
        initial_trainable_fingerprint=dict(softmax=state_fingerprint(canonical),mixture=state_fingerprint(mixed)),
        softmax='Canonical PhotoZHead output300; CE suppresses a never-positive bin via gradient p_j. This is inductive pressure, not impossibility proof.',
        grid_and_loss='Original finite logits use log_softmax for stable CE; p(z)=softmax can underflow to zero in float32. No additional inference floor is applied to either trained head.',
        mixture='Output9 replaces output300; identical hidden trunk and QLoRA, bounded continuous Gaussian component parameters; global truncation couples bins. Architectural control, not promised improvement.'))
    for arm,desc in ARMS.items():
        cfg=json.loads((FEAS/'design/configs'/f"zcut0.6_{desc['condition']}_hsc_qlora_seed42.json").read_text())
        cfg.update(version='isolated_highz_H1_v1',name=arm,condition=desc['condition'],head=desc['head'],training_authorized=True,P_inference_authorized=True,
            output_dir=str(OUT/'training'/arm),primary_checkpoint='fixed_final_epoch10',source_selection=policy['secondary_checkpoint'],
            initial_head=str(initial/'softmax_head.pt'),initial_trainable_state=str(initial/f"{desc['head']}_trainable.pt"),
            expected_initial_trainable_tensor_sha256=state_fingerprint(canonical if desc['head']=='softmax' else mixed),
            train_path=str(OUT/'splits'/f"{desc['condition']}_train.npz"),validation_path=str(OUT/'splits'/f"{desc['condition']}_validation.npz"),
            order_path=str(OUT/'splits/presentation_order_seed42.npy'))
        cfg['train_sha256']=sha(cfg['train_path']);cfg['validation_sha256']=sha(cfg['validation_path']);cfg['order_sha256']=sha(cfg['order_path']);cfg['initial_head_tensor_sha256']=state_fingerprint(head)
        write(OUT/'protocol/configs'/f'{arm}.json',cfg)
        (OUT/'training'/arm).mkdir()
    # Baselines use source labels only; compute before P predictions exist.
    edges=np.linspace(0,4.5,301,dtype=np.float32);centers=(edges[:-1]+edges[1:])/2
    import torch as _torch
    from aion_magnitude.utils import make_redshift_grid
    te,tc=make_redshift_grid(0,4.5,300);edges=te.numpy();centers=tc.numpy()
    prior={}
    for c in composition:
        z=read(OUT/'splits'/f'{c}_train.npz')['z'];bins=_torch.bucketize(_torch.from_numpy(z),te).numpy()-1
        hist=np.bincount(np.clip(bins,0,299),minlength=300)+.001;prior[c]=hist/hist.sum()
    boundary=np.exp(-.5*((centers.astype(float)-float(centers[centers<.6][-1]))/.015)**2);boundary=np.maximum(boundary,1e-12);boundary/=boundary.sum()
    np.savez_compressed(OUT/'protocol/baselines.npz',boundary=boundary,redshift_edges=edges,redshift_centers=centers,**prior)
    assets=[p for parent in [OUT/'splits',OUT/'protocol'] for p in parent.rglob('*') if p.is_file()]
    manifest=dict(created_utc=now(),base_sha=json.loads((OUT/'isolation_audit/worktree.json').read_text())['base_sha'],execution_commit=git('rev-parse','HEAD'),branch=BRANCH,worktree=str(ROOT),
        code_sha256={str(ROOT/p):sha(ROOT/p) for p in CODE},request_sha256=sha(ROOT/'redshift_1102.md'),
        frozen_feasibility_hashes_sha256=sha(OUT/'isolation_audit/feasibility_hashes.json'),assets_sha256={str(p):sha(p) for p in assets},
        P_sha256={str(p):sha(p) for p in [OUT/'evaluation/locked_P.npz',OUT/'evaluation/P_inputs.jsonl']},head_tests_sha256=sha(OUT/'head_tests/verification.json'),
        environment_versions={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','bitsandbytes','numpy','scipy']},gpu_allowlist=SLICES,gpu_denylist=OTHER_SLICES,arms=list(ARMS))
    write(OUT/'manifest.json',manifest)
    for p in CODE+['redshift_1102.md']:
        dest=OUT/'protocol/code_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
    write(OUT/'isolation_audit/after_preparation.json',protected_check())
    make_report()
    write(OUT/'status.json',dict(updated_utc=now(),stage='prepared_correctness_gates_passed',pending=QUEUE,active={},failed={}))
    print('PREPARED',json.dumps(composition),params,flush=True)


def validate():
    m=json.loads((OUT/'manifest.json').read_text())
    assert git('branch','--show-current')==BRANCH and git('rev-parse','HEAD')==m['execution_commit']
    verify(m['code_sha256']);verify(m['assets_sha256'])
    assert sha(OUT/'head_tests/verification.json')==m['head_tests_sha256']
    for n,v in m['environment_versions'].items():assert importlib.metadata.version(n)==v
    return m


def config(arm):
    assert arm in ARMS;return json.loads((OUT/'protocol/configs'/f'{arm}.json').read_text())


def install_worker_guard():
    """Training cannot open P files or write outside its execution root.

    Code/cache reads and cooperative device-lease descriptors are permitted;
    all newly generated data are restricted to this root by OS audit events.
    """
    def guard(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        text=os.fsdecode(args[0]);p=Path(text).resolve()
        flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        mode=args[1] if len(args)>1 and isinstance(args[1],str) else ''
        writing=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)) or any(x in mode for x in 'wax+')
        if writing and not p.is_relative_to(OUT) and text not in ['/dev/null','/dev/tty']:raise PermissionError(f'H1 artifact write outside own root: {p}')
        forbidden=(p.is_relative_to(OUT/'evaluation') or p.is_relative_to(FEAS) or p.is_relative_to(SHARED) and ('filter-aware-runs' in str(p) or 'reports' in p.parts))
        # Explicitly deny the shared full cohorts and any other branch's artifacts.
        allowed_shared=[PKG/'model_initialisation',PKG/'manifest.json']
        if p.is_relative_to(ARC) and not p.is_relative_to(OUT) and not any(p==x or p.is_relative_to(x) for x in allowed_shared):forbidden=True
        if forbidden:raise PermissionError(f'P/foreign-experiment data prohibited in H1 training: {p}')
    sys.addaudithook(guard)


@contextmanager
def own_gpu():
    from scripts.qwen_survey_gpu_guard import driver_uuids,memory_info
    selected=os.environ.get('CUDA_VISIBLE_DEVICES','');assert selected in SLICES and selected not in OTHER_SLICES
    uuids=driver_uuids();assert uuids==[selected.removeprefix('MIG-')],uuids
    inherited=os.environ.get('HIGHZ_DEVICE_LEASE_FD')
    if inherited:
        fd=int(inherited);assert os.path.exists(f'/proc/self/fd/{fd}')
        target=Path(os.readlink(f'/proc/self/fd/{fd}'))
        assert target.name==selected.removeprefix('MIG-')+'.lock'
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield dict(uuid=selected,inherited_lease=True,memory_before_gib=memory_info(uuids[0]))
    else:
        lock=Path('/scratch/.tmp-gsm/aion_output/gpu_locks')/str(os.getuid())/(uuids[0]+'.lock')
        with lock.open('r') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);info=memory_info(uuids[0]);assert info['used']<.25 and info['free']>=8
            yield dict(uuid=selected,inherited_lease=False,memory_before_gib=info)


def build(cfg):
    import torch
    from aion_magnitude.matched_training import build_fresh_model,state_fingerprint
    from aion_magnitude.highz_h1_head import replace_head
    model,tok,device=build_fresh_model(cfg)
    if cfg['head']=='mixture':replace_head(model)
    actual=state_fingerprint({k:v for k,v in model.named_parameters() if v.requires_grad})
    assert actual==cfg['expected_initial_trainable_tensor_sha256'],actual
    return model,tok,device


def smoke(model,tok,device,cfg,train,run):
    import torch
    from aion_magnitude.matched_training import rng_state,restore_rng,state_fingerprint
    from aion_magnitude.qwen_posttraining import make_text_collator
    from aion_magnitude.metrics import redshift_cross_entropy_loss,predict_photoz_from_logits
    from aion_magnitude.utils import make_redshift_grid,set_random_seed
    names={k for k,p in model.named_parameters() if p.requires_grad}
    initial={k:p.detach().cpu().clone() for k,p in model.named_parameters() if k in names};rng=rng_state();set_random_seed(421102)
    edge,center=make_redshift_grid(0,4.5,300);stats={}
    adapter_ids={id(p) for n,p in model.named_parameters() if n in names and n.startswith('qwen.')}
    for phase in ['head_only','joint']:
        model.train();model.freeze_qwen_representation=phase=='head_only'
        for p in model.qwen.parameters():
            # Only the canonical adapter parameters participate.
            p.requires_grad_(id(p) in adapter_ids and phase=='joint')
        if phase=='head_only':model.qwen.eval()
        model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats(device)
        batch=make_text_collator(tok,max_length=512)([dict(text=str(train['text'][0]),z_spec=torch.tensor(float(train['z'][0])),object_id=str(train['targetid'][0]))])
        z=batch.pop('z_spec').to(device);batch.pop('object_id');batch={k:v.to(device) for k,v in batch.items()};start=time.time()
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(**batch);loss=redshift_cross_entropy_loss(logits.float(),z,edges=edge.to(device))
        assert torch.isfinite(logits).all() and torch.isfinite(loss)
        loss.backward();torch.cuda.synchronize(device)
        grads={part:sum(float(p.grad.float().square().sum()) for n,p in model.named_parameters() if n in names and p.grad is not None and (n.startswith('photoz_head.') if part=='head' else n.startswith('qwen.')))**.5 for part in ['head','adapter']}
        assert grads['head']>0 and (phase=='head_only' or grads['adapter']>0)
        output_grad=sum(float(p.grad.float().norm()) for p in model.photoz_head.net[6].parameters());assert output_grad>0
        with torch.no_grad():
            for p in model.photoz_head.parameters():
                if p.grad is not None:p.add_(p.grad,alpha=-1e-5)
        assert any(not torch.equal(p.detach().cpu(),initial[n]) for n,p in model.named_parameters() if n.startswith('photoz_head.'))
        pred=predict_photoz_from_logits(logits.float(),center.to(device))
        stats[phase]=dict(loss=float(loss),seconds=time.time()-start,gradient_norms=grads,output_gradient_norm=output_grad,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
            initial_logits_min=float(logits.min()),initial_logits_max=float(logits.max()),initial_probability_mass_above_cutoff=float(pred['pz'][:,center.to(device)>=.6].sum()),only_T_record=True)
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n in names:p.copy_(initial[n].to(p.device))
        model.zero_grad(set_to_none=True)
    for n,p in model.named_parameters():
        if n in names:p.requires_grad_(True)
    assert state_fingerprint({n:p for n,p in model.named_parameters() if n in names})==cfg['expected_initial_trainable_tensor_sha256']
    restore_rng(rng);model.freeze_qwen_representation=False;torch.cuda.empty_cache()
    write(run/'smoke_test.json',dict(passed=True,phases=stats,canonical_initial_state_restored=True,RNG_restored=True,official_presentations_consumed=0))


def train(arm):
    import torch
    from aion_magnitude.highz_h1_training import train_loop
    m=validate();cfg=config(arm);run=Path(cfg['output_dir']);assert run==OUT/'training'/arm
    # All model/split identities verified before installation of P/write guard.
    assert sha(cfg['base_identity_path'])==cfg['base_identity_sha256']
    base=json.loads(Path(cfg['base_identity_path']).read_text())
    # Full shared base files are verified once by preparation/launch; workers use
    # read-only size/mtime plus the frozen identity, avoiding six redundant 8GB scans.
    proof=json.loads((OUT/'protocol/base_verification.json').read_text())
    for p,info in base['files'].items():
        s=Path(p).stat();assert s.st_size==proof[p]['size'] and s.st_mtime_ns==proof[p]['mtime_ns']
    with (run/'training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (run/'complete.json').exists():return
        install_worker_guard()
        data={label:read(cfg[label+'_path']) for label in ['train','validation']}
        assert len(data['train']['z'])==10489 and len(data['validation']['z'])==3745 and np.all(data['validation']['z']<.6)
        if cfg['condition']=='lowz_only':assert np.all(data['train']['z']<.6)
        assert not np.intersect1d(data['train']['targetid'],data['validation']['targetid']).size
        orders=np.load(cfg['order_path'])
        resume=bool(list((run/'checkpoints').glob('resume-slot*.pt')))
        with own_gpu() as gpu:
            assert gpu['memory_before_gib']['free']>=8 and gpu['memory_before_gib']['used']<.25
            started=time.time();model,tok,device=build(cfg)
            write(run/'initialization_proof.json',dict(created_utc=now(),gpu=gpu,verified_fingerprint=cfg['expected_initial_trainable_tensor_sha256'],head=cfg['head'],source_only=True))
            if not resume:smoke(model,tok,device,cfg,data['train'],run)
            write(run/'config.json',cfg)
            result=train_loop(model,tok,device,data['train']['text'].tolist(),data['train']['z'],data['train']['targetid'],
                data['validation']['text'].tolist(),data['validation']['z'],data['validation']['targetid'],cfg,run,saved_orders=orders,resume=resume)
            assert result['global_update']==6560 and result['examples_seen']==104890
            write(run/'resource_cost.json',dict(finished_utc=now(),seconds_this_process=time.time()-started,MIG_hours_this_process=(time.time()-started)/3600,gpu=gpu,resumed=resume))


def freeze_final():
    validate();models={}
    for arm in ARMS:
        run=OUT/'training'/arm;done=json.loads((run/'complete.json').read_text());proof=json.loads((run/'completion_checks.json').read_text())
        assert done['global_update']==6560 and done['examples_seen']==104890 and len(done['history'])==10 and proof['passed']
        replay=json.loads((run/'source_validation_replay.json').read_text());assert all(v['exact'] for v in replay.values())
        final=json.loads((run/'final/metadata.json').read_text());selected=json.loads((run/'selected/metadata.json').read_text())
        for label,meta in [('final',final),('selected',selected)]:
            verify({str(run/label/p):h for p,h in meta['checkpoint_sha256'].items()})
            assert (run/label/'trainable_state.pt').exists()
        assert final['global_update']==6560
        assert all(h['gradient_diagnostics']['head_output_grad_max']>0 for h in done['history'])
        models[arm]=dict(primary_directory=str(run/'final'),primary_update=6560,primary_sha256=final['checkpoint_sha256'],
            secondary_directory=str(run/'selected'),secondary_update=done['best_update'],secondary_sha256=selected['checkpoint_sha256'],secondary_differs_from_final=done['best_update']!=6560,primary_trainable_state_sha256=sha(run/'final/trainable_state.pt'),secondary_trainable_state_sha256=sha(run/'selected/trainable_state.pt'))
    gate=dict(created_utc=now(),all_six_fixed_final_complete=True,models=models,evaluation_policy_sha256=sha(OUT/'protocol/evaluation_policy.json'))
    path=OUT/'evaluation/final_checkpoints_frozen.json'
    if path.exists():assert json.loads(path.read_text())['models']==models
    else:write(path,gate)
    return gate


def infer(arm):
    import torch
    from aion_magnitude.qwen_posttraining import TextRedshiftDataset,make_text_collator,evaluate_text_photoz_model
    from aion_magnitude.utils import make_redshift_grid
    from aion_magnitude.matched_training import state_fingerprint
    from torch.utils.data import DataLoader
    m=validate();gate=freeze_final();cfg=config(arm);folder=OUT/'evaluation'/arm;folder.mkdir(exist_ok=True)
    verify(m['P_sha256']);d=read(OUT/'evaluation/locked_P.npz');records=list(map(json.loads,(OUT/'evaluation/P_inputs.jsonl').open()))
    np.testing.assert_array_equal(np.array([int(r['targetid']) for r in records]),d['targetid']);assert len(d['z'])==61696
    state=gate['models'][arm];verify({str(Path(state['primary_directory'])/p):h for p,h in state['primary_sha256'].items()})
    with (folder/'inference.lock').open('a+') as lock, own_gpu() as gpu:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model,tok,device=build(cfg)
        assert sha(Path(state['primary_directory'])/'trainable_state.pt')==state['primary_trainable_state_sha256']
        current=dict(model.named_parameters());saved=torch.load(Path(state['primary_directory'])/'trainable_state.pt',map_location='cpu',weights_only=True)
        with torch.no_grad():
            for k,v in saved.items():current[k].copy_(v.to(current[k].device))
        model.eval();model.freeze_qwen_representation=False
        for p in model.parameters():p.requires_grad_(False)
        before=state_fingerprint({k:current[k] for k in saved});edge,center=make_redshift_grid(0,4.5,300)
        def ev(text,z,ids):
            loader=DataLoader(TextRedshiftDataset(text,z,list(map(str,ids))),batch_size=8,shuffle=False,collate_fn=make_text_collator(tok,max_length=512))
            return evaluate_text_photoz_model(model,loader,device=device,redshift_edges=edge,redshift_centers=center)
        source=read(cfg['validation_path']);ref=torch.load(Path(state['primary_directory'])/'validation_predictions.pt',map_location='cpu',weights_only=False)
        replay=ev(source['text'][:64].tolist(),source['z'][:64],source['targetid'][:64]);diff={k:float((replay[k]-ref[k][:64]).abs().max()) for k in ['logits','pz','z_p16','z_p50','z_p84']};assert all(v==0 for v in diff.values()),diff
        write(folder/'source_replay.json',dict(n=64,exact=True,differences=diff))
        started=time.time()
        for start in range(0,len(d['z']),256):
            end=min(start+256,len(d['z']));path=folder/f'{start:06d}_{end:06d}.npz'
            if path.exists():
                old=read(path);np.testing.assert_array_equal(old['targetid'],d['targetid'][start:end]);np.testing.assert_array_equal(old['z_spec'],d['z'][start:end]);continue
            texts=[r['text'] for r in records[start:end]];assert max(map(len,tok(texts,truncation=False)['input_ids']))<=512
            pred=ev(texts,d['z'][start:end],d['targetid'][start:end]);assert torch.isfinite(pred['logits']).all() and torch.isfinite(pred['pz']).all()
            tmp=path.with_name(path.stem+'.partial.npz');np.savez_compressed(tmp,targetid=d['targetid'][start:end],**{k:pred[k].numpy() for k in ['logits','pz','z_spec','z_p16','z_p50','z_p84','redshift_edges','redshift_centers']});tmp.replace(path)
            write(folder/'progress.json',dict(updated_utc=now(),completed=end,n=61696,seconds=time.time()-started,gpu=gpu))
        assert state_fingerprint({k:current[k] for k in saved})==before
    write(folder/'complete.json',dict(finished_utc=now(),n=61696,primary_update=6560,prediction_sha256={p.name:sha(p) for p in folder.glob('??????_??????.npz')},seconds=time.time()-started))


@contextmanager
def reserve_spares():
    """Hold only the two allowed cooperative locks throughout this pilot queue."""
    from scripts.qwen_survey_gpu_guard import memory_info
    handles={};own=[]
    try:
        protected_check()
        for gpu in SLICES:
            uuid=gpu.removeprefix('MIG-');path=Path('/scratch/.tmp-gsm/aion_output/gpu_locks')/str(os.getuid())/(uuid+'.lock')
            h=path.open('r')
            try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:h.close();continue
            info=memory_info(uuid)
            if info['free']<8 or info['used']>=.25:h.close();continue
            lease=(OUT/'isolation_audit'/f'{uuid}.reservation.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);own.append(lease);handles[gpu]=h
        yield handles
    finally:
        for h in [*handles.values(),*own]:h.close()


def run_queue(stage,jobs,leases):
    active={};pending=list(jobs);finished=[];failed={}
    while active or pending:
        for gpu in leases:
            if gpu in active:
                arm,proc,log=active[gpu];code=proc.poll()
                if code is not None:
                    log.close();del active[gpu]
                    if code==0:finished.append(arm)
                    else:failed[arm]=dict(exit_code=code,log=str(OUT/'logs'/f'{stage}_{arm}.log'))
            if gpu not in active and pending and not failed:
                from scripts.qwen_survey_gpu_guard import memory_info
                mem=memory_info(gpu.removeprefix('MIG-'))
                if mem['free']<8 or mem['used']>=.25:continue
                protected_check();arm=pending.pop(0);fd=leases[gpu].fileno();env=environment(gpu);env['HIGHZ_DEVICE_LEASE_FD']=str(fd)
                log=(OUT/'logs'/f'{stage}_{arm}.log').open('a');action='train' if stage=='source_training' else 'infer'
                command=[str(PYTHON),'-B',str(Path(__file__).resolve()),action,'--arm',arm]
                proc=subprocess.Popen(command,cwd=ROOT,env=env,pass_fds=(fd,),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                active[gpu]=(arm,proc,log);write(OUT/'logs'/f'{stage}_{arm}_launch.json',dict(utc=now(),gpu=gpu,pid=proc.pid,command=command))
                # Avoid simultaneous base-weight read storms.
                time.sleep(8)
        write(OUT/'status.json',dict(updated_utc=now(),stage=stage,pending=pending,active={g:dict(arm=a,pid=p.pid) for g,(a,p,l) in active.items()},finished=finished,failed=failed,protected_slices=OTHER_SLICES))
        if failed and not active:raise RuntimeError(f'H1 failure; later launches gated, independent arms/checkpoints preserved: {failed}')
        if active or pending:time.sleep(15)


def make_report():
    from highz_h1_1102_report import pending_report
    pending_report()


def supervise():
    os.nice(10)
    with (OUT/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);validate()
        while True:
            with reserve_spares() as leases:
                if not leases:
                    write(OUT/'status.json',dict(updated_utc=now(),stage='queued_waiting_for_independent_resources',pending=QUEUE,active={},failed={},protected_slices=OTHER_SLICES));make_report()
                else:
                    write(OUT/'isolation_audit/active_reservations.json',dict(utc=now(),supervisor_pid=os.getpid(),slices=list(leases),other_slices_never_acquired=OTHER_SLICES))
                    jobs=[a for a in QUEUE if not (OUT/'training'/a/'complete.json').exists()]
                    make_report();run_queue('source_training',jobs,leases);freeze_final()
                    jobs=[a for a in QUEUE if not (OUT/'evaluation'/a/'complete.json').exists()]
                    run_queue('locked_P_fixed_final',jobs,leases)
                    break
            time.sleep(30)
        from highz_h1_1102_report import report
        report();write(OUT/'isolation_audit/final_protection_check.json',protected_check())
        write(OUT/'complete.json',dict(finished_utc=now(),arms=6,primary='fixed_final_epoch10',seed=42,no_expansion_launched=True))
        write(OUT/'status.json',dict(updated_utc=now(),stage='complete',active={},pending=[],failed={},arms=6))


def base_verify():
    cfg=json.loads((FEAS/'design/configs/zcut0.6_lowz_only_hsc_qlora_seed42.json').read_text());identity=json.loads(Path(cfg['base_identity_path']).read_text());result={}
    assert sha(cfg['base_identity_path'])==cfg['base_identity_sha256']
    for p,info in identity['files'].items():
        assert sha(p)==info['sha256'];s=Path(p).stat();result[p]=dict(sha256=info['sha256'],size=s.st_size,mtime_ns=s.st_mtime_ns)
    write(OUT/'protocol/base_verification.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','base-verify','supervise','train','infer','report','freeze']);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='base-verify':base_verify()
    elif a.action=='supervise':supervise()
    elif a.action=='train':train(a.arm)
    elif a.action=='infer':infer(a.arm)
    elif a.action=='freeze':freeze_final()
    else:
        from highz_h1_1102_report import report
        report()

if __name__=='__main__':main()
