#!/usr/bin/env python3
"""Versioned, gated execution of Bidirectional_1029.md. No new-model P inference API."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'notebooks')]
import numpy as np

PKG = (ROOT/'filter-aware-runs/1028/matched-training-design').resolve()
BENCH = (ROOT/'filter-aware-runs/1027/hsc-legacy-paired-full').resolve()
OUT = (ROOT/'filter-aware-runs/1028/original-bidirectional-transfer').resolve()
EXPECTED = 'bc7ac99fda7629f1235fb1a4c547e699fa9039984e82a474ebe654d418183661'
SLICES = ['MIG-1bba1f26-8b2b-5b97-9bb1-c886ab9bb8b7', 'MIG-0a2e21a6-8961-56b1-9eb4-997ae914386e', 'MIG-6c85a6c1-35aa-5ff1-ae36-a468e3e230b6']
NAMES = [f'{s}-{m}' for s in ('hsc','legacy') for m in ('qlora','dora')]
BINS = [0., .2, .4, .6, 1., 4.5]

def now(): return datetime.now(timezone.utc).isoformat()
def read(p):
    with np.load(p, allow_pickle=False) as f: return {k:f[k] for k in f.files}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''): h.update(block)
    return h.hexdigest()
def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.partial');q.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');q.replace(p)
def verify(mapping, relative=False):
    for p,h in mapping.items():
        path=ROOT/p if relative else Path(p)
        assert sha(path)==h, f'Protected artifact changed: {path}'
def texts(data, domain):
    from aion_magnitude.filter_photometry import feature_columns,serialize_rows
    kind='galaxiesml-spec' if domain=='hsc' else 'desi-ls'
    names=[n for _,v,e in feature_columns(kind) for n in (v,e)]+['MW_EBV']
    return serialize_rows(dict(features=data[f'{domain}_features'],group=data['galaxy_id'] if domain=='hsc' else data['legacy_group']),dict(feature_names=names),catalogue=kind,level=0)
def ordered_texts(p, ids):
    rows={str(r['targetid']):r['text'] for r in map(json.loads,Path(p).open())}
    return [rows[str(x)] for x in ids]
def subset(d, ix): return {k:v[ix] for k,v in d.items()}
def load_manifest():
    m=json.loads((OUT/'manifest.json').read_text())
    assert sha(PKG/'manifest.json')==EXPECTED
    verify(m['execution_code_sha256'],relative=True)
    assert sha(OUT/'cohort.npz')==m['cohort_sha256']
    assert sha(OUT/'policy.json')==m['policy_sha256']
    return m

def prepare():
    if OUT.exists(): raise FileExistsError(f'Execution already exists: {OUT}')
    assert sha(PKG/'manifest.json')==EXPECTED
    p=json.loads((PKG/'manifest.json').read_text())
    verify(p['code_sha256'],relative=True);verify(p['immutable_data_sha256']);verify(p['final_config_sha256']);verify(p['original_artifact_sha256'])
    for name,v in p['environment_versions'].items(): assert importlib.metadata.version(name)==v
    assert sha(BENCH/'manifest.json')==p['benchmark_manifest_sha256']
    original=json.loads((BENCH/'manifest.json').read_text())
    assert sha(BENCH/'cohort.npz')==original['cohort_sha256']
    cohort=PKG/'original_model_exposure/mutual_cohort.npz';d=read(cohort)
    assert len(d['z'])==61310 and len(np.unique(d['targetid']))==61310 and np.all(d['category']=='P')
    OUT.mkdir();shutil.copy2(cohort,OUT/'cohort.npz')
    identity=dict(n=61310,cohort_sha256=sha(cohort),targetid_sha256=hashlib.sha256(d['targetid'].tobytes()).hexdigest(),
                  galaxy_id_sha256=hashlib.sha256(d['galaxy_id'].tobytes()).hexdigest(),label_sha256=hashlib.sha256(d['z'].tobytes()).hexdigest(),
                  ordering='Exact prepared P_mutual order; no re-selection',excluded_original_P=386,
                  exposure_audit_sha256=sha(PKG/'original_model_exposure/audit.json'),exposure_flags_sha256=sha(PKG/'original_model_exposure/exposure_flags.npz'))
    write(OUT/'cohort_identity.json',identity)
    models={};protected={};configs={}
    for name in NAMES:
        cp=PKG/'original_model_exposure'/f'original-{name}-hsc.json';cfg=json.loads(cp.read_text());configs[str(cp)]=sha(cp)
        assert sha(cohort)==cfg['cohort_sha256'] and cfg['n']==61310
        models[name]=dict(directory=cfg['directory'],config=cfg['training']);verify(cfg['protected_sha256']);protected.update(cfg['protected_sha256'])
    input_hashes={};audit={}
    for domain in ('hsc','legacy'):
        assert sha(BENCH/'inputs'/f'level0_{domain}.jsonl')==original['input_sha256'][f'level0_{domain}']
        ts=texts(d,domain);old=ordered_texts(BENCH/'inputs'/f'level0_{domain}.jsonl',d['targetid']);assert ts==old
        path=OUT/'inputs'/f'{domain}.jsonl';path.parent.mkdir(exist_ok=True)
        with path.open('w') as f:
            for oid,t in zip(d['targetid'],ts): f.write(json.dumps(dict(targetid=str(oid),text=t))+'\n')
        input_hashes[str(path)]=sha(path)
        audit[domain]=dict(n=len(ts),example=ts[0],archived_inputs_byte_equal=True,
            physical_fields='HSC grizy cmodel' if domain=='hsc' else 'Legacy grzW1W2 Tractor',
            schema='galaxiesml-spec_native_photometry_specz_v1' if domain=='hsc' else 'desi-ls_native_photometry_specz_v1')
    write(OUT/'input_compatibility.json',dict(domains=audit,native_cells='Original trained source schema',cross_domain_cells='interface-stress diagnostic',
          explanation='Cross-domain records retain actual destination-survey named filters and schema. No positional band replacement, imputation, or fictitious source photometry. Source-specific Level-0 contracts make these interface-stress diagnostics, not clean filter-aware transfer.'))
    b=np.lib.format.open_memmap(OUT/'bootstrap_indices.npy',mode='w+',dtype=np.uint32,shape=(1000,61310))
    rng=np.random.default_rng(1029)
    for i in range(1000): b[i]=rng.integers(0,61310,61310,dtype=np.uint32)
    b.flush();del b
    # Controller-only partition extraction. New model workers never open these full files.
    full=read(PKG/'paired_split/paired_cohort.npz');tv=OUT/'source_only';tv.mkdir()
    for domain in ('hsc','legacy'):
        rec=ordered_texts(PKG/'representation_audit'/f'{domain}_inputs.jsonl',full['targetid'])
        for label,category,count in [('train','T',16600),('validation','V',4257)]:
            ix=np.flatnonzero(full['category']==category);assert len(ix)==count
            np.savez_compressed(tv/f'{domain}_{label}.npz',targetid=full['targetid'][ix],z=full['z'][ix],text=np.array([rec[i] for i in ix]))
    shutil.copy2(PKG/'model_initialisation/presentation_order_seed42.npy',tv/'presentation_order_seed42.npy')
    policy=dict(created_utc=now(),stage_order='All four original source replays -> historical eight cells -> complete Stage-1 report -> four fresh seed-42 training workers -> source-only convergence report',
        n_historical=61310,new_training_counts=dict(train=16600,validation=4257),locked_P_count=61696,new_P_inference_authorized=False,seeds=[42],
        redshift_bins=BINS,bootstrap=dict(n=1000,seed=1029,shared_across_all_cells=True,sha256=sha(OUT/'bootstrap_indices.npy')),
        metric_convention='Existing repository probability-mass CE, discrete CRPS, lower median NMAD/bias, bin-centre quantiles/PIT',
        uncertainty='Paired galaxy sampling conditional on frozen models; no training-seed uncertainty',cross_domain_interpretation='interface-stress diagnostic',
        training_protocol='Unchanged prepared seed-42 configurations; 10380 updates and 166000 presentations each; no Stage-1-based tuning',
        launcher_deviation='Versioned source-only worker replaces task_1028.py train because prepared entrypoint reads locked P before slicing. Original launcher/entrypoint remain byte-identical. Same GPU locks, fresh factory, optimizer/scheduler/resume semantics; extra diagnostics and source-only prediction exports.',
        source_only_assets={str(p):sha(p) for p in tv.glob('*')})
    write(OUT/'policy.json',policy)
    files=[Path(__file__),ROOT/'notebooks/bidirectional_1029_report.py',ROOT/'aion_magnitude/bidirectional_1029_training.py']
    manifest=dict(created_utc=now(),package=str(PKG),package_manifest_sha256=EXPECTED,request_sha256=sha(ROOT/'Bidirectional_1029.md'),
        cohort_sha256=sha(OUT/'cohort.npz'),cohort_identity_sha256=sha(OUT/'cohort_identity.json'),policy_sha256=sha(OUT/'policy.json'),
        base_model=original['base_model'],models=models,protected_sha256=protected,input_sha256=input_hashes,original_inference_config_sha256=configs,
        execution_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},prepared_code_sha256=p['code_sha256'],
        source_only_assets=policy['source_only_assets'],redshift_edges=original['redshift_edges'],redshift_centers=original['redshift_centers'])
    write(OUT/'manifest.json',manifest)
    for path in files+[ROOT/'Bidirectional_1029.md']:
        dest=OUT/'code_snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    write(OUT/'audit.json',dict(created_utc=now(),prepared_manifest_exact=True,all_package_hashes_verified=True,models_verified=list(models),
        new_model_P_isolation='Separate pre-exec staging process produces T/V-only NPZ; worker opens only T/V and saved order; audit hook denies original P-containing datasets',
        output_safety='New historical directory; new model outputs are original approved future_runs paths; require own resumable state to resume',launcher_adjustment=policy['launcher_deviation']))
    print('PREPARED',OUT,flush=True)

def replay(name):
    import torch
    from filter_metadata_counterfactual import load_model
    from qwen_zero_shot_sanity import reserve_one_gpu,evaluate
    m=load_manifest();verify(m['protected_sha256']);source,method=name.split('-')
    with reserve_one_gpu() as gpu:
        assert gpu['memory_before_gib']['used']<.25
        model,tok=load_model(m,name);config=m['models'][name]['config']
        if source=='legacy':
            proof=json.loads((PKG/'legacy_checkpoint_audit'/f'{method}_validation_replay.json').read_text())
            ts=proof['input_texts'];ref=read(PKG/'legacy_checkpoint_audit'/f'{method}_validation_reference.npz');n=len(ts)
            data=dict(targetid=ref['targetid'][:n],z=ref['z_spec'][:n]);ev=evaluate(model,tok,data,ts,config)
            diffs={k:float(np.max(np.abs(ev[k].numpy()-ref[k][:n]))) for k in ('logits','pz','z_p16','z_p50','z_p84')}
            assert all(x==0 for x in diffs.values()),diffs
        else:
            ref=read(BENCH/f'original-{method}_replay_reference.npz');ts=ordered_texts(BENCH/'inputs/level0_replay.jsonl',ref['targetid']);n=len(ts)
            ev=evaluate(model,tok,dict(targetid=ref['targetid'],z=ref['z']),ts,config)
            torch.testing.assert_close(ev['logits'],torch.from_numpy(ref['logits']),rtol=0,atol=0)
            torch.testing.assert_close(ev['pz'],torch.softmax(torch.from_numpy(ref['logits']),dim=-1),rtol=0,atol=0)
            diffs={'logits':0.,'pz':0.}
            # Validate archive reuse in its exact original chunk/batch order in both domains.
            old=read(BENCH/'cohort.npz');ix=np.arange(256);small=subset(old,ix)
            for domain in ('hsc','legacy'):
                tsold=ordered_texts(BENCH/'inputs'/f'level0_{domain}.jsonl',small['targetid'])
                evold=evaluate(model,tok,dict(targetid=small['targetid'],z=small['z']),tsold,config)
                archive=read(BENCH/f'original-{method}'/f'{domain}_000000_000256.npz')
                for k in ('logits','pz','z_spec','z_p16','z_p50','z_p84','redshift_edges','redshift_centers'): np.testing.assert_array_equal(evold[k].numpy(),archive[k])
        write(OUT/'replays'/f'{name}.json',dict(created_utc=now(),n=n,source_validation_exact=True,max_abs_difference=diffs,gpu=gpu,
              archived_both_domains_first256_exact=source=='hsc',trained_head_loaded=True,checkpoint_hashes_unchanged=True))
    verify(m['protected_sha256'])

def frozen(name,domain):
    import torch
    from filter_metadata_counterfactual import load_model
    from qwen_zero_shot_sanity import reserve_one_gpu,evaluate,save_predictions
    m=load_manifest();assert all((OUT/'replays'/f'{x}.json').exists() for x in NAMES)
    verify(m['protected_sha256']);d=read(OUT/'cohort.npz');folder=OUT/'cells'/name/domain;folder.mkdir(parents=True,exist_ok=True)
    source,method=name.split('-');started=time.time()
    if source=='hsc':
        from hsc_legacy_paired_report import load_predictions
        old=read(BENCH/'cohort.npz');p=load_predictions(BENCH,f'original-{method}',domain,old)
        lookup={int(x):i for i,x in enumerate(old['targetid'])};ix=np.array([lookup[int(x)] for x in d['targetid']])
        hashes={str(path):sha(path) for path in sorted((BENCH/f'original-{method}').glob(f'{domain}_??????_??????.npz'))}
        for start in range(0,len(ix),256):
            end=min(start+256,len(ix));path=folder/f'{start:06d}_{end:06d}.npz'
            if path.exists(): continue
            q={k:(v if k.startswith('redshift_') else v[ix[start:end]]) for k,v in p.items()}
            np.testing.assert_array_equal(q['targetid'],d['targetid'][start:end]);np.testing.assert_array_equal(q['z_spec'],d['z'][start:end])
            tmp=path.with_name(path.stem+'.partial.npz');np.savez_compressed(tmp,**q);tmp.replace(path)
        write(folder/'reuse_proof.json',dict(archived_sha256=hashes,input_strings_verified_equal=True,grid_verified_equal=True,exact_live_replay='256 original archive-order objects, both domains',sliced_reordered=True))
        gpu=None
    else:
        path=OUT/'inputs'/f'{domain}.jsonl';assert sha(path)==m['input_sha256'][str(path)];ts=ordered_texts(path,d['targetid'])
        with reserve_one_gpu() as gpu:
            assert gpu['memory_before_gib']['used']<.25
            model,tok=load_model(m,name);config=m['models'][name]['config']
            lengths=[len(x) for x in tok(ts,truncation=False)['input_ids']];assert max(lengths)<=config['max_length']
            write(folder/'token_audit.json',dict(n=len(ts),min=min(lengths),max=max(lengths),truncated=0))
            for start in range(0,len(ts),256):
                end=min(start+256,len(ts));path=folder/f'{start:06d}_{end:06d}.npz'
                if path.exists():
                    old=read(path);np.testing.assert_array_equal(old['targetid'],d['targetid'][start:end]);np.testing.assert_array_equal(old['z_spec'],d['z'][start:end]);continue
                sm=dict(targetid=d['targetid'][start:end],z=d['z'][start:end],features=d[f'{domain}_features'][start:end])
                ev=evaluate(model,tok,sm,ts[start:end],config)
                for k in ('redshift_edges','redshift_centers'):np.testing.assert_array_equal(ev[k].numpy(),np.array(m[k],np.float32))
                assert torch.isfinite(ev['logits']).all() and torch.isfinite(ev['pz']).all()
                tmp=path.with_name(path.stem+'.partial.npz');save_predictions(tmp,sm,ev);tmp.replace(path)
                write(folder/'progress.json',dict(updated_utc=now(),completed=end,n=61310,seconds_this_process=time.time()-started,gpu=gpu))
    verify(m['protected_sha256'])
    write(folder/'complete.json',dict(finished_utc=now(),n=61310,reused=source=='hsc',seconds_this_process=time.time()-started,gpu=gpu,
          interpretation='native' if source==domain else 'interface-stress diagnostic',prediction_sha256={p.name:sha(p) for p in folder.glob('??????_??????.npz')}))

def gpu_free(gpu):
    # This parent imports no torch/CUDA; NVML is safe across differently visible workers.
    import importlib.util
    spec=importlib.util.spec_from_file_location('guard',ROOT/'scripts/qwen_survey_gpu_guard.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    folder=Path('/scratch/.tmp-gsm/aion_output/gpu_locks')/str(os.getuid());folder.mkdir(parents=True,exist_ok=True)
    with (folder/f'{gpu[4:]}.lock').open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        info=g.memory_info(gpu[4:]);return info['used']<.25 and info['free']>=8

def run_queue(stage,jobs):
    active={};pending=list(jobs);failed={};done=[];logs=OUT/'logs';logs.mkdir(exist_ok=True)
    while pending or active:
        for gpu in SLICES:
            if gpu in active:
                job,proc,handle=active[gpu];code=proc.poll()
                if code is not None:
                    handle.close();del active[gpu]
                    if code==0:done.append(job['name'])
                    else:failed[job['name']]=dict(exit_code=code,log=str(logs/f"{job['name']}.log"))
            if gpu not in active and pending and not failed and gpu_free(gpu):
                job=pending.pop(0);handle=(logs/f"{job['name']}.log").open('a')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PIXI_CACHE_DIR='/scratch/.tmp-gsm/pixi-cache')
                command=['pixi','run','--locked','python',str(Path(__file__).resolve()),*job['args']]
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
                active[gpu]=(job,proc,handle);write(logs/f"{job['name']}_launch.json",dict(created_utc=now(),command=command,pid=proc.pid,gpu=gpu))
        write(OUT/'status.json',dict(updated_utc=now(),stage=stage,pending=[j['name'] for j in pending],active={g:dict(job=j['name'],pid=p.pid) for g,(j,p,h) in active.items()},finished=done,failed=failed,new_P_inference=False,seeds=[42]))
        if failed and not active:raise RuntimeError(f'Workers failed; outputs retained; no subsequent stage: {failed}')
        if pending or active:time.sleep(15)

def supervise():
    with (OUT/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);load_manifest()
        run_queue('original_source_validation_replays',[dict(name='replay-'+n,args=['replay','--model',n]) for n in NAMES if not (OUT/'replays'/f'{n}.json').exists()])
        jobs=[]
        # Original Legacy cells first; archive slicing uses remaining slots briefly.
        for n in NAMES[::-1]:
            for d in ('hsc','legacy'):
                if not (OUT/'cells'/n/d/'complete.json').exists():jobs.append(dict(name=f'original-{n}-{d}',args=['frozen','--model',n,'--domain',d]))
        run_queue('historical_eight_cells',jobs)
        from bidirectional_1029_report import historical,consolidated
        if not (OUT/'stage1_complete.json').exists():historical()
        assert json.loads((OUT/'stage1_complete.json').read_text())['complete']
        # No historical metric is read here to alter the immutable training jobs.
        consolidated()
        jobs=[]
        for method in ('dora','qlora'):
            for domain in ('hsc','legacy'):
                name=f'{domain}-{method}-seed42';cfg=json.loads((PKG/'training_configs'/f'{name}.json').read_text());path=Path(cfg['output_dir'])
                if (path/'complete.json').exists():
                    assert (path/'execution_1029.json').exists(),'Foreign completion cannot satisfy this run'
                else:jobs.append(dict(name=name,args=['train','--model',f'{domain}-{method}']))
        run_queue('seed42_source_only_training',jobs)
        consolidated(final=True)
        write(OUT/'complete.json',dict(finished_utc=now(),historical_complete=True,seed42_training_complete=True,new_P_evaluated=False,seeds_43_44_launched=False))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','replay','frozen','supervise','train','report']);p.add_argument('--model',choices=NAMES);p.add_argument('--domain',choices=['hsc','legacy']);args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='replay':replay(args.model)
    elif args.action=='frozen':frozen(args.model,args.domain)
    elif args.action=='supervise':supervise()
    elif args.action=='train':
        from aion_magnitude.bidirectional_1029_training import worker
        worker(args.model)
    else:
        from bidirectional_1029_report import historical,consolidated
        historical();consolidated()

if __name__=='__main__':main()
