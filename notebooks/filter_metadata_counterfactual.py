#!/usr/bin/env python3
"""Frozen filter-metadata experiment: prepare, smoke, replay, inference, report."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'notebooks'))
import numpy as np
from hsc_legacy_paired_benchmark import read,sha,now
from aion_magnitude.filter_metadata_counterfactual import build_conditions,intervene_text,nonmetadata

BENCHMARK=ROOT/'filter-aware-runs/1027/hsc-legacy-paired-full'
SEED=1027031
BINS=[0.,.2,.4,.6,1.,4.5]


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def ordered_texts(path,expected_ids):
    records=[json.loads(line) for line in Path(path).open()]
    np.testing.assert_array_equal(np.array([int(r['targetid']) for r in records[:len(expected_ids)]],np.int64),expected_ids)
    return [r['text'] for r in records[:len(expected_ids)]]


def check_protected(m,full=False):
    protected=m['protected_sha256'] if full else m['core_protected_sha256']
    for file,expected in protected.items():
        assert sha(file)==expected,f'Protected original changed: {file}'


def token_audit(tokenizer,texts,baseline,out,key):
    flat=[];lengths=[];changed=[];differ=[];total=0
    for start in range(0,len(texts),256):
        encoded=tokenizer(texts[start:start+256],truncation=False,padding=False)['input_ids']
        for j,tokens in enumerate(encoded,start):
            a=np.asarray(tokens,np.int32);flat.append(a);lengths.append(len(a));total+=len(a)
            if baseline is None:
                changed.append(0);differ.append(0)
            else:
                b=baseline['flat'][baseline['offsets'][j]:baseline['offsets'][j+1]];n=min(len(a),len(b))
                changed.append(int((a[:n]!=b[:n]).sum())+abs(len(a)-len(b)))
                differ.append(int(not np.array_equal(a,b)))
    lengths=np.asarray(lengths,np.int32);offsets=np.r_[0,np.cumsum(lengths,dtype=np.int64)]
    data=dict(flat=np.concatenate(flat),lengths=lengths,offsets=offsets,changed_positions=np.asarray(changed,np.int32))
    path=out/'tokens'/f'{key}.npz';path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,**data)
    a=dict(n=len(texts),min=int(lengths.min()),median=float(np.median(lengths)),p95=float(np.quantile(lengths,.95)),
        max=int(lengths.max()),total_tokens=total,changed_records=int(sum(differ)),
        changed_record_fraction=float(np.mean(differ)),changed_position_count=int(sum(changed)),
        change_definition='Positionwise token-ID mismatch on shared positions plus absolute length difference; not edit distance',
        changed_position_fraction=float(sum(changed)/sum(np.maximum(lengths,baseline['lengths']))) if baseline is not None else 0.,
        token_file_sha256=sha(path))
    if baseline is not None:
        delta=lengths-baseline['lengths'];a.update(length_delta_min=int(delta.min()),length_delta_max=int(delta.max()),
            length_delta_mean=float(delta.mean()),equal_length_fraction=float((delta==0).mean()),
            length_delta_histogram={str(k):int((delta==k).sum()) for k in np.unique(delta)})
    return data,a


def write_inputs(out,name,domain,ids,texts):
    path=out/'inputs'/f'{name}_{domain}.jsonl'
    with path.open('w') as f:
        for tid,text in zip(ids,texts,strict=True):
            f.write(json.dumps(dict(targetid=str(tid),text=text))+'\n')
    return sha(path)


def prepare(args):
    from transformers import AutoTokenizer
    from aion_magnitude.filter_photometry import load_registry,DEFAULT_REGISTRY,serialize_rows,feature_columns
    from hsc_legacy_paired_report import load_predictions,object_metrics,metrics
    out=args.output_dir
    if out.exists():raise FileExistsError('Use a new experiment directory; original and completed outputs are immutable')
    out.mkdir(parents=True);(out/'inputs').mkdir();(out/'reference').mkdir()
    original=json.loads((BENCHMARK/'manifest.json').read_text());original_policy=json.loads((BENCHMARK/'evaluation_policy.json').read_text())
    original_results=json.loads((BENCHMARK/'results.json').read_text());all_data=read(BENCHMARK/'cohort.npz')
    rows=np.flatnonzero(all_data['category']=='P');assert len(rows)==61696
    np.testing.assert_array_equal(rows,np.arange(len(rows)))
    data={k:v[rows] for k,v in all_data.items()}
    for model in ('B','C'):
        assert not data[f'{model}_train'].any() and not data[f'{model}_val'].any()
    eligible=read(BENCHMARK/'eligible_pairs.npz');lookup={int(t):i for i,t in enumerate(eligible['targetid'])}
    ix=np.array([lookup[int(t)] for t in data['targetid']])
    for k in data:np.testing.assert_equal(data[k],eligible[k][ix])
    np.savez_compressed(out/'cohort.npz',**data)
    registry=load_registry(verify_curves=True);mappings=build_conditions(registry,np.unique(data['legacy_release']))
    write(out/'metadata_mappings_v1.json',mappings)
    protected={str(BENCHMARK/f):sha(BENCHMARK/f) for f in ['manifest.json','evaluation_policy.json','cohort.npz','eligible_pairs.npz','results.json','base_model_identity.json']}
    protected[str(DEFAULT_REGISTRY)]=sha(DEFAULT_REGISTRY)
    models={k:original['models'][k] for k in ('B','C')}
    for model in models.values():protected.update(model['sha256'])
    core_protected=protected.copy();reproduction={}
    for model in ('B','C'):
        reproduction[model]={}
        for domain in ('hsc','legacy'):
            pred=load_predictions(BENCHMARK,model,domain,all_data)
            pred={k:(v if k.startswith('redshift_') else v[rows]) for k,v in pred.items() if k!='features'}
            computed=metrics(object_metrics(pred),np.arange(len(rows)))
            expected=original_results['models'][model]['P']['full'][domain]
            for key in expected:np.testing.assert_allclose(computed[key],expected[key],rtol=0,atol=1e-7)
            reproduction[model][domain]=computed
            np.savez_compressed(out/'reference'/f'{model}_{domain}.npz',**pred)
            for path in sorted((BENCHMARK/model).glob(f'{domain}_??????_??????.npz')):
                protected[str(path.resolve())]=sha(path)
    write(out/'archived_baseline_reproduction.json',dict(created_utc=now(),n=len(rows),metrics=reproduction,
        exact_archived_metric_reproduction=True,scope='Full independent P only; archived predictions, not yet live replay'))
    tokenizer=AutoTokenizer.from_pretrained(original['base_model'],local_files_only=True)
    audits={};inputs={};maximum=original['max_length']
    for domain,kind,group in [('hsc','galaxiesml-spec',data['galaxy_id']),('legacy','desi-ls',data['legacy_group'])]:
        names=[n for _,v,e in feature_columns(kind) for n in (v,e)]+['MW_EBV']
        native=dict(features=data[f'{domain}_features'],group=group)
        base=serialize_rows(native,dict(feature_names=names),catalogue=kind,level=2,registry=registry)
        archived=ordered_texts(BENCHMARK/'inputs'/f'level2_{domain}.jsonl',data['targetid']);assert base==archived
        btexts=serialize_rows(native,dict(feature_names=names),catalogue=kind,level=1,registry=registry)
        assert btexts==ordered_texts(BENCHMARK/'inputs'/f'level1_{domain}.jsonl',data['targetid'])
        inputs[f'B_{domain}']=write_inputs(out,'B',domain,data['targetid'],btexts)
        _,a=token_audit(tokenizer,btexts,None,out,f'B_{domain}');audits[f'B_{domain}']=a;maximum=max(maximum,a['max'])
        protected[str((BENCHMARK/'inputs'/f'level1_{domain}.jsonl').resolve())]=sha(BENCHMARK/'inputs'/f'level1_{domain}.jsonl')
        protected[str((BENCHMARK/'inputs'/f'level2_{domain}.jsonl').resolve())]=sha(BENCHMARK/'inputs'/f'level2_{domain}.jsonl')
        baseline_tokens=None
        for name,condition in mappings['conditions'].items():
            texts=[intervene_text(text,condition['mapping']) for text in base]
            assert all(nonmetadata(a)==nonmetadata(b) for a,b in zip(texts,base))
            if name=='correct':assert texts==base
            key=f'{name}_{domain}';inputs[key]=write_inputs(out,name,domain,data['targetid'],texts)
            tokens,a=token_audit(tokenizer,texts,baseline_tokens,out,key);audits[key]=a;maximum=max(maximum,a['max'])
            if name=='correct':baseline_tokens=tokens
            print('PREPARED',key,'max_tokens',a['max'],'length_delta',a.get('length_delta_mean',0),flush=True)
    write(out/'tokenization_audit.json',dict(created_utc=now(),common_max_length=maximum,conditions=audits,
        truncated_records=0,photometry_names_missingness_ebv_byte_identical=True,
        placebo_length_matching={domain:audits[f'placebo-length-structure_{domain}']['equal_length_fraction'] for domain in ('hsc','legacy')}))
    smoke=np.array(sorted(sorted(range(len(rows)),key=lambda i:hashlib.sha256(f'{SEED}:{data["galaxy_id"][i]}'.encode()).digest())[:32]))
    np.save(out/'smoke_indices.npy',smoke)
    bootstrap=np.lib.format.open_memmap(out/'bootstrap_indices.npy',mode='w+',dtype=np.uint32,shape=(1000,len(rows)))
    rng=np.random.default_rng(SEED)
    for i in range(1000):bootstrap[i]=rng.integers(0,len(rows),size=len(rows),dtype=np.uint32)
    bootstrap.flush();del bootstrap
    policy=dict(version='metadata_counterfactual_policy_v1',created_utc=now(),request=str(ROOT/'Filter-Meta_Counter.md'),
        benchmark=str(BENCHMARK.resolve()),n=len(rows),category='P only; no V or T',conditions=list(mappings['conditions']),
        mappings_sha256=sha(out/'metadata_mappings_v1.json'),cohort_sha256=sha(out/'cohort.npz'),
        identifiers_sha256=hashlib.sha256(data['targetid'].tobytes()).hexdigest(),galaxy_order='Identical to original P order; no resampling or membership changes',
        primary_metric='CE of probability mass in identical 300 bins over [0,4.5]',
        metrics=['CE','CRPS','NMAD','outlier |delta|>0.15','mean bias','median bias','coverage68','PIT','mean absolute normalized residual'],
        redshift_edges=BINS,redshift_counts=np.histogram(data['z'],bins=BINS)[0].tolist(),
        bootstrap=dict(replicates=1000,seed=SEED,file_sha256=sha(out/'bootstrap_indices.npy'),
            rule='Same saved galaxy indices reused for every model/condition/domain and principal contrast; percentile 95%; conditional on fixed checkpoints and this cohort, no training-seed uncertainty'),
        smoke=dict(n=32,rule=f'Lowest SHA256({SEED}:galaxy_id), restored to original order',indices_sha256=sha(out/'smoke_indices.npy'),
            labels_used=False,purpose='Implementation and computational feasibility only; no counterfactual redshift metrics scored or selected'),
        baseline_gate='Archived full-P B/C metrics reproduced first; live first-256 replay for B/C both domains before smoke. Full B/C live replay, exact logits/PDFs, must finish before any full counterfactual inference.',
        perturbation_selection='Every condition fixed from registry/seeds before new counterfactual predictions; no target-label construction or checkpoint/condition selection',
        designs=dict(both_domains='Each condition applied to both real observational representations, with globally consistent physical-profile mappings',
                     target_only='Derived contrast using perturbed Legacy and unchanged correct-C HSC; no separate inference needed; labelled target-side effect'),
        max_length=maximum,token_audit_sha256=sha(out/'tokenization_audit.json'),
        resource_plan='All 12 counterfactuals in both domains on all 61,696 P galaxies, after B/C full replay; three idle MIG slices; durable chunked background queue',
        inference_weights_updated=False,calibration=False,training=False)
    write(out/'evaluation_policy_v1.json',policy)
    files=[Path(__file__),ROOT/'aion_magnitude/filter_metadata_counterfactual.py',ROOT/'notebooks/report_filter_metadata_counterfactual.py',
           ROOT/'notebooks/supervise_filter_metadata_counterfactual.py',ROOT/'aion_magnitude/filter_photometry.py',ROOT/'aion_magnitude/metrics.py',
           ROOT/'notebooks/hsc_legacy_paired_report.py',ROOT/'notebooks/qwen_zero_shot_sanity.py']
    manifest=dict(created_utc=now(),benchmark=str(BENCHMARK.resolve()),models=models,base_model=original['base_model'],
        policy_sha256=sha(out/'evaluation_policy_v1.json'),cohort_sha256=sha(out/'cohort.npz'),inputs_sha256=inputs,
        metadata_mappings_sha256=sha(out/'metadata_mappings_v1.json'),protected_sha256=protected,core_protected_sha256=core_protected,
        max_length=maximum,redshift_edges=original['redshift_edges'],redshift_centers=original['redshift_centers'],
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        archived_original_policy_sha256=sha(BENCHMARK/'evaluation_policy.json'),reference_prediction_sha256={p.name:sha(p) for p in (out/'reference').glob('*.npz')})
    write(out/'manifest.json',manifest)
    snap=out/'code_snapshot'
    for path in files+[ROOT/'Filter-Meta_Counter.md']:
        dest=snap/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    check_protected(manifest,full=True)
    print('POLICY FROZEN',manifest['policy_sha256'],'n',len(rows),'conditions',len(mappings['conditions']),'max_length',maximum,flush=True)


def load_model(m,which):
    import torch
    from peft import PeftModel,prepare_model_for_kbit_training,get_peft_model_state_dict
    from safetensors.torch import load_file
    from aion_magnitude.FM_Qwen import load_frozen_qwen
    from aion_magnitude.qwen_posttraining import QwenPhotoZModel,qwen_hidden_size
    info=m['models'][which];config=info['config'];selected=Path(info['directory'])
    base,tokenizer=load_frozen_qwen(m['base_model'],device='cuda',load_in_4bit=True,torch_dtype='bf16',local_files_only=True)
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
    qwen=PeftModel.from_pretrained(base,selected/'adapter',is_trainable=False)
    expected=load_file(str(selected/'adapter/adapter_model.safetensors'));actual=get_peft_model_state_dict(qwen)
    assert actual.keys()==expected.keys()
    for key,tensor in actual.items():torch.testing.assert_close(tensor.cpu(),expected[key],rtol=0,atol=0,check_dtype=False)
    model=QwenPhotoZModel(qwen,hidden_size=qwen_hidden_size(qwen),n_z_bins=config['n_z_bins'],head_hidden_dim=config['head_hidden_dim'],pooling=config['pooling']).to('cuda')
    state=torch.load(selected/'photoz_head.pt',map_location='cpu',weights_only=True);model.photoz_head.load_state_dict(state,strict=True)
    for key,tensor in model.photoz_head.state_dict().items():torch.testing.assert_close(tensor.cpu(),state[key],rtol=0,atol=0)
    model.requires_grad_(False);model.eval();assert not any(p.requires_grad for p in model.parameters())
    return model,tokenizer


def validate_run(out):
    m=json.loads((out/'manifest.json').read_text())
    assert sha(out/'evaluation_policy_v1.json')==m['policy_sha256'] and sha(out/'cohort.npz')==m['cohort_sha256']
    assert sha(out/'metadata_mappings_v1.json')==m['metadata_mappings_sha256']
    for file,expected in m['code_sha256'].items():assert sha(ROOT/file)==expected,f'Prepared code changed: {file}'
    check_protected(m)
    return m,read(out/'cohort.npz')


def exact_replay(ev,ref,indices):
    # Exact means exact, rather than a tolerance selected after examining counterfactuals.
    for key in ['logits','pz','z_spec','z_p16','z_p50','z_p84']:
        np.testing.assert_array_equal(ev[key].numpy(),ref[key][indices],err_msg=f'Archived replay mismatch: {key}')
    for key in ['redshift_edges','redshift_centers']:np.testing.assert_array_equal(ev[key].numpy(),ref[key])


def smoke(args):
    import torch
    from qwen_zero_shot_sanity import reserve_one_gpu,evaluate
    out=args.output_dir;m,d=validate_run(out);which=args.model;config=m['models'][which]['config'].copy();config['max_length']=m['max_length']
    record=dict(started_utc=now(),model=which,baseline_replays={},counterfactuals={},labels_used_for_counterfactual_smoke=False)
    with reserve_one_gpu() as gpu:
        assert gpu['memory_before_gib']['used']<.25
        model,tokenizer=load_model(m,which)
        for domain in ('hsc','legacy'):
            name='B' if which=='B' else 'correct';key=f'{name}_{domain}';path=out/'inputs'/f'{key}.jsonl'
            assert sha(path)==m['inputs_sha256'][key];texts=ordered_texts(path,d['targetid'])
            ix=np.arange(256);ref=read(out/'reference'/f'{which}_{domain}.npz')
            ev=evaluate(model,tokenizer,dict(targetid=d['targetid'][ix],z=d['z'][ix]),texts[:256],config);exact_replay(ev,ref,ix)
            record['baseline_replays'][domain]=dict(n=256,logits_and_pdfs_exact=True,max_abs_logit_difference=0.)
        write(out/'smoke'/f'{which}_baseline.json',record)
        print('LIVE BASELINE EXACT',which,'256 per domain',flush=True)
        if which=='C':
            # B's first-256 gate may be on another slice: wait without starting perturbations.
            deadline=time.time()+600
            while not (out/'smoke/B.json').exists():
                if time.time()>deadline:raise TimeoutError('B baseline gate did not finish; no perturbations run')
                time.sleep(2)
            ix=np.load(out/'smoke_indices.npy');names=json.loads((out/'metadata_mappings_v1.json').read_text())['conditions']
            for name in names:
                if name=='correct':continue
                for domain in ('hsc','legacy'):
                    key=f'{name}_{domain}';path=out/'inputs'/f'{key}.jsonl';assert sha(path)==m['inputs_sha256'][key]
                    texts=ordered_texts(path,d['targetid']);start=time.time()
                    ev=evaluate(model,tokenizer,dict(targetid=d['targetid'][ix],z=np.zeros(len(ix),np.float32)),[texts[i] for i in ix],config)
                    assert torch.isfinite(ev['logits']).all() and torch.isfinite(ev['pz']).all()
                    torch.testing.assert_close(ev['pz'].sum(1),torch.ones(len(ix)),rtol=1e-5,atol=1e-6)
                    folder=out/'smoke/predictions';folder.mkdir(exist_ok=True)
                    np.savez_compressed(folder/f'{key}.npz',targetid=d['targetid'][ix],logits=ev['logits'].numpy(),pz=ev['pz'].numpy(),redshift_edges=ev['redshift_edges'].numpy())
                    record['counterfactuals'][key]=dict(n=len(ix),seconds=time.time()-start,pdf_finite_normalized=True,scored_against_real_redshifts=False)
                    print('SMOKE OK',key,len(ix),flush=True)
        check_protected(m);record['finished_utc']=now();record['gpu']=gpu;write(out/'smoke'/f'{which}.json',record)


def infer(args):
    from qwen_zero_shot_sanity import reserve_one_gpu,evaluate,save_predictions
    out=args.output_dir;m,d=validate_run(out);which=args.model;name=args.condition
    conditions=json.loads((out/'metadata_mappings_v1.json').read_text())['conditions']
    assert (which=='B' and name=='B') or (which=='C' and name in conditions),'B is an external Level-1 reference; interventions use C only'
    if name not in ('B','correct'):assert (out/'gates/full_baseline_replay.json').exists(),'Full baseline replay is a mandatory gate'
    domains=['hsc','legacy'] if args.domain=='both' else [args.domain]
    job=f'{name}-{args.domain}';status=out/'jobs'/job;status.mkdir(parents=True,exist_ok=True)
    if (status/'complete.json').exists():return
    config=m['models'][which]['config'].copy();config['max_length']=m['max_length'];start_time=time.time()
    with reserve_one_gpu() as gpu:
        assert gpu['memory_before_gib']['used']<.25
        write(status/'running.json',dict(started_utc=now(),pid=os.getpid(),model=which,condition=name,domains=domains,gpu=gpu))
        model,tokenizer=load_model(m,which);hashes={}
        for domain in domains:
            path=out/'inputs'/f'{name}_{domain}.jsonl';assert sha(path)==m['inputs_sha256'][f'{name}_{domain}']
            texts=ordered_texts(path,d['targetid']);dest=out/'predictions'/name/domain;dest.mkdir(parents=True,exist_ok=True)
            ref=read(out/'reference'/f'{which}_{domain}.npz') if name in ('B','correct') else None
            for start in range(0,len(texts),256):
                end=min(start+256,len(texts));file=dest/f'{start:06d}_{end:06d}.npz'
                if file.exists():
                    saved=read(file);np.testing.assert_array_equal(saved['targetid'],d['targetid'][start:end]);hashes[str(file)]=sha(file);continue
                small=dict(targetid=d['targetid'][start:end],z=d['z'][start:end],features=d[f'{domain}_features'][start:end])
                ev=evaluate(model,tokenizer,small,texts[start:end],config)
                assert np.isfinite(ev['logits'].numpy()).all() and np.isfinite(ev['pz'].numpy()).all(),'Nonfinite prediction; preserve completed chunks and stop'
                if ref is not None:exact_replay(ev,ref,np.arange(start,end))
                tmp=file.with_name(file.stem+'.partial.npz');save_predictions(tmp,small,ev);tmp.replace(file);hashes[str(file)]=sha(file)
                write(status/'status.json',dict(updated_utc=now(),model=which,condition=name,domain=domain,completed=end,total=len(texts),seconds=time.time()-start_time,baseline_exact=ref is not None))
                print('PROGRESS',name,domain,end,'/',len(texts),f'{time.time()-start_time:.1f}s',flush=True)
        check_protected(m)
        write(status/'complete.json',dict(finished_utc=now(),seconds=time.time()-start_time,model=which,condition=name,domains=domains,
            n=len(texts),baseline_replay_exact=ref is not None,checkpoint_hashes_unchanged=True,prediction_sha256=hashes))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','smoke','infer','report']);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--model',choices=['B','C'],default='C');p.add_argument('--condition',default='correct');p.add_argument('--domain',choices=['hsc','legacy','both'],default='both')
    args=p.parse_args()
    if args.stage=='prepare':prepare(args)
    elif args.stage=='smoke':smoke(args)
    elif args.stage=='infer':infer(args)
    else:
        from report_filter_metadata_counterfactual import report
        report(args.output_dir)


if __name__=='__main__':main()
