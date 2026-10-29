#!/usr/bin/env python3
"""Small, independent Stage-1 physics probes and frozen-representation benchmark."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'notebooks'),str(ROOT)]
import numpy as np
import diagnose_hsc_clauds_stage0 as d
import stage1_sed_forward as f
OUT=ROOT/'reports/stage1/1027'
SEED=102701
QUERY='\nInfer redshift from spectral shape:'
N_BINS=90


def stable_seed(*args):
    return int.from_bytes(hashlib.sha256(':'.join(map(str,args)).encode()).digest()[:4],'little')


def require_stage05():
    p=ROOT/'reports/stage05/1027';review=json.loads((p/'correctness_review.json').read_text());assert review['stage05_complete'] and review['no_blocking_correctness_problem_for_stage1']
    return dict(report_sha256=d.sha(p/'report.md'),review_sha256=d.sha(p/'correctness_review.json'),completed_utc=json.loads((p/'summary.json').read_text())['completed_utc'])


def serialize(record,flux,sigma,curves):
    # No label, SED family/identity, intrinsic nuisance, survey name or seed.
    normalized,errors,scale=f.normalize_observation(flux,sigma)
    slots=np.random.default_rng(record['order_seed']).choice(11,len(record['bands']),replace=False)
    order=np.argsort(slots);features=np.zeros((11,9));lines=['Galaxy relative AB fnu photometry; amplitude marginalized; response wavelengths in Angstrom.']
    for j in order:
        b=record['metadata_bands'][j];curve=curves[b];valid=np.isfinite(normalized[j]) and np.isfinite(errors[j])
        number=lambda x:f'{x:+.6e}' if np.isfinite(x) else 'NA'
        q=curve['quantiles'];lines.append(f'band={b} pivot={curve["pivot"]:.1f} width={curve["width"]:.1f} q10={q[0]:.1f} q50={q[1]:.1f} q90={q[2]:.1f} flux={number(normalized[j])} sigma={number(errors[j])} valid={int(valid)}')
        features[slots[j]]=[np.arcsinh(normalized[j]) if valid else 0.,np.log(errors[j]) if valid else 0.,curve['pivot']/1e4,curve['width']/1e4,*list(q/1e4),float(valid),1.]
    return '\n'.join(lines)+QUERY,features,normalized,errors,scale


def reasoning_cases():
    rng=np.random.default_rng(SEED);cases=[]
    def add(kind,question,options,correct,extra):
        permutation=rng.permutation(4);opts=[options[i] for i in permutation];answer=int(np.flatnonzero(permutation==correct)[0]);cases.append(dict(id=len(cases),kind=kind,question=question,options=opts,correct_index=answer,**extra))
    for i,z in enumerate([.1,.25,.4,.6,.9,1.3,2.,3.]):
        rest=4000.;obs=rest*(1+z);values=[z,z+.4,max(0,z-.2),z+1.]
        add('explicit_break_arithmetic',f'A spectral break is at {rest:.0f} Angstrom in the rest frame and {obs:.0f} Angstrom observed. What is its cosmological redshift?', [f'{v:.2f}' for v in values],0,dict(rest=rest,observed=obs,truth=z))
        add('generic_arithmetic',f'Compute {obs:.0f} divided by {rest:.0f}, minus one.',[f'{v:.2f}' for v in values],0,dict(rest=rest,observed=obs,truth=z))
    for i,(name,rest) in enumerate([('hydrogen Balmer series limit',3646.),('hydrogen Lyman-alpha',1216.),('hydrogen Lyman series limit',912.),('hydrogen H-alpha',6563.)]*2):
        z=[.25,.5,1.,2.,.75,1.5,2.5,3.][i];obs=rest*(1+z)
        add('named_feature_without_rest_value',f'The {name} feature is observed at {obs:.1f} Angstrom. What redshift does this imply?',[f'{v:.2f}' for v in [z,z+.5,max(0,z-.2),z+1.]],0,dict(feature=name,rest=rest,observed=obs,truth=z))
    for obs in [5000,5500,6000,6500,7000,7500,8000,8500]:
        add('break_ambiguity',f'Photometry shows one break near {obs} Angstrom, with no further colour or line information. Can this alone distinguish a rest-frame 4000 Angstrom break from a 1216 Angstrom Lyman break?', ['It must be the 4000 Angstrom break','It must be the Lyman break','Both interpretations are possible','The redshift must be zero'],2,dict(observed=obs))
    for i in range(8):
        z=[.1,.3,.5,.8,1.,1.5,2.,3.][i];consistent=i%2==0;obs=4000*(1+z)+(0 if consistent else 400)
        add('physical_consistency',f'A 4000 Angstrom rest-frame break is reported at {obs:.0f} Angstrom observed, with redshift z={z:.2f}. Is this wavelength/redshift relation consistent?',['Yes','No','Only the flux can determine this','Only the filter width can determine this'],0 if consistent else 1,dict(rest=4000,observed=obs,supplied_z=z))
    for i in range(8):
        flatnu=i%2==0;kind='constant f_nu' if flatnu else 'constant f_lambda'
        question=f'A spectrum has {kind} across two photon-counting filters. Their pivots are 4800 and 9000 Angstrom. Compare their band-averaged AB f_nu after full response normalization.'
        add('cross_filter_flux_rule',question,['The redder filter has larger f_nu','The bluer filter has larger f_nu','Both f_nu values are equal','The redshift is necessarily 1'],2 if flatnu else 0,dict(flat_fnu=flatnu))
    return cases


def prepare(out):
    from transformers import AutoTokenizer
    from aion_magnitude.filter_photometry import load_registry
    gate=require_stage05();assert not (out/'manifest.json').exists();out.mkdir(parents=True,exist_ok=True)
    test_log=(out/'forward-tests.log').read_text();assert 'passed' in test_log and 'failed' not in test_log
    load_registry(verify_curves=True);curves=f.responses(ROOT);rng=np.random.default_rng(SEED)
    sedparams=[];spectra=[];records=[];rawflux=[];rawsigma=[];cleanflux=[];numeric=[];normflux=[];normsigma=[];texts=[];scales=[]
    empirical=['CWW_E_ext.sed','CWW_Sbc_ext.sed','CWW_Scd_ext.sed','CWW_Im_ext.sed'];families=['break4000','lyman_continuum','weak_break','empirical']
    for split,n in [('train',128),('validation',24),('test',32),('family_holdout',24)]:
        for i in range(n):
            family='double_break_emission' if split=='family_holdout' else families[i%4]
            param=dict(sed_id=len(sedparams),split=split,family=family,slope=float(rng.uniform(-1.1,1.1)),av=float(rng.uniform(0,.6)),break_strength=float(rng.uniform(1,1.08) if family=='weak_break' else rng.uniform(1.25,2.5)),equivalent_width=float(rng.uniform(20,100)),log_lnu=float(rng.uniform(27,30)),template=empirical[int(rng.integers(4))])
            shape=f.sed_shape(param,template_dir=out/'templates');param['spectrum_sha256']=hashlib.sha256(shape.tobytes()).hexdigest();sedparams.append(param);spectra.append(shape)
            for realization in range(2):
                z=float(rng.uniform(.05,4.1));sn=float(np.exp(rng.uniform(np.log(10),np.log(100))));latent_id=f'{param["sed_id"]}:{realization}'
                systems=['HSC','Legacy'] if split in ('train','validation') else list(f.SYSTEMS)
                for system in systems:
                    bands=list(f.SYSTEMS[system]);clean=f.observe_clean(param,z,bands,curves,out/'templates')
                    flux,sigma=f.add_noise(clean,sn,np.random.default_rng(stable_seed(latent_id,system)))
                    variants=['correct']
                    if split=='test' and system=='CLAUDS':variants+=['wrong_metadata','missing_bands','noisy']
                    if split=='test' and system=='HSC':variants+=['redshift_plus03']
                    for variant in variants:
                        zz=z;cc=clean.copy();ff=flux.copy();ee=sigma.copy();metadata=bands.copy()
                        if variant=='wrong_metadata':metadata=list(np.roll(bands,1))
                        if variant=='missing_bands':
                            miss=np.random.default_rng(stable_seed(latent_id,'missing')).choice(len(bands),2,replace=False);ff[miss]=np.nan;ee[miss]=np.nan
                        if variant=='noisy':ff,ee=f.add_noise(clean,sn/5,np.random.default_rng(stable_seed(latent_id,'noisy')))
                        if variant=='redshift_plus03':zz=z+.3;cc=f.observe_clean(param,zz,bands,curves,out/'templates');ff,ee=f.add_noise(cc,sn,np.random.default_rng(stable_seed(latent_id,system,variant)))
                        rec=dict(index=len(records),sed_id=param['sed_id'],latent_id=latent_id,split=split,family=family,z=zz,sn_nuisance=sn/(5 if variant=='noisy' else 1),system=system,variant=variant,bands=bands,metadata_bands=metadata,order_seed=stable_seed(latent_id,system,'order'),log_lnu=param['log_lnu'])
                        text,x,nf,ne,scale=serialize(rec,ff,ee,curves);records.append(rec);texts.append(text);numeric.append(x);scales.append(scale)
                        for dest,values in [(rawflux,ff),(rawsigma,ee),(cleanflux,cc),(normflux,nf),(normsigma,ne)]:dest.append(np.pad(values,(0,11-len(values)),constant_values=np.nan))
    assert len({p['spectrum_sha256'] for p in sedparams})==len(sedparams)
    splits={s:{p['sed_id'] for p in sedparams if p['split']==s} for s in ('train','validation','test','family_holdout')}
    assert all(not (v&w) for i,v in enumerate(splits.values()) for w in list(splits.values())[i+1:])
    assert all(r['system'] in ('HSC','Legacy') for r in records if r['split']=='train')
    np.savez_compressed(out/'spectra.npz',rest_wavelength_angstrom=f.REST_GRID,emergent_lnu_shape=np.array(spectra))
    np.savez_compressed(out/'photometry.npz',raw_flux_nanomaggy=np.array(rawflux),raw_sigma_nanomaggy=np.array(rawsigma),clean_flux_nanomaggy=np.array(cleanflux),normalized_flux=np.array(normflux),normalized_sigma=np.array(normsigma),numeric=np.array(numeric),normalization_scale=np.array(scales),z=np.array([r['z'] for r in records],np.float32),sed_id=np.array([r['sed_id'] for r in records]),split=np.array([r['split'] for r in records]))
    d.write(out/'sed_parameters.json',sedparams)
    with (out/'records.jsonl').open('w') as file:
        for r in records:file.write(json.dumps(r)+'\n')
    with (out/'inputs.jsonl').open('w') as file:
        for i,text in enumerate(texts):file.write(json.dumps(dict(index=i,text=text))+'\n')
    cases=reasoning_cases();d.write(out/'reasoning_cases.json',cases)
    tokenizer=AutoTokenizer.from_pretrained('/arc/home/gsm/hf_models/Qwen3.5-4B-Base',local_files_only=True);enc=tokenizer(texts,truncation=False,padding=False);lengths=list(map(len,enc['input_ids']));assert max(lengths)<=1024
    d.write(out/'input_examples.json',[dict(record_index=i,text=texts[i],input_ids=enc['input_ids'][i],last_token=tokenizer.convert_ids_to_tokens(enc['input_ids'][i][-1])) for i in [0,128,512,608,len(texts)-1]])
    for name,curve in curves.items():np.savez_compressed(out/f'response_{name}.npz',**curve)
    config=dict(seed=SEED,train_seds=128,validation_seds=24,test_seds=32,family_holdout_seds=24,redshifts_per_sed=2,redshift_uniform=[.05,4.1],normalization='RMS of observed valid fnu, same scale applied to errors; raw physical fluxes also saved',
        source_systems=['HSC','Legacy'],held_out_systems=['CLAUDS','new_filters'],source_labels=512,z_min=0,z_max=4.5,n_z_bins=N_BINS,probe_seeds=[11,23,37],probe_updates=400,probe_batch_size=64,probe_lr=.002,probe_weight_decay=.0001,head_hidden_dim=32,numerical_set_mlp_hidden=680,primary_pooling='last',secondary_pooling='mean',
        random_architecture_seed=1701,max_tokens=1024,spectrum_family_test='double_break_emission entirely absent from source supervised training',noise='independent Gaussian sky-equivalent error at redshift-independent nuisance RMS S/N; extra noisy reobservation regenerated from clean flux',
        igm='simplified mean Lyman-series opacity screen plus ionizing suppression; not a calibrated full IGM model',dust='controlled power-law optical depth, not a fitted empirical law',absolute_luminosity='Independent log Lnu(5500 A) uniform 27–30 before observation; all supervised inputs remove amplitude',
        no_real_target_catalogue=True,no_clauds_labels=True,clauds_success_claimed=False,trainable_backbone=False,adapter_training=False,reasoning='Four-way conditional next-token scoring, no-example versus fixed two-example conditions; base and Instruct kept separate')
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),stage05_gate=gate,config=config,records=len(records),sed_count=len(sedparams),split_counts={s:sum(r['split']==s for r in records) for s in splits},max_token_length=max(lengths),median_token_length=float(np.median(lengths)),registry_sha256=d.sha(ROOT/'configs/filter_registry_v1.json'),
        source_code_sha256={p.name:d.sha(p) for p in [Path(__file__),ROOT/'scripts/stage1_sed_forward.py']},templates=json.loads((out/'templates/manifest.json').read_text()),prepared_sha256={str(p.relative_to(out)):d.sha(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.log'})
    d.write(out/'manifest.json',manifest);print(json.dumps(dict(records=len(records),sed_count=len(sedparams),split_counts=manifest['split_counts'],max_tokens=max(lengths))),flush=True)


def check(out):
    m=json.loads((out/'manifest.json').read_text())
    for p,v in m['prepared_sha256'].items():assert d.sha(out/p)==v,p
    return m


def reasoning_prompts(cases,tokenizer,instruct):
    result=[];examples='Question: A rest wavelength of 2000 Angstrom is observed at 3000 Angstrom. Redshift?\nA: 0.50; B: 1.00; C: 2.00; D: 3.00\nAnswer: A\n\nQuestion: Compute 7200/3600 - 1.\nA: 2.00; B: 1.00; C: 0.00; D: 3.00\nAnswer: B\n\n'
    for few in (False,True):
        for case in cases:
            question='Question: '+case['question']+'\n'+'; '.join(f'{chr(65+j)}: {v}' for j,v in enumerate(case['options']))+'\nAnswer:'
            prefix=examples if few else ''
            text=prefix+question
            if instruct:text=tokenizer.apply_chat_template([dict(role='user',content='Choose the correct option letter.\n'+text)],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:'
            result.append(dict(case_id=case['id'],few_examples=few,text=text))
    return result


def load_backbone(model_name,out):
    from aion_magnitude.FM_Qwen import load_frozen_qwen
    if model_name=='random':
        import torch
        from transformers import AutoModel,AutoTokenizer,BitsAndBytesConfig
        path=Path('/scratch/.tmp-gsm/stage1-random-text-1701');assert (path/'config.json').exists(),'Initialize random control first'
        q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
        model=AutoModel.from_pretrained(path,local_files_only=True,quantization_config=q,device_map={'':0},dtype=torch.bfloat16)
        tok=AutoTokenizer.from_pretrained('/arc/home/gsm/hf_models/Qwen3.5-4B-Base',local_files_only=True)
        if tok.pad_token_id is None:tok.pad_token=tok.eos_token
        model.eval();model.requires_grad_(False);return model,tok
    return load_frozen_qwen('/arc/home/gsm/hf_models/'+('Qwen3.5-4B' if model_name=='instruct' else 'Qwen3.5-4B-Base'),device='cuda',load_in_4bit=True,torch_dtype='bf16',local_files_only=True)


def init_random(out):
    import torch
    from transformers import AutoConfig,AutoModel
    m=check(out);path=Path('/scratch/.tmp-gsm/stage1-random-text-1701')
    if path.exists():
        metadata=json.loads((path/'initialization.json').read_text());assert metadata['seed']==m['config']['random_architecture_seed']
        for name,digest in metadata['files'].items():assert d.sha(path/name)==digest,name
        d.write(out/'random_architecture.json',metadata);return
    torch.manual_seed(m['config']['random_architecture_seed']);config=AutoConfig.from_pretrained('/arc/home/gsm/hf_models/Qwen3.5-4B-Base',local_files_only=True).text_config
    model=AutoModel.from_config(config,dtype=torch.bfloat16);model.requires_grad_(False);model.eval();path.mkdir(parents=True);model.save_pretrained(path,max_shard_size='2GB')
    d.write(path/'initialization.json',dict(seed=m['config']['random_architecture_seed'],class_name=type(model).__name__,parameters=sum(p.numel() for p in model.parameters()),frozen=True,text_architecture='same full text stack, vision branch unused in both conditions',quantization='reloaded NF4 double-quantized, matching pretrained frozen representation',files={p.name:d.sha(p) for p in path.glob('*.safetensors')}))
    d.write(out/'random_architecture.json',json.loads((path/'initialization.json').read_text()));print(path,flush=True)


def reason(out,model_name):
    import torch
    import qwen_zero_shot_sanity as pilot
    from aion_magnitude.FM_Qwen import pool_qwen_hidden_states
    check(out);folder=out/f'reasoning-{model_name}';folder.mkdir(exist_ok=True)
    cases=json.loads((out/'reasoning_cases.json').read_text())
    with pilot.reserve_one_gpu() as gpu:
        model,tok=load_backbone(model_name,out);prompts=reasoning_prompts(cases,tok,model_name=='instruct');letters=[tok.encode(' '+c,add_special_tokens=False) for c in 'ABCD'];assert all(len(x)==1 for x in letters);ids=torch.tensor([x[0] for x in letters],device='cuda');rows=[]
        with torch.inference_mode():
            for start in range(0,len(prompts),4):
                batch=prompts[start:start+4];enc=tok([x['text'] for x in batch],padding=True,truncation=False,return_tensors='pt');inputs={k:v.to('cuda') for k,v in enc.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16):hidden=model(**inputs,use_cache=False,return_dict=True).last_hidden_state
                emb=pool_qwen_hidden_states(hidden,inputs['attention_mask'],pooling='last');weights=model.get_input_embeddings().weight[ids]
                logits=(emb.float()@weights.float().T).cpu().numpy()
                for j,p in enumerate(batch):
                    case=cases[p['case_id']];prediction=int(logits[j].argmax());rows.append(dict(model=model_name,case_id=case['id'],kind=case['kind'],few_examples=p['few_examples'],prediction_index=prediction,correct_index=case['correct_index'],correct=prediction==case['correct_index'],option_logits=logits[j].tolist(),text=p['text'],input_ids=enc['input_ids'][j].tolist(),attention_mask=enc['attention_mask'][j].tolist()))
        d.write(folder/'results.json',rows)
        d.table(folder/'summary.csv',[dict(model=model_name,kind=kind,few_examples=few,n=sum(r['kind']==kind and r['few_examples']==few for r in rows),accuracy=float(np.mean([r['correct'] for r in rows if r['kind']==kind and r['few_examples']==few]))) for kind in sorted({r['kind'] for r in rows}) for few in (False,True)])
        d.write(folder/'status.json',dict(phase='complete',gpu=gpu,backbone_frozen=True,candidate_token_ids=ids.cpu().tolist(),scoring='Tied LM unembedding; conditional next-token logits for four single-token space-prefixed option letters; no generation or photo-z fine-tuning'))
        print(folder,flush=True)


def extract(out,model_name):
    import torch
    import qwen_zero_shot_sanity as pilot
    from aion_magnitude.FM_Qwen import pool_qwen_hidden_states
    m=check(out);assert json.loads((out/'reasoning-base/status.json').read_text())['phase']=='complete','Run inference-only reasoning first'
    path=out/f'embeddings-{model_name}.npz'
    if path.exists():raise FileExistsError(path)
    texts=[json.loads(l)['text'] for l in (out/'inputs.jsonl').read_text().splitlines()];last=[];mean=[];lengths=[]
    with pilot.reserve_one_gpu() as gpu:
        model,tok=load_backbone(model_name,out);assert not any(p.requires_grad for p in model.parameters())
        with torch.inference_mode():
            for start in range(0,len(texts),4):
                enc=tok(texts[start:start+4],padding=True,truncation=False,return_tensors='pt');assert enc['input_ids'].shape[1]<=m['config']['max_tokens'];inputs={k:v.to('cuda') for k,v in enc.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16):hidden=model(**inputs,use_cache=False,return_dict=True).last_hidden_state
                last.append(pool_qwen_hidden_states(hidden,inputs['attention_mask'],pooling='last').float().cpu().numpy());mean.append(pool_qwen_hidden_states(hidden,inputs['attention_mask'],pooling='mean').float().cpu().numpy());lengths.extend(enc['attention_mask'].sum(dim=1).tolist())
                if start%128==0:print(model_name,start,'/',len(texts),flush=True)
        np.savez_compressed(path,last=np.concatenate(last),mean=np.concatenate(mean),lengths=lengths)
        d.write(out/f'extraction-{model_name}.json',dict(phase='complete',gpu=gpu,backbone_frozen=True,rows=len(texts),dimensions=last[0].shape[1],quantization='NF4 double quantization, BF16 compute',sha256=d.sha(path)))


def prediction_metrics(logits,z):
    import torch
    from aion_magnitude.metrics import predict_photoz_from_logits
    from aion_magnitude.utils import make_redshift_grid
    edges,centers=make_redshift_grid(0,4.5,N_BINS);l=torch.from_numpy(np.asarray(logits,np.float32));pred=predict_photoz_from_logits(l,centers=centers)
    return dict(logits=l.numpy(),z_spec=np.asarray(z,np.float32),redshift_edges=edges.numpy(),redshift_centers=centers.numpy(),**{k:v.numpy() for k,v in pred.items() if k in ('pz','z_p16','z_p50','z_p84')})


def split_masks(records):
    masks={}
    for split in ('validation','test','family_holdout'):
        for system in f.SYSTEMS:
            for variant in sorted({r['variant'] for r in records if r['split']==split and r['system']==system}):
                masks[f'{split}_{system}_{variant}']=np.array([r['split']==split and r['system']==system and r['variant']==variant for r in records])
    return masks


def fit(out):
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import ExtraTreesRegressor
    from aion_magnitude.metrics import point_photoz_metrics
    m=check(out);records=[json.loads(l) for l in (out/'records.jsonl').read_text().splitlines()];a=d.npz(out/'photometry.npz');train=a['split']=='train';z=a['z'];masks=split_masks(records);rows=[];curves=[];parameters=[]
    train_idx=np.flatnonzero(train);y=torch.from_numpy(np.clip((z[train]/4.5*N_BINS).astype(np.int64),0,N_BINS-1));(out/'predictions').mkdir(exist_ok=True)
    # Exactly identical head/optimizer budget for pretrained and full random Qwen.
    sources={}
    for name in ('base','random'):
        e=d.npz(out/f'embeddings-{name}.npz')
        for pooling in ('last','mean'):sources[f'{name}_{pooling}']=e[pooling]
    sources['base_last_shuffled_labels']=sources['base_last']
    class NumericSetMLP(nn.Module):
        def __init__(self):
            super().__init__();self.band=nn.Sequential(nn.Linear(9,32),nn.GELU(),nn.Linear(32,32),nn.GELU());self.head=nn.Sequential(nn.Linear(32,680),nn.GELU(),nn.Linear(680,N_BINS))
        def forward(self,x):
            mask=x[:,:,8:9];h=self.band(x);pooled=(h*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1);return self.head(pooled)
    sources['numeric_set_mlp']=a['numeric']
    for name,features in sources.items():
        scaler=StandardScaler().fit(features[train] if features.ndim==2 else features[train].reshape(-1,9))
        if features.ndim==2:x=scaler.transform(features).astype(np.float32)
        else:
            # Normalize numeric values/metadata but retain exact padding mask.
            x=scaler.transform(features.reshape(-1,9)).reshape(features.shape).astype(np.float32);x[:,:,8]=features[:,:,8]
        x=torch.from_numpy(x)
        for seed in m['config']['probe_seeds']:
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=NumericSetMLP() if features.ndim==3 else nn.Sequential(nn.Linear(features.shape[1],32),nn.GELU(),nn.Linear(32,N_BINS))
            target=y.clone()
            if 'shuffled' in name:target=target[torch.from_numpy(np.random.default_rng(seed+1000).permutation(len(target)))]
            optimizer=torch.optim.AdamW(model.parameters(),lr=m['config']['probe_lr'],weight_decay=m['config']['probe_weight_decay']);model.train();start=time.monotonic()
            for step in range(m['config']['probe_updates']):
                idx=rng.choice(len(train_idx),m['config']['probe_batch_size'],replace=False);optimizer.zero_grad(set_to_none=True);logits=model(x[train_idx[idx]]);loss=nn.functional.cross_entropy(logits,target[idx]);loss.backward();optimizer.step()
                if step in (0,99,199,399):curves.append(dict(model=name,seed=seed,step=step+1,training_ce=float(loss.detach())))
            model.eval()
            with torch.no_grad():logits=torch.cat([model(x[i:i+256]) for i in range(0,len(x),256)]).numpy()
            pred=prediction_metrics(logits,z);np.savez_compressed(out/'predictions'/f'{name}_seed{seed}.npz',**pred)
            torch.save(dict(state_dict=model.state_dict(),scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,configuration=m['config']),out/'predictions'/f'{name}_seed{seed}_probe.pt')
            parameters.append(dict(model=name,seed=seed,trainable_parameters=sum(p.numel() for p in model.parameters()),training_rows=int(train.sum()),optimizer_updates=m['config']['probe_updates'],seconds=time.monotonic()-start))
            for group,mask in masks.items():
                if mask.any():rows.append(dict(model=name,seed=seed,stratum=group,**d.metrics(pred,mask)))
            print(name,seed,'complete',flush=True)
    # Existing sklearn numerical/tabular infrastructure; scalar output stays scalar.
    x=a['numeric'].reshape(len(records),-1);tree=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=2,max_features=.8,n_jobs=2,random_state=SEED).fit(x[train],z[train]);pred=tree.predict(x)
    np.savez_compressed(out/'predictions/numeric_extra_trees.npz',z_p50=pred,z_spec=z)
    for group,mask in masks.items():
        if mask.any():rows.append(dict(model='numeric_extra_trees',seed=SEED,stratum=group,n=int(mask.sum()),**point_photoz_metrics(torch.from_numpy(pred[mask]),torch.from_numpy(z[mask]))))
    d.table(out/'performance.csv',rows);d.table(out/'training_curves.csv',curves);d.table(out/'parameter_budgets.csv',parameters)
    d.write(out/'fit_status.json',dict(phase='complete',backbones_updated=False,supervision='Synthetic source-filter observations only',target_filter_training=False,real_target_labels=False,configuration_fixed_before_probes=True))


def physical_baseline(out):
    from scipy.special import logsumexp
    m=check(out);(out/'predictions').mkdir(exist_ok=True);curves=f.responses(ROOT);records=[json.loads(l) for l in (out/'records.jsonl').read_text().splitlines()];a=d.npz(out/'photometry.npz');centers=(np.arange(N_BINS)+.5)*4.5/N_BINS
    bank=[]
    for family in ('break4000','weak_break','lyman_continuum'):
        for slope in (-.8,.4):
            for av in (0.,.4):bank.append(dict(family=family,slope=slope,av=av,break_strength=1.05 if family=='weak_break' else 1.8,equivalent_width=0.,log_lnu=28.,template='CWW_E_ext.sed'))
    for template in ('CWW_E_ext.sed','CWW_Sbc_ext.sed','CWW_Scd_ext.sed','CWW_Im_ext.sed'):
        for av in (0.,.4):bank.append(dict(family='empirical',slope=0.,av=av,break_strength=1.,equivalent_width=0.,log_lnu=28.,template=template))
    keys=sorted({b for bands in f.SYSTEMS.values() for b in bands});cube=[]
    for i,param in enumerate(bank):
        cube.append([f.observe_clean(param,float(z),keys,curves,out/'templates') for z in centers]);print('template',i+1,'/',len(bank),flush=True)
    cube=np.array(cube);np.savez_compressed(out/'template_bank.npz',flux=cube,redshift_centers=centers,filters=keys);d.write(out/'template_bank_parameters.json',bank)
    logits=np.zeros((len(records),N_BINS),np.float32)
    for i,r in enumerate(records):
        nb=len(r['bands']);values=a['raw_flux_nanomaggy'][i,:nb];errors=a['raw_sigma_nanomaggy'][i,:nb];valid=np.isfinite(values)&np.isfinite(errors)&(errors>0)
        bidx=[keys.index(k) for k in r['metadata_bands']];model=cube[:,:,bidx][:,:,valid];values=values[valid];iv=1/errors[valid]**2
        numerator=np.sum(model*values*iv,axis=2);denominator=np.sum(model**2*iv,axis=2);amplitude=np.maximum(numerator/np.maximum(denominator,1e-200),0)
        chi2=np.sum((values[None,None,:]-model*amplitude[:,:,None])**2*iv,axis=2);logits[i]=logsumexp(-.5*chi2,axis=0)-np.log(len(bank))
    pred=prediction_metrics(logits,a['z']);np.savez_compressed(out/'predictions/physical_template.npz',**pred)
    d.table(out/'physical_baseline_performance.csv',[dict(model='physical_template',seed=0,stratum=name,**d.metrics(pred,mask)) for name,mask in split_masks(records).items() if mask.any()])
    d.write(out/'physical_baseline_status.json',dict(phase='complete',normalization='nonnegative profile-likelihood amplitude per SED/redshift',template_count=len(bank),redshift_grid_bins=N_BINS,template_prototypes='fixed analytic/CWW parameter grid, never evaluation SED identities or their truth parameters',pdf='equal-template profile-likelihood pseudo-posterior; not a calibrated Bayesian posterior'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,choices=['prepare','init-random','reason','extract','fit','physical']);p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--model',choices=['base','instruct','random']);args=p.parse_args()
    if args.stage=='prepare':prepare(args.output_dir)
    elif args.stage=='init-random':init_random(args.output_dir)
    elif args.stage=='reason':reason(args.output_dir,args.model)
    elif args.stage=='extract':extract(args.output_dir,args.model)
    elif args.stage=='fit':fit(args.output_dir)
    else:physical_baseline(args.output_dir)


if __name__=='__main__':main()
