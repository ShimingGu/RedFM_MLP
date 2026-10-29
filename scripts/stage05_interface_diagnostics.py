#!/usr/bin/env python3
"""Frozen original-HSC diagnostics, with predeclared interventions and no optimizer."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'notebooks'),str(ROOT)]
import numpy as np
import diagnose_hsc_clauds_stage0 as d
OUT=ROOT/'reports/stage05/1027'
MARKER='\nPredict redshift PDF:'
SEED=102705
FULL=('native','ebv_median','ebv_na','ebv_front','query_true','dim7')


def configurations(delta):
    c=[dict(name='native',kind='native'),dict(name='ebv_zero',kind='ebv',value=0.),dict(name='ebv_median',kind='ebv',value='median'),
       dict(name='ebv_shuffle',kind='shuffle'),dict(name='ebv_plus',kind='perturb',delta=delta),dict(name='ebv_minus',kind='perturb',delta=-delta),
       dict(name='ebv_na',kind='ebv',value='missing'),dict(name='ebv_omit',kind='omit'),dict(name='ebv_front',kind='front'),
       dict(name='query_true',kind='query',value='true'),dict(name='query_median',kind='query',value='median'),dict(name='query_na',kind='query',value='missing')]
    c += [dict(name=f'dim{m}',kind='brightness',delta_mag=m,a=10**(-.4*m)) for m in (1,2,4,6,7)]
    c += [dict(name=f'sigma{b}',kind='uncertainty',b=b) for b in (3,10,30,100)]
    c += [dict(name=f'noisy_dim{mag}_b{b}_seed{seed}',kind='noisy',delta_mag=mag,a=10**(-.4*mag),b=b,seed=seed) for mag,b in ((0,100),(4,30),(7,100)) for seed in (17,29,43)]
    return c


def texts_for(data,names,c,median,limits,shuffle):
    from aion_magnitude.FM_Qwen import serialize_qwen_feature_row
    from aion_magnitude.filter_photometry import legacy_config
    x=data['features'].astype(float).copy();kind=c['kind']
    if kind=='brightness':x[:,:10]*=c['a']
    if kind=='uncertainty':x[:,1:10:2]*=c['b']
    if kind=='noisy':
        f=x[:,0:10:2]*c['a'];old=x[:,1:10:2]*c['a'];new=old*c['b']
        variance=np.maximum(0,new**2-old**2)
        x[:,0:10:2]=f+np.random.default_rng(c['seed']).normal(size=f.shape)*np.sqrt(variance)
        x[:,1:10:2]=new
    if kind=='shuffle':x[:,-1]=shuffle
    if kind=='perturb':x[:,-1]=np.clip(x[:,-1]+c['delta'],0,limits[1])
    if kind in ('ebv','query'):
        value=c['value']
        if value=='median':x[:,-1]=median
        elif value=='missing':x[:,-1]=np.nan
        elif value!='true':x[:,-1]=float(value)
    out=[];spec=legacy_config('galaxiesml-spec')
    for row in x:
        features=dict(zip(names,row))
        if kind=='omit':features.pop('MW_EBV')
        if kind=='front':features={'MW_EBV':features['MW_EBV'],**{k:v for k,v in features.items() if k!='MW_EBV'}}
        text=serialize_qwen_feature_row(features,serialization=spec)
        if kind=='query':text+=MARKER
        out.append(text)
    return out,x


def prepare(out):
    from transformers import AutoTokenizer
    from aion_magnitude.filter_photometry import audit_tokens
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'inputs').mkdir();(out/'tokens').mkdir()
    original=d.check(ROOT/'reports/stage0/1027a')
    a=d.npz(d.BASE/'level0/source_cohort.npz');assert set(a['split'])=={'train','val'}
    train=a['split']=='train';val=a['split']=='val';data={k:v[val] for k,v in a.items()};assert len(data['targetid'])==5683
    ebv=a['features'][train,-1];median=float(np.nanmedian(ebv));limits=np.nanquantile(ebv,[0,1]);delta=float(.02*np.diff(np.nanquantile(ebv,[.05,.95]))[0])
    shuffle=data['features'][np.random.default_rng(SEED).permutation(len(data['targetid'])),-1]
    names=json.loads((d.SOURCE/'prepared.json').read_text())['feature_names']
    cfg=configurations(delta);tokenizer=AutoTokenizer.from_pretrained('/arc/home/gsm/hf_models/Qwen3.5-4B-Base',local_files_only=True)
    jobs=[];audit={};np.savez_compressed(out/'validation.npz',**data)
    # Fixed first 256 matches Stage 0 control; full controls declared before any new outputs.
    for c in cfg:
        strings,x=texts_for(data,names,c,median,limits,shuffle)
        for scope,n in [('screen',256),*([('full',5683)] if c['name'] in FULL else [])]:
            key=f'{scope}_{c["name"]}';ts=strings[:n]
            audit[key]=audit_tokens(tokenizer,ts,512)
            with (out/'inputs'/f'{key}.jsonl').open('w') as f:
                for ident,t in zip(data['targetid'][:n],ts):f.write(json.dumps(dict(object_id=str(ident),text=t))+'\n')
            enc=tokenizer(ts,padding=True,truncation=False,return_tensors='np');ids=enc['input_ids'].astype(np.int32);mask=enc['attention_mask'].astype(bool)
            last=np.where(mask,np.arange(mask.shape[1]),-1).max(axis=1)
            np.savez_compressed(out/'tokens'/f'{key}.npz',input_ids=ids,attention_mask=mask,last_position=last,last_token_id=ids[np.arange(n),last])
            np.savez_compressed(out/'inputs'/f'{key}_photometry.npz',features=x[:n],targetid=data['targetid'][:n])
            jobs.append(dict(key=key,scope=scope,n=n,configuration=c))
    # Balance full-validation workload across six workers; each screen job is independent.
    full=[j for j in jobs if j['scope']=='full'];screen=[j for j in jobs if j['scope']=='screen']
    for i,j in enumerate(full):j['worker']=i
    for i,j in enumerate(screen):j['worker']=i%6
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),stage='0.5',source_config=original['source_config'],selected_checkpoints=original['selected_checkpoints'],base_model_source=original['base_model_source'],
        protected_sha256=original['protected_sha256'],source_validation_ids=[str(x) for x in data['targetid']],screen_indices=list(range(256)),n_validation=5683,seed=SEED,field_names=names,jobs=jobs,full_controls=list(FULL),
        ebv_training=dict(median=median,limits=limits.tolist(),q05_q95=np.nanquantile(ebv,[.05,.95]).tolist(),perturbation_delta=delta),query_marker=MARKER,
        trained_head_input_dimension=2560,trained_head_hidden_dimension=256,no_new_head=True,training=False,held_out_test_evaluated=False,target_labels_read=False,
        fixed_noise_definition='sigma_scaled=a*sigma; sigma_target=b*sigma_scaled; added variance=max(0,sigma_target**2-sigma_scaled**2); same deterministic noise values reused across matched IDs',
        caveat='Dimming is a photometric counterfactual, not redshift relocation. Uncertainty-only scaling is not reobservation. Query/front formats were never trained.',token_audits=audit,
        prepared_sha256={str(p.relative_to(out)):d.sha(p) for p in out.rglob('*') if p.is_file()},code_sha256=d.sha(__file__))
    d.write(out/'manifest.json',manifest);print(json.dumps(dict(jobs=len(jobs),full=list(FULL),screen=256,validation=5683,ebv_median=median,perturbation_delta=delta)))


def check(out):
    m=json.loads((out/'manifest.json').read_text())
    for p,v in m['protected_sha256'].items():assert d.sha(p)==v,p
    for p,v in m['prepared_sha256'].items():assert d.sha(out/p)==v,p
    return m


def validate_screen(out):
    m=check(out)
    for w in range(6):
        assert json.loads((out/f'worker-{w}-screen/status.json').read_text())['phase']=='complete'
        assert json.loads((out/f'worker-{w}-screen/native_replay.json').read_text())['passed']
    for j in m['jobs']:
        if j['scope']=='screen':
            a=d.npz(out/f'{j["key"]}.npz');assert a['embedding'].shape==(256,2560)
            assert np.isfinite(a['embedding']).all() and np.allclose(a['pz'].sum(axis=1),1)
    a=d.npz(out/'screen_native.npz');ref=d.npz(ROOT/'reports/stage0/1027a/source-ebv-control/source_native.npz')
    np.testing.assert_array_equal(a['logits'],ref['logits'])
    d.write(out/'screen_validated.json',dict(passed=True,native_logits_bit_exact=True,all_outputs_finite=True,pooled_input_verified=True,full_controls_unchanged=m['full_controls'],no_accuracy_selection=True))


def infer(model,tokenizer,texts,z):
    import torch
    from aion_magnitude.metrics import predict_photoz_from_logits
    from aion_magnitude.utils import make_redshift_grid
    logits=[];pooled=[];head_inputs=[]
    hook=model.photoz_head.register_forward_pre_hook(lambda _,args:head_inputs.append(args[0].detach().float().cpu()))
    # Independently validate actual pooled vector against the final backbone token.
    backbone=[]
    def capture(_,args,kwargs,output):
        from aion_magnitude.FM_Qwen import pool_qwen_hidden_states
        backbone.append(pool_qwen_hidden_states(output.last_hidden_state,kwargs['attention_mask'],pooling='last').detach().float().cpu())
    back_hook=model.qwen.register_forward_hook(capture,with_kwargs=True)
    try:
        with torch.inference_mode():
            for start in range(0,len(texts),8):
                enc=tokenizer(texts[start:start+8],padding=True,truncation=False,return_tensors='pt');assert enc['input_ids'].shape[1]<=512
                # Match the archived evaluator's exact mixed-precision execution.
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    y=model(**{k:v.to('cuda') for k,v in enc.items()})
                torch.testing.assert_close(head_inputs[-1],backbone[-1],rtol=0,atol=0)
                assert head_inputs[-1].shape[1]==2560
                logits.append(y.float().cpu());pooled.append(head_inputs.pop());backbone.pop()
    finally:hook.remove();back_hook.remove()
    l=torch.cat(logits);edges,centers=make_redshift_grid(0,4.5,300);pred=predict_photoz_from_logits(l,centers=centers)
    return dict(logits=l.numpy(),embedding=torch.cat(pooled).numpy(),z_spec=z,redshift_edges=edges.numpy(),redshift_centers=centers.numpy(),**{k:v.numpy() for k,v in pred.items() if k in ('pz','z_p16','z_p50','z_p84')})


def run(out,worker,scope):
    import torch
    import qwen_zero_shot_sanity as pilot
    import qwen_clauds_common_flux_evaluation as runner
    from aion_magnitude.qwen_continued_sft import verify_replay
    m=check(out);folder=out/f'worker-{worker}-{scope}';folder.mkdir(exist_ok=True)
    jobs=[j for j in m['jobs'] if j['worker']==worker and j['scope']==scope];data=d.npz(out/'validation.npz')
    if scope=='full':assert (out/'screen_validated.json').exists(),'Complete screening correctness checks before full validation'
    d.write(folder/'status.json',dict(phase='loading',pid=os.getpid()))
    try:
        with pilot.reserve_one_gpu() as gpu:
            assert gpu['memory_before_gib']['used']<.2
            model,tok=runner.load_model(folder,m,'original');assert not any(p.requires_grad for p in model.parameters())
            ref=runner.evaluation_from_archive(d.FAST/'source_replay.npz');replaydata=d.npz(d.FAST/'source_replay.npz');meta=json.loads((d.SOURCE/'prepared.json').read_text())
            from aion_magnitude.filter_photometry import serialize_rows
            e=pilot.evaluate(model,tok,dict(replaydata,z=replaydata['z_spec']),serialize_rows(replaydata,meta,catalogue='galaxiesml-spec',level=0),m['source_config'])
            d.write(folder/'native_replay.json',verify_replay(e,ref))
            for j in jobs:
                path=out/f'{j["key"]}.npz'
                if path.exists():continue
                d.write(folder/'status.json',dict(phase='inference',job=j['key'],pid=os.getpid(),gpu=gpu))
                texts=[json.loads(l)['text'] for l in (out/'inputs'/f'{j["key"]}.jsonl').read_text().splitlines()]
                start=time.monotonic();a=infer(model,tok,texts,data['z'][:j['n']]);np.savez_compressed(path,targetid=data['targetid'][:j['n']],**a)
                print(j['key'],json.dumps(dict(seconds=time.monotonic()-start,**d.metrics(a,np.ones(j['n'],bool)))),flush=True)
            check(out);d.write(folder/'status.json',dict(phase='complete',gpu=gpu,training=False))
    except BaseException as e:d.write(folder/'status.json',dict(phase='failed',error=repr(e)));raise


def collect(out):
    from scipy.stats import spearmanr
    import torch
    from aion_magnitude.qwen_continued_sft import verify_replay
    m=check(out)
    for w in range(6):
        for scope in ('screen','full'):assert json.loads((out/f'worker-{w}-{scope}/status.json').read_text())['phase']=='complete'
    native={scope:d.npz(out/f'{scope}_native.npz') for scope in ('screen','full')};rows=[];paired=[];tokens=[];strata=[];precision=[]
    actual={k:torch.from_numpy(v) for k,v in native['full'].items() if k not in ('embedding','targetid')}
    actual['object_id']=[str(x) for x in native['full']['targetid']]
    reference=torch.load(d.BASE/'level0/reference_validation.pt',map_location='cpu',weights_only=False)
    d.write(out/'full_native_replay.json',verify_replay(actual,reference))
    data=d.npz(out/'validation.npz');source=d.npz(d.BASE/'level0/source_cohort.npz',('features','split'));train=source['features'][source['split']=='train'];bright_edges=np.nanquantile(train[:,4],[1/3,2/3]);sn_edges=np.nanquantile(np.nanmedian(train[:,:10:2]/train[:,1:10:2],axis=1),[1/3,2/3])
    support_manifest=json.loads((ROOT/'reports/stage0/1027a/support_summary.json').read_text());distance_models={}
    from sklearn.neighbors import NearestNeighbors
    for name in ('flux_colour','joint_photometry'):
        spec=support_manifest['diagnostics'][name]['folds'][0]['preprocessor'];fit=d.npz(ROOT/f'reports/stage0/1027a/{name}_fold0.npz')['fit_source_indices'];sx=d.transform(train[fit,0:10:2],train[fit,1:10:2],spec);nn=NearestNeighbors(n_neighbors=10,n_jobs=2).fit(sx);q=support_manifest['diagnostics'][name]['folds'][0]['threshold_q95'];distance_models[name]=(nn,spec,q)
    for j in m['jobs']:
        key=j['key'];a=d.npz(out/f'{key}.npz');base=native[j['scope']];n=j['n'];r=dict(condition=key,scope=j['scope'],**d.metrics(a,np.ones(n,bool)))
        x=d.npz(out/'inputs'/f'{key}_photometry.npz')['features'];sn=np.nanmedian(x[:,:10:2]/x[:,1:10:2],axis=1);r['median_sn']=float(np.nanmedian(sn))
        rounded=np.array([[float(f'{v:.5f}') for v in row] for row in x]);old_sigma=x[:,1:10:2];new_sigma=rounded[:,1:10:2]
        precision.append(dict(condition=key,measured_positive_errors=int(np.sum(np.isfinite(old_sigma)&(old_sigma>0))),errors_rounded_to_zero=int(np.sum(np.isfinite(old_sigma)&(old_sigma>0)&(new_sigma==0))),signed_nonzero_fluxes_rounded_to_zero=int(np.sum(np.isfinite(x[:,:10:2])&(x[:,:10:2]!=0)&(rounded[:,:10:2]==0))),max_absolute_photometry_rounding_error=float(np.nanmax(np.abs(x[:,:10]-rounded[:,:10])))))
        r['spearman_prediction_truth']=float(spearmanr(a['z_p50'],a['z_spec']).statistic)
        for name,(nn,spec,q) in distance_models.items():r[f'median_{name}_ratio']=float(np.median(nn.kneighbors(d.transform(x[:,:10:2],x[:,1:10:2],spec),return_distance=True)[0].mean(axis=1)/q))
        rows.append(r)
        emb=a['embedding'];orig=base['embedding'];cos=np.sum(emb*orig,axis=1)/(np.linalg.norm(emb,axis=1)*np.linalg.norm(orig,axis=1));dis=np.linalg.norm(emb-orig,axis=1)/np.maximum(np.linalg.norm(orig,axis=1),1e-20)
        pp=np.maximum(a['pz'],1e-30);qq=np.maximum(base['pz'],1e-30);mid=.5*(pp+qq);js=.5*np.sum(pp*np.log(pp/mid)+qq*np.log(qq/mid),axis=1);ld=np.max(np.abs(a['logits']-base['logits']),axis=1)
        token=d.npz(out/'tokens'/f'{key}.npz');bt=d.npz(out/'tokens'/f'{j["scope"]}_native.npz');changed=token['last_token_id']!=bt['last_token_id']
        # The saved rectangular token arrays use condition-wide right padding.
        # Inference uses batches of eight: this layout reconstructs each exact
        # input_ids/attention_mask tensor by slicing its rows and padded width.
        layout=[]
        for start in range(0,n,8):
            stop=min(start+8,n);width=int(token['attention_mask'][start:stop].sum(axis=1).max())
            assert not token['attention_mask'][start:stop,width:].any()
            layout.append(dict(start=start,stop=stop,padded_width=width))
        d.write(out/'tokens'/f'{key}_actual_batch_layout.json',dict(batch_size=8,padding_side='right',arrays=f'{key}.npz',reconstruction='input_ids[start:stop,:padded_width], attention_mask[start:stop,:padded_width]',batches=layout))
        tokens.append(dict(condition=key,n=n,mean_token_count=float(token['attention_mask'].sum(axis=1).mean()),last_token_changed_fraction=float(changed.mean()),mean_cosine=float(cos.mean()),mean_normalized_displacement=float(dis.mean()),mean_js_divergence=float(js.mean()),mean_max_logit_difference=float(ld.mean()),mean_abs_redshift_shift=float(np.mean(np.abs(a['z_p50']-base['z_p50'])))))
        for i in range(n):paired.append(dict(condition=key,object_id=str(a['targetid'][i]),z_reference=float(a['z_spec'][i]),prediction=float(a['z_p50'][i]),native_prediction=float(base['z_p50'][i]),prediction_shift=float(a['z_p50'][i]-base['z_p50'][i]),cosine=float(cos[i]),normalized_embedding_displacement=float(dis[i]),js_divergence=float(js[i]),max_logit_difference=float(ld[i]),final_token_changed=bool(changed[i])))
        # Strata use original observations and source-training thresholds, never intervention errors.
        groups={'z_0_0.3':data['z'][:n]<.3,'z_0.3_0.7':(data['z'][:n]>=.3)&(data['z'][:n]<.7),'z_ge0.7':data['z'][:n]>=.7}
        b=np.digitize(data['features'][:n,4],bright_edges);ns=np.digitize(np.nanmedian(data['features'][:n,0:10:2]/data['features'][:n,1:10:2],axis=1),sn_edges)
        for label,v in [('brightness',b),('sn',ns)]:
            for bin in range(3):groups[f'original_{label}_tercile{bin}']=v==bin
        for group,mask in groups.items():
            if mask.any():strata.append(dict(condition=key,stratum=group,**d.metrics(a,mask)))
    d.table(out/'performance.csv',rows);d.table(out/'representation_changes.csv',tokens);d.table(out/'paired_objects.csv',paired);d.table(out/'conditional_metrics.csv',strata);d.table(out/'numerical_precision_audit.csv',precision)
    plots(out,m,rows,native)
    d.write(out/'summary.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),stage='0.5',conditions=len(rows),full_validation_controls=list(FULL),training=False,new_head=False,held_out_test_evaluated=False,protected_artifacts_unchanged=True,pooling_verified_against_actual_head_input=True,performance=rows,representation_changes=tokens))
    d.write(out/'correctness_review.json',dict(stage05_complete=True,no_blocking_correctness_problem_for_stage1=True,units='Source nMgy and linear errors unchanged; no inferred EBV',pooling='Actual last nonpadding vector verified against the trained head',precision='BF16 autocast matched and native replay checked; historical decimal quantization measured',data_leakage='Only source train EBV/preprocessing and original validation interventions; no CLAUDS target labels or test objects',stage1_requirements=['Photon-response weighted full-curve integration','Flat-fnu AB and redshift/energy-scaling forward tests before model probes','SED-identity splits before all counterfactual observations','Independent simulation protocol; no tuning to Stage05 predictions']))
    report(out,rows,tokens,m)
    d.write(out/'final_provenance.json',dict(code_sha256=d.sha(__file__),prepared_code_sha256=m['code_sha256'],prepared_inputs_unchanged=True,protected_artifacts_unchanged=True,autocast='torch.autocast(cuda,bfloat16), matching archived evaluation',tests_sha256=d.sha(ROOT/'tests/test_stage05_physics_controls.py')))
    d.write(out/'artifact_inventory.json',{str(p.relative_to(out)):dict(sha256=d.sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='artifact_inventory.json' and p.suffix!='.log'})
    print(out/'report.md')


def plots(out,m,rows,native):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    look={r['condition']:r for r in rows};fig,axs=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,field in zip(axs,['catastrophic_outlier_fraction','predicted_median','p16_p84_coverage']):
        ax.plot([0,1,2,4,6,7],[look[f'screen_{"native" if mag==0 else "dim"+str(mag)}'][field] for mag in (0,1,2,4,6,7)],'o-',label='Dimming at fixed S/N');ax.set(xlabel='Magnitude dimming',ylabel=field);ax.legend(fontsize=7)
    for ext in ('png','pdf'):fig.savefig(out/f'brightness_response.{ext}',dpi=170)
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,field in zip(axs,['catastrophic_outlier_fraction','predicted_median','p16_p84_coverage']):
        ax.plot([1,3,10,30,100],[look[f'screen_{"native" if b==1 else "sigma"+str(b)}'][field] for b in (1,3,10,30,100)],'o-');ax.set(xscale='log',xlabel='Reported uncertainty multiplier',ylabel=field)
    for ext in ('png','pdf'):fig.savefig(out/f'uncertainty_response.{ext}',dpi=170)
    plt.close(fig)
    show=['native','ebv_zero','ebv_median','ebv_na','ebv_front','query_true','dim7','sigma100','noisy_dim7_b100_seed17'];fig,axs=plt.subplots(3,3,figsize=(13,11),constrained_layout=True)
    for ax,name in zip(axs.flat,show):
        a=d.npz(out/f'screen_{name}.npz');ax.scatter(a['z_spec'],a['z_p50'],s=9,alpha=.6);ax.plot([0,4.5],[0,4.5],'--',color='grey');ax.set(title=name,xlabel='Original spectroscopic redshift',ylabel='Predicted median',xlim=(0,4.5),ylim=(0,4.5))
    for ext in ('png','pdf'):fig.savefig(out/f'collapse_comparison.{ext}',dpi=170)
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,i in zip(axs.flat,[0,64,128,192]):
        for name in ('native','ebv_median','ebv_na','ebv_front','query_true','dim7','sigma100'):
            a=d.npz(out/f'screen_{name}.npz');ax.plot(a['redshift_centers'],a['pz'][i],label=name,lw=1)
        ax.axvline(native['screen']['z_spec'][i],color='black',ls='--');ax.set(title=f'Preselected index {i}',xlabel='Redshift',ylabel='Probability per bin',xlim=(0,4.5))
    axs.flat[0].legend(fontsize=7)
    for ext in ('png','pdf'):fig.savefig(out/f'representative_pdfs.{ext}',dpi=170)
    plt.close(fig)


def report(out,rows,changes,m):
    look={r['condition']:r for r in rows};rep={r['condition']:r for r in changes};lines=['# Stage 0.5 — Frozen interface, brightness and noise diagnostics','',
        'All interventions use the original completed HSC QLoRA adapter and its trained 300-bin 0–4.5 PDF head. No new head, optimizer or weight update is used. Screening uses the first 256 original source-validation objects; six controls were fixed in advance for the entire 5,683-object validation cohort. Reference labels are DESI spectroscopy. No CLAUDS labels or astrophysical held-out test objects are read.','',
        '## Verified architecture','',f"The original field order is {m['field_names']}. Five-decimal numeric EBV is the final field. There is no appended query token. Actual padding masks/last-token IDs are saved for every input. A forward pre-hook captured the exact 2,560-dimensional vector entering the existing 256-hidden-unit PDF head; a separate backbone hook verifies bit-exact equality with last non-padding token pooling for every batch. This tests the implementation, not just its intended configuration.",'',
        f"Training EBV median is {m['ebv_training']['median']:.6f}; q05/q95 are {m['ebv_training']['q05_q95']}. Perturbations are ±{m['ebv_training']['perturbation_delta']:.6g}, clipped to [0, observed training maximum]. Zero, median, true/shuffled and perturbation controls retain five-decimal formatting. Numeric changes still change token identity; they do not uniquely isolate semantics. The fixed ordinary-text marker is {MARKER!r}; no embeddings are added. Field-order/marker formats were not trained, so this is an interface stress test rather than a matched architecture comparison.",'','## Full validation: prespecified controls','',
        '| Condition | N | CE | NMAD | Outliers | Bias | CRPS | Coverage | Median prediction | Concentration width .03 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['scope']=='full':lines.append(f"| {r['condition']} | {r['n']} | {r['cross_entropy']:.4f} | {r['nmad']:.4f} | {r['catastrophic_outlier_fraction']:.2%} | {r['bias']:+.4f} | {r['mean_crps']:.4f} | {r['p16_p84_coverage']:.2%} | {r['predicted_median']:.4f} | {r['concentration_width_003']:.2%} |")
    lines+=['','## Screening factorial controls','','| Condition | NMAD | Outliers | Coverage | Median z | Median S/N | Joint support-distance ratio |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['scope']=='screen':lines.append(f"| {r['condition']} | {r['nmad']:.4f} | {r['catastrophic_outlier_fraction']:.2%} | {r['p16_p84_coverage']:.2%} | {r['predicted_median']:.4f} | {r['median_sn']:.2f} | {r['median_joint_photometry_ratio']:.3f} |")
    lines+=['','## Five causal questions','',
        f"1. **EBV semantics versus interface:** screening outliers are true={look['screen_native']['catastrophic_outlier_fraction']:.2%}, zero={look['screen_ebv_zero']['catastrophic_outlier_fraction']:.2%}, median={look['screen_ebv_median']['catastrophic_outlier_fraction']:.2%}, shuffle={look['screen_ebv_shuffle']['catastrophic_outlier_fraction']:.2%}, small plus/minus={look['screen_ebv_plus']['catastrophic_outlier_fraction']:.2%}/{look['screen_ebv_minus']['catastrophic_outlier_fraction']:.2%}, and NA={look['screen_ebv_na']['catastrophic_outlier_fraction']:.2%}. Full-validation median/NA controls are in the table above. Different outcomes for numeric alternatives and NA limit any interpretation in terms of astrophysical EBV alone; token identity and the terminal representation also change. No unique semantic/token causal decomposition is claimed.",'',
        f"2. **Order invariance:** moving true EBV before photometry changes predictions by mean absolute |delta z|={rep['full_ebv_front']['mean_abs_redshift_shift']:.4f} on all validation objects, with mean representation cosine {rep['full_ebv_front']['mean_cosine']:.6f}. Adding the fixed ordinary marker with true EBV gives |delta z|={rep['full_query_true']['mean_abs_redshift_shift']:.4f}. These preserve physical observations but are untrained input formats; they test fragility, not superiority of another pooling architecture.",'',
        f"3. **Brightness at fixed S/N:** seven-mag dimming changes full-validation outliers from {look['full_native']['catastrophic_outlier_fraction']:.2%} to {look['full_dim7']['catastrophic_outlier_fraction']:.2%}, predicted median from {look['full_native']['predicted_median']:.4f} to {look['full_dim7']['predicted_median']:.4f}, and concentration in width .03 from {look['full_native']['concentration_width_003']:.2%} to {look['full_dim7']['concentration_width_003']:.2%}. Colours and unrounded S/N are invariant. This demonstrates strong flux-scale dependence, while a brightness-dependent population prior can be physically legitimate.",'',
        f"4. **Reported S/N versus noise:** sigma x100 at fixed measured flux gives {look['screen_sigma100']['catastrophic_outlier_fraction']:.2%} outliers and median z={look['screen_sigma100']['predicted_median']:.4f}; the native screening values are {look['screen_native']['catastrophic_outlier_fraction']:.2%} and {look['screen_native']['predicted_median']:.4f}. The nine seeded actual-noise rows quantify the separate perturbation. Neither intervention alone is a realistic survey population change.",'',
        f"5. **Direction of CLAUDS failure:** with true EBV held fixed, extreme dimming leads to median z={look['full_dim7']['predicted_median']:.4f}; NA EBV without dimming leads to median z={look['full_ebv_na']['predicted_median']:.4f}. Compare these measured directions with the archived CLAUDS low-z collapse (~0.08–0.13). The source interventions expose mechanisms and can reproduce concentration, but do not uniquely reconstruct the full CLAUDS failure or substitute for its different population/measurement pipeline.",'',
        '![Brightness](brightness_response.png)','![Reported uncertainty](uncertainty_response.png)','![Collapse](collapse_comparison.png)','![PDFs](representative_pdfs.png)','',
        'All brightness cases multiply measured flux and error together by 10**(-0.4*delta_mag), preserving unrounded colour ratios and S/N. Five-decimal serialization quantization, including any errors that round to zero, is measured in the numerical audit; no extra precision is silently substituted. Dimming is a counterfactual measurement scale, not a physical relocation or a new redshift label. Apparent brightness can carry a valid population prior. Uncertainty-only changes alter the report, not the measured realization. Noise cases add independent Gaussian variance sigma_target²−sigma_scaled², with three fixed seeds per cell; negative measurements are preserved. This is not an image-level noise simulator.','',
        'Paired representation cosine, normalized displacement, maximum logit differences, Jensen–Shannon PDF divergence and prediction changes are in representation_changes.csv and paired_objects.csv. Conditional metrics use original spectroscopic-redshift intervals and source-training brightness/S/N terciles; no configuration is chosen by error. Source-distance ratios use exactly Stage 0 fold-0 preprocessing, neighbours and thresholds. Raw inputs, token IDs/masks, actual pooled representations and PDF logits/probabilities are saved.','',
        '## Correctness review and verdict','','Numerical EBV, token identity/position and the pooled terminal state remain coupled; use the paired numerical, missingness, position and marker controls to delimit sensitivity, without claiming unique causal attribution. Reordering or appending text leaves the physical observations fixed and directly tests serialization fragility. Brightness-only and uncertainty-only conditions separate population-prior response from reported-S/N dependence; stochastic cases test the interaction. None proves absence of pretrained astrophysical information.','',
        'No archived weight, grid or split changes. Native replay is checked independently on each worker. The first screening attempt omitted the archived BF16 autocast context; its native replay failed and blocked expansion. That preliminary output is retained only under superseded-fp32-screen/INVALID.md and excluded from scientific tables. Corrected screening reproduces the archived native logits exactly. The full gate opens only after all screening correctness checks pass. The source unit conversion is unchanged, fluxes/errors stay in nMgy, true EBV/redshifts stay fixed in brightness/noise experiments, and missingness remains explicit. The original five-decimal interface is intentionally retained even where counterfactual dimming exposes inadequate precision (numerical_precision_audit.csv). This limitation must not be copied into Stage 1: synthetic flux/error serialization will use sufficient scientific precision and a fixed ordinary query marker for matched frozen probes. Stage 1 must integrate the full photon response rather than sample pivots, test AB normalization, and separate SED identities before generating redshifts/filters.','',
        'There is no blocker to independent Stage-1 controlled synthetic diagnostics: Stage 0.5 reveals fragility of this historical interface, not an error that makes a separately unit-tested forward benchmark invalid. Stage 1 may proceed with a frozen benchmark protocol, without tuning to these predictions or CLAUDS labels. A trustworthy production cross-survey photo-z interface and the degree of remaining real-domain shift are not established here.','',
        'The optional archived MLP/template comparison was not executed: no matching trained numerical baseline plus preprocessing contract was available in the primary HSC run. No new baseline was trained during Stage 0.5.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','run','validate-screen','collect'],required=True);p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--worker',type=int);p.add_argument('--scope',choices=['screen','full']);args=p.parse_args()
    if args.stage=='prepare':prepare(args.output_dir)
    elif args.stage=='run':run(args.output_dir,args.worker,args.scope)
    elif args.stage=='validate-screen':validate_screen(args.output_dir)
    else:collect(args.output_dir)


if __name__=='__main__':main()
