#!/usr/bin/env python3
"""Audit historical Legacy models and prepare matched training; no implicit training."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import re
import shutil
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'notebooks'))
import numpy as np
from scipy.spatial import cKDTree
from hsc_legacy_paired_benchmark import read,sha,now,unit_vectors,nearby
from filter_metadata_counterfactual import write

BENCH=ROOT/'filter-aware-runs/1027/hsc-legacy-paired-full'
RUNS=Path('/arc/home/gsm/aion_output/figures')
BINS=[0.,.2,.4,.6,1.,4.5]
SEEDS=[42,43,44]


def instrument_hash(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def telemetry(folder,updates_per_epoch=3741,head_epochs=3):
    files=sorted(folder.glob('checkpoint-update-*.pt'));result={'head_warmup':[],'joint':[]};raw=[]
    for a,b in zip(files,files[1:]):
        ia=int(a.stem.split('-')[-1]);ib=int(b.stem.split('-')[-1]);seconds=b.stat().st_mtime-a.stat().st_mtime
        if ib-ia!=100 or seconds<=0 or (ia-1)//updates_per_epoch!=(ib-1)//updates_per_epoch:continue
        phase='head_warmup' if ib<=updates_per_epoch*head_epochs else 'joint'
        result[phase].append(seconds/100);raw.append(dict(first=str(a),last=str(b),phase=phase,seconds=seconds,updates=100))
    summary={}
    for phase,values in result.items():
        if values:summary[phase]=dict(intervals=len(values),seconds_per_update_median=float(np.median(values)),
            seconds_per_update_q25_q75=np.quantile(values,[.25,.75]).tolist(),min=float(min(values)),max=float(max(values)))
    return dict(summary=summary,raw_intervals=raw,interpretation='Actual adjacent checkpoint mtimes, exactly 100 updates, excluding epoch boundaries; includes checkpoint I/O. Median/IQR limit interruption sensitivity; elapsed wall time is not a hardware-normalized FLOP measure.')


def inventory(out):
    import torch
    from safetensors.torch import load_file
    from aion_magnitude.matched_training import state_fingerprint
    folder=out/'legacy_checkpoint_audit';folder.mkdir(parents=True,exist_ok=True)
    result={};protected={};telemetries={}
    for method in ['qlora','dora']:
        run=RUNS/f'qwen-desi-ls-{method}-specz-e10'
        meta=json.loads((run/'prepared.json').read_text());contract=json.loads((run/'run_contract.json').read_text())
        selected=run/method;res=torch.load(selected/'result.pt',map_location='cpu',weights_only=False,mmap=True)
        assert len(res['history'])==10 and res['metadata']['use_dora']==(method=='dora')
        config=json.loads((selected/'adapter/adapter_config.json').read_text())
        assert config['use_dora']==(method=='dora') and config['r']==8
        adapter=load_file(str(selected/'adapter/adapter_model.safetensors'));head=torch.load(selected/'photoz_head.pt',map_location='cpu',weights_only=True)
        assert head['net.6.weight'].shape[0]==300
        checkpoint_dir=Path(contract['checkpoint_dir']);files=sorted(checkpoint_dir.glob('checkpoint-update-*.pt'));latest=files[-1]
        resume=torch.load(latest,map_location='cpu',weights_only=False,mmap=True)
        best_epoch=min(res['history'],key=lambda x:x['val_cross_entropy'])['epoch']
        assert best_epoch==9 and res['history'][-1]['val_cross_entropy']==res['final_metrics']['val']['cross_entropy']
        log=run/'multigpu_logs/adapter.log';log_text=log.read_text(errors='replace')
        hashes={str(p.resolve()):sha(p) for p in [selected/'result.pt',selected/'photoz_head.pt',*sorted((selected/'adapter').glob('*')),run/'prepared.json',run/'run_contract.json',log,latest] if p.is_file()};protected.update(hashes)
        result[method]=dict(run=str(run),selected_directory=str(selected),base_model=contract['model_source']['model'],
            adapter_tensor_sha256=state_fingerprint(adapter),head_tensor_sha256=state_fingerprint(head),sha256=hashes,
            adapter_config=config,training_config=res['metadata']['config'],source_cache_key=meta['cache_key'],
            source_split_counts=meta['split_counts'],feature_names=meta['feature_names'],serialization=contract['serialization'],
            selection=dict(rule='Minimum original source-validation CE over ten epochs',selected_epoch_zero_based=best_epoch,
                completed_optimizer_updates=37410,selected_equals_fixed_final_weights=True,
                evidence='Archived epoch 9 is best; original code restores best only after all epochs. Thus selected and last-budget weights coincide in these runs; no separate final adapter was exported.'),
            last_resumable_checkpoint=dict(path=str(latest),global_update=resume['global_update'],epoch=resume['epoch'],next_batch_index=resume['next_batch_index'],best_val_loss=resume['best_val_loss'],
                latest_trainable_tensor_sha256=state_fingerprint(resume['trainable_state']),latest_best_tensor_sha256=state_fingerprint(resume['best_trainable_state']),
                warning='This is update 37400, not the final 37410 state. Its stored best snapshot is epoch 8, not the selected epoch 9 export.'),
            original_validation_metrics=res['final_metrics']['val'],history=res['history'],
            resume_log_records=[x for x in log_text.replace('\r','\n').splitlines() if 'resuming ' in x],
            artifact_times={p.name:datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat() for p in [selected/'result.pt',selected/'photoz_head.pt',selected/'adapter/adapter_model.safetensors']},
            validation_replay='pending',source_ID_recovery='pending',compatible_grid=True)
        np.savez_compressed(folder/f'{method}_validation_reference.npz',targetid=np.asarray(res['val_evaluation']['object_id'],np.int64),
            **{k:res['val_evaluation'][k].numpy() for k in ['logits','pz','z_spec','z_p16','z_p50','z_p84','redshift_edges','redshift_centers']})
        telemetries[method]=telemetry(checkpoint_dir)
    assert result['qlora']['source_cache_key']==result['dora']['source_cache_key']
    write(folder/'inventory.json',dict(created_utc=now(),models=result,original_artifact_sha256=protected,
        discovery='Both completed exports exist at their actual metadata-recorded run paths. QLoRA export time 2026-10-27 07:14 UTC; DoRA 12:44 UTC, after the earlier initial discovery pass. Earlier reports remain immutable historical artifacts.'))
    write(out/'feasibility/historical_telemetry.json',telemetries)
    print('COMPLETED LEGACY ARTIFACTS INVENTORIED',flush=True)


def physical_overlap(data):
    """Union all physical sky representations and identifiers; reject cross-split ties."""
    radius=2*np.sin(np.deg2rad(1/3600)/2);n=len(data['z']);pairs=set()
    for key in ['targetid','galaxy_id','legacy_group']:
        ids=data[key];order=np.argsort(ids,kind='stable');same=np.flatnonzero(ids[order][1:]==ids[order][:-1])
        pairs.update(tuple(sorted((int(order[j]),int(order[j+1])))) for j in same)
    h=unit_vectors(data['ra'],data['dec']);l=unit_vectors(data['legacy_ra'],data['legacy_dec'])
    for points in [h,l]:
        q=cKDTree(points).query_pairs(radius,output_type='ndarray');pairs.update(tuple(map(int,x)) for x in q)
    neighbours=cKDTree(h).query_ball_tree(cKDTree(l),radius)
    for i,js in enumerate(neighbours):pairs.update(tuple(sorted((i,int(j)))) for j in js if i!=j)
    crosses=[pair for pair in sorted(pairs) if data['category'][pair[0]]!=data['category'][pair[1]]]
    return dict(tolerance_arcsec=1.,total_distinct_object_pairs=len(pairs),cross_category_pairs=crosses,
        policy='TARGETID, HSC imaging ID, Legacy release/brick/object identity; HSC/HSC, Legacy/Legacy and HSC/Legacy spherical positions. Self cross-representation pairs excluded.')


def exposure(out):
    legacy=read(out/'legacy_checkpoint_audit/recovered_cohort/cohort.npz')
    assert json.loads((out/'legacy_checkpoint_audit/recovered_cohort/recovery_proof.json').read_text())['archived_cache_key_exact']
    data=read(BENCH/'eligible_pairs.npz');hdf=read(BENCH/'raw-audit/hdf_metadata.npz');n=len(data['z'])
    hlookup={int(t):i for i,t in enumerate(hdf['targetid'])};p_hsc=data['galaxy_id']
    flags={k:v for k,v in data.items() if any(k.startswith(model+'_') for model in ['original-qlora','original-dora','A','B','C'])}
    for method in ['qlora','dora']:
        for split in ['train','val','test']:
            source=np.flatnonzero(legacy['split']==split)
            exact=np.isin(data['targetid'],legacy['targetid'][source]);group=np.isin(data['legacy_group'],legacy['group'][source])
            match=[hlookup[int(t)] for t in legacy['targetid'][source] if int(t) in hlookup]
            hsc_group=np.isin(p_hsc,np.asarray([f'hsc:{i}' for i in hdf['hsc_object_id'][match]]))
            sky=nearby(data['ra'],data['dec'],legacy['ra'][source],legacy['dec'][source])
            sky|=nearby(data['legacy_ra'],data['legacy_dec'],legacy['ra'][source],legacy['dec'][source])
            key=f'legacy-{method}_{split}'
            flags[key+'_exact']=exact;flags[key+'_imaging']=group|hsc_group;flags[key+'_positional']=sky;flags[key]=exact|group|hsc_group|sky
    old=data['category']=='P';contaminated=np.zeros(n,bool)
    for method in ['qlora','dora']:
        contaminated|=flags[f'legacy-{method}_train']|flags[f'legacy-{method}_val']
    mutual=old&~contaminated;overlap=physical_overlap(data)
    folder=out/'original_model_exposure';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'exposure_flags.npz',targetid=data['targetid'],original_category=data['category'],P_mutual=mutual,**flags)
    ix=np.flatnonzero(mutual);np.savez_compressed(folder/'mutual_cohort.npz',**{k:v[ix] for k,v in data.items()})
    with (folder/'mutual_TARGETIDs.txt').open('w') as f:f.write(''.join(f'{int(x)}\n' for x in data['targetid'][ix]))
    write(folder/'audit.json',dict(created_utc=now(),existing_P=int(old.sum()),mutual_P=int(mutual.sum()),excluded_P=int((old&contaminated).sum()),
        flags_P={k:int(v[old].sum()) for k,v in flags.items()},flags_full_pool={k:int(v.sum()) for k,v in flags.items()},
        historical_qlora_dora_cohort_identical=True,physical_overlap=overlap,
        original_model_independence='P_mutual excludes training and source-validation exposure across all seven historical models. Previously evaluated held-out tests are explicitly flagged, not claimed never-inspected.',
        no_full_inference_performed=True))
    print('MUTUAL INDEPENDENT P',int(mutual.sum()),'excluded',int((old&contaminated).sum()),'cross-split candidates',len(overlap['cross_category_pairs']),flush=True)


def prepare(out):
    import torch
    from transformers import AutoTokenizer
    from aion_magnitude.filter_photometry import serialize_rows,feature_columns,audit_tokens,DEFAULT_REGISTRY
    from aion_magnitude.matched_training import initial_head,state_fingerprint,train_order
    from hsc_legacy_paired_report import photometric_summary
    data=read(BENCH/'eligible_pairs.npz');audit=json.loads((out/'original_model_exposure/audit.json').read_text())
    if audit['physical_overlap']['cross_category_pairs']:
        raise ValueError('Cross-category physical neighbours exist; preserve fixed P and review a minimal geometry-only split correction')
    counts={c:int((data['category']==c).sum()) for c in ['T','V','P']};assert counts==dict(T=16600,V=4257,P=61696)
    split=out/'paired_split';split.mkdir(exist_ok=True);np.savez_compressed(split/'paired_cohort.npz',**data)
    redshifts={};distributions={}
    for c,name in [('T','train'),('V','validation'),('P','evaluation')]:
        ix=np.flatnonzero(data['category']==c);np.savez_compressed(split/f'{name}_membership.npz',targetid=data['targetid'][ix],galaxy_id=data['galaxy_id'][ix],legacy_group=data['legacy_group'][ix],row=ix,z=data['z'][ix])
        redshifts[name]=dict(n=len(ix),min=float(data['z'][ix].min()),max=float(data['z'][ix].max()),histogram_edges=BINS,histogram=np.histogram(data['z'][ix],bins=BINS)[0].tolist())
        for domain,bands in [('hsc',['g','r','i','z','y']),('legacy',['g','r','z','W1','W2'])]:distributions[f'{name}_{domain}']=photometric_summary(data[f'{domain}_features'][ix],bands)
    write(split/'audit.json',dict(created_utc=now(),counts=counts,physical_overlap=audit['physical_overlap'],redshift_distributions=redshifts,
        same_labels_for_both_domains=True,quality_policy=json.loads((BENCH/'audit.json').read_text())['label_audit'],
        suitability='Historical exposure labels are a fixed, geometry-clean allocation, not inherited model weights. Fresh bases make T/V/P valid for the new models. Historical Legacy exposure of P is irrelevant to fresh-model training but is audited separately.',
        evaluation_status='Locked, previously analysed paired benchmark; not a prospective blind test',
        prospective_replication='No additional uninspected eligible set exists within this fully audited pool under the unchanged policy. Reserve a new external matched catalogue before training for a future replication; excluded/ambiguous rows are not silently promoted.'))
    write(split/'photometric_distributions.json',distributions)
    base_identity=json.loads((BENCH/'base_model_identity.json').read_text());base=Path(base_identity['model'])
    for file,info in base_identity['files'].items():assert sha(file)==info['sha256'],f'Original pre-adaptation base changed: {file}'
    write(out/'model_initialisation/base_model_identity.json',base_identity)
    model_config=json.loads((base/'config.json').read_text());hidden=model_config.get('hidden_size',model_config.get('text_config',{}).get('hidden_size'))
    assert hidden==2560
    for seed in SEEDS:
        path=out/'model_initialisation'/f'head_seed{seed}.pt';state=initial_head(hidden,seed);torch.save(state,path)
        np.save(out/'model_initialisation'/f'presentation_order_seed{seed}.npy',train_order(counts['T'],seed,10))
    rep=out/'representation_audit';rep.mkdir(exist_ok=True);tokenizer=AutoTokenizer.from_pretrained(str(base),local_files_only=True)
    input_hashes={};tokens={}
    for domain,kind,group in [('hsc','galaxiesml-spec',data['galaxy_id']),('legacy','desi-ls',data['legacy_group'])]:
        names=[n for _,v,e in feature_columns(kind) for n in [v,e]]+['MW_EBV']
        texts=serialize_rows(dict(features=data[f'{domain}_features'],group=group),dict(feature_names=names),catalogue=kind,level=1)
        assert all('pivot_nm' not in t and 'width_nm' not in t for t in texts)
        path=rep/f'{domain}_inputs.jsonl'
        with path.open('w') as f:
            for tid,text in zip(data['targetid'],texts):f.write(json.dumps(dict(targetid=str(tid),text=text))+'\n')
        input_hashes[domain]=sha(path);tokens[domain]=audit_tokens(tokenizer,texts,512)
    write(rep/'input_contract.json',dict(version='matched_level1_v1',precision=5,unit='AB_nanomaggy',sigma='one_sigma',missing='NA; valid signed Legacy flux retained',
        schema='Common existing Level-1 template, identical field structure; real survey, cmodel/Tractor, filter/instrument identifiers and EBV provenance differ',
        ordering='Header, five bands in fixed increasing nominal wavelength order for each true system, each with flux/sigma, then EBV',
        extra_physical_metadata=False,registry_read_only=True,registry_sha256=sha(DEFAULT_REGISTRY),tokens=tokens,input_sha256=input_hashes))
    config_folder=out/'training_configs';config_folder.mkdir(exist_ok=True)
    for seed in SEEDS:
        state=torch.load(out/'model_initialisation'/f'head_seed{seed}.pt',map_location='cpu',weights_only=True)
        for method in ['qlora','dora']:
            for domain in ['hsc','legacy']:
                cfg=dict(version='matched_training_v1',domain=domain,method=method,seed=seed,initial_adapter=None,
                    initial_head=str((out/'model_initialisation'/f'head_seed{seed}.pt').resolve()),initial_head_tensor_sha256=state_fingerprint(state),
                    cohort=str((split/'paired_cohort.npz').resolve()),cohort_sha256=sha(split/'paired_cohort.npz'),
                    input_path=str((rep/f'{domain}_inputs.jsonl').resolve()),input_sha256=input_hashes[domain],
                    presentation_order_sha256=sha(out/'model_initialisation'/f'presentation_order_seed{seed}.npy'),
                    output_dir=str((out/'future_runs'/f'{domain}-{method}-seed{seed}').resolve()),checkpoint_interval_updates=100,
                    training=dict(model_path=str(base),max_length=512,pooling='last',n_z_bins=300,z_min=0.,z_max=4.5,head_hidden_dim=256,
                        epochs=10,head_warmup_epochs=3,batch_size=1,eval_batch_size=8,gradient_accumulation_steps=16,
                        learning_rate=2e-4,lora_learning_rate=1e-5,weight_decay=.01,warmup_fraction=.03,
                        max_grad_norm=1.,lora_max_grad_norm=.1,lora_rank=8,lora_alpha=16,lora_dropout=.05,lora_target_modules='q_proj,k_proj,v_proj,o_proj',seed=seed,device='cuda',local_files_only=True))
                write(config_folder/f'{domain}-{method}-seed{seed}.draft.json',cfg)
    evaluation=dict(version='matched_evaluation_v1',created_utc=now(),seeds=SEEDS,pilot_seed=42,counts=counts,
        split_membership_sha256={p.name:sha(p) for p in split.glob('*membership.npz')},
        evaluation='Locked previously analysed P; same IDs/order/reference z in every cell; no held-out tuning or calibration',
        representation='Common Level-1 only; physical filter IDs, native flux/sigma, EBV; no pivots or widths',
        grid=dict(n_bins=300,z_min=0.,z_max=4.5,probability='Softmax mass per bin; CE bucketize left boundary convention inherited from evaluator'),
        source_validation='Only own-source V CE, end of each epoch, common cadence; save best and fixed-final-budget states separately. No automatic P evaluation during training.',
        budget=dict(epochs=10,head_warmup_epochs=3,updates_per_epoch=1038,total_updates=10380,head_only_updates=3114,joint_updates=7266,
                    total_examples=166000,effective_batch=16,epoch_final_batch=8,partial_batch='Loss divided by actual eight examples, not nominal sixteen'),
        budget_justification='One-seed four-model implementation/convergence pilot at an evidence-backed historical 3-head+7-joint schedule; original native HSC/Legacy curves still improve late. Fresh Level-1 convergence is not established. Any future common-budget revision/three-seed expansion uses source validation only before locked-P scoring, never whichever transfer result is favourable.',
        bootstrap=dict(replicates=1000,seed=1028,unit='paired physical galaxy',shared_indices=True,confidence='Percentile 95%, directly on paired differences, conditional on trained checkpoint'),
        seed_analysis='Report every matched seed, then means and sample dispersion; separate galaxy-bootstrap uncertainty from training-seed variability; do not assert QLoRA/DoRA stochastic bitwise equivalence',
        redshift_edges=BINS,aggregation=['native population-weighted','equal-bin macro average, including explicitly labelled macro NMAD'],
        metrics=['CE','CRPS','NMAD','outlier fraction abs(delta)>0.15','mean/median bias','68% coverage','PIT','per-object normalized residual','new/rescued catastrophic counts'],
        contrasts=dict(P_HL='CE(H,L)-CE(H,H)',P_LH='CE(L,H)-CE(L,L)',G_HL='CE(H,L)-CE(L,L)',G_LH='CE(L,H)-CE(H,H)',
            method_robustness='P_DoRA-P_QLoRA for each direction, plus absolute target CE differences',interaction='[CE(H,L)-CE(H,H)]-[CE(L,L)-CE(L,H)]'),
        primary_selected_state='Source-V selected checkpoint; fixed-final-budget state is a separately labelled secondary trajectory control',
        caveats='Photometric observational-system transfer, not pure filter transmission; pipelines, depths, noise, extinction/calibration and grizy versus grzW1W2 information content differ. No strict high-z extrapolation; P has 400 objects above z=1 and none above z=2.',
        full_training_started=False,original_model_full_inference_started=False)
    write(out/'evaluation_policy/evaluation_policy_v1.json',evaluation)
    print('MATCHED DATA AND DRAFT CONFIGS PREPARED',counts,flush=True)


def replay(out,method):
    import torch
    import qwen_survey_posttraining as pipeline
    from filter_metadata_counterfactual import load_model
    from qwen_zero_shot_sanity import reserve_one_gpu,evaluate
    from aion_magnitude.metrics import summarize_pdf_metrics
    inv=json.loads((out/'legacy_checkpoint_audit/inventory.json').read_text())['models'][method]
    source=read(out/'legacy_checkpoint_audit/recovered_cohort/cohort.npz');ix=np.flatnonzero(source['split']=='val')[:64]
    data={k:v[ix] for k,v in source.items()};meta=json.loads((Path(inv['run'])/'prepared.json').read_text())
    texts=pipeline.texts_for(argparse.Namespace(catalogue_kind='desi-ls'),data,meta)
    manifest=dict(models={method:dict(config=inv['training_config'],directory=inv['selected_directory'])},base_model=inv['base_model'])
    ref=read(out/'legacy_checkpoint_audit'/f'{method}_validation_reference.npz');np.testing.assert_array_equal(data['targetid'],ref['targetid'][:64])
    with reserve_one_gpu() as gpu:
        assert gpu['memory_before_gib']['used']<.25
        model,tokenizer=load_model(manifest,method)
        lengths=[len(tokenizer(t,truncation=False)['input_ids']) for t in texts];assert max(lengths)<=512
        started=time.time();ev=evaluate(model,tokenizer,data,texts,inv['training_config']);torch.cuda.synchronize();seconds=time.time()-started
        diffs={k:float(np.max(np.abs(ev[k].numpy()-ref[k][:64]))) for k in ['logits','pz','z_p16','z_p50','z_p84']}
        exact=all(v==0 for v in diffs.values())
        baseline={k:torch.from_numpy(ref[k][:64] if not k.startswith('redshift_') else ref[k]) for k in ref if k!='targetid'}
        result=dict(created_utc=now(),n=64,cohort='Original source validation, no paired P scoring',gpu=gpu,
            exact_numerical_replay=exact,max_abs_difference=diffs,metrics=summarize_pdf_metrics(ev),reference_metrics=summarize_pdf_metrics(baseline),seconds=seconds,
            input_tokens_min_max=[min(lengths),max(lengths)],input_texts=texts,serializer=inv['serialization'])
        write(out/'legacy_checkpoint_audit'/f'{method}_validation_replay.json',result)
        assert exact,'Replay mismatch recorded; investigate historical evaluator before full inference'
    print('LEGACY REPLAY EXACT',method,flush=True)


def initialization_smoke(out,seed):
    import torch
    from aion_magnitude.matched_training import build_fresh_model,state_fingerprint
    from qwen_zero_shot_sanity import reserve_one_gpu
    with reserve_one_gpu() as gpu:
        assert gpu['memory_before_gib']['used']<.25
        for method in ['qlora','dora']:
            hashes={};counts={};timings={};memory={}
            for domain in ['hsc','legacy']:
                cfg=json.loads((out/'training_configs'/f'{domain}-{method}-seed{seed}.draft.json').read_text())
                torch.cuda.reset_peak_memory_stats()
                model,tokenizer,device=build_fresh_model(cfg)
                state={k:p.detach().cpu() for k,p in model.named_parameters() if p.requires_grad};hashes[domain]=state_fingerprint(state)
                counts[domain]=dict(head=sum(p.numel() for p in model.photoz_head.parameters()),adapter=sum(p.numel() for p in model.qwen.parameters() if p.requires_grad))
                data=read(cfg['cohort']);train=np.flatnonzero(data['category']=='T')[:8]
                records=[json.loads(line) for line in Path(cfg['input_path']).open()];texts=[records[i]['text'] for i in train]
                model.eval();batch=tokenizer(texts,padding=True,truncation=False,return_tensors='pt');assert batch['input_ids'].shape[1]<=512
                batch={k:v.to(device) for k,v in batch.items()}
                with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    logits=model(**batch);assert logits.shape==(8,300) and torch.isfinite(logits).all()
                    torch.cuda.synchronize();start=time.time()
                    for _ in range(3):model(**batch)
                    torch.cuda.synchronize();timings[domain]=(time.time()-start)/24
                    memory[domain]=torch.cuda.max_memory_allocated()/2**30
                path=out/'model_initialisation'/f'{method}-seed{seed}';path.mkdir(exist_ok=True)
                if domain=='hsc':torch.save(state,path/'canonical_initial_trainable_state.pt')
                del state,model,tokenizer,batch,logits;gc.collect();torch.cuda.empty_cache()
            assert hashes['hsc']==hashes['legacy'] and counts['hsc']==counts['legacy']
            write(out/'model_initialisation'/f'{method}-seed{seed}'/'proof.json',dict(created_utc=now(),seed=seed,method=method,gpu=gpu,
                source_initial_trainable_tensor_sha256=hashes,identical_source_initial_states=True,parameter_counts=counts,
                seconds_per_object_batch8=timings,peak_allocated_gib_forward_only=memory,forward_only=True,optimizer_updates=0,test_evaluated=False))
    print('FRESH INITIALIZATION VERIFIED',seed,flush=True)


def train(out,config_path,resume):
    from aion_magnitude.matched_training import build_fresh_model,train_loop,state_fingerprint
    from qwen_zero_shot_sanity import reserve_one_gpu
    config=json.loads(config_path.read_text());assert config_path.name.endswith('.json') and '.draft.' not in config_path.name
    package=json.loads((out/'manifest.json').read_text())
    import importlib.metadata
    for name,expected in package['environment_versions'].items():assert importlib.metadata.version(name)==expected,'Prepared environment changed'
    assert sha(config_path)==package['final_config_sha256'][str(config_path.resolve())]
    for file,expected in package['code_sha256'].items():assert sha(ROOT/file)==expected,'Prepared code changed; review/version package before training'
    base_identity=json.loads(Path(config['base_identity_path']).read_text());assert sha(config['base_identity_path'])==config['base_identity_sha256']
    for file,info in base_identity['files'].items():assert sha(file)==info['sha256'],'Verified pre-adaptation base changed'
    assert sha(config['cohort'])==config['cohort_sha256'] and sha(config['input_path'])==config['input_sha256']
    records=[json.loads(line) for line in Path(config['input_path']).open()];data=read(config['cohort'])
    np.testing.assert_array_equal(np.array([int(r['targetid']) for r in records],np.int64),data['targetid'])
    t=np.flatnonzero(data['category']=='T');v=np.flatnonzero(data['category']=='V')
    run=Path(config['output_dir'])
    if run.exists() and not resume:raise FileExistsError('New full training requires a fresh run or explicit --resume')
    with reserve_one_gpu():
        model,tokenizer,device=build_fresh_model(config)
        initial=state_fingerprint({k:p for k,p in model.named_parameters() if p.requires_grad})
        assert initial==config['expected_initial_trainable_tensor_sha256']
        train_loop(model,tokenizer,device,[records[i]['text'] for i in t],data['z'][t],data['targetid'][t],
            [records[i]['text'] for i in v],data['z'][v],data['targetid'][v],config,run,resume=resume)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['inventory','exposure','prepare','replay','initialization-smoke','finalize','report','train','frozen-inference'])
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--method',choices=['qlora','dora']);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--config',type=Path);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.output_dir
    if args.stage=='inventory':inventory(out)
    elif args.stage=='exposure':exposure(out)
    elif args.stage=='prepare':prepare(out)
    elif args.stage=='replay':replay(out,args.method)
    elif args.stage=='initialization-smoke':initialization_smoke(out,args.seed)
    elif args.stage=='train':train(out,args.config,args.resume)
    else:
        from task_1028_package import finalize,report,frozen_inference
        if args.stage=='finalize':finalize(out)
        elif args.stage=='report':report(out)
        else:frozen_inference(out,args.config)


if __name__=='__main__':main()
