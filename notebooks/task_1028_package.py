"""Finalize and report the audited matched-training package; no auto training."""
from pathlib import Path
import json
import importlib.metadata
import shlex
import shutil

import numpy as np

from task_1028 import ROOT,BENCH,RUNS,BINS,SEEDS,read,sha,write,now


def verify_paired_measurements(out):
    from aion_magnitude.survey_catalogues import flux_features
    from hsc_legacy_paired_benchmark import separation
    d=read(out/'paired_split/paired_cohort.npz');h=read(BENCH/'raw-audit/hdf_metadata.npz');p=read(BENCH/'raw-audit/tractor_matches.npz');s=read(BENCH/'raw-audit/redshift_matches.npz')
    def indices(ids):
        lookup={int(v):i for i,v in enumerate(ids)}
        return np.array([lookup[int(t)] for t in d['targetid']])
    hi=indices(h['targetid']);pi=indices(p['TARGETID'])
    text=lambda name:np.char.strip(s[name].astype('U'))
    q=(s['ZCAT_PRIMARY']!=0)&(text('SPECTYPE')=='GALAXY')&(text('OBJTYPE')=='TGT')&(s['ZWARN']==0)&(s['COADD_FIBERSTATUS']==0)&np.isfinite(s['DELTACHI2'])&(s['DELTACHI2']>40)&np.isfinite(s['Z'])&(s['Z']>=0)&(s['Z']<=4.5)&np.isfinite(s['ZERR'])&(s['ZERR']>=0)&(text('SURVEY')=='main')&np.isin(text('PROGRAM'),['bright','dark'])
    source=np.flatnonzero(q);assert len(np.unique(s['TARGETID'][source]))==len(source);zi=source[indices(s['TARGETID'][source])]
    np.testing.assert_equal(d['hsc_features'],h['hsc_features'][hi]);np.testing.assert_array_equal(d['hsc_valid'],h['hsc_valid'][hi])
    pairs=[flux_features(p['FLUX_'+b][pi],p['FLUX_IVAR_'+b][pi],p['NOBS_'+b][pi]) for b in ['G','R','Z','W1','W2']]
    reconstructed=np.column_stack([x for pair in pairs for x in pair[:2]]+[p['EBV'][pi]]).astype(np.float32)
    valid=np.column_stack([pair[2] for pair in pairs]);np.testing.assert_equal(d['legacy_features'],reconstructed);np.testing.assert_array_equal(d['legacy_valid'],valid)
    assert np.all(valid.sum(1)>=3) and np.all(d['hsc_valid'].sum(1)>=3)
    np.testing.assert_array_equal(d['z'],h['z_hdf'][hi].astype(np.float32));np.testing.assert_array_equal(d['z_iron'],s['Z'][zi])
    assert np.all(np.abs(h['z_hdf'][hi]-s['Z'][zi])<=1e-6*(1+np.abs(s['Z'][zi])))
    distance=separation(d['ra'],d['dec'],d['legacy_ra'],d['legacy_dec']);assert np.all(distance<=1.)
    legacy_group=np.array([f'ls:{r}:{b}:{o}' for r,b,o in zip(p['RELEASE'][pi],p['BRICKID'][pi],p['OBJID'][pi])])
    np.testing.assert_array_equal(d['legacy_group'],legacy_group)
    np.testing.assert_array_equal(d['galaxy_id'],np.array([f'hsc:{v}' for v in h['hsc_object_id'][hi]]))
    write(out/'paired_split/eligibility_recheck.json',dict(created_utc=now(),n=len(hi),all_raw_metadata_rows_matched=True,
        hsc_features_and_validity_exact=True,original_tractor_flux_ivar_nobs_ebv_reconstruction_exact=True,
        shared_spectroscopic_quality_passed=True,native_HDF_labels_exact=True,native_Iron_agreement_tolerance=1e-6,
        max_pair_separation_arcsec=float(distance.max()),imaging_identities_exact=True,
        rows_by_split={c:int((d['category']==c).sum()) for c in ['T','V','P']},
        raw_metadata_sha256={str(path.resolve()):sha(path) for path in [BENCH/'raw-audit/hdf_metadata.npz',BENCH/'raw-audit/tractor_matches.npz',BENCH/'raw-audit/redshift_matches.npz']},
        quality_scope='DR1 Iron quality, same as established policy; HDF-specific fiber-status discrepancies remain documented and are not silently used as a new cut',
        images_or_spectral_pixels_read=False))


def verify_raw_backreferences(out):
    import fitsio
    from aion_magnitude.survey_catalogues import flux_features
    folder=out/'legacy_checkpoint_audit/recovered_cohort';data=read(folder/'cohort.npz');meta=json.loads((folder/'cohort_metadata.json').read_text())
    rng=np.random.default_rng(1028);checked=[]
    columns=['TARGETID','RA','DEC','EBV']+[f'{prefix}_{band}' for band in ['G','R','Z','W1','W2'] for prefix in ['FLUX','FLUX_IVAR','NOBS']]
    for split in ['train','val','test']:
        for i in rng.choice(np.flatnonzero(data['split']==split),size=8,replace=False):
            path=meta['sources'][int(data['source_file_index'][i])+1]['path']
            with fitsio.FITS(path) as file:row=file[1].read(rows=[int(data['source_row'][i])],columns=columns)
            assert int(row['TARGETID'][0])==int(data['targetid'][i])
            assert row['RA'][0]==data['ra'][i] and row['DEC'][0]==data['dec'][i]
            pairs=[flux_features(row['FLUX_'+b],row['FLUX_IVAR_'+b],row['NOBS_'+b]) for b in ['G','R','Z','W1','W2']]
            features=np.column_stack([x for pair in pairs for x in pair[:2]]+[row['EBV']]).astype(np.float32)
            np.testing.assert_equal(features[0],data['features'][i]);checked.append(dict(split=split,targetid=str(data['targetid'][i]),file=path,row=int(data['source_row'][i])))
    write(folder/'raw_row_backreference_check.json',dict(created_utc=now(),seed=1028,checked=checked,all_identifiers_coordinates_features_exact=True))


def code_files():
    return [ROOT/p for p in ['notebooks/task_1028.py','notebooks/task_1028_package.py','notebooks/task_1028_report.py','notebooks/recover_legacy_cohort_1028.py',
        'notebooks/launch_matched_training_1028.py','aion_magnitude/matched_training.py','aion_magnitude/filter_photometry.py',
        'aion_magnitude/qwen_posttraining.py','aion_magnitude/FM_Qwen.py','aion_magnitude/metrics.py','aion_magnitude/models.py','aion_magnitude/utils.py',
        'notebooks/filter_metadata_counterfactual.py','tests/test_matched_training_1028.py','pixi.toml','pixi.lock']]


def finalize(out):
    import torch
    from aion_magnitude.matched_training import state_fingerprint
    if (out/'manifest.json').exists():raise FileExistsError('Prepared package is immutable')
    verify_paired_measurements(out)
    verify_raw_backreferences(out)
    inventory=json.loads((out/'legacy_checkpoint_audit/inventory.json').read_text());recovery=json.loads((out/'legacy_checkpoint_audit/recovered_cohort/recovery_proof.json').read_text())
    models=inventory['models']
    for method in ['qlora','dora']:
        proof=json.loads((out/'legacy_checkpoint_audit'/f'{method}_validation_replay.json').read_text());assert proof['exact_numerical_replay']
        models[method]=dict(models[method],validation_replay='Exact logits/PDFs/quantiles on 64 original source-validation objects',
            source_ID_recovery='Verified deterministic reconstruction; all archived ordered validation/test IDs and labels match',ready_for_matched_inference=True)
    write(out/'legacy_checkpoint_audit/verified_inventory.json',dict(created_utc=now(),models=models,recovery_proof_sha256=sha(out/'legacy_checkpoint_audit/recovered_cohort/recovery_proof.json')))
    configs={};counts={};initialization={}
    for seed in SEEDS:
        head=torch.load(out/'model_initialisation'/f'head_seed{seed}.pt',weights_only=True,map_location='cpu')
        assert all(state_fingerprint(head)!=m['head_tensor_sha256'] for m in models.values()),'A historical trained head was restored'
        for method in ['qlora','dora']:
            p=json.loads((out/'model_initialisation'/f'{method}-seed{seed}'/'proof.json').read_text())
            assert p['identical_source_initial_states'] and p['optimizer_updates']==0
            initialization[f'{method}-seed{seed}']=p
            counts[method]=p['parameter_counts']['hsc']
            for domain in ['hsc','legacy']:
                name=f'{domain}-{method}-seed{seed}'
                config=json.loads((out/'training_configs'/f'{name}.draft.json').read_text())
                config['expected_initial_trainable_tensor_sha256']=p['source_initial_trainable_tensor_sha256'][domain]
                config['base_identity_path']=str((out/'model_initialisation/base_model_identity.json').resolve())
                config['base_identity_sha256']=sha(out/'model_initialisation/base_model_identity.json')
                write(out/'training_configs'/f'{name}.json',config);configs[name]=config
    assert counts['dora']['adapter']-counts['qlora']['adapter']==102400
    write(out/'model_initialisation/summary.json',dict(created_utc=now(),all_source_branches_identical_within_seed_method=True,
        shared_heads_across_all_four_branches=True,parameter_counts=counts,dora_additional_trainables=102400,
        seeds=SEEDS,seed_streams=dict(head='seed',adapter='seed+10000',order='seed+20000+epoch',dropout='seed+30000'),
        caveat='Identical source-branch initialization is verified. Different methods and different token sequence lengths can consume randomness differently; stochastic trajectories are not claimed bitwise equivalent.'))
    rng=np.random.default_rng(1028);bootstrap=np.lib.format.open_memmap(out/'evaluation_policy/bootstrap_indices.npy',mode='w+',dtype=np.uint32,shape=(1000,61696))
    for i in range(1000):bootstrap[i]=rng.integers(0,61696,size=61696,dtype=np.uint32)
    bootstrap.flush();del bootstrap
    write(out/'evaluation_policy/bootstrap_identity.json',dict(seed=1028,draws=1000,n=61696,sha256=sha(out/'evaluation_policy/bootstrap_indices.npy'),shared_across_all_cells_and_seeds=True))
    exposure=json.loads((out/'original_model_exposure/audit.json').read_text())
    original_cfgs={};hsc_manifest=json.loads((BENCH/'manifest.json').read_text())
    for source in ['legacy','hsc']:
        for method in ['qlora','dora']:
            if source=='legacy':
                info=models[method];directory=info['selected_directory'];settings=info['training_config'];protected=info['sha256'];base=info['base_model']
            else:
                info=hsc_manifest['models'][f'original-{method}'];directory=info['directory'];settings=info['config'];protected=info['sha256'];base=hsc_manifest['base_model']
            for domain in ['hsc','legacy']:
                name=f'original-{source}-{method}-{domain}'
                config=dict(kind='historical_frozen',model=f'{source}-{method}',directory=directory,training=settings,base_model=base,serializer_level=0,
                    domain=domain,cohort=str((out/'original_model_exposure/mutual_cohort.npz').resolve()),cohort_sha256=sha(out/'original_model_exposure/mutual_cohort.npz'),
                    n=exposure['mutual_P'],protected_sha256=protected,output=str((out/'future_original_inference'/name).resolve()),
                    evaluation_scope='Mutually independent of all seven historical models; previous held-out test evaluation acknowledged',expensive_full_inference_executed=False)
                write(out/'original_model_exposure'/f'{name}.json',config);original_cfgs[name]=config
    for name,config in configs.items():
        for state in ['selected','final']:
            for domain in ['hsc','legacy']:
                plan=dict(kind='new_matched_frozen',model=name,checkpoint_state=state,domain=domain,serializer_level=1,
                    checkpoint_metadata=str((Path(config['output_dir'])/state/'metadata.json').resolve()),base_model=config['training']['model_path'],
                    training=config['training'],cohort=config['cohort'],cohort_sha256=config['cohort_sha256'],n=61696,
                    output=str((out/'future_matched_inference'/f'{name}-{state}-{domain}').resolve()),
                    evaluation_scope='Locked previously analysed P only; new fresh-base models, identical training membership',expensive_full_inference_executed=False)
                write(out/'evaluation_policy'/f'infer-{name}-{state}-{domain}.json',plan)
    timings=json.loads((out/'feasibility/historical_telemetry.json').read_text());per_model={};training=0.;inference=0.
    for method in ['qlora','dora']:
        t=timings[method]['summary'];head=t['head_warmup']['seconds_per_update_median'];joint=t['joint']['seconds_per_update_median']
        values=[p['seconds_per_object_batch8'][domain] for key,p in initialization.items() if key.startswith(method) for domain in ['hsc','legacy']]
        fwd=float(np.median(values));val=4257*10*fwd/3600
        hours=(3114*head+7266*joint)/3600
        per_model[method]=dict(head_warmup_seconds_per_update=head,joint_seconds_per_update=joint,
            head_training_proxy_MIG_GPU_hours=3114*head/3600,joint_training_proxy_MIG_GPU_hours=7266*joint/3600,
            source_validation_MIG_GPU_hours=val,training_plus_validation_MIG_GPU_hours=hours+val,
            measured_Level1_batch8_seconds_per_object=fwd,
            caveat='Training proxy from actual native-input historical checkpoint intervals; new Level-1 gradient throughput was not measured because Qwen training is not authorised. IQR describes recorded timing variation, not uncertainty in this representation transfer.')
        training+=2*(hours+val);inference+=4*61696*fwd/3600
    q=per_model['qlora']['training_plus_validation_MIG_GPU_hours'];d=per_model['dora']['training_plus_validation_MIG_GPU_hours']
    checkpoint_bytes={method:Path(models[method]['last_resumable_checkpoint']['path']).stat().st_size for method in ['qlora','dora']}
    selected_bytes={method:sum(Path(p).stat().st_size for p in m['sha256'] if p.endswith(('adapter_model.safetensors','photoz_head.pt','adapter_config.json'))) for method,m in models.items()}
    storage_run={method:2*checkpoint_bytes[method]+2*selected_bytes[method]+2*(out/'model_initialisation'/f'{method}-seed42/canonical_initial_trainable_state.pt').stat().st_size+2*4257*300*8 for method in ['qlora','dora']}
    # use the measured historical file sizes; the last head state variable is no longer used below.
    storage_weights=2*sum(storage_run.values());prediction_bytes=8*61696*300*8
    cost=dict(created_utc=now(),units='MIG-GPU hours on comparable H100 1g.12gb slices; not full-H100-equivalent hours',per_model=per_model,
        one_seed_four_models_training_validation=training,three_seeds_twelve_models_training_validation=3*training,
        additional_two_seeds_training_validation=2*training,
        selected_state_paired_inference_one_seed=inference,selected_and_final_inference_one_seed=2*inference,
        one_seed_total_with_selected_inference=training+inference,three_seed_total_with_selected_inference=3*(training+inference),
        three_free_slices_one_seed_training_queue_hours=max(2*q,d),three_free_slices_twelve_models_training_queue_hours=2*q+2*d,
        scheduling='Two DoRA branches plus one QLoRA start; second QLoRA follows on its released slice. For twelve branches, six DoRA jobs then six QLoRA jobs balance four waves over three slices. Existing four CLAUDS slices are not counted.',
        storage=dict(measured_historical_resume_bytes=checkpoint_bytes,measured_adapter_head_bytes=selected_bytes,
            projected_bytes_per_model_run=storage_run,one_seed_four_runs_bytes=storage_weights,three_seed_twelve_runs_bytes=3*storage_weights,
            selected_state_raw_prediction_bytes_per_seed=prediction_bytes,selected_and_final_raw_prediction_bytes_per_seed=2*prediction_bytes,
            convention='Uncompressed float32 logits+probability mass dominate prediction estimate; NPZ often smaller, no assumed compression ratio. Two rotating full resume slots, selected/final exports and validation PDFs retained; base checkpoint is shared.'))
    write(out/'feasibility/cost_estimate.json',cost)
    commands=[]
    for seed in SEEDS:
        commands.append(f'PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/launch_matched_training_1028.py --package {shlex.quote(str(out.resolve()))} --seeds {seed}')
    commands.append(f'PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/launch_matched_training_1028.py --package {shlex.quote(str(out.resolve()))} --seeds 42 43 44')
    (out/'launch_commands.txt').write_text('# Prepared only. Full Qwen training needs a separate user go-ahead.\n'+'\n'.join(commands)+'\n\n# Original-model inference: separately authorise before execution.\n'+
        '\n'.join(f'CUDA_VISIBLE_DEVICES=MIG-1bba1f26-8b2b-5b97-9bb1-c886ab9bb8b7 PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/task_1028.py frozen-inference --output-dir {shlex.quote(str(out.resolve()))} --config {shlex.quote(str((out/"original_model_exposure"/(name+".json")).resolve()))}' for name in original_cfgs)+'\n')
    files=code_files()
    manifest=dict(version='task_1028_matched_package_v1',created_utc=now(),training_launched=False,full_frozen_inference_launched=False,
        specification_sha256=sha(ROOT/'task_1028.md'),benchmark_manifest_sha256=sha(BENCH/'manifest.json'),
        original_artifact_sha256=inventory['original_artifact_sha256'],
        recovered_cohort_sha256=sha(out/'legacy_checkpoint_audit/recovered_cohort/cohort.npz'),
        paired_cohort_sha256=sha(out/'paired_split/paired_cohort.npz'),
        final_config_sha256={str((out/'training_configs'/f'{name}.json').resolve()):sha(out/'training_configs'/f'{name}.json') for name in configs},
        evaluation_policy_sha256=sha(out/'evaluation_policy/evaluation_policy_v1.json'),
        immutable_data_sha256={str(p.resolve()):sha(p) for base in [out/'paired_split',out/'representation_audit',out/'model_initialisation'] for p in base.rglob('*') if p.is_file()},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    manifest['environment_versions']={name:importlib.metadata.version(name) for name in ['torch','transformers','peft','bitsandbytes','numpy']}
    write(out/'manifest.json',manifest)
    for p in files+[ROOT/'task_1028.md']:
        dest=out/'code_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    print('RUN-READY PACKAGE FROZEN; NO QWEN TRAINING LAUNCHED',flush=True)


def frozen_inference(out,config_path):
    """Runnable plan only until a separate explicit full-inference request."""
    import torch
    from filter_metadata_counterfactual import load_model
    from qwen_zero_shot_sanity import reserve_one_gpu,evaluate,save_predictions
    from aion_magnitude.filter_photometry import feature_columns,serialize_rows
    cfg=json.loads(config_path.read_text());data=read(cfg['cohort']);assert sha(cfg['cohort'])==cfg['cohort_sha256']
    if cfg['kind']=='new_matched_frozen':
        ix=np.flatnonzero(data['category']=='P');data={k:v[ix] for k,v in data.items()}
        meta=json.loads(Path(cfg['checkpoint_metadata']).read_text());directory=Path(cfg['checkpoint_metadata']).parent
        protected={str(directory/p):v for p,v in meta['checkpoint_sha256'].items()}
    else:directory=Path(cfg['directory']);protected=cfg['protected_sha256']
    for file,expected in protected.items():assert sha(file)==expected
    assert len(data['z'])==cfg['n']
    domain=cfg['domain'];kind='galaxiesml-spec' if domain=='hsc' else 'desi-ls'
    names=[n for _,v,e in feature_columns(kind) for n in [v,e]]+['MW_EBV']
    d=dict(features=data[f'{domain}_features'],group=data['galaxy_id'] if domain=='hsc' else data['legacy_group'])
    texts=serialize_rows(d,dict(feature_names=names),catalogue=kind,level=cfg['serializer_level'])
    folder=Path(cfg['output']);folder.mkdir(parents=True,exist_ok=True)
    m=dict(base_model=cfg['base_model'],models={'model':dict(directory=str(directory),config=cfg['training'])})
    with reserve_one_gpu():
        model,tokenizer=load_model(m,'model')
        if cfg['kind']=='new_matched_frozen':
            source=read(cfg['cohort']);vi=np.flatnonzero(source['category']=='V')[:64]
            source_config=meta['config'];records=[json.loads(line) for line in Path(source_config['input_path']).open()]
            assert sha(source_config['input_path'])==source_config['input_sha256']
            check=dict(targetid=source['targetid'][vi],z=source['z'][vi])
            reference=torch.load(directory/'validation_predictions.pt',map_location='cpu',weights_only=False)
            np.testing.assert_array_equal(np.asarray(reference['object_id'][:64],np.int64),check['targetid'])
            ev=evaluate(model,tokenizer,check,[records[i]['text'] for i in vi],cfg['training'])
            differences={key:float(np.max(np.abs(ev[key].numpy()-reference[key][:64].numpy()))) for key in ['logits','pz','z_p16','z_p50','z_p84']}
            write(folder/'source_validation_replay.json',dict(n=64,max_abs_difference=differences,exact=all(v==0 for v in differences.values())))
            assert all(v==0 for v in differences.values()),'New-model source-validation replay failed before any P inference'
        assert max(len(x) for x in tokenizer(texts,truncation=False)['input_ids'])<=cfg['training']['max_length']
        for start in range(0,len(texts),256):
            end=min(start+256,len(texts));path=folder/f'{start:06d}_{end:06d}.npz'
            if path.exists():np.testing.assert_array_equal(read(path)['targetid'],data['targetid'][start:end]);continue
            small=dict(targetid=data['targetid'][start:end],z=data['z'][start:end],features=d['features'][start:end])
            ev=evaluate(model,tokenizer,small,texts[start:end],cfg['training']);tmp=path.with_name(path.stem+'.partial.npz');save_predictions(tmp,small,ev);tmp.replace(path)
    for file,expected in protected.items():assert sha(file)==expected
    write(folder/'complete.json',dict(finished_utc=now(),config=cfg,checkpoint_hashes_unchanged=True,n=len(texts)))


def report(out):
    from task_1028_report import report as write_report
    write_report(out)
