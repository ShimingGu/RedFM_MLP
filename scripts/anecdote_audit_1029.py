"""Verify actual compatible checkpoints and reconstruct supervised exposures."""
import json,time
from pathlib import Path
import numpy as np
import torch
import fitsio
from anecdote_common_1029 import *
def run():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'checkpoint_provenance.json').exists()
    protected={}
    for p in (ROOT/'reports').rglob('*'):
        if p.is_file() and 'stage4' not in p.parts:protected[str(p)]=d.sha(p)
    for folder in ['configs','aion_magnitude']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:protected[str(p)]=d.sha(p)
    d.write(OUT/'protected_before.json',protected)
    from safetensors import safe_open
    cache=OUT/'base_content_hashes.json'
    basefiles=json.loads(cache.read_text()) if cache.exists() else {p.name:d.sha(p) for p in BASE.iterdir() if p.is_file()}
    d.write(cache,basefiles)
    config=json.loads((BASE/'config.json').read_text());pro=dict(base_model='Qwen3.5-4B-Base',base_path=str(BASE),base_files_sha256=basefiles,hidden_dimension=config['text_config']['hidden_size'],families={},selected='One rank8 QLoRA checkpoint per family; extra DoRA arms deliberately omitted before downstream results. 0.8B checkpoints incompatible with the primary4B base.',historical_base_revision='Adapter revision fields are null. Current full content hashes, matching local base path/config, historical BS file fingerprints, adapter shapes and native replay establish the executable contract; no invented upstream revision SHA.')
    results={};states={}
    for family,folder in [('BS',BS),('BC',BC)]:
        r=torch.load(folder/'result.pt',map_location='cpu',weights_only=False,mmap=True);results[family]=r
        c=json.loads((folder/'adapter/adapter_config.json').read_text());assert Path(c['base_model_name_or_path']).resolve()==BASE.resolve();assert c['r']==8 and c['lora_alpha']==16 and not c['use_dora'];assert r['metadata']['config']['model_path']=='Qwen3.5-4B-Base'
        with safe_open(str(folder/'adapter/adapter_model.safetensors'),framework='pt',device='cpu') as f:states[family]={k:tuple(f.get_slice(k).get_shape()) for k in f.keys()}
        files={str(p.resolve()):d.sha(p) for p in (folder/'adapter').glob('*') if p.is_file()};files[str((folder/'photoz_head.pt').resolve())]=d.sha(folder/'photoz_head.pt');files[str((folder/'result.pt').resolve())]=d.sha(folder/'result.pt')
        for p,h in files.items():protected[p]=h
        pro['families'][family]=dict(directory=str(folder.resolve()),metadata=r['metadata'],adapter_config=c,files_sha256=files,tensor_shapes=states[family],training_labels='DESI spectroscopic native HDF redshifts' if family=='BS' else 'Historical Phosphoros ZPHOT, not spectroscopic',historical_input='native signed grizy flux/error+EBV, five decimals, max512' if family=='BS' else '11 median-imputed magnitudes using historical ZP23 convention, five decimals, max2048',original_grid=[r['metadata']['config']['z_min'],r['metadata']['config']['z_max']],original_pooling='last nonpadding token; faithful native replay audited separately')
    assert states['BS']==states['BC'],'Adapter architectures differ'
    meta=json.loads((ROOT/'reports/stage0/1027a/manifest.json').read_text());finger=meta['base_model_source']['files']
    for f in finger:
        st=Path(f['path']).stat();assert st.st_size==f['bytes'] and st.st_mtime_ns==f['mtime_ns']
    pro['historical_BS_base_fingerprints_match']=True
    # Recover BC identity sampling/partitions; inspect no archived test predictions.
    from aion_magnitude.clauds_bands import select_catalogue_row_indices
    from aion_magnitude.dataset import make_random_split
    with fitsio.FITS(str(OLD_CAT)) as f:
        rr=select_catalogue_row_indices(f[1].get_nrows(),max_rows=300000,sample_mode='random',seed=42);a=f[1].read(rows=rr,columns=['ID','RA','DEC','ZPHOT'])
    good=np.isfinite(a['ZPHOT'])&(a['ZPHOT']!=-99);a=a[good];rr=rr[good];role=make_random_split(len(a),seed=42).astype('U12');r=results['BC'];vi=ids(r['val_evaluation']['object_id']);ti=ids(r['test_evaluation']['object_id'])
    assert np.array_equal(ids(a['ID'][role=='val']),vi);assert np.array_equal(ids(a['ID'][role=='test']),ti)
    np.testing.assert_array_equal(a['ZPHOT'][role=='val'].astype(np.float32),r['val_evaluation']['z_spec'].numpy())
    bc=dict(object_id=ids(a['ID']),source_row=rr,ra=a['RA'],dec=a['DEC'],historical_role=role)
    np.savez_compressed(OUT/'BC_historical_identity.npz',**bc)
    # BS historical train membership has an independently archived source cache.
    old=d.npz(d.BASE/'level0/source_cohort.npz',('targetid','split','group'));hv=ids(results['BS']['val_evaluation']['object_id']);ht=ids(results['BS']['test_evaluation']['object_id'])
    assert set(hv)==set(ids(old['targetid'][old['split']=='val']))
    hh=d.npz(RAW/'hdf_metadata.npz',('targetid','hsc_object_id','hsc_ra','hsc_dec'));lookup={str(t):i for i,t in enumerate(hh['targetid'])};allid=np.r_[ids(old['targetid'][old['split']=='train']),hv,ht];ix=np.array([lookup[x] for x in allid]);role=np.r_[np.full(22444,'train'),np.full(len(hv),'val'),np.full(len(ht),'test')]
    bs=dict(object_id=allid,source_row=ix,ra=hh['hsc_ra'][ix],dec=hh['hsc_dec'][ix],group=ids(hh['hsc_object_id'][ix]),historical_role=role)
    assert len(set(allid))==111980;np.savez_compressed(OUT/'BS_historical_identity.npz',**bs)
    (OUT/'replay').mkdir(exist_ok=True)
    # Save only known historical validation examples for native execution replay.
    from aion_magnitude.clauds_bands import ALL_BAND_FLUX_COLUMNS
    from aion_magnitude.utils import flux_to_ab_mag
    names=['u','u_star','g','r','i','z','y','Y','J','H','Ks'];cols=[ALL_BAND_FLUX_COLUMNS[b] for b in names]
    tr=np.flatnonzero(bc['historical_role']=='train');val=np.flatnonzero(bc['historical_role']=='val')[:32]
    with fitsio.FITS(str(OLD_CAT)) as f:raw=f[1].read(rows=bc['source_row'][np.r_[tr,val]],columns=cols)
    mags=np.column_stack([flux_to_ab_mag(raw[c].astype(float),mag_zero_point=23.)[0].astype(np.float32) for c in cols]);med=np.nanmedian(mags[:len(tr)],axis=0);vv=np.where(np.isfinite(mags[len(tr):]),mags[len(tr):],med)
    from aion_magnitude.FM_Qwen import serialize_qwen_feature_row
    import qwen_posttraining_comparison as oldpipeline
    _,serializer=oldpipeline.qwen_settings(oldpipeline.build_parser().parse_args(['--stage','prepare']))
    texts=[serialize_qwen_feature_row(dict(zip([b+'_mag' for b in names],row)),serialization=serializer) for row in vv]
    (OUT/'replay/BC_texts.json').write_text(json.dumps(texts,ensure_ascii=False))
    np.savez_compressed(OUT/'replay/BC.npz',object_id=vi[:32],features=vv,**{k:r['val_evaluation'][k].numpy()[:32] if k not in ['redshift_edges','redshift_centers'] else r['val_evaluation'][k].numpy() for k in ['logits','pz','z_spec','z_p16','z_p50','z_p84','redshift_edges','redshift_centers']})
    from aion_magnitude.filter_photometry import serialize_rows
    replay=d.npz(d.FAST/'source_replay.npz');metadata=json.loads((d.SOURCE/'prepared.json').read_text());texts=serialize_rows(replay,metadata,catalogue='galaxiesml-spec',level=0);(OUT/'replay/BS_texts.json').write_text(json.dumps(texts));np.savez_compressed(OUT/'replay/BS.npz',object_id=ids(replay['targetid']),**{k:v for k,v in replay.items() if k!='targetid'})
    d.write(OUT/'exposure_recovery.json',dict(BC_train=int((bc['historical_role']=='train').sum()),BC_validation=int((bc['historical_role']=='val').sum()),BC_test=int((bc['historical_role']=='test').sum()),BC_original_validation_IDs_and_ZPHOT_bit_exact=True,BC_original_test_IDs_exact=True,BC_original_test_metrics_inspected=False,BC_training_label_type='ZPHOT pseudo-label',BS_train=22444,BS_validation=len(hv),BS_test=len(ht),BS_original_membership_verified=True,BC_native_training_magnitude_medians=med.tolist(),native_AB_convention_warning='BC ZP23 historical values are used only for native replay, not physically calibrated main inputs',elapsed_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    d.write(OUT/'protected_before.json',protected);d.write(OUT/'checkpoint_provenance.json',pro)
    print('CHECKPOINT AND EXPOSURE RECOVERY COMPLETE',len(bs['object_id']),len(bc['object_id']),flush=True)
if __name__=='__main__':run()
