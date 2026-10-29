"""Frozen benchmark scoring; T2 labels accessible only after source/model gate."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
from stage2_data import d,take,FILTERS,make_rows
from stage3_models_1029 import Model,KPATH
import stage2_physics as pdf
import stage2_numerical as base
from stage3_train_1029 import logits
A=ROOT/'reports/stage3a/1029';B=ROOT/'reports/stage3b/1029'
def restrict(a,bandset):
    a={k:v.copy() for k,v in a.items()}
    names=None if bandset=='full' else [f'HSC.{b}' for b in ('grizy' if bandset=='HSC' else 'griz')]
    if names is not None:
        ok=np.isin(a['band_id'],[FILTERS.index(b) for b in names]);a['flux'][~ok]=np.nan;a['sigma'][~ok]=np.nan;a['band_id'][~ok]=-1
    return a
def wrong_metadata(a):
    a={k:v.copy() for k,v in a.items()}
    for i in range(len(a['flux'])):
        ok=np.isfinite(a['flux'][i])&np.isfinite(a['sigma'][i])&(a['sigma'][i]>0)&(a['band_id'][i]>=0);a['band_id'][i,ok]=np.roll(a['band_id'][i,ok],1)
    return a
_K=None
def physical_row(args):
    global _K
    if _K is None:_K=d.npz(KPATH)['kernel']
    import stage2a5_audit as nnls
    from stage2a6_identifiability import two_profile
    f,e,ids=args;valid=np.isfinite(f)&np.isfinite(e)&(e>0)&(ids>=0)
    if valid.sum()<3:return np.zeros(300),np.zeros(300)
    p12=-.5*nnls.profile((f,e,ids,_K,0.));p2=-.5*two_profile(f[valid],e[valid],_K[ids[valid]][:,:,[6,5]])[0]
    return p12,p2
def physical():
    assert (A/'benchmark_lock.json').exists();a=d.npz(A/'T2_photometry.npz');start=time.time()
    for bandset in ['full','HSC','overlap']:
        path=B/f'physical_label_free_T2_{bandset}.npz'
        if path.exists():continue
        aa=restrict(a,bandset)
        with ProcessPoolExecutor(max_workers=6) as pool:res=list(pool.map(physical_row,zip(aa['flux'],aa['sigma'],aa['band_id']),chunksize=16))
        np.savez_compressed(path,object_id=a['object_id'],D12=np.array([x[0] for x in res]),D2=np.array([x[1] for x in res]))
        print('PHYSICS label-free',bandset,len(res),time.time()-start,flush=True)
    d.write(B/'physical_provenance.json',dict(kernel_sha256=d.sha(KPATH),target_labels_read=False,D12='NNLS perz, measured errors, uniform z density, no population prior, profile pseudo-posterior',D2='fixed6/5 constrained profile diagnostic, no real adequacy/calibration claim',seconds=time.time()-start))
def bootstrap(a,b,z):
    x=(np.abs(a-z)/(1+z)>.15).astype(float);y=(np.abs(b-z)/(1+z)>.15).astype(float);diff=x-y;rng=np.random.default_rng(1029)
    vals=np.array([diff[rng.integers(0,len(z),len(z))].mean() for _ in range(2000)])
    return dict(n=len(z),outlier_difference_R2_minus_R1=float(diff.mean()),outlier_difference_CI95=np.quantile(vals,[.025,.975]).tolist(),relative_outlier_change=float(diff.mean()/y.mean()) if y.mean() else None,bootstrap_unit='unique galaxy; same paired resample',conditional_on_frozen_seed=True)
def c4_data():
    k=d.npz(KPATH)['kernel'];zgrid=pdf.grid()[1];rng=np.random.default_rng(1029);zz=rng.choice(np.flatnonzero((zgrid>=.1)&(zgrid<=1.2)),128);weights=rng.dirichlet([1,1],128)
    systems={'HSC':[f'HSC.{x}' for x in 'grizy'],'Legacy':['DECam.g','DECam.r','DECam.z','WISE.W1','WISE.W2'],'full':['MegaCam.u','MegaCam.u_star',*[f'HSC.{x}' for x in 'grizy'],'VIRCAM.Y','VIRCAM.J','VIRCAM.H','VIRCAM.Ks']};out={}
    for system,bands in systems.items():
        ids=np.array([FILTERS.index(x) for x in bands]);clean=np.einsum('bik,ik->ib',k[ids][:,zz][:,:,[6,5]],weights);sigma=np.maximum(np.median(np.abs(clean),axis=1),1e-20)[:,None]/40;sig=np.broadcast_to(sigma,clean.shape).copy();f=clean+rng.normal(size=clean.shape)*sig
        features=np.column_stack([v for j in range(len(bands)) for v in [f[:,j],sig[:,j]]]+[np.full(128,np.nan)]);a=make_rows(np.arange(128),features,bands,zgrid[zz],np.full(128,'control'),np.zeros(128),np.zeros(128),system);a['group']=np.array([f'C4:{i}' for i in range(128)]);out[system]=a
    return out
def save_eval(tag,ll,a,rows,conditional):
    pred=pdf.prediction(ll,a['z']);valid=np.isfinite(a['flux'])&np.isfinite(a['sigma'])&(a['sigma']>0)&(a['band_id']>=0)
    nvalid=valid.sum(1);assert np.isfinite(pred['pz']).all() and np.allclose(pred['pz'].sum(1),1,atol=1e-5)
    np.savez_compressed(B/'predictions'/f'{tag}.npz',**pred,object_id=a['object_id'],group=a['group'],n_valid=nvalid)
    metrics=pdf.metric(pred);rows.append(dict(condition=tag,insufficient_band_count=int((nvalid<3).sum()),**metrics))
    for lo,hi in [(0,.5),(.5,1),(1,1.5),(1.5,2),(2,3),(3,4.50001),(0,1),(1,2),(2,4.50001)]:
        mask=(a['z']>=lo)&(a['z']<hi)
        if mask.any():conditional.append(dict(condition=tag,stratum=f'z[{lo},{hi})',**pdf.metric(pred,mask)))
    for n in np.unique(nvalid):
        mask=nvalid==n;conditional.append(dict(condition=tag,stratum=f'usable_bands={n}',**pdf.metric(pred,mask)))
    # Brightness/SNR cut points are fixed on eligible HSC source training only.
    legacy=bool(np.all(a['catalogue']=='Legacy'));reference='z' if legacy else 'i';h=d.npz(A/f"{'Legacy' if legacy else 'HSC'}_training_joint_coverage.npz");edgesm=np.nanquantile(h['magnitude'],[1/3,2/3]);edgess=np.nanquantile(h['median_SN'],[1/3,2/3])
    bids=[FILTERS.index(x) for x in ['DECam.z','MzLS.z']] if legacy else [FILTERS.index('HSC.i')]
    mm=np.isin(a['band_id'],bids);f=np.where(mm,a['flux'],0).sum(1);mag=22.5-2.5*np.log10(np.where(f>0,f,np.nan));sn=np.nanmedian(np.where(valid,np.abs(a['flux']/a['sigma']),np.nan),axis=1)
    for key,v,edges in [(reference+'_brightness',mag,edgesm),('median_SN',sn,edgess)]:
        for j,(lo,hi) in enumerate(zip(np.r_[-np.inf,edges],np.r_[edges,np.inf])):
            mask=(v>=lo)&(v<hi)
            if mask.any():conditional.append(dict(condition=tag,stratum=f'source_{key}_tercile{j}',**pdf.metric(pred,mask)))
    for label,mask in [('fainter_than_source_'+reference+'_q99',mag>np.nanquantile(h['magnitude'],.99)),('lower_SN_than_source_q01',sn<np.nanquantile(h['median_SN'],.01))]:
        if mask.any():conditional.append(dict(condition=tag,stratum=label,**pdf.metric(pred,mask)))
    from aion_magnitude.metrics import pit_values
    pit=pit_values(torch.tensor(pred['pz']),torch.tensor(a['z']),edges=torch.tensor(pred['redshift_edges'])).numpy()
    d.write(B/'predictions'/f'{tag}_calibration.json',dict(pit_hist=np.histogram(pit,np.linspace(0,1,21))[0].tolist(),pit_edges=np.linspace(0,1,21).tolist(),PIT_mean=float(pit.mean()),PIT_std=float(pit.std()),source_magnitude_terciles=edgesm.tolist(),source_SN_terciles=edgess.tolist()))
    return pred
def evaluate():
    import fitsio
    lock=json.loads((B/'source_gate_and_model_lock.json').read_text());assert lock['passed'] and lock['all_controls_frozen']
    assert d.sha(A/'benchmark.json')==lock['protocol_sha256']
    for p,sha in lock['model_sha256'].items():assert d.sha(B/p)==sha
    assert not (B/'target_labels_opened.json').exists(),'Independent target evaluation is immutable'
    identity=d.npz(A/'T2_identity_locked.npz');a=d.npz(A/'T2_photometry.npz')
    for key in ['T2_identity_locked.npz','T2_photometry.npz']:
        assert d.sha(A/key)==json.loads((A/'benchmark.json').read_text())['dataset_sha256'][key]
    d.write(B/'target_labels_opened.json',dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_model_lock_sha256=d.sha(B/'source_gate_and_model_lock.json'),before_access=True,architecture_or_checkpoint_changes_allowed=False))
    with fitsio.FITS(str(A/'external/specz_unique.fits')) as f:s=f[1].read(rows=identity['spec_row'],columns=['Id_specz','specz','flag','Confidence_level','survey'])
    assert np.array_equal(s['Id_specz'],identity['spec_id']);assert np.all((s['flag']==4)&(s['Confidence_level']>=97))
    z=s['specz'];valid=np.isfinite(a['flux'])&np.isfinite(a['sigma'])&(a['sigma']>0)&(a['band_id']>=0)
    inrange=np.isfinite(z)&(z>0)&(z<=4.5);keep=inrange&(valid.sum(1)>=3)
    d.write(B/'target_eligibility.json',dict(reserved=len(z),finite_positive_in_grid=int(inrange.sum()),nonfinite=int((~np.isfinite(z)).sum()),nonpositive=int((z<=0).sum()),above_grid=int((z>4.5).sum()),insufficient_full_bands=int((inrange&(valid.sum(1)<3)).sum()),eligible=int(keep.sum()),z_bin_counts=np.histogram(z[keep],[0,.5,1,1.5,2,3,4.5])[0].tolist(),spectroscopy_surveys={str(k):int((s['survey'][keep]==k).sum()) for k in np.unique(s['survey'][keep])}))
    a=take(a,keep);a['z']=z[keep].astype(np.float32);np.savez_compressed(B/'T2_truth_eligible.npz',object_id=a['object_id'],group=a['group'],z=a['z'],source_row=identity['source_row'][keep],spec_id=identity['spec_id'][keep],survey=s['survey'][keep])
    (B/'predictions').mkdir(exist_ok=True);rows=[];conditional=[];paired=[];source_rows=[];source_cond=[];preds={}
    records=[torch.load(p,map_location='cpu',weights_only=False) for p in sorted((B/'models').glob('*/model.pt'))];common_rows=[];c4rows=[];c4cond=[];c4=c4_data();s3rows=[];s3cond=[]
    s3=d.npz(ROOT/'reports/stage2a/1028/mixed_eligible_trainval.npz');hv=d.npz(A/'HSC_eligible.npz');ids=np.intersect1d(s3['object_id'][s3['split']=='val'],hv['object_id'][hv['split']=='val']);ids=np.array(sorted(ids,key=lambda x:__import__('hashlib').sha256(str(x).encode()).hexdigest())[:128]);s3=take(s3,np.isin(s3['object_id'],ids));np.savez_compressed(B/'S3_controlled_validation_views.npz',**s3)
    for rec in records:
        model=Model(rec['arm']);model.load_state_dict(rec['state']);model.eval();tag=f"{rec['source']}_{rec['arm']}_seed{rec['seed']}"
        # Independent source-transfer population excludes every galaxy shared with other survey.
        if rec['arm'] in ['R1','R2'] and rec['source'] in ['HSC','Legacy']:
            source=d.npz(A/f"{rec['source']}_eligible.npz");other='Legacy' if rec['source']=='HSC' else 'HSC';aa=d.npz(A/f'{other}_eligible.npz');ind=(aa['split']=='val')&~np.isin(aa['group'],source['group']);aa=take(aa,ind)
            p=save_eval('E1_' +tag if rec['source']=='HSC' else 'E2_'+tag,logits(model,base.tensor_rows(aa)),aa,source_rows,source_cond)
            preds[('source',tag)]=p
        if rec['arm'] in ['R1','R2']:
            own=d.npz(A/f"{rec['source']}_eligible.npz");own=take(own,own['split']=='val');vt=base.tensor_rows(own)
            for step in [2500,5000]:
                model.load_state_dict(rec['snapshots'][step]);cp=pdf.prediction(logits(model,vt),own['z']);common_rows.append(dict(condition=tag,update=step,population='source_validation',**pdf.metric(cp)))
                tp=pdf.prediction(logits(model,base.tensor_rows(a)),a['z']);common_rows.append(dict(condition=tag,update=step,population='T2_full',**pdf.metric(tp)))
            model.load_state_dict(rec['state'])
        if rec['source']=='Mixed' and rec['arm'] in ['R0','R1','R2','C1','C2','C3']:
            for system,aa in c4.items():
                save_eval(f'C4_consistent_{tag}_{system}',logits(model,base.tensor_rows(aa)),aa,c4rows,c4cond)
                ww=wrong_metadata(aa);save_eval(f'C4_wrong_metadata_{tag}_{system}',logits(model,base.tensor_rows(ww)),ww,c4rows,c4cond)
            for survey in ['HSC','Legacy']:
                aa=take(s3,s3['catalogue']==survey);save_eval(f'S3_actual_{tag}_{survey}',logits(model,base.tensor_rows(aa)),aa,s3rows,s3cond)
        for bandset in ['full','HSC','overlap']:
            aa=restrict(a,bandset);p=save_eval('T2_'+tag+'_'+bandset,logits(model,base.tensor_rows(aa)),aa,rows,conditional);preds[(bandset,tag)]=p
            if bandset=='full' and rec['source']=='Mixed' and rec['arm'] in ['R1','R2']:
                ww=wrong_metadata(aa);save_eval('T2_'+tag+'_wrong_metadata',logits(model,base.tensor_rows(ww)),ww,rows,conditional)
        print('EVALUATED',tag,flush=True)
    for source in ['HSC','Legacy','Mixed']:
        for seed in [11,23]:
            for bandset in ['full','HSC','overlap']+(['source'] if source in ['HSC','Legacy'] else []):
                r1=preds[(bandset,f'{source}_R1_seed{seed}')];r2=preds[(bandset,f'{source}_R2_seed{seed}')]
                row=bootstrap(r2['z_p50'],r1['z_p50'],r1['z_spec']);row.update(source=source,seed=seed,bandset=bandset,NMAD_difference=pdf.metric(r2)['nmad']-pdf.metric(r1)['nmad'],CRPS_difference=pdf.metric(r2)['crps']-pdf.metric(r1)['crps'],CE_difference=pdf.metric(r2)['ce']-pdf.metric(r1)['ce']);paired.append(row)
                for lo,hi in [(0,1),(1,2),(2,4.50001)]:
                    mm=(r1['z_spec']>=lo)&(r1['z_spec']<hi)
                    if mm.any():paired.append(dict(source=source,seed=seed,bandset=bandset,stratum=f'z[{lo},{hi})',**bootstrap(r2['z_p50'][mm],r1['z_p50'][mm],r1['z_spec'][mm])))
    controls=[]
    for seed in [11,23]:
        r2=preds[('full',f'Mixed_R2_seed{seed}')]
        for arm in ['C1','C2','C3']+(['R0'] if seed==11 else []):
            cp=preds[('full',f'Mixed_{arm}_seed{seed}')];record=bootstrap(r2['z_p50'],cp['z_p50'],r2['z_spec']);record['outlier_difference_R2_minus_control']=record.pop('outlier_difference_R2_minus_R1');record.update(control=arm,seed=seed,CE_difference=pdf.metric(r2)['ce']-pdf.metric(cp)['ce'],CRPS_difference=pdf.metric(r2)['crps']-pdf.metric(cp)['crps']);controls.append(record)
    d.write(B/'capacity_information_control_bootstrap.json',controls)
    for bandset in ['full','HSC','overlap']:
        scores=d.npz(B/f'physical_label_free_T2_{bandset}.npz');assert np.array_equal(scores['object_id'],identity['object_id'].astype('U30'))
        for arm in ['D12','D2']:save_eval(f'T2_R3_{arm}_{bandset}',scores[arm][keep],restrict(a,bandset),rows,conditional)
    qpath=B/'qwen_frozen/T2_label_free_predictions.npz'
    if qpath.exists():
        q=d.npz(qpath);assert np.array_equal(q['object_id'],identity['object_id'].astype('U30'));save_eval('T2_Qwen_original_HSC_explicit_NA',q['logits'][keep],restrict(a,'HSC'),rows,conditional)
    # Historical256 remain an explicitly exploratory ZPHOT agreement experiment.
    t1=d.npz(ROOT/'reports/stage2a/1028/clauds_photometry.npz');old=d.npz(d.FAST/'clauds_predictions.npz',('targetid','z_spec'));assert np.array_equal(t1['object_id'],old['targetid'].astype('U30'));t1['z']=old['z_spec'];t1rows=[];t1cond=[]
    for rec in records:
        if rec['arm'] not in ['R1','R2'] and rec['source']!='Mixed':continue
        model=Model(rec['arm']);model.load_state_dict(rec['state']);tag=f"{rec['source']}_{rec['arm']}_seed{rec['seed']}"
        for bandset in ['full','HSC']:
            aa=restrict(t1,bandset);save_eval('T1_exploratory_'+tag+'_'+bandset,logits(model,base.tensor_rows(aa)),aa,t1rows,t1cond)
    for system,aa in c4.items():np.savez_compressed(B/f'C4_consistent_data_{system}.npz',**aa)
    d.table(B/'target_metrics.csv',rows);d.table(B/'target_conditional_metrics.csv',conditional);d.table(B/'source_transfer_metrics.csv',source_rows);d.table(B/'source_transfer_conditional_metrics.csv',source_cond);d.write(B/'paired_bootstrap.json',paired);d.table(B/'common_update_metrics.csv',common_rows);d.table(B/'counterfactual_metrics.csv',c4rows);d.table(B/'T1_exploratory_metrics.csv',t1rows);d.table(B/'T1_exploratory_conditional.csv',t1cond)
    d.table(B/'S3_controlled_validation_metrics.csv',s3rows)
    d.write(B/'evaluation_complete.json',dict(target_labels_opened_once=True,models_not_reselected=True,source_test_evaluated=False,source_metrics=len(source_rows),target_metrics=len(rows),completed_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
if __name__=='__main__':
    torch.set_num_threads(2)
    globals()[sys.argv[1]]()
