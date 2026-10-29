"""Resume immutable prediction collection after exact-ID dtype guard repair.

Never retrains or reselects a candidate; never reopens the external specz table.
"""
from pathlib import Path
import sys,json,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import stage3_evaluate_1029 as e
from stage2_data import d,take,FILTERS
import stage2_physics as pdf
import stage2_numerical as base
A=e.A;B=e.B
def append_saved(tag,a,rows,conditional):
    p=d.npz(B/'predictions'/f'{tag}.npz');assert np.array_equal(p['object_id'],a['object_id']);assert np.array_equal(p['z_spec'],a['z'])
    valid=np.isfinite(a['flux'])&np.isfinite(a['sigma'])&(a['sigma']>0)&(a['band_id']>=0);n=valid.sum(1)
    rows.append(dict(condition=tag,insufficient_band_count=int((n<3).sum()),**pdf.metric(p)))
    masks=[(f'z[{lo},{hi})',(a['z']>=lo)&(a['z']<hi)) for lo,hi in [(0,.5),(.5,1),(1,1.5),(1.5,2),(2,3),(3,4.50001),(0,1),(1,2),(2,4.50001)]]
    masks.extend((f'usable_bands={j}',n==j) for j in np.unique(n))
    legacy=bool(np.all(a['catalogue']=='Legacy'));reference='z' if legacy else 'i';h=d.npz(A/f"{'Legacy' if legacy else 'HSC'}_training_joint_coverage.npz");em=np.nanquantile(h['magnitude'],[1/3,2/3]);es=np.nanquantile(h['median_SN'],[1/3,2/3]);ids=[FILTERS.index(x) for x in ['DECam.z','MzLS.z']] if legacy else [FILTERS.index('HSC.i')]
    mm=np.isin(a['band_id'],ids);ff=np.where(mm,a['flux'],0).sum(1);mag=22.5-2.5*np.log10(np.where(ff>0,ff,np.nan))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);sn=np.nanmedian(np.where(valid,np.abs(a['flux']/a['sigma']),np.nan),axis=1)
    for key,v,edges in [(reference+'_brightness',mag,em),('median_SN',sn,es)]:
        for j,(lo,hi) in enumerate(zip(np.r_[-np.inf,edges],np.r_[edges,np.inf])):masks.append((f'source_{key}_tercile{j}',(v>=lo)&(v<hi)))
    masks.extend([('fainter_than_source_'+reference+'_q99',mag>np.nanquantile(h['magnitude'],.99)),('lower_SN_than_source_q01',sn<np.nanquantile(h['median_SN'],.01))])
    for label,m in masks:
        if m.any():conditional.append(dict(condition=tag,stratum=label,**pdf.metric(p,m)))
    return p
def run():
    lock=json.loads((B/'source_gate_and_model_lock.json').read_text());assert lock['passed']
    for path,sha in lock['model_sha256'].items():assert d.sha(B/path)==sha
    repair=json.loads((B/'collection_ID_dtype_repair.json').read_text())
    for name,sha in repair['saved_predictions_sha256'].items():assert d.sha(B/'predictions'/name)==sha
    truth=d.npz(B/'T2_truth_eligible.npz');identity=d.npz(A/'T2_identity_locked.npz');allphot=d.npz(A/'T2_photometry.npz');keep=np.isin(allphot['object_id'],truth['object_id']);a=take(allphot,keep);assert np.array_equal(a['object_id'],truth['object_id']);a['z']=truth['z'];rows=[];conditional=[];source_rows=[];source_cond=[];common=[];c4rows=[];c4cond=[];s3rows=[];s3cond=[]
    records=[torch.load(p,map_location='cpu',weights_only=False) for p in sorted((B/'models').glob('*/model.pt'))];c4={name:d.npz(B/'physical_diagnostics'/f'input_{name}.npz') for name in ['HSC','Legacy','full']};s3=d.npz(B/'S3_controlled_validation_views.npz')
    for rec in records:
        tag=f"{rec['source']}_{rec['arm']}_seed{rec['seed']}"
        for bandset in ['full','HSC','overlap']:append_saved('T2_'+tag+'_'+bandset,e.restrict(a,bandset),rows,conditional)
        if rec['source']=='Mixed' and rec['arm'] in ['R1','R2']:append_saved('T2_'+tag+'_wrong_metadata',e.wrong_metadata(a),rows,conditional)
        if rec['arm'] in ['R1','R2'] and rec['source'] in ['HSC','Legacy']:
            source=d.npz(A/f"{rec['source']}_eligible.npz");other='Legacy' if rec['source']=='HSC' else 'HSC';aa=d.npz(A/f'{other}_eligible.npz');aa=take(aa,(aa['split']=='val')&~np.isin(aa['group'],source['group']));append_saved(('E1_' if rec['source']=='HSC' else 'E2_')+tag,aa,source_rows,source_cond)
        if rec['source']=='Mixed' and rec['arm'] in ['R0','R1','R2','C1','C2','C3']:
            for system,aa in c4.items():
                append_saved(f'C4_consistent_{tag}_{system}',aa,c4rows,c4cond);append_saved(f'C4_wrong_metadata_{tag}_{system}',e.wrong_metadata(aa),c4rows,c4cond)
            for system in ['HSC','Legacy']:append_saved(f'S3_actual_{tag}_{system}',take(s3,s3['catalogue']==system),s3rows,s3cond)
        if rec['arm'] in ['R1','R2']:
            model=e.Model(rec['arm']);own=d.npz(A/f"{rec['source']}_eligible.npz");own=take(own,own['split']=='val');vt=base.tensor_rows(own)
            for step in [2500,5000]:
                model.load_state_dict(rec['snapshots'][step]);p=pdf.prediction(e.logits(model,vt),own['z']);common.append(dict(condition=tag,update=step,population='source_validation',**pdf.metric(p)))
                p=pdf.prediction(e.logits(model,base.tensor_rows(a)),a['z']);common.append(dict(condition=tag,update=step,population='T2_full',**pdf.metric(p)))
        print('COLLECTED frozen',tag,flush=True)
    paired=[]
    for source in ['HSC','Legacy','Mixed']:
        for seed in [11,23]:
            for bandset in ['full','HSC','overlap']+(['source'] if source in ['HSC','Legacy'] else []):
                pre=('E1_' if source=='HSC' else 'E2_') if bandset=='source' else 'T2_';suf='' if bandset=='source' else '_'+bandset
                r1=d.npz(B/'predictions'/f'{pre}{source}_R1_seed{seed}{suf}.npz');r2=d.npz(B/'predictions'/f'{pre}{source}_R2_seed{seed}{suf}.npz');row=e.bootstrap(r2['z_p50'],r1['z_p50'],r1['z_spec']);row.update(source=source,seed=seed,bandset=bandset,NMAD_difference=pdf.metric(r2)['nmad']-pdf.metric(r1)['nmad'],CRPS_difference=pdf.metric(r2)['crps']-pdf.metric(r1)['crps'],CE_difference=pdf.metric(r2)['ce']-pdf.metric(r1)['ce']);paired.append(row)
                for lo,hi in [(0,1),(1,2),(2,4.50001)]:
                    mm=(r1['z_spec']>=lo)&(r1['z_spec']<hi)
                    if mm.any():paired.append(dict(source=source,seed=seed,bandset=bandset,stratum=f'z[{lo},{hi})',**e.bootstrap(r2['z_p50'][mm],r1['z_p50'][mm],r1['z_spec'][mm])))
    for bandset in ['full','HSC','overlap']:
        scores=d.npz(B/f'physical_label_free_T2_{bandset}.npz');assert np.array_equal(scores['object_id'],identity['object_id'].astype('U30'))
        for arm in ['D12','D2']:e.save_eval(f'T2_R3_{arm}_{bandset}',scores[arm][keep],e.restrict(a,bandset),rows,conditional)
    q=d.npz(B/'qwen_frozen/T2_label_free_predictions.npz');assert np.array_equal(q['object_id'],identity['object_id'].astype('U30'));e.save_eval('T2_Qwen_original_HSC_explicit_NA',q['logits'][keep],e.restrict(a,'HSC'),rows,conditional)
    t1=d.npz(ROOT/'reports/stage2a/1028/clauds_photometry.npz');old=d.npz(d.FAST/'clauds_predictions.npz',('targetid','z_spec'));assert np.array_equal(t1['object_id'],old['targetid'].astype('U30'));t1['z']=old['z_spec'];t1rows=[];t1cond=[]
    for rec in records:
        if rec['arm'] not in ['R1','R2'] and rec['source']!='Mixed':continue
        model=e.Model(rec['arm']);model.load_state_dict(rec['state']);tag=f"{rec['source']}_{rec['arm']}_seed{rec['seed']}"
        for bandset in ['full','HSC']:
            aa=e.restrict(t1,bandset);e.save_eval('T1_exploratory_'+tag+'_'+bandset,e.logits(model,base.tensor_rows(aa)),aa,t1rows,t1cond)
    for system,aa in c4.items():np.savez_compressed(B/f'C4_consistent_data_{system}.npz',**aa)
    for filename,data in [('target_metrics',rows),('target_conditional_metrics',conditional),('source_transfer_metrics',source_rows),('source_transfer_conditional_metrics',source_cond),('common_update_metrics',common),('counterfactual_metrics',c4rows),('T1_exploratory_metrics',t1rows),('T1_exploratory_conditional',t1cond),('S3_controlled_validation_metrics',s3rows)]:d.table(B/f'{filename}.csv',data)
    d.write(B/'paired_bootstrap.json',paired)
    for name,sha in repair['saved_predictions_sha256'].items():assert d.sha(B/'predictions'/name)==sha
    d.write(B/'evaluation_complete.json',dict(target_labels_opened_once=True,external_spec_table_not_reopened_on_resume=True,all_preexisting_primary_prediction_hashes_unchanged=True,models_not_reselected=True,source_test_evaluated=False,source_metrics=len(source_rows),target_metrics=len(rows),completed_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),collection_repair='Exact ID type normalization only; missing common-snapshot tables recollected from identical frozen states; no new candidate or tuning'))
    print('EVALUATION COMPLETE',len(rows),flush=True)
if __name__=='__main__':torch.set_num_threads(2);run()
