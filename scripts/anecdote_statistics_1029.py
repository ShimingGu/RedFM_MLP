"""Paired galaxy intervals with subset and initialization variation kept separate."""
import json,csv
import numpy as np
from pathlib import Path
from scipy.special import logsumexp
from anecdote_common_1029 import *
def read_metrics():
    with (OUT/'metrics.csv').open() as f:return list(csv.DictReader(f))
def pred(row,cohort='query'):
    p=d.npz(OUT/'predictions'/f"{row['tag']}.npz");m=p['split']==cohort
    return {k:(v if k in ['redshift_edges','redshift_centers'] else v[m]) for k,v in p.items()}
def per_object(p):
    z=p['z_spec'];delta=(p['z_p50']-z)/(1+z);bins=np.minimum(np.searchsorted(p['redshift_edges'],z,side='left')-1,299).clip(0);ce=logsumexp(p['logits'].astype(float),axis=1)-p['logits'][np.arange(len(z)),bins];return (np.abs(delta)>.15).astype(float),ce
def paired(first,second):
    assert np.array_equal(first['object_id'],second['object_id']);assert np.array_equal(first['z_spec'],second['z_spec']);a,ac=per_object(first);b,bc=per_object(second);diff=a-b;dce=ac-bc;n=len(a);rng=np.random.default_rng(1029);out=[];ce=[];af=[];bf=[]
    for _ in range(1000):
        ix=rng.integers(0,n,n);out.append(diff[ix].mean());ce.append(dce[ix].mean());af.append(a[ix].mean());bf.append(b[ix].mean())
    return dict(n=n,delta_outlier=float(diff.mean()),delta_outlier_CI95=np.quantile(out,[.025,.975]).tolist(),delta_CE=float(dce.mean()),delta_CE_CI95=np.quantile(ce,[.025,.975]).tolist(),first_outlier_CI95=np.quantile(af,[.025,.975]).tolist(),B0_outlier_CI95=np.quantile(bf,[.025,.975]).tolist(),unit='unique query galaxy, predictions resampled together',conditional_on_support_subset_and_head_seed=True)
def run():
    assert (OUT/'query_evaluation_complete.json').exists();rows=read_metrics();query=[r for r in rows if r['evaluation_cohort']=='query'];index={(r['task'],r['arm'],r['format'],r['pooling'],r['loss'],r['K'],r['subset_seed'],r['head_seed']):r for r in query};results=[]
    for r in query:
        if r['arm'] not in ['BS','BC']:continue
        base=index[(r['task'],'B0',r['format'],r['pooling'],r['loss'],r['K'],r['subset_seed'],r['head_seed'])];values=paired(pred(r),pred(base));values.update({k:r[k] for k in ['task','arm','K','format','pooling','loss','subset_seed','head_seed','head_learned']});values['B0_head_learned']=base['head_learned'];values['primary_budget']=int(r['K']) in [64,256];values['primary_readout']=r['format']=='canonical' and r['pooling']=='last' and r['loss']=='hard';results.append(values)
    d.write(OUT/'paired_representation_bootstrap.json',results)
    exposure=[]
    for r in rows:
        if r['evaluation_cohort']!='exposure_query' or r['arm'] not in ['BS','BC'] or int(r['K']) not in [64,256] or r['loss']!='hard' or r['format']!='canonical' or r['pooling']!='last':continue
        base=index[(r['task'],'B0','canonical','last','hard',r['K'],r['subset_seed'],r['head_seed'])];v=paired(pred(r,'exposure_query'),pred(base,'exposure_query'));v.update({k:r[k] for k in ['task','arm','K','subset_seed','head_seed']});exposure.append(v)
    d.write(OUT/'historical_exposure_diagnostic_bootstrap.json',exposure)
    controls=[]
    for r in query:
        if r['arm'] not in ['BS','BC'] or int(r['K']) not in [64,256] or r['format']!='canonical' or r['pooling']!='last' or r['loss'] not in ['hard','smooth3','smooth10']:continue
        for arm in ['NUM','KNN','GLOBAL']:
            key=(r['task'],arm,'canonical','last',r['loss'],r['K'],r['subset_seed'],r['head_seed']) if arm=='NUM' else (r['task'],arm,'canonical','numeric','fixed_numeric',r['K'],r['subset_seed'],'0')
            base=index[key];v=paired(pred(r),pred(base));v['control_outlier_CI95']=v.pop('B0_outlier_CI95');v.update({k:r[k] for k in ['task','arm','K','loss','subset_seed','head_seed','head_learned']});v['control']=arm;v['control_development_labels']=base.get('development_label_budget','0');controls.append(v)
    d.write(OUT/'paired_numerical_control_bootstrap.json',controls)
    summary=[]
    groups={tuple(r[k] for k in ['task','arm','K','format','pooling','loss']) for r in results}
    for key in sorted(groups):
        rr=[r for r in results if tuple(r[k] for k in ['task','arm','K','format','pooling','loss'])==key];out=np.array([r['delta_outlier'] for r in rr]);ce=np.array([r['delta_CE'] for r in rr]);by_subset=[np.mean([r['delta_outlier'] for r in rr if r['subset_seed']==str(seed)]) for seed in [101,202,303]];by_head=[np.mean([r['delta_outlier'] for r in rr if r['head_seed']==str(seed)]) for seed in [11,23]]
        summary.append(dict(zip(['task','arm','K','format','pooling','loss'],key),replicates=len(rr),delta_outlier_mean=float(out.mean()),delta_outlier_min=float(out.min()),delta_outlier_max=float(out.max()),delta_outlier_SD=float(out.std(ddof=1)),delta_CE_mean=float(ce.mean()),delta_CE_min=float(ce.min()),delta_CE_max=float(ce.max()),replicates_outlier_improved=int((out<0).sum()),replicates_CE_improved=int((ce<0).sum()),paired_outlier_CI_below_zero=sum(r['delta_outlier_CI95'][1]<0 for r in rr),paired_CE_CI_below_zero=sum(r['delta_CE_CI95'][1]<0 for r in rr),subset_mean_outlier_range=[float(min(by_subset)),float(max(by_subset))],head_seed_mean_outlier_range=[float(min(by_head)),float(max(by_head))],all_heads_pass_learning_gate=all(r['head_learned']=='True' and r['B0_head_learned']=='True' for r in rr)))
    d.table(OUT/'paired_summary.csv',summary)
    # Predeclared thresholds: require each replicate's galaxy-CI upper bound below threshold.
    efficiency=[]
    for task in TASKS:
        for arm in BACKBONES:
            for threshold in [.1,.2,.3,.4,.5]:
                accepted=None
                for K in [16,32,64,128,256,512]:
                    rr=[r for r in results if r['task']==task and int(r['K'])==K and r['primary_readout'] and (r['arm']==arm if arm!='B0' else r['arm']=='BS')];upper=[r['first_outlier_CI95'][1] if arm!='B0' else r['B0_outlier_CI95'][1] for r in rr]
                    learned=all(r['head_learned']=='True' if arm!='B0' else r['B0_head_learned']=='True' for r in rr)
                    if len(upper)==6 and learned and all(v<=threshold for v in upper):accepted=K;break
                efficiency.append(dict(task=task,arm=arm,outlier_threshold=threshold,minimum_supported_training_K=accepted,fixed_development_label_cost=256,criterion='Each of6 fixed support/head replicates has conditional galaxy-bootstrap upper95% outlier bound below threshold; no independence claim over repeated galaxies'))
    d.write(OUT/'relative_sample_efficiency.json',efficiency)
    print('PAIRED STATISTICS COMPLETE',len(results),len(summary),flush=True)
if __name__=='__main__':run()
