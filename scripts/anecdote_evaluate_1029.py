"""Locked head evaluation and inexpensive K-label controls on identical queries."""
import json,time,sys
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from anecdote_common_1029 import *
from anecdote_heads_1029 import Head,data,numerical_features,standardize
import stage2_physics as pdf
def from_logits(ll,z,edges=None):
    edges=pdf.grid()[0] if edges is None else edges;centers=(edges[:-1]+edges[1:])/2;lp=np.asarray(ll,float)+np.log(np.diff(edges));lp-=logsumexp(lp,axis=1,keepdims=True);p=np.exp(lp);cdf=p.cumsum(1)
    def quant(q):return centers[np.minimum((cdf<q).sum(1),len(centers)-1)]
    assert np.isfinite(lp).all() and np.allclose(p.sum(1),1)
    return dict(logits=lp.astype(np.float32),pz=p.astype(np.float32),z_spec=np.asarray(z,np.float32),z_p16=quant(.16),z_p50=quant(.5),z_p84=quant(.84),redshift_edges=edges,redshift_centers=centers)
def reference_heads():
    from aion_magnitude.models import PhotoZHead
    folder=OUT/'original_head_references';folder.mkdir(exist_ok=True)
    pro=json.loads((OUT/'checkpoint_provenance.json').read_text())
    for arm in ['BS','BC']:
        source=BS if arm=='BS' else BC;head=PhotoZHead(2560,n_z_bins=300,hidden_dim=256);head.load_state_dict(torch.load(source/'photoz_head.pt',map_location='cpu',weights_only=True));head.eval().requires_grad_(False);c=pro['families'][arm]['metadata']['config'];edges=np.linspace(c['z_min'],c['z_max'],301,dtype=np.float32)
        for task in TASKS:
            a,x,_=data(task,arm);raw=[]
            with torch.inference_mode():
                for i in range(0,len(x),256):raw.append(head(torch.tensor(x[i:i+256])).numpy())
            pred=from_logits(np.concatenate(raw),np.zeros(len(x)),edges);pred.pop('z_spec');np.savez_compressed(folder/f'{task}_{arm}.npz',object_id=a['object_id'],**pred)
    d.write(folder/'manifest.json',dict(protocol='Frozen historical head on the same shared canonical cached last-token representations; FP32 head evaluation. Separate from native historical replay.',native_grid_BC='0.00500003..5.99249983,300bins; do not compare bin-mass CE across grids',no_fresh_head_initialization_from_history=True,files_sha256={p.name:d.sha(p) for p in folder.glob('*.npz')}))
def all_labels(task,a):
    source=d.npz(OUT/f'{task}_support_development_labels.npz');sealed=d.npz(OUT/f'{task}_query_labels_sealed.npz');lookup={x:float(z) for f in [source,sealed] for x,z in zip(f['object_id'],f['z'])};y=np.array([lookup[x] for x in a['object_id']],np.float32);assert np.all(np.isfinite(y)&(y>=0)&(y<=4.5));return y
def write_prediction(tag,p,a):
    folder=OUT/'predictions';folder.mkdir(exist_ok=True);keep=np.isin(a['split'],['query','exposure_query']);saved={k:(v if k in ['redshift_edges','redshift_centers'] else v[keep]) for k,v in p.items()};np.savez_compressed(folder/f'{tag}.npz',**saved,object_id=a['object_id'][keep],split=a['split'][keep])
def statistics(tag,p,a,meta,rows,conditional):
    for split in ['query','exposure_query']:
        mask=a['split']==split
        if not mask.any():continue
        r=pdf.metric(p,mask);rows.append(dict(tag=tag,evaluation_cohort=split,**meta,**r,interval_width_below003_fraction=float(np.mean((p['z_p84'][mask]-p['z_p16'][mask])<.03))))
        for lo,hi in [(0,.5),(.5,1),(1,1.5),(1.5,2),(2,3),(3,4.50001),(0,1),(1,2),(2,4.50001)]:
            mm=mask&(p['z_spec']>=lo)&(p['z_spec']<hi)
            if mm.any():conditional.append(dict(tag=tag,evaluation_cohort=split,stratum=f'z[{lo},{hi})',**meta,**pdf.metric(p,mm)))
        valid=np.isfinite(a['flux'])&np.isfinite(a['sigma'])&(a['sigma']>0)&(a['band_id']>=0);nvalid=valid.sum(1);mmi=a['band_id']==FILTERS.index('HSC.i');ff=np.where(mmi,a['flux'],0).sum(1);mag=22.5-2.5*np.log10(np.where(ff>0,ff,np.nan))
        import warnings
        with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);sn=np.nanmedian(np.where(valid,np.abs(a['flux']/a['sigma']),np.nan),axis=1)
        support=a['split']=='support_pool'
        for key,values in [('i_magnitude',mag),('median_SN',sn)]:
            edges=np.nanquantile(values[support],[1/3,2/3])
            for j,(lo,hi) in enumerate(zip(np.r_[-np.inf,edges],np.r_[edges,np.inf])):
                mm=mask&(values>=lo)&(values<hi)
                if mm.any():conditional.append(dict(tag=tag,evaluation_cohort=split,stratum=f'supportpool_{key}_tercile{j}',**meta,**pdf.metric(p,mm)))
        for count in np.unique(nvalid[mask]):
            mm=mask&(nvalid==count);conditional.append(dict(tag=tag,evaluation_cohort=split,stratum=f'usable_bands={count}',**meta,**pdf.metric(p,mm)))
        # Fixed original HSC-source support diagnostic, when the old T2 assignment exists.
        if meta['task']=='C_SPEC':
            support=d.npz(ROOT/'reports/stage3a/1029/T2_support.npz');lookup={x:bool(s) for x,s in zip(support['object_id'],support['common_support'])};common=np.array([lookup.get(x,False) for x in a['object_id']])
            for flag in [False,True]:
                mm=mask&(common==flag)
                if mm.any():conditional.append(dict(tag=tag,evaluation_cohort=split,stratum=f'original_HSC_source_common_support={flag}',**meta,**pdf.metric(p,mm)))
        from aion_magnitude.metrics import pit_values
        pit=pit_values(torch.tensor(p['pz'][mask]),torch.tensor(p['z_spec'][mask]),edges=torch.tensor(p['redshift_edges'])).numpy();(OUT/'calibration').mkdir(exist_ok=True);d.write(OUT/'calibration'/f'{tag}_{split}.json',dict(pit_hist=np.histogram(pit,np.linspace(0,1,21))[0].tolist(),pit_mean=float(pit.mean()),pit_std=float(pit.std()),pit_edges=np.linspace(0,1,21).tolist(),n=int(mask.sum())))
def baseline(task,a,y,K,subset,arm):
    selected=json.loads((OUT/f'{task}_nested_subsets.json').read_text())[str(subset)][str(K)];lookup={x:i for i,x in enumerate(a['object_id'])};tr=np.array([lookup[x] for x in selected]);centers=pdf.grid()[1].astype(float)
    if arm=='KNN':
        from sklearn.neighbors import NearestNeighbors
        x=numerical_features(a);xx,_,_=standardize(x,tr);nn=NearestNeighbors(n_neighbors=min(10,K),n_jobs=2).fit(xx[tr]);distance,index=nn.kneighbors(xx);weights=1/(distance+1e-3);weights/=weights.sum(1,keepdims=True);z=y[tr][index];width=np.maximum(.03*(1+z),np.std(y[tr])/np.sqrt(K));component=-.5*((centers[None,None,:]-z[:,:,None])/width[:,:,None])**2;component-=logsumexp(component,axis=2,keepdims=True);mass=np.sum(weights[:,:,None]*np.exp(component),axis=1);ll=np.log(mass.clip(1e-300));extra=dict(k_neighbors=min(10,K),fitted_labels=K,development_labels_used=0)
    else:
        origin='BC' if task=='GMS' else 'BS';ref=d.npz(OUT/'original_head_references'/f'{task}_{origin}.npz');assert np.array_equal(ref['object_id'],a['object_id']);x=ref['z_p50'].astype(float);xc=x[tr]-x[tr].mean();yc=y[tr]-y[tr].mean();beta=float((np.dot(xc,yc)+.1)/(np.dot(xc,xc)+.1));alpha=float(y[tr].mean()-beta*x[tr].mean());center=alpha+beta*x;sigma=max(float(np.sqrt(np.sum((y[tr]-center[tr])**2)/max(K-2,1))),.03);ll=-.5*((centers[None,:]-center[:,None])/sigma)**2;extra=dict(original_source_head=origin,alpha=alpha,beta=beta,sigma=sigma,fitted_labels=K,development_labels_used=0,unbounded_affine_centers_outside_grid=int(((center<0)|(center>4.5)).sum()))
    return from_logits(ll,y),extra
def evaluate():
    assert not (OUT/'query_labels_opened.json').exists()
    assert json.loads((OUT/'prequery_verification.json').read_text())['passed']
    locks={task:json.loads((OUT/f'{task}_head_selection_locked.json').read_text()) for task in TASKS}
    for task,lock in locks.items():
        assert lock['complete']
        for path,sha in lock['head_files_sha256'].items():assert d.sha(OUT/path)==sha
    d.write(OUT/'all_head_selection_locked.json',dict(tasks=locks,protocol_sha256=d.sha(OUT/'protocol.json'),query_evaluated=False))
    reference_heads()
    d.write(OUT/'query_labels_opened.json',dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),head_selection_lock_sha256=d.sha(OUT/'all_head_selection_locked.json'),no_candidate_or_hyperparameter_changes_permitted=True,evaluation_code_sha256=d.sha(Path(__file__))))
    rows=[];conditional=[];runtime=[]
    for task in TASKS:
        a=d.npz(OUT/f'{task}_photometry.npz');y=all_labels(task,a)
        for path in sorted((OUT/'heads').glob(task+'_*/head.pt')):
            r=torch.load(path,map_location='cpu',weights_only=False);_,x,_=data(task,r['arm'],r['format'],r['pooling']);xx=np.clip((x-r['mean'])/r['std'],-10,10).astype(np.float32);model=Head(r['feature_dim']);model.load_state_dict(r['state']);model.eval().requires_grad_(False);start=time.time();raw=[]
            with torch.inference_mode():
                for i in range(0,len(xx),512):raw.append(model(torch.tensor(xx[i:i+512])).numpy())
            p=from_logits(np.concatenate(raw),y);meta={k:r[k] for k in ['task','arm','K','subset_seed','head_seed','loss','format','pooling','best_update','head_learned']};lookup={x:i for i,x in enumerate(a['object_id'])};tr=np.array([lookup[x] for x in r['train_IDs']]);meta.update(support_z_ge2=int((y[tr]>=2).sum()),support_z_ge1=int((y[tr]>=1).sum()),development_label_budget=256);write_prediction(r['tag'],p,a);statistics(r['tag'],p,a,meta,rows,conditional);runtime.append(dict(tag=r['tag'],CPU_threads=2,seconds=time.time()-start,n=len(xx)))
        for subset in [101,202,303]:
            for K in [16,32,64,128,256,512]:
                for arm in ['KNN','GLOBAL']:
                    p,extra=baseline(task,a,y,K,subset,arm);tag=f'{task}_{arm}_K{K}_subset{subset}';meta=dict(task=task,arm=arm,K=K,subset_seed=subset,head_seed=0,loss='fixed_numeric',format='canonical',pooling='numeric',best_update=-1,head_learned=True,development_label_budget=0,baseline_extra=extra);write_prediction(tag,p,a);statistics(tag,p,a,meta,rows,conditional)
        # Original frozen heads remain distinct native-grid references, never K0 on fresh curve.
        for arm in ['BS','BC']:
            ref=d.npz(OUT/'original_head_references'/f'{task}_{arm}.npz');ref['z_spec']=y;mask=(a['split']=='query')&(y>=ref['redshift_edges'][0])&(y<=ref['redshift_edges'][-1]);meta=dict(task=task,arm='ORIGINAL_'+arm,K=0,subset_seed=0,head_seed=0,loss='original_head',format='canonical',pooling='last',native_grid_min=float(ref['redshift_edges'][0]),native_grid_max=float(ref['redshift_edges'][-1]),outside_native_grid_query=int(((a['split']=='query')&~mask).sum()));rows.append(dict(tag=f'{task}_ORIGINAL_{arm}',evaluation_cohort='query_native_grid_only',**meta,**pdf.metric(ref,mask)))
        print('QUERY EVALUATED',task,'rows',len(rows),flush=True)
    d.table(OUT/'metrics.csv',rows);d.table(OUT/'conditional_metrics.csv',conditional);d.table(OUT/'query_runtime.csv',runtime);d.write(OUT/'query_evaluation_complete.json',dict(complete=True,metric_conditions=len(rows),query_labels_used_for_selection=False,backbone_or_adapter_updated=False,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
if __name__=='__main__':torch.set_num_threads(2);evaluate()
