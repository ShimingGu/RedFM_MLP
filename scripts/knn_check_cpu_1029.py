"""Retrospective Stage 4.5: audit, cached readout reproduction, and fixed kNN diagnostics."""
import os,sys,json,csv,time,hashlib
from pathlib import Path
import numpy as np,torch
from scipy.special import logsumexp
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'notebooks'),str(ROOT)]
from anecdote_common_1029 import *
from anecdote_heads_1029 import numerical_features,standardize,Head
from anecdote_evaluate_1029 import baseline,from_logits
import stage2_physics as pdf
import stage2_numerical as numerical
from aion_magnitude.models import PhotoZHead
DEST=ROOT/'reports/stage4_5/readout_knn_discrepancy_1029';(DEST/'predictions').mkdir(exist_ok=True)
torch.set_num_threads(2)
def savejson(name,x):d.write(DEST/name,x)
def table(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (DEST/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def inputs():
    a=d.npz(OUT/'GMS_photometry.npz');lab=[d.npz(OUT/f'GMS_{part}.npz') for part in ['support_development_labels','query_labels_sealed']];ll={str(i):float(z) for l in lab for i,z in zip(l['object_id'],l['z'])};y=np.array([ll[x] for x in a['object_id']],np.float32);q=np.flatnonzero(a['split']=='query'); dev=np.flatnonzero(a['split']=='development'); subsets=json.loads((OUT/'GMS_nested_subsets.json').read_text()); ix={v:i for i,v in enumerate(a['object_id'])};tr={int(s):np.array([ix[x] for x in subsets[s]['256']]) for s in ['101','202','303']};return a,y,q,dev,tr
def pred_np(p,idx,ids,name):
    out={k:(np.asarray(v) if k in ['redshift_edges','redshift_centers'] else np.asarray(v)[idx]) for k,v in p.items()};out['object_id']=ids;np.savez_compressed(DEST/'predictions'/f'{name}.npz',**out)
def metric(p):return pdf.metric(p)
def subset_prediction(p,sel,z=None):
    r={k:(v if k in ['redshift_edges','redshift_centers'] else v[sel]) for k,v in p.items() if k not in ['object_id','split']};
    if z is not None:r['z_spec']=z
    return r
def get_existing(name,qids,z):
    p=d.npz(OUT/'predictions'/f'{name}.npz'); lookup={x:i for i,x in enumerate(p['object_id'])};sel=np.array([lookup[x] for x in qids]);return subset_prediction(p,sel,z)
def knn(x,tr,q,y,dist='euclidean',weighted=False):
    if dist=='euclidean': xx,_,_=standardize(x,tr)
    elif dist=='cosine':
        xx=np.asarray(x,dtype=np.float64)-np.mean(x[tr],axis=0);norm=np.linalg.norm(xx,axis=1,keepdims=True);xx=xx/np.maximum(norm,1e-30)
    else:raise ValueError(dist)
    nn=NearestNeighbors(n_neighbors=10,metric=dist,n_jobs=2).fit(xx[tr]);distance,index=nn.kneighbors(xx[q]); weights=1/(distance+1e-3) if weighted else np.ones_like(distance);weights/=weights.sum(1,keepdims=True)
    return distance,index,weights
def kde(ytrain,index,weights,zq):
    centers=pdf.grid()[1].astype(float);zn=ytrain[index];width=np.maximum(.03*(1+zn),np.std(ytrain)/np.sqrt(len(ytrain)));component=-.5*((centers[None,None,:]-zn[:,:,None])/width[:,:,None])**2;component-=logsumexp(component,axis=2,keepdims=True);mass=np.sum(weights[:,:,None]*np.exp(component),axis=1);return from_logits(np.log(mass.clip(1e-300)),zq)
def audit():
    a,y,q,dev,trs=inputs();cfg=json.loads((OUT/'protocol.json').read_text());verify=json.loads((OUT/'verification.json').read_text());assert verify['completed'];headlock=json.loads((OUT/'GMS_head_selection_locked.json').read_text());features={};checks={}
    for arm in ['B0','BS','BC']:
        manifest=json.loads((OUT/'features'/arm/'manifest.json').read_text());path=OUT/'features'/arm/'GMS_canonical.npz';assert d.sha(path)==manifest['feature_files']['GMS_canonical']['sha256'];f=d.npz(path);assert np.array_equal(f['object_id'],a['object_id']);features[arm]=f['last'];checks[arm]=d.sha(path)
    features['NUM']=numerical_features(a);records=[];maxdiff=0
    for arm in ['B0','BS','BC','NUM']:
      for loss in ['hard','smooth3']:
       for seed in [101,202,303]:
        for hs in [11,23]:
            tag=f'GMS_{arm}_canonical_last_{loss}_K256_subset{seed}_seed{hs}';path=OUT/'heads'/tag/'head.pt';assert d.sha(path)==headlock['head_files_sha256'][str(path.relative_to(OUT))];r=torch.load(path,map_location='cpu',weights_only=False);xx=np.clip((features[arm][q]-r['mean'])/r['std'],-10,10).astype(np.float32);m=Head(xx.shape[1]);m.load_state_dict(r['state']);m.eval()
            with torch.inference_mode(): out=m(torch.tensor(xx)).numpy()
            p=from_logits(out,y[q]);old=get_existing(tag,a['object_id'][q],y[q]);err=float(np.max(np.abs(p['pz']-old['pz'])));maxdiff=max(maxdiff,err);assert err<2e-6,(tag,err);pred_np(p,np.arange(len(q)),a['object_id'][q],tag)
            records.append(dict(tag=tag,arm=arm,loss=loss,subset_seed=seed,head_seed=hs,best_update=r['best_update'],head_learned=r['head_learned'],train_hard_CE_initial=r['history'][0]['train_CE'],train_hard_CE_final=r['history'][-1]['train_CE'],development_hard_CE_initial=r['initial_development_CE'],development_hard_CE_best=r['best_development_CE'],**metric(p)))
    table('soft_label_readout_metrics.csv',records)
    source=np.flatnonzero(np.isin(a['split'],['support_pool','development']));
    # No objects or physical source groups shared between any primary partition.
    roles=['support_pool','development','query'];identity={};
    for x in roles:
      for z in roles:
       if roles.index(x)>=roles.index(z):continue
       i=np.flatnonzero(a['split']==x);j=np.flatnonzero(a['split']==z);sep,near=nearest(take(a,i),take(a,j));identity[x+'__'+z]=dict(shared_id=int(len(set(a['object_id'][i])&set(a['object_id'][j]))),shared_group=int(len(set(a['group'][i])&set(a['group'][j]))),min_arcsec=float(sep.min()),within_1arcsec=int((sep<=1).sum()),within_2arcsec=int((sep<=2).sum()))
    exposure=d.npz(OUT/'GMS_full_exposure_audit.npz');ei={v:i for i,v in enumerate(exposure['object_id'])};idx=np.array([ei[v] for v in a['object_id'][np.r_[source,q]]]); flags={k:int(exposure[k][idx].sum()) for k in ['BS_train_match','BS_val_match','BC_train_match','BC_val_match','BS_train_ambiguous','BS_val_ambiguous','BC_train_ambiguous','BC_val_ambiguous']};
    assert all(v['shared_id']==0 and v['shared_group']==0 and v['within_1arcsec']==0 for v in identity.values());assert not any(flags.values())
    savejson('asset_and_identity_audit.json',dict(feature_hashes=checks,existing_head_fits_reproduced=len(records),maximum_head_pdf_replay_error=maxdiff,identity=identity,known_historical_exposures=flags,historical_stage4_verification=verify,retrospective_query=True))
    print('AUDIT HEADS',len(records),'maxdiff',maxdiff,'identity',identity,'flags',flags,flush=True)
def feature_sets(a):
    t=numerical.tensor_rows(a);mask=t['mask'].numpy();shape=np.arcsinh(t['f'].numpy())*mask;brightness=t['brightness'].numpy()[:,None];relative=np.log(t['e'].numpy()).clip(-20,20)*mask;absolute=np.where(mask,np.log(np.where(mask,a['sigma'],1)).clip(-20,20),0); q=np.where(t['qa'].numpy(),t['q'].numpy(),0);qa=t['qa'].numpy().astype(float);ebv=t['ebv'].numpy()[:,None];ea=t['ebv_available'].numpy()[:,None].astype(float);m=mask.astype(float)
    return {'shape_mask':np.c_[shape,m], 'brightness_only':brightness, 'shape_brightness':np.c_[shape,m,brightness], 'shape_brightness_relative_error':np.c_[shape,m,brightness,relative], 'measurement_selection_only':np.c_[absolute,m,q,qa,ebv,ea], 'original_without_quality_ebv':np.c_[np.arcsinh(t['f'].numpy())*m,relative,np.arcsinh(t['f'].numpy()/t['e'].numpy()).clip(-20,20)*m,m,brightness]}
def compute():
    a,y,q,dev,trs=inputs();qid=a['object_id'][q]; zq=y[q];x=numerical_features(a); fs=feature_sets(a);metrics=[];neighbours=[];null=[]; audit=[];existing=[]
    hidden={}
    for arm in ['B0','BS','BC']:
        m=json.loads((OUT/'features'/arm/'manifest.json').read_text());path=OUT/'features'/arm/'GMS_canonical.npz';assert d.sha(path)==m['feature_files']['GMS_canonical']['sha256']; h=d.npz(path);hidden[arm]=h['last'];hidden[arm+'_meanpool']=h['mean']
    oldhead=PhotoZHead(2560,n_z_bins=300,hidden_dim=256);oldhead.load_state_dict(torch.load(BS/'photoz_head.pt',map_location='cpu',weights_only=True));oldhead.eval().requires_grad_(False)
    with torch.inference_mode(): hidden['BS_supervised_head_penultimate_canonical']=oldhead.net[:-1](torch.tensor(hidden['BS'])).numpy()
    for seed,tr in trs.items():
      # Reproduce original exactly, with independent implementation and query-label perturbation check.
      dist,ind,w=knn(x,tr,q,y,'euclidean',True);p=kde(y[tr],ind,w,zq);original=get_existing(f'GMS_KNN_K256_subset{seed}',qid,zq);diff=float(np.max(np.abs(p['pz']-original['pz'])));assert diff<2e-6,(seed,diff)
      yy=y.copy();yy[q]=y[q][::-1]+.12345; p2,_=baseline('GMS',a,yy,256,seed,'KNN');d2=float(np.max(np.abs(p2['pz'][q]-p['pz'])));assert d2<2e-6
      rng=np.random.default_rng(1029);perm=rng.permutation(len(q));dd,ii,ww=knn(x,tr,q[perm],y,'euclidean',True);pr=kde(y[tr],ii,ww,zq[perm]);inv=np.argsort(perm);ordererr=float(np.max(np.abs(pr['pz'][inv]-p['pz'])));assert ordererr<1e-7
      audit.append(dict(subset_seed=seed,pdf_max_difference=diff,query_label_perturbation_pdf_max_difference=d2,query_row_permutation_pdf_max_difference=ordererr))
      pred_np(p,np.arange(len(q)),qid,f'photo_original_subset{seed}');metrics.append(dict(method='photo_original',distance='euclidean_weighted',subset_seed=seed,train_labels=256,dev_labels=0,**metric(p)))
      for r in range(20):
        sh=np.random.default_rng(1029000+100*seed+r).permutation(y[tr]);npred=kde(sh,ind,w,zq);null.append(dict(subset_seed=seed,permutation=r,**metric(npred)))
      for label,point in [('constant_median',np.median(y[tr])),('constant_mean',np.mean(y[tr]))]:
        err=(point-zq)/(1+zq);metrics.append(dict(method=label,distance='none',subset_seed=seed,train_labels=256,dev_labels=0,n=len(q),outlier_fraction=float(np.mean(abs(err)>.15)),nmad=float(1.48*np.median(abs(err-np.median(err)))),bias=float(np.mean(err)),ce='',crps='',coverage_68=''))
      allindex=np.tile(np.arange(len(tr)),(len(q),1));weights=np.full_like(allindex,1/len(tr),dtype=float);gp=kde(y[tr],allindex,weights,zq);pred_np(gp,np.arange(len(q)),qid,f'global_kde_subset{seed}');metrics.append(dict(method='global_kde',distance='none',subset_seed=seed,train_labels=256,dev_labels=0,**metric(gp)))
      variants=[('photo_uniform',x,'euclidean')]+[(name,arr,'euclidean') for name,arr in fs.items()]
      variants += [(f'hidden_{arm}_{ds}',hidden[arm],ds) for arm in ['B0','BS','BC'] for ds in ['euclidean','cosine']]
      variants += [(f'hidden_{arm}_meanpool_euclidean',hidden[arm+'_meanpool'],'euclidean') for arm in ['B0','BS','BC']]
      variants += [('BS_supervised_head_penultimate_canonical',hidden['BS_supervised_head_penultimate_canonical'],'euclidean')]
      for method,arr,ds in variants:
        dsq,nn,ww=knn(arr,tr,q,y,ds,False);pp=kde(y[tr],nn,ww,zq);pred_np(pp,np.arange(len(q)),qid,f'{method}_subset{seed}');metrics.append(dict(method=method,distance=ds,subset_seed=seed,train_labels=256,dev_labels=0,**metric(pp)))
        if method in ['photo_uniform','hidden_B0_euclidean','hidden_BS_euclidean','hidden_BC_euclidean','hidden_B0_cosine','hidden_BS_cosine','hidden_BC_cosine']:
         zn=y[tr][nn];err=(pp['z_p50']-zq)/(1+zq);query_ra=np.deg2rad(a['ra'][q]);query_dec=np.deg2rad(a['dec'][q]);near_ra=np.deg2rad(a['ra'][tr][nn[:,0]]);near_dec=np.deg2rad(a['dec'][tr][nn[:,0]]);angle=2*np.arcsin(np.sqrt(np.sin((query_dec-near_dec)/2)**2+np.cos(query_dec)*np.cos(near_dec)*np.sin((query_ra-near_ra)/2)**2))*180/np.pi*3600
         for i in range(len(q)):neighbours.append(dict(method=method,subset_seed=seed,object_id=qid[i],z_query=float(zq[i]),neighbour_ids=json.dumps(a['object_id'][tr][nn[i]].tolist()),neighbour_z=json.dumps(zn[i].tolist()),distances=json.dumps(dsq[i].tolist()),nearest_distance=float(dsq[i,0]),nearest_angular_arcsec=float(angle[i]),nearest_abs_dz=float(abs(zn[i,0]-zq[i])),neighbour_z_std=float(np.std(zn[i])),effective_neighbours=float(1/np.square(ww[i]).sum()),bias=float(err[i]),outlier=int(abs(err[i])>.15)))
      # Detailed diagnostics for the exact weighted original rather than uniform version.
      zn=y[tr][ind];err=(p['z_p50']-zq)/(1+zq);r1=np.deg2rad(a['ra'][q]);d1=np.deg2rad(a['dec'][q]);r2=np.deg2rad(a['ra'][tr][ind[:,0]]);d2=np.deg2rad(a['dec'][tr][ind[:,0]]);ang=2*np.arcsin(np.sqrt(np.sin((d1-d2)/2)**2+np.cos(d1)*np.cos(d2)*np.sin((r1-r2)/2)**2))*180/np.pi*3600
      for i in range(len(q)):neighbours.append(dict(method='photo_original',subset_seed=seed,object_id=qid[i],z_query=float(zq[i]),neighbour_ids=json.dumps(a['object_id'][tr][ind[i]].tolist()),neighbour_z=json.dumps(zn[i].tolist()),distances=json.dumps(dist[i].tolist()),nearest_distance=float(dist[i,0]),nearest_angular_arcsec=float(ang[i]),nearest_abs_dz=float(abs(zn[i,0]-zq[i])),neighbour_z_std=float(np.std(zn[i])),effective_neighbours=float(1/np.square(w[i]).sum()),bias=float(err[i]),outlier=int(abs(err[i])>.15)))
      print('CPU SUBSET',seed,'done',flush=True)
    table('knn_metrics.csv',metrics);table('neighbourhood_diagnostics.csv',neighbours);table('permutation_null_metrics.csv',null);savejson('knn_reproduction_checks.json',audit)
    print('CPU COMPLETE',len(metrics),'conditions',len(neighbours),'query-neighbour rows',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
