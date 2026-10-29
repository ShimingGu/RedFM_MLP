"""Produce matched-cohort assessment and paired galaxy bootstrap figures."""
import sys,json,csv,time
from pathlib import Path
import numpy as np,pandas as pd,torch
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'notebooks'),str(ROOT)]
from anecdote_common_1029 import *
from knn_check_cpu_1029 import inputs,knn,kde,metric,table,pred_np,subset_prediction,feature_sets
from anecdote_heads_1029 import numerical_features,standardize
from aion_magnitude.models import PhotoZHead
DEST=ROOT/'reports/stage4_5/readout_knn_discrepancy_1029'
torch.set_num_threads(2)
def loadp(name):return d.npz(DEST/'predictions'/f'{name}.npz')
def normalize(p,ids,z):
 ix={v:i for i,v in enumerate(p['object_id'])};ii=np.array([ix[x] for x in ids]);r=subset_prediction(p,ii,z);r['object_id']=ids;return r
def add_native_space(a,y,q,trs):
 n=d.npz(DEST/'BS_native_features.npz');assert d.sha(DEST/'BS_native_features.npz')==json.loads((DEST/'native_extraction.json').read_text())['native_features_sha256'];index={x:i for i,x in enumerate(n['object_id'])}; qn=np.array([index[v] for v in a['object_id'][q]]);yn=np.array([y[np.flatnonzero(a['object_id']==x)[0]] for x in n['object_id']],np.float32)
 head=PhotoZHead(2560,300,256);head.load_state_dict(torch.load(BS/'photoz_head.pt',map_location='cpu',weights_only=True));head.eval().requires_grad_(False)
 with torch.inference_mode():hidden=head.net[:-1](torch.tensor(n['last'])).numpy()
 rows=[]
 for seed,tr in trs.items():
  tn=np.array([index[v] for v in a['object_id'][tr]])
  for name,x,dist in [('BS_native_hidden_euclidean',n['last'],'euclidean'),('BS_native_hidden_cosine',n['last'],'cosine'),('BS_native_supervised_head_penultimate',hidden,'euclidean')]:
   dd,ix,w=knn(x,tn,qn,yn,dist,False);p=kde(yn[tn],ix,w,y[q]);tag=f'{name}_subset{seed}';pred_np(p,np.arange(len(q)),a['object_id'][q],tag);rows.append(dict(method=name,subset_seed=seed,distance=dist,train_labels=256,dev_labels=0,**metric(p)))
 table('native_hidden_knn_metrics.csv',rows)
 base=pd.read_csv(DEST/'knn_metrics.csv');native=pd.DataFrame(rows);pd.concat([base[base.method.str.contains('hidden|penultimate')],native],ignore_index=True).to_csv(DEST/'hidden_knn_metrics.csv',index=False)
def event(p,z):return np.abs((p['z_p50']-z)/(1+z))>.15
def obj_metrics(p,z):
 pz=np.asarray(p['pz'],float);edges=p['redshift_edges'];cent=p['redshift_centers'];idx=(np.searchsorted(edges,z,side='left')-1).clip(0,len(cent)-1);cdf=np.cumsum(pz,axis=1);return dict(outlier=event(p,z).astype(float),ce=-np.log(np.maximum(pz[np.arange(len(z)),idx],1e-300)),crps=np.square(cdf-(cent[None,:]>=z[:,None])).sum(axis=1)*np.mean(np.diff(cent)))
def main():
 a,y,q,dev,trs=inputs();ids=a['object_id'][q];z=y[q];n=1024;assert len(q)==n;assert all((OUT/'verification.json').exists() for _ in [0]);add_native_space(a,y,q,trs)
 native=normalize(d.npz(DEST/'archived_native_all_clean.npz'),ids,z);arch=d.npz(DEST/'archived_native_all_clean.npz');nn={v:i for i,v in enumerate(arch['object_id'])};np.testing.assert_array_equal(np.array([arch['z_spec'][nn[x]] for x in ids]),z)
 canon=normalize(d.npz(OUT/'original_head_references/GMS_BS.npz'),ids,z)
 for name,p in [('BS_historical_native',native),('BS_historical_canonical',canon)]:np.savez_compressed(DEST/'predictions'/f'{name}.npz',**p)
 rows=[]
 def add(method,p,s=0,h=0,train=0,devlabels=0):
  assert np.array_equal(p['object_id'],ids);r=metric(p);rows.append(dict(method=method,subset_seed=s,head_seed=h,train_labels=train,development_labels=devlabels,**r));return p
 add('BS_historical_native',native);add('BS_historical_canonical',canon)
 groups={'BS_historical_native':[native]*6,'BS_historical_canonical':[canon]*6}
 for arm in ['B0','BS','BC','NUM']:
  for loss in ['hard','smooth3']:
   name=f'{arm}_{loss}';groups[name]=[]
   for s in [101,202,303]:
    for h in [11,23]:
     tag=f'GMS_{arm}_canonical_last_{loss}_K256_subset{s}_seed{h}';groups[name].append(add(name,normalize(loadp(tag),ids,z),s,h,256,256))
 for method in ['photo_original','photo_uniform','global_kde','hidden_B0_euclidean','hidden_BS_euclidean','hidden_BC_euclidean','hidden_B0_cosine','hidden_BS_cosine','hidden_BC_cosine','hidden_B0_meanpool_euclidean','hidden_BS_meanpool_euclidean','hidden_BC_meanpool_euclidean','BS_supervised_head_penultimate_canonical','BS_native_hidden_euclidean','BS_native_hidden_cosine','BS_native_supervised_head_penultimate']:
  groups[method]=[]
  for s in [101,202,303]:
   p=add(method,normalize(loadp(f'{method}_subset{s}'),ids,z),s,0,256,0);groups[method].extend([p,p])
 groups['BS_native_smooth3']=[]
 for s in [101,202,303]:
  for h in [11,23]:groups['BS_native_smooth3'].append(add('BS_native_smooth3',normalize(loadp(f'BS_native_smooth3_subset{s}_seed{h}'),ids,z),s,h,256,256))
 table('native_vs_canonical_metrics.csv',[r for r in rows if r['method'] in ['BS_historical_native','BS_historical_canonical','BS_native_smooth3','BS_smooth3']]);table('integrated_model_metrics.csv',rows)
 stats=pd.DataFrame(rows);coarse=[]
 for name,g in stats.groupby('method',sort=False):
  coarse.append(dict(method=name,n_fits=len(g),train_labels=int(g.train_labels.iloc[0]),dev_labels=int(g.development_labels.iloc[0]),outlier_mean=g.outlier_fraction.mean(),outlier_min=g.outlier_fraction.min(),outlier_max=g.outlier_fraction.max(),nmad_mean=g.nmad.mean(),ce_mean=g.ce.mean(),crps_mean=g.crps.mean(),coverage_mean=g.coverage_68.mean(),bias_mean=g.bias.mean()))
 table('integrated_summary.csv',coarse)
 # Same galaxy bootstrap indices shared across all comparisons and all fits.
 rng=np.random.default_rng(1029);boot=rng.integers(0,n,size=(1000,n));pairs=[('BS_historical_native','BS_historical_canonical'),('BS_smooth3','BS_hard'),('BS_smooth3','photo_original'),('NUM_smooth3','photo_original'),('BS_smooth3','NUM_smooth3'),('BS_smooth3','B0_smooth3'),('BC_smooth3','B0_smooth3'),('BS_native_smooth3','BS_smooth3'),('BS_native_smooth3','photo_original'),('BS_historical_native','photo_original'),('BS_native_hidden_euclidean','hidden_BS_euclidean'),('hidden_BS_euclidean','hidden_B0_euclidean'),('hidden_BS_cosine','hidden_B0_cosine'),('hidden_BS_meanpool_euclidean','hidden_B0_meanpool_euclidean'),('hidden_BS_euclidean','photo_uniform')];out=[]
 for left,right in pairs:
  aa=groups[left];bb=groups[right];assert len(aa)==len(bb)==6;ma=[obj_metrics(p,z) for p in aa];mb=[obj_metrics(p,z) for p in bb]
  for measure in ['outlier','ce','crps']:
   dif=np.stack([u[measure]-v[measure] for u,v in zip(ma,mb)]);perobject=dif.mean(0);draws=perobject[boot].mean(1);row=dict(left=left,right=right,metric=measure,delta_mean=float(perobject.mean()),bootstrap_low=float(np.quantile(draws,.025)),bootstrap_high=float(np.quantile(draws,.975)),fit_delta_min=float(dif.mean(1).min()),fit_delta_max=float(dif.mean(1).max()),bootstrap_unit='1024 galaxies jointly resampled; fixed fits averaged, not independent fits');
   if measure=='outlier':row.update(both_wrong_mean=float(np.mean([np.sum(event(u,z)&event(v,z)) for u,v in zip(aa,bb)])),left_only_wrong_mean=float(np.mean([np.sum(event(u,z)&~event(v,z)) for u,v in zip(aa,bb)])),right_only_wrong_mean=float(np.mean([np.sum(~event(u,z)&event(v,z)) for u,v in zip(aa,bb)])))
   out.append(row)
 table('paired_comparisons.csv',out)
 # Strata, distribution and threshold tails; per fit reports uncertainty separately.
 bins=[0,.15,.3,.5,.7,1,4.50001];labels=['0–0.15','0.15–0.3','0.3–0.5','0.5–0.7','0.7–1','1–4.5'];zs=[];tails=[]
 for name in ['BS_historical_native','BS_historical_canonical','BS_hard','BS_smooth3','NUM_smooth3','photo_original','hidden_BS_euclidean','hidden_BS_meanpool_euclidean','BS_native_smooth3']:
  for j,(lo,hi) in enumerate(zip(bins[:-1],bins[1:])):
   mm=(z>=lo)&(z<hi); nquery=int(mm.sum());support=np.mean([np.sum((y[tr]>=lo)&(y[tr]<hi)) for tr in trs.values()]);val=np.array([np.mean(event(p,z)[mm]) for p in groups[name]]) if nquery else np.array([np.nan]);zs.append(dict(method=name,redshift_bin=labels[j],lo=lo,hi=hi,n_query=nquery,mean_support_count=support,outlier_mean=float(np.mean(val)),fit_range_low=float(np.min(val)),fit_range_high=float(np.max(val))))
  err=np.stack([np.abs((p['z_p50']-z)/(1+z)) for p in groups[name]])
  for t in [.05,.1,.15,.2]:tails.append(dict(method=name,threshold=t,fraction=float(np.mean(err>t))))
 table('conditional_redshift_metrics.csv',zs);table('error_threshold_tail.csv',tails)
 qstats=dict(n=len(z),minimum=float(z.min()),q05=float(np.quantile(z,.05)),q25=float(np.quantile(z,.25)),median=float(np.median(z)),q75=float(np.quantile(z,.75)),q95=float(np.quantile(z,.95)),maximum=float(z.max()),per_bin_query_counts=[int(np.sum((z>=lo)&(z<hi))) for lo,hi in zip(bins[:-1],bins[1:])]);d.write(DEST/'query_redshift_distribution.json',qstats)
 # Scalar baselines have no PDF and are maintained as separate rows.
 k=pd.read_csv(DEST/'knn_metrics.csv');const=k[k.method.isin(['constant_mean','constant_median','global_kde'])];const.to_csv(DEST/'constant_baseline_and_query_distribution.csv',index=False)
 nd=pd.read_csv(DEST/'neighbourhood_diagnostics.csv',dtype={'object_id':str});photo=nd[nd.method=='photo_original'];orig={};
 for kname in ['nearest_distance','nearest_angular_arcsec','nearest_abs_dz','neighbour_z_std','effective_neighbours']:
  orig[kname]={str(quant):float(photo[kname].quantile(quant)) for quant in [.05,.5,.95]}
 orig['nearest_abs_dz_lt_0.01']=float((photo.nearest_abs_dz<.01).mean());orig['nearest_abs_dz_lt_0.05']=float((photo.nearest_abs_dz<.05).mean());orig['nearest_abs_dz_lt_0.1']=float((photo.nearest_abs_dz<.1).mean());d.write(DEST/'original_neighbourhood_summary.json',orig)
 # Exact photometric-neighbour differences are diagnostic, never selection criteria.
 idmap={x:i for i,x in enumerate(a['object_id'])};shapes=feature_sets(a)['shape_mask'];photodiff=[]
 for row in photo.itertuples(index=False):
  i=idmap[row.object_id];j=idmap[json.loads(row.neighbour_ids)[0]];both=np.isfinite(a['flux'][i,:5])&np.isfinite(a['flux'][j,:5]);fi=a['flux'][i,:5];fj=a['flux'][j,:5];ratio=np.abs(fi[both]-fj[both])/(np.abs(fi[both])+np.abs(fj[both])+1e-12)
  photodiff.append(dict(object_id=row.object_id,subset_seed=row.subset_seed,nearest_object_id=a['object_id'][j],nearest_angular_arcsec=row.nearest_angular_arcsec,median_symmetric_relative_flux_difference=float(np.median(ratio)) if len(ratio) else np.nan,normalized_shape_distance=float(np.linalg.norm(shapes[i]-shapes[j])),query_i_flux=float(fi[2]),neighbor_i_flux=float(fj[2])))
 table('nearest_neighbour_photometric_differences.csv',photodiff)
 # Matched distance and magnitude strata per support subset, with fixed labels.
 cond=[];distances={} 
 for seed in [101,202,303]:
  g=photo[photo.subset_seed==seed].set_index('object_id').loc[ids];ds=g.nearest_distance.to_numpy();qs=np.quantile(ds,[0,1/3,2/3,1]);qq=np.digitize(ds,qs[1:3]);f=a['flux'][q,2];m=22.5-2.5*np.log10(np.where(f>0,f,np.nan)); valid=np.isfinite(m); mag_edges=np.nanquantile(m,[1/3,2/3]);mg=np.digitize(m,mag_edges)
  for kind,ix in [('nearest_distance_tercile',qq),('apparent_i_mag_tercile',mg)]:
   for j in range(3):
    sel=(ix==j)&(valid if kind.startswith('apparent') else True);cond.append(dict(subset_seed=seed,stratum=kind,group=j,n=int(sel.sum()),outliers=int(g.outlier.to_numpy()[sel].sum()),outlier_fraction=float(g.outlier.to_numpy()[sel].mean()) if sel.any() else np.nan,edges=json.dumps(qs.tolist() if kind.startswith('nearest') else mag_edges.tolist())))
 table('original_knn_conditioned_metrics.csv',cond)
 # Descriptive label distributions: fixed query, K support, and the original BS training cohort.
 historical=d.npz(d.BASE/'level0/source_cohort.npz');ztrain=historical['z'][historical['split']=='train'];assert len(ztrain)==22444
 hbins=np.linspace(0,1.5,31);fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
 ax.hist(ztrain,bins=hbins,density=True,histtype='step',linewidth=2,label='Historical BS train (n=22444)');ax.hist(z,bins=hbins,density=True,histtype='step',linewidth=2,label='Current query (n=1024)')
 for seed,tr in trs.items():ax.hist(y[tr],bins=hbins,density=True,histtype='step',alpha=.5,label=f'K support seed {seed} (n=256)')
 ax.set(xlabel='Spectroscopic redshift',ylabel='Density',title='GMS redshift distribution by historical/current role');ax.legend(fontsize=8);ax.grid(alpha=.2)
 for ext in ['png','pdf']:fig.savefig(DEST/f'redshift_distribution.{ext}',dpi=160)
 plt.close(fig)
 sourcecounts=[]
 for lo,hi in zip(bins[:-1],bins[1:]):sourcecounts.append(dict(lo=lo,hi=hi,original_BS_train=int(np.sum((ztrain>=lo)&(ztrain<hi))),query=int(np.sum((z>=lo)&(z<hi))),mean_K_support=float(np.mean([np.sum((y[tr]>=lo)&(y[tr]<hi)) for tr in trs.values()]))))
 table('redshift_source_support_counts.csv',sourcecounts)
 # Same-cohort overview. error bars represent fit range; not galaxy CIs.
 fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True);order=['BS_historical_native','BS_historical_canonical','BS_hard','BS_smooth3','BS_native_smooth3','NUM_smooth3','photo_original','hidden_BS_euclidean','hidden_BS_meanpool_euclidean','BS_native_hidden_euclidean'];names=['BS old / native','BS old / canonical','BS fresh hard','BS fresh sigma3','BS native fresh sigma3','MLP sigma3','Photometric kNN','BS hidden kNN','BS mean-pool kNN','BS native hidden kNN'];cmap=['#1c7430','#777777','#cc6633','#e49a37','#f6be6b','#41825e','#8b6bac','#588bc0','#64a8d8','#3276a7'];
 for ax,column,title,scale in [(axs[0],'outlier_fraction','Outliers (%, lower is better)',100),(axs[1],'nmad','NMAD (lower is better)',1)]:
  for i,(name,color) in enumerate(zip(order,cmap)):
   vals=stats.loc[stats.method==name,column].to_numpy()*scale;mean=vals.mean();ax.barh(i,mean,color=color);ax.errorbar(mean,i,xerr=[[mean-vals.min()],[vals.max()-mean]],fmt='none',ecolor='black',capsize=2);ax.text(max(vals)+(.4 if scale==100 else .002),i,(f'{mean:.2f}' if scale==100 else f'{mean:.3f}'),va='center',fontsize=8)
  ax.set_yticks(range(len(order)),names);ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.18);ax.set_xlim(right=ax.get_xlim()[1]*1.12)
 fig.suptitle('GMS identical 1024 galaxies; K=256 + 256 development for learned heads; whiskers=fit range',fontsize=10)
 for ext in ['png','pdf']:fig.savefig(DEST/f'figure1_same_cohort.{ext}',dpi=160)
 plt.close(fig)
 # Use each support subset's leave-one-out nearest-neighbour distances as reference.
 fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
 space={'photo_uniform':numerical_features(a)}
 for arm in ['B0','BS','BC']:space[f'hidden_{arm}_euclidean']=d.npz(OUT/'features'/arm/'GMS_canonical.npz')['last']
 for method,label in [('photo_uniform','Photometry'),('hidden_B0_euclidean','B0 hidden'),('hidden_BS_euclidean','BS hidden'),('hidden_BC_euclidean','BC hidden')]:
  g=nd[nd.method==method].copy();rank=[]
  for seed,tr in trs.items():
   xx,_,_=standardize(space[method],tr);ref=NearestNeighbors(n_neighbors=2,n_jobs=2).fit(xx[tr]).kneighbors(xx[tr])[0][:,1];query_dist=g.loc[g.subset_seed==seed,'nearest_distance'].to_numpy();rank.extend(np.searchsorted(np.sort(ref),query_dist,side='right')/len(ref))
  g['rank']=rank;edges=np.linspace(0,1,11);mid=[];vals=[]
  for lo,hi in zip(edges[:-1],edges[1:]):
   v=g[(g['rank']>=lo)&(g['rank']<hi if hi<1 else g['rank']<=hi)];mid.append((lo+hi)/2);vals.append(v.nearest_abs_dz.median() if len(v) else np.nan)
  ax.plot(mid,vals,'o-',label=label)
 random_abs=[]
 for seed,tr in trs.items():
  for j in range(20):
   rng=np.random.default_rng(1029000+100*seed+j);random_abs.extend(np.abs(z-y[tr][rng.integers(0,len(tr),size=len(z))]).tolist())
 ax.axhline(np.median(random_abs),color='gray',ls=':',label='Random support label')
 ax.set(xlabel='Percentile in support leave-one-out distance distribution',ylabel='Median |z(query) − z(nearest)|',title='Neighbourhood redshift agreement; space-specific reference');ax.legend();ax.grid(alpha=.2)
 for ext in ['png','pdf']:fig.savefig(DEST/f'figure2_neighbourhoods.{ext}',dpi=160)
 plt.close(fig)
 # True-redshift strata (small-bin counts visible), plus tail probabilities.
 fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True); zz=pd.DataFrame(zs);xx=np.arange(len(labels))
 for name,label in [('photo_original','Photometric kNN'),('BS_historical_native','BS native'),('BS_smooth3','BS fresh sigma3'),('hidden_BS_euclidean','BS hidden kNN')]:
  g=zz[zz.method==name];axs[0].plot(xx,g.outlier_mean*100,'o-',label=label)
 axs[0].set(xticks=xx,xticklabels=[f'{s}\nq={qstats["per_bin_query_counts"][i]}\nK≈{np.mean([np.sum((y[tr]>=bins[i])&(y[tr]<bins[i+1])) for tr in trs.values()]):.0f}' for i,s in enumerate(labels)],xlabel='True redshift (exploratory)',ylabel='Outliers (%)',title='Conditional query failures');axs[0].tick_params(axis='x',labelsize=8);axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
 for name,label in [('photo_original','Photometric kNN'),('BS_historical_native','BS native'),('BS_smooth3','BS fresh sigma3'),('hidden_BS_euclidean','BS hidden kNN')]:
  g=pd.DataFrame(tails);g=g[g.method==name];axs[1].plot(g.threshold,g.fraction*100,'o-',label=label)
 axs[1].axvline(.15,ls=':',color='gray');axs[1].set(xlabel='Normalized absolute-error threshold',ylabel='Fraction above threshold (%)',title='Error tail; 0.15 is the primary threshold');axs[1].grid(alpha=.2);axs[1].legend(fontsize=8)
 for ext in ['png','pdf']:fig.savefig(DEST/f'figure3_redshift_and_tails.{ext}',dpi=160)
 plt.close(fig)
 # Stage4 hard CE learning histories; correct names, no imaginary soft-loss curve.
 fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
 for loss,color in [('hard','tab:red'),('smooth3','tab:blue')]:
  for seed in [11,23]:
   r=json.loads((OUT/'heads'/f'GMS_BS_canonical_last_{loss}_K256_subset101_seed{seed}'/'summary.json').read_text());his=pd.DataFrame(r['history']);axs[0].plot(his['update'],his.train_CE,color=color,alpha=.5,label=f'{loss} seed{seed}');axs[1].plot(his['update'],his.development_CE,color=color,alpha=.5,label=f'{loss} seed{seed}');axs[1].axvline(r['best_update'],color=color,ls=':',alpha=.4)
 for ax,title in zip(axs[:2],['Training hard CE','Development hard CE; dotted = selected step']):ax.set(xlabel='Update',ylabel='Hard CE',title=title);ax.legend(fontsize=7);ax.grid(alpha=.2)
 for i,loss in enumerate(['hard','smooth3']):
  g=stats[stats.method=='BS_'+loss];axs[2].bar(i,g.coverage_68.mean(),color=['tab:red','tab:blue'][i],alpha=.7);axs[2].text(i,g.coverage_68.mean()+.02,f'CRPS {g.crps.mean():.3f}\nCE {g.ce.mean():.3f}',ha='center',fontsize=8)
 axs[2].axhline(.68,ls=':',color='gray');axs[2].set(xticks=[0,1],xticklabels=['Hard','Sigma3'],ylim=(0,1),ylabel='68% interval coverage',title='Query calibration; six-fit mean')
 for ext in ['png','pdf']:fig.savefig(DEST/f'figure4_learning_curves.{ext}',dpi=160)
 plt.close(fig)
 print('REPORT DATA COMPLETE',flush=True)
if __name__=='__main__':main()
