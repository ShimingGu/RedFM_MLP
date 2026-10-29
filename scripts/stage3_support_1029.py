"""Source-defined descriptive support, no target redshift or photo-z access."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
from stage2_data import d,FILTERS,take
from stage3_data_1029 import OUT as A
def grizy(a):
    f=[];e=[]
    for b in 'grizy':
        mm=a['band_id']==FILTERS.index('HSC.'+b);f.append(np.where(mm,a['flux'],0).sum(1));e.append(np.where(mm,a['sigma'],0).sum(1))
    return np.array(f).T,np.array(e).T
def describe(a):
    f,e=grizy(a);valid=np.isfinite(a['flux'])&np.isfinite(a['sigma'])&(a['sigma']>0)&(a['band_id']>=0)
    m=22.5-2.5*np.log10(np.where(f[:,2]>0,f[:,2],np.nan));sn=np.abs(f[:,2]/e[:,2]);c=np.where((f[:,0]>0)&(f[:,1]>0),-2.5*np.log10(np.maximum(f[:,0],1e-30)/np.maximum(f[:,1],1e-30)),np.nan)
    def q(v):return np.nanquantile(v,[.01,.1,.5,.9,.99]).tolist() if np.isfinite(v).any() else None
    return dict(n=len(f),i_magnitude=q(m),i_SN=q(sn),g_minus_r_positive=q(c),available_band_histogram={str(i):int((valid.sum(1)==i).sum()) for i in range(12)},any_bad_quality_fraction=float(np.any(a['quality']>0,axis=1).mean()),sky_ra_dec_range=[float(np.min(a['ra'])),float(np.max(a['ra'])),float(np.min(a['dec'])),float(np.max(a['dec']))])
def run():
    from sklearn.neighbors import NearestNeighbors
    h=d.npz(A/'HSC_eligible.npz');h=take(h,h['split']=='train');sf,se=grizy(h)
    sets={name:d.npz(A/f'{name}_photometry.npz') for name in ['T2','unlabeled_population']};sets['T1']=d.npz(ROOT/'reports/stage2a/1028/clauds_photometry.npz')
    rng=np.random.default_rng(1029);fold=rng.integers(0,5,len(sf));support={name:{} for name in sets}
    for joint in [False,True]:
        ratios={name:[] for name in sets}
        for j in range(5):
            train=fold!=j;cal=~train;spec=d.fit_transform(sf[train],se[train],joint);sx=d.transform(sf[train],se[train],spec);cx=d.transform(sf[cal],se[cal],spec)
            nn=NearestNeighbors(n_neighbors=10,n_jobs=2,algorithm='ball_tree').fit(sx);q95=np.quantile(nn.kneighbors(cx)[0].mean(1),.95)
            for name,a in sets.items():
                tf,te=grizy(a);tx=d.transform(tf,te,spec);ratios[name].append(nn.kneighbors(tx)[0].mean(1)/q95)
        for name,v in ratios.items():support[name]['joint' if joint else 'flux_colour']=np.median(v,axis=0)
    summaries={}
    for name,a in sets.items():
        f,e=grizy(a);complete=(np.isfinite(f)&np.isfinite(e)&(e>0)).all(1);s=support[name];common=complete&(s['joint']<=1)&(s['flux_colour']<=1)
        np.savez_compressed(A/f'{name}_support.npz',object_id=a['object_id'],complete_grizy=complete,common_support=common,**s)
        summaries[name]=dict(**describe(a),common_support_count=int(common.sum()),common_support_fraction=float(common.mean()),source_distance_joint_q10_q50_q90=np.quantile(s['joint'],[.1,.5,.9]).tolist())
    summaries['HSC_train']=describe(h)
    summaries['definition']='Stage0 signed asinh flux/colour and joint error/SNR feature contract; five source-only folds; k10 mean distance/source-calibration q95; completegrizy and both median ratios<=1; target labels unopened; descriptive support, not a density or confidence probability'
    summaries['limitations']='HSC nominal grizy responses only; no forced substitution of DECam or BASS. Legacy source wavelength/SNR/color coverage tabulated separately. UV/NIR not source-observed; WISE absent on target. Local COSMOS photometric sample only.'
    d.write(A/'photometric_support.json',summaries)
    # Full eligible source redshift/magnitude/colour/SN joint arrays for audit.
    for name in ['HSC','Legacy','Mixed']:
        a=d.npz(A/f'{name}_eligible.npz');a=take(a,a['split']=='train');f=a['flux'];e=a['sigma'];ids=a['band_id'];z=a['z'];fields={}
        for b,key in [('g','g'),('r','r'),('z','z'),('i','i')]:
            mm=np.isin(ids,[j for j,s in enumerate(FILTERS) if s.split('.')[-1]==b]);v=np.where(mm,f,0).sum(1);fields[key]=np.where(mm.any(1),v,np.nan)
        ref=fields['i'] if name=='HSC' else fields['z'];mag=22.5-2.5*np.log10(np.where(ref>0,ref,np.nan));valid=np.isfinite(f)&np.isfinite(e)&(e>0);sn=np.nanmedian(np.where(valid,np.abs(f/e),np.nan),axis=1)
        gr=-2.5*np.log10(np.where((fields['g']>0)&(fields['r']>0),fields['g']/fields['r'],np.nan));rz=-2.5*np.log10(np.where((fields['r']>0)&(fields['z']>0),fields['r']/fields['z'],np.nan))
        np.savez_compressed(A/f'{name}_training_joint_coverage.npz',group=a['group'],z=z,magnitude=mag,median_SN=sn,g_minus_r=gr,r_minus_z=rz,catalogue=a['catalogue'],available_bands=valid.sum(1))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for name in ['HSC','Legacy','Mixed']:
        a=d.npz(A/f'{name}_training_joint_coverage.npz');axs[0,0].hist(a['z'],bins=np.linspace(0,4.5,46),histtype='step',label=name);axs[1,0].scatter(a['z'][::20],a['magnitude'][::20],s=2,label=name,alpha=.5);axs[1,1].scatter(a['z'][::20],np.log10(a['median_SN'][::20]),s=2,label=name,alpha=.5)
    for name,a in [('HSC_train',h),*sets.items()]:
        f,e=grizy(a);mag=22.5-2.5*np.log10(np.where(f[:,2]>0,f[:,2],np.nan));sn=np.abs(f[:,2]/e[:,2]);axs[0,1].hist(mag,bins=np.linspace(13,31,40),histtype='step',density=True,label=name);axs[0,2].hist(np.log10(np.maximum(sn,1e-10)),bins=np.linspace(-1,5,40),histtype='step',density=True,label=name)
    for name,v in support.items():axs[1,2].hist(np.log10(v['joint']),bins=np.linspace(-1,3,40),histtype='step',label=name,density=True)
    for ax,title in zip(axs.flat,['Actual eligible source spec-z','Positive i AB magnitude','i S/N','Source redshift vs magnitude','Source redshift vs median S/N','Source-defined joint distance/q95']):ax.set_title(title);ax.legend(fontsize=7)
    axs[1,0].invert_yaxis();axs[1,2].axvline(0,color='gray')
    for ext in ['png','pdf']:fig.savefig(A/f'readiness_coverage.{ext}',dpi=170)
    print('SUPPORT COMPLETE',summaries,flush=True)
if __name__=='__main__':run()
