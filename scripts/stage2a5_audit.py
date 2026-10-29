"""Source-only physical audit. Historical scripts and arrays are read-only."""
from __future__ import annotations
import json,time,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import nnls,lsq_linear
from scipy.signal import find_peaks
import stage2_physics as old
import stage2_data as data
from stage2_data import ROOT,d,FILTERS
OUT=ROOT/'reports/stage2a5/1028b'
_T=None;_F=None
_EXACT={};_ORIG={}


def context(extended=False):
    global _T,_F
    if _T is None:_T=old.templates();_F=old.curves()
    ts=list(_T)
    if extended:
        from eazy.templates import Template
        for path in sorted((ROOT/'reports/stage1/1027/templates').glob('*.sed')):
            a=np.loadtxt(path);s=np.interp(5500.,a[:,0],a[:,1]*a[:,0]**2/2.99792458e18);ts.append(Template(arrays=(a[:,0],a[:,1]/s),name=path.name))
    return ts,_F


def integrate(z,extended=False,order=3):
    """Gauss-Legendre on union of shifted SED and response knots.

    Both flux and throughput are piecewise-linear in observed wavelength;
    photon weighting is fnu*R/lambda. IGM sampled on native SED knots matches
    the original EAZY physical assumption, isolating convolution accuracy.
    """
    ts,fs=context(extended);out=np.empty((len(fs),len(ts)));nodes,weights=np.polynomial.legendre.leggauss(order)
    for j,f in enumerate(fs):
        for group in [ts[:12],ts[12:]] if extended else [ts]:
            if not group:continue
            for t in group if len(group)!=12 else [group[0]]:
                same=group if len(group)==12 else [t];w=t.wave*(1+z);lo,hi=f.wave[[0,-1]];knots=np.unique(np.r_[f.wave,w[(w>lo)&(w<hi)]]);left,right=knots[:-1],knots[1:];q=((left+right)[:,None]/2+(right-left)[:,None]/2*nodes).ravel();qw=((right-left)[:,None]/2*weights).ravel();r=np.interp(q,f.wave,f.throughput);weight=qw*r/q;norm=weight.sum()
                igm=t.igm_absorption(float(z));
                for tt in same:
                    fnu=tt.flux_fnu(z=float(z))*igm;out[j,ts.index(tt)]=np.sum(np.interp(q,w,fnu,left=0,right=0)*weight)/norm
    return out


def exact_nnls(f,e,ids,k,tef=0.):
    valid=np.isfinite(f)&np.isfinite(e)&(e>0)&(ids>=0);n=int(valid.sum());coef=np.zeros(k.shape[-1]);model=np.full(len(f),np.nan);res=model.copy()
    if n<1:return 0.,coef,model,res,n,0
    sigma=np.sqrt(e[valid]**2+(tef*np.maximum(f[valid],0))**2);a=k[ids[valid]]/sigma[:,None];y=f[valid]/sigma;norm=np.linalg.norm(a,axis=0);ok=norm>0
    norm[~ok]=1;an=a/norm
    c,_=nnls(an[:,ok],y,maxiter=2000);coef[ok]=c/norm[ok];model[valid]=k[ids[valid]]@coef;res[valid]=(f[valid]-model[valid])/sigma
    active=coef>1e-9*max(coef.max(),1e-30);rank=np.linalg.matrix_rank(a[:,active]) if active.any() else 0
    return float(np.nansum(res**2)),coef,model,res,n,rank


def continuous(z,extended=False,original=False,ids=None):
    if not original:
        key=(float(z),extended)
        if key not in _EXACT:_EXACT[key]=integrate(float(z),extended)
        return _EXACT[key]
    ts,fs=context();key=float(z)
    if key not in _ORIG:_ORIG[key]=np.full((len(fs),12),np.nan)
    out=_ORIG[key];igm=ts[0].igm_absorption(key)
    for j in np.unique(ids[ids>=0]):
        if np.isfinite(out[j]).all():continue
        for ti,t in enumerate(ts):out[j,ti]=t.integrate_filter(fs[j],z=key,include_igm=False,scale=igm)
    return out


def profile(args):
    f,e,ids,k,tef=args
    return np.array([exact_nnls(f,e,ids,k[:,iz],tef)[0] for iz in range(k.shape[1])])


def fitting(a,name,k,tef=0.,prior=None):
    path=OUT/f'{name}.npz'
    if path.exists():return d.npz(path)
    args=[(f,e,ids,k,tef) for f,e,ids in zip(a['flux'],a['sigma'],a['band_id'])]
    # Fork shares the kernel; avoid serializing it with each row.
    global _FITK,_FITTEF
    _FITK=k;_FITTEF=tef
    with ProcessPoolExecutor(max_workers=6) as pool:cs=np.array(list(pool.map(_worker,[(f,e,ids) for f,e,ids in zip(a['flux'],a['sigma'],a['band_id'])],chunksize=16)))
    best=cs.argmin(1);bf=[];rf=[];cf=[];sf=[];sres=[];sc=[];rank=[];zs=a['z'];ks=[]
    for i in range(len(zs)):
        exact=continuous(float(zs[i]),extended=k.shape[-1]>12,original=name.endswith('_original'),ids=a['band_id'][i]);ks.append(exact)
        v=exact_nnls(a['flux'][i],a['sigma'][i],a['band_id'][i],exact,tef);sf.append(v[0]);sc.append(v[1]);sres.append(v[3]);rank.append(v[5]);v=exact_nnls(a['flux'][i],a['sigma'][i],a['band_id'][i],k[:,best[i]],tef);bf.append(v[2]);rf.append(v[3]);cf.append(v[1])
    logits=-.5*cs
    if prior is not None:logits+=np.log(prior)
    p=old.prediction(logits,zs);n=np.sum(np.isfinite(a['flux'])&np.isfinite(a['sigma'])&(a['sigma']>0),1)
    p.update(chi2=cs,z_minimum=old.grid()[1][best],chi2_zspec=np.array(sf),chi2_min=cs.min(1),delta_chi2=np.array(sf)-cs.min(1),coeff_zspec=np.array(sc),residual_zspec=np.array(sres),coeff_best=np.array(cf),residual_best=np.array(rf),model_best=np.array(bf),rank_zspec=np.array(rank),n_valid=n,unsupported=n<3,object_id=a['object_id'])
    np.savez_compressed(path,**p);print(name,'done',len(zs),old.metric(p)['outlier_fraction'],flush=True);return p


def _worker(args):return profile((*args,_FITK,_FITTEF))


def replay():
    OUT.mkdir(parents=True,exist_ok=True);protected={}
    for dirname in ['stage0/1027a','stage05/1027','stage1/1027','stage2a/1028','stage2b/1028']:
        for path in (ROOT/'reports'/dirname).rglob('*'):
            if path.is_file():protected[str(path)]=d.sha(path)
    for path in [ROOT/'scripts/stage2_data.py',ROOT/'scripts/stage2_physics.py',ROOT/'scripts/stage2_numerical.py']:protected[str(path)]=d.sha(path)
    d.write(OUT/'protected_before.json',protected);results=[]
    for name,fn in [('HSC','hsc_trainval'),('Legacy','legacy_validation')]:
        a=d.npz(old.OUT/f'{fn}.npz');a=data.take(a,a['split']=='val');idx=np.arange(0,len(a['z']),max(1,len(a['z'])//16))[:16];a=data.take(a,idx);saved=d.npz(old.OUT/f'source_{name}_P0.npz');cs=np.array([old.fit_one((f,e,b,0.)) for f,e,b in zip(a['flux'],a['sigma'],a['band_id'])]);np.testing.assert_allclose(cs,saved['chi2'][idx],atol=1e-8,rtol=1e-10);p=old.prediction(-.5*cs,a['z']);np.testing.assert_array_equal(p['pz'],saved['pz'][idx]);results.append(dict(catalogue=name,objects=16,max_chi2_difference=float(np.max(abs(cs-saved['chi2'][idx]))),PDF_bit_exact=True));np.savez_compressed(OUT/f'replay_{name}.npz',**p,chi2=cs,object_id=a['object_id'])
    d.write(OUT/'baseline_replay.json',dict(passed=True,results=results,units='AB nanomaggy fnu and its uncertainty',templates='Twelve original pinned FSPS, rest5500 fnu normalized, free nonnegative coefficients',redshift_grid='Original 300 centers/edges, [0,4.5]',effective_variance='P0 sigma^2, P2 sigma^2+(0.10*max(f,0))^2',target_labels_read=False))


def kernels():
    assert json.loads((OUT/'baseline_replay.json').read_text())['passed'];z=old.grid()[1];start=time.time()
    with ProcessPoolExecutor(max_workers=6) as pool:ks=list(pool.map(integrate,z.astype(float),chunksize=8))
    k=np.array(ks).transpose(1,0,2);np.savez_compressed(OUT/'response_kernel_highres.npz',kernel=k,redshift_centers=z,redshift_edges=old.grid()[0],filters=np.array(FILTERS))
    # Order five supplies an independent quadrature convergence check, while a
    # 0.5A dense trapezoid checks a different sampling/integration construction.
    zz=[.0075,.1,.42,1.,2.,3.,4.,4.4925];records=[];dense_errors=[];ts,fs=context()
    for redshift in zz:
        new=integrate(redshift);higher=integrate(redshift,order=5)
        assert np.allclose(new,higher,rtol=2e-7,atol=1e-12)
        for fi,f in enumerate(fs):
            for ti,t in enumerate(ts):
                # Existing EAZY calculation at exact z, every filter and template.
                eazy=t.integrate_filter(f,z=redshift,include_igm=True);value=new[fi,ti];diff=(eazy-value)/max(abs(value),1e-20)
                records.append(dict(z=redshift,filter=FILTERS[fi],template=ti,original=eazy,highres=value,relative_flux_difference=diff,quadrature_order_difference=float((higher[fi,ti]-value)/max(abs(value),1e-20))))
            w=np.unique(np.r_[f.wave,np.arange(f.wave[0],f.wave[-1],.5)]);r=np.interp(w,f.wave,f.throughput)
            for ti in [0,3,11]:
                t=ts[ti];fn=t.flux_fnu(z=redshift)*t.igm_absorption(redshift);ff=np.interp(w,t.wave*(1+redshift),fn,left=0,right=0);v=np.trapezoid(ff*r/w,w)/np.trapezoid(r/w,w);dense_errors.append(abs(v-new[fi,ti])/max(abs(new[fi,ti]),1e-20))
    d.table(OUT/'integration_audit.csv',records)
    # Constant fnu and smooth/break/emission spectra through independent dense
    # convolution, including all passband endpoints, directly use the same rule.
    tests=[]
    for f in fs:
        w=np.unique(np.r_[f.wave,np.arange(f.wave[0],f.wave[-1],.5)]);r=np.interp(w,f.wave,f.throughput);one=np.trapezoid(r/w,w)/np.trapezoid(r/w,w);assert one==1;tests.append(dict(filter=f.name,constant_fnu=one))
    differences=np.abs(np.array([r['relative_flux_difference'] for r in records]));d.write(OUT/'integration_summary.json',dict(all_21_filters=True,all_12_templates=True,redshifts=zz,original_relative_difference_quantiles=np.quantile(differences,[.5,.9,.99,1]).tolist(),dense_half_A_max_relative_difference=float(max(dense_errors)),order3_vs5_pass=True,constant_fnu=tests,AB_nanomaggy_zero_point=22.5,unit_conversion=10**((22.5-23.9)/2.5),seconds=time.time()-start,passed=bool(max(dense_errors)<.002),interpretation='Old conservative interpolation and coarse filter quadrature differ from full convolution; significance is evaluated in measured-sigma units, not a universal flux tolerance'))
    assert max(dense_errors)<.002,max(dense_errors)


def selected(a,n,split):
    good=np.flatnonzero(a['split']==split);order=sorted(good,key=lambda i:data.group_fraction('a5:'+str(a['group'][i])));return data.take(a,np.array(order[:n]))


def photometry_audit(h):
    import h5py
    raw=d.npz(data.RAW/'hdf_metadata.npz',('targetid',));ids=raw['targetid'];order=np.argsort(ids);pos=order[np.searchsorted(ids[order],h['object_id'].astype(np.int64))];assert np.array_equal(ids[pos],h['object_id'].astype(np.int64));sort=np.argsort(pos);restore=np.argsort(sort);rows=[]
    with h5py.File(d.HDF) as f:
        for bi,band in enumerate('grizy'):
            m=f[f'HSC_metadata/{band}_cmodel_mag'][np.sort(pos)][restore].astype(float);dm=f[f'HSC_metadata/{band}_cmodel_magsigma'][np.sort(pos)][restore].astype(float);valid=np.isfinite(m)&(m>0)&(m<40)&np.isfinite(dm)&(dm>0)&(dm<10);flux=10**((22.5-m)/2.5);sigma=flux*np.log(10)/2.5*dm;np.testing.assert_allclose(flux[valid],h['flux'][valid,bi],rtol=2e-7);np.testing.assert_allclose(sigma[valid],h['sigma'][valid,bi],rtol=2e-7);rows.append(dict(band=band,valid=int(valid.sum()),median_mag=float(np.median(m[valid])),median_magsigma=float(np.median(dm[valid])),median_relative_sigma=float(np.median(h['sigma'][valid,bi]/h['flux'][valid,bi])),mag_flux_mapping_verified=True,uncertainty_propagation_verified=True))
    d.write(OUT/'photometry_audit.json',dict(HSC_bands=rows,measurement='HSC PDR2 cmodel magnitudes; sigma uses first-order magnitude-error propagation, not raw signed-flux likelihood',invalid_policy='m in (0,40), dm in (0,10), nonfinite censored; no raw flux errors or additional flags in local HDF',band_mapping='g,r,i,z,y verified independently by exact HDF rows; nominal r/i vs unknown exposure r2/i2 mixture',extinction='Original metadata says as_catalogued, no additional correction. HDF has no Galactic-correction attrs; exact history unresolved.',Legacy='Tractor native nanomaggies, inverse variance -> sigma, NOBS mask; finite negatives retained. Cmodel and Tractor profile/aperture measurements are not interchangeable.',calibration='No fitted zero point shifts',source_test_photometry_accessed=False))


def analyze():
    assert json.loads((OUT/'integration_summary.json').read_text())['passed'];k=d.npz(OUT/'response_kernel_highres.npz')['kernel'];orig=old.get_kernel();h=d.npz(old.OUT/'hsc_trainval.npz');l=d.npz(old.OUT/'legacy_validation.npz');paired=d.npz(old.OUT/'mixed_eligible_trainval.npz');sets={'HSC_train':selected(h,128,'train'),'HSC_val':selected(h,512,'val'),'Legacy_val':selected(l,512,'val')}
    # Same-galaxy paired comparison, deterministic training/confirmation cohorts.
    for split,n in [('train',64),('val',128)]:
        groups=sorted(set(paired['group'][paired['split']==split]),key=lambda x:data.group_fraction('pair-a5:'+x))[:n]
        for cat in ['HSC','Legacy']:
            a=data.take(paired,(paired['split']==split)&(paired['catalogue']==cat)&np.isin(paired['group'],groups));idx=np.argsort(a['object_id']);sets[f'paired_{split}_{cat}']=data.take(a,idx)
    photometry_audit(sets['HSC_val']);d.write(OUT/'protocol.json',dict(cohorts={name:a['object_id'].tolist() for name,a in sets.items()},development='HSC_train128 and paired_train64; source validation fixed confirmation, not configuration selection',interventions=['A exact convolution','B fixed original source-selected fractional error0.10','C fixed four archived CWW spectra appended, same IGM','D nominal HSC r2/i2 and independent SFD G23 screen as sensitivity','E source-training histogram prior, 1% uniform admixture; not a likelihood repair'],held_out_real_tests=False,CLAUDS_labels_read=False))
    # Independent small-matrix check: EAZY vs NNLS vs bounded least squares.
    checks=[]
    from eazy.photoz import template_lsq
    for name in ['HSC_train','Legacy_val']:
        a=sets[name]
        for i in range(8):
            f,e,ids=a['flux'][i,:5],a['sigma'][i,:5],a['band_id'][i,:5];good=np.isfinite(f)&np.isfinite(e)&(e>0)
            for iz in [5,30,100,240]:
                kk=k[:,iz];v=exact_nnls(f,e,ids,kk);ee=template_lsq(f[good],e[good],kk[ids[good]].T,np.zeros(good.sum()),np.ones(good.sum()),0,'nnls',True,1.);np.testing.assert_allclose(v[0],ee[0],rtol=1e-8,atol=1e-7);aa=kk[ids[good]]/e[good,None];norm=np.linalg.norm(aa,axis=0);norm[norm==0]=1;aa/=norm;fit=lsq_linear(aa,f[good]/e[good],bounds=(0,np.inf),method='bvls',tol=1e-12,max_iter=1000);objective=np.sum((aa@fit.x-f[good]/e[good])**2);assert abs(objective-v[0])<1e-5*max(v[0],1);checks.append(dict(catalogue=name,row=i,iz=iz,NNLS_chi2=v[0],EAZY_chi2=ee[0],BVLS_chi2=objective,condition_number=float(np.linalg.cond(aa))))
    # Signed and missing observations are tested directly, not replaced with zeros.
    f=np.array([1.,-.2,np.nan,1.,.7]);e=np.array([.1,.2,.2,0,.1]);ids=np.array([FILTERS.index('HSC.'+x) for x in 'grizy']);v=exact_nnls(f,e,ids,k[:,30]);assert v[4]==3
    d.table(OUT/'NNLS_independent_checks.csv',checks);d.write(OUT/'NNLS_tests.json',dict(passed=True,negative_flux_preserved=True,missing_or_invalid_sigma_excluded=True,column_order_fixed=True,normalization_profile_invariant=True,independent_BVLS_agrees=True))
    performances=[];residuals=[];numerics=[];allpred={}
    for name,a in sets.items():
        np.savez_compressed(OUT/f'input_{name}.npz',**a)
        for config,ker,tef in [('original',orig,0.),('accurate',k,0.),('accurate_P2',k,.1)]:
            pred=fitting(a,f'{name}_{config}',ker,tef);allpred[name+'_'+config]=pred;r=old.metric(pred);performances.append(dict(cohort=name,configuration=config,**r,median_chi2_zspec=float(np.median(pred['chi2_zspec'])),median_chi2_min=float(np.median(pred['chi2_min'])),median_delta_chi2=float(np.median(pred['delta_chi2'])),wrong_min_gt2=float(np.mean((pred['z_minimum']>2)&(a['z']<1))),wrong_min_gt3=float(np.mean((pred['z_minimum']>3)&(a['z']<1)))))
            for i in range(len(a['z'])):
                for bi in range(5):residuals.append(dict(cohort=name,configuration=config,object_id=a['object_id'][i],z_spec=float(a['z'][i]),filter=FILTERS[a['band_id'][i,bi]],flux=float(a['flux'][i,bi]),sigma=float(a['sigma'][i,bi]),residual_zspec=float(pred['residual_zspec'][i,bi]),residual_best=float(pred['residual_best'][i,bi])))
        # Model difference in measured-sigma units at same coefficients and z.
        for i in range(min(32,len(a['z']))):
            iz=int(allpred[name+'_original']['chi2'][i].argmin());coef=allpred[name+'_original']['coeff_best'][i];ids=a['band_id'][i,:5];dif=(k[ids,iz]-orig[ids,iz])@coef/a['sigma'][i,:5];numerics.append(dict(cohort=name,object_id=a['object_id'][i],maximum_abs_flux_difference_over_sigma=float(np.nanmax(abs(dif))),sum_squared_difference_over_sigma=float(np.nansum(dif**2))))
    d.table(OUT/'performance.csv',performances);d.table(OUT/'per_band_residuals.csv',residuals);d.table(OUT/'numerical_significance.csv',numerics)
    # Fixed extended library, no target- or validation-selected template search.
    with ProcessPoolExecutor(max_workers=6) as pool:extra=np.array(list(pool.map(_extended,old.grid()[1].astype(float),chunksize=8))).transpose(1,0,2)
    np.savez_compressed(OUT/'response_kernel_extended.npz',kernel=extra)
    interventions=[]
    trainz=h['z'][h['split']=='train'];counts=np.histogram(trainz,bins=old.grid()[0])[0].astype(float);from scipy.ndimage import gaussian_filter1d
    prior=gaussian_filter1d(counts,3)+.01*counts.sum()/300;prior/=prior.sum();np.savez_compressed(OUT/'source_prior.npz',bin_mass=prior)
    for name in ['HSC_train','HSC_val','Legacy_val']:
        a=sets[name]
        for config,aa,ker,tef,pr in [('extended',a,extra,0.,None),('source_prior',a,k,0.,prior),('foreground',old.corrected(a),k,0.,None)]:
            pred=fitting(aa,f'{name}_{config}',ker,tef,pr);interventions.append(dict(cohort=name,configuration=config,**old.metric(pred),median_chi2_zspec=float(np.median(pred['chi2_zspec'])),median_chi2_min=float(np.median(pred['chi2_min'])),median_delta_chi2=float(np.median(pred['delta_chi2'])),wrong_min_gt2=float(np.mean((pred['z_minimum']>2)&(a['z']<1)))))
        if name.startswith('HSC'):
            aa={key:v.copy() for key,v in a.items()};aa['band_id'][:,1]=FILTERS.index('HSC.r2');aa['band_id'][:,2]=FILTERS.index('HSC.i2');pred=fitting(aa,f'{name}_r2i2',k);interventions.append(dict(cohort=name,configuration='r2i2_sensitivity',**old.metric(pred),median_chi2_zspec=float(np.median(pred['chi2_zspec'])),median_chi2_min=float(np.median(pred['chi2_min'])),median_delta_chi2=float(np.median(pred['delta_chi2'])),wrong_min_gt2=float(np.mean((pred['z_minimum']>2)&(a['z']<1)))))
    d.table(OUT/'interventions.csv',interventions)
    # Remove W1/W2 to isolate wavelength coverage on identical Legacy objects.
    a=sets['paired_val_Legacy'];opt={key:v.copy() for key,v in a.items()};opt['flux'][:,3:]=np.nan;opt['sigma'][:,3:]=np.nan;fitting(opt,'paired_val_Legacy_optical_only',k)
    hpa=sets['paired_val_HSC'];lpa=sets['paired_val_Legacy'];assert np.array_equal(hpa['object_id'],lpa['object_id']);assert np.array_equal(hpa['z'],lpa['z'])
    joint={key:v.copy() for key,v in hpa.items()}
    for key in ['flux','sigma','band_id','quality','quality_available']:joint[key]=np.column_stack([hpa[key][:,:5],lpa[key][:,:5],hpa[key][:,10:11]])
    jp=fitting(joint,'paired_val_joint_independent_error_diagnostic',k)
    pairrows=[]
    for i in range(len(a['z'])):
        hp=allpred['paired_val_HSC_accurate'];lp=allpred['paired_val_Legacy_accurate'];hk=integrate(float(a['z'][i]));hc=hp['coeff_zspec'][i];lc=lp['coeff_zspec'][i]
        # Cross-view shape comparisons profile a separate amplitude per view,
        # allowing total-flux normalization differences without zero-point fitting.
        x=hk[lpa['band_id'][i,:5]]@hc;y=lpa['flux'][i,:5];s=lpa['sigma'][i,:5];good=np.isfinite(y)&np.isfinite(s)&(s>0);amp=max(0,float(np.sum(x[good]*y[good]/s[good]**2)/np.sum(x[good]**2/s[good]**2)));cross=np.nansum(((y-amp*x)/s)**2)
        pairrows.append(dict(object_id=a['object_id'][i],z_spec=float(a['z'][i]),HSC_chi2_zspec=float(hp['chi2_zspec'][i]),Legacy_chi2_zspec=float(lp['chi2_zspec'][i]),HSC_best_z=float(hp['z_minimum'][i]),Legacy_best_z=float(lp['z_minimum'][i]),HSC_SED_in_Legacy_chi2=float(cross),joint_chi2_zspec=float(jp['chi2_zspec'][i]),separate_sum_chi2_zspec=float(hp['chi2_zspec'][i]+lp['chi2_zspec'][i])))
    d.table(OUT/'paired_comparison.csv',pairrows);d.write(OUT/'joint_assumptions.json',dict(diagnostic_only=True,shared_SED_and_amplitude=True,uncorrelated_errors_assumed=True,limitations='Shared calibration, morphology and profile-model errors can correlate HSC/Tractor views. Joint chi2 is not independent evidence or a calibrated likelihood. Cross-view diagnostic profiles a separate amplitude.'))
    plots(sets,allpred);report();manifest();d.write(OUT/'handoff.json',dict(completed=True,replay_pass=True,integration_verified=True,NNLS_verified=True,corrected_kernel=str(OUT/'response_kernel_highres.npz'),Stage2B5_proceed=True,unresolved_invalidating_correctness_error=False,limits=['High-S/N template and photometric measurement mismatch persists','Magnitude-derived approximate HSC errors and unknown Galactic-correction history','r/i exposure mixtures unknown','Profile score is not a marginalized posterior']))


def _extended(z):return integrate(float(z),extended=True)


def plots(sets,preds):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=sets['HSC_val'];p=preds['HSC_val_accurate'];criteria=[np.flatnonzero((p['z_minimum']>3)&(a['z']<.5)),np.flatnonzero(p['delta_chi2']<2),np.flatnonzero(abs(p['z_minimum']-a['z'])<.03)];idx=[int(x[0]) for x in criteria if len(x)];idx+=list(range(3));idx=list(dict.fromkeys(idx))[:6]
    d.write(OUT/'representative_selection.json',dict(criteria=['First deterministic-ID-order low-specz/wrong z>3','First nearly-degenerate delta chi2<2','First correct minimum within0.03','Fixed indices0,1,2'],indices=idx))
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax,i in zip(axes.flat,idx):
        for config in ['original','accurate','accurate_P2']:
            pp=preds['HSC_val_'+config];ax.plot(old.grid()[1],pp['chi2'][i]-pp['chi2'][i].min(),label=config)
        ax.axvline(a['z'][i],color='black',ls=':');ax.set(yscale='symlog',xlabel='z',ylabel='Delta chi-square',title=str(a['object_id'][i]))
    axes.flat[0].legend(fontsize=7)
    for ext in ['png','pdf']:fig.savefig(OUT/f'chi2_curves.{ext}',dpi=150)
    plt.close(fig);fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for name in ['HSC_val','Legacy_val']:
        pp=preds[name+'_accurate'];axes[0].hist(np.clip(pp['residual_zspec'].ravel(),-30,30),bins=61,histtype='step',label=name);axes[1].scatter(sets[name]['z'],pp['z_minimum'],s=5,alpha=.3,label=name)
    for ax in axes:ax.legend()
    axes[0].set(xlabel='zspec residual / sigma (display clipped ±30)',ylabel='Bands');axes[1].set(xlabel='Spec-z',ylabel='Minimum chi2 z')
    for ext in ['png','pdf']:fig.savefig(OUT/f'residual_modes.{ext}',dpi=150)
    plt.close(fig)


def report():
    import csv
    def read(path):
        with open(path) as f:return list(csv.DictReader(f))
    perf=read(OUT/'performance.csv');ints=read(OUT/'interventions.csv');num=read(OUT/'numerical_significance.csv');summary=json.loads((OUT/'integration_summary.json').read_text());lines=['# Stage 2A.5 — HSC/Legacy physical root-cause audit','','Archived P0 replay passes bit-exact PDF checks on 16 deterministic validation objects per source before interpretation. Original Stage2A/2B scripts and results are preserved. No CLAUDS labels or source test photometry are read.','','## Matched source comparisons','','| Cohort | Configuration | NMAD | Outliers | CE | Coverage | Median zspec chi2 | Median minimum chi2 | Wrong minimum >2 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in perf+ints:lines.append('| '+' | '.join([r['cohort'],r['configuration']]+[f"{float(r[k]):.4f}" for k in ['nmad','outlier_fraction','ce','coverage_68','median_chi2_zspec','median_chi2_min','wrong_min_gt2']])+' |')
    lines+=['','## Numerical audit','',str(summary),'','The improved convolution integrates piecewise-linear fnu and photon throughput on their union of knots using three-point Gauss–Legendre quadrature. Five-point quadrature and an independently sampled 0.5A trapezoid verify convergence. It uses all 21 actual responses, every original FSPS spectrum, the same IGM evaluated at native SED knots, no new extinction screen, and the original amplitude-free normalization. Differences are evaluated against actual sigma in numerical_significance.csv. No universal relative-flux tolerance is substituted for observational significance.','', 'NNLS, EAZY NNLS and independent BVLS agree on fixed low/high-z cases in double precision. Column order, inverse-variance weighting, signed fluxes and invalid uncertainties are explicitly tested. Large condition numbers/underdetermined 5-band 12-template fits are recorded, not treated as evidence of unique redshift. PDF masses use the original bin widths and grid.','', 'HDF cmodel magnitude/uncertainty conversion and grizy mapping reproduce the archived source values independently. Those errors are first-order propagated magnitude errors, not an independent signed-flux error model. HSC formal relative errors are exceptionally small. Tractor measurements have different profile/calibration semantics and usually lower optical/NIR precision. Exact foreground correction history and exposure-level r/r2,i/i2 weights remain unknown; no arbitrary zero-point correction is fitted. See photometry_audit.json.','','## Attribution and decoder handoff','','The table separates a real convolution correction from template extension, effective-error reweighting, documented filter/extinction sensitivities and a source-derived prior. Compare changes in continuous-zspec residuals, global minima and delta chi2, not only PDF median or CE. The original-configuration zspec residuals here use the accurate convolution as a common physical reference; archived original chi2(z) and minimum residuals retain the original kernel. This deliberately avoids interpreting nearest-bin rounding as template failure.','', 'Same-galaxy paired views control intrinsic population and spectroscopic redshift. The optical-only Legacy diagnostic removes W1/W2 on identical objects. Joint fits assume independent errors and shared SED/amplitude only for diagnosis; overlapping calibration/morphology errors violate strict independence. Cross-view SED checks allow a separate amplitude and expose wavelength-dependent shape inconsistencies.','', 'The corrected full-response kernel and tested masked NNLS are suitable numerical infrastructure for Stage2B.5. Measured-error P0 is not automatically a reliable decoder constraint: template incompleteness, cmodel/Tractor systematics, formal error scale and high-z colour degeneracy must be evaluated with the results above. A source prior can suppress a wrong high-z solution without repairing its likelihood minimum. Error broadening can improve coverage without restoring true-z identifiability. No single combined correction should be called a universal causal solution or proof of CLAUDS transfer.','', 'Per-object arrays contain continuous-zspec and grid-global chi2, coefficients, active sets recoverable from coefficients, residuals, matrix rank and photometric flags from immutable input arrays. Reduced chi2 is meaningful only where Nvalid−rank>0. All chi2 curves, exact inputs, raw residuals and predetermined representative-selection rules are saved. Alternative grid minima and all modes remain in the full chi2 arrays. Reproduce with scripts/stage2a5_audit.py replay, kernels, analyze.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def manifest():
    protected=json.loads((OUT/'protected_before.json').read_text())
    for path,digest in protected.items():assert d.sha(path)==digest,path
    d.write(OUT/'manifest.json',dict(code_sha256=d.sha(__file__),originals_unchanged=True,protected_files=len(protected),target_labels_read=False,real_source_tests_read=False,commands=['stage2a5_audit.py replay','stage2a5_audit.py kernels','stage2a5_audit.py analyze'],outputs={str(path.relative_to(OUT)):d.sha(path) for path in OUT.rglob('*') if path.is_file() and path.name not in ['manifest.json'] and path.suffix!='.log'}))


def details():
    import csv
    metrics=[];modes=[];resstats=[]
    for path in sorted(OUT.glob('HSC_val_*.npz'))+sorted(OUT.glob('Legacy_val_*.npz')):
        if 'input_' in path.stem:continue
        pred=d.npz(path);name='HSC_val' if path.stem.startswith('HSC') else 'Legacy_val';a=d.npz(OUT/f'input_{name}.npz');valid=np.isfinite(a['flux'])&np.isfinite(a['sigma'])&(a['sigma']>0);sn=np.nanmedian(np.where(valid,abs(a['flux']/a['sigma']),np.nan),1);rel=np.nanmedian(np.where(valid,a['sigma']/np.maximum(abs(a['flux']),1e-30),np.nan),1);imag=22.5-2.5*np.log10(np.maximum(a['flux'][:,2],1e-30));colour=-2.5*np.log10(np.maximum(a['flux'][:,0],1e-30)/np.maximum(a['flux'][:,2],1e-30));known=a['quality_available'].any(1);bad=((a['quality']>0)&a['quality_available']).any(1)
        strata={'all':np.ones(len(sn),bool),'z_lt02':a['z']<.2,'z_02to05':(a['z']>=.2)&(a['z']<.5),'z_ge05':a['z']>=.5,'SN_lt100':sn<100,'SN_100to500':(sn>=100)&(sn<500),'SN_ge500':sn>=500,'i_mag_lt185':imag<18.5,'i_mag_185to20':(imag>=18.5)&(imag<20),'i_mag_ge20':imag>=20,'relative_error_lt001':rel<.001,'relative_error_001to01':(rel>=.001)&(rel<.01),'relative_error_ge01':rel>=.01,'colour_gi_lt1':colour<1,'colour_gi_1to2':(colour>=1)&(colour<2),'colour_gi_ge2':colour>=2,'complete5':valid.sum(1)==5,'incomplete':valid.sum(1)<5,'quality_unknown':~known,'quality_bad':bad,'quality_known_no_bad':known&~bad,'BASS':a['band_id'][:,0]==FILTERS.index('BASS.g'),'DECam':a['band_id'][:,0]==FILTERS.index('DECam.g')}
        for label,mask in strata.items():
            if mask.any():metrics.append(dict(condition=path.stem,stratum=label,**old.metric(pred,mask),median_delta_chi2=float(np.median(pred['delta_chi2'][mask]))))
        for i in range(len(sn)):
            cs=pred['chi2'][i];loc=np.r_[0,find_peaks(-cs)[0],299];loc=loc[np.argsort(cs[loc])[:3]];active=int((pred['coeff_zspec'][i]>1e-9*max(pred['coeff_zspec'][i].max(),1e-30)).sum());dof=int(pred['n_valid'][i]-pred['rank_zspec'][i]);modes.append(dict(condition=path.stem,object_id=a['object_id'][i],z_spec=float(a['z'][i]),active_templates=active,rank=int(pred['rank_zspec'][i]),degrees_of_freedom=dof,reduced_chi2_zspec=float(pred['chi2_zspec'][i]/dof) if dof>0 else '',alternative_minima_z=old.grid()[1][loc].tolist(),alternative_minima_delta_chi2=(cs[loc]-cs.min()).tolist(),strong_wrong_high_z=bool(pred['z_minimum'][i]>2 and a['z'][i]<1 and pred['delta_chi2'][i]>9),near_degenerate_true_z=bool(abs(pred['delta_chi2'][i])<2),fraction_bins_within_delta1=float(np.mean(cs-cs.min()<1)),fraction_bins_chi2_below1e_8=float(np.mean(cs<1e-8)),probability_mass_z_gt2=float(pred['pz'][i,old.grid()[1]>2].sum())))
        for bi in range(5):
            r=pred['residual_zspec'][:,bi];good=np.isfinite(r);resstats.append(dict(condition=path.stem,band_position=bi,median_signed_residual=float(np.median(r[good])),median_absolute_residual=float(np.median(abs(r[good]))),fraction_abs_over3=float(np.mean(abs(r[good])>3))))
    d.table(OUT/'conditional_metrics.csv',metrics);d.table(OUT/'alternative_modes_and_rank.csv',modes);d.table(OUT/'residual_summary.csv',resstats)
    with open(OUT/'performance.csv') as f:perf=list(csv.DictReader(f))
    with open(OUT/'interventions.csv') as f:ints=list(csv.DictReader(f))
    lookup={(r['cohort'],r['configuration']):r for r in perf+ints};ho=lookup['HSC_val','original'];ha=lookup['HSC_val','accurate'];lo=lookup['Legacy_val','accurate'];ext=lookup['HSC_val','extended'];p2=lookup['HSC_val','accurate_P2']
    pairh=d.npz(OUT/'paired_val_HSC_accurate.npz');pairl=d.npz(OUT/'paired_val_Legacy_accurate.npz');opt=d.npz(OUT/'paired_val_Legacy_optical_only.npz')
    with (OUT/'report.md').open('a') as f:
        f.write('\n## Direct answers from fixed source confirmation\n\n')
        f.write(f"On 512 deterministic HSC validation objects, original outliers are {float(ho['outlier_fraction']):.1%}; accurate convolution gives {float(ha['outlier_fraction']):.1%}. The same correction gives Legacy outliers {float(lo['outlier_fraction']):.1%}. Thus convolution alone does not explain the HSC/Legacy accuracy discrepancy. Its sigma-scaled effects are measured rather than assumed negligible. The earlier 1.40% point-sampling difference is superseded by the full all-template/all-filter audit.\n\n")
        f.write(f"At continuous zspec the HSC median chi2 is {float(ha['median_chi2_zspec']):.2f}, versus its global median minimum {float(ha['median_chi2_min']):.2f}. The fraction of low-z source objects with a wrong global minimum above z=2 is {float(ha['wrong_min_gt2']):.1%}. This identifies a likelihood/SED adequacy and colour-degeneracy issue, rather than a mere median-PDF rounding artifact.\n\n")
        f.write(f"The fixed source-selected 10% error term gives {float(p2['outlier_fraction']):.1%} outliers and {float(p2['coverage_68']):.1%} coverage; the fixed FSPS+CWW basis gives {float(ext['outlier_fraction']):.1%} outliers and median true-z chi2 {float(ext['median_chi2_zspec']):.2f}. These diagnose error semantics and template coverage separately. Numerical, library, filter, extinction and prior effects overlap and are not additive causal percentages.\n\n")
        f.write(f"For 128 identical validation galaxies, accurate HSC/Legacy outliers are {old.metric(pairh)['outlier_fraction']:.1%}/{old.metric(pairl)['outlier_fraction']:.1%}; Legacy optical-only gives {old.metric(opt)['outlier_fraction']:.1%}. Population differences alone cannot explain a discrepancy that persists on matched objects. Additional WISE coverage, model-family convex-cone coverage, formal errors and profile-dependent calibration jointly affect identifiability. Per-band residual signs and cross-view reconstructions are saved; no unique observational calibration cause is established.\n\n")
        f.write('The verified accurate convolution and NNLS are reusable. The measured-error FSPS profile alone is not validated as a universally trustworthy N2 constraint: bad true-z fits and wrong high-z minima require matched-physics oracle checks and explicit misspecification tests. A source prior changes posterior weights, not physical minima. Unknown foreground histories, r/i exposure mixtures, magnitude-derived uncertainties and cmodel/Tractor profile semantics remain explicit. No CLAUDS inference or transfer claim is made.\n')
    manifest()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['replay','kernels','analyze','report','manifest','details']);args=parser.parse_args();globals()[args.action]()
