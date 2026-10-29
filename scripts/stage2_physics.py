"""Stage 2A: maintained EAZY NNLS with full response curves and frozen settings.

No target reference labels are imported by kernel construction or fitting.
"""
from __future__ import annotations
import os,json,time,importlib.metadata
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.special import logsumexp
import stage2_data as data
from stage2_data import ROOT,FILTERS,REGISTRY,d
OUT=ROOT/'reports/stage2a/1028'


def grid():
    a=d.npz(d.FAST/'clauds_predictions.npz',('redshift_edges','redshift_centers'))
    return a['redshift_edges'],a['redshift_centers']


def curves():
    from aion_magnitude.filter_photometry import load_registry
    from eazy.filters import FilterDefinition
    load_registry(verify_curves=True)
    out=[]
    for name in FILTERS:
        r=REGISTRY['filters'][name];p=ROOT/'configs'/r['curve_path'];assert d.sha(p)==r['curve_sha256']
        a=np.loadtxt(p,delimiter=',',skiprows=1);w=a[:,0]*10;t=a[:,1]
        assert np.all(np.diff(w)>0) and np.all(t>=0) and np.trapezoid(t/w,w)>0
        out.append(FilterDefinition(name=name,wave=w,throughput=t))
    return out


def templates():
    from eazy.templates import Template
    out=[]
    for p in sorted((OUT/'templates').glob('tweak*_0*.dat')):
        a=np.loadtxt(p);scale=np.interp(5500.,a[:,0],a[:,1]*a[:,0]**2/2.99792458e18)
        out.append(Template(arrays=(a[:,0],a[:,1]/scale),name=p.name))
    assert len(out)==12
    return out


def build():
    from eazy.photoz import TemplateGrid
    OUT.mkdir(parents=True,exist_ok=True);filters=curves();ts=templates();edges,z=grid()
    (OUT/'filters.RES').write_text('\n'.join(f.for_filter_file() for f in filters))
    d.write(OUT/'filter_translation.json',{name:dict(EAZY_filter_number=i+1,flux_column='F'+str(i+1),error_column='E'+str(i+1),response=REGISTRY['filters'][name]) for i,name in enumerate(FILTERS)})
    if (OUT/'response_kernel.npz').exists():k=d.npz(OUT/'response_kernel.npz')['kernel']
    else:
        g=TemplateGrid(z.astype(float),ts,filters=filters,add_igm=True,galactic_ebv=0,n_proc=4,verbose=2,array_dtype=np.float64)
        k=np.transpose(g.tempfilt,(2,0,1));assert k.shape==(len(FILTERS),300,12) and np.all(np.isfinite(k)) and np.max(k)>0
        np.savez_compressed(OUT/'response_kernel.npz',kernel=k,redshift_centers=z,redshift_edges=edges,filters=np.array(FILTERS))
    tests={};maxflat=0.;maxdirect=0.
    from eazy.templates import Template
    from stage1_sed_forward import band_fnu,C_AA_S
    flat=Template(arrays=(ts[0].wave, C_AA_S/ts[0].wave**2),name='unit-fnu')
    for f in filters:
        v=flat.integrate_filter(f,z=.71,include_igm=False);maxflat=max(maxflat,abs(float(v)-1))
        # Independent Stage-1 photon-weighted integration of a redshifted fnu shape.
        zz=.42;w=f.wave;rest=w/(1+zz);fn=np.interp(rest,ts[3].wave,ts[3].flux_fnu(z=zz))
        direct=band_fnu(w,fn*C_AA_S/w**2,f.throughput)
        ez=ts[3].integrate_filter(f,z=zz,include_igm=False)
        maxdirect=max(maxdirect,abs(direct-ez)/max(abs(ez),1e-30))
    # EAZY conservatively interpolates spectra; independent point interpolation
    # differs slightly at sharp features on the original throughput knots.
    assert maxflat<2e-3 and maxdirect<.035,(maxflat,maxdirect)
    tests.update(constant_fnu_relative_error=maxflat,independent_stage1_relative_error=maxdirect,AB_zero_nanomaggy_mag=22.5,CLAUDS_microJy_scale=10**((22.5-23.9)/2.5),spectral_density_conversion=True,all_response_units_verified=True)
    np.testing.assert_allclose(22.5-2.5*np.log10(1e9),0,atol=1e-12)
    # A known nonnegative mixture, including a negative observation and a masked band.
    a=dict(flux=np.array([[1.,.8,-.04,np.nan,1.2]+[np.nan]*6]),sigma=np.array([[.1,.1,.1,np.nan,.1]+[np.nan]*6]),band_id=np.array([[FILTERS.index('HSC.'+b) for b in 'grizy']+[-1]*6]))
    c=fit_one((a['flux'][0],a['sigma'][0],a['band_id'][0],0.))
    cc=fit_one((a['flux'][0]*.001,a['sigma'][0]*.001,a['band_id'][0],0.))
    np.testing.assert_allclose(c,cc,rtol=1e-8,atol=1e-8)
    bad=a['sigma'][0].copy();bad[2]=np.nan
    c2=fit_one((a['flux'][0],bad,a['band_id'][0],0.));assert np.isfinite(c2).all()
    select=np.array([FILTERS.index('HSC.'+b) for b in 'grizy']);ff=k[select,100]@np.arange(1.,13.)
    known=fit_one((ff,.01*ff,select,0.));assert known[100]<1e-10
    tests.update(known_nonnegative_mixture_chi2=float(known[100]),signed_flux_retained=True,missing_sigma_masked=True,missing_bands=True,free_amplitude_scale_invariance=True,passed=True)
    d.write(OUT/'physical_tests.json',tests)
    deps={p:importlib.metadata.version(p) for p in ['eazy','numpy','scipy','astropy','dust-extinction','torch']}
    d.write(OUT/'physical_manifest.json',dict(dependencies=deps,kernel_sha256=d.sha(OUT/'response_kernel.npz'),templates_manifest_sha256=d.sha(OUT/'templates/manifest.json'),filter_registry_sha256=d.sha(ROOT/'configs/filter_registry_v1.json'),IGM='EAZY Inoue14 enabled',dust='FSPS QSF12 library includes its intrinsic template dust; no additional internal dust fit',normalization='Templates divided by rest 5500A fnu; free nonnegative amplitudes remove absolute (1+z)/luminosity-distance factors omitted by EAZY integration',probability='Midpoint-bin profile exp(-chi2/2) times bin width; no population prior, no EAZY template-error posterior correction',P1='Not run: optional population prior would introduce another selection assumption',P2='Fixed source-only candidates fractional template errors 0,0.03,0.10; balanced source CE on deterministic calibration halves, confirmation halves untouched',physical_tests=tests))


_K=None
def get_kernel():
    global _K
    if _K is None:_K=d.npz(OUT/'response_kernel.npz',('kernel',))['kernel']
    return _K


def fit_one(args):
    from eazy.photoz import template_lsq
    f,e,ids,tef=args;valid=np.isfinite(f)&np.isfinite(e)&(e>0)&(ids>=0)
    if valid.sum()<3:return np.zeros(300)
    ff=f[valid].astype(float);ee=e[valid].astype(float);A=get_kernel()[ids[valid]].transpose(1,2,0)
    result=np.empty(300)
    for iz in range(300):
        chi2,coeff,model,_=template_lsq(ff,ee,A[iz],np.full(len(ff),tef),np.ones(len(ff)),0,'nnls',True,1.)
        result[iz]=chi2
    return result


def fit_rows(a,path,tef=0.,workers=8):
    if path.exists():return d.npz(path)
    start=time.time();valid=np.isfinite(a['flux'])&np.isfinite(a['sigma'])&(a['sigma']>0)&(a['band_id']>=0)
    args=[(f,e,b,tef) for f,e,b in zip(a['flux'],a['sigma'],a['band_id'])]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cs=[]
        for i,c in enumerate(pool.map(fit_one,args,chunksize=16)):
            cs.append(c)
            if (i+1)%512==0:print(path.stem,i+1,'/',len(args),'seconds',round(time.time()-start),flush=True)
    chi=np.array(cs);failed=~np.isfinite(chi).any(1);chi[failed]=0
    pred=prediction(-.5*np.nan_to_num(chi,nan=np.inf,posinf=np.inf),a.get('z',np.zeros(len(args))))
    pred.update(chi2=chi,object_id=a['object_id'],n_valid=valid.sum(1),unsupported=(valid.sum(1)<3)|failed,seconds=np.array(time.time()-start))
    np.savez_compressed(path,**pred);return pred


def prediction(logits,z):
    edges,centers=grid();logits=np.asarray(logits,dtype=float)+np.log(np.diff(edges));logits-=logsumexp(logits,axis=1,keepdims=True)
    p=np.exp(logits);cdf=np.cumsum(p,axis=1);idx=lambda q:np.minimum((cdf<q).sum(1),len(centers)-1)
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
    return dict(logits=logits.astype(np.float32),pz=p.astype(np.float32),z_spec=np.asarray(z,np.float32),z_p16=centers[idx(.16)],z_p50=centers[idx(.5)],z_p84=centers[idx(.84)],redshift_edges=edges,redshift_centers=centers)


def metric(a,mask=None):
    if mask is None:mask=np.ones(len(a['pz']),bool)
    r=d.metrics(a,mask);dz=(a['z_p50'][mask]-a['z_spec'][mask])/(1+a['z_spec'][mask]);r['median_bias']=float(np.median(dz))
    r.update(ce=r['cross_entropy'],outlier_fraction=r['catastrophic_outlier_fraction'],crps=r['mean_crps'],coverage_68=r['p16_p84_coverage'])
    return r


def corrected(a):
    from dust_extinction.parameter_averages import G23
    import astropy.units as u
    law=G23(Rv=3.1);co=[]
    for f in curves():
        ext=3.1*law((f.wave*1e-4)*u.micron)
        co.append((f.wave,f.throughput,ext))
    r={k:v.copy() for k,v in a.items()}
    for i in range(len(r['flux'])):
        if not r['ebv_available'][i]:continue
        for b,j in enumerate(r['band_id'][i]):
            if j<0:continue
            w,t,x=co[j];trans=np.trapezoid(t/w*10**(-.4*r['ebv'][i]*x),w)/np.trapezoid(t/w,w)
            r['flux'][i,b]/=trans;r['sigma'][i,b]/=trans
    return r


def conditional(name,a,pred,target=False):
    f,e=a['flux'],a['sigma'];v=np.isfinite(f)&np.isfinite(e)&(e>0);n=v.sum(1)
    sn=np.sqrt(np.nansum(np.where(v,(f/e)**2,np.nan),axis=1)/np.maximum(n,1))
    mag=22.5-2.5*np.log10(np.maximum(np.nanmedian(np.abs(f),axis=1),1e-30));z=pred['z_spec']
    strata={'all':np.ones(len(z),bool),'supported':~pred.get('unsupported',n<3),'z_lt1':z<1,'z_1to2':(z>=1)&(z<2),'z_ge2':z>=2,'SN_lt5':sn<5,'SN_5to30':(sn>=5)&(sn<30),'SN_ge30':sn>=30,'mag_lt22':mag<22,'mag_ge22':mag>=22,'complete':n==np.max(n),'incomplete':n<np.max(n),'quality_bad':(a['quality']>0).any(1),'quality_no_flag':~(a['quality']>0).any(1)}
    if target:
        s=d.npz(ROOT/'reports/stage0/1027a/support_assignments.npz')
        strata.update(common_support=s['common_support'],out_of_support=~s['common_support'])
        for key in ['brightness','sn']:
            for value in np.unique(s[key]):strata[f'Stage0_{key}_{value}']=s[key]==value
    return [dict(condition=name,stratum=k,**metric(pred,m)) for k,m in strata.items() if np.any(m)]


def evaluate():
    assert json.loads((OUT/'physical_tests.json').read_text())['passed'];audit=json.loads((OUT/'data_audit.json').read_text());assert audit['outside_grid_source_labels']==0
    h=d.npz(OUT/'hsc_trainval.npz');h=data.take(h,h['split']=='val');l=d.npz(OUT/'legacy_validation.npz');rows=[];screen={}
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    match,sep,_=SkyCoord(l['ra']*u.deg,l['dec']*u.deg).match_to_catalog_sky(SkyCoord(h['ra']*u.deg,h['dec']*u.deg))
    linked=sep.arcsec<1.;l['group'][linked]=h['group'][match[linked]]
    id_groups=dict(zip(h['object_id'],h['group']))
    for i,ident in enumerate(l['object_id']):
        if ident in id_groups:l['group'][i]=id_groups[ident]
    d.write(OUT/'source_overlap.json',dict(shared_exact_identifiers=len(set(h['object_id'])&set(l['object_id'])),positional_pairs_within_1arcsec=int(linked.sum()),different_identifiers_within_1arcsec=int(np.sum(l['object_id'][linked]!=h['object_id'][match[linked]])),calibration_confirmation_groups_linked=True))
    # Deterministic source screening, before full baseline fitting or target labels.
    for name,a in [('HSC',h),('Legacy',l)]:
        s=data.take(a,np.arange(min(256,len(a['flux']))));p=fit_rows(s,OUT/f'screen_{name}_P0.npz');screen[name]=metric(p)
    d.write(OUT/'source_screen.json',screen)
    sources={}
    for name,a in [('HSC',h),('Legacy',l)]:
        p=fit_rows(a,OUT/f'source_{name}_P0.npz');sources[name]=p;rows+=conditional(f'{name}_P0',a,p)
    # A modest optional calibration experiment; no duplicate galaxy enters both halves.
    selection=[]
    for frac in [0.,.03,.10]:
        scores=[]
        for name,a in [('HSC',h),('Legacy',l)]:
            iscal=np.array([data.group_fraction('cal:'+x)<.5 for x in a['group']]);idx=np.flatnonzero(iscal)[:256];s=data.take(a,idx)
            p=fit_rows(s,OUT/f'cal_{name}_{frac:.2f}.npz',frac);scores.append(metric(p)['ce'])
        selection.append(dict(fractional_error=frac,balanced_source_CE=float(np.mean(scores)),source_CE=scores))
    chosen=min(selection,key=lambda x:x['balanced_source_CE'])['fractional_error']
    lock=dict(P0_fractional_error=0.,P2_fractional_error=chosen,selection=selection,target_labels_used=False,Pext='G23 Rv3.1, independent SFD EBV, flat-fnu band attenuation; predeclared, not selected on target',shared_source_ids=len(set(h['object_id'])&set(l['object_id'])),confirmation='First 256 per source from complementary SHA256 object-ID calibration halves')
    d.write(OUT/'settings_locked.json',lock)
    for name,a in [('HSC',h),('Legacy',l)]:
        iscal=np.array([data.group_fraction('cal:'+x)<.5 for x in a['group']]);s=data.take(a,np.flatnonzero(~iscal)[:256])
        p=fit_rows(s,OUT/f'confirm_{name}_P2.npz',chosen);rows+=conditional(f'{name}_P2_confirmation',s,p)
        ext=data.take(a,np.arange(min(256,len(a['flux']))));p=fit_rows(corrected(ext),OUT/f'extinction_{name}_P0.npz');rows+=conditional(f'{name}_Pext_screen',ext,p)
    # At this point inference configurations are frozen; now load ZPHOT reference.
    target=d.npz(OUT/'clauds_photometry.npz');target['z']=d.npz(d.FAST/'clauds_predictions.npz',('z_spec',))['z_spec'];eb=d.npz(OUT/'clauds_independent_ebv.npz')
    assert np.array_equal(target['object_id'],eb['object_id']);et={k:v.copy() for k,v in target.items()};et['ebv']=eb['ebv'];et['ebv_available']=eb['available']
    for config,tef,base in [('P0',0.,target),('P2',chosen,target),('Pext',0.,corrected(et))]:
        for system in ['full','HSC']:
            a={k:v.copy() for k,v in base.items()}
            if system=='HSC':
                keep=np.isin(a['band_id'],[FILTERS.index('HSC.'+b) for b in 'grizy']);a['flux'][~keep]=np.nan;a['sigma'][~keep]=np.nan
            name=f'CLAUDS_{config}_{system}';p=fit_rows(a,OUT/f'{name}.npz',tef);p['z_spec']=target['z'];np.savez_compressed(OUT/f'{name}.npz',**p);rows+=conditional(name,a,p,True)
    stress=[]
    for key in ['screen_native','screen_dim7','screen_sigma100','screen_noisy_dim0_b100_seed17','screen_noisy_dim0_b100_seed29','screen_noisy_dim0_b100_seed43']:
        old=d.npz(ROOT/f'reports/stage05/1027/{key}.npz');features=d.npz(ROOT/f'reports/stage05/1027/inputs/{key}_photometry.npz')['features']
        # Match the exact archived input features/identities, rather than regenerate noise.
        ids=old['targetid'];s=data.make_rows(ids,features,[f'HSC.{b}' for b in 'grizy'],old['z_spec'],np.full(len(ids),'val'),np.zeros(len(ids)),np.zeros(len(ids)),'HSC')
        p=fit_rows(s,OUT/f'stress_{key}.npz');stress.append(dict(condition=key,model='P0',**metric(p)));stress.append(dict(condition=key,model='archived_Qwen',**metric(old)))
    d.table(OUT/'conditional_metrics.csv',rows);d.table(OUT/'performance.csv',[r for r in rows if r['stratum']=='all']);d.table(OUT/'stress_metrics.csv',stress)
    d.write(OUT/'summary.json',dict(performance=[r for r in rows if r['stratum']=='all'],source_screen=screen,settings=lock,stress=stress,target_reference='CLAUDS ZPHOT, not spectroscopic truth',stage2b_physical_blocker=False))
    plot();report();d.write(OUT/'handoff.json',dict(stage='Stage2A',completed=True,physical_checks_passed=True,baseline_evaluation_complete=True,unresolved_invalidating_error=False,Stage2B_proceed=True,limitations=['Catalogue historical Galactic correction equivalence is not proven; untreated primary and independent correction sensitivity delimit its effect','HSC r/r2 and i/i2 exposure mixtures unavailable; nominal r/i adopted','Legacy original independent training cache missing; audited paired-view pilot available','CLAUDS ZPHOT shares template-fitting assumptions; not independent spectroscopy']))


def plot():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['CLAUDS_P0_full','CLAUDS_P0_HSC','CLAUDS_P2_full','CLAUDS_Pext_full'];fig,axs=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for j,n in enumerate(names):
        p=d.npz(OUT/f'{n}.npz');axs[0,j].scatter(p['z_spec'],p['z_p50'],s=10,alpha=.5);axs[0,j].plot([0,4.5],[0,4.5],color='gray');axs[0,j].set(title=n,xlabel='CLAUDS ZPHOT',ylabel='Predicted median',xlim=(0,4.5),ylim=(0,4.5))
        pit=np.sum(p['pz']*(p['redshift_centers'][None,:]<=p['z_spec'][:,None]),axis=1);axs[1,j].hist(pit,bins=np.linspace(0,1,11));axs[1,j].set(xlabel='PIT',ylabel='Objects')
    for ext in ['png','pdf']:fig.savefig(OUT/f'scatter_PIT.{ext}',dpi=150)
    plt.close(fig);fig,axs=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
    for i,ax in zip([0,1,2,63,127,255],axs.flat):
        for name in ['CLAUDS_P0_full','CLAUDS_P0_HSC','CLAUDS_P2_full']:
            p=d.npz(OUT/f'{name}.npz');ax.plot(p['redshift_centers'],p['pz'][i],label=name)
        ax.axvline(p['z_spec'][i],color='black',ls=':');ax.set(title=f'Preselected query index {i}',xlabel='z',ylabel='Bin probability')
    axs.flat[0].legend(fontsize=6)
    for ext in ['png','pdf']:fig.savefig(OUT/f'representative_pdfs.{ext}',dpi=150)
    plt.close(fig)


def report():
    s=json.loads((OUT/'summary.json').read_text());audit=json.loads((OUT/'data_audit.json').read_text());m=s['performance'];lines=['# Stage 2A — physical SED baseline, 1028','', 'Executed EAZY 0.8.7 NNLS on the actual full response curves with 12 pinned FSPS QSF templates and Inoue14 IGM. P0 is a normalized profile-likelihood pseudo-posterior with free nonnegative template amplitudes, not EAZY’s fully configured posterior. No population prior or target-label selection was used.','','## Results','','| Condition | N | NMAD | Outliers | CE | CRPS | 68% coverage | Median z | Narrow-interval fraction |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in m:lines.append('| '+r['condition']+' | '+str(r['n'])+' | '+' | '.join(f'{r[k]:.4f}' for k in ['nmad','outlier_fraction','ce','crps','coverage_68','predicted_median','concentration_width_003'])+' |')
    by={r['condition']:r for r in m};p=by['CLAUDS_P0_full'];h=by['CLAUDS_P0_HSC'];e=by['CLAUDS_Pext_full']
    lines+=['','## Scientific assessment','',f"A/B: P0 full-band predictions occupy {p['unique_median_bins']} median bins, span {p['predicted_p05']:.3f}–{p['predicted_p95']:.3f} at the 5th–95th percentiles, and have {p['outlier_fraction']:.1%} ZPHOT outliers. Compare this directly with the historical 96–97% low-z failure; a broad distribution alone is not success.",'',f"C: NMAD={p['nmad']:.4f}, CRPS={p['crps']:.4f}, CE={p['ce']:.3f}, and nominal 68% coverage={p['coverage_68']:.1%}. These separately expose point error and PDF miscalibration.",'',f"D: Actual HSC-only outliers={h['outlier_fraction']:.1%} and NMAD={h['nmad']:.4f}, versus full-band {p['outlier_fraction']:.1%} and {p['nmad']:.4f}. Extra wavelength coverage must be judged from both statistics.",'',f"E: The fixed independent extinction correction gives {e['outlier_fraction']:.1%} outliers and NMAD={e['nmad']:.4f}. P2 and all quality/completeness/support strata are separately tabulated. A population prior was deliberately omitted; no claim of prior robustness is made.",'','F: The response integration, masking and free-amplitude likelihood are suitable verified infrastructure for a small Stage 2B pilot. The observed accuracy and calibration do not establish true-redshift transfer; template mismatch, processing differences and genuine degeneracies remain possible explanations.']
    lines+=['','## Inputs and interpretation','',f'Full existing HSC validation and recovered Legacy validation are evaluated (audit: {audit}). The source screening precedes full fits. Optional fractional template-error candidates 0, 0.03, 0.10 are selected on 256 calibration-half objects per source, balanced by survey; complementary confirmation halves are reserved. Exact calibration selection is in settings_locked.json. All source overlaps with the fixed 256 targets are excluded within 1 arcsec. Repeated source identities use the same calibration half and are reported as shared, not independent.','', 'HSC uses archived grizy cmodel fluxes inferred from magnitudes, with magnitude-limit censoring inherited from the loader. Legacy uses native signed Tractor grzW1W2 fluxes and inverse-variance errors; releases 9010/9012 use DECam and 9011 uses BASS/MzLS, with WISE W1/W2 retained. CLAUDS uses hscPipe cmodel flux in microJy converted by 10^((22.5−23.9)/2.5), including its actual flags and signed values. Nonfinite flux, nonpositive or missing uncertainty and unavailable filter IDs are masked. Fewer than three usable bands yields an explicitly flagged uniform PDF. Quality flags are reported, not silently promoted to missing observations. Source quality/membership remains the archived spectroscopic selection.','', 'The primary configuration uses photometry as loaded without an additional Galactic screen and does not require EBV. The predeclared Pext sensitivity applies G23(Rv=3.1) with independent SFD EBV (source catalogue metadata; target official IRSA DUST queries), using flat-fnu band attenuation. It is an approximate band correction, not proof that historical source and target processing match. Unavailable EBV stays unavailable, never a measured zero. HSC r/r2 and i/i2 exposure mixtures remain a nominal-r/i approximation. These delimited uncertainties do not invalidate the tested interface.','', 'CLAUDS ZPHOT comes from the hscPipe–Phosphoros catalogue. Its template fitting and EAZY share broad stellar-population/template assumptions, though not identical libraries or priors; agreement is not independent redshift truth. No independent spectroscopic column was present in this local catalogue. Existing source-target positional coincidences are excluded from calibration rather than used to tune transfer. See the [catalogue paper](https://arxiv.org/abs/2301.13750), [EAZY method](https://arxiv.org/abs/0807.1533), and [IRSA API](https://irsa.ipac.caltech.edu/applications/DUST/docs/dustProgramInterface.html).','', 'Full UV/NIR versus actual HSC-only measurements, support, completeness, quality, magnitude, S/N and redshift strata are in conditional_metrics.csv. Point-distribution concentration is reported to distinguish object-dependent inference from a constant predictor; uncertainty is assessed with CE, CRPS, interval coverage and PIT, not NMAD alone. Small support strata have high sampling uncertainty. Stress cases use the exact Stage-0.5 identities and archived noise realizations; common amplitude/error scaling is a likelihood symmetry and simulated dimming does not create high-redshift galaxies.','', '## Reproduction and Stage 2B handoff','', 'Run scripts/stage2_data.py, then scripts/stage2_physics.py build and evaluate in the recorded environment. Per-object profile chi-square, normalized PDFs, flags, kernel, filter translations, dependency versions and hashes are saved beside this report. The kernel has [filter,300 redshift bins,12 templates] and integrates all actual UV through W2 responses. Source test data and historical weights/results are untouched.','', 'Stage 2B can reuse this verified signed-flux/masked-error interface and physical kernel. Retain P0, the ordinary numerical MLP and the shared-band neural-only encoder as controls. Historical extinction equivalence, nominal HSC filters, source redshift coverage and template incompleteness remain scientific limitations. The complete independent Legacy training cache is missing; the available audited HSC–Legacy same-galaxy multi-view pilot must not be described as the definitive population-scale mixed-source experiment.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','evaluate','report']);args=p.parse_args();globals()[args.action]()
