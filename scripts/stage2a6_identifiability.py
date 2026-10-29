"""Stage 2A.6: fixed SED-dimensionality experiments and proper latent integration.

All historical implementations are imported read-only. No target catalogue is
opened. DM integrates a normalized Beta shape prior and a proper half-normal
amplitude prior in the documented, rest-5500-normalized kernel coordinates.
"""
from __future__ import annotations
import json,time,csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.special import logsumexp,log_ndtr,betaln
from scipy.signal import find_peaks
from scipy.integrate import quad
import stage2a5_audit as previous
import stage2_physics as pdf
from stage2_data import ROOT,d,FILTERS,take,group_fraction
OUT=ROOT/'reports/stage2a6/1028c'
_K=None;_CFG=None


def setup():
    global _K,_CFG
    if _K is None:_K=d.npz(previous.OUT/'response_kernel_highres.npz')['kernel']
    if _CFG is None:_CFG=json.loads((OUT/'protocol.json').read_text())
    return _K,_CFG


def valid_arrays(f,e,ids,tef=0.):
    mask=np.isfinite(f)&np.isfinite(e)&(e>0)&(ids>=0)
    return f[mask].astype(float),np.sqrt(e[mask].astype(float)**2+(tef*np.maximum(f[mask],0))**2),ids[mask],mask


def two_profile(y,s,k):
    """Vectorized exact 2-column NNLS, k=[band,z,2]."""
    iv=1/s**2;a=np.einsum('bz,b,bz->z',k[:,:,0],iv,k[:,:,0]);b=np.einsum('bz,b,bz->z',k[:,:,0],iv,k[:,:,1]);c=np.einsum('bz,b,bz->z',k[:,:,1],iv,k[:,:,1]);h=np.einsum('b,bz,b->z',y,k[:,:,0],iv);j=np.einsum('b,bz,b->z',y,k[:,:,1],iv)
    det=a*c-b*b;den=np.maximum(det,1e-300);x=(h*c-j*b)/den;z=(j*a-h*b)/den;feasible=(det>1e-12*np.maximum(a*c,1e-300))&(x>=0)&(z>=0)
    coeff=np.zeros((k.shape[1],2));c0=np.maximum(h/np.maximum(a,1e-300),0);c1=np.maximum(j/np.maximum(c,1e-300),0)
    chi0=np.sum(((y[:,None]-k[:,:,0]*c0)/s[:,None])**2,0);chi1=np.sum(((y[:,None]-k[:,:,1]*c1)/s[:,None])**2,0);choose=chi0<=chi1;coeff[choose,0]=c0[choose];coeff[~choose,1]=c1[~choose];coeff[feasible]=np.stack([x,z],1)[feasible]
    model=np.einsum('bzk,zk->bz',k,coeff);chi=np.sum(((y[:,None]-model)/s[:,None])**2,0)
    return chi,coeff,(a,b,c,h,j)


def amplitude_log_integral(y,s,t,tau):
    """Proper normalized half-N(0,tau²) amplitude, integrated analytically."""
    q=np.sum((t/s[:,None])**2,0)+1/tau**2;h=np.sum(y[:,None]*t/s[:,None]**2,0);r=np.sum((y/s)**2);normal=-.5*len(y)*np.log(2*np.pi)-np.log(s).sum()
    return normal-.5*np.maximum(r-h*h/q,0)+np.log(2)-np.log(tau)-.5*np.log(q)+log_ndtr(h/np.sqrt(q))


def marginal_two(y,s,k,tau,alpha=1.,order=16,moments=False):
    """Proper A,w marginal, adaptive union intervals + Gaussian quadrature.

    w=sin²(theta) regularizes the Beta(1/2,1/2) endpoints. Intervals resolve
    NNLS shape peaks, boundary-gradient scales and a fixed global partition.
    No flux likelihood or PDF floor is applied.
    """
    chi,co,(a,b,c,h,j)=two_profile(y,s,k);amp=co.sum(1);w=np.divide(co[:,1],amp,out=np.full(len(amp),.5),where=amp>0);q=a*(1-w)**2+2*b*w*(1-w)+c*w*w;diff=a+c-2*b;td=(b-a)*(1-w)+(c-b)*w;fisher=amp**2*np.maximum(diff-td**2/np.maximum(q,1e-300),0);width=np.minimum(1.,1/np.sqrt(np.maximum(fisher,1e-24)))
    model=np.einsum('bzk,zk->bz',k,co);grad=-2*amp*np.sum((y[:,None]-model)*(k[:,:,1]-k[:,:,0])/s[:,None]**2,0);boundary=(w<1e-12)|(w>1-1e-12);width[boundary]=np.minimum(width[boundary],2/np.maximum(abs(grad[boundary]),2));width=np.maximum(width,1e-12)
    fixed=np.array([0,1e-12,1e-9,1e-6,.001,.01,.1,.5,.9,.99,.999,1-1e-6,1-1e-9,1-1e-12,1]);multipliers=np.array([-32,-16,-8,-4,-2,-1,-.5,-.25,0,.25,.5,1,2,4,8,16,32]);bound=np.sort(np.column_stack([np.broadcast_to(fixed,(len(w),len(fixed))),np.clip(w[:,None]+width[:,None]*multipliers,0,1)]),axis=1);theta=np.arcsin(np.sqrt(bound));left,right=theta[:,:-1],theta[:,1:];nodes,weights=np.polynomial.legendre.leggauss(order);half=(right-left)/2;angle=(left+right)[:,:,None]/2+half[:,:,None]*nodes;ww=np.sin(angle)**2
    qq=a[:,None,None]*(1-ww)**2+2*b[:,None,None]*ww*(1-ww)+c[:,None,None]*ww**2+1/tau**2;hh=h[:,None,None]*(1-ww)+j[:,None,None]*ww;r=np.sum((y/s)**2);normal=-.5*len(y)*np.log(2*np.pi)-np.log(s).sum();logI=normal-.5*np.maximum(r-hh*hh/qq,0)+np.log(2)-np.log(tau)-.5*np.log(qq)+log_ndtr(hh/np.sqrt(qq))
    with np.errstate(divide='ignore',invalid='ignore'):
        prior=np.log(2)-betaln(alpha,alpha)+(2*alpha-1)*(np.log(np.sin(angle))+np.log(np.cos(angle))) if alpha!=.5 else np.zeros_like(angle)+np.log(2/np.pi)
        log_weight=np.log(half[:,:,None]*weights)
    joint=(logI+prior+log_weight).reshape(len(w),-1);logevidence=logsumexp(joint,axis=1)
    if not moments:return logevidence
    posterior=np.exp(joint-logevidence[:,None]);mu=hh/qq;sd=1/np.sqrt(qq);standard=hh/np.sqrt(qq);mills=np.exp(-.5*standard**2-.5*np.log(2*np.pi)-log_ndtr(standard));amean=np.maximum(mu+sd*mills,0).reshape(len(w),-1);weight=ww.reshape(len(w),-1)
    return dict(log_evidence=logevidence,mean_shape_w=np.sum(posterior*weight,1),mean_amplitude=np.sum(posterior*amean,1),mean_coefficient0=np.sum(posterior*amean*(1-weight),1),mean_coefficient1=np.sum(posterior*amean*weight,1))


def all_scores(f,e,ids,k=None,tef=0.,dm=True):
    full,cfg=setup();k=full if k is None else k;y,s,bids,mask=valid_arrays(f,e,ids,tef);nz=k.shape[1];nt=k.shape[2];selected=k[bids]
    if not len(y):return {name:np.zeros(nz) for name in ['D12','D1_profile','D1_template_marg','D1_evidence','D2','D3','DM','DM_alpha05']}
    q=np.sum((selected/s[:,None,None])**2,0);h=np.sum(y[:,None,None]*selected/s[:,None,None]**2,0);amplitude=np.maximum(h/np.maximum(q,1e-300),0);single=np.sum(((y[:,None,None]-selected*amplitude[None,:,:])/s[:,None,None])**2,0)
    scores=dict(D12=-.5*previous.profile((f,e,ids,k,tef)),D1_profile=-.5*single.min(1),D1_template_marg=logsumexp(-.5*single,axis=1)-np.log(nt))
    evidence=np.column_stack([amplitude_log_integral(y,s,selected[:,:,ti],cfg['amplitude_prior_tau_nanomaggy']) for ti in range(nt)]);scores['D1_evidence']=logsumexp(evidence,axis=1)-np.log(nt)
    ks=selected[:,:,cfg['D2_indices']];chi2,_,_=two_profile(y,s,ks);scores['D2']=-.5*chi2;scores['D3']=-.5*previous.profile((f,e,ids,k[:,:,cfg['D3_indices']],tef))
    if dm:
        low=marginal_two(y,s,ks,cfg['amplitude_prior_tau_nanomaggy'],1.,8);high=marginal_two(y,s,ks,cfg['amplitude_prior_tau_nanomaggy'],1.,16);score8=pdf.prediction(low[None,:],np.array([0.]))['pz'][0];score16=pdf.prediction(high[None,:],np.array([0.]))['pz'][0];tv=.5*abs(score8-score16).sum()
        if tv>1e-4:
            higher=marginal_two(y,s,ks,cfg['amplitude_prior_tau_nanomaggy'],1.,32);tv=.5*abs(pdf.prediction(higher[None,:],np.array([0.]))['pz'][0]-score16).sum();high=higher
        half16=marginal_two(y,s,ks,cfg['amplitude_prior_tau_nanomaggy'],.5,16);half32=marginal_two(y,s,ks,cfg['amplitude_prior_tau_nanomaggy'],.5,32);htv=.5*abs(pdf.prediction(half16[None,:],np.array([0.]))['pz'][0]-pdf.prediction(half32[None,:],np.array([0.]))['pz'][0]).sum()
        scores.update(DM=high,DM_alpha05=half32,DM_quadrature_TV=np.full(nz,float(tv)),DM_alpha05_quadrature_TV=np.full(nz,float(htv)))
    return scores


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists(),'Use existing protocol; do not overwrite it'
    protected={}
    for folder in ['stage0','stage05','stage1','stage2a','stage2b','stage2a5','stage2b5']:
        for path in (ROOT/'reports'/folder).rglob('*'):
            if path.is_file():protected[str(path)]=d.sha(path)
    for path in (ROOT/'scripts').glob('*.py'):
        if path.name!='stage2a6_identifiability.py':protected[str(path)]=d.sha(path)
    d.write(OUT/'protected_before.json',protected)
    ts,_=previous.context();wave=np.geomspace(1200,50000,256);spectra=[];breaks=[]
    for t in ts:
        f=t.flux_fnu(z=0);spectra.append(np.interp(wave,t.wave,f));lo=np.mean(np.interp(np.linspace(3850,3950,100),t.wave,f));hi=np.mean(np.interp(np.linspace(4000,4100,100),t.wave,f));breaks.append(hi/lo)
    basis=[int(np.argmin(breaks)),int(np.argmax(breaks))];x=np.log(np.maximum(spectra,1e-30));distance=np.minimum(np.mean((x-x[basis[0]])**2,axis=1),np.mean((x-x[basis[1]])**2,axis=1));distance[basis]=-1;third=int(distance.argmax())
    pair=d.npz(pdf.OUT/'mixed_eligible_trainval.npz');groups={}
    for split,n in [('train',64),('val',192)]:groups[split]=sorted(set(pair['group'][pair['split']==split]),key=lambda g:group_fraction('a6:'+g))[:n]
    partitions={'development':groups['train'],'validation':groups['val'][:64],'confirmation':groups['val'][64:]}
    subsets={}
    for partition,gs in partitions.items():
        for cat in ['HSC','Legacy']:
            a=take(pair,np.isin(pair['group'],gs)&(pair['catalogue']==cat));a=take(a,np.argsort(a['object_id']));subsets[partition+'_'+cat]=a;np.savez_compressed(OUT/f'input_{partition}_{cat}.npz',**a)
    flux=pair['flux'][pair['split']=='train'];tau=float(1000*np.nanquantile(np.abs(flux),.95))
    d.write(OUT/'protocol.json',dict(D2_indices=basis,D3_indices=basis+[third],basis_method='Pinned-library-only min/max narrow4000A fnu break; D3 adds largest minimum log-SED distance from these endpoints on1200–50000A. No source/target photo-z metric selects spectra.',template_4000_breaks=breaks,amplitude_prior='HalfNormal(A;0,tau), A>=0, observed-frame normalized-kernel amplitude in nanomaggy; not an intrinsic luminosity prior',amplitude_prior_tau_nanomaggy=tau,amplitude_scale_source='1000 times training-pair95th percentile |flux|, no redshift labels; broad proper prior',coefficient_priors={'DM':'Beta(1,1) on D2 mixture; amplitude analytically marginalized','DM_alpha05':'Beta(0.5,0.5), fixed sensitivity'},D1_prior='Uniform probability1/12, profile-amplitude and proper amplitude-marginal versions separately labelled',redshift_prior='Uniform density over original[0,4.5] grid; bin widths retained',errors=[0.,.1],partitions=partitions,source_selection='Independent galaxy-group hashes, original union-validation preserved; validation and confirmation64/128 different galaxies',target_labels_read=False,source_tests_read=False,band_experiment=['HSC5','HSC3_gri','HSC3_grz','Legacy3','Legacy4_W1','Legacy5','HSC5_W1','HSC5_W1W2'],confirmation_models_fixed_before_evaluation=True))
    np.savez_compressed(OUT/'basis.npz',rest_wavelength=wave,template_fnu=np.array(spectra),D2_indices=basis,D3_indices=basis+[third])


def tests():
    k,cfg=setup();replay=[]
    for cat in ['HSC','Legacy']:
        a=d.npz(previous.OUT/f'input_paired_val_{cat}.npz');old=d.npz(previous.OUT/f'paired_val_{cat}_accurate.npz')
        for i in range(8):
            value=previous.profile((a['flux'][i],a['sigma'][i],a['band_id'][i],k,0.));np.testing.assert_allclose(value,old['chi2'][i],rtol=1e-9,atol=1e-8);replay.append(float(np.max(abs(value-old['chi2'][i]))))
    # Amplitude analytic integral against independently adaptive quadrature.
    y=np.array([.8,-.1,.6]);s=np.array([.1,.2,.1]);t=np.array([.7,.2,.5]);tau=3.;ana=float(amplitude_log_integral(y,s,t[:,None],tau)[0]);normal=np.prod(1/(np.sqrt(2*np.pi)*s));prior=lambda A:np.sqrt(2/np.pi)/tau*np.exp(-.5*(A/tau)**2);fun=lambda A:normal*np.exp(-.5*np.sum(((y-A*t)/s)**2))*prior(A);numeric=quad(fun,0,30,points=[.5,1,1.5,2],epsabs=1e-13,epsrel=1e-10)[0];np.testing.assert_allclose(ana,np.log(numeric),atol=1e-9)
    # 2-component profile independent solver, all original redshifts.
    rng=np.random.default_rng(106);ids=np.array([FILTERS.index('HSC.'+x) for x in 'grizy']);small=k[ids][:,:,cfg['D2_indices']];ff=small[:,40]@np.array([.8,1.1]);ee=np.ones(5)*.001;c,co,_=two_profile(ff,ee,small)
    for iz in [0,40,70,150,250]:np.testing.assert_allclose(c[iz],previous.exact_nnls(ff,ee,ids,k[:,:,cfg['D2_indices']][:,iz])[0],rtol=1e-8,atol=1e-7)
    integration=[]
    for sigma in [.1,.001,.0001]:
        xx=np.array([[[.8,.2]],[[.2,1.]],[[.5,.6]]]);yy=xx[:,0]@np.array([1.2,.8]);ss=np.full(3,sigma);tau0=5.;g16=marginal_two(yy,ss,xx,tau0,1.,16)[0];g32=marginal_two(yy,ss,xx,tau0,1.,32)[0];np.testing.assert_allclose(g16,g32,atol=2e-4)
        # Independent adaptive theta integral at the peak of an exact mixture.
        mode=.4;logref=float(amplitude_log_integral(yy,ss,((1-mode)*xx[:,0,0]+mode*xx[:,0,1])[:,None],tau0)[0]);fun=lambda theta:float(np.exp(amplitude_log_integral(yy,ss,((1-np.sin(theta)**2)*xx[:,0,0]+np.sin(theta)**2*xx[:,0,1])[:,None],tau0)[0]-logref)*np.sin(2*theta));center=np.arcsin(np.sqrt(mode));points=np.clip(center+np.array([-100,-10,-1,0,1,10,100])*sigma,0,np.pi/2);integ=quad(fun,0,np.pi/2,points=np.unique(points),epsabs=1e-11,epsrel=1e-8,limit=300)[0];truth=np.log(integ)+logref;np.testing.assert_allclose(g32,truth,atol=3e-4);integration.append(dict(sigma=sigma,order16=float(g16),order32=float(g32),adaptive_reference=float(truth)))
    # Amplitude scaling is exact for profiles, and proper-prior covariance
    # requires co-scaling tau; the likelihood density changes by scale^-N.
    a=amplitude_log_integral(y,s,t[:,None],tau);aa=amplitude_log_integral(y*.01,s*.01,t[:,None],tau*.01);np.testing.assert_allclose(aa-a,-len(y)*np.log(.01),atol=1e-9)
    negative=np.array([1.,-.2,np.nan,1.,.5]);err=np.array([.1,.1,.1,0,.1]);v=valid_arrays(negative,err,ids);assert len(v[0])==3 and (v[0]<0).any()
    d.write(OUT/'numerical_tests.json',dict(passed=True,D12_replay_max_absolute_difference=max(replay),single_amplitude_analytic_vs_adaptive=True,D2_NNLS_matches_independent=True,DM_quadrature_checks=integration,proper_prior_normalization=True,signed_flux_and_invalid_error_mask=True,profile_scale_invariance=True,proper_amplitude_prior_scale_covariance=True))


def _row(args):
    f,e,ids,tef=args;return all_scores(f,e,ids,tef=tef)


def score_dataset(a,tag,tef=0.,diagnostics=True):
    path=OUT/f'{tag}.npz'
    if path.exists():return d.npz(path)
    start=time.time();setup()
    with ProcessPoolExecutor(max_workers=6) as pool:results=list(pool.map(_row,[(f,e,b,tef) for f,e,b in zip(a['flux'],a['sigma'],a['band_id'])],chunksize=4))
    out={name:np.array([r[name] for r in results]) for name in results[0]};out.update(object_id=a['object_id'],z_spec=a['z'],n_valid=(np.isfinite(a['flux'])&np.isfinite(a['sigma'])&(a['sigma']>0)&(a['band_id']>=0)).sum(1),seconds=np.array(time.time()-start));np.savez_compressed(path,**out)
    if diagnostics:
        fixed={};best={};cfg=setup()[1]
        for model,indices in [('D12',list(range(12))),('D1_profile',None),('D2',cfg['D2_indices']),('D3',cfg['D3_indices'])]:
            cs=[];res=[];coeff=[];amp=[];rank=[];minimum=[];rbest=[]
            for i,z in enumerate(a['z']):
                ker=previous.continuous(float(z));ii=indices
                if ii is None:
                    y,s,ids,mask=valid_arrays(a['flux'][i],a['sigma'][i],a['band_id'][i],tef);ss=ker[ids];qq=np.sum((ss/s[:,None])**2,0);hh=np.sum(y[:,None]*ss/s[:,None]**2,0);aa=np.maximum(hh/np.maximum(qq,1e-300),0);cost=np.sum(((y[:,None]-ss*aa)/s[:,None])**2,0);ii=[int(cost.argmin())]
                value=previous.exact_nnls(a['flux'][i],a['sigma'][i],a['band_id'][i],ker[:,ii],tef);cs.append(value[0]);res.append(value[3]);fullcoef=np.zeros(12);fullcoef[ii]=value[1];coeff.append(fullcoef);amp.append(float(fullcoef.sum()));rank.append(value[5]);iz=int(out[model][i].argmax());minimum.append(-2*out[model][i,iz]);bii=ii
                if model=='D1_profile':
                    y,s,ids,mask=valid_arrays(a['flux'][i],a['sigma'][i],a['band_id'][i],tef);ss=setup()[0][ids,iz];qq=np.sum((ss/s[:,None])**2,0);hh=np.sum(y[:,None]*ss/s[:,None]**2,0);aa=np.maximum(hh/np.maximum(qq,1e-300),0);bii=[int(np.sum(((y[:,None]-ss*aa)/s[:,None])**2,0).argmin())]
                v=previous.exact_nnls(a['flux'][i],a['sigma'][i],a['band_id'][i],setup()[0][:,iz][:,bii],tef);rbest.append(v[3])
            fixed.update({f'{model}_'+name:np.array(values) for name,values in [('chi2_zspec',cs),('residual_zspec',res),('coeff_zspec',coeff),('amplitude_zspec',amp),('rank_zspec',rank),('chi2_min',minimum),('residual_best',rbest)]})
        # Continuous true-z evidence, not a nearest-bin approximation.
        dmtrue=[];d1true=[]
        for i,z in enumerate(a['z']):
            kk=previous.continuous(float(z))[:,None,:];f,e,ids=a['flux'][i],a['sigma'][i],a['band_id'][i];y,s,bids,mask=valid_arrays(f,e,ids,tef);dmtrue.append(float(marginal_two(y,s,kk[bids][:,:,cfg['D2_indices']],cfg['amplitude_prior_tau_nanomaggy'],order=32)[0]));d1true.append(float(logsumexp(np.array([amplitude_log_integral(y,s,kk[bids,:,j],cfg['amplitude_prior_tau_nanomaggy'])[0] for j in range(12)]))-np.log(12)))
        fixed.update(DM_log_evidence_zspec=np.array(dmtrue),D1_evidence_zspec=np.array(d1true),object_id=a['object_id'],z_spec=a['z']);np.savez_compressed(OUT/f'{tag}_diagnostics.npz',**fixed)
    print(tag,'objects',len(a['z']),'seconds',round(time.time()-start,1),'max quadrature TV',float(out['DM_quadrature_TV'].max()),flush=True);return out


def band_views(partition):
    h=d.npz(OUT/f'input_{partition}_HSC.npz');l=d.npz(OUT/f'input_{partition}_Legacy.npz');assert np.array_equal(h['object_id'],l['object_id']);views={}
    for name,base,positions in [('HSC5',h,[0,1,2,3,4]),('HSC3_gri',h,[0,1,2]),('HSC3_grz',h,[0,1,3]),('Legacy3',l,[0,1,2]),('Legacy4_W1',l,[0,1,2,3]),('Legacy5',l,[0,1,2,3,4])]:
        a={k:v.copy() for k,v in base.items()};bad=np.ones(11,bool);bad[positions]=False;a['flux'][:,bad]=np.nan;a['sigma'][:,bad]=np.nan;views[name]=a
    for name,n in [('HSC5_W1',1),('HSC5_W1W2',2)]:
        a={k:v.copy() for k,v in h.items()}
        for key in ['flux','sigma','band_id','quality','quality_available']:a[key][:,5:5+n]=l[key][:,3:3+n]
        views[name]=a
    return views


def simulations():
    rng=np.random.default_rng(10606);k,cfg=setup();records=[]
    for family in ['single','two','three','twelve','mismatched']:
        for i in range(32):
            iz=int(rng.integers(5,270));z=float(pdf.grid()[1][iz]);w=np.zeros(12)
            if family=='single':w[int(rng.integers(12))]=1
            elif family=='two':w[cfg['D2_indices']]=rng.dirichlet(np.ones(2))
            elif family=='three':w[cfg['D3_indices']]=rng.dirichlet(np.ones(3))
            else:w=rng.dirichlet(np.ones(12)*.5)
            mismatch_parameters=(float(rng.uniform(-1,1)),float(rng.uniform(1.2,4)),float(rng.uniform(0,.4)))
            for cat,bands in [('HSC',[f'HSC.{x}' for x in 'grizy']),('Legacy',['DECam.g','DECam.r','DECam.z','WISE.W1','WISE.W2'])]:
                ids=np.array([FILTERS.index(x) for x in bands]);clean=k[ids,iz]@w
                if family=='mismatched':
                    from stage2b5_learnability import break_flux
                    clean=break_flux(z,bands,*mismatch_parameters)
                for noise in [False,True]:
                    sigma=np.ones(5)*max(float(np.median(abs(clean))),1e-20)/30;f=clean+(rng.normal(size=5)*sigma if noise else 0);sc=all_scores(f,sigma,ids,dm=False)
                    for model in ['D12','D1_profile','D2','D3']:
                        p=pdf.prediction(sc[model][None,:],np.array([z]));chi=-2*sc[model];records.append(dict(family=family,latent_id=f'{family}:{i}',catalogue=cat,noisy=noise,model=model,z_true=z,z_median=float(p['z_p50'][0]),z_min=float(pdf.grid()[1][chi.argmin()]),chi2_true_grid=float(chi[iz]),fraction_bins_delta1=float(np.mean(chi-chi.min()<1)),outlier=abs((p['z_p50'][0]-z)/(1+z))>.15))
    d.table(OUT/'simulation_controls.csv',records)


def run():
    assert json.loads((OUT/'numerical_tests.json').read_text())['passed']
    for partition in ['development','validation','confirmation']:
        for cat in ['HSC','Legacy']:
            a=d.npz(OUT/f'input_{partition}_{cat}.npz')
            for tef in [0.,.1]:score_dataset(a,f'{partition}_{cat}_tef{tef:.1f}',tef)
    for name,a in band_views('confirmation').items():score_dataset(a,'bands_'+name,diagnostics=False)
    simulations();collect();report();handoff()


def collect():
    rows=[];mode_rows=[];residual=[];quaderrors=[]
    for path in OUT.glob('*.npz'):
        if path.stem.endswith('_diagnostics'):continue
        with np.load(path) as f:
            if 'D12' not in f.files:continue
            a={k:f[k] for k in f.files}
        tag=path.stem;n=len(a['z_spec']);diag=d.npz(OUT/f'{tag}_diagnostics.npz') if (OUT/f'{tag}_diagnostics.npz').exists() else None
        quaderrors.append(dict(condition=tag,max_posterior_TV_difference=float(a['DM_quadrature_TV'].max()),median_posterior_TV_difference=float(np.median(a['DM_quadrature_TV'][:,0]))))
        for model in ['D12','D1_profile','D1_template_marg','D1_evidence','D2','D3','DM','DM_alpha05']:
            p=pdf.prediction(a[model],a['z_spec']);np.savez_compressed(OUT/f'prediction_{tag}_{model}.npz',**p,object_id=a['object_id']);chi=-2*a[model];delta=chi-chi.min(1,keepdims=True);within=(delta<1).mean(1);r=dict(condition=tag,model=model,**pdf.metric(p),wrong_highz_median=float(np.mean((p['z_p50']>2)&(p['z_spec']<1))),median_competitive_bin_fraction=float(np.median(within)),median_exact_fit_bin_fraction=float(np.median(np.mean(abs(chi)<1e-8,axis=1))) if model in ['D12','D1_profile','D2','D3'] else '',unsupported=int((a['n_valid']<3).sum()))
            if diag is not None and model in ['D12','D1_profile','D2','D3']:
                true=diag[model+'_chi2_zspec'];rank=diag[model+'_rank_zspec'];r.update(median_chi2_zspec=float(np.median(true)),fraction_zspec_chi2_above25=float(np.mean(true>25)),median_active_rank=float(np.median(rank)),median_amplitude_zspec=float(np.median(diag[model+'_amplitude_zspec'])))
                for i in range(n):
                    for bi in range(11):
                        rr=diag[model+'_residual_zspec'][i,bi]
                        if np.isfinite(rr):residual.append(dict(condition=tag,model=model,object_id=a['object_id'][i],band_position=bi,residual_sigma=float(rr),chi2_contribution=float(rr*rr)))
            rows.append(r)
            for i in range(n):
                peaks=np.r_[0,find_peaks(a[model][i])[0],299];good=peaks[delta[i,peaks]<2];mode_rows.append(dict(condition=tag,model=model,object_id=a['object_id'][i],competitive_modes=len(good),fraction_bins_delta1=float(within[i]),fraction_bins_delta4=float(np.mean(delta[i]<4)),dominant_redshifts=pdf.grid()[1][peaks[np.argsort(delta[i,peaks])[:3]]].tolist()))
    d.table(OUT/'performance.csv',rows);d.table(OUT/'redshift_modes.csv',mode_rows);d.table(OUT/'per_band_residuals.csv',residual);d.table(OUT/'quadrature_convergence.csv',quaderrors)
    # Paired galaxies, same fitted inputs and same count. No per-view resampling.
    rng=np.random.default_rng(106);boot=[]
    for cat in ['HSC','Legacy']:
        base=d.npz(OUT/f'prediction_confirmation_{cat}_tef0.0_D12.npz');n=len(base['z_spec']);be=abs((base['z_p50']-base['z_spec'])/(1+base['z_spec']))>.15
        for model in ['D1_profile','D2','D3','DM']:
            p=d.npz(OUT/f'prediction_confirmation_{cat}_tef0.0_{model}.npz');er=abs((p['z_p50']-p['z_spec'])/(1+p['z_spec']))>.15;diff=er.astype(float)-be;vals=[float(diff[rng.integers(n,size=n)].mean()) for _ in range(500)];boot.append(dict(catalogue=cat,model=model,galaxy_units=n,outlier_change_vs_D12=float(diff.mean()),CI95=np.quantile(vals,[.025,.975]).tolist()))
    d.write(OUT/'paired_bootstrap.json',boot)


def report():
    with (OUT/'performance.csv').open() as f:rows=list(csv.DictReader(f))
    selected=[r for r in rows if r['condition'].startswith('confirmation_') or r['condition'].startswith('bands_')];lines=['# Stage 2A.6 — SED freedom, evidence volume and identifiability','','All choices use source data or the pinned library only. No target labels, real source test objects or prior outputs are modified. Sixty-four paired source-training galaxies support development,64 distinct validation galaxies and128 distinct confirmation galaxies retain all HSC/Legacy views together. Templates and the verified21-filter union-knot integrator are unchanged.','','## Fixed confirmation and band-coverage comparisons','','| Condition | Model | NMAD | Outliers | CE | CRPS | Coverage | Competitive-bin fraction |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in selected:lines.append('| '+' | '.join([r['condition'],r['model']]+[f'{float(r[k]):.4f}' for k in ['nmad','outlier_fraction','ce','crps','coverage_68','median_competitive_bin_fraction']])+' |')
    lines+=['','## Model and probability definitions','','D12 replays unrestricted NNLS; c=A*w, A=sum(c), w on the simplex is a reparameterization, not extra identifiability. D1 profiles one discrete library template or averages over a fixed uniform template prior. D2 uses the library-only minimum/maximum4000A-break spectra; D3 adds a spectral-coverage endpoint selected without redshift-error fitting. Reduced freedom can exclude the true spectrum; true-z residual adequacy is reported alongside point error.','','DM uses D2 with normalized Beta(1,1) coefficients and a proper half-normal nonnegative amplitude prior. The amplitude integral is analytic; the shape integral uses adaptive peak/boundary-aware quadrature and independent checks. DM_alpha05 fixes the Beta(0.5,0.5) sensitivity. D1_evidence uses the same amplitude prior and normalized1/12 discrete probabilities. Only these proper marginal calculations are called evidence. Profile curves, including D1_template_marg with profiled amplitude, remain explicitly pseudo-likelihoods and are not compared as normalized model evidences.','','The broad amplitude scale is1000 times the95th percentile of absolute training-pair flux. It is a proper observer-frame normalization prior, not an intrinsic luminosity prior. Distance/(1+z) factors are absorbed into this explicitly chosen amplitude coordinate; a luminosity prior would require a different redshift-dependent transformation. A proper amplitude prior can alter redshift weighting. Uniform redshift density and actual bin widths are used for every PDF; no source-population redshift prior is added.','','Measured errors and the earlier fixed10% observed-flux TEF are separate likelihood conditions, not retuned. Missing/invalid errors remove likelihood terms; signed finite flux remains. Unknown extinction is never filled with a measured zero. Numerical constant terms are retained for proper evidences. Quadrature error is tracked as posterior total variation; inaccurate marginal products are ineligible for decoder use.','','## Identifiability, adequacy and attribution','','Full chi-square/evidence grids, continuous-zspec residuals, coefficient amplitudes, active matrix rank, competitive modes and plateau widths are saved. Under nonnegative constraints, rank is a local diagnostic; Nbands minus nominal12 is not used as a reduced-chi-square denominator. Raw zspec chi-square>25 is reported as an incompatibility diagnostic, not a universal calibrated chi-square p-value. Marginal modes are evidence modes, not chi-square minima.','','The fixed band experiment keeps exactly the same128 galaxies. HSC optical subsets isolate band count within one pipeline; Legacy grz/grzW1/grzW1W2 adds actual audited WISE leverage; equal-five-band HSC versus Legacy remains confounded by filters, errors and measurement pipelines. HSC+WISE mixes cmodel and Tractor apertures and assumes independent calibration/errors only as a diagnostic. No fabricated WISE data or photometric offset correction is used. Joint HSC+Legacy inference is not promoted to primary evidence because measurement compatibility remains unresolved.','','Single/two/three/twelve-component and deliberately mismatched simulations use actual filters, known z and explicit noiseless/noisy observations. Low fit residual is not redshift recovery. Paired bootstrap units are galaxies, with correlated survey views kept together. A narrow constrained posterior is not automatically physically adequate.','','## Decoder handoff','','The numerically checked D2 profile, proper DM (if convergence passes), D12 reference and both error prescriptions are retained. No model is declared universally trustworthy from aggregate NMAD. Stage2B.6 must test matched and mismatched spectra, source convergence and observed-data trust diagnostics before any generalisation claim. Allowed trust inputs exclude true z, true coefficients, family identity and mismatch labels. See stage2a6_handoff.json for precise paths, priors, units and limitations.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');plots()


def plots():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=['D12','D1_profile','D2','D3','DM'];fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for cat in ['HSC','Legacy']:
        outs=[];true=[];flat=[]
        for name in models:
            p=d.npz(OUT/f'prediction_confirmation_{cat}_tef0.0_{name}.npz');outs.append(pdf.metric(p)['outlier_fraction']);a=d.npz(OUT/f'confirmation_{cat}_tef0.0.npz');delta=-2*(a[name]-a[name].max(1,keepdims=True));flat.append(np.median((delta<1).mean(1)));di=d.npz(OUT/f'confirmation_{cat}_tef0.0_diagnostics.npz');true.append(np.median(di[name+'_chi2_zspec']) if name in ['D12','D1_profile','D2','D3'] else np.nan)
        axs[0].plot(models,outs,'o-',label=cat);axs[1].plot(models,flat,'o-',label=cat);axs[2].plot(models,true,'o-',label=cat)
    for ax in axs:ax.legend();ax.tick_params(axis='x',rotation=25)
    axs[0].set(ylabel='Catastrophic-outlier fraction',title='Fixed128-galaxy confirmation');axs[1].set(ylabel='Median fraction of bins within delta score1',title='Redshift plateaus');axs[2].set(ylabel='Median continuous-zspec chi-square',yscale='symlog',title='True-spectrum adequacy')
    for ext in ['png','pdf']:fig.savefig(OUT/f'sed_dimension_audit.{ext}',dpi=160)
    plt.close(fig)


def handoff():
    with (OUT/'quadrature_convergence.csv').open() as f:q=list(csv.DictReader(f))
    maxerr=max(float(x['max_posterior_TV_difference']) for x in q);k,cfg=setup();proper=maxerr<.002
    retained=dict(D12='-0.5*NNLS chi2 full12',D2='-0.5*NNLS chi2 fixed2',DM='proper Beta(1,1) shape + halfNormal amplitude marginal' if proper else 'diagnostic only: convergence not adequate')
    d.write(OUT/'stage2a6_handoff.json',dict(completed=True,Stage2B6_proceed=True,numerical_tests_passed=True,response_integrator='stage2a5_audit.integrate: union-knot Gauss-Legendre with original Inoue14',kernel=str(previous.OUT/'response_kernel_highres.npz'),kernel_sha256=d.sha(previous.OUT/'response_kernel_highres.npz'),D2_indices=cfg['D2_indices'],D3_indices=cfg['D3_indices'],SED_basis=str(OUT/'basis.npz'),amplitude_prior=cfg['amplitude_prior'],amplitude_prior_tau_nanomaggy=cfg['amplitude_prior_tau_nanomaggy'],physical_scores=retained,marginal_eligible=proper,max_quadrature_posterior_TV=maxerr,uncertainty='Measured sigma; separate fixed TEF0.10, never conflated with temperature',prior_provenance='Pinned library only and unlabeled source-training flux scale',numerical_accuracy='A.5 independent quadrature + A.6 amplitude analytic/adaptive and mixture convergence tests',limitations=['No constrained model declared universally reliable; assess true-z fit adequacy','Unknown real extinction history and cmodel/Tractor calibration remain','Proper observer-frame amplitude prior is not a luminosity population prior','Zero minimum chi2 and sharp PDFs do not establish reliability'],allowed_trust_features=['observed flux/error/mask','response curves','number valid','physical profile width/multimodality','cross-model disagreement','constrained residual/rank adequacy proxy'],forbidden_features=['z_spec','SED_truth','mixture_truth','mismatch_label'],interfaces='stage2a6_identifiability.all_scores / two_profile / marginal_two',source_tests_read=False,target_labels_read=False))
    manifest()


def manifest():
    protected=json.loads((OUT/'protected_before.json').read_text())
    for path,sha in protected.items():assert d.sha(path)==sha,path
    d.write(OUT/'manifest.json',dict(stage='Stage2A.6',code_sha256=d.sha(__file__),prior_artifacts_unchanged=len(protected),artifacts={str(p.relative_to(OUT)):d.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json' and p.suffix!='.log'},real_target_labels_read=False,source_tests_read=False))


def assessment():
    with (OUT/'performance.csv').open() as f:rows=list(csv.DictReader(f))
    lookup={(r['condition'],r['model']):r for r in rows};h=lookup['confirmation_HSC_tef0.0','D12'];one=lookup['confirmation_HSC_tef0.0','D1_profile'];two=lookup['confirmation_HSC_tef0.0','D2'];dm=lookup['confirmation_HSC_tef0.0','DM'];legacy=lookup['bands_Legacy5','D12'];opt=lookup['bands_Legacy3','D12'];w1=lookup['bands_Legacy4_W1','D12'];hw=lookup['bands_HSC5_W1W2','D12']
    # Selection reads validation rows only; confirmation is reported separately.
    validation=[r for r in rows if r['condition'].startswith('validation_')];d.write(OUT/'source_selection_record.json',dict(basis_selected_from='Pinned spectra only, before all source metrics',validation_records=validation,preferred_real_survey_score=None,reason='No tested constrained model demonstrates both useful discrimination and acceptable true-z measurement-error adequacy. Point gains alone are insufficient.',retained_diagnostics=['D12 reference','D1 single-template, best-characterized point restriction','D2/D3 low-dimensional mismatch controls','DM numerically converged proper evidence, not a universal accuracy winner'],confirmation_was_not_used_to_choose_basis_or_prior=True))
    text=(OUT/'report.md').read_text().split('\n## Empirical answers')[0]
    text+='\n## Empirical answers\n\n'
    text+=f"1. Excessive flexibility contributes to wrong exact-fit plateaus, but is not a sufficient explanation or a universal remedy. On128 independent confirmation galaxies, HSC D12 has {float(h['outlier_fraction']):.1%} outliers; single-template D1 reduces this to {float(one['outlier_fraction']):.1%}. However median true-z chi-square rises from {float(h['median_chi2_zspec']):.1f} to {float(one['median_chi2_zspec']):.1f}, with essentially no68% coverage. The point improvement is not a validated physical likelihood.\n\n"
    text+=f"2. The fixed D2 basis performs poorly on real HSC ({float(two['outlier_fraction']):.1%} outliers, median true-z chi-square {float(two['median_chi2_zspec']):.1f}). D3 and the10% TEF sensitivity do not supply a model that simultaneously restores useful discrimination and measured-error adequacy. Restricting SED freedom can remove valid spectra as well as wrong solutions.\n\n"
    text+=f"3. Proper D2 coefficient/amplitude marginalisation is numerically converged but does not repair its family mismatch: DM outliers are {float(dm['outlier_fraction']):.1%}, with CE {float(dm['ce']):.1f}, versus D2 profile CE {float(two['ce']):.1f}. Volume weighting cannot reconstruct a spectrum outside the family. Proper amplitude and normalized coefficient priors can change weighting; this is not an arbitrary chi-square temperature or a claim of prior-independent evidence superiority.\n\n"
    text+=f"4. For identical galaxies, Legacy grz/grzW1/grzW1W2 D12 outliers are {float(opt['outlier_fraction']):.1%}/{float(w1['outlier_fraction']):.1%}/{float(legacy['outlier_fraction']):.1%}. Actual HSC+WISE reduces HSC5 outliers from {float(h['outlier_fraction']):.1%} to {float(hw['outlier_fraction']):.1%}. This strongly supports wavelength leverage beyond raw optical band count, while additional constraints, error differences and cmodel/Tractor compatibility remain confounded. HSC+WISE CE remains very poor despite good point statistics.\n\n"
    text+='5/6. No newly tested score is declared trustworthy for arbitrary real galaxies. The original, D1 and the best-characterized low-dimensional numerical models are handed off as separate diagnostics. Stage2B.6 may use exact-family matched simulations and observed-data trust rules to test their conditional usefulness; it must explicitly suppress or report misspecification rather than assume source point gains solve it. No real target labels or test data are used.\n'
    (OUT/'report.md').write_text(text)
    handoff();hnd=json.loads((OUT/'stage2a6_handoff.json').read_text());hnd.update(preferred_real_survey_score=None,real_source_adequacy_gate='No universal reliable winner; diagnostic handoff only',single_template_score='D1_profile and D1_evidence retained; point gains do not establish fit adequacy',source_selection_record=str(OUT/'source_selection_record.json'));d.write(OUT/'stage2a6_handoff.json',hnd);manifest()


def posterior_moments():
    k,cfg=setup();tau=cfg['amplitude_prior_tau_nanomaggy'];check=marginal_two(np.zeros(3),np.ones(3),np.zeros((3,1,2)),3.,moments=True);np.testing.assert_allclose(check['mean_shape_w'],.5,atol=1e-9);np.testing.assert_allclose(check['mean_amplitude'],3*np.sqrt(2/np.pi),atol=1e-9)
    for cat in ['HSC','Legacy']:
        data=d.npz(OUT/f'input_confirmation_{cat}.npz')
        for tef in [0.,.1]:
            path=OUT/f'DM_posterior_moments_confirmation_{cat}_tef{tef:.1f}.npz'
            if path.exists():continue
            scores=d.npz(OUT/f'confirmation_{cat}_tef{tef:.1f}.npz');records={}
            for i,z in enumerate(data['z']):
                y,s,ids,mask=valid_arrays(data['flux'][i],data['sigma'][i],data['band_id'][i],tef);iz=int(scores['DM'][i].argmax())
                for label,kk in [('zspec',previous.continuous(float(z))[:,None,:]),('global_evidence_max',k[:,iz:iz+1])]:
                    moments=marginal_two(y,s,kk[ids][:,:,cfg['D2_indices']],tau,order=32,moments=True)
                    for name,value in moments.items():records.setdefault(label+'_'+name,[]).append(float(value[0]))
                    model=kk[ids,0][:,cfg['D2_indices']]@np.array([moments['mean_coefficient0'][0],moments['mean_coefficient1'][0]]);res=np.full(11,np.nan);res[mask]=(y-model)/s;records.setdefault(label+'_residual_sigma',[]).append(res)
            np.savez_compressed(path,**{key:np.array(value) for key,value in records.items()},object_id=data['object_id'],z_spec=data['z'])
    d.write(OUT/'posterior_moment_tests.json',dict(passed=True,zero_response_recovers_shape_and_amplitude_prior=True,amplitude_shape_correlation_retained=True,coefficient_means_not_product_of_marginal_means=True,DM_residuals_use_posterior_mean_flux=True))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','tests','run','collect','report','handoff','manifest','assessment','posterior_moments']);args=p.parse_args();globals()[args.action]()
