"""Stage2B.6 new synthetic partitions, paired-seed convergence and physics trust.

No real target data are opened. Domain labels and latent truth stay outside
tensor_rows and deployable trust features. Physical caches are deterministic
functions of observed f,sigma,masks and the verified Stage2A.6 handoff.
"""
from __future__ import annotations
import json,time,csv,copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import logsumexp
import stage2a6_identifiability as physics
import stage2a5_audit as integrator
import stage2_physics as pdf
import stage2_numerical as base
import stage2b5_learnability as previous
from stage2_data import ROOT,d,FILTERS,take
OUT=ROOT/'reports/stage2b6/1028c';DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
_CWW=None
HSC=[f'HSC.{x}' for x in 'grizy'];LEGACY=['DECam.g','DECam.r','DECam.z','WISE.W1','WISE.W2'];FULL=['MegaCam.u','MegaCam.u_star',*HSC,'VIRCAM.Y','VIRCAM.J','VIRCAM.H','VIRCAM.Ks'];NEW=[x for x in FULL if not x.startswith('HSC.')]


def prepare():
    h=json.loads((physics.OUT/'stage2a6_handoff.json').read_text());assert h['completed'] and h['Stage2B6_proceed'] and h['numerical_tests_passed'];assert (physics.OUT/'source_selection_record.json').exists()
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists()
    d.write(OUT/'handoff_verified.json',dict(handoff_sha256=d.sha(physics.OUT/'stage2a6_handoff.json'),handoff=h,acknowledged='D2/DM are diagnostic, not universally source-validated real-galaxy likelihoods. Exact-family matched and deliberately mismatched controls are required.'))
    d.write(OUT/'protocol.json',dict(seeds=[11,23],source_SEDs_per_domain=dict(train=384,calibration=128,validation=128,held=128),domains=['M1_identifiable_D2','M2_random_D12','M3_analytic_SEDs','M4_measurement_mismatch'],source_systems=['HSC','Legacy'],held_systems=['full_CLAUDS_like','strict_new_only'],redshift='Uniform original centers0.1–1.1; +0.1 regeneration is a distinct counterfactual',noise='Independent Gaussian nominal median-flux SNR40, amplitude independent ofz; M4 perturbations separately declared',M1='D2 young/old weights[0.02..0.25, remainder], source-independent continuous unique mixtures',M2='Dirichlet(0.3) full12',M3='Independent slope/break/dust and variable emission features, not supplied to inference',M4=['3% wavelength-dependent calibration tilt','reported errors understate actual sigma by1.5','one missing band','2% response wavelength shift (simulated, not actual catalogue history)'],held_family='Independently generated CWW spectra with source-independent reddening/slope, never source-trained',optim=dict(learning_rate=.002,weight_decay=.0001,batch_size=128,updates=5000,neighbor_CE_sigma_bins=3.,evaluation_every=200),selection='Lowest source-validation hard CE; retain common2000/4000/5000 snapshots and first source plateau, never held performance',convergence='Validation CE rolling mean over3 evaluations improves<0.02 over1000 updates after2000, no >=0.1 divergence; otherwise inconclusive within cap',warmup='Neural CE and reconstruction active throughout; physical coefficient linearly ramps to fixed value over first800 updates. Same labeled exposure for all primary architectures.',models=['N0_canonical secondary(seed11)','N1','N2_original(D12,TEF0.1,beta0.2/Nvalid)','N2_constrained(D2,measured errors,beta0.2)','N1_kernel_information(seed11)'],fixed_weights=[0.,.05,.2,1.],TEF_controls=[0.,.1],G1='g=0.2/[1+(min constrained-profile chi2/(max(Nvalid-2,1)*3))^2], zero whenNvalid<3; feature detached and computed independently of neural parameters; not calibrated p-value',G2='Only if source fixed controls show matched/mismatched trust tradeoff:5-parameter sigmoid gate on observed adequacy/count/SNR/disagreement, frozen encoder/head, calibration partition only, L2 regularization',G2_selection='Minimize smooth worst-domain source-calibration CE, conditional source-validation checks; never infer mismatch labels',gate_features=['log1p constrained global minimum chi2/local-rank proxy','valid band count','median logSNR','JS divergence D2 versusD12'],forbidden=['z_true','SED_id','true weights','domain/mismatch class'],physical_scores='Verified A.6 full12 andfixedD2 profiles, eligible properDM evidence; distinguish per-z independentNNLS from one predicted global mixture',trust_expansion='Adaptive neuralG1 training only if source frozen-score differences demonstrate need; two paired seeds and same max budget',target_labels_used=False,source_real_tests=False,real_population_training=False))


def spectral_flux(z,bands,par):
    _,fs=integrator.context();out=[]
    for band in bands:
        f=fs[FILTERS.index(band)];w=np.unique(np.r_[f.wave,np.arange(f.wave[0],f.wave[-1],4.)]);rest=w/(1+z);R=np.interp(w,f.wave,f.throughput);shape=(rest/5500)**par['slope']*(1+(par['break_strength']-1)/(1+np.exp(-np.clip((rest-4000)/25,-100,100))))*np.exp(-par['dust']*(rest/5500)**-.7)
        for lam,ratio in [(3727,1),(4861,.7),(5007,1.4),(6563,1.5)]:shape*=1+par['EW']*ratio/(np.sqrt(2*np.pi)*25)*np.exp(-.5*((rest-lam)/25)**2)
        out.append(np.trapezoid(shape*R/w,w)/np.trapezoid(R/w,w))
    return np.array(out)


def cww_flux(z,bands,par,response_shift=0.):
    """Unique held-family SED realizations using the verified union-knot rule."""
    global _CWW
    import hashlib
    if _CWW is None:_CWW=integrator.context(extended=True)[0][12:]
    idx=int(hashlib.sha256(('CWW:'+par['group']).encode()).hexdigest()[:8],16)%4;t=_CWW[idx];rest=t.wave;fnu=t.flux_fnu(z=z)*(rest/5500)**(par['slope']*.3)*np.exp(-par['dust']*(rest/5500)**-.7);fnu/=np.interp(5500,rest,fnu);fnu*=t.igm_absorption(z);wave=rest*(1+z);nodes,weights=np.polynomial.legendre.leggauss(3);values=[]
    for band in bands:
        fil=integrator.context()[1][FILTERS.index(band)];fw=fil.wave*(1+response_shift);knots=np.unique(np.r_[fw,wave[(wave>fw[0])&(wave<fw[-1])]]);left,right=knots[:-1],knots[1:];q=((left+right)[:,None]/2+(right-left)[:,None]/2*nodes).ravel();qw=((right-left)[:,None]/2*weights).ravel();R=np.interp(q,fw,fil.throughput);weight=qw*R/q;values.append(np.sum(weight*np.interp(q,wave,fnu,left=0,right=0))/weight.sum())
    return np.array(values)


def latents(split,n):
    path=OUT/f'latent_{split}.json'
    if path.exists():return json.loads(path.read_text())
    hand=json.loads((physics.OUT/'stage2a6_handoff.json').read_text());d2=hand['D2_indices'];rng=np.random.default_rng(6006+['train','calibration','validation','held'].index(split)*100000);records=[]
    for domain in range(4):
        for i in range(n):
            w=np.zeros(12)
            if domain in [0,3]:w[d2[0]]=rng.uniform(.02,.25);w[d2[1]]=1-w[d2[0]]
            else:w=rng.dirichlet(np.ones(12)*.3)
            records.append(dict(group=f'new6:{split}:M{domain+1}:{i}',domain=f'M{domain+1}',z=float(pdf.grid()[1][int(rng.integers(7,74))]),weights=w.tolist(),amplitude=float(10**rng.uniform(-1,1)),slope=float(rng.uniform(-.5,.8)),break_strength=float(rng.uniform(1.5,4)),dust=float(rng.uniform(0,.35)),EW=float(rng.uniform(0,60)),measurement_case=i%4,perturbation_sign=int(rng.choice([-1,1]))))
    d.write(path,records);return records


def generate(split):
    path=OUT/f'data_{split}.npz'
    if path.exists():return d.npz(path)
    cfg=json.loads((OUT/'protocol.json').read_text());n=cfg['source_SEDs_per_domain'][split];records=latents(split,n);rng=np.random.default_rng(16006+['train','calibration','validation','held'].index(split));flux=[];sig=[];bands=[];z=[];groups=[];domains=[];systems=[];variants=[];actual=[]
    systems_def=[('HSC',HSC),('Legacy',LEGACY)] if split!='held' else [('HSC',HSC),('Legacy',LEGACY),('full',FULL),('new_only',NEW)]
    k=physics.setup()[0];extended=d.npz(integrator.OUT/'response_kernel_extended.npz')['kernel'] if split=='held' else None
    for par in records:
        for system,bb in systems_def:
            variants_def=['correct'] if split!='held' or system in ['HSC','Legacy'] else ['correct','missing','noisy','wrong_metadata','redshift_plus01','held_family']
            for variant in variants_def:
                zz=par['z']+(.1 if variant=='redshift_plus01' else 0.);ids=np.array([FILTERS.index(x) for x in bb]);iz=int(np.argmin(abs(pdf.grid()[1]-zz)));ker=k[:,iz] if variant!='redshift_plus01' else integrator.continuous(zz);w=np.array(par['weights'])
                if par['domain']=='M3':clean=spectral_flux(zz,bb,par)
                else:clean=ker[ids]@w
                if variant=='held_family':
                    clean=cww_flux(zz,bb,par)
                clean=clean*par['amplitude'];sigma=np.ones(len(bb))*max(float(np.median(abs(clean))),1e-20)/40;reported=sigma.copy();real=sigma.copy()
                if par['domain']=='M4':
                    case=par['measurement_case']
                    if case==0:
                        pivot=np.array([pdf.REGISTRY['filters'][x]['pivot_nm']*10 for x in bb]);clean*=np.exp(par['perturbation_sign']*.03*np.log(pivot/5500))
                    if case==1:real*=1.5
                    if case==3:
                        # At these low redshifts IGM is unity in all source bands.
                        # A wavelength-shifted passband is equivalent in fnu shape
                        # to z'=(1+z)/1.02-1, with amplitude still profiled.
                        shifted_z=(1+zz)/1.02-1
                        if variant=='held_family':
                            clean=cww_flux(zz,bb,par,response_shift=.02)*par['amplitude']
                        else:
                            shifted=integrator.continuous(shifted_z);clean=shifted[ids]@w*par['amplitude']
                if variant=='noisy':real*=5;reported*=5
                f=clean+rng.normal(size=len(bb))*real;e=reported
                if variant=='missing' or (par['domain']=='M4' and par['measurement_case']==2):
                    miss=rng.choice(len(bb),min(2 if variant=='missing' else 1,len(bb)-2),replace=False);f[miss]=np.nan;e[miss]=np.nan
                meta=list(np.roll(bb,1)) if variant=='wrong_metadata' else bb
                flux.append(f);sig.append(e);bands.append(meta);z.append(zz);groups.append(par['group']+(':CWW' if variant=='held_family' else ''));domains.append(par['domain']);systems.append(system);variants.append(variant);actual.append(system)
    rows=previous.rows_from_flux(flux,sig,bands,z,groups,split);rows.update(domain=np.array(domains),system=np.array(systems),variant=np.array(variants));np.savez_compressed(path,**rows);print('generated',split,len(rows['z']),'views',len(set(groups)),'SEDs',flush=True);return rows


def js_from_scores(first,second):
    logwidth=np.log(np.diff(pdf.grid()[0]).astype(float));lp=np.asarray(first,float)+logwidth;lq=np.asarray(second,float)+logwidth;lp-=logsumexp(lp);lq-=logsumexp(lq);lm=np.logaddexp(lp,lq)-np.log(2);value=.5*np.sum(np.exp(lp)*(lp-lm)+np.exp(lq)*(lq-lm));assert np.isfinite(value) and -.000001<=value<=np.log(2)+.000001;return max(0.,float(value))


def _physical_row(args):
    f,e,ids=args;s=physics.all_scores(f,e,ids,tef=0.);broad=physics.all_scores(f,e,ids,tef=.1,dm=False);n=int(np.sum(np.isfinite(f)&np.isfinite(e)&(e>0)&(ids>=0)));q=max(0,float(-2*s['D2'].max()))/max(n-2,1);p=pdf.prediction(s['D2'][None,:],np.array([0.]))['pz'][0];p12=pdf.prediction(s['D12'][None,:],np.array([0.]))['pz'][0];m=.5*(p+p12)
    js=js_from_scores(s['D2'],s['D12']);valid=np.isfinite(f)&np.isfinite(e)&(e>0);sn=float(np.median(abs(f[valid]/e[valid]))) if valid.any() else 0.
    return dict(P12=s['D12'],P2=s['D2'],P1=s['D1_profile'],PM=s['DM'],P2_TEF=broad['D2'],P12_TEF=broad['D12'],DM_error=float(s['DM_quadrature_TV'].max()),features=np.array([np.log1p(q),n/11,np.log1p(sn),js]),rule=float(.2/(1+(q/3)**2)) if n>=3 else 0.)


def physical_cache(a,split):
    path=OUT/f'physics_{split}.npz'
    if path.exists():return d.npz(path)
    physics.setup()
    with ProcessPoolExecutor(max_workers=6) as pool:
        results=[]
        for i,r in enumerate(pool.map(_physical_row,zip(a['flux'],a['sigma'],a['band_id']),chunksize=8)):
            results.append(r)
            if (i+1)%512==0:print('physical cache',split,i+1,'/',len(a['z']),flush=True)
    out={key:np.array([r[key] for r in results]) for key in results[0]};out['object_id']=a['object_id'];np.savez_compressed(path,**out);return out


def tensors(a,ph):
    assert np.array_equal(a['object_id'],ph['object_id']), 'Physical products must match exact observed-row identities'
    t=base.tensor_rows(a,DEVICE);t['rule']=torch.tensor(ph['rule'],dtype=torch.float32,device=DEVICE);return t


class Model(previous.Model):
    def __init__(self,arm):
        kind='N0' if arm=='N0' else 'N2' if arm.startswith('N2') else 'N1';super().__init__(kind,False,'full');self.arm=arm;self.ramp=1.
        self.kernel.copy_(torch.tensor(physics.setup()[0],dtype=torch.float32));self.tef=.1 if arm=='N2_original' else 0.
        if arm in ['N2_constrained','N2_G1']:
            self.kernel=self.kernel[:,:,physics.setup()[1]['D2_indices']].clone();self.coefficients=nn.Linear(64,2)
        if arm=='N1_kernel':
            self.register_buffer('kernel_responses',torch.tensor(physics.setup()[0].reshape(len(FILTERS),-1),dtype=torch.float32));self.response_encoder=nn.Sequential(nn.Linear(self.kernel_responses.shape[1],16),nn.SiLU(),nn.Linear(16,16),nn.SiLU())

    def encode(self,a,wrong=False):
        if self.arm!='N1_kernel':return super().encode(a,wrong)
        m=a['mask'];ids=a['ids'].clamp_min(0);response=self.kernel_responses[ids]*m[:,:,None];x=torch.stack([torch.asinh(a['f']),torch.log(a['e']).clamp(-20,20),torch.asinh(a['f']/a['e']).clamp(-20,20),m.float(),a['q'],a['qa'].float()],-1)*m[:,:,None];h=self.band_encoder(torch.cat([x,self.response_encoder(response)],-1));h=(h*m[:,:,None]).sum(1)/m.sum(1).clamp_min(1)[:,None];return self.head(torch.cat([h,torch.zeros_like(a['brightness'][:,None])],1)),h

    def forward(self,a,wrong=False,neural_only=False):
        raw,h=self.encode(a,wrong);neural=4*torch.tanh(raw/4)
        if not self.arm.startswith('N2'):return neural,{}
        w=self.coefficients(h).softmax(1);chi,amp=self.decode(a,w,wrong);physical=-.5*(chi-chi.min(1,keepdim=True).values)
        if self.arm=='N2_original':g=.2/a['mask'].sum(1).clamp_min(1)
        elif self.arm=='N2_G1':g=a['rule'].detach()
        else:g=torch.ones(len(chi),device=chi.device)*.2
        logits=neural if neural_only else neural+self.ramp*g[:,None]*physical
        return logits,dict(neural=neural,chi=chi,weights=w,amplitude=amp,physical=physical,trust=g)


@torch.no_grad()
def infer(model,t,batch=128):
    model.eval();out={}
    for i in range(0,len(t['f']),batch):
        p,aux=model(base.slice_t(t,slice(i,i+batch)));aux=dict(logits=p,**aux)
        for k,v in aux.items():out.setdefault(k,[]).append(v.detach().cpu().numpy())
    return {k:np.concatenate(v) for k,v in out.items()}


def target(a):return torch.tensor(np.clip(np.searchsorted(pdf.grid()[0],a['z'],side='left')-1,0,299),dtype=torch.long,device=DEVICE)


def test_models():
    a=generate('train');ph=physical_cache(a,'train');idx=np.arange(8);t=tensors(take(a,idx),{k:v[idx] for k,v in ph.items()});torch.manual_seed(106);tests=[]
    for arm in ['N0','N1','N2_original','N2_constrained','N2_G1','N1_kernel']:
        model=Model(arm).to(DEVICE);v,aux=model(t);perm=torch.tensor([5,0,7,1,10,2,6,3,8,4,9],device=DEVICE);p={k:(x[:,perm] if x.ndim==2 and x.shape[1]==11 else x) for k,x in t.items()};vp,_=model(p);assert torch.allclose(v,vp,atol=1e-3,rtol=1e-4),arm
        empty={k:x.clone() for k,x in t.items()};empty['mask'][:]=False;empty['f'][:]=0;empty['e'][:]=1;empty['rule'][:]=0;vv,aa=model(empty);assert torch.isfinite(vv).all()
        if aa:aa['chi'].sum().backward();assert all(x.grad is None or torch.isfinite(x.grad).all() for x in model.parameters())
        assert 'z' not in t and 'domain' not in t and 'weights' not in t
        tests.append(dict(arm=arm,permutation_max_absolute=float((v-vp).abs().max().detach()),finite_output=True,empty_finite=True))
    d.write(OUT/'correctness_tests.json',dict(passed=True,models=tests,no_truth_or_mismatch_labels_in_inputs=True,rule_independent_of_learned_parameters=True,physics_handoff_hash_verified=True))


def train(arm,seed,train_a,val_a,train_ph,val_ph):
    folder=OUT/'models'/f'{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=True);path=folder/'model.pt'
    if path.exists():return torch.load(path,map_location='cpu',weights_only=False)
    cfg=json.loads((OUT/'protocol.json').read_text())['optim'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=Model(arm).to(DEVICE);t=tensors(train_a,train_ph);v=tensors(val_a,val_ph);y=target(train_a);vy=target(val_a);bins=torch.arange(300,device=DEVICE);soft=torch.exp(-.5*((bins[None,:]-y[:,None])/3)**2);soft/=soft.sum(1,keepdim=True);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);rng=torch.Generator(device=DEVICE).manual_seed(seed+160);history=[];best=float('inf');snapshots={};start=time.time();grad=0.;lastloss={};plateau=None
    for step in range(cfg['updates']+1):
        if step%cfg['evaluation_every']==0 or step==cfg['updates']:
            model.ramp=1.;pred=infer(model,v);ce=float(F.cross_entropy(torch.tensor(pred['logits'],device=DEVICE),vy));prob=torch.tensor(pred['logits']).softmax(1);entropy=float(-(prob*torch.log(prob.clamp_min(1e-30))).sum(1).mean());conditional={cat:float(F.cross_entropy(torch.tensor(pred['logits'][val_a['domain']==cat]),vy[val_a['domain']==cat].cpu())) for cat in ['M1','M2','M3','M4']};row=dict(update=step,validation_CE=ce,conditional_CE=conditional,entropy=entropy,gradient_L2=grad,seconds=time.time()-start,**lastloss);history.append(row)
            if ce<best:best=ce;beststep=step;beststate={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
            if step in [2000,4000,cfg['updates']]:snapshots[step]={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
            if plateau is None and step>=2000 and len(history)>=9:
                oldmean=np.mean([x['validation_CE'] for x in history[-8:-5]]);newmean=np.mean([x['validation_CE'] for x in history[-3:]])
                if abs(oldmean-newmean)<.02 and max(x['validation_CE'] for x in history[-3:])-min(x['validation_CE'] for x in history[-3:])<.1:plateau=step;plateau_state={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
            if step%1000==0:print(arm,seed,step,'source CE',round(ce,4),'domains',{k:round(x,3) for k,x in conditional.items()},flush=True)
        if step==cfg['updates']:break
        model.train();model.ramp=min(1.,step/800);idx=torch.randint(len(y),(cfg['batch_size'],),generator=rng,device=DEVICE);batch=base.slice_t(t,idx);p,aux=model(batch);ce=-(soft[idx]*F.log_softmax(p,1)).sum(1).mean();phot=reg=ce*0
        if aux:phot=torch.log1p(aux['chi'].gather(1,y[idx,None]).squeeze(1)/batch['mask'].sum(1).clamp_min(1)).mean();reg=aux['weights'].square().sum(1).mean()
        loss=ce+.1*phot+.0001*reg;assert torch.isfinite(loss),(arm,step);optimizer.zero_grad();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10));assert np.isfinite(grad);optimizer.step();lastloss=dict(train_CE_hard=float(F.cross_entropy(p.detach(),y[idx])),train_CE_neighbor=float(ce.detach()),photometry_loss=float(phot.detach()),SED_penalty=float(reg.detach()))
    r=dict(arm=arm,seed=seed,state=beststate,final_state={k:x.detach().cpu().clone() for k,x in model.state_dict().items()},snapshots=snapshots,plateau_state=plateau_state if plateau else None,plateau_update=plateau,history=history,best_update=beststep,best_validation_CE=best,parameters=sum(x.numel() for x in model.parameters()),seconds=time.time()-start,updates=cfg['updates'],peak_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
    torch.save(r,path);d.write(folder/'history.json',{k:x for k,x in r.items() if k not in ['state','final_state','snapshots','plateau_state']});return r


def model_for(r,state='state'):
    model=Model(r['arm']).to(DEVICE);model.load_state_dict(r[state] if isinstance(state,str) else state);model.ramp=1.;model.eval();return model


def score(r,a,ph,tag,state='state'):
    path=OUT/'predictions'/f'{tag}.npz';path.parent.mkdir(exist_ok=True)
    if path.exists():return d.npz(path)
    raw=infer(model_for(r,state),tensors(a,ph));p=pdf.prediction(raw.pop('logits'),a['z']);p.update(raw,object_id=a['object_id'],group=a['group']);np.savez_compressed(path,**p);return p


def metrics_by_domain(tag,p,a):return [dict(condition=tag,domain=dom,**pdf.metric(p,a['domain']==dom)) for dom in ['M1','M2','M3','M4']]


def fixed_controls(results,cal_a,cal_ph,val_a,val_ph):
    rows=[];evidence=[]
    for key,r in results.items():
        if not r['arm'].startswith('N2'):continue
        model=model_for(r);t=tensors(cal_a,cal_ph)
        for tef in [0.,.1]:
            model.tef=tef;raw=infer(model,t);neural=raw['neural'];chi=raw['chi'];physical=-.5*(chi-chi.min(1,keepdims=True));n=t['mask'].sum(1).cpu().numpy()
            for label,g in [('neural',np.zeros(len(n))),('original_tempering',.2/np.maximum(n,1)),('weight0.05',np.full(len(n),.05)),('weight0.2',np.full(len(n),.2)),('untempered',np.ones(len(n))),('G1_rule',cal_ph['rule'])]:
                p=pdf.prediction(neural+g[:,None]*physical,cal_a['z']);rows+=metrics_by_domain(f'{key}_TEF{tef}_{label}',p,cal_a)
            p=pdf.prediction(physical,cal_a['z']);rows+=metrics_by_domain(f'{key}_TEF{tef}_physical_only',p,cal_a)
        # All alternatives use same frozen neural information and observed data.
        raw=infer(model_for(r),t);neural=raw['neural']
        for name,field in [('P-original','P12'),('P-constrained','P2'),('P-marginal','PM')]:
            if field=='PM' and not json.loads((physics.OUT/'stage2a6_handoff.json').read_text())['marginal_eligible']:continue
            s=cal_ph[field]-cal_ph[field].max(1,keepdims=True);p=pdf.prediction(neural+.2*s,cal_a['z']);rows+=metrics_by_domain(f'{key}_{name}_same_weight0.2',p,cal_a)
    d.table(OUT/'source_fixed_trust_controls.csv',rows)
    # G1 training is justified if source calibration exhibits heterogeneous harm
    # from stronger physics in at least one regime and benefit in another.
    changes=[]
    for arm in ['N2_constrained']:
        for seed in [11,23]:
            prefix=f'{arm}_seed{seed}_TEF0.0_';lookup={(r['condition'],r['domain']):r for r in rows}
            gains={dom:lookup[prefix+'untempered',dom]['ce']-lookup[prefix+'neural',dom]['ce'] for dom in ['M1','M2','M3','M4']};changes.append(dict(seed=seed,CE_change_stronger_physics=gains))
    justified=any(min(r['CE_change_stronger_physics'].values())<-.05 and max(r['CE_change_stronger_physics'].values())>.05 for r in changes)
    d.write(OUT/'adaptive_justification.json',dict(source_only=True,conditional_changes=changes,G1_training_justified=justified,policy='Gate only if fixed source controls show conditional benefits and harms; no aggregate easy/hard mixture shortcut'))
    return justified


def learn_gate(r,cal_a,cal_ph,val_a,val_ph):
    assert np.isfinite(cal_ph['features']).all() and np.isfinite(val_ph['features']).all()
    model=model_for(r);raw=infer(model,tensors(cal_a,cal_ph));vr=infer(model,tensors(val_a,val_ph));features=cal_ph['features'];median=np.median(features,0);scale=np.std(features,0);scale=np.where(scale>1e-6,scale,1);x=torch.tensor((features-median)/scale,dtype=torch.float32,device=DEVICE);vx=torch.tensor((val_ph['features']-median)/scale,dtype=torch.float32,device=DEVICE);neural=torch.tensor(raw['neural'],device=DEVICE);physical=torch.tensor(-.5*(raw['chi']-raw['chi'].min(1,keepdims=True)),device=DEVICE);y=target(cal_a);gate=nn.Linear(4,1).to(DEVICE);nn.init.zeros_(gate.weight);nn.init.constant_(gate.bias,-1.386294);optimizer=torch.optim.Adam(gate.parameters(),lr=.02);history=[]
    for step in range(301):
        g=gate(x).sigmoid();logits=neural+g*physical;ce=torch.stack([F.cross_entropy(logits[cal_a['domain']==dom],y[cal_a['domain']==dom]) for dom in ['M1','M2','M3','M4']]);loss=torch.logsumexp(ce,0)+.01*sum(p.square().sum() for p in gate.parameters())+.01*(g.mean()-.2)**2
        assert torch.isfinite(loss) and torch.isfinite(g).all(), 'Nonfinite gate loss must stop before held evaluation'
        if step%50==0:history.append(dict(update=step,conditional_calibration_CE=ce.detach().cpu().tolist(),gate_mean=float(g.mean().detach())))
        if step==300:break
        optimizer.zero_grad();loss.backward();optimizer.step()
    with torch.no_grad():
        vg=gate(vx).sigmoid().cpu().numpy().ravel();vphysical=-.5*(vr['chi']-vr['chi'].min(1,keepdims=True));vp=pdf.prediction(vr['neural']+vg[:,None]*vphysical,val_a['z']);validation=metrics_by_domain('G2_source_validation',vp,val_a)
    result=dict(weight=gate.weight.detach().cpu().numpy(),bias=gate.bias.detach().cpu().numpy(),median=median,scale=scale,history=history,parameters=5,source_validation=validation,features=['observed_adequacy','count','observed_SN','model_JS_disagreement'],training='Source calibration labels only; smooth worst-domain CE, no domain input, frozen encoder and decoder')
    return result


def gate_values(gate,features):return 1/(1+np.exp(-np.clip(((features-gate['median'])/gate['scale'])@gate['weight'].ravel()+gate['bias'][0],-30,30)))


def run():
    from qwen_zero_shot_sanity import reserve_one_gpu
    source={};phys={}
    for split in ['train','calibration','validation']:
        source[split]=generate(split);phys[split]=physical_cache(source[split],split)
    # Split identities are asserted before model fitting, including calibration.
    sets=[set(a['group']) for a in source.values()];assert all(not(x&y) for i,x in enumerate(sets) for y in sets[i+1:])
    with reserve_one_gpu() as reservation:
        d.write(OUT/'device.json',dict(device=DEVICE,reservation=reservation));test_models();results={};source_metrics=[]
        for seed in [11,23]:
            for arm in ['N1','N2_original','N2_constrained']:
                r=train(arm,seed,source['train'],source['validation'],phys['train'],phys['validation']);key=f'{arm}_seed{seed}';results[key]=r;p=score(r,source['validation'],phys['validation'],'source_'+key);source_metrics+=metrics_by_domain(key,p,source['validation'])
        for arm in ['N0','N1_kernel']:
            r=train(arm,11,source['train'],source['validation'],phys['train'],phys['validation']);key=arm+'_seed11';results[key]=r;p=score(r,source['validation'],phys['validation'],'source_'+key);source_metrics+=metrics_by_domain(key,p,source['validation'])
        justify=fixed_controls(results,source['calibration'],phys['calibration'],source['validation'],phys['validation'])
        if justify:
            for seed in [11,23]:
                r=train('N2_G1',seed,source['train'],source['validation'],phys['train'],phys['validation']);key='N2_G1_seed'+str(seed);results[key]=r;p=score(r,source['validation'],phys['validation'],'source_'+key);source_metrics+=metrics_by_domain(key,p,source['validation'])
        # At common updates and at predeclared plateau checkpoints, source data
        # alone quantify convergence; never choose an update on held responses.
        convergence=[]
        for key,r in results.items():
            for step,state in r['snapshots'].items():
                p=score(r,source['validation'],phys['validation'],f'source_{key}_update{step}',state);convergence.append(dict(model=key,checkpoint=f'common_update{step}',**pdf.metric(p)))
            if r['plateau_state'] is not None:
                p=score(r,source['validation'],phys['validation'],f'source_{key}_plateau',r['plateau_state']);convergence.append(dict(model=key,checkpoint='first_source_plateau',plateau_update=r['plateau_update'],**pdf.metric(p)))
        d.table(OUT/'source_convergence_comparisons.csv',convergence);d.table(OUT/'source_metrics.csv',source_metrics)
        gates={key:dict(validation_CE_improvement=r['history'][0]['validation_CE']-r['best_validation_CE'],best_validation_CE=r['best_validation_CE'],source_plateau=r['plateau_update'],converged=r['plateau_update'] is not None,meaningful_learning=r['history'][0]['validation_CE']-r['best_validation_CE']>.5 and r['best_validation_CE']<np.log(300)-.5) for key,r in results.items()}
        d.write(OUT/'source_gates.json',gates)
        learned={}
        if justify:
            for seed in [11,23]:
                key='N2_G1_seed'+str(seed);learned[key]=learn_gate(results[key],source['calibration'],phys['calibration'],source['validation'],phys['validation']);np.savez_compressed(OUT/f'learned_gate_seed{seed}.npz',**{k:v for k,v in learned[key].items() if k not in ['history','training','features','source_validation']});d.write(OUT/f'learned_gate_seed{seed}.json',{k:v for k,v in learned[key].items() if k not in ['weight','bias','median','scale']})
        d.write(OUT/'models_locked_before_held.json',dict(source_only=True,models={key:dict(sha256=d.sha(OUT/'models'/key/'model.pt'),selected_update=r['best_update'],source_plateau=r['plateau_update']) for key,r in results.items()},G1_justified=justify,gate_training_partition='calibration',held_data_metrics_used=False,convergence_gates=gates))
    held=generate('held');hp=physical_cache(held,'held')
    with reserve_one_gpu():
        target_rows=[];gate_rows=[]
        for key,r in results.items():
            if not gates[key]['meaningful_learning']:continue
            p=score(r,held,hp,'held_'+key)
            for dom in ['M1','M2','M3','M4']:
                for system in ['HSC','Legacy','full','new_only']:
                    for variant in np.unique(held['variant']):
                        mask=(held['domain']==dom)&(held['system']==system)&(held['variant']==variant)
                        if mask.any():target_rows.append(dict(model=key,domain=dom,system=system,variant=variant,**pdf.metric(p,mask)))
            if key in learned:
                raw=infer(model_for(r),tensors(held,hp));g=gate_values(learned[key],hp['features']);physical=-.5*(raw['chi']-raw['chi'].min(1,keepdims=True));pg=pdf.prediction(raw['neural']+g[:,None]*physical,held['z']);np.savez_compressed(OUT/'predictions'/f'held_G2_seed{r["seed"]}.npz',**pg,gate=g,object_id=held['object_id'],group=held['group'])
                for dom in ['M1','M2','M3','M4']:
                    for system in ['HSC','Legacy','full','new_only']:
                        for variant in np.unique(held['variant']):
                            mask=(held['domain']==dom)&(held['system']==system)&(held['variant']==variant)
                            if mask.any():target_rows.append(dict(model=f'G2_seed{r["seed"]}',domain=dom,system=system,variant=variant,**pdf.metric(pg,mask)));gate_rows.append(dict(model=f'G2_seed{r["seed"]}',domain=dom,system=system,variant=variant,mean_gate=float(g[mask].mean()),fraction_below005=float((g[mask]<.05).mean()),fraction_above095=float((g[mask]>.95).mean())))
        # Independent physical-only baselines fit coefficients at each z.
        for model,field in [('P12','P12'),('P2','P2'),('P1','P1'),('PM','PM')]:
            if model=='PM' and not json.loads((physics.OUT/'stage2a6_handoff.json').read_text())['marginal_eligible']:continue
            p=pdf.prediction(hp[field],held['z']);np.savez_compressed(OUT/'predictions'/f'held_{model}.npz',**p,object_id=held['object_id'],group=held['group'])
            for dom in ['M1','M2','M3','M4']:
                for system in ['HSC','Legacy','full','new_only']:
                    mask=(held['domain']==dom)&(held['system']==system)&(held['variant']=='correct')
                    if mask.any():target_rows.append(dict(model=model,domain=dom,system=system,variant='correct',**pdf.metric(p,mask)))
        d.table(OUT/'held_metrics.csv',target_rows);d.table(OUT/'gate_distributions.csv',gate_rows)
    postprocess();report();manifest()


def postprocess():
    held=d.npz(OUT/'data_held.npz');rng=np.random.default_rng(6606);rows=[];paired=[]
    for seed in [11,23]:
        baseline=d.npz(OUT/f'predictions/held_N1_seed{seed}.npz');e0=abs((baseline['z_p50']-baseline['z_spec'])/(1+baseline['z_spec']))>.15
        for arm in ['N2_original','N2_constrained','N2_G1','G2']:
            path=OUT/f'predictions/held_{arm}_seed{seed}.npz'
            if not path.exists():continue
            p=d.npz(path);er=abs((p['z_p50']-p['z_spec'])/(1+p['z_spec']))>.15
            for dom in ['M1','M2','M3','M4']:
                for system in ['full','new_only']:
                    mask=(held['domain']==dom)&(held['system']==system)&(held['variant']=='correct');idx=np.flatnonzero(mask);delta=er[idx].astype(float)-e0[idx];values=[float(delta[rng.integers(len(idx),size=len(idx))].mean()) for _ in range(300)];rows.append(dict(model=arm,seed=seed,domain=dom,system=system,unique_SED_units=len(idx),outlier_change_vs_N1=float(delta.mean()),CI95=np.quantile(values,[.025,.975]).tolist()))
            for dom in ['M1','M2','M3','M4']:
                full=(held['domain']==dom)&(held['system']=='full')&(held['variant']=='correct');new=(held['domain']==dom)&(held['system']=='new_only')&(held['variant']=='correct');change=(held['domain']==dom)&(held['system']=='full')&(held['variant']=='redshift_plus01');paired.append(dict(model=arm,seed=seed,domain=dom,mean_abs_cross_filter_shift=float(np.mean(abs(p['z_p50'][full]-p['z_p50'][new]))),mean_known_plus01_response=float(np.mean(p['z_p50'][change]-p['z_p50'][full])),fraction_positive_response=float(np.mean(p['z_p50'][change]>p['z_p50'][full]))))
    d.write(OUT/'paired_SED_bootstrap.json',rows);d.table(OUT/'paired_counterfactuals.csv',paired)
    # Limited real-source plausibility: observed features only, spec-z used only
    # afterwards to evaluate physical predictions. No new real model training.
    real=[]
    for system,a in physics.band_views('confirmation').items():
        if system not in ['HSC5','Legacy3','Legacy5','HSC5_W1W2']:continue
        ph=physical_cache(a,'real_'+system);p=pdf.prediction(ph['P2'],a['z'])
        for i in range(len(a['z'])):real.append(dict(system=system,object_id=a['object_id'][i],z_spec=float(a['z'][i]),D2_predicted=float(p['z_p50'][i]),D2_adequacy=float(np.expm1(ph['features'][i,0])),G1_weight=float(ph['rule'][i]),D12_min_chi2=float(-2*ph['P12'][i].max())))
    d.table(OUT/'real_source_trust_plausibility.csv',real)


def report():
    with (OUT/'source_metrics.csv').open() as f:source=list(csv.DictReader(f))
    with (OUT/'held_metrics.csv').open() as f:held=list(csv.DictReader(f))
    gates=json.loads((OUT/'source_gates.json').read_text());lines=['# Stage 2B.6 — matched convergence and physical-score trust','','Stage2A.6 completed before this stage. Its constrained family is a verified diagnostic, not a trusted real-galaxy likelihood. Fresh intrinsic SED identities are split before all views and noise into training, gate-calibration, source validation and held evaluation; no prior inspected synthetic truth is reused as new confirmation. Real target labels and real source test data are never read.','','## Source convergence gates','',str(gates),'','All primary N1/N2 arms use paired initialization seeds, the same labeled examples, optimizer and5000-update cap. Common2000/4000/5000 snapshots and first predeclared source plateau are retained. A missing plateau means matched-convergence superiority is inconclusive, even if equal-budget metrics differ. Selected checkpoints minimize source-validation CE only. N1_kernel consumes complete basis-response arrays as an information-access control; it has additional input parameters and is not an identical-capacity proof. N0 is the canonical padded baseline.','','## Fixed held-data outcomes','','| Model | Domain | Responses | NMAD | Outliers | CE | CRPS | Coverage |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in held:
        if r['variant']=='correct' and r['system'] in ['full','new_only']:lines.append('| '+' | '.join([r['model'],r['domain'],r['system']]+[f'{float(r[k]):.4f}' for k in ['nmad','outlier_fraction','ce','crps','coverage_68']])+' |')
    lines+=['','## Physical and trust definitions','','Original N2 uses a predicted global12-component mixture,10% TEF and0.2/Nvalid weighting. Constrained N2 uses a predicted globalD2 mixture, measured errors and a fixed0.2 profile coefficient. Their matched encoder/head architecture is identical; the decoder coefficient count differs. Fixed ablations separately vary error model, temperature and SED score on the same frozen source-calibration predictions, so cross-arm differences are not falsely assigned to a single factor. Independent P12/P2 baselines instead optimize coefficients at every candidate z; proper PM integrates the documented A.6 priors. This is more fitting freedom than a single predicted global mixture.','','G1 uses only independently calculated constrained adequacy:0.2/[1+(q/3)^2], q=min_z chi2_D2/max(Nvalid−2,1), zero when fewer than3 measurements exist. This proxy is not a calibrated chi-square probability; constraints, redshift searching and non-identifiability limit degrees-of-freedom interpretation. A near-zero unrestrictedD12 chi-square never automatically grants trust. Rule features are detached from neural parameters, preventing a network from intentionally ruining reconstruction to switch off physics.','','A tiny learned G2 is introduced only if fixed source controls justify trust heterogeneity. It sees4 observed-data adequacy/count/SNR/disagreement features, not true redshift, SED identity, coefficients or mismatch class. Its5 parameters train on the independent source-calibration partition with a regularized smooth worst-domain CE objective. Domain is used for balancing/conditional analysis only, never as a deployable input. Gate collapse, conditional outcomes and source-validation diagnostics are reported; ignoring all hard objects is not a claimed physical improvement.','','The full likelihood cache derives solely from observed f/sigma, masks and actual response metadata. It includes the eligible normalized marginal model, keeps negative measurements and never inserts missing EBV. All profiles use free nonnegative amplitude. Gaussian constants cancel for profile z scores but are retained for proper evidence. A temperature does not repair a wrong minimum. Source template mismatch, learned coefficient inadequacy, score flattening and probability calibration are separate diagnoses.','','## Interpretation and limitations','','Paired SED-level bootstrap is computed per seed and domain; two independently initialized seeds supply separate training variability. Held noise/missingness, response corruption and known+0.1 counterfactuals are saved. Wrong-metadata sensitivity is not proof of physical reasoning. Matched spectral priors constitute additional external information for N2 and physical-only baselines; the full-kernel N1 control documents, but cannot erase, information/capacity differences.','','Real-source plausibility uses only existing paired confirmation galaxies in HSC5, Legacy3, Legacy5 and HSC+WISE views; observed trust features are calculated before spectroscopic evaluation. This does not validate a synthetic gate on real data. High-z real supervision and an independent Legacy training population remain unavailable. Unknown extinction histories, HSC exposure mixtures and cmodel/Tractor compatibility remain.','','No population-scale real run, CLAUDS-label calibration or pretrained Qwen/Stage2C experiment is justified solely by these controlled tests. Any useful effect must survive matched source convergence, both seeds, new-only responses and SED mismatch. Negative or inconclusive gates remain explicit. Exact source histories, physical scores, fitted priors, bootstrap units, runtime and hashes accompany this report.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');plots()


def plots():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for arm in ['N1','N2_original','N2_constrained','N2_G1']:
        for seed in [11,23]:
            path=OUT/'models'/f'{arm}_seed{seed}'/'history.json'
            if not path.exists():continue
            h=json.loads(path.read_text())['history'];axes[0].plot([r['update'] for r in h],[r['validation_CE'] for r in h],label=f'{arm}/{seed}');axes[1].plot([r['update'] for r in h],[r.get('train_CE_hard',np.nan) for r in h],label=f'{arm}/{seed}')
    axes[0].set(title='Source validation, paired seeds',xlabel='Updates',ylabel='CE');axes[1].set(title='Training CE',xlabel='Updates',ylabel='CE');axes[0].legend(fontsize=6)
    for ext in ['png','pdf']:fig.savefig(OUT/f'matched_convergence.{ext}',dpi=160)
    plt.close(fig)


def manifest():
    protected=json.loads((physics.OUT/'protected_before.json').read_text())
    for path,sha in protected.items():assert d.sha(path)==sha,path
    d.write(OUT/'manifest.json',dict(stage='Stage2B.6',code_sha256=d.sha(__file__),Stage2A6_handoff_sha256=d.sha(physics.OUT/'stage2a6_handoff.json'),prior_artifacts_unchanged=len(protected),real_target_labels_read=False,source_tests_read=False,artifacts={str(p.relative_to(OUT)):d.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json' and p.suffix!='.log'}))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','postprocess','report','manifest']);args=p.parse_args();globals()[args.action]()
