"""Stage 2B small, source-selected numerical/physical pilot; no text backbone.

Target labels are opened only after every checkpoint and source gate is frozen.
"""
from __future__ import annotations
import json,time,copy,os,inspect
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import stage2_data as data
import stage2_physics as phys
from stage2_data import ROOT,FILTERS,REGISTRY,d
OUT=ROOT/'reports/stage2b/1028';A=phys.OUT
SEED=11


def response_features():
    fs=phys.curves();lo=min(f.wave.min() for f in fs);hi=max(f.wave.max() for f in fs)
    wave=np.geomspace(lo,hi,256);full=[];summary=[]
    for name,f in zip(FILTERS,fs):
        r=np.interp(wave,f.wave,f.throughput,left=0,right=0);r/=np.trapezoid(r,np.log(wave))
        full.append(r);entry=REGISTRY['filters'][name];cdf=np.cumsum(f.throughput/f.wave*np.gradient(f.wave));cdf/=cdf[-1]
        q=np.interp([.1,.9],cdf,f.wave);s=np.zeros(256);s[:4]=[np.log(entry['pivot_nm']*10/5500),np.log(entry['width_nm']/entry['pivot_nm']),np.log(q[0]/5500),np.log(q[1]/5500)];summary.append(s)
    # Full support includes W2 and the exact UV/IR curve endpoints.
    assert wave[-1]>=max(fs[FILTERS.index('WISE.W2')].wave) and wave[0]<=min(fs[FILTERS.index('MegaCam.u')].wave)
    return np.array(full,np.float32),np.array(summary,np.float32),wave


def tensor_rows(a,device='cpu'):
    f=np.asarray(a['flux'],float);e=np.asarray(a['sigma'],float);bid=a['band_id'];m=np.isfinite(f)&np.isfinite(e)&(e>0)&(bid>=0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);s=np.nanmedian(np.where(m,np.abs(f),np.nan),axis=1);fallback=np.nanmedian(np.where(m,e,np.nan),axis=1)
    s=np.where(np.isfinite(s)&(s>1e-30),s,np.where(np.isfinite(fallback)&(fallback>1e-30),fallback,1.))
    # Mask before arithmetic; safe placeholders are internal masked storage only.
    ff=np.where(m,f/s[:,None],0);ee=np.where(m,e/s[:,None],1)
    qa=np.asarray(a['quality_available'],bool)&m;q=np.where(qa,np.asarray(a['quality'],float),0)
    out=dict(f=ff,e=ee,ids=bid.astype(np.int64),mask=m,q=q,qa=qa,brightness=np.log10(s),ebv=np.where(a['ebv_available'],a['ebv'],0),ebv_available=a['ebv_available'])
    return {k:torch.as_tensor(v,dtype=torch.bool if k in ['mask','qa','ebv_available'] else torch.long if k=='ids' else torch.float32,device=device) for k,v in out.items()}


def slice_t(a,indices):return {k:v[indices] for k,v in a.items()}


class Model(nn.Module):
    def __init__(self,kind='N1',response='full',brightness=False,tef=.1):
        super().__init__();full,summary,wave=response_features();self.kind=kind;self.response_mode=response;self.use_brightness=brightness;self.tef=tef
        r=summary if response=='summary' else full
        self.register_buffer('responses',torch.from_numpy(r));self.register_buffer('wrong_order',torch.from_numpy(np.random.default_rng(921).permutation(len(FILTERS))))
        self.register_buffer('kernel',torch.tensor(phys.get_kernel(),dtype=torch.float32));self.register_buffer('edges',torch.tensor(phys.grid()[0]))
        if kind=='N0':
            self.mlp=nn.Sequential(nn.Linear(11*(256+6)+1,32),nn.SiLU(),nn.Linear(32,64),nn.SiLU(),nn.Linear(64,300))
        else:
            self.response_encoder=nn.Sequential(nn.Linear(256,16),nn.SiLU(),nn.Linear(16,16),nn.SiLU())
            self.band_encoder=nn.Sequential(nn.Linear(22,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
            self.head=nn.Sequential(nn.Linear(65,256),nn.SiLU(),nn.Linear(256,300))
            if kind=='N2':self.coefficients=nn.Linear(64,12)

    def encode(self,a,wrong=False):
        mask=a['mask'];f=a['f'];e=a['e'];ids=a['ids'].clamp_min(0)
        if self.response_mode=='shuffled' or wrong:ids=self.wrong_order[ids]
        response=self.responses[ids];response=response*mask[:,:,None]
        x=torch.stack([torch.asinh(f),torch.log(e).clamp(-20,20),torch.asinh(f/e).clamp(-20,20),mask.float(),a['q'],a['qa'].float()],dim=-1)*mask[:,:,None]
        bright=a['brightness'][:,None] if self.use_brightness else torch.zeros_like(a['brightness'][:,None])
        if self.kind=='N0':return self.mlp(torch.cat([torch.cat([x,response],dim=-1).flatten(1),bright],dim=1)),None
        r=self.response_encoder(response);h=self.band_encoder(torch.cat([x,r],dim=-1));h=(h*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
        return self.head(torch.cat([h,bright],dim=1)),h

    def decode(self,a,weights,wrong=False):
        ids=a['ids'].clamp_min(0)
        if wrong:ids=self.wrong_order[ids]
        t=torch.einsum('nbzk,nk->nbz',self.kernel[ids],weights)
        # Rescale each candidate shape before profiling to avoid small denominators;
        # this is an exact amplitude reparameterization, with no learned free band flux.
        mask=a['mask'].float();t=t*mask[:,:,None]
        norm2=(t.square().sum(1)/mask.sum(1).clamp_min(1)[:,None]).clamp_min(1e-30)
        t=t/norm2.sqrt()[:,None,:]
        var=a['e'].square()+(self.tef*a['f'].clamp_min(0)).square();iv=mask/var.clamp_min(1e-20)
        num=(a['f'][:,:,None]*t*iv[:,:,None]).sum(1);den=(t.square()*iv[:,:,None]).sum(1).clamp_min(1e-20);amp=(num/den).clamp_min(0)
        chi=((a['f'][:,:,None]-amp[:,None,:]*t).square()*iv[:,:,None]).sum(1)
        return chi,amp

    def forward(self,a,wrong=False,neural_only=False):
        raw,h=self.encode(a,wrong);neural=4*torch.tanh(raw/4)
        if self.kind!='N2':return neural,{}
        w=self.coefficients(h).softmax(1);chi,amp=self.decode(a,w,wrong)
        # Explicit tempered profile score. Source-fixed beta=0.2; no normalization
        # determinant is added: var depends on observed f, not on candidate z.
        score=-.5*(chi-chi.min(1,keepdim=True).values)/a['mask'].sum(1).clamp_min(1)[:,None]
        logits=neural if neural_only else neural+.2*score
        return logits,dict(chi=chi,weights=w,amplitude=amp,neural=neural,physical_score=score)


def tests():
    hand=json.loads((A/'handoff.json').read_text());assert hand['Stage2B_proceed'] and not hand['unresolved_invalidating_error']
    torch.manual_seed(1701);a=d.npz(A/'mixed_eligible_trainval.npz');a=data.take(a,np.arange(8));t=tensor_rows(a);results={}
    perm=torch.tensor([8,2,4,1,10,0,6,3,9,5,7]);p={k:(v[:,perm] if v.ndim==2 and v.shape[1]==11 else v) for k,v in t.items()}
    scaled={k:v.copy() for k,v in a.items()};scaled['flux']*=.0001;scaled['sigma']*=.0001;sc=tensor_rows(scaled)
    for name in ['N1','N2']:
        model=Model(name);model.eval();v,aux=model(t);vp,_=model(p)
        err=float((v-vp).abs().max().detach());assert torch.allclose(v,vp,atol=3e-4,rtol=3e-5),err
        vs,_=model(sc);assert torch.allclose(v,vs,atol=3e-4,rtol=3e-5)
        # Unavailable feature storage can contain arbitrary finite values without effect.
        altered={k:v.clone() for k,v in t.items()};bad=~t['mask'];altered['f'][bad]=17;altered['q'][bad]=92;altered['qa'][bad]=True
        vv,_=model(altered);assert torch.allclose(v,vv,atol=1e-5)
        assert torch.isfinite(v).all() and torch.allclose(v.softmax(1).sum(1),torch.ones(len(v)),atol=1e-6)
        empty={k:v.clone() for k,v in t.items()};empty['mask'][:]=False;empty['f'][:]=0;empty['e'][:]=1
        empty_logits,empty_aux=model(empty);assert torch.isfinite(empty_logits).all()
        if empty_aux:
            assert torch.all(empty_aux['chi']==0)
            empty_aux['chi'].sum().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        results[name]=dict(permutation_max_abs=err,masked_storage_invariance=True,common_scale_invariance=True,normalized_pdf=True)
    # No uncertainty -> no observation. Signed negative f remains an observation.
    test={k:v.copy() for k,v in a.items()};test['flux'][0,0]=-1.;test['sigma'][0,1]=np.nan;t2=tensor_rows(test)
    assert t2['mask'][0,0] and t2['f'][0,0]<0 and not t2['mask'][0,1]
    # Differentiate a synthetic basis mixture at a known grid point. Weights alone
    # are optimized; correct-direction finite differences must match autograd.
    model=Model('N2');obs={k:v[:1].clone() for k,v in t.items()};truth=torch.softmax(torch.arange(12,dtype=torch.float32)*.2,0);ids=obs['ids'].clamp_min(0)
    obs['f']=(model.kernel[ids[0],90]@truth)[None,:];obs['f'][~obs['mask']]=0;obs['e']=torch.ones_like(obs['f'])*.05
    par=torch.zeros((1,12),requires_grad=True)
    loss=model.decode(obs,par.softmax(1))[0][0,90];g=torch.autograd.grad(loss,par)[0];assert torch.isfinite(g).all() and g.abs().sum()>0
    step=.01/max(float(g.norm()),1.);new=model.decode(obs,(par-step*g).softmax(1))[0][0,90];assert new<loss
    eps=1e-3;plus=par.detach().clone();minus=plus.clone();plus[0,3]+=eps;minus[0,3]-=eps
    fd=(model.decode(obs,plus.softmax(1))[0][0,90]-model.decode(obs,minus.softmax(1))[0][0,90])/(2*eps)
    assert torch.allclose(fd,g[0,3],atol=.02,rtol=.03),(fd,g)
    assert 'z' not in tensor_rows(a) and 'z' not in inspect.signature(Model.forward).parameters
    direct=phys.templates()[2].integrate_filter(phys.curves()[FILTERS.index('HSC.i')],z=float(phys.grid()[1][90]),include_igm=True)
    cache=phys.get_kernel()[FILTERS.index('HSC.i'),90,2];assert np.isclose(direct,cache,rtol=2e-5)
    results.update(all_missing_finite_neutral_physics=True,signed_flux=True,absent_uncertainty_excluded=True,AB_normalization=json.loads((A/'physical_tests.json').read_text())['AB_zero_nanomaggy_mag']==22.5,kernel_direct_agreement=True,finite_correct_gradient=True,gradient_finite_difference=float(fd.detach()),gradient_autograd=float(g[0,3]),decoder_inputs_no_labels=True,no_fixed_filter_position=True,passed=True)
    OUT.mkdir(parents=True,exist_ok=True);d.write(OUT/'correctness_tests.json',results)


def prepare():
    assert json.loads((OUT/'correctness_tests.json').read_text())['passed']
    a=d.npz(A/'mixed_eligible_trainval.npz');subsets={}
    for split,n in [('train',1024),('val',512)]:
        groups=sorted(set(a['group'][a['split']==split]),key=lambda x:data.group_fraction('pilot:'+x));groups=groups[:n]
        subsets[split]=data.take(a,(a['split']==split)&np.isin(a['group'],groups))
        np.savez_compressed(OUT/f'real_{split}.npz',**subsets[split])
    assert not set(subsets['train']['group'])&set(subsets['val']['group'])
    records=[json.loads(x) for x in (ROOT/'reports/stage1/1027/records.jsonl').read_text().splitlines()];s=d.npz(ROOT/'reports/stage1/1027/photometry.npz');n=len(records)
    syn=dict(object_id=np.array([str(r['index']) for r in records]),group=s['sed_id'].astype(str),flux=s['raw_flux_nanomaggy'],sigma=s['raw_sigma_nanomaggy'],band_id=np.full((n,11),-1,np.int16),quality=np.zeros((n,11)),quality_available=np.zeros((n,11),bool),ebv=np.full(n,np.nan),ebv_available=np.zeros(n,bool),z=s['z'],split=s['split'],catalogue=np.array([r['system'] for r in records]))
    for i,r in enumerate(records):syn['band_id'][i,:len(r['metadata_bands'])]=[FILTERS.index(x) for x in r['metadata_bands']]
    np.savez_compressed(OUT/'synthetic.npz',**syn)
    full,summary,w=response_features();np.savez_compressed(OUT/'response_features.npz',full=full,summary=summary,wavelength_angstrom=w)
    config=dict(seed=SEED,real_training_galaxies=1024,real_training_views=2048,real_validation_galaxies=512,real_validation_views=1024,real_updates=1000,synthetic_updates=800,batch_size=64,optimizer='AdamW',learning_rate=.002,weight_decay=.0001,checkpoint='Lowest source-validation CE, checked each 50 updates',selection='All training/validation selections fixed using identity hashes before model fitting',label_range_policy='Exclude outside [0,4.5], never clip; audited zero out-of-range source labels',neural_bound=4.,physics_beta=.2,physics_score='-0.5*(chi2-min chi2)/number valid bands',alpha=.1,gamma=.0001,template_error=float(json.loads((A/'settings_locked.json').read_text())['P2_fractional_error']),physical_basis='Verified Stage2A FSPS QSF12, positive simplex coefficients, actual responses; shared coefficients across candidate z',amplitude='Nonnegative per-candidate weighted projection, exact shape rescaling for stability',normalization='Median absolute valid flux; fallback median valid sigma, then internal scale1 if no observations. Same factor divides f and sigma; no measured zeros fabricated',EBV='Value and available recorded separately; omitted from primary neural inputs. Numerical placeholder masked and never used as measurement.',brightness_control='N1_brightness supplies explicit log10 flux scale; primary all models remove amplitude',augmentation='Separate N1_noise and N2_noise: sigma multiplier in {1,3,10,30}, fresh Gaussian incremental variance, source labels only. Common amplitude scaling is already an exact normalization symmetry',response_grid=[float(w[0]),float(w[-1]),len(w)],wrong_response='Fixed RNG921 permutation of full responses, not optimized on target. N1_shuffled trained with wrong metadata; all primary models also get post-hoc wrong-response diagnostics',gate2='Each primary model validation CE below uniform log(300)-0.5, source NMAD<0.12 and outliers<0.30 per survey. If underfit, one source-only extension to 2000/1600 updates allowed.',gate3='N2 must improve both source balanced CE and synthetic held-out-filter outliers over N1 to justify expanded seeds or data. Otherwise only the fixed small target diagnostic.',expansion='Only three-seed promising primary models if Gate3 passes; no larger hyperparameter sweep',target_labels_used=False,definitive_full_mixed_source_population_pending=True)
    d.write(OUT/'pilot_protocol.json',config);d.write(OUT/'dataset_audit.json',json.loads((A/'mixed_overlap_audit.json').read_text()))


def augment(a,rng,noise=False):
    b={k:v.clone() for k,v in a.items()};n=len(b['f']);order=torch.rand((n,11),device=b['f'].device,generator=rng).argsort(1)
    for k in ['f','e','mask','ids','q','qa']:b[k]=b[k].gather(1,order)
    if noise:
        values=torch.tensor([1.,3.,10.,30.],device=b['f'].device);factor=values[torch.randint(4,(n,1),device=b['f'].device,generator=rng)]
        b['f']+=torch.randn(b['f'].shape,device=b['f'].device,generator=rng)*b['e']*torch.sqrt(factor.square()-1)*b['mask'];b['e']*=factor
        # Re-normalize the reobserved data without exposing an amplitude shortcut.
        scale=torch.nanmedian(torch.where(b['mask'],b['f'].abs(),torch.nan),dim=1).values.clamp_min(1e-10);b['f']/=scale[:,None];b['e']/=scale[:,None]
    return b


@torch.no_grad()
def infer(model,t,batch=128,wrong=False,neural_only=False):
    model.eval();logits=[];chi=[];coef=[]
    for i in range(0,len(t['f']),batch):
        p,aux=model(slice_t(t,slice(i,i+batch)),wrong=wrong,neural_only=neural_only);logits.append(p.cpu().numpy())
        if aux:chi.append(aux['chi'].cpu().numpy());coef.append(aux['weights'].cpu().numpy())
    out=dict(logits=np.concatenate(logits))
    if chi:out.update(physical_chi2=np.concatenate(chi),sed_weights=np.concatenate(coef))
    return out


def train_one(name,train,val,domain,seed=SEED,updates=None):
    for cohort in [train,val]:
        assert np.all(np.isfinite(cohort['z'])&(cohort['z']>=0)&(cohort['z']<=4.5)), 'Exclude out-of-grid source labels in preparation; never clip them'
    cfg=json.loads((OUT/'pilot_protocol.json').read_text());folder=OUT/'models';folder.mkdir(exist_ok=True);stem=f'{domain}_{name}_seed{seed}';path=folder/f'{stem}.pt'
    if path.exists():return torch.load(path,map_location='cpu',weights_only=False)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);device='cuda' if torch.cuda.is_available() else 'cpu'
    kind='N0' if name=='N0' else 'N2' if name.startswith('N2') else 'N1';response='summary' if name=='N1_summary' else 'shuffled' if name=='N1_shuffled' else 'full'
    model=Model(kind,response,brightness=name=='N1_brightness',tef=cfg['template_error']).to(device);torch.cuda.reset_peak_memory_stats() if device=='cuda' else None
    tt=tensor_rows(train,device);vt=tensor_rows(val,device);z=torch.tensor(train['z'],device=device);target=(torch.bucketize(z,model.edges.to(device))-1).clamp(0,299).long()
    vy=torch.tensor(np.searchsorted(phys.grid()[0],val['z'],side='left')-1,device=device).clamp(0,299).long()
    optim=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);rng=torch.Generator(device=device).manual_seed(seed+130);history=[];best=float('inf');beststate=None;start=time.time()
    steps=updates or cfg['real_updates' if domain=='real' else 'synthetic_updates']
    for step in range(steps+1):
        if step%50==0 or step==steps:
            model.eval();pred=infer(model,vt);ce=float(F.cross_entropy(torch.tensor(pred['logits'],device=device),vy));history.append(dict(update=step,validation_ce=ce,seconds=time.time()-start))
            if ce<best:best=ce;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};beststep=step
            if step%200==0:print(domain,name,seed,'step',step,'val CE',round(ce,4),flush=True)
        if step==steps:break
        model.train();idx=torch.randint(len(z),(cfg['batch_size'],),device=device,generator=rng);b=augment(slice_t(tt,idx),rng,noise=name.endswith('_noise'));p,aux=model(b);loss=F.cross_entropy(p,target[idx]);phot=reg=torch.tensor(0.,device=device)
        if aux:
            truechi=aux['chi'].gather(1,target[idx,None]).squeeze(1)/b['mask'].sum(1).clamp_min(1);phot=torch.log1p(truechi).mean();reg=aux['weights'].square().sum(1).mean();loss=loss+cfg['alpha']*phot+cfg['gamma']*reg
        assert torch.isfinite(loss),dict(name=name,step=step)
        optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optim.step()
    result=dict(name=name,domain=domain,seed=seed,model_kind=kind,response=response,brightness=name=='N1_brightness',state_dict=beststate,history=history,best_step=beststep,best_validation_ce=best,updates=steps,trainable_parameters=sum(p.numel() for p in model.parameters()),seconds=time.time()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,template_error=cfg['template_error'])
    torch.save(result,path);d.write(OUT/f'{stem}_history.json',{k:v for k,v in result.items() if k!='state_dict'});return result


def load_model(result,device):
    model=Model(result['model_kind'],result['response'],result['brightness'],result['template_error']).to(device);model.load_state_dict(result['state_dict']);model.eval();return model


def score(result,a,stem,wrong=False,neural_only=False):
    path=OUT/'predictions'/f'{stem}.npz';path.parent.mkdir(exist_ok=True)
    if path.exists():return d.npz(path)
    device='cuda' if torch.cuda.is_available() else 'cpu';model=load_model(result,device);t=tensor_rows(a,device);raw=infer(model,t,wrong=wrong,neural_only=neural_only);p=phys.prediction(raw.pop('logits'),a.get('z',np.zeros(len(a['flux']))));p.update(raw);n=t['mask'].sum(1).cpu().numpy();p.update(n_valid=n,unsupported=n<3,object_id=a['object_id'])
    # An all-missing set has no measured information: explicit uniform fallback.
    no=n<1
    if no.any():
        uniform=phys.prediction(np.zeros((int(no.sum()),300)),a['z'][no]);
        for k in ['logits','pz','z_p16','z_p50','z_p84']:p[k][no]=uniform[k]
    np.savez_compressed(path,**p);return p


def source_metrics(result,a,stem):
    p=score(result,a,stem);rows=[]
    for cat in np.unique(a['catalogue']):
        mask=a['catalogue']==cat;s=data.take(a,mask);pp={k:(v[mask] if np.ndim(v)>0 and k not in ['redshift_edges','redshift_centers'] else v) for k,v in p.items()};rows+=phys.conditional(stem+'_'+cat,s,pp)
    return rows


def synthetic_metrics(result,a,records,stem):
    p=score(result,a,stem);rows=[]
    for split in ['validation','test','family_holdout']:
        for system in np.unique(a['catalogue']):
            for variant in ['correct','missing_bands','noisy','wrong_metadata','redshift_plus03']:
                mask=(a['split']==split)&(a['catalogue']==system)&np.array([r['variant']==variant for r in records])
                if not mask.any():continue
                rows.append(dict(condition=stem,split=split,system=system,variant=variant,**phys.metric(p,mask)))
    for family in sorted(set(r['family'] for r in records)):
        mask=np.array([r['split'] in ['test','family_holdout'] and r['family']==family and r['variant']=='correct' for r in records]);
        if mask.any():rows.append(dict(condition=stem,split='family',system=family,variant='correct',**phys.metric(p,mask)))
    # Matched latent identities separate cross-filter consistency from correctness.
    pairs={};lookup={(r['latent_id'],r['system'],r['variant']):i for i,r in enumerate(records)}
    for r in records:
        if r['split']=='test' and r['system']=='HSC' and r['variant']=='correct':
            i=r['index'];j=lookup[(r['latent_id'],'CLAUDS','correct')];k=lookup[(r['latent_id'],'HSC','redshift_plus03')];pairs[r['latent_id']]=dict(HSC=float(p['z_p50'][i]),CLAUDS=float(p['z_p50'][j]),redshift_plus03=float(p['z_p50'][k]),z=float(a['z'][i]))
    shifts=np.array([v['CLAUDS']-v['HSC'] for v in pairs.values()]);response=np.array([v['redshift_plus03']-v['HSC'] for v in pairs.values()]);d.write(OUT/f'{stem}_paired.json',dict(pairs=pairs,mean_abs_cross_filter_shift=float(np.mean(abs(shifts))),mean_known_plus03_response=float(response.mean()),fraction_correct_change_sign=float((response>0).mean())))
    return rows


def run():
    assert json.loads((OUT/'correctness_tests.json').read_text())['passed'];cfg=json.loads((OUT/'pilot_protocol.json').read_text());train=d.npz(OUT/'real_train.npz');val=d.npz(OUT/'real_val.npz');syn=d.npz(OUT/'synthetic.npz');records=[json.loads(x) for x in (ROOT/'reports/stage1/1027/records.jsonl').read_text().splitlines()];st=data.take(syn,syn['split']=='train');sv=data.take(syn,syn['split']=='validation')
    from contextlib import nullcontext
    from qwen_zero_shot_sanity import reserve_one_gpu
    lock=reserve_one_gpu() if torch.cuda.is_available() else nullcontext({})
    with lock as reservation:
        d.write(OUT/'device.json',dict(reservation=reservation,device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',torch=torch.__version__))
        real={};synthetic={};source=[];synthrows=[]
        # Exact requested pilot order precedes controls or expansion.
        for name in ['N0','N1','N2']:
            real[name]=train_one(name,train,val,'real');source+=source_metrics(real[name],val,'real_'+name)
        for name in ['N0','N1','N2']:
            synthetic[name]=train_one(name,st,sv,'synthetic');synthrows+=synthetic_metrics(synthetic[name],syn,records,'synthetic_'+name)
        def learnable(result):
            p=score(result,val,'real_'+result['name']);r=[phys.metric(p,val['catalogue']==cat) for cat in ['HSC','Legacy']]
            return all(x['ce']<np.log(300)-.5 and x['nmad']<.12 and x['outlier_fraction']<.3 for x in r)
        gate2={k:learnable(v) for k,v in real.items()}
        if not all(gate2.values()):
            # A source-only optimization extension is permitted once; no target data yet.
            for name,ok in gate2.items():
                if not ok:
                    previous=OUT/'models'/f'real_{name}_seed{SEED}.pt';previous.rename(previous.with_suffix('.initial.pt'));(OUT/'predictions'/f'real_{name}.npz').rename(OUT/'predictions'/f'real_{name}_initial.npz')
                    real[name]=train_one(name,train,val,'real',updates=2000)
            source=[]
            for name,r in real.items():source+=source_metrics(r,val,'real_'+name)
            gate2={k:learnable(v) for k,v in real.items()}
        # Required response/brightness/augmentation controls remain small fixed pilots.
        for name in ['N1_summary','N1_shuffled','N1_brightness','N1_noise','N2_noise']:
            real[name]=train_one(name,train,val,'real');source+=source_metrics(real[name],val,'real_'+name)
        for name in ['N1_summary','N1_shuffled']:
            synthetic[name]=train_one(name,st,sv,'synthetic');synthrows+=synthetic_metrics(synthetic[name],syn,records,'synthetic_'+name)
        testmask=np.array([r['split']=='test' and r['system']=='new_filters' and r['variant']=='correct' for r in records])
        p1=score(synthetic['N1'],syn,'synthetic_N1');p2=score(synthetic['N2'],syn,'synthetic_N2');sm1=phys.metric(p1,testmask);sm2=phys.metric(p2,testmask)
        gate3=all(gate2.values()) and real['N2']['best_validation_ce']<real['N1']['best_validation_ce'] and sm2['outlier_fraction']<sm1['outlier_fraction']
        if gate3:
            for seed in [23,37]:
                for name in ['N0','N1','N2']:
                    key=name+'_seed'+str(seed);real[key]=train_one(name,train,val,'real',seed=seed);source+=source_metrics(real[key],val,'real_'+key)
                    synthetic[key]=train_one(name,st,sv,'synthetic',seed=seed);synthrows+=synthetic_metrics(synthetic[key],syn,records,'synthetic_'+key)
        gates=dict(gate1=True,gate2=gate2,gate3=gate3,synthetic_new_filters_N1=sm1,synthetic_new_filters_N2=sm2,expanded_seeds=[23,37] if gate3 else [],full_population_expansion=False,reason='Full original independent Legacy training cache unavailable; only audited paired-view pilot. Further expansion requires interpretable gains, not target-label tuning.')
        d.write(OUT/'gates_before_target.json',gates)
        d.write(OUT/'models_locked.json',dict(target_labels_used_for_selection=False,models={k:dict(best_update=v['best_step'],validation_ce=v['best_validation_ce'],sha256=d.sha(OUT/'models'/f"real_{v['name']}_seed{v['seed']}.pt")) for k,v in real.items()},gates=gates))
        # First and only target-label evaluation, after all controls, checkpoints,
        # source-based decisions and expansions have been frozen.
        target=d.npz(A/'clauds_photometry.npz');target['z']=d.npz(d.FAST/'clauds_predictions.npz',('z_spec',))['z_spec'];targetrows=[]
        for name,result in real.items():
            for system in ['full','HSC']:
                a={k:v.copy() for k,v in target.items()}
                if system=='HSC':
                    keep=np.isin(a['band_id'],[FILTERS.index('HSC.'+b) for b in 'grizy']);a['flux'][~keep]=np.nan;a['sigma'][~keep]=np.nan
                stem=f'CLAUDS_{name}_{system}';p=score(result,a,stem);targetrows+=phys.conditional(stem,a,p,True)
            if name in ['N0','N1','N2']:
                p=score(result,target,f'CLAUDS_{name}_wrong_response',wrong=True);targetrows+=phys.conditional(f'CLAUDS_{name}_wrong_response',target,p,True)
            if name=='N2':
                p=score(result,target,'CLAUDS_N2_neural_only',neural_only=True);targetrows+=phys.conditional('CLAUDS_N2_neural_only',target,p,True)
        stress=[]
        for name in ['N0','N1','N2']:
            for condition in ['screen_native','screen_dim7','screen_sigma100','screen_noisy_dim0_b100_seed17']:
                old=d.npz(ROOT/f'reports/stage05/1027/{condition}.npz');f=d.npz(ROOT/f'reports/stage05/1027/inputs/{condition}_photometry.npz')['features'];a=data.make_rows(old['targetid'],f,[f'HSC.{b}' for b in 'grizy'],old['z_spec'],np.full(len(f),'val'),np.zeros(len(f)),np.zeros(len(f)),'HSC');p=score(real[name],a,'stress_'+name+'_'+condition);stress.append(dict(model=name,condition=condition,**phys.metric(p)))
        d.table(OUT/'source_conditional_metrics.csv',source);d.table(OUT/'synthetic_metrics.csv',synthrows);d.table(OUT/'target_conditional_metrics.csv',targetrows);d.table(OUT/'stress_metrics.csv',stress)
        d.table(OUT/'source_performance.csv',[r for r in source if r['stratum']=='all']);d.table(OUT/'target_performance.csv',[r for r in targetrows if r['stratum']=='all']);d.write(OUT/'summary.json',dict(gates=gates,source=[r for r in source if r['stratum']=='all'],target=[r for r in targetrows if r['stratum']=='all'],synthetic=synthrows,target_reference='ZPHOT, not spectroscopic truth',no_target_training_or_selection=True))
    plots();report()


def plots():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for name in ['N0','N1','N2']:
        h=json.loads((OUT/f'real_{name}_seed11_history.json').read_text())['history'];ax[0].plot([r['update'] for r in h],[r['validation_ce'] for r in h],label=name)
        p=d.npz(OUT/f'predictions/CLAUDS_{name}_full.npz');ax[1].hist(p['z_p50'],bins=np.linspace(0,4.5,31),histtype='step',label=name);pit=(p['pz']*(p['redshift_centers'][None,:]<=p['z_spec'][:,None])).sum(1);ax[2].hist(pit,bins=np.linspace(0,1,11),histtype='step',label=name)
    for a in ax:a.legend()
    ax[0].set(xlabel='Updates',ylabel='Source validation CE');ax[1].set(xlabel='CLAUDS predicted median z',ylabel='Objects');ax[2].set(xlabel='CLAUDS PIT',ylabel='Objects')
    for ext in ['png','pdf']:fig.savefig(OUT/f'training_distribution_PIT.{ext}',dpi=150)
    plt.close(fig);fig,axs=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for j,name in enumerate(['N0','N1','N2']):
        for i,system in enumerate(['full','HSC']):
            p=d.npz(OUT/f'predictions/CLAUDS_{name}_{system}.npz');axs[i,j].scatter(p['z_spec'],p['z_p50'],s=10,alpha=.5);axs[i,j].plot([0,4.5],[0,4.5],color='gray');axs[i,j].set(title=f'{name} {system}',xlabel='CLAUDS ZPHOT',ylabel='Predicted median',xlim=(0,4.5),ylim=(0,4.5))
    for ext in ['png','pdf']:fig.savefig(OUT/f'target_scatter.{ext}',dpi=150)
    plt.close(fig)


def report():
    s=json.loads((OUT/'summary.json').read_text());lines=['# Stage 2B — numerical set encoder and differentiable SED pilot','', 'This is the executed small paired HSC+Legacy source pilot. The full independent Legacy training cache is unavailable, so the definitive population-scale mixed-source experiment remains pending. Stage 2A completed its physical checks and baseline evaluation first; handoff.json and the Stage 2B correctness tests pass. No Qwen/LLM training or Stage 2C was performed.','','## Source and target results','','| Condition | N | NMAD | Outliers | CE | CRPS | Coverage | Median z | Narrow fraction |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['source']+s['target']:
        lines.append('| '+r['condition']+' | '+str(r['n'])+' | '+' | '.join(f'{r[k]:.4f}' for k in ['nmad','outlier_fraction','ce','crps','coverage_68','predicted_median','concentration_width_003'])+' |')
    lines+=['','## Architecture and controls','','N0 is a padded numerical MLP with flux/error/mask/quality and sampled responses. N1 encodes each numeric response (256→16→16), shares a band MLP (22→64→64), pools by masked mean, and predicts 300 bins (65→256→300). N2 has exactly the same encoder/head plus a 64→12 positive simplex SED-coefficient predictor. It projects the verified 12 FSPS bases through K[filter,z,component], profiles a nonnegative amplitude at every candidate z, and combines bounded neural logits with a source-fixed tempered profile score. No arbitrary per-filter flux decoder is learned. Exact counts, runtime and peak GPU allocation are in each history JSON; optimizer and label budgets are matched for primary pilots, with any source-only underfit extension recorded.','','For N2, logits = 4*tanh(raw/4) + 0.2*(-0.5*(chi2−min chi2)/Nvalid). Effective variance is measured sigma² plus the Stage-2A source-selected fractional-error term applied to max(observed flux,0). The score is a tempered profile pseudo-likelihood, not a marginalized physical posterior. Candidate shape RMS rescaling is an exact amplitude reparameterization. CE plus 0.1*log(1+reduced true-z chi2) and 1e-4*sum(w²) trains coefficients; true source z indexes the training loss only and is absent from inference inputs. The explicit positive beta and bounded correction prevent silent removal of the physical score. Neural-only ablation is separately labelled.','','The response grid covers the complete registered UV through W2 wavelength range. Full curves, compact four-number response summaries, and a fixed shuffled-response training control are compared. Post-hoc response corruption is a diagnostic, not proof of physics. All primary models remove common amplitude using the median absolute valid flux; brightness is supplied only to N1_brightness. N1_noise/N2_noise are separate source-only incremental-variance augmentations with error multipliers 1,3,10,30. Common dimming is a normalization symmetry, not a realistic high-z population. EBV value and availability are saved separately and excluded from the primary neural representation. Missing numerical storage is masked before aggregation and likelihood; no unavailable flux is treated as a precise zero.','','## Splits, support and gates','', 'The real pilot uses 1,024 training and 512 validation galaxies, each with two actual survey views (2,048/1,024 observations), deterministic identity hashes and survey-balanced counts. Original HSC and Legacy source test assignments are excluded. Validation is the union of both original validation memberships, so no validation galaxy enters training through its other survey. Identity, imaging group, positional and target overlaps are audited. See dataset_audit.json. The paired population mostly lies below z=1, has very few z>1 objects and none at z≥2; brightness/noise augmentation supplies no missing high-z labels.','',f"Gate results: {s['gates']}",'','The synthetic experiment reuses the Stage-1 disjoint SED-instance splits, source filters, held-out CLAUDS/new-filter observations and entirely held-out double-break/emission family. Its 512 source labels are distinct from real training. The same 300-bin head is used for all new models. synthetic_metrics.csv separates systems, families, missing/noisy/wrong-response variants; paired JSONs quantify identical-latent filter changes and the known +0.3 redshift change. Stage 1’s physical baseline remains a context result with a different template bank and 90-bin PDF grid; CE must not be compared across grids without rebinning. The Stage-1 observation model uses simplified IGM/dust, so it does not exactly match the Stage-2A FSPS/Inoue14 physical basis.','','## Scientific interpretation','', 'N1 versus N0 isolates the tested shared-set architecture under a small supervised budget; N2 versus N1 adds the explicit SED decoder and reconstruction objective. Equal parameter counts do not imply equal optimization or effective capacity. Use source convergence and the frozen synthetic gate before interpreting target differences. No target result selected any checkpoint, response encoding, error floor, augmentation or training budget.','', 'CLAUDS labels are Phosphoros ZPHOT and share broad template-fitting assumptions with the physical baseline. Object-dependent predictions, useful outlier rates and calibrated PDFs must be considered together. Accuracy against this reference is exploratory agreement, not independent spectroscopic confirmation. Underfit source models, missing high-z support, brightness/noise shift, unproven historical extinction equivalence, nominal HSC r/i curves and template mismatch limit attribution. A numerical or physical gain cannot establish a foundation-model/pretraining advantage; Stage 2C has not been initiated.','','## Reproduction','','Run scripts/stage2_numerical.py tests, prepare and run, using reproduce.sh and requirements-executed.txt. The source protocol is saved before training; models_locked.json is written before opening CLAUDS labels. Per-object PDFs, N2 per-z chi-square/SED weights, histories, hashes, controls, source/target conditional metrics and plots are separate from Stage 2A.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['tests','prepare','run','report']);a=p.parse_args();globals()[a.action]()
