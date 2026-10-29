"""Small source-only synthetic learnability and physical-decoder audit."""
from __future__ import annotations
import json,time,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import stage2_numerical as original
import stage2a5_audit as physics
from stage2_data import ROOT,d,FILTERS,take
import stage2_physics as pdf
OUT=ROOT/'reports/stage2b5/1028b';DEVICE='cuda' if torch.cuda.is_available() else 'cpu'


class Model(original.Model):
    def __init__(self,name,coarse=False,response='full'):
        super().__init__(name,response=response,tef=.1);self.coarse=coarse
        self.kernel.copy_(torch.tensor(d.npz(physics.OUT/'response_kernel_highres.npz')['kernel'],dtype=torch.float32))
        if name=='N0':self.register_buffer('canonical_pivot',torch.tensor([f.pivot for f in physics.context()[1]],dtype=torch.float32))
        if coarse:
            if name=='N0':self.mlp[-1]=nn.Linear(64,30)
            else:self.head[-1]=nn.Linear(256,30)
    def encode(self,a,wrong=False):
        if self.kind=='N0':
            # A conventional padded MLP requires a reproducible representation.
            # Canonical wavelength ordering removes random slot assignments;
            # full responses and masks remain explicit numerical inputs.
            order=(self.canonical_pivot[a['ids'].clamp_min(0)]+(~a['mask']).float()*1e10).argsort(1)
            a={k:(v.gather(1,order) if v.ndim==2 and v.shape[1]==11 else v) for k,v in a.items()}
        x,h=super().encode(a,wrong)
        if self.coarse:x=x.repeat_interleave(10,dim=1)
        return x,h


def target(a,device=DEVICE):
    assert np.all(np.isfinite(a['z'])&(a['z']>=0)&(a['z']<=4.5))
    return torch.tensor(np.clip(np.searchsorted(pdf.grid()[0],a['z'],side='left')-1,0,299),dtype=torch.long,device=device)


def entropy(logits):
    logp=F.log_softmax(logits,1);return -(logp.exp()*logp).sum(1).mean().item()


def train(a,v,name,tag,steps=1200,lr=.002,smooth=0.,coarse=False,seed=11,replay=False,shuffle=False,response='full'):
    folder=OUT/'runs'/tag;folder.mkdir(parents=True,exist_ok=True);path=folder/'model.pt'
    if path.exists():return torch.load(path,map_location='cpu',weights_only=False)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=(original.Model(name,response=response,tef=.1) if replay else Model(name,coarse,response)).to(DEVICE);initial={k:x.detach().cpu().clone() for k,x in model.state_dict().items()};tt=original.tensor_rows(a,DEVICE);vt=original.tensor_rows(v,DEVICE);y=target(a);vy=target(v)
    if shuffle:y=y[torch.randperm(len(y),generator=torch.Generator().manual_seed(18)).to(DEVICE)]
    optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001);rng=torch.Generator(device=DEVICE).manual_seed(seed+130);history=[];best=1e30;start=time.time();initial_logits=None;gradient_last={};update_accum=0.
    if smooth:
        bins=torch.arange(300,device=DEVICE);soft=torch.exp(-.5*((bins[None,:]-y[:,None])/smooth).square());soft/=soft.sum(1,keepdim=True)
    for step in range(steps+1):
        if step%50==0 or step==steps:
            model.eval();raw=original.infer(model,tt);val=original.infer(model,vt);tl=torch.tensor(raw['logits'],device=DEVICE);vl=torch.tensor(val['logits'],device=DEVICE);trce=F.cross_entropy(tl,y).item();ce=F.cross_entropy(vl,vy).item();head=[(k,x) for k,x in model.named_parameters() if 'head' in k or 'mlp' in k]
            record=dict(update=step,train_CE=trce,source_validation_CE=ce,train_entropy=entropy(tl),validation_entropy=entropy(vl),parameter_change_L2=float(sum((x.detach().cpu()-initial[k]).square().sum().item() for k,x in model.named_parameters())**.5),head_change_L2=float(sum((x.detach().cpu()-initial[k]).square().sum().item() for k,x in head)**.5),bins_above_uniform_half=float((vl.softmax(1)>1/600).float().sum(1).mean()),gradient=gradient_last,seconds=time.time()-start)
            history.append(record)
            if initial_logits is None:initial_logits=raw['logits'].copy();initial_validation_logits=val['logits'].copy()
            if ce<best:best=ce;best_step=step;beststate={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
            if step%400==0:print(tag,step,round(trce,3),round(ce,3),flush=True)
        if step==steps:break
        model.train();idx=torch.randint(len(y),(64,),generator=rng,device=DEVICE);batch=original.augment(original.slice_t(tt,idx),rng);logits,aux=model(batch)
        ce=-(soft[idx]*F.log_softmax(logits,1)).sum(1).mean() if smooth else F.cross_entropy(logits,y[idx]);phot=reg=ce*0
        if aux:phot=torch.log1p(aux['chi'].gather(1,y[idx,None]).squeeze(1)/batch['mask'].sum(1).clamp_min(1)).mean();reg=aux['weights'].square().sum(1).mean()
        loss=ce+.1*phot+.0001*reg;assert torch.isfinite(loss)
        if aux and step in [0,400,steps-1]:
            pars=list(model.coefficients.parameters());g1=torch.autograd.grad(ce,pars,retain_graph=True,allow_unused=True);g2=torch.autograd.grad(.1*phot,pars,retain_graph=True,allow_unused=True);x=torch.cat([x.flatten() for x in g1 if x is not None]);z=torch.cat([x.flatten() for x in g2 if x is not None]);cos=float(F.cosine_similarity(x,z,dim=0).detach());gradient_last=dict(CE_to_coeff=float(x.norm().detach()),photometry_to_coeff=float(z.norm().detach()),CE_photometry_cosine=cos,neural_logit_range=float((aux['neural'].max(1).values-aux['neural'].min(1).values).median().detach()),physical_logit_range=float((.2*(aux['physical_score'].max(1).values-aux['physical_score'].min(1).values)).median().detach()))
        optim.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10);assert torch.isfinite(norm);gradient_last={**gradient_last,'total_gradient_L2_before_clipping':float(norm)};optim.step()
    result=dict(name=name,tag=tag,seed=seed,state=beststate,final_state={k:x.detach().cpu().clone() for k,x in model.state_dict().items()},initial_state=initial,history=history,best_step=best_step,best_validation_CE=best,steps=steps,lr=lr,smooth=smooth,coarse=coarse,replay=replay,response=response,parameters=sum(x.numel() for x in model.parameters()),seconds=time.time()-start,optimizer='AdamW weight_decay0.0001; no scheduler (same as original)',shuffled_training_labels=shuffle)
    torch.save(result,path);d.write(folder/'history.json',{k:x for k,x in result.items() if k not in ['state','final_state','initial_state']});np.savez_compressed(folder/'initial_final_logits.npz',initial_train=initial_logits,initial_validation=initial_validation_logits,final_train=raw['logits'],final_validation=val['logits']);return result


def model_for(r,final=False):
    model=(original.Model(r['name'],response=r['response'],tef=.1) if r['replay'] else Model(r['name'],r['coarse'],r['response'])).to(DEVICE);model.load_state_dict(r['final_state' if final else 'state']);model.eval();return model


def evaluate(r,a,tag,wrong=False):
    path=OUT/'predictions'/f'{tag}.npz';path.parent.mkdir(exist_ok=True)
    if path.exists():return d.npz(path)
    model=model_for(r);raw=original.infer(model,original.tensor_rows(a,DEVICE),wrong=wrong);pred=pdf.prediction(raw.pop('logits'),a['z']);pred.update(raw);pred['object_id']=a['object_id'];np.savez_compressed(path,**pred);return pred


def prepare():
    assert json.loads((physics.OUT/'handoff.json').read_text())['Stage2B5_proceed'];assert (physics.OUT/'conditional_metrics.csv').exists();OUT.mkdir(parents=True,exist_ok=True)
    d.write(OUT/'protocol.json',dict(frozen_before_training=True,source_only=True,real_CLAUDS_evaluation=False,replay='All original five arms, original kernel/seed11/800 updates/AdamW0.002/0.0001, same random batches and permutation generator; instrument training/gradients without changing objective',tiny='32 deterministic elementary L0 source objects, N0,N1 2000 updates; N2 32 matched-basis source objects2000 updates',ladder=dict(L0='One strong 4000A break, tiny independent intrinsic jitter, noiseless, z0.1–1.1, HSC5',L1='Independent slope/break variations, noiseless HSC5',L2_luminosity='Independent amplitude factor, f and sigma scaled together',L2_dust='Independent power-law internal attenuation',L2_noise='SNR30 independent Gaussian reobservation',L2_missing='One random source band removed',L2_weak='Weak or absent breaks; ambiguity explicitly assessed',L3='Same intrinsic SED/redshift through HSC and Legacy responses, training/source-validation groups disjoint',L4='Original frozen Stage1 synthetic benchmark; new source-only protocol distinctly labelled'),source_counts='1024 unique SEDs training +256 independent SEDs validation for new ladders; two views per SED for L3. Original512 train views/128 SEDs retained separately.',seeds=[11,23],screen=['Original300 hard-label CE lr0.002','Original300 hard-label CE lr0.0005','300-bin Gaussian neighbor-label CE sigma3 lr0.002','30-output head repeated into300 bins, hard-label CE lr0.002'],screen_selection='Original synthetic source-validation hard CE only; no strict held-out filters/families',steps=1200,gate2='N0,N1 L0 independent validation CE improves >=0.5 and outliers<0.15; correct known-mixture oracle chi2 behavior',gate3='At least one numerical architecture gains >=0.3 validation CE on independent source SED instances',gate4='Only trained N1 vs N2, matched/mismatched/oracle and held-out controls with SED bootstrap; no automatic physics claim',gate5='No population-scale real runs; no new target-label evaluations',kernel='Verified Stage2A.5 union-knot Gauss quadrature; original kernel used only for exact failure replay'))


def rows_from_flux(flux,sigma,bands,z,groups,split):
    n=len(z);f=np.full((n,11),np.nan);e=f.copy();ids=np.full((n,11),-1,np.int16)
    for i,(ff,ee,bb) in enumerate(zip(flux,sigma,bands)):f[i,:len(ff)]=ff;e[i,:len(ee)]=ee;ids[i,:len(bb)]=[FILTERS.index(x) for x in bb]
    return dict(flux=f,sigma=e,band_id=ids,quality=np.zeros_like(f),quality_available=np.zeros_like(f,bool),ebv=np.full(n,np.nan),ebv_available=np.zeros(n,bool),z=np.array(z,np.float32),object_id=np.array([f'{g}:view{i}' for i,g in enumerate(groups)]),group=np.array(groups),split=np.full(n,split),catalogue=np.array(['synthetic']*n))


def break_flux(z,bands,slope=.2,strength=3.,av=0.):
    # Separate observed-wavelength high-resolution direct convolution, not FSPS K.
    _,curves=physics.context();out=[]
    for band in bands:
        curve=curves[FILTERS.index(band)];w=np.unique(np.r_[curve.wave,np.arange(curve.wave[0],curve.wave[-1],4.)]);rest=w/(1+z);r=np.interp(w,curve.wave,curve.throughput);shape=(rest/5500)**slope*(1+(strength-1)/(1+np.exp(-np.clip((rest-4000)/20,-100,100))))*np.exp(-av*(rest/5500)**-.7);out.append(np.trapezoid(shape*r/w,w)/np.trapezoid(r/w,w))
    return np.array(out)


HSC=[f'HSC.{x}' for x in 'grizy'];LS=['DECam.g','DECam.r','DECam.z','WISE.W1','WISE.W2']
def generate(task,split,n):
    path=OUT/f'{task}_{split}.npz'
    if path.exists():return d.npz(path)
    rng=np.random.default_rng(1802+sum(map(ord,task))+int(split=='val')*10000);flux=[];sigma=[];bands=[];z=[];groups=[];params=[]
    for i in range(n):
        zz=float(rng.uniform(.1,1.1));slope=.2+float(rng.uniform(-.001,.001));strength=3+float(rng.uniform(-.001,.001));av=0.
        if task!='L0':slope=float(rng.uniform(-.4,.8));strength=float(rng.uniform(1.8,4.5))
        if task in ['L2_dust','L2_noise','L2_missing','L3']:av=float(rng.uniform(0,.4))
        if task=='L2_weak':strength=float(rng.uniform(1,1.05))
        amp=10**rng.uniform(-3,3) if task=='L2_luminosity' else 1.;systems=[HSC,LS] if task=='L3' else [HSC]
        for bb in systems:
            ff=break_flux(zz,bb,slope,strength,av)*amp;ee=np.ones(len(bb))*np.median(abs(ff))*(1/30 if task in ['L2_noise','L2_missing','L3'] else .01)
            if task in ['L2_noise','L2_missing','L3']:ff+=rng.normal(size=len(bb))*ee
            if task=='L2_missing':miss=int(rng.integers(len(bb)));ff[miss]=np.nan;ee[miss]=np.nan
            flux.append(ff);sigma.append(ee);bands.append(bb);z.append(zz);groups.append(f'{task}:{split}:{i}');params.append(dict(slope=slope,strength=strength,av=av,z=zz,group=groups[-1]))
    a=rows_from_flux(flux,sigma,bands,z,groups,split);np.savez_compressed(path,**a);d.write(OUT/f'{task}_{split}_truth.json',params);return a


def matched(split,n):
    path=OUT/f'matched_{split}_{n}.npz'
    if path.exists():return d.npz(path)
    rng=np.random.default_rng(917+int(split=='val'));k=d.npz(physics.OUT/'response_kernel_highres.npz')['kernel'];flux=[];sigma=[];bands=[];zz=[];groups=[];weights=[];indices=[]
    for i in range(n):
        iz=int(rng.integers(7,74));w=rng.dirichlet(np.ones(12)*.15)
        for bb in [HSC,LS]:
            bid=[FILTERS.index(x) for x in bb];f=k[bid,iz]@w;e=np.ones(5)*np.median(f)*.01;flux.append(f);sigma.append(e);bands.append(bb);zz.append(float(pdf.grid()[1][iz]));groups.append(f'matched:{split}:{i}');weights.append(w);indices.append(iz)
    a=rows_from_flux(flux,sigma,bands,zz,groups,split);np.savez_compressed(path,**a);np.savez_compressed(OUT/f'matched_{split}_{n}_truth.npz',weights=np.array(weights),redshift_index=np.array(indices));return a


def oracle():
    a=matched('val',64);truth=d.npz(OUT/'matched_val_64_truth.npz');k=d.npz(physics.OUT/'response_kernel_highres.npz')['kernel'];rows=[];allcs=[]
    for i in range(len(a['z'])):
        ids=a['band_id'][i,:5];f=a['flux'][i,:5];e=a['sigma'][i,:5];w=truth['weights'][i];iz=truth['redshift_index'][i];t=np.einsum('bzk,k->bz',k[ids],w);amp=np.maximum(0,np.sum(f[:,None]*t/e[:,None]**2,0)/np.sum(t*t/e[:,None]**2,0));known=np.sum(((f[:,None]-amp*t)/e[:,None])**2,0);cs=physics.profile((f,e,ids,k,0.));fixed=physics.exact_nnls(f,e,ids,k[:,iz]);assert known[iz]<1e-15 and fixed[0]<1e-15
        rows.append(dict(object_id=a['object_id'][i],z=a['z'][i],known_mixture_best_z=float(pdf.grid()[1][known.argmin()]),known_true_chi2=float(known[iz]),NNLS_true_chi2=fixed[0],NNLS_profile_best_z=float(pdf.grid()[1][cs.argmin()]),fraction_bins_NNLS_chi2_below1=float((cs<1).mean()),known_oracle_within_bin=abs(int(known.argmin())-int(iz))<=1));allcs.append(cs)
    d.table(OUT/'matched_oracle.csv',rows);np.savez_compressed(OUT/'matched_oracle_profiles.npz',chi2=np.array(allcs));d.write(OUT/'oracle_tests.json',dict(passed=True,known_mixture_correct_fraction=float(np.mean([r['known_oracle_within_bin'] for r in rows])),true_z_NNLS_exact=True,warning='With 12 free mixtures and only5 bands, zero chi2 may span many z; a different global argmin within a plateau is not solver failure. Known-mixture oracle has more intrinsic information than neural models.'))
    # Original mismatched source training/validation only; do not touch strict tests.
    a=d.npz(ROOT/'reports/stage2b/1028/synthetic.npz');a=take(a,np.isin(a['split'],['train','validation']));idx=np.linspace(0,len(a['z'])-1,96).astype(int);a=take(a,idx);mismatch=[]
    for i in range(len(a['z'])):
        km=physics.continuous(float(a['z'][i]));fit=physics.exact_nnls(a['flux'][i],a['sigma'][i],a['band_id'][i],km,0.);cs=physics.profile((a['flux'][i],a['sigma'][i],a['band_id'][i],k,0.));mismatch.append(dict(object_id=a['object_id'][i],z_spec=float(a['z'][i]),chi2_z_true=fit[0],chi2_min=float(cs.min()),delta_chi2=float(fit[0]-cs.min()),z_min=float(pdf.grid()[1][cs.argmin()]),median_abs_residual=float(np.nanmedian(abs(fit[3])))))
    d.table(OUT/'mismatched_source_physics.csv',mismatch)


def initial_tests():
    torch.manual_seed(9);a=matched('train',16);t=original.tensor_rows(a,DEVICE);model=Model('N2').to(DEVICE);v,aux=model(t);perm=torch.randperm(11,device=DEVICE);tp={k:(x[:,perm] if x.ndim==2 and x.shape[1]==11 else x) for k,x in t.items()};vp,_=model(tp);assert torch.allclose(v,vp,rtol=1e-4,atol=5e-4)
    empty={k:x.clone() for k,x in t.items()};empty['mask'][:]=False;empty['f'][:]=0;empty['e'][:]=1;vv,aa=model(empty);assert torch.isfinite(vv).all();aa['chi'].sum().backward();assert all(x.grad is None or torch.isfinite(x.grad).all() for x in model.parameters());assert 'z' not in t
    scaled={k:x.copy() for k,x in a.items()};scaled['flux']*=.001;scaled['sigma']*=.001;vs,_=model(original.tensor_rows(scaled,DEVICE));assert torch.allclose(v,vs,rtol=1e-4,atol=5e-4)
    # Kernel cache/direct agreement already independently verified in A.5.
    np.testing.assert_allclose(model.kernel.cpu().numpy(),d.npz(physics.OUT/'response_kernel_highres.npz')['kernel'],rtol=1e-7,atol=1e-7)
    d.write(OUT/'correctness_tests.json',dict(passed=True,permutation=True,empty_finite_output_and_gradients=True,scaling_symmetry=True,no_truth_in_inference_inputs=True,corrected_kernel_copied=True,negative_flux_and_invalid_sigma='Inherited tested tensor_rows/decoder; masks checked in priorStage2B and sourceA.5'))


def replay_all():
    a=d.npz(ROOT/'reports/stage2b/1028/synthetic.npz');tr=take(a,a['split']=='train');va=take(a,a['split']=='validation');summary=[]
    for label in ['N0','N1','N2','N1_summary','N1_shuffled']:
        name='N2' if label=='N2' else 'N0' if label=='N0' else 'N1';response='summary' if label.endswith('summary') else 'shuffled' if label.endswith('shuffled') else 'full';r=train(tr,va,name,'replay_'+label,steps=800,replay=True,response=response);arch=torch.load(ROOT/f'reports/stage2b/1028/models/synthetic_{label}_seed11.pt',map_location='cpu',weights_only=False)
        assert r['best_step']==arch['best_step']==0;ce_diff=abs(r['best_validation_CE']-arch['best_validation_ce']);assert ce_diff<1e-4,ce_diff
        final=r['history'][-1];summary.append(dict(model=label,initial_train_CE=r['history'][0]['train_CE'],final_train_CE=final['train_CE'],initial_validation_CE=r['history'][0]['source_validation_CE'],final_validation_CE=final['source_validation_CE'],best_update=r['best_step'],parameter_change_L2=final['parameter_change_L2'],head_change_L2=final['head_change_L2'],final_entropy=final['validation_entropy']))
    d.table(OUT/'update_zero_reproduction.csv',summary)


def physical_controls(r,a,tag):
    model=model_for(r);t=original.tensor_rows(a,DEVICE);rows=[]
    with torch.no_grad():
        logits,aux=model(t);y=target(a);n=t['mask'].sum(1).clamp_min(1);true=aux['chi'].gather(1,y[:,None]).squeeze(1)/n
        for label,ll in [('combined',logits),('neural_only',aux['neural']),('physical_only',.2*aux['physical_score'])]:rows.append(dict(experiment=tag,condition=label,**pdf.metric(pdf.prediction(ll.cpu().numpy(),a['z'])),median_reduced_true_z_chi2=float(true.median()),median_physical_logit_range=float((.2*(aux['physical_score'].max(1).values-aux['physical_score'].min(1).values)).median())))
    d.table(OUT/f'{tag}_physical_neural_controls.csv',rows)


def elementary_oracle(a,task):
    # Known fixed L0 family: independent grid/profile calculation, no network.
    k=np.array([break_flux(float(z),HSC) for z in pdf.grid()[1]]).T;cs=[]
    for f,e in zip(a['flux'][:,:5],a['sigma'][:,:5]):
        amp=np.maximum(0,np.sum(f[:,None]*k/e[:,None]**2,0)/np.sum(k*k/e[:,None]**2,0));cs.append(np.sum(((f[:,None]-amp*k)/e[:,None])**2,0))
    pred=pdf.prediction(-.5*np.array(cs),a['z']);np.savez_compressed(OUT/f'{task}_physical_only.npz',**pred,chi2=np.array(cs));return pdf.metric(pred)


def score_row(r,a,tag):
    pred=evaluate(r,a,tag);met=pdf.metric(pred);met.update(entropy=float(np.mean(-np.sum(pred['pz']*pred['logits'],1))));return dict(experiment=tag,**met)


def ladder():
    tr=generate('L0','train',1024);va=generate('L0','val',256);results=[];tiny=[]
    for name in ['N0','N1']:
        small=take(tr,np.arange(32));r=train(small,small,name,'tiny_'+name,steps=2000);tiny.append(score_row(r,small,'tiny_'+name))
    m=matched('train',32);r=train(m,m,'N2','tiny_N2_matched',steps=2000);tiny.append(score_row(r,m,'tiny_N2_matched'));d.table(OUT/'tiny_overfit.csv',tiny)
    l0={}
    for name in ['N0','N1']:
        r=train(tr,va,name,'L0_'+name,steps=2000);l0[name]=r;results.append(score_row(r,va,'L0_'+name))
    physical=elementary_oracle(va,'L0');passed=all(x['outlier_fraction']<.15 for x in results) and all(r['history'][0]['source_validation_CE']-r['best_validation_CE']>.5 for r in l0.values())
    d.write(OUT/'elementary_gate.json',dict(passed=passed,results=results,physical_oracle=physical,tiny=tiny))
    if not passed:raise RuntimeError('Elementary learnability gate failed; diagnose source training before architectural comparisons')
    # Promising elementary results justify one independently initialized seed.
    for name in ['N0','N1']:
        r=train(tr,va,name,'L0_'+name+'_seed23',steps=2000,seed=23);results.append(score_row(r,va,'L0_'+name+'_seed23'))
    # Source-only screen on original training and source-validation; held filters
    # and families remain unread by metric or model-selection code.
    syn=d.npz(ROOT/'reports/stage2b/1028/synthetic.npz');st=take(syn,syn['split']=='train');sv=take(syn,syn['split']=='validation');screen=[]
    for label,lr,smooth,coarse in [('hard300',.002,0.,False),('lower_lr',.0005,0.,False),('neighbor_CE',.002,3.,False),('coarse30',.002,0.,True)]:
        rr=train(st,sv,'N1','screen_'+label,lr=lr,smooth=smooth,coarse=coarse);screen.append(dict(label=label,lr=lr,smooth=smooth,coarse=coarse,CE=rr['best_validation_CE'],best_update=rr['best_step']))
    selected=min(screen,key=lambda x:x['CE']);d.write(OUT/'source_optimization_selection.json',dict(candidates=screen,selected=selected,target_or_held_filter_labels_used=False))
    options={k:selected[k] for k in ['lr','smooth','coarse']}
    for task in ['L1','L2_luminosity','L2_dust','L2_noise','L2_missing','L2_weak','L3']:
        traina=generate(task,'train',1024);vala=generate(task,'val',256)
        for name in ['N0','N1']:
            r=train(traina,vala,name,task+'_'+name,**options);results.append(score_row(r,vala,task+'_'+name))
        if task=='L3':
            r=train(traina,vala,'N2','L3_N2_mismatched',**options);results.append(score_row(r,vala,'L3_N2_mismatched'));physical_controls(r,vala,'L3_mismatched')
    # Matched-basis source-family validation with identical labels and budgets.
    mt=matched('train',1024);mv=matched('val',256);matched_results={}
    for name in ['N0','N1','N2']:
        r=train(mt,mv,name,'matched_'+name,**options);matched_results[name]=r;results.append(score_row(r,mv,'matched_'+name))
    physical_controls(matched_results['N2'],mv,'matched')
    # Label-shuffled control learns a different target and cannot establish
    # predictive validation accuracy; report it alongside the honest fit.
    r=train(tr,va,'N1','L0_label_shuffled',steps=1200,shuffle=True,**options);results.append(score_row(r,va,'L0_label_shuffled'))
    # L4 uses the source-selected procedure, never alters original truth/splits.
    l4={}
    for name in ['N0','N1','N2']:
        r=train(st,sv,name,'L4_'+name,**options);l4[name]=r;results.append(score_row(r,sv,'L4_'+name+'_source_validation'))
    d.write(OUT/'locked_before_heldout.json',dict(source_screen=selected,checkpoints={name:d.sha(OUT/'runs'/r['tag']/'model.pt') for name,r in l4.items()},source_validation_improvement={name:r['history'][0]['source_validation_CE']-r['best_validation_CE'] for name,r in l4.items()},real_target_evaluated=False))
    # Evaluate original strict independent sets only if this new protocol has
    # actually learned source-validation; no interpretation of update-zero arms.
    held=syn['split'].astype(str);records=[json.loads(x) for x in (ROOT/'reports/stage1/1027/records.jsonl').read_text().splitlines()];heldrows=[]
    for name,r in l4.items():
        gain=r['history'][0]['source_validation_CE']-r['best_validation_CE']
        if r['best_step']==0 or gain<.3:continue
        pred=evaluate(r,syn,'L4_'+name+'_all_frozen')
        for split in ['test','family_holdout']:
            for system in ['HSC','Legacy','CLAUDS','new_filters']:
                for variant in ['correct','wrong_metadata','noisy','missing_bands','redshift_plus03']:
                    mask=np.array([x['split']==split and x['system']==system and x['variant']==variant for x in records])
                    if mask.any():heldrows.append(dict(model=name,split=split,system=system,variant=variant,**pdf.metric(pred,mask)))
    d.table(OUT/'ladder_metrics.csv',results);d.table(OUT/'heldout_metrics.csv',heldrows)
    d.write(OUT/'gates.json',dict(implementation=True,elementary=passed,source_validation_learning=True,original_L4_trained_models_evaluated_on_heldout=sorted(set(x['model'] for x in heldrows)),population_scale_real_training=False,CLAUDS_labels_used=False,physics_benefit='Assess matched/mismatched/oracle results with SED bootstrap; no automatic superiority declaration'))


def run():
    from qwen_zero_shot_sanity import reserve_one_gpu
    from contextlib import nullcontext
    assert (OUT/'protocol.json').exists()
    with reserve_one_gpu() if DEVICE=='cuda' else nullcontext({}) as info:
        d.write(OUT/'device.json',dict(device=DEVICE,reservation=info));initial_tests();replay_all();oracle();ladder()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args();globals()[args.action]()
