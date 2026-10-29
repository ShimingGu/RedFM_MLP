"""Fresh readouts only. This process never loads a Qwen or adapter model."""
import json,time,sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from anecdote_common_1029 import *
import stage2_numerical as numerical
import stage2_physics as pdf
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
_FEATURES={}
class Head(nn.Module):
    def __init__(self,dim):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim,32),nn.SiLU(),nn.Dropout(.1),nn.Linear(32,300))
    def forward(self,x):return self.net(x)
def numerical_features(a):
    t=numerical.tensor_rows(a);m=t['mask'];x=torch.stack([torch.asinh(t['f']),torch.log(t['e']).clamp(-20,20),torch.asinh(t['f']/t['e']).clamp(-20,20),m.float(),t['q'],t['qa'].float()],-1)*m[:,:,None]
    return torch.cat([x.flatten(1),t['brightness'][:,None],t['ebv'][:,None],t['ebv_available'][:,None].float()],1).numpy()
def data(task,arm,fmt='canonical',pool='last'):
    key=(task,arm,fmt,pool)
    if key in _FEATURES:return _FEATURES[key]
    a=d.npz(OUT/f'{task}_photometry.npz');z=d.npz(OUT/f'{task}_support_development_labels.npz');lookup={x:float(v) for x,v in zip(z['object_id'],z['z'])};y=np.array([lookup.get(x,np.nan) for x in a['object_id']],np.float32)
    if arm=='NUM':x=numerical_features(a)
    else:
        folder=OUT/'features'/arm;manifest=json.loads((folder/'manifest.json').read_text());path=folder/f'{task}_{fmt}.npz';assert d.sha(path)==manifest['feature_files'][task+'_'+fmt]['sha256'];cache=d.npz(path);assert np.array_equal(cache['object_id'],a['object_id']);x=cache[pool]
    _FEATURES[key]=(a,x,y);return a,x,y
def standardize(x,tr):
    mean=x[tr].mean(0);std=x[tr].std(0);positive=std[std>0];floor=max(float(np.median(positive))*.05,1e-4) if len(positive) else 1.;std=np.maximum(std,floor);xx=np.clip((x-mean)/std,-10,10).astype(np.float32);return xx,mean.astype(np.float32),std.astype(np.float32)
def filetag(task,arm,K,subset,seed,loss,fmt,pool):return f'{task}_{arm}_{fmt}_{pool}_{loss}_K{K}_subset{subset}_seed{seed}'
def fit(task,arm,K,subset,seed,loss='hard',fmt='canonical',pool='last'):
    tag=filetag(task,arm,K,subset,seed,loss,fmt,pool);folder=OUT/'heads'/tag;folder.mkdir(parents=True,exist_ok=True);path=folder/'head.pt'
    if path.exists():return json.loads((folder/'summary.json').read_text())
    cfg=json.loads((OUT/'protocol.json').read_text());a,x,y=data(task,arm,fmt,pool);selected=json.loads((OUT/f'{task}_nested_subsets.json').read_text())[str(subset)][str(K)];index={v:i for i,v in enumerate(a['object_id'])};tr=np.array([index[v] for v in selected]);va=np.flatnonzero(a['split']=='development');assert len(tr)==K and not set(tr)&set(va);assert np.all(np.isfinite(y[np.r_[tr,va]]));assert np.all((y[np.r_[tr,va]]>=0)&(y[np.r_[tr,va]]<=4.5));xx,mean,std=standardize(x,tr)
    tx=torch.tensor(xx[tr],device=DEVICE);vx=torch.tensor(xx[va],device=DEVICE);edges=pdf.grid()[0];labels=np.minimum(np.searchsorted(edges,y[np.r_[tr,va]],side='left')-1,299).clip(0);ty=torch.tensor(labels[:K],device=DEVICE);vy=torch.tensor(labels[K:],device=DEVICE)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=Head(x.shape[1]).to(DEVICE);initial_sha=state_hash(model.state_dict());optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['optim']['lr'],weight_decay=cfg['optim']['weight_decay']);assert set(id(p) for g in optimizer.param_groups for p in g['params'])==set(id(p) for p in model.parameters());assert all(p.requires_grad for p in model.parameters())
    width=10. if loss=='smooth10' else 3.
    soft=torch.exp(-.5*((torch.arange(300,device=DEVICE)[None,:]-ty[:,None])/width)**2);soft/=soft.sum(1,keepdim=True);rng=torch.Generator(device=DEVICE).manual_seed(subset*100+seed);history=[];best=float('inf');start=time.time();grad=0.
    for step in range(cfg['optim']['updates']+1):
        if step%cfg['optim']['validation_every']==0:
            model.eval()
            with torch.no_grad():v=model(vx);training=float(F.cross_entropy(model(tx),ty));validation=float(F.cross_entropy(v,vy));assert torch.isfinite(v).all()
            history.append(dict(update=step,train_CE=training,development_CE=validation,gradient_norm=grad))
            if validation<best:best=validation;beststep=step;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if step==cfg['optim']['updates']:break
        model.train();ix=torch.randint(K,(min(64,K),),generator=rng,device=DEVICE);ll=model(tx[ix]);value=F.cross_entropy(ll,ty[ix]) if loss=='hard' else -(soft[ix]*F.log_softmax(ll,1)).sum(1).mean();assert torch.isfinite(value);optimizer.zero_grad();value.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5));assert np.isfinite(grad);optimizer.step()
    model.load_state_dict(beststate);model.eval()
    with torch.no_grad():ll=model(vx).cpu().numpy()
    pred=pdf.prediction(ll,y[va]);metrics=pdf.metric(pred);learned=bool(beststep>0 and history[0]['development_CE']-best>=.05 and history[0]['train_CE']-min(r['train_CE'] for r in history)>=.3)
    summary=dict(task=task,arm=arm,K=K,subset_seed=subset,head_seed=seed,loss=loss,format=fmt,pooling=pool,tag=tag,feature_dim=x.shape[1],parameters=sum(p.numel() for p in model.parameters()),initial_head_sha256=initial_sha,best_update=beststep,initial_development_CE=history[0]['development_CE'],best_development_CE=best,development_CE_gain=history[0]['development_CE']-best,head_learned=learned,development_metrics=metrics,history=history,seconds=time.time()-start,all_optimizer_parameters_fresh_head_only=True,backbone_model_loaded_in_training_process=False,protocol_sha256=d.sha(OUT/'protocol.json'),train_IDs=selected,normalizer_fit_IDs=selected)
    torch.save(dict(state=beststate,mean=mean,std=std,**summary),path);d.write(folder/'summary.json',summary);print('HEAD',tag,'step',beststep,'devCE',round(best,4),'gain',round(summary['development_CE_gain'],3),flush=True);return summary
def pilot():
    assert all((OUT/'features'/b/'manifest.json').exists() for b in BACKBONES)
    results=[]
    for task in TASKS:
        for arm in BACKBONES:
            for K in [64,256]:
                for loss in ['hard','smooth3']+(['smooth10'] if task=='C_ZPHOT' and (OUT/'source_only_sparse_amendment.json').exists() else []):results.append(fit(task,arm,K,101,11,loss))
    gates={}
    for task in TASKS:
        eligible=[arm for arm in BACKBONES if any(r['head_learned'] for r in results if r['task']==task and r['arm']==arm and r['K']==256)];gates[task]=dict(passed=len(eligible)>=2,learning_backbones=eligible)
    passed=all(g['passed'] for g in gates.values());d.write(OUT/'pilot_gate.json',dict(passed=passed,tasks=gates,query_labels_opened=False,source_development_only=True,results=[{k:r[k] for k in ['tag','best_update','head_learned','development_CE_gain']} for r in results]));print('PILOT GATE',passed,gates,flush=True)
def train(task):
    assert json.loads((OUT/'pilot_gate.json').read_text())['tasks'][task]['passed']
    for arm in BACKBONES+['NUM']:
        for subset in [101,202,303]:
            for seed in [11,23]:
                for K in [16,32,64,128,256,512]:fit(task,arm,K,subset,seed)
                for K in [64,256]:
                    fit(task,arm,K,subset,seed,'smooth3')
                    if arm!='NUM':
                        fit(task,arm,K,subset,seed,'hard','canonical','mean')
                        if task in ['GMS','C_SPEC']:fit(task,arm,K,subset,seed,'hard','alternate','last')
                if task=='C_ZPHOT' and (OUT/'source_only_sparse_amendment.json').exists():
                    for K in [16,32,64,128,256,512]:fit(task,arm,K,subset,seed,'smooth10')
    files=sorted((OUT/'heads').glob(task+'_*/head.pt'));d.write(OUT/f'{task}_head_selection_locked.json',dict(task=task,complete=True,head_files_sha256={str(p.relative_to(OUT)):d.sha(p) for p in files},query_labels_opened=False,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())));print('TASK HEADS COMPLETE',task,len(files),flush=True)
if __name__=='__main__':
    from qwen_zero_shot_sanity import reserve_one_gpu
    torch.set_num_threads(2)
    with reserve_one_gpu() as gpu:
        if sys.argv[1]=='pilot':pilot()
        else:train(sys.argv[1])
