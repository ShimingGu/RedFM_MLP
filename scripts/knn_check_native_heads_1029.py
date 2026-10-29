"""Six fresh BS heads on frozen native features, identical Stage4 sigma3 protocol."""
import sys,time,json
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'notebooks'),str(ROOT)]
from anecdote_common_1029 import *
from anecdote_heads_1029 import Head,standardize
from anecdote_evaluate_1029 import from_logits
from qwen_zero_shot_sanity import reserve_one_gpu
import stage2_physics as pdf
DEST=ROOT/'reports/stage4_5/readout_knn_discrepancy_1029'
def main():
 torch.set_num_threads(2);start=time.time();a=d.npz(DEST/'BS_native_features.npz');phot=d.npz(OUT/'GMS_photometry.npz');clean=np.isin(phot['split'],['support_pool','development','query']);assert np.array_equal(a['object_id'],phot['object_id'][clean]);source=d.npz(OUT/'GMS_support_development_labels.npz');truth=d.npz(OUT/'GMS_query_labels_sealed.npz');lookup={str(i):float(z) for f in [source,truth] for i,z in zip(f['object_id'],f['z'])};y=np.array([lookup[v] for v in a['object_id']],np.float32);ids={x:i for i,x in enumerate(a['object_id'])};splits=json.loads((OUT/'GMS_nested_subsets.json').read_text());va=np.flatnonzero(a['split']=='development');qu=np.flatnonzero(a['split']=='query');cfg=json.loads((OUT/'protocol.json').read_text());edges=pdf.grid()[0];history=[];models=DEST/'native_fresh_heads';models.mkdir(exist_ok=True)
 with reserve_one_gpu() as gpu:
  for subset in [101,202,303]:
   tr=np.array([ids[v] for v in splits[str(subset)]['256']]);xx,mean,std=standardize(a['last'],tr);tx=torch.tensor(xx[tr],device='cuda');vx=torch.tensor(xx[va],device='cuda');labels=np.minimum(np.searchsorted(edges,y[np.r_[tr,va]],side='left')-1,299).clip(0);ty=torch.tensor(labels[:256],device='cuda');vy=torch.tensor(labels[256:],device='cuda');soft=torch.exp(-.5*((torch.arange(300,device='cuda')[None,:]-ty[:,None])/3.)**2);soft/=soft.sum(1,keepdim=True)
   for seed in [11,23]:
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=Head(2560).to('cuda');initial=state_hash(model.state_dict());old_summary=json.loads((OUT/'heads'/f'GMS_BS_canonical_last_smooth3_K256_subset{subset}_seed{seed}'/'summary.json').read_text());assert initial==old_summary['initial_head_sha256'];opt=torch.optim.AdamW(model.parameters(),lr=cfg['optim']['lr'],weight_decay=cfg['optim']['weight_decay']);rng=torch.Generator(device='cuda').manual_seed(subset*100+seed);best=float('inf');hist=[]
    for step in range(cfg['optim']['updates']+1):
     if step%cfg['optim']['validation_every']==0:
      model.eval()
      with torch.no_grad():tl=float(F.cross_entropy(model(tx),ty));vl=float(F.cross_entropy(model(vx),vy))
      hist.append(dict(update=step,train_hard_CE=tl,dev_hard_CE=vl))
      if vl<best:best=vl;bst=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
     if step==cfg['optim']['updates']:break
     model.train();ix=torch.randint(256,(64,),generator=rng,device='cuda');loss=-(soft[ix]*F.log_softmax(model(tx[ix]),1)).sum(1).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
    model.load_state_dict(state);model.eval()
    with torch.inference_mode(): logits=model(torch.tensor(xx[qu],device='cuda')).cpu().numpy()
    p=from_logits(logits,y[qu]);tag=f'BS_native_smooth3_subset{subset}_seed{seed}';np.savez_compressed(DEST/'predictions'/f'{tag}.npz',object_id=a['object_id'][qu],**p);torch.save(dict(state=state,mean=mean,std=std,initial_head_sha256=initial,train_IDs=a['object_id'][tr].tolist(),history=hist,best_update=bst,best_dev_hard_CE=best),models/f'{tag}.pt'); row=dict(tag=tag,subset_seed=subset,head_seed=seed,best_update=bst,head_learned=bool(bst>0 and hist[0]['dev_hard_CE']-best>=.05 and hist[0]['train_hard_CE']-min(h['train_hard_CE'] for h in hist)>=.3),history=hist,metrics=pdf.metric(p));history.append(row);print('NATIVE HEAD',subset,seed,bst,row['metrics']['outlier_fraction'],flush=True)
 d.write(DEST/'native_head_training.json',dict(results=history,GPU=gpu,elapsed_seconds=time.time()-start,source_train_labels=256,development_labels=256,backbone_modified=False,targets='3-bin Gaussian centered on hard label bin; development hard CE',initialization= 'same seed as Stage4 fresh heads'))
if __name__=='__main__':main()
