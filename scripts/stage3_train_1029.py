"""Frozen-protocol source-only compact real-survey benchmark."""
from pathlib import Path
import sys,json,time,copy
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'notebooks'),str(ROOT)]
from stage3_models_1029 import Model,OUT as A
from stage2_data import d,take
import stage2_numerical as base
import stage2_physics as pdf
OUT=ROOT/'reports/stage3b/1029'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
def targets(a):
    z=a['z'];assert np.all(np.isfinite(z)&(z>=0)&(z<=4.5))
    return torch.tensor(np.minimum(np.searchsorted(pdf.grid()[0],z,side='left')-1,299).clip(0),dtype=torch.long,device=DEVICE)
@torch.no_grad()
def logits(model,t):
    model.eval();n=len(t['f']);return np.concatenate([model(base.slice_t(t,slice(i,i+512))).cpu().numpy() for i in range(0,n,512)])
def freeze():
    assert not (A/'benchmark.json').exists()
    assert (A/'identity_split_audit.json').exists() and json.loads((A/'model_contract_tests.json').read_text())['passed']
    cfg=dict(stage='Stage3 compact real-survey benchmark',seeds=[11,23],experiments={'E1':'HSC source -> independently grouped Legacy validation','E2':'Legacy source -> independently grouped HSC validation','E3':'Mixed independent populations -> spatially reserved COSMOS T2'},models_primary=['R1','R2'],controls=['R0 seed11 Mixed','C1 Mixed both seeds','C2 Mixed both seeds','C3 Mixed both seeds','R1_brightness seed11 Mixed'],model_details=json.loads((A/'representation_definition.json').read_text()),optim=dict(updates=5000,batch_size=256,learning_rate=.002,weight_decay=.0001,evaluate_every=250,neighbor_sigma_bins=1.,grad_clip=10.),loss='Unit-bin Gaussian neighbor CE sigma1; hard CE selects checkpoints',selection='Minimum survey-balanced source validation hard CE; common2500/5000 snapshots; no target tuning',mixed_sampling='One deterministic view per physical galaxy fixed in preparation; half batch uniformly sampled from each observed survey. Repeated galaxies never receive extra views or labels.',convergence='Final three-evaluation mean improves <0.02 over mean1000 updates earlier, after3000 updates; no validation divergence>0.1. Otherwise common-budget inference only; no matched-convergence claim.',learning_gate='Every primary source/seed best_update>0; validation CE improvement>=0.5; outliers<0.35 and NMAD<0.12; finite normalized PDFs; no nonfinite gradients. Fail -> no T2 labels.',grid=dict(bins=300,zmin=0,zmax=4.5,outside='Exclude and count; never clip target truth'),point='PDF median at first cumulative probability>=0.5 on bin centers',primary_endpoint='Paired galaxy outlier fraction R2-R1; abs(zmedian-zspec)/(1+zspec)>0.15; T2 full band; per seed',uncertainty='2000 paired galaxy bootstrap replicates, seed1029; 2.5/97.5 percentiles conditional on trained model. Training-seed range and SD separate.',secondary=['NMAD1.48 centered absolute deviation','mean normalized bias','hard CE','CRPS','nominal68%coverage','PIT histogram','PDF concentration','failure/insufficient bands','conditional z<1/1..2/>=2 and finer bins','source-defined brightness and SNR terciles','usable bands'],target='T2 identity locked before source fitting, spectroscopic labels read only after source gate; oldT1 exploratory separately; T3 omitted',target_subsets={'full':'all11 actual available response bands','HSC':'actual HSC grizy','overlap':'actual HSC g,r,i,z (no filter substitutions)'},counterfactuals={'C4_wrong_metadata':'Fixed measured flux/error, rotate valid filter IDs by one, no claim of physically generated data','C4_consistent':'Source-only synthetic positive control:128 fixed D2 SEDs actual HSC/Legacy/full response integration, independent noise SNR40; not real galaxies or high-z supervision','C4_paired':'Existing held validation S3 actual HSC/Legacy views, galaxy-paired metrics'},physical_baselines='D12 measured-error NNLS primary; D2 constrained diagnostic only, no universal real-galaxy calibration claim; no new score selection',brightness='R1_brightness limited Mixed source-only ablation fixed in advance; primary all amplitude normalized',Qwen='Frozen original HSC checkpoint, same T2 eligible identities in grizy source-format with explicit unavailable EBV; no new training; replay required; label-free inference may precede final label opening',success='Both seeds improve T2 outlier endpoint and their conditional 95% bootstrap intervals exclude0; probability metrics/calibration must be reported regardless. External FSPS information claim only if comparable-capacity and generic/full-response controls do not account for result. No architecture-only attribution to R2. Unsupported z>=2 remains exploratory.',failure='Any identity, unit, response hash or source-learning gate failure stops target label access. A valid compact common-budget comparison is not automatically full convergence or population-wide faint-target representativeness.',forbidden=['Qwen training','Stage2C','target checkpoint selection','target kernel/basis selection','source-test evaluation'])
    cfg['target_eligibility']='Frozen identities and flag4/CL>=97; after model lock retain0<specz<=4.5 and >=3 full available bands; count negative/zero and outside-grid references and insufficient-band failures separately; no target-derived relabeling or clipping'
    cfg['C4_seed_and_domain']='seed1029;128 fixed D2 mixtures; z uniform original centers0.1..1.2; Dirichlet(1,1); independent Gaussian nominalSNR40; amplitude1; HSC/Legacy/full responses; no target labels'
    cfg['dataset_sha256']={p.name:d.sha(p) for p in A.glob('*.npz')};cfg['implementation_sha256']={p.name:d.sha(p) for p in (ROOT/'scripts').glob('stage3*1029.py')}
    d.write(A/'benchmark.json',cfg);d.write(A/'benchmark_lock.json',dict(sha256=d.sha(A/'benchmark.json'),created_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),target_redshift_labels_opened=False));d.write(OUT/'benchmark_reference.json',dict(path=str(A/'benchmark.json'),sha256=d.sha(A/'benchmark.json')))
def train(source,arm,seed):
    folder=OUT/'models'/f'{source}_{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=True);path=folder/'model.pt'
    if path.exists():return
    cfg=json.loads((A/'benchmark.json').read_text())['optim'];a=d.npz(A/f'{source}_eligible.npz');tr=take(a,a['split']=='train');va=take(a,a['split']=='val');t=base.tensor_rows(tr,DEVICE);v=base.tensor_rows(va,DEVICE);y=targets(tr);vy=targets(va)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=Model(arm).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(seed+1029);surveys=np.unique(tr['catalogue']);by=[np.flatnonzero(tr['catalogue']==s) for s in surveys];history=[];best=float('inf');start=time.time();grad=0.;trainloss=None;snapshots={}
    bins=torch.arange(300,device=DEVICE);soft=torch.exp(-.5*((bins[None,:]-y[:,None])/cfg['neighbor_sigma_bins'])**2);soft/=soft.sum(1,keepdim=True)
    for step in range(cfg['updates']+1):
        if step%cfg['evaluate_every']==0:
            ll=logits(model,v);ce_by={s:float(F.cross_entropy(torch.tensor(ll[va['catalogue']==s],device=DEVICE),vy[va['catalogue']==s])) for s in np.unique(va['catalogue'])};ce=float(np.mean(list(ce_by.values())))
            row=dict(update=step,validation_CE=ce,validation_CE_by_survey=ce_by,train_batch_hard_CE=trainloss,gradient_L2=grad,seconds=time.time()-start);history.append(row)
            if ce<best:best=ce;beststep=step;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if step in [2500,5000]:snapshots[step]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(source,arm,seed,step,'valCE',round(ce,4),flush=True)
        if step==cfg['updates']:break
        model.train();idx=np.concatenate([rng.choice(pool,cfg['batch_size']//len(by),replace=True) for pool in by]);rng.shuffle(idx);idx=torch.tensor(idx,device=DEVICE);pred=model(base.slice_t(t,idx));loss=-(soft[idx]*F.log_softmax(pred,1)).sum(1).mean();assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip']));assert np.isfinite(grad);opt.step();trainloss=float(F.cross_entropy(pred.detach(),y[idx]))
    model.load_state_dict(beststate);ll=logits(model,v);pred=pdf.prediction(ll,va['z']);metrics=pdf.metric(pred)
    past=np.mean([r['validation_CE'] for r in history[-7:-4]]);recent=np.mean([r['validation_CE'] for r in history[-3:]]);improvement=float(past-recent)
    plateau=bool(-.1<=improvement<.02);passed=bool(beststep>0 and history[0]['validation_CE']-best>=.5 and metrics['outlier_fraction']<.35 and metrics['nmad']<.12)
    rec=dict(source=source,arm=arm,seed=seed,state=beststate,snapshots=snapshots,history=history,best_update=beststep,parameters=sum(p.numel() for p in model.parameters()),seconds=time.time()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated() if DEVICE=='cuda' else 0,validation_metrics=metrics,source_learning_gate=passed,final_plateau=plateau,final_window_improvement=improvement,validation_galaxies=len(va['z']),training_galaxies=len(tr['z']))
    torch.save(rec,path);d.write(folder/'summary.json',{k:v for k,v in rec.items() if k not in ['state','snapshots']});np.savez_compressed(folder/'validation_predictions.npz',**pred,object_id=va['object_id'],group=va['group']);print('DONE',source,arm,seed,beststep,metrics,flush=True)
def run():
    assert d.sha(A/'benchmark.json')==json.loads((A/'benchmark_lock.json').read_text())['sha256']
    torch.set_num_threads(2)
    for source in ['HSC','Legacy','Mixed']:
        for seed in [11,23]:
            for arm in ['R1','R2']:train(source,arm,seed)
    for seed in [11,23]:
        for arm in ['C1','C2','C3']:train('Mixed',arm,seed)
    for arm in ['R0','R1_brightness']:train('Mixed',arm,11)
    summaries=[json.loads(p.read_text()) for p in sorted((OUT/'models').glob('*/summary.json'))]
    primary=[r for r in summaries if r['arm'] in ['R1','R2']]
    passed=len(primary)==12 and all(r['source_learning_gate'] for r in primary)
    lock=dict(passed=passed,primary_models=len(primary),all_control_models=len(summaries),all_controls_frozen=len(summaries)==20,model_sha256={str(p.relative_to(OUT)):d.sha(p) for p in sorted((OUT/'models').glob('*/model.pt'))},protocol_sha256=d.sha(A/'benchmark.json'),target_labels_opened=False,created_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    d.write(OUT/'source_gate_and_model_lock.json',lock);print('SOURCE GATE',lock['passed'],flush=True)
if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='run':
        from qwen_zero_shot_sanity import reserve_one_gpu
        with reserve_one_gpu() as gpu:
            d.write(OUT/'GPU_reservation.json',gpu);run()
