"""Isolated H1 fixed-budget loop derived from the validated seed-42 loop.

The loop below preserves its optimisation operations, adding diagnostics, saved
orders, finite guards, and source-only prediction exports. No P-data API exists.
"""
from __future__ import annotations
from contextlib import nullcontext
import json
import hashlib
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .matched_training import (accumulation_window_size,build_fresh_model,state_fingerprint,rng_state,restore_rng,save_torch,export_model)
from .qwen_posttraining import QwenPosttrainingConfig,TextRedshiftDataset,make_text_collator,evaluate_text_photoz_model,_linear_warmup_decay_lambda
from .metrics import redshift_cross_entropy_loss,summarize_pdf_metrics
from .utils import make_redshift_grid,set_random_seed

def train_loop(model,tokenizer,device,train_texts,train_z,train_ids,val_texts,val_z,val_ids,config,output_dir,*,resume=False,export=export_model,checkpoint_hook=None, saved_orders=None):
    """Train/validate only. Evaluation-cohort data are never passed to this loop."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    c=QwenPosttrainingConfig(**config['training']).normalized()
    assert c.batch_size==1,'Exact partial-window normalization currently supports microbatch=1'
    n=len(train_z);updates_per_epoch=(n+c.gradient_accumulation_steps-1)//c.gradient_accumulation_steps
    budget=updates_per_epoch*c.epochs;warmup=int(budget*c.warmup_fraction);adapter_start=updates_per_epoch*c.head_warmup_epochs
    adapter_budget=budget-adapter_start;adapter_warmup=int(adapter_budget*c.warmup_fraction)
    head_params=list(model.photoz_head.parameters());adapter_params=[p for p in model.qwen.parameters() if p.requires_grad]
    names={k for k,p in model.named_parameters() if p.requires_grad}
    assert adapter_params and head_params
    state=lambda:{k:p.detach().cpu().clone() for k,p in model.named_parameters() if k in names}
    def load_state(s):
        assert set(s)==names
        with torch.no_grad():
            for k,p in model.named_parameters():
                if k in names:p.copy_(s[k].to(p.device))
    optimizer=torch.optim.AdamW([dict(params=head_params,lr=c.learning_rate,weight_decay=c.weight_decay),
                                dict(params=adapter_params,lr=c.lora_learning_rate,weight_decay=0.)])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,[
        lambda step:_linear_warmup_decay_lambda(step,warmup_steps=warmup,total_steps=budget),
        lambda step:0. if step<adapter_start else _linear_warmup_decay_lambda(step-adapter_start,warmup_steps=adapter_warmup,total_steps=adapter_budget)])
    assert saved_orders is not None
    orders=saved_orders
    assert orders.shape==(c.epochs,n)
    for order in orders:np.testing.assert_array_equal(np.sort(order),np.arange(n))
    collate=make_text_collator(tokenizer,max_length=c.max_length)
    val_loader=DataLoader(TextRedshiftDataset(val_texts,val_z,[str(x) for x in val_ids]),batch_size=c.eval_batch_size,shuffle=False,collate_fn=collate)
    edges,centers=make_redshift_grid(c.z_min,c.z_max,c.n_z_bins)
    epoch0=batch0=update=examples=0;history=[];best=float('inf');best_state=None;best_update=None
    if resume:
        files=list((out/'checkpoints').glob('resume-slot*.pt'))
        if not files:raise FileNotFoundError('No matched-run checkpoint to resume')
        checkpoints=[]
        for f in files:
            proof=f.with_suffix('.sha256')
            if proof.exists() and hashlib.sha256(f.read_bytes()).hexdigest()==proof.read_text().strip():
                value=torch.load(f,map_location='cpu',weights_only=False)
                assert value['config']==config and value['dataset_sizes']==(n,len(val_z))
                checkpoints.append(value)
        if not checkpoints:raise RuntimeError('No own checkpoint with a matching SHA256; refuse unverified resume')
        saved=max(checkpoints,key=lambda s:(s['global_update'],s['next_epoch'],s['next_batch']))
        assert saved['config']==config and saved['dataset_sizes']==(n,len(val_z))
        load_state(saved['trainable_state']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        epoch0=saved['next_epoch'];batch0=saved['next_batch'];update=saved['global_update'];examples=saved['examples_seen']
        history=saved['history'];best=saved['best_ce'];best_state=saved['best_state'];best_update=saved['best_update'];restore_rng(saved['rng'])
    else:set_random_seed(config['seed']+30000)
    initial=torch.load(config['initial_trainable_state'],map_location='cpu',weights_only=True)
    assert set(initial)==names
    # Diagnostics are checkpointed with the optimizer, so resume retains epoch loss totals.
    diagnostics=(saved.get('diagnostics') if resume else None) or dict(epochs={},first_backward={},process_intervals=[])
    process_started=time.time();process_update_start=update
    def changes():
        result={}
        current=state()
        for label in ('head','adapter'):
            keys=[k for k in current if ('photoz_head' in k)==(label=='head')]
            sq=sum(float((current[k].float()-initial[k].float()).square().sum()) for k in keys)
            base=sum(float(initial[k].float().square().sum()) for k in keys)
            result[label]=dict(l2_change=math.sqrt(sq),relative_l2_change=math.sqrt(sq/max(base,1e-30)),parameter_l2_norm=math.sqrt(sum(float(current[k].float().square().sum()) for k in keys)))
        return result
    def source_prediction_stats(ev):
        p=ev['pz'].float();med=ev['z_p50'].float()
        return dict(z_median_std=float(med.std()),mean_entropy=float(-(p*p.clamp_min(1e-30).log()).sum(1).mean()),
                    uniform_mass_ce=math.log(c.n_z_bins),mean_mass_above_cutoff=float((p*torch.clamp((edges[1:]-torch.maximum(edges[:-1],torch.tensor(.6)))/torch.diff(edges),0,1)).sum(1).mean()))
    def source_prediction_export(folder,split,texts,z,ids):
        loader=DataLoader(TextRedshiftDataset(texts,z,[str(x) for x in ids]),batch_size=c.eval_batch_size,shuffle=False,collate_fn=collate)
        value=evaluate_text_photoz_model(model,loader,device=device,redshift_edges=edges,redshift_centers=centers)
        save_torch(folder/f'{split}_predictions.pt',value)
        return value
    interval=config['checkpoint_interval_updates'];optimizer.zero_grad(set_to_none=True)
    def checkpoint(next_epoch,next_batch):
        value=dict(format_version='matched_training_resume_v1',config=config,dataset_sizes=(n,len(val_z)),
            next_epoch=next_epoch,next_batch=next_batch,global_update=update,examples_seen=examples,
            history=history,best_ce=best,best_state=best_state,best_update=best_update,trainable_state=state(),
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),diagnostics=diagnostics)
        path=out/'checkpoints'/f'resume-slot{(update//interval)%2}.pt'
        save_torch(path,value)
        proof=path.with_suffix('.sha256');tmp=proof.with_suffix('.partial')
        tmp.write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n');tmp.replace(proof)
        if checkpoint_hook is not None:checkpoint_hook(value)
    for epoch in range(epoch0,c.epochs):
        epoch_started=time.time()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        warm=epoch<c.head_warmup_epochs;model.train();model.freeze_qwen_representation=warm
        for p in adapter_params:p.requires_grad_(not warm)
        if warm:model.qwen.eval()
        phase='head_warmup' if warm else 'joint'
        diag=diagnostics['epochs'].setdefault(str(epoch),dict(loss_sum=0.,examples=0,updates=0,head_grad_sum=0.,head_grad_max=0.,adapter_grad_sum=0.,adapter_grad_max=0.,head_zero_grad_updates=0,adapter_zero_grad_updates=0,head_output_grad_max=0.))
        phase_first=phase not in diagnostics['first_backward']
        if phase_first and device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        order=orders[epoch]
        dataset=TextRedshiftDataset([train_texts[i] for i in order],np.asarray(train_z)[order],[str(train_ids[i]) for i in order])
        loader=DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=collate,generator=torch.Generator().manual_seed(config['seed']+30000+epoch))
        for batch_index,batch in enumerate(loader):
            if epoch==epoch0 and batch_index<batch0:continue
            z=batch.pop('z_spec').to(device);batch.pop('object_id')
            inputs={k:v.to(device) for k,v in batch.items()}
            context=torch.autocast(device_type='cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
            started_step=time.time()
            with context:logits=model(**inputs);loss=redshift_cross_entropy_loss(logits.float(),z,edges=edges.to(device))
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite source-training loss at epoch {epoch}, batch {batch_index}')
            denominator=accumulation_window_size(n,batch_index,c.gradient_accumulation_steps)
            (loss/denominator).backward();examples+=1
            diag['loss_sum']+=float(loss.detach());diag['examples']+=1
            if phase_first:
                if device.type=='cuda':torch.cuda.synchronize(device)
                diagnostics['first_backward'][phase]=dict(epoch=epoch,batch_index=batch_index,global_update_before=update,
                    seconds_forward_backward=time.time()-started_step,
                    allocated_gib=torch.cuda.memory_allocated(device)/2**30 if device.type=='cuda' else 0.,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0.,
                    peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else 0.)
                (out/'memory_throughput.json').write_text(json.dumps(diagnostics['first_backward'],indent=2)+'\n');phase_first=False
            if (batch_index+1)%c.gradient_accumulation_steps==0 or batch_index+1==n:
                output_grads=[p.grad.detach().float().norm() for p in model.photoz_head.net[6].parameters() if p.grad is not None]
                diag['head_output_grad_max']=max(diag['head_output_grad_max'],sum(float(v) for v in output_grads))
                hg=float(torch.nn.utils.clip_grad_norm_(head_params,c.max_grad_norm,error_if_nonfinite=True))
                ag=float(torch.nn.utils.clip_grad_norm_(adapter_params,c.lora_max_grad_norm,error_if_nonfinite=True)) if not warm else 0.
                diag['head_grad_sum']+=hg;diag['head_grad_max']=max(diag['head_grad_max'],hg);diag['head_zero_grad_updates']+=int(hg==0)
                diag['adapter_grad_sum']+=ag;diag['adapter_grad_max']=max(diag['adapter_grad_max'],ag);diag['adapter_zero_grad_updates']+=int(ag==0);diag['updates']+=1
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);update+=1
                if update==1 or update==adapter_start+1 or update%interval==0:
                    checkpoint(epoch,batch_index+1)
                    progress=dict(updated_utc=time.time(),epoch=epoch,phase=phase,global_update=update,examples_seen=examples,
                        best_source_val_ce=best if math.isfinite(best) else None,head_grad_norm=hg,adapter_grad_norm=ag,
                        lr=[g['lr'] for g in optimizer.param_groups],seconds_per_update_this_process=(time.time()-process_started)/(update-process_update_start),
                        new_model_P_evaluated=False)
                    (out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                if update==1 or update%100==0:print('MATCHED UPDATE',update,'/',budget,'examples',examples,flush=True)
        if hasattr(model.photoz_head,'reset_source_diagnostics'):model.photoz_head.reset_source_diagnostics()
        evaluation=evaluate_text_photoz_model(model,val_loader,device=device,redshift_edges=edges,redshift_centers=centers)
        scores=summarize_pdf_metrics(evaluation)
        scale_diagnostics=model.photoz_head.source_diagnostics() if config.get('head')=='mixture' else dict(architecture='300-bin softmax',scale_parameter=None)
        assert all(math.isfinite(float(v)) for v in scores.values())
        history.append(dict(epoch=epoch,global_update=update,examples_seen=examples,phase=phase,val=scores,
            training_loss=diag['loss_sum']/diag['examples'],gradient_diagnostics=diag.copy(),parameter_changes=changes(),
            epoch_wall_seconds_this_process=time.time()-epoch_started,gpu_memory=dict(allocated_gib=torch.cuda.memory_allocated(device)/2**30 if device.type=='cuda' else 0.,reserved_gib=torch.cuda.memory_reserved(device)/2**30 if device.type=='cuda' else 0.,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0.,peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else 0.),
            prediction_diagnostics={**source_prediction_stats(evaluation),**scale_diagnostics},lr=[g['lr'] for g in optimizer.param_groups]))
        if scores['cross_entropy']<best:
            best=scores['cross_entropy'];best_state=state();best_update=update
            export(model,out/'selected',dict(selected_on='source-validation CE only',global_update=update,epoch=epoch,val=scores,config=config))
            save_torch(out/'selected/validation_predictions.pt',evaluation)
            save_torch(out/'selected/optimizer_scheduler_rng.pt',dict(config=config,global_update=update,examples_seen=examples,
                next_epoch=epoch+1,next_batch=0,trainable_state=best_state,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),history=history))
        checkpoint(epoch+1,0);batch0=0
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    assert update==budget and examples==n*c.epochs
    if epoch0==c.epochs:
        evaluation=evaluate_text_photoz_model(model,val_loader,device=device,redshift_edges=edges,redshift_centers=centers)
    for p in adapter_params:p.requires_grad_(True)
    export(model,out/'final',dict(global_update=update,examples_seen=examples,fixed_budget=True,config=config))
    save_torch(out/'final/validation_predictions.pt',evaluation)
    save_torch(out/'final/trainable_state.pt',state())
    save_torch(out/'final/optimizer_scheduler_rng.pt',dict(config=config,global_update=update,examples_seen=examples,
        next_epoch=c.epochs,next_batch=0,trainable_state=state(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),history=history))
    # Source-only exports occur after the entire fixed optimization budget, leaving training RNG untouched.
    source_prediction_export(out/'final','training',train_texts,train_z,train_ids)
    assert best_state is not None;load_state(best_state)
    source_prediction_export(out/'selected','training',train_texts,train_z,train_ids)
    save_torch(out/'selected/trainable_state.pt',best_state)
    result=dict(primary_checkpoint='fixed_final_epoch10',secondary_checkpoint='minimum_lowz_validation_CE',training_complete=True,global_update=update,examples_seen=examples,best_update=best_update,best_source_val_ce=best,
                history=history,test_evaluated=False,training_seed=config['seed'],final_budget_state_saved_before_best_restore=True)
    diagnostics['process_intervals'].append(dict(seconds=time.time()-process_started,updates=update-process_update_start))
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    # Frozen source-validation replay of both exports, without any P access.
    replay_proofs={}
    del optimizer,scheduler
    if device.type=='cuda':torch.cuda.empty_cache()
    for label in ('selected','final'):
        # Use current architecture for replay by restoring exact exported trainable tensors.
        restored=torch.load(out/label/'trainable_state.pt',map_location='cpu',weights_only=True);load_state(restored)
        if export is export_model:
            from peft import get_peft_model_state_dict
            from safetensors.torch import load_file
            saved_adapter=load_file(str(out/label/'adapter/adapter_model.safetensors'))
            current_adapter=get_peft_model_state_dict(model.qwen)
            assert set(saved_adapter)==set(current_adapter)
            for key,value in current_adapter.items():torch.testing.assert_close(value.cpu(),saved_adapter[key],rtol=0,atol=0,check_dtype=False)
            saved_head=torch.load(out/label/'photoz_head.pt',map_location='cpu',weights_only=True)
            for key,value in model.photoz_head.state_dict().items():torch.testing.assert_close(value.cpu(),saved_head[key],rtol=0,atol=0)
        model.eval()
        ref=torch.load(out/label/'validation_predictions.pt',map_location='cpu',weights_only=False)
        replay_loader=DataLoader(TextRedshiftDataset(val_texts[:64],val_z[:64],[str(x) for x in val_ids[:64]]),batch_size=c.eval_batch_size,shuffle=False,collate_fn=collate)
        ev=evaluate_text_photoz_model(model,replay_loader,device=device,redshift_edges=edges,redshift_centers=centers)
        differences={k:float((ev[k]-ref[k][:64]).abs().max()) for k in ('logits','pz','z_p16','z_p50','z_p84')}
        assert all(v==0 for v in differences.values()),differences
        replay_proofs[label]=dict(n=64,exact=True,max_abs_difference=differences)
    (out/'source_validation_replay.json').write_text(json.dumps(replay_proofs,indent=2)+'\n')
    flags=[]
    for row in history:
        if row['val']['cross_entropy']>=math.log(c.n_z_bins)-.1:flags.append(f"epoch {row['epoch']}: validation CE near/worse than uniform mass baseline")
        if row['prediction_diagnostics']['z_median_std']<.01:flags.append(f"epoch {row['epoch']}: nearly constant predicted median")
        if row['phase']=='joint' and row['gradient_diagnostics']['adapter_zero_grad_updates']==row['gradient_diagnostics']['updates']:flags.append(f"epoch {row['epoch']}: inactive adapter")
    if len(history)>c.head_warmup_epochs and history[c.head_warmup_epochs]['val']['cross_entropy']>history[c.head_warmup_epochs-1]['val']['cross_entropy']+.5:flags.append('Major deterioration at head/joint transition')
    for row in history:
        g=row['gradient_diagnostics']
        if g['head_zero_grad_updates']==g['updates']:flags.append(f"epoch {row['epoch']}: inactive head")
        if g['head_grad_max']>1e4 or g['adapter_grad_max']>1e3:flags.append(f"epoch {row['epoch']}: exceptionally large pre-clipping gradient")
    if history[-1]['training_loss']>=history[0]['training_loss']-.05:flags.append('Training loss failed to improve substantially')
    if best_update!=min(history,key=lambda x:x['val']['cross_entropy'])['global_update']:flags.append('Checkpoint selection anomaly')
    convergence=dict(flags=flags,baseline_uniform_ce=math.log(c.n_z_bins),best_source_val_ce=best,
        final_source_val_ce=history[-1]['val']['cross_entropy'],final_minus_best_ce=history[-1]['val']['cross_entropy']-best,
        assessment='Source-only diagnostic thresholds; no automatic protocol rescue or P inference',history=history)
    (out/'convergence.json').write_text(json.dumps(convergence,indent=2)+'\n')
    checks=dict(passed=True,updates=update,presentations=examples,epochs=len(history),source_validation_replay=replay_proofs,
        source_training_and_validation_predictions_saved=True,selected_and_final_exports=True,optimizer_scheduler_rng_retained=True,
        source_initialization_verified=True,P_opened=False,P_scored=False,seeds=[config['seed']])
    (out/'completion_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


from .bidirectional_1029_training import install_P_guard
