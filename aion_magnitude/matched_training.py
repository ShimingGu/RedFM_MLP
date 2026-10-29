"""Fresh-base matched HSC/Legacy training, with explicit common initialization."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .models import PhotoZHead
from .qwen_posttraining import QwenPosttrainingConfig,create_qlora_photoz_model,TextRedshiftDataset,make_text_collator,evaluate_text_photoz_model,_linear_warmup_decay_lambda
from .metrics import redshift_cross_entropy_loss,summarize_pdf_metrics
from .utils import make_redshift_grid,set_random_seed


def state_fingerprint(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def initial_head(hidden_size,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=PhotoZHead(hidden_size,300,256)
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def accumulation_window_size(total,batch_index,accumulation):
    start=(batch_index//accumulation)*accumulation
    return min(accumulation,total-start)


def train_order(n,seed,epochs):
    return np.stack([np.random.default_rng(seed+20000+epoch).permutation(n) for epoch in range(epochs)]).astype(np.int32)


def build_fresh_model(config):
    """Source identity never enters parameter construction; no adapter restore API."""
    if config.get('initial_adapter') is not None or config.get('source_adapter') is not None:
        raise ValueError('Matched branches must start from the pre-adaptation base')
    base=Path(config['training']['model_path'])
    if (base/'adapter_config.json').exists():raise ValueError('A PEFT directory is not the pre-adaptation base')
    if not (base/'model.safetensors.index.json').exists():raise ValueError('Expected verified sharded Qwen base')
    settings=config['training'].copy();settings['seed']=config['seed']+10000
    c=QwenPosttrainingConfig(**settings).normalized();set_random_seed(c.seed)
    model,tokenizer,device=create_qlora_photoz_model(c,use_dora=config['method']=='dora')
    head=torch.load(config['initial_head'],map_location='cpu',weights_only=True)
    assert state_fingerprint(head)==config['initial_head_tensor_sha256']
    model.photoz_head.load_state_dict(head,strict=True)
    for k,v in model.photoz_head.state_dict().items():torch.testing.assert_close(v.cpu(),head[k],rtol=0,atol=0)
    return model,tokenizer,device


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu())
    if s['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])


def save_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
    torch.save(value,tmp);tmp.replace(path)


def export_model(model,folder,metadata):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    model.qwen.save_pretrained(folder/'adapter')
    save_torch(folder/'photoz_head.pt',{k:v.detach().cpu() for k,v in model.photoz_head.state_dict().items()})
    metadata=dict(metadata)
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(2**20),b''):h.update(block)
        return h.hexdigest()
    metadata['checkpoint_sha256']={str(p.relative_to(folder)):digest(p) for p in [folder/'photoz_head.pt',*sorted((folder/'adapter').glob('*'))] if p.is_file()}
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


def train_loop(model,tokenizer,device,train_texts,train_z,train_ids,val_texts,val_z,val_ids,config,output_dir,*,resume=False,export=export_model,checkpoint_hook=None):
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
    orders=train_order(n,config['seed'],c.epochs)
    collate=make_text_collator(tokenizer,max_length=c.max_length)
    val_loader=DataLoader(TextRedshiftDataset(val_texts,val_z,[str(x) for x in val_ids]),batch_size=c.eval_batch_size,shuffle=False,collate_fn=collate)
    edges,centers=make_redshift_grid(c.z_min,c.z_max,c.n_z_bins)
    epoch0=batch0=update=examples=0;history=[];best=float('inf');best_state=None;best_update=None
    if resume:
        files=list((out/'checkpoints').glob('resume-slot*.pt'))
        if not files:raise FileNotFoundError('No matched-run checkpoint to resume')
        checkpoints=[torch.load(f,map_location='cpu',weights_only=False) for f in files]
        saved=max(checkpoints,key=lambda s:(s['global_update'],s['next_epoch'],s['next_batch']))
        assert saved['config']==config and saved['dataset_sizes']==(n,len(val_z))
        load_state(saved['trainable_state']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        epoch0=saved['next_epoch'];batch0=saved['next_batch'];update=saved['global_update'];examples=saved['examples_seen']
        history=saved['history'];best=saved['best_ce'];best_state=saved['best_state'];best_update=saved['best_update'];restore_rng(saved['rng'])
    else:set_random_seed(config['seed']+30000)
    interval=config['checkpoint_interval_updates'];optimizer.zero_grad(set_to_none=True)
    def checkpoint(next_epoch,next_batch):
        value=dict(format_version='matched_training_resume_v1',config=config,dataset_sizes=(n,len(val_z)),
            next_epoch=next_epoch,next_batch=next_batch,global_update=update,examples_seen=examples,
            history=history,best_ce=best,best_state=best_state,best_update=best_update,trainable_state=state(),
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state())
        save_torch(out/'checkpoints'/f'resume-slot{(update//interval)%2}.pt',value)
        if checkpoint_hook is not None:checkpoint_hook(value)
    for epoch in range(epoch0,c.epochs):
        warm=epoch<c.head_warmup_epochs;model.train();model.freeze_qwen_representation=warm
        for p in adapter_params:p.requires_grad_(not warm)
        if warm:model.qwen.eval()
        order=orders[epoch]
        dataset=TextRedshiftDataset([train_texts[i] for i in order],np.asarray(train_z)[order],[str(train_ids[i]) for i in order])
        loader=DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=collate,generator=torch.Generator().manual_seed(config['seed']+30000+epoch))
        for batch_index,batch in enumerate(loader):
            if epoch==epoch0 and batch_index<batch0:continue
            z=batch.pop('z_spec').to(device);batch.pop('object_id')
            inputs={k:v.to(device) for k,v in batch.items()}
            context=torch.autocast(device_type='cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
            with context:logits=model(**inputs);loss=redshift_cross_entropy_loss(logits.float(),z,edges=edges.to(device))
            denominator=accumulation_window_size(n,batch_index,c.gradient_accumulation_steps)
            (loss/denominator).backward();examples+=1
            if (batch_index+1)%c.gradient_accumulation_steps==0 or batch_index+1==n:
                torch.nn.utils.clip_grad_norm_(head_params,c.max_grad_norm)
                if not warm:torch.nn.utils.clip_grad_norm_(adapter_params,c.lora_max_grad_norm)
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);update+=1
                if update%interval==0:checkpoint(epoch,batch_index+1)
                if update==1 or update%100==0:print('MATCHED UPDATE',update,'/',budget,'examples',examples,flush=True)
        evaluation=evaluate_text_photoz_model(model,val_loader,device=device,redshift_edges=edges,redshift_centers=centers)
        scores=summarize_pdf_metrics(evaluation);history.append(dict(epoch=epoch,global_update=update,examples_seen=examples,phase='head_warmup' if warm else 'joint',val=scores))
        if scores['cross_entropy']<best:
            best=scores['cross_entropy'];best_state=state();best_update=update
            export(model,out/'selected',dict(selected_on='source-validation CE only',global_update=update,epoch=epoch,val=scores,config=config))
            save_torch(out/'selected/validation_predictions.pt',evaluation)
        checkpoint(epoch+1,0);batch0=0
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    assert update==budget and examples==n*c.epochs
    for p in adapter_params:p.requires_grad_(True)
    export(model,out/'final',dict(global_update=update,examples_seen=examples,fixed_budget=True,config=config))
    save_torch(out/'final/validation_predictions.pt',evaluation)
    save_torch(out/'final/trainable_state.pt',state())
    assert best_state is not None;load_state(best_state)
    save_torch(out/'selected/trainable_state.pt',best_state)
    result=dict(training_complete=True,global_update=update,examples_seen=examples,best_update=best_update,best_source_val_ce=best,
                history=history,test_evaluated=False,training_seed=config['seed'],final_budget_state_saved_before_best_restore=True)
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
