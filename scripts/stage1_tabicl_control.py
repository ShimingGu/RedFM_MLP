#!/usr/bin/env python3
"""Small existing-API TabICL in-context regressor comparison; no gradient updates."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'notebooks'),str(ROOT)]
import numpy as np
import torch
import stage1_physics_benchmark as b
from aion_magnitude.metrics import point_photoz_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=b.OUT);p.add_argument('--checkpoint-revision');args=p.parse_args();out=args.output_dir;b.check(out)
    from huggingface_hub import HfApi,hf_hub_download
    repository='jingang/TabICL';saved=json.loads((out/'tabicl_protocol.json').read_text()) if (out/'tabicl_protocol.json').exists() else {}
    revision=args.checkpoint_revision or saved.get('revision') or HfApi().model_info(repository).sha;name='tabicl-regressor-v2-20260212.ckpt'
    path=hf_hub_download(repo_id=repository,filename=name,revision=revision,cache_dir='/scratch/.tmp-gsm/stage1-tabicl-cache')
    import qwen_zero_shot_sanity as pilot
    from aion_magnitude.table_models import create_table_regressor
    config=argparse.Namespace(device='cuda',model_path=path,allow_model_download=False,n_estimators=4,table_batch_size=4,offload_mode='auto',seed=b.SEED)
    b.d.write(out/'tabicl_protocol.json',dict(repo=repository,revision=revision,checkpoint=name,checkpoint_path=path,checkpoint_sha256=b.d.sha(path),n_estimators=4,train_rows=512,fit='In-context source X/y; no gradient updates',feature_information='same normalized photometry, response summaries, masks; randomly permuted slots, 99 numerical features',target_labels_used_for_fit=False))
    a=b.d.npz(out/'photometry.npz');train=a['split']=='train';x=a['numeric'].reshape(len(train),-1);records=[json.loads(l) for l in (out/'records.jsonl').read_text().splitlines()]
    with pilot.reserve_one_gpu() as gpu:
        model=create_table_regressor('tabicl',config,n_features=x.shape[1]);model.fit(x[train],a['z'][train]);pred=np.asarray(model.predict(x));(out/'predictions').mkdir(exist_ok=True)
        np.savez_compressed(out/'predictions/tabicl.npz',z_p50=pred,z_spec=a['z'])
        b.d.table(out/'tabicl_performance.csv',[dict(model='tabicl',seed=b.SEED,stratum=name,n=int(mask.sum()),**point_photoz_metrics(torch.from_numpy(pred[mask]),torch.from_numpy(a['z'][mask]))) for name,mask in b.split_masks(records).items() if mask.any()])
        b.d.write(out/'tabicl_status.json',dict(phase='complete',gpu=gpu,gradient_updates=False,scalar_output=True))


if __name__=='__main__':main()
