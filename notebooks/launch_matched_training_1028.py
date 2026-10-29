#!/usr/bin/env python3
"""Explicit future training launcher. This is prepared, not run by task_1028."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import time

from task_1028 import ROOT,sha,write,now

SLICES=['MIG-1bba1f26-8b2b-5b97-9bb1-c886ab9bb8b7','MIG-0a2e21a6-8961-56b1-9eb4-997ae914386e','MIG-6c85a6c1-35aa-5ff1-ae36-a468e3e230b6']


def main(out,seeds):
    manifest=json.loads((out/'manifest.json').read_text());queue=[]
    for method in ['dora','qlora']:
        for seed in seeds:
            for domain in ['hsc','legacy']:
                name=f'{domain}-{method}-seed{seed}';path=out/'training_configs'/f'{name}.json'
                assert sha(path)==manifest['final_config_sha256'][str(path.resolve())]
                cfg=json.loads(path.read_text())
                if not (Path(cfg['output_dir'])/'complete.json').exists():queue.append((name,path,cfg))
    logs=out/'training_launch_logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(PIXI_CACHE_DIR='/scratch/.tmp-gsm/pixi-cache',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    with (out/'training-queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);active={};failed={};finished=[]
        while queue or active:
            for gpu in SLICES:
                if gpu in active:
                    name,process,handle,cfg=active[gpu];code=process.poll()
                    if code is not None:
                        handle.close();del active[gpu]
                        if code==0 and (Path(cfg['output_dir'])/'complete.json').exists():finished.append(name)
                        else:failed[name]=dict(exit_code=code,log=str(logs/f'{name}.log'))
                if gpu not in active and queue and not failed:
                    name,path,cfg=queue.pop(0);handle=(logs/f'{name}.log').open('a');child_env=env.copy();child_env['CUDA_VISIBLE_DEVICES']=gpu
                    command=['pixi','run','--locked','python',str(ROOT/'notebooks/task_1028.py'),'train','--output-dir',str(out),'--config',str(path)]
                    if Path(cfg['output_dir']).exists():command.append('--resume')
                    process=subprocess.Popen(command,cwd=ROOT,env=child_env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
                    active[gpu]=(name,process,handle,cfg);write(logs/f'{name}_launch.json',dict(created_utc=now(),command=command,gpu=gpu,pid=process.pid))
            write(out/'training_queue_status.json',dict(updated_utc=now(),seeds=seeds,pending=[n for n,p,c in queue],
                active={g:dict(job=n,pid=p.pid) for g,(n,p,h,c) in active.items()},finished=finished,failed=failed))
            if failed and not active:raise RuntimeError(f'Preserved failed training checkpoints: {failed}')
            if queue or active:time.sleep(15)
    write(out/'training_queue_complete.json',dict(finished_utc=now(),seeds=seeds,full_P_inference_launched=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',choices=[42,43,44],default=[42]);args=p.parse_args()
    main(args.package.resolve(),args.seeds)
