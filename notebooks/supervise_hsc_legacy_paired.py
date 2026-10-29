#!/usr/bin/env python3
"""Durable three-slice inference queue and automatic report; never trains."""
import argparse
from collections import deque
import os
from pathlib import Path
import subprocess
import time

from hsc_legacy_paired_benchmark import ROOT,MODELS,sha,write,now
import json

SLICES=['MIG-1bba1f26-8b2b-5b97-9bb1-c886ab9bb8b7',
        'MIG-0a2e21a6-8961-56b1-9eb4-997ae914386e',
        'MIG-6c85a6c1-35aa-5ff1-ae36-a468e3e230b6']


def main(out):
    manifest=json.loads((out/'manifest.json').read_text())
    queue=deque(n for n in ['original-dora','B','C','original-qlora','A'] if not (out/n/'complete.json').exists())
    active={};finished={};failed={};logs=out/'execution_logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PIXI_CACHE_DIR='/scratch/.tmp-gsm/pixi-cache')
    prefix=['pixi','run','--locked','python',str(ROOT/'notebooks/hsc_legacy_paired_benchmark.py')]
    while queue or active:
        for gpu in SLICES:
            if gpu in active:
                model,process,handle=active[gpu]
                status=process.poll()
                if status is not None:
                    handle.close();del active[gpu]
                    if status==0 and (out/model/'complete.json').exists():finished[model]=dict(exit_code=status,finished_utc=now())
                    else:failed[model]=dict(exit_code=status,log=str(logs/f'{model}.log'))
            if gpu not in active and queue and not failed:
                for file,expected in manifest['code_sha256'].items():
                    assert sha(ROOT/file)==expected, f'Prepared scientific code changed: {file}; preserve completed chunks and review before resuming'
                model=queue.popleft();handle=(logs/f'{model}.log').open('a')
                command=prefix+['run','--output-dir',str(out),'--model',model]
                worker_env=env.copy();worker_env['CUDA_VISIBLE_DEVICES']=gpu
                process=subprocess.Popen(command,cwd=ROOT,env=worker_env,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                active[gpu]=(model,process,handle)
                write(out/f'{model}_launch.json',dict(created_utc=now(),pid=process.pid,gpu=gpu,command=command,environment={k:worker_env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','TOKENIZERS_PARALLELISM','PIXI_CACHE_DIR']}))
        write(out/'supervisor_status.json',dict(updated_utc=now(),pid=os.getpid(),phase='inference',
            pending=list(queue),active={g:dict(model=m,pid=p.pid) for g,(m,p,h) in active.items()},finished=finished,failed=failed))
        if failed and not active:raise RuntimeError(f'Inference failed; inspect preserved logs: {failed}')
        if queue or active:time.sleep(15)
    assert all((out/m/'complete.json').exists() for m in MODELS)
    write(out/'supervisor_status.json',dict(updated_utc=now(),pid=os.getpid(),phase='analysis',finished=finished))
    with (logs/'report.log').open('a') as handle:
        subprocess.run(prefix+['report','--output-dir',str(out)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
    write(out/'supervisor_status.json',dict(updated_utc=now(),pid=os.getpid(),phase='complete',finished=finished))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    main(parser.parse_args().output_dir.resolve())
