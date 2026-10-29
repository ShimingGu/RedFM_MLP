#!/usr/bin/env python3
"""Persistent smoke -> exact full replay -> full counterfactual -> report queue."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import time

from filter_metadata_counterfactual import ROOT,write,validate_run,now,check_protected

SLICES=['MIG-1bba1f26-8b2b-5b97-9bb1-c886ab9bb8b7',
        'MIG-0a2e21a6-8961-56b1-9eb4-997ae914386e',
        'MIG-6c85a6c1-35aa-5ff1-ae36-a468e3e230b6']


def stage_queue(out,stage,jobs,env):
    pending=list(jobs);active={};completed={};failed={};logs=out/'execution_logs';logs.mkdir(exist_ok=True)
    prefix=['pixi','run','--locked','python',str(ROOT/'notebooks/filter_metadata_counterfactual.py')]
    def done(name):
        if stage=='smoke':return (out/'smoke'/f'{name}.json').exists()
        return (out/'jobs'/name/'complete.json').exists()
    pending=[(n,a) for n,a in pending if not done(n)]
    while pending or active:
        for gpu in SLICES:
            if gpu in active:
                name,process,handle=active[gpu];status=process.poll()
                if status is not None:
                    handle.close();del active[gpu]
                    if status==0 and done(name):completed[name]=dict(finished_utc=now(),exit_code=0)
                    else:failed[name]=dict(exit_code=status,log=str(logs/f'{stage}-{name}.log'))
            if gpu not in active and pending and not failed:
                validate_run(out)
                name,args=pending.pop(0);handle=(logs/f'{stage}-{name}.log').open('a')
                worker_env=env.copy();worker_env['CUDA_VISIBLE_DEVICES']=gpu
                command=prefix+args+['--output-dir',str(out)]
                process=subprocess.Popen(command,cwd=ROOT,env=worker_env,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                active[gpu]=(name,process,handle)
                write(out/'launches'/f'{stage}-{name}.json',dict(created_utc=now(),pid=process.pid,gpu=gpu,command=command))
        write(out/'supervisor_status.json',dict(updated_utc=now(),pid=os.getpid(),phase=stage,
            pending=[n for n,a in pending],active={g:dict(job=n,pid=p.pid) for g,(n,p,h) in active.items()},completed=completed,failed=failed))
        if failed and not active:raise RuntimeError(f'Queue stopped without dropping failed conditions: {failed}')
        if pending or active:time.sleep(10)
    for name,args in jobs:assert done(name)


def main(out):
    with (out/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m,_=validate_run(out);check_protected(m,full=True)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PIXI_CACHE_DIR='/scratch/.tmp-gsm/pixi-cache')
        stage_queue(out,'smoke',[(model,['smoke','--model',model]) for model in ['B','C']],env)
        stage_queue(out,'baseline',[
            ('B-both',['infer','--model','B','--condition','B','--domain','both']),
            ('correct-hsc',['infer','--model','C','--condition','correct','--domain','hsc']),
            ('correct-legacy',['infer','--model','C','--condition','correct','--domain','legacy'])],env)
        proofs={}
        for name in ['B-both','correct-hsc','correct-legacy']:
            proof=json.loads((out/'jobs'/name/'complete.json').read_text());assert proof['baseline_replay_exact']
            proofs[name]=proof
        gate=out/'gates/full_baseline_replay.json'
        if not gate.exists():write(gate,dict(created_utc=now(),full_P_n=61696,all_B_C_HSC_Legacy_logits_and_pdfs_exact=True,
            archived_metrics_reproduced=True,jobs=proofs,policy_sha256=m['policy_sha256']))
        names=json.loads((out/'evaluation_policy_v1.json').read_text())['conditions']
        jobs=[(f'{name}-{domain}',['infer','--model','C','--condition',name,'--domain',domain])
              for name in names if name!='correct' for domain in ['hsc','legacy']]
        stage_queue(out,'counterfactual',jobs,env)
        write(out/'supervisor_status.json',dict(updated_utc=now(),pid=os.getpid(),phase='analysis',failed={}))
        with (out/'execution_logs/report.log').open('a') as log:
            subprocess.run(['pixi','run','--locked','python',str(ROOT/'notebooks/filter_metadata_counterfactual.py'),
                            'report','--output-dir',str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(out/'supervisor_status.json',dict(updated_utc=now(),pid=os.getpid(),phase='complete',failed={}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();main(args.output_dir.resolve())
