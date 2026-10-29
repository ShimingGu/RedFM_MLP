#!/usr/bin/env python3
"""Source validation intervention: distinguish missing EBV/token effects.

Uses the first 256 archived source validation objects, with no checkpoint or
hyperparameter selection. Only the original frozen model is evaluated.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'notebooks'),str(ROOT)]
import numpy as np
import torch
import diagnose_hsc_clauds_stage0 as d
import qwen_clauds_common_flux_evaluation as runner
import qwen_zero_shot_sanity as pilot
from aion_magnitude.filter_photometry import legacy_config,audit_tokens
from aion_magnitude.FM_Qwen import serialize_qwen_feature_row
from aion_magnitude.qwen_continued_sft import verify_replay


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT/'reports/stage0/1027a');args=p.parse_args()
    out=args.output_dir;m=d.check(out);folder=out/'source-ebv-control'
    if (folder/'comparison.csv').exists():raise FileExistsError(folder)
    folder.mkdir(exist_ok=True);(folder/'inputs').mkdir(exist_ok=True)
    a=d.npz(d.BASE/'level0/source_cohort.npz');idx=np.flatnonzero(a['split']=='val')[:256]
    data={k:v[idx] for k,v in a.items()};assert np.all(data['split']=='val')
    names=json.loads((d.SOURCE/'prepared.json').read_text())['feature_names'];spec=legacy_config('galaxiesml-spec')
    texts={k:[] for k in ('source_native','source_missing_ebv','source_omit_ebv','source_target_schema')}
    for row in data['features']:
        features=dict(zip(names,row));texts['source_native'].append(serialize_qwen_feature_row(features,serialization=spec))
        features['MW_EBV']=np.nan;texts['source_missing_ebv'].append(serialize_qwen_feature_row(features,serialization=spec))
        texts['source_omit_ebv'].append(serialize_qwen_feature_row({k:v for k,v in features.items() if k!='MW_EBV'},serialization=spec))
        texts['source_target_schema'].append(serialize_qwen_feature_row(features,serialization=replace(spec,schema_name='clauds_native_photometry_specz_v1')))
    for name,strings in texts.items():
        with (folder/'inputs'/f'{name}.jsonl').open('w') as f:
            for ident,t in zip(data['targetid'],strings):f.write(json.dumps(dict(object_id=str(ident),text=t))+'\n')
    d.write(folder/'protocol.json',dict(ids=[str(x) for x in data['targetid']],selection='first 256 original source validation objects in archived order',checkpoint='original frozen HSC QLoRA plus trained PDF head',
        variants=list(texts),target_labels_used=False,weights_updated=False,code_sha256=d.sha(__file__),input_sha256={p.name:d.sha(p) for p in (folder/'inputs').glob('*.jsonl')}))
    d.write(folder/'status.json',dict(phase='starting',pid=os.getpid()))
    with pilot.reserve_one_gpu() as gpu:
        model,tokenizer=runner.load_model(folder,m,'original');rows=[]
        ref=torch.load(d.BASE/'level0/reference_validation.pt',map_location='cpu',weights_only=False)
        ref={k:(v[:256] if k=='object_id' or (isinstance(v,torch.Tensor) and v.ndim>0 and k not in ('redshift_edges','redshift_centers')) else v) for k,v in ref.items()}
        for name,strings in texts.items():
            audit_tokens(tokenizer,strings,512);e=pilot.evaluate(model,tokenizer,data,strings,m['source_config'])
            if name=='source_native':d.write(folder/'native_replay.json',verify_replay(e,ref))
            pilot.save_predictions(folder/f'{name}.npz',data,e)
            result=dict(condition=name,**d.metrics({k:v.numpy() for k,v in e.items() if isinstance(v,torch.Tensor)},np.ones(256,bool)))
            rows.append(result);print(json.dumps(result),flush=True)
        d.table(folder/'comparison.csv',rows);d.check(out)
        d.write(folder/'status.json',dict(phase='complete',gpu=gpu,weights_updated=False,test_evaluated=False))


if __name__=='__main__':main()
