#!/usr/bin/env python3
"""Freeze whole-P evaluation under unchanged scientific rules; reuse exact chunks."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from hsc_legacy_paired_benchmark import prepare,read,write,sha,MODELS,now
from audit_hsc_legacy_measurements import main as supplement


def main(source,out):
    out.mkdir(parents=True,exist_ok=False)
    (out/'raw-audit').symlink_to((source/'raw-audit').resolve(),target_is_directory=True)
    prepare(argparse.Namespace(output_dir=out,n_primary=0,n_validation=256))
    supplement(out)
    old=json.loads((source/'manifest.json').read_text());new=json.loads((out/'manifest.json').read_text())
    assert old['code_sha256']==new['code_sha256'] and old['registry_sha256']==new['registry_sha256']
    assert old['max_length']==new['max_length'] and old['protected_sha256']==new['protected_sha256']
    a,b=read(source/'cohort.npz'),read(out/'cohort.npz');lookup={int(t):i for i,t in enumerate(b['targetid'])}
    positions=np.array([lookup[int(t)] for t in a['targetid']])
    for key in a:np.testing.assert_equal(a[key],b[key][positions])
    # Verify actual input bytes in addition to code/schema/feature identity.
    for path in (source/'inputs').glob('level*_*.jsonl'):
        if 'replay' in path.name:continue
        expected={r['targetid']:r['text'] for r in map(json.loads,path.open())}
        actual={r['targetid']:r['text'] for r in map(json.loads,(out/'inputs'/path.name).open()) if r['targetid'] in expected}
        assert actual==expected
    reused=[]
    for model in MODELS:
        assert (source/model/'complete.json').exists(),f'Wait for first pass to finish: {model}'
        (out/model).mkdir()
        for domain in ('hsc','legacy'):
            for path in sorted((source/model).glob(f'{domain}_??????_??????.npz')):
                data=read(path);pos=np.array([lookup[int(t)] for t in data['targetid']]);start=int(pos[0]);end=int(pos[-1])+1
                if start%256 or not np.array_equal(pos,np.arange(start,end)):continue
                dest=out/model/f'{domain}_{start:06d}_{end:06d}.npz';shutil.copy2(path,dest)
                assert sha(path)==sha(dest)
                reused.append(dict(source=str(path.resolve()),destination=str(dest.resolve()),sha256=sha(dest),rows=len(pos)))
    for name in ['base_model_identity.json','tests.log']:
        shutil.copy2(source/name,out/name)
    write(out/'first_pass_reuse.json',dict(created_utc=now(),source=str(source.resolve()),chunks=reused,
        input_bytes_exact=True,all_feature_label_and_exposure_arrays_exact=True,checkpoint_and_grid_unchanged=True,
        scientific_rules='Full independent P pool under the exact first-pass cuts; no changes from prediction performance; same source-only support rule and predeclared bin-merge algorithm; first-pass V subset retained'))
    print('FULL COHORT PREPARED; reused',sum(x['rows'] for x in reused),'prediction rows',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();main(args.source,args.output_dir)
