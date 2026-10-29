"""Exact historical DESI-Legacy reconstruction; never evaluates source test."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aion_magnitude.survey_catalogues import build_cohort,fingerprints,source_paths
OUT=ROOT/'reports/stage3a/1029'
OUT.mkdir(parents=True,exist_ok=True)
meta=json.loads(Path('/arc/home/gsm/aion_output/figures/qwen-desi-ls-qlora-specz-e10/prepared.json').read_text())
paths=source_paths('desi-ls','/arc/projects/ots/Cosmic_Imprint_of_Time/DESI-LS','')
assert fingerprints(paths)==meta['sources'],'Historical input fingerprint changed'
t=time.time()
result=build_cohort(kind='desi-ls')
data,actual=result
assert actual['n_rows']==meta['n_rows']
assert actual['split_counts']==meta['split_counts']
keep=data['split']!='test'
np.savez_compressed(OUT/'legacy_membership_only.npz',**{k:data[k] for k in ['targetid','group','split']})
np.savez_compressed(OUT/'legacy_exact_trainval.npz',**{k:v[keep] for k,v in data.items()})
val=np.load(ROOT/'reports/stage2a/1028/legacy_validation.npz')
idx=np.argsort(data['targetid'][data['split']=='val'])
oldidx=np.argsort(val['object_id'].astype(np.int64))
np.testing.assert_array_equal(data['targetid'][data['split']=='val'][idx],val['object_id'].astype(np.int64)[oldidx])
np.testing.assert_array_equal(data['z'][data['split']=='val'][idx],val['z'][oldidx])
(OUT/'legacy_reconstruction.json').write_text(json.dumps(dict(metadata=actual,elapsed_seconds=time.time()-t,exact_original_input_fingerprints=True,original_validation_ids_and_redshifts_exact=True,source_test_evaluated=False,source_test_rows_saved=False),indent=2))
print('RECOVERY COMPLETE',actual['split_counts'],flush=True)
