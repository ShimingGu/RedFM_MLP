#!/usr/bin/env python3
"""Recover the historical Legacy sample, verified against archived IDs and labels."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'notebooks'))
import numpy as np
from hsc_legacy_paired_benchmark import sha,read,now
from filter_metadata_counterfactual import write


def instrument_positions(code):
    """Add observational bookkeeping only; sampling, RNG and sorting are untouched."""
    anchor='for path in paths[1:]:'
    assert code.count(anchor)==1
    code=code.replace(anchor,'for source_file_index, path in enumerate(paths[1:]):')
    anchor='"targetid": rows["TARGETID"][good].astype(np.int64),'
    assert code.count(anchor)==1
    code=code.replace(anchor,anchor+'\n                            "ra": rows["RA"][good].astype(np.float64),\n                            "dec": rows["DEC"][good].astype(np.float64),\n                            "source_file_index": np.full(int(good.sum()), source_file_index, dtype=np.int16),\n                            "source_row": (np.flatnonzero(valid)[good] + start).astype(np.int64),')
    return code


def main(out):
    import torch
    import qwen_survey_posttraining as pipeline
    from aion_magnitude import survey_catalogues as surveys
    root=Path('/arc/home/gsm/aion_output/figures');run=root/'qwen-desi-ls-qlora-specz-e10'
    original=json.loads((run/'prepared.json').read_text())
    args=pipeline.parser().parse_args(['--stage','prepare','--catalogue-kind','desi-ls','--method','qlora','--output-dir',str(run)])
    settings=pipeline.cohort_settings(args);key_settings=settings.copy();key_settings.pop('hsc_file')
    source=Path(surveys.__file__).read_text();fingerprints=surveys.fingerprints(surveys.source_paths('desi-ls',args.desi_root,args.hsc_file))
    assert fingerprints==original['sources'],'Raw input ordering or fingerprints changed'
    key=pipeline.digest(dict(settings=key_settings,schema=surveys.SCHEMA_VERSION,code=hashlib.sha256(source.encode()).hexdigest(),sources=fingerprints))
    assert key==original['cache_key'],'Historical preparation source/config identity not recovered'
    folder=out/'legacy_checkpoint_audit/recovered_cohort'
    if folder.exists():raise FileExistsError('Never overwrite a recovered cohort')
    folder.mkdir(parents=True);started=time.time()
    (folder/'archived_loader.py').write_text(source)
    code=instrument_positions(source);(folder/'instrumented_loader.py').write_text(code)
    spec=importlib.util.spec_from_file_location('legacy_loader_1028',folder/'instrumented_loader.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    data,metadata=loader.build_cohort(**settings)
    for field in ['split_counts','n_rows','eligible_photometry_rows','duplicate_imaging_rows_removed_after_sampling','quality_selected_redshifts','redshift_rows_scanned']:
        assert metadata[field]==original[field],f'Historical sampling mismatch: {field}'
    evidence={}
    for method in ['qlora','dora']:
        result_path=root/f'qwen-desi-ls-{method}-specz-e10'/method/'result.pt'
        result=torch.load(result_path,map_location='cpu',weights_only=False,mmap=True)
        assert result['metadata']['survey_cohort_cache_key']==key
        evidence[method]={}
        for split in ['val','test']:
            rows=np.flatnonzero(data['split']==split);ev=result[split+'_evaluation'];ids=np.asarray(ev['object_id'],np.int64)
            np.testing.assert_array_equal(data['targetid'][rows],ids)
            np.testing.assert_array_equal(data['z'][rows],ev['z_spec'].numpy())
            evidence[method][split]=dict(n=len(rows),ordered_TARGETIDs_exact=True,all_specz_exact=True,
                targetids_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
    assert not np.isin(data['targetid'][data['split']=='train'],data['targetid'][data['split']!='train']).any()
    np.savez_compressed(folder/'cohort.npz',**data)
    metadata['cache_key']=key;write(folder/'cohort_metadata.json',metadata)
    write(folder/'recovery_proof.json',dict(created_utc=now(),seconds=time.time()-started,
        archived_cache_key_exact=True,original_code_sha256=hashlib.sha256(source.encode()).hexdigest(),cache_key=key,
        raw_fingerprints_and_order_exact=True,sampler='Original NumPy PCG64(seed=42) random priorities, global lowest 300000, then original imaging-group deduplication and SHA256 group split',
        instrumentation='Only RA/DEC and exact raw source_file_index/source_row bookkeeping added to sampled records; priority RNG, thresholds, numeric features, sorting and split logic unchanged',
        independent_archival_evidence=evidence,all_historical_counts_exact=True,
        train_identity_status='Deterministically reconstructed with exact archived code/config/raw fingerprints and full ordered archived validation/test identities and labels verified for both independently trained methods',
        cohort_sha256=sha(folder/'cohort.npz'),no_training=True,original_artifacts_modified=False))
    print('HISTORICAL COHORT VERIFIED',metadata['split_counts'],'seconds',time.time()-started,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    main(parser.parse_args().output_dir)
