#!/usr/bin/env python3
"""Supplement paired audit with original feature, quality and provenance checks."""
import argparse
from pathlib import Path

import h5py
import numpy as np

from hsc_legacy_paired_benchmark import read,write,sha,WARM,MODELS,now


def main(out):
    raw=out/'raw-audit';h=read(raw/'hdf_metadata.npz');p=read(raw/'tractor_matches.npz');s=read(raw/'redshift_matches.npz')
    d=read(out/'eligible_pairs.npz');source=read(WARM/'level0/source_cohort.npz')
    hlookup={int(t):i for i,t in enumerate(h['targetid'])};plookup={int(t):i for i,t in enumerate(p['TARGETID'])}
    src=np.array([hlookup[int(t)] for t in source['targetid']]);np.testing.assert_equal(h['hsc_features'][src],source['features'])
    rows=np.array([hlookup[int(t)] for t in d['targetid']]);pr=np.array([plookup[int(t)] for t in d['targetid']])
    text=lambda k:np.char.strip(s[k].astype('U'))
    q=(s['ZCAT_PRIMARY']!=0)&(text('SPECTYPE')=='GALAXY')&(text('OBJTYPE')=='TGT')&(s['ZWARN']==0)&(s['COADD_FIBERSTATUS']==0)&np.isfinite(s['DELTACHI2'])&(s['DELTACHI2']>40)&np.isfinite(s['Z'])&(s['Z']>=0)&(s['Z']<=4.5)&np.isfinite(s['ZERR'])&(s['ZERR']>=0)
    qlookup={int(s['TARGETID'][i]):i for i in np.flatnonzero(q)};zr=np.array([qlookup[int(t)] for t in d['targetid']])
    data=Path('/arc/projects/ots/Cosmic_Imprint_of_Time')
    with h5py.File(data/'GalaxiesMLSpec/DESI_HSC_64x64_v2.hdf5','r') as f:
        fiber=f['DESI_fibermap/COADD_FIBERSTATUS'][:][rows];obj=f['DESI_fibermap/OBJTYPE'][:][rows].astype('U')
    paths=[data/'GalaxiesMLSpec/provenance/zenodo_record_16989593.json',
           data/'GalaxiesMLSpec/provenance/publisher_description.html',
           data/'DESI-LS/provenance/lsdr9_photometry_README.md']
    legacy_keys=np.array([f'{a}:{b}:{c}' for a,b,c in zip(p['RELEASE'],p['BRICKID'],p['OBJID'])])
    exposure={}
    for model in MODELS:
        ids=read(out/f'{model}_exposure_ids.npz')
        for split in ('train','val','test'):
            sr=np.array([plookup[int(t)] for t in ids[split]])
            same=np.isin(legacy_keys[pr],legacy_keys[sr])
            extra=same&~d[f'{model}_{split}']
            exposure[f'{model}_{split}']=dict(same_tractor_imaging_key=int(same.sum()),additional_unflagged=int(extra.sum()))
            assert not extra.any(),'Additional alternate-ID exposure requires a new audited policy/cohort before inference can be used'
    result=dict(created_utc=now(),changes_to_eligibility=False,
        original_hsc_features=dict(rows=len(src),all_reconstructed_raw_train_validation_features_exactly_equal=True),
        quality=dict(rows=len(rows),hdf_fields_available=['OBJTYPE','COADD_FIBERSTATUS'],
            hdf_missing=['ZCAT_PRIMARY','ZWARN','SPECTYPE','DELTACHI2','SURVEY','PROGRAM'],
            fiberstatus_equal=int((fiber==s['COADD_FIBERSTATUS'][zr]).sum()),hdf_fiberstatus_nonzero=int((fiber!=0).sum()),
            objtype_equal=int((np.char.strip(obj)==text('OBJTYPE')[zr]).sum()),
            applied_selection='Archived DR1 Iron uniform galaxy quality; HDF lacks complete spectral-quality flags; agreement of individual fields cannot prove identical reductions'),
        tractor_quality=dict(maskbits_nonzero=int((p['MASKBITS'][pr]!=0).sum()),
            nobs={b:dict(zero=int((p['NOBS_'+b][pr]<=0).sum()),min=int(p['NOBS_'+b][pr].min()),median=float(np.median(p['NOBS_'+b][pr]))) for b in ['G','R','Z','W1','W2']},
            new_maskbits_cut=False),
        uncertainty_comparison=dict(hdf_iron_zerr_equal=int((h['zerr_hdf'][rows]==s['ZERR'][zr]).sum()),rows=len(rows)),
        alternate_legacy_imaging_identity_exposure=exposure,
        provenance_sha256={str(p):sha(p) for p in paths},
        positional_policy='All raw HDF TARGETIDs have exact unique Tractor counterparts; no position-only recovery was needed. Every retained counterpart must additionally pass the independent 1-arcsec HSC/Tractor positional verification.')
    write(out/'supplementary_audit.json',result)
    print(result['quality'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    main(parser.parse_args().output_dir)
