#!/usr/bin/env python3
"""Read raw HSC/DESI metadata and exact-ID Tractor counterparts; no inference."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import fitsio
import h5py
import numpy as np
from aion_magnitude.survey_catalogues import exact_ids, fingerprints, magnitude_features

DATA=Path('/arc/projects/ots/Cosmic_Imprint_of_Time')
HDF=DATA/'GalaxiesMLSpec/DESI_HSC_64x64_v2.hdf5'
PHOT=DATA/'DESI-LS/catalogues/ls_dr9/iron/v1.1/tractorphot'
ZCAT=DATA/'DESI-LS/catalogues/desi_dr1/iron/zcatalog/v1/zall-pix-iron.fits'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir
    if out.exists():raise FileExistsError('Preserve prior audit; choose a new directory')
    out.mkdir(parents=True);started=time.time()
    with h5py.File(HDF,'r') as f:
        ids=exact_ids(f['targetid'][:]);assert np.array_equal(ids,f['DESI_fibermap/TARGETID'][:])
        h=dict(targetid=ids,hsc_object_id=exact_ids(f['object_id'][:]),hsc_ra=f['HSC_metadata/ra'][:],hsc_dec=f['HSC_metadata/dec'][:],
               target_ra=f['DESI_fibermap/TARGET_RA'][:],target_dec=f['DESI_fibermap/TARGET_DEC'][:],z_hdf=f['DESI_fibermap/DESI_redshift'][:],
               zerr_hdf=f['DESI_fibermap/DESI_redshift_err'][:],hdf_release=f['DESI_fibermap/RELEASE'][:],hdf_brickid=f['DESI_fibermap/BRICKID'][:],hdf_objid=f['DESI_fibermap/BRICK_OBJID'][:])
        pairs=[magnitude_features(f[f'HSC_metadata/{b}_cmodel_mag'][:],f[f'HSC_metadata/{b}_cmodel_magsigma'][:]) for b in ('g','r','i','z','y')]
        h['hsc_features']=np.column_stack([x for pair in pairs for x in pair[:2]]+[f['DESI_fibermap/EBV'][:]]).astype(np.float32)
        h['hsc_valid']=np.column_stack([pair[2] for pair in pairs])
        h['hdf_legacy_features']=np.column_stack([x for b in ('G','R','Z','W1','W2') for x in (f['DESI_fibermap/FLUX_'+b][:],f['DESI_fibermap/FLUX_IVAR_'+b][:])]+[f['DESI_fibermap/EBV'][:]])
    assert len(np.unique(ids))==len(ids)
    np.savez_compressed(out/'hdf_metadata.npz',**h)
    order=np.argsort(ids);wanted=ids[order]
    def match(values):
        pos=np.searchsorted(wanted,values);good=pos<len(wanted);good[good]&=wanted[pos[good]]==values[good]
        return good
    columns=['TARGETID','RELEASE','BRICKID','OBJID','RA','DEC','EBV','MASKBITS','PHOTSYS']
    # PHOTSYS is not present in every Tractor VAC; RELEASE supplies actual instrument provenance.
    with fitsio.FITS(str(sorted(PHOT.glob('*.fits'))[0])) as f:available=f[1].get_colnames()
    columns=[c for c in columns if c in available]
    columns += [f'{prefix}_{b}' for b in ('G','R','Z','W1','W2') for prefix in ('FLUX','FLUX_IVAR','NOBS')]
    rows=[];sources=[]
    for i,path in enumerate(sorted(PHOT.glob('*.fits'))):
        with fitsio.FITS(str(path)) as f:
            tid=f[1].read(columns=['TARGETID'])['TARGETID'];indices=np.flatnonzero(match(tid))
            if len(indices):
                found=f[1].read(rows=indices,columns=columns);rows.append(found)
                sources.append(dict(path=str(path),matched_rows=len(found)))
        if i%10==0:print('Tractor shards',i+1,'matched',sum(len(x) for x in rows),flush=True)
    if not rows:raise ValueError('No exact-ID Tractor matches')
    phot=np.concatenate(rows);np.savez_compressed(out/'tractor_matches.npz',**{k:phot[k] for k in phot.dtype.names})
    print('TRACTOR COMPLETE',len(phot),flush=True)
    zcols=['TARGETID','Z','ZERR','ZCAT_PRIMARY','SPECTYPE','OBJTYPE','ZWARN','COADD_FIBERSTATUS','DELTACHI2','SURVEY','PROGRAM']
    zs=[]
    with fitsio.FITS(str(ZCAT)) as f:
        n=f[1].get_nrows()
        for start in range(0,n,200000):
            a=f[1].read(rows=np.arange(start,min(start+200000,n)),columns=zcols);good=match(a['TARGETID'])
            if good.any():zs.append(a[good])
            if start%2000000==0:print('ZCAT rows',start,'/',n,'matched',sum(len(x) for x in zs),flush=True)
    z=np.concatenate(zs);np.savez_compressed(out/'redshift_matches.npz',**{k:z[k] for k in z.dtype.names})
    audit=dict(hdf_rows=len(ids),tractor_rows=len(phot),tractor_unique_ids=len(np.unique(phot['TARGETID'])),zcat_rows=len(z),sources=sources,
               source_fingerprints=fingerprints([HDF,ZCAT,*sorted(PHOT.glob('*.fits'))]),seconds=time.time()-started,
               image_or_spectral_pixels_read=False,training=False,model_inference=False,
               hdf_legacy_limitation='DESI targeting fibermap flux/ivar/EBV/RELEASE exist but NOBS is absent; use original Tractor VAC measurements and NOBS instead, auditing numerical differences')
    (out/'raw_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('AUDIT COMPLETE',json.dumps({k:v for k,v in audit.items() if k not in ('sources','source_fingerprints')}),flush=True)


if __name__=='__main__':main()
