"""Verified Stage-2 physical inputs; no target redshift labels are read here."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'notebooks'),str(ROOT)]
import diagnose_hsc_clauds_stage0 as d
from aion_magnitude.filter_photometry import filter_key,CLAUDS_BANDS
PAIRS=Path('/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1027/hsc-legacy-paired-full')
RAW=PAIRS/'raw-audit'
REGISTRY=json.loads((ROOT/'configs/filter_registry_v1.json').read_text())
FILTERS=sorted(REGISTRY['filters'])


def group_fraction(group):
    return int.from_bytes(hashlib.sha256(f'42:{group}'.encode()).digest()[:8],'big')/2**64


def archive_validation_ids():
    import torch
    p=Path('/arc/home/gsm/aion_output/figures/qwen-desi-ls-qlora-specz-e10/qlora/result.pt')
    result=torch.load(p,map_location='cpu',weights_only=False,mmap=True)['val_evaluation']
    return np.array([int(x) for x in result['object_id']],np.int64),result['z_spec'].numpy(),p


def make_rows(ids,features,band_names,z,split,ra,dec,catalogue,quality=None,quality_available=None):
    n=len(ids);f=np.full((n,11),np.nan);e=f.copy();bid=np.full((n,11),-1,np.int16)
    f[:,:len(band_names)]=features[:,:2*len(band_names):2];e[:,:len(band_names)]=features[:,1:2*len(band_names):2]
    if isinstance(band_names[0],str):
        bid[:,:len(band_names)]=[FILTERS.index(b) for b in band_names]
    else:raise ValueError('Use a fixed filter mapping or replace band IDs after construction')
    q=np.zeros_like(f);qa=np.zeros_like(f,dtype=bool)
    if quality is not None:q[:,:len(band_names)]=quality
    if quality_available is not None:qa[:,:len(band_names)]=quality_available
    return dict(object_id=ids.astype('U30'),group=ids.astype('U30'),flux=f,sigma=e,band_id=bid,quality=q,quality_available=qa,ebv=features[:,-1].astype(float),ebv_available=np.isfinite(features[:,-1]),ra=np.asarray(ra),dec=np.asarray(dec),z=np.asarray(z,np.float32),split=np.asarray(split,dtype='U10'),catalogue=np.full(n,catalogue,dtype='U10'))


def take(a,indices):return {k:v[indices] for k,v in a.items()}


def recover_hsc(out):
    # This archived file contains only the original train/validation cohorts.
    a=d.npz(d.BASE/'level0/source_cohort.npz');assert set(a['split'])=={'train','val'}
    with np.load(RAW/'hdf_metadata.npz',allow_pickle=False) as f:
        ids=f['targetid'];order=np.argsort(ids);where=order[np.searchsorted(ids[order],a['targetid'])];assert np.array_equal(ids[where],a['targetid'])
        ra=f['hsc_ra'][where];dec=f['hsc_dec'][where]
    rows=make_rows(a['targetid'],a['features'],[f'HSC.{b}' for b in 'grizy'],a['z'],a['split'],ra,dec,'HSC')
    rows['group']=a['group']
    np.savez_compressed(out/'hsc_trainval.npz',**rows)
    return rows


def recover_legacy(out):
    """Read only original validation photometry after exact TARGETID lookup."""
    import fitsio
    from aion_magnitude.survey_catalogues import flux_features
    ids,z,reference=archive_validation_ids();lookup={int(x):i for i,x in enumerate(ids)};sorted_ids=np.sort(ids);found=np.zeros(len(ids),int)
    bands=['g','r','z','W1','W2'];cols=['TARGETID','RELEASE','BRICKID','OBJID','RA','DEC','EBV','MASKBITS']+[f'{prefix}_{b.upper()}' for b in bands for prefix in ['FLUX','FLUX_IVAR','NOBS','MW_TRANSMISSION']]
    data={};sources=[];meta=json.loads(Path('/arc/home/gsm/aion_output/figures/qwen-desi-ls-qlora-specz-e10/prepared.json').read_text())
    paths=[Path(p['path']) for p in meta['sources'] if 'tractorphot-' in p['path']]
    for path in paths:
        with fitsio.FITS(str(path)) as f:
            allids=f[1].read(columns=['TARGETID'])['TARGETID'];pos=np.searchsorted(sorted_ids,allids);valid=pos<len(sorted_ids);valid[valid]&=sorted_ids[pos[valid]]==allids[valid];where=np.flatnonzero(valid)
            if not len(where):continue
            raw=f[1].read(rows=where,columns=cols)
        for r in raw:
            i=lookup[int(r['TARGETID'])];found[i]+=1
            for k in cols:
                if k not in data:data[k]=np.empty(len(ids),dtype=raw[k].dtype)
                data[k][i]=r[k]
        sources.append(dict(path=str(path),rows_read=len(raw),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns))
        print(path.name,len(raw),'validation rows',flush=True)
    assert np.all(found==1),dict(missing=int((found==0).sum()),duplicates=int((found>1).sum()))
    pairs=[flux_features(data[f'FLUX_{b.upper()}'],data[f'FLUX_IVAR_{b.upper()}'],data[f'NOBS_{b.upper()}']) for b in bands]
    features=np.column_stack([x for p in pairs for x in p[:2]]+[data['EBV']]).astype(np.float32)
    rows=make_rows(ids,features,['DECam.g','DECam.r','DECam.z','WISE.W1','WISE.W2'],z,np.full(len(ids),'val'),data['RA'],data['DEC'],'Legacy',quality=np.repeat((data['MASKBITS']!=0)[:,None],5,axis=1),quality_available=np.ones((len(ids),5),bool))
    for i,release in enumerate(data['RELEASE']):
        rows['band_id'][i,:5]=[FILTERS.index(filter_key('desi-ls',b,int(release))) for b in bands]
    groups=np.array([f'ls:{int(r)}:{int(b)}:{int(o)}' for r,b,o in zip(data['RELEASE'],data['BRICKID'],data['OBJID'])])
    assert np.all([.2<=group_fraction(g)<.25 for g in groups]),'Original Legacy validation membership must remain unchanged'
    np.savez_compressed(out/'legacy_validation_raw.npz',**data)
    np.savez_compressed(out/'legacy_validation.npz',**rows)
    # Independent archived query values must reproduce the existing loader.
    old=d.npz(d.FAST/'desi-ls_predictions.npz');order=np.argsort(ids);pos=order[np.searchsorted(ids[order],old['targetid'])]
    assert np.array_equal(ids[pos],old['targetid']);np.testing.assert_array_equal(features[pos],old['features']);np.testing.assert_array_equal(z[pos],old['z_spec'])
    d.write(out/'legacy_recovery.json',dict(validation_objects=len(ids),reference=str(reference),reference_sha256=d.sha(reference),photometry_sources=sources,all_validation_ids_found_once=True,archived_256_query_features_bit_exact=True,source_test_photometry_read=False,quality=meta['quality'],original_cache_missing=True,original_validation_recovered=True))
    return rows


def recover_target(out):
    import fitsio
    old=d.npz(d.COMMON/'cohort.npz',('targetid','source_row','flux_nanomaggy','sigma_nanomaggy','notObserved','isNoData','hasBadPhotometry','isClean'))
    with fitsio.FITS(str(d.CAT)) as f:raw=f[1].read(rows=old['source_row'],columns=['ID','RA','DEC'])
    assert np.array_equal(raw['ID'],old['targetid']);n=len(raw)
    # No ZPHOT is read or placed in this physical-input artifact.
    rows=dict(object_id=old['targetid'].astype('U30'),group=old['targetid'].astype('U30'),flux=old['flux_nanomaggy'],sigma=old['sigma_nanomaggy'],band_id=np.repeat(np.array([[FILTERS.index(filter_key('clauds',b)) for b in CLAUDS_BANDS]],np.int16),n,axis=0),quality=old['hasBadPhotometry'].astype(float),quality_available=np.ones((n,11),bool),ebv=np.full(n,np.nan),ebv_available=np.zeros(n,bool),ra=raw['RA'],dec=raw['DEC'],split=np.full(n,'diagnostic'),catalogue=np.full(n,'CLAUDS'))
    np.savez_compressed(out/'clauds_photometry.npz',**rows)
    d.write(out/'target_provenance.json',dict(ids=rows['object_id'].tolist(),source_rows=old['source_row'].tolist(),reference='ZPHOT, not spectroscopy; loaded only after configurations are frozen',local_independent_spectroscopic_column=False,conversion=10**((22.5-23.9)/2.5),availability='Original notObserved/isNoData and nonfinite/-99 rules; signed finite flux retained',quality='Original hasBadPhotometry/isClean retained; no new quality cut'))
    return rows


def remove_target_overlap(source,target):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    a=SkyCoord(source['ra']*u.deg,source['dec']*u.deg);b=SkyCoord(target['ra']*u.deg,target['dec']*u.deg);_,sep,_=a.match_to_catalog_sky(b);keep=sep.arcsec>1.
    return take(source,keep),dict(radius_arcsec=1.,source_objects_excluded=int((~keep).sum()),excluded_source_ids=source['object_id'][~keep].tolist(),minimum_retained_arcsec=float(sep.arcsec[keep].min()))


def mixed_pilot(out,target):
    with np.load(PAIRS/'eligible_pairs.npz',allow_pickle=False) as a:
        # First inspect membership/identity only. Never evaluate original HSC test galaxies.
        eligible=a['original-qlora_train']|a['original-qlora_val'];indices=np.flatnonzero(eligible)
        fields=['targetid','galaxy_id','ra','dec','legacy_ra','legacy_dec','pair_separation_arcsec','z','hsc_features','legacy_features','legacy_release','legacy_group','original-qlora_train','original-qlora_val']
        data={k:a[k][indices] for k in fields}
    role=np.array([group_fraction(g) for g in data['legacy_group']]);legacy_train=role<.2;legacy_val=(role>=.2)&(role<.25)
    # Union-validation policy: never train a validation galaxy via another view.
    train=data['original-qlora_train']&legacy_train
    val=(data['original-qlora_val']&(legacy_train|legacy_val))|(data['original-qlora_train']&legacy_val)
    assert not np.any(train&val)
    frames=[]
    for name,features,ra,dec in [('HSC',data['hsc_features'],data['ra'],data['dec']),('Legacy',data['legacy_features'],data['legacy_ra'],data['legacy_dec'])]:
        rows=make_rows(data['targetid'],features,[f'HSC.{b}' for b in 'grizy'] if name=='HSC' else ['DECam.g','DECam.r','DECam.z','WISE.W1','WISE.W2'],data['z'],np.where(train,'train',np.where(val,'val','excluded')),ra,dec,name)
        if name=='Legacy':
            for i,r in enumerate(data['legacy_release']):rows['band_id'][i,:5]=[FILTERS.index(filter_key('desi-ls',b,int(r))) for b in ('g','r','z','W1','W2')]
        frames.append(rows)
    a={k:np.concatenate([frame[k] for frame in frames]) for k in frames[0]};a=take(a,np.isin(a['split'],['train','val']));a,overlap=remove_target_overlap(a,target)
    assert not (set(a['group'][a['split']=='train'])&set(a['group'][a['split']=='val']))
    # A conservative sub-arcsecond cross-split positional audit; reject ambiguous groups.
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    tr=np.flatnonzero((a['split']=='train')&(a['catalogue']=='HSC'));va=np.flatnonzero((a['split']=='val')&(a['catalogue']=='HSC'))
    _,sep,_=SkyCoord(a['ra'][tr]*u.deg,a['dec'][tr]*u.deg).match_to_catalog_sky(SkyCoord(a['ra'][va]*u.deg,a['dec'][va]*u.deg));bad=set(a['group'][tr[sep.arcsec<1.]])
    a=take(a,~np.isin(a['group'],list(bad)))
    np.savez_compressed(out/'mixed_eligible_trainval.npz',**a)
    d.write(out/'mixed_overlap_audit.json',dict(legacy_original_cache_missing=True,setting='Audited same-galaxy HSC+Legacy multi-view subset, not the complete independently sampled Legacy cohort',source_id_match_and_position_radius_arcsec=1.,original_hsc_test_objects_excluded=True,original_legacy_test_group_assignments_excluded=True,source_val_union_reserved=True,paired_train_galaxies=len(set(a['group'][a['split']=='train'])),paired_val_galaxies=len(set(a['group'][a['split']=='val'])),ambiguous_train_groups_excluded=len(bad),target_overlap=overlap,source_reference='Native HDF/DESI spectroscopic labels with independently audited Iron agreement <=1e-6*(1+z)',full_mixed_population_training_pending=True))
    return a


def independent_ebv(out,target):
    from concurrent.futures import ThreadPoolExecutor
    import urllib.request,urllib.parse,xml.etree.ElementTree as ET
    folder=out/'extinction_queries';folder.mkdir(exist_ok=True)
    def one(i):
        ident=target['object_id'][i];path=folder/f'{ident}.xml';url='https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?'+urllib.parse.urlencode({'locstr':f'{target["ra"][i]:.8f} {target["dec"][i]:.8f} equ j2000'})
        try:
            raw=path.read_bytes() if path.exists() else urllib.request.urlopen(url,timeout=30).read();tree=ET.fromstring(raw)
            value=tree.find('.//refPixelValueSFD');assert value is not None
            ebv=float(value.text.strip().split()[0]);assert 0<=ebv<2;path.write_bytes(raw)
            return ebv,dict(object_id=str(ident),url=url,ebv_sfd=ebv,xml_sha256=d.sha(path))
        except Exception as error:return np.nan,dict(object_id=str(ident),url=url,error=repr(error))
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(one,range(len(target['object_id']))))
    values=np.array([v[0] for v in results]);np.savez_compressed(out/'clauds_independent_ebv.npz',object_id=target['object_id'],ebv=values,available=np.isfinite(values));d.write(out/'independent_extinction_provenance.json',dict(provider='IRSA DUST',map='SFD refPixelValueSFD; no redshift or photometric fitting',records=[r[1] for r in results],available=int(np.isfinite(values).sum()),failed=int((~np.isfinite(values)).sum()),primary_photometry='Uncorrected/as-loaded P0; this independent map is only a predeclared correction sensitivity and separately identified metadata'))


def recover(out):
    out.mkdir(parents=True,exist_ok=True);target=recover_target(out);hsc=recover_hsc(out);legacy=recover_legacy(out)
    hsc,a=remove_target_overlap(hsc,target);legacy,c=remove_target_overlap(legacy,target)
    np.savez_compressed(out/'hsc_trainval.npz',**hsc);np.savez_compressed(out/'legacy_validation.npz',**legacy)
    mixed=mixed_pilot(out,target);independent_ebv(out,target)
    d.write(out/'data_audit.json',dict(hsc_rows=len(hsc['object_id']),hsc_train=int((hsc['split']=='train').sum()),hsc_val=int((hsc['split']=='val').sum()),legacy_original_validation=len(legacy['object_id']),clauds_queries=len(target['object_id']),filters=FILTERS,source_hsc_target_overlap=a,source_legacy_target_overlap=c,source_split_ranges='Existing source cohorts already require 0<=z<=4.5; no high-z labels are clipped',outside_grid_source_labels=int(np.sum((hsc['z']<0)|(hsc['z']>4.5))+np.sum((legacy['z']<0)|(legacy['z']>4.5))),target_labels_read=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'reports/stage2a/1028');recover(parser.parse_args().output_dir)
