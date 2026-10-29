"""Stage3 identity/photometry preparation. New target redshifts stay unopened."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
from stage2_data import d,FILTERS,make_rows,take,RAW,group_fraction
from aion_magnitude.filter_photometry import CLAUDS_BANDS,filter_key,clauds_flux_to_nanomaggy
OUT=ROOT/'reports/stage3a/1029'
def xyz(ra,dec):
    r=np.deg2rad(ra);de=np.deg2rad(dec)
    return np.c_[np.cos(de)*np.cos(r),np.cos(de)*np.sin(r),np.sin(de)]
def nearest(a,b,k=1):
    from scipy.spatial import cKDTree
    dist,idx=cKDTree(xyz(b['ra'],b['dec'])).query(xyz(a['ra'],a['dec']),k=k,workers=2)
    return np.rad2deg(2*np.arcsin(np.clip(dist/2,0,1)))*3600,idx
def target():
    import fitsio
    specfile=OUT/'external/specz_unique.fits'
    with fitsio.FITS(str(specfile)) as f:
        s=f[1].read(columns=['Id_specz','ra_corrected','dec_corrected','flag','Confidence_level','Priority','survey','GroupSize'])
    good=(s['flag']==4)&(s['Confidence_level']>=97)&(s['Priority']==1)&np.isfinite(s['ra_corrected'])&np.isfinite(s['dec_corrected'])
    sidx=np.flatnonzero(good);s=s[good];print('high-quality positions',len(s),flush=True)
    with fitsio.FITS(str(d.CAT)) as f:
        a=f[1].read(columns=['ID','RA','DEC','isStar','isParent_HSC-I','isDuplicated_HSC-I','isSky_HSC-I'])
    region=(a['RA']>147)&(a['RA']<153)&(a['DEC']>-1)&(a['DEC']<5)
    indices=np.flatnonzero(region);a=a[region]
    pos={'ra':a['RA'],'dec':a['DEC']};sp={'ra':s['ra_corrected'],'dec':s['dec_corrected']}
    sep,j=nearest(sp,pos,k=2)
    ok=(sep[:,0]<.5)&(sep[:,1]>1.)
    counts=np.bincount(j[ok,0],minlength=len(a));ok&=counts[j[:,0]]==1
    pick=j[:,0];ok&=~a['isStar'][pick].astype(bool)&~a['isParent_HSC-I'][pick].astype(bool)&~a['isDuplicated_HSC-I'][pick].astype(bool)&~a['isSky_HSC-I'][pick].astype(bool)
    old=d.npz(d.COMMON/'cohort.npz',('targetid',))['targetid'];ok&=~np.isin(a['ID'][pick],old)
    h=d.npz(RAW/'hdf_metadata.npz',('hsc_ra','hsc_dec'))
    hs,_=nearest(sp,dict(ra=h['hsc_ra'],dec=h['hsc_dec']));ok&=hs>1.
    use=np.flatnonzero(ok)
    # Freeze selection without reading specz or any photo-z field.
    priority=np.array([hashlib.sha256(f'1029:{int(x)}'.encode()).hexdigest() for x in a['ID'][pick[use]]]);use=use[np.argsort(priority)[:2048]]
    rr=indices[pick[use]]
    np.savez_compressed(OUT/'T2_candidate_identity.npz',source_row=rr,spec_row=sidx[use],object_id=a['ID'][pick[use]],spec_id=s['Id_specz'][use],ra=a['RA'][pick[use]],dec=a['DEC'][pick[use]],separation_arcsec=sep[use,0],second_neighbor_arcsec=sep[use,1],survey=s['survey'][use],flag=s['flag'][use],confidence=s['Confidence_level'][use])
    audit=dict(local_photometry_rows=5474883,COSMOS_rectangle_rows=int(region.sum()),coordinate_rectangle_deg=[147,153,-1,5],spec_rows_quality=int(good.sum()),quality='flag=4 (no broad-line flag); confidence>=97; Priority=1',crossmatch='corrected spec J2000 positions; closest<0.5 arcsec; second photometric neighbor>1 arcsec; unique counterpart',candidates_before_cap=int(ok.sum()),candidate_count=len(use),cap=2048,old_256_excluded=True,all_historical_HSC_positions_excluded_radius_arcsec=1.,target_labels_read=False,spec_file_sha256=d.sha(specfile),actual_columns_read=['Id_specz','ra_corrected','dec_corrected','flag','Confidence_level','Priority','survey','GroupSize'],remaining='Exclude historical Legacy sampled membership by position before final lock')
    d.write(OUT/'target_candidate_audit.json',audit);print(audit,flush=True)
def legacy():
    import fitsio
    membership=d.npz(OUT/'legacy_membership_only.npz');ids=membership['targetid'];order=np.argsort(ids);sortedids=ids[order]
    raw={};found=np.zeros(len(ids),int)
    meta=json.loads(Path('/arc/home/gsm/aion_output/figures/qwen-desi-ls-qlora-specz-e10/prepared.json').read_text())
    for record in meta['sources'][1:]:
        with fitsio.FITS(record['path']) as f:
            allids=f[1].read(columns=['TARGETID'])['TARGETID'];p=np.searchsorted(sortedids,allids);ok=p<len(ids);ok[ok]&=sortedids[p[ok]]==allids[ok]
            rows=f[1].read(rows=np.flatnonzero(ok),columns=['TARGETID','RELEASE','BRICKID','OBJID','RA','DEC','MASKBITS']) if ok.any() else None
        if rows is None:continue
        dst=order[p[ok]];found[dst]+=1
        for key in rows.dtype.names:
            if key not in raw:raw[key]=np.empty(len(ids),dtype=rows[key].dtype)
            raw[key][dst]=rows[key]
        print('identity',Path(record['path']).name,len(rows),flush=True)
    assert np.all(found==1)
    groups=np.array([f'ls:{r}:{b}:{o}' for r,b,o in zip(raw['RELEASE'],raw['BRICKID'],raw['OBJID'])])
    np.savez_compressed(OUT/'legacy_all_membership_astrometry.npz',targetid=ids,group=groups,split=membership['split'],ra=raw['RA'],dec=raw['DEC'])
    a=d.npz(OUT/'legacy_exact_trainval.npz');keep=membership['split']!='test';assert np.array_equal(ids[keep],a['targetid'])
    rr=make_rows(a['targetid'],a['features'],['DECam.g','DECam.r','DECam.z','WISE.W1','WISE.W2'],a['z'],a['split'],raw['RA'][keep],raw['DEC'][keep],'Legacy',quality=np.repeat((raw['MASKBITS'][keep]!=0)[:,None],5,axis=1),quality_available=np.ones((keep.sum(),5),bool))
    for i,r in enumerate(raw['RELEASE'][keep]):rr['band_id'][i,:5]=[FILTERS.index(filter_key('desi-ls',b,int(r))) for b in ['g','r','z','W1','W2']]
    rr['group']=a['group'];np.savez_compressed(OUT/'legacy_trainval_unified.npz',**rr)
    print('Legacy unified ready',len(a['z']),flush=True)
def target_phot(identity,name):
    import fitsio
    suffix=['MegaCam-u','MegaCam-uS','HSC-G','HSC-R','HSC-I','HSC-Z','HSC-Y','VIRCAM-Y','VIRCAM-J','VIRCAM-H','VIRCAM-Ks']
    cols=['ID','RA','DEC']+[f'{p}_{s}' for s in suffix for p in ['FLUX_CMODEL','FLUXERR_CMODEL','notObserved','isNoData','hasBadPhotometry','isClean']]
    with fitsio.FITS(str(d.CAT)) as f:a=f[1].read(rows=identity['source_row'],columns=cols)
    assert np.array_equal(a['ID'],identity['object_id'])
    f=np.column_stack([a[f'FLUX_CMODEL_{s}'] for s in suffix]);e=np.column_stack([a[f'FLUXERR_CMODEL_{s}'] for s in suffix])
    notobs=np.column_stack([a[f'notObserved_{s}'] for s in suffix]);nodata=np.column_stack([a[f'isNoData_{s}'] for s in suffix])
    f,e,valid=clauds_flux_to_nanomaggy(f,e,not_observed=notobs,no_data=nodata)
    n=len(a);rows=dict(object_id=a['ID'].astype('U30'),group=np.array([f'clauds:{x}' for x in a['ID']]),flux=f,sigma=e,band_id=np.tile([FILTERS.index(filter_key('clauds',b)) for b in CLAUDS_BANDS],(n,1)),quality=np.column_stack([a[f'hasBadPhotometry_{s}'] for s in suffix]).astype(float),quality_available=np.ones_like(f,bool),ebv=np.full(n,np.nan),ebv_available=np.zeros(n,bool),ra=a['RA'],dec=a['DEC'],split=np.full(n,'target'),catalogue=np.full(n,'CLAUDS'))
    np.savez_compressed(OUT/f'{name}_photometry.npz',**rows);return rows
def finalize():
    from collections import Counter
    h=d.npz(ROOT/'reports/stage2a/1028/hsc_trainval.npz');l=d.npz(OUT/'legacy_trainval_unified.npz')
    member=d.npz(OUT/'legacy_all_membership_astrometry.npz')
    target=d.npz(OUT/'T2_candidate_identity.npz');sep,_=nearest(target,member)
    target=take(target,sep>1.)
    np.savez_compressed(OUT/'T2_identity_locked.npz',**target)
    t=target_phot(target,'T2')
    raw=d.npz(RAW/'tractor_matches.npz',('TARGETID','RELEASE','BRICKID','OBJID','RA','DEC'))
    hm=d.npz(RAW/'hdf_metadata.npz',('targetid','hsc_object_id','hsc_ra','hsc_dec'))
    order=np.argsort(raw['TARGETID']);pos=order[np.searchsorted(raw['TARGETID'][order],hm['targetid'])]
    raw={k:v[pos] for k,v in raw.items()};assert np.array_equal(raw['TARGETID'],hm['targetid'])
    ls=np.array([f'ls:{r}:{b}:{o}' for r,b,o in zip(raw['RELEASE'],raw['BRICKID'],raw['OBJID'])])
    hg=np.array([f'hsc:{o}' for o in hm['hsc_object_id']])
    dd=np.rad2deg(np.linalg.norm(xyz(raw['RA'],raw['DEC'])-xyz(hm['hsc_ra'],hm['hsc_dec']),axis=1))*3600
    valid=(dd<=1.)&(raw['RELEASE']>0)
    counts=Counter(ls[valid]);ambig=set(ls[valid][np.array([counts[g]>1 for g in ls[valid]])])
    mapping={str(hg[i]):str(ls[i]) for i in np.flatnonzero(valid) if ls[i] not in ambig}
    h['group']=np.array([mapping.get(str(g),str(g)) for g in h['group']],dtype='U80')
    test=set(member['group'][member['split']=='test'])
    test.update(ls[valid&np.array([group_fraction(g)>=.25 for g in hg])])
    valgroups=set(h['group'][h['split']=='val'])|set(l['group'][l['split']=='val'])
    # Different IDs within one arcsec are excluded as ambiguous, not silently joined.
    from scipy.spatial import cKDTree
    pooled={k:np.concatenate([h[k],l[k]]) for k in ['ra','dec','group']}
    close=cKDTree(xyz(pooled['ra'],pooled['dec'])).query_pairs(np.deg2rad(1/3600),output_type='ndarray')
    close=close[pooled['group'][close[:,0]]!=pooled['group'][close[:,1]]]
    ambiguous_spatial=set(pooled['group'][close.ravel()]);ambig.update(ambiguous_spatial)
    htest=np.array([group_fraction(g)>=.25 for g in hg])
    testsky=dict(ra=np.r_[hm['hsc_ra'][htest],member['ra'][member['split']=='test']],dec=np.r_[hm['hsc_dec'][htest],member['dec'][member['split']=='test']])
    sep_test,_=nearest(pooled,testsky);test.update(pooled['group'][sep_test<=1.])
    audit={'T2_final_count':len(t['object_id']),'T2_legacy_historical_membership_exclusions':int((sep<=1).sum()),'T2_historical_source_overlap_policy':'Exclude target matches to all HDF positions and all sampled Legacy train/val/test astrometry, without reading test photometry or evaluating test labels','source_ambiguous_Legacy_groups':len(ambig),'HSC_to_Legacy_mapping_count':len(mapping),'source_identity_match_max_arcsec':1.,'group_rule':'verified unique same TARGETID and <=1 arcsec map to Legacy imaging key; ambiguity removed; any historical test collision excluded; union validation wins'}
    audit.update(different_identity_close_pairs=len(close),ambiguous_spatial_groups=len(ambiguous_spatial),source_old_test_spatial_collision_rows=int((sep_test<=1).sum()))
    for name,a in [('HSC',h),('Legacy',l)]:
        original=len(a['z']);bad=np.isin(a['group'],list(test|ambig))
        # Independent spatial confirmation: quarantine the entire COSMOS rectangle.
        sky=(a['ra']>147)&(a['ra']<153)&(a['dec']>-1)&(a['dec']<5)
        a=take(a,~bad&~sky)
        a['split'][np.isin(a['group'],list(valgroups))]='val'
        audit[name]=dict(original_trainval=original,excluded_test_or_ambiguous=int(bad.sum()),excluded_spatial_COSMOS=int((sky&~bad).sum()),train=int((a['split']=='train').sum()),validation=int((a['split']=='val').sum()))
        assert not set(a['group'][a['split']=='train'])&set(a['group'][a['split']=='val'])
        np.savez_compressed(OUT/f'{name}_eligible.npz',**a)
        if name=='HSC':h=a
        else:l=a
    mixed={k:np.concatenate([h[k],l[k]]) for k in h}
    # One deterministic observed view per shared galaxy prevents repeated-view weighting.
    order=np.lexsort((mixed['catalogue'],mixed['group']));keep=[]
    gs=mixed['group'][order];starts=np.r_[0,np.flatnonzero(gs[1:]!=gs[:-1])+1,len(gs)]
    for lo,hi in zip(starts[:-1],starts[1:]):
        block=order[lo:hi];keep.append(block[int(hashlib.sha256(gs[lo].encode()).hexdigest()[:8],16)%len(block)])
    m=take(mixed,np.array(keep));assert len(set(m['group']))==len(m['z'])
    np.savez_compressed(OUT/'Mixed_eligible.npz',**m)
    audit['mixed_unique_galaxies']=len(m['z']);audit['mixed_shared_galaxies']=int(len(mixed['z'])-len(m['z']));audit['mixed_view_rule']='One deterministic SHA256 chosen real view per galaxy; balanced survey sampling; never synthesize joint apertures'
    d.write(OUT/'identity_split_audit.json',audit)
    bins=[0,.5,1,1.5,2,3,4.5];coverage=[];stats=[]
    for name,a in [('HSC',h),('Legacy',l),('Mixed',m)]:
        for split in ['train','val']:
            aa=take(a,a['split']==split);z=aa['z'];counts,_=np.histogram(z,bins)
            for j,n in enumerate(counts):coverage.append(dict(cohort=name,split=split,zlo=bins[j],zhi=bins[j+1],n=int(n),fraction=float(n/len(z))))
            valid=np.isfinite(aa['flux'])&np.isfinite(aa['sigma'])&(aa['sigma']>0)&(aa['band_id']>=0)
            for bid in np.unique(aa['band_id'][valid]):
                mm=valid&(aa['band_id']==bid);ff=aa['flux'][mm];ee=aa['sigma'][mm];zz=np.broadcast_to(z[:,None],mm.shape)[mm]
                for lo,hi in zip(bins[:-1],bins[1:]):
                    takez=(zz>=lo)&(zz<hi);v=ff[takez];er=ee[takez]
                    if not len(v):continue
                    positive=v[v>0];stats.append(dict(cohort=name,split=split,filter=FILTERS[bid],zlo=lo,zhi=hi,n=len(v),flux_q10_q50_q90=np.quantile(v,[.1,.5,.9]).tolist(),sigma_q10_q50_q90=np.quantile(er,[.1,.5,.9]).tolist(),sn_q10_q50_q90=np.quantile(np.abs(v/er),[.1,.5,.9]).tolist(),magnitude_positive_q10_q50_q90=np.quantile(22.5-2.5*np.log10(positive),[.1,.5,.9]).tolist() if len(positive) else [],negative_fraction=float(np.mean(v<0))))
    d.table(OUT/'source_redshift_coverage.csv',coverage);d.write(OUT/'source_joint_photometry_coverage.json',stats)
    hashes={p.name:d.sha(p) for p in OUT.glob('*.npz')}
    d.write(OUT/'dataset_hashes.json',hashes)
    print('STAGE3 DATA READY',audit,flush=True)
def background():
    import fitsio
    rng=np.random.default_rng(1029);rr=np.sort(rng.choice(5474883,12000,replace=False))
    with fitsio.FITS(str(d.CAT)) as f:a=f[1].read(rows=rr,columns=['ID','RA','DEC','isStar','isParent_HSC-I','isDuplicated_HSC-I','isSky_HSC-I'])
    good=~a['isStar'].astype(bool)&~a['isParent_HSC-I'].astype(bool)&~a['isDuplicated_HSC-I'].astype(bool)&~a['isSky_HSC-I'].astype(bool)
    idx=np.flatnonzero(good)[:4096];ident=dict(source_row=rr[idx],object_id=a['ID'][idx]);assert len(idx)==4096
    np.savez_compressed(OUT/'unlabeled_population_identity.npz',**ident);target_phot(ident,'unlabeled_population')
    d.write(OUT/'unlabeled_population_definition.json',dict(random_seed=1029,sampling='Uniform without replacement12000 row candidates then first4096 nonstellar, nonparent, nonduplicated, nonsky I-band rows; independent of flux/z',labels_read=False,population='Local COSMOS catalogue; not all CLAUDS fields'))
if __name__=='__main__':globals()[sys.argv[1]]()
