"""Identity-first downstream cohorts; seal query labels for the new readout study."""
import json
import numpy as np
import fitsio
from anecdote_common_1029 import *
from aion_magnitude.filter_photometry import CLAUDS_BANDS,filter_key,clauds_flux_to_nanomaggy
def exposure(a,bs,bc):
    result={};known=np.zeros(len(a['object_id']),bool);uncertain=known.copy();validation=known.copy()
    for family,member in [('BS',bs),('BC',bc)]:
        for role in ['train','val']:
            b=take(member,member['historical_role']==role);sep,ix=nearest(a,b,k=2);exact=np.isin(a['object_id'],b['object_id']);matched=exact|((sep[:,0]<=1)&(sep[:,1]>1));ambig=(~exact)&(((sep[:,0]>1)&(sep[:,0]<=2))|(sep[:,1]<=1))
            result[f'{family}_{role}_match']=matched;result[f'{family}_{role}_nearest_arcsec']=sep[:,0];result[f'{family}_{role}_ambiguous']=ambig
            if role=='train':known|=matched
            else:validation|=matched
            uncertain|=ambig
    category=np.where(known,'historical_training_exposed',np.where(uncertain,'uncertain_exposure',np.where(validation,'historical_validation_only','clean_no_train_or_val')))
    result['exposure_category']=category
    return result
def clauds_phot(rr,expected):
    suffix=['MegaCam-u','MegaCam-uS','HSC-G','HSC-R','HSC-I','HSC-Z','HSC-Y','VIRCAM-Y','VIRCAM-J','VIRCAM-H','VIRCAM-Ks'];prefix=['FLUX_CMODEL','FLUXERR_CMODEL','notObserved','isNoData','hasBadPhotometry'];flags=['isStar','isParent_HSC-I','isDuplicated_HSC-I','isSky_HSC-I']
    with fitsio.FITS(str(d.CAT)) as f:a=f[1].read(rows=rr,columns=['ID','RA','DEC',*flags,*[f'{p}_{s}' for s in suffix for p in prefix]])
    assert np.array_equal(ids(a['ID']),expected)
    ff=np.column_stack([a[f'FLUX_CMODEL_{s}'] for s in suffix]);ee=np.column_stack([a[f'FLUXERR_CMODEL_{s}'] for s in suffix]);no=np.column_stack([a[f'notObserved_{s}'] for s in suffix]);nd=np.column_stack([a[f'isNoData_{s}'] for s in suffix]);ff,ee,_=clauds_flux_to_nanomaggy(ff,ee,not_observed=no,no_data=nd)
    features=np.column_stack([x for j in range(11) for x in [ff[:,j],ee[:,j]]]+[np.full(len(ff),np.nan)])
    rows=make_rows(expected,features,[filter_key('clauds',b) for b in CLAUDS_BANDS],np.zeros(len(ff)),np.full(len(ff),'unassigned'),a['RA'],a['DEC'],'CLAUDS',quality=np.column_stack([a[f'hasBadPhotometry_{s}'] for s in suffix]).astype(float),quality_available=np.ones_like(ff,bool));rows['group']=np.array([f'clauds:{x}' for x in expected])
    good=~np.any(np.column_stack([a[c].astype(bool) for c in flags]),axis=1)
    return rows,good
def partition(task,a,z,category):
    from scipy.spatial import cKDTree
    excluded_uncertain=int((category=='uncertain_exposure').sum());excluded_validation=int((category=='historical_validation_only').sum())
    valid=np.isfinite(a['flux'])&np.isfinite(a['sigma'])&(a['sigma']>0)&(a['band_id']>=0)
    labelok=np.isfinite(z)&(z>=0)&(z<=4.5);labelok&=z>0 if task=='C_SPEC' else True
    eligible=labelok&(valid.sum(1)>=3)
    # Exclude every selected close/blended identity, rather than split a pair.
    close=cKDTree(xyz(a['ra'],a['dec'])).query_pairs(np.deg2rad(1/3600),output_type='ndarray');bad=np.unique(close.ravel());eligible[bad]=False
    clean=np.flatnonzero(eligible&(category=='clean_no_train_or_val'));clean=clean[order(a['object_id'][clean],task+':clean')]
    required=2304 if task!='C_SPEC' else min(len(clean),2048)
    assert required>=1536,(task,len(clean))
    clean=clean[:required];diag=np.flatnonzero(eligible&(category=='historical_training_exposed'));diag=diag[order(a['object_id'][diag],task+':exposure')[:128]]
    use=np.r_[clean,diag];a=take(a,use);z=np.asarray(z[use],np.float32);category=category[use]
    nq=1024 if task!='C_SPEC' else 768;nd=256;ns=len(clean)-nq-nd;assert ns>=512
    role=np.r_[np.full(ns,'support_pool'),np.full(nd,'development'),np.full(nq,'query'),np.full(len(diag),'exposure_query')];a['split']=role;a['exposure_category']=category
    arrays={k:v for k,v in a.items() if k!='z'};np.savez_compressed(OUT/f'{task}_photometry.npz',**arrays)
    supervised=np.isin(role,['support_pool','development']);np.savez_compressed(OUT/f'{task}_support_development_labels.npz',object_id=a['object_id'][supervised],z=z[supervised],split=role[supervised]);np.savez_compressed(OUT/f'{task}_query_labels_sealed.npz',object_id=a['object_id'][~supervised],z=z[~supervised],split=role[~supervised])
    seeds={}
    for seed in [101,202,303]:
        support=np.flatnonzero(role=='support_pool');support=support[np.random.default_rng(seed).permutation(len(support))];seeds[str(seed)]={str(k):a['object_id'][support[:k]].tolist() for k in [16,32,64,128,256,512]}
    d.write(OUT/f'{task}_nested_subsets.json',seeds)
    audit=dict(task=task,clean=len(clean),support_pool=ns,development=nd,query=nq,exposure_query=len(diag),uncertain_exposure_excluded=excluded_uncertain,validation_only_excluded=excluded_validation,range='0<=z<=4.5, positive spec-z; never clipped',close_pairs_excluded=len(close),outside_or_invalid_label_candidates=int((~labelok).sum()),minimum_full_bands=3,identical_galaxies_for_all_backbones=True,query_labels_sealed=True,interpretation='Exploratory support/query reuse of Stage3 T2; previously inspected, not new confirmation' if task=='C_SPEC' else 'Genuine off-diagonal transfer comparison; historical catalog evaluation membership retained; no historically trained objects in primary support/dev/query',source_role='Historical HSC test membership for the entire clean downstream dataset; new head-only support/dev/query roles, no changes to old partitions' if task=='GMS' else 'Historical BC test identities for clean ZPHOT experiment' if task=='C_ZPHOT' else 'New exploratory split of old T2',subset_seeds=[101,202,303])
    d.write(OUT/f'{task}_cohort_manifest.json',audit);print('COHORT',audit,flush=True)
def run():
    assert (OUT/'checkpoint_provenance.json').exists();assert not (OUT/'data_ready.json').exists()
    bs=d.npz(OUT/'BS_historical_identity.npz');bc=d.npz(OUT/'BC_historical_identity.npz');audits={}
    # Entire historical HSC test membership is eligible for new head-only roles.
    g=take(bs,bs['historical_role']=='test');diag=take(bs,bs['historical_role']=='train');diag=take(diag,order(diag['object_id'],'GMS:diagnostic')[:256]);g={k:np.concatenate([g[k],diag[k]]) for k in g};exp=exposure(g,bs,bc);np.savez_compressed(OUT/'GMS_full_exposure_audit.npz',object_id=g['object_id'],ra=g['ra'],dec=g['dec'],historical_role=g['historical_role'],**exp)
    choose=np.r_[np.flatnonzero(exp['exposure_category']=='clean_no_train_or_val')[order(g['object_id'][exp['exposure_category']=='clean_no_train_or_val'],'GMS:candidates')[:2800]],np.flatnonzero(exp['exposure_category']=='historical_training_exposed')[:256]];g=take(g,choose);category=exp['exposure_category'][choose]
    raw=d.npz(RAW/'hdf_metadata.npz',('targetid','hsc_features','z_hdf','hsc_object_id'));ix=g['source_row'];assert np.array_equal(ids(raw['targetid'][ix]),g['object_id']);a=make_rows(g['object_id'],raw['hsc_features'][ix],[f'HSC.{b}' for b in 'grizy'],raw['z_hdf'][ix],np.full(len(ix),'unassigned'),g['ra'],g['dec'],'HSC');a['group']=np.array([f'hsc:{x}' for x in raw['hsc_object_id'][ix]]);a['historical_role']=g['historical_role'];partition('GMS',a,raw['z_hdf'][ix],category)
    # T2 was already inspected in Stage3: make a new exploratory split, not a new confirmation.
    truth=d.npz(ROOT/'reports/stage3b/1029/T2_truth_eligible.npz');phot=d.npz(ROOT/'reports/stage3a/1029/T2_photometry.npz');index={x:i for i,x in enumerate(phot['object_id'])};a=take(phot,np.array([index[x] for x in truth['object_id']]));exp=exposure(a,bs,bc);np.savez_compressed(OUT/'C_SPEC_full_exposure_audit.npz',object_id=a['object_id'],ra=a['ra'],dec=a['dec'],**exp);partition('C_SPEC',a,truth['z'],exp['exposure_category'])
    # Clean ZPHOT identities come from BC's original test, disjoint from T2.
    c=take(bc,bc['historical_role']=='test');exp=exposure(c,bs,bc);np.savez_compressed(OUT/'C_ZPHOT_full_exposure_audit.npz',object_id=c['object_id'],ra=c['ra'],dec=c['dec'],historical_role=c['historical_role'],**exp)
    gms=d.npz(OUT/'GMS_photometry.npz');sep_gms,_=nearest(c,gms)
    clean=np.flatnonzero((exp['exposure_category']=='clean_no_train_or_val')&~np.isin(c['object_id'],truth['object_id'])&(sep_gms>2.));clean=clean[order(c['object_id'][clean],'C_ZPHOT:candidates')[:8192]];c=take(c,clean)
    diag=take(bc,bc['historical_role']=='train');diag=take(diag,order(diag['object_id'],'C_ZPHOT:diagnostic')[:512]);c={k:np.concatenate([c[k],diag[k]]) for k in c};exp=exposure(c,bs,bc);a,good=clauds_phot(c['source_row'],c['object_id'])
    with fitsio.FITS(str(OLD_CAT)) as f:z=f[1].read(rows=c['source_row'],columns=['ID','ZPHOT']);assert np.array_equal(ids(z['ID']),c['object_id'])
    a=take(a,good);a['historical_role']=c['historical_role'][good];partition('C_ZPHOT',a,z['ZPHOT'][good],exp['exposure_category'][good])
    for task in TASKS:
        exp=d.npz(OUT/f'{task}_full_exposure_audit.npz');audits[task]={str(c):int((exp['exposure_category']==c).sum()) for c in np.unique(exp['exposure_category'])}
    d.write(OUT/'historical_exposure_summary.json',audits)
    d.write(OUT/'data_ready.json',dict(ready=True,all_three_backbones_same_primary_galaxies=True,no_historical_train_or_validation_exposure_in_primary_cohorts=True,Paper='II CLAUDS frozen-readout branch',not_a_Stage2C_or_physics_kernel_experiment=True,task_definitions={task:json.loads((OUT/f'{task}_cohort_manifest.json').read_text()) for task in TASKS}))
if __name__=='__main__':run()
