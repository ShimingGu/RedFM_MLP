#!/usr/bin/env python3
"""Frozen, same-galaxy HSC/Legacy benchmark. Prepare/audit before inference."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'notebooks'))
import numpy as np
from scipy.spatial import cKDTree

RUNS = Path('/arc/home/gsm/aion_output/figures')
WARM = ROOT / 'filter-aware-runs/1023/hsc-warmstart'
MODELS = ('original-qlora', 'original-dora', 'A', 'B', 'C')
SEED = 1027


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def unit_vectors(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))


def separation(ra, dec, ra2, dec2):
    d = np.linalg.norm(unit_vectors(ra, dec)-unit_vectors(ra2, dec2), axis=1)
    return np.rad2deg(2*np.arcsin(np.minimum(d/2, 1))) * 3600


def nearby(ra, dec, source_ra, source_dec, radius=1.):
    """Spherical positional exposure, including alternate catalogue IDs."""
    if not len(source_ra):
        return np.zeros(len(ra), bool)
    distances = cKDTree(unit_vectors(source_ra, source_dec)).query(unit_vectors(ra, dec), k=1)[0]
    return distances <= 2*np.sin(np.deg2rad(radius/3600)/2)


def unique_mask(values):
    _, inverse, count = np.unique(values, return_inverse=True, return_counts=True)
    return count[inverse] == 1


def categories(train, val):
    return np.where(train, 'T', np.where(val, 'V', 'P'))


def fixed_bins(z, minimum=40):
    """Pre-prediction adjacent merges, left to right; merge a small tail backward."""
    initial = np.array([0., .2, .4, .6, 1., 2., 4.5])
    bins = np.minimum(np.searchsorted(initial, z, side='right')-1, len(initial)-2)
    counts = np.bincount(bins, minlength=len(initial)-1)
    edges = [initial[0]]; accumulated = 0
    for i, n in enumerate(counts):
        accumulated += int(n)
        if accumulated >= minimum:
            edges.append(initial[i+1]); accumulated = 0
    if edges[-1] != initial[-1]:
        if len(edges) > 1:
            edges[-1] = initial[-1]
        else:
            edges.append(initial[-1])
    counts2, _ = np.histogram(z, bins=edges)
    return dict(initial_edges=initial.tolist(), initial_counts=counts.tolist(), edges=list(map(float, edges)),
                counts=counts2.tolist(), minimum=minimum,
                merge_rule='Accumulate adjacent bins left-to-right until >=40; merge final small tail backward',
                standardization='Equal weight for each nonempty merged spec-z bin; macro-average within-bin metrics, including NMAD (not pooled weighted NMAD)')


def support_values(features, indices, scale):
    f = features[:, np.array(indices)*2].astype(float)
    e = features[:, np.array(indices)*2+1].astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        a = np.arcsinh(f/scale)
        return np.column_stack((a, np.log10(e), np.arcsinh(f/e), a[:, 0]-a[:, 1], a[:, 1]-a[:, 2]))


def fit_support(train_features, hsc, legacy):
    scale = np.nanmedian(train_features[:, [1, 3, 7]], axis=0)
    x = support_values(train_features, [0, 1, 3], scale)
    low, high = np.nanquantile(x, [.005, .995], axis=0)
    def inside(f, idx):
        v = support_values(f, idx, scale)
        return np.all(np.isfinite(v) & (v >= low) & (v <= high), axis=1)
    a, b = inside(hsc, [0, 1, 3]), inside(legacy, [0, 1, 2])
    return a & b, dict(fit_rows=len(train_features), fit='HSC source training photometry only; no target labels',
        rule='Both HSC and Legacy must have valid common g/r/z and all 11 transformed coordinates within HSC training marginal 0.5–99.5 percentiles',
        coordinates=['asinh_flux_g','asinh_flux_r','asinh_flux_z','log10_sigma_g','log10_sigma_r','log10_sigma_z',
                     'asinh_snr_g','asinh_snr_r','asinh_snr_z','asinh_g_minus_r','asinh_r_minus_z'],
        scale=scale.tolist(), lower=low.tolist(), upper=high.tolist(),
        hsc_inside=int(a.sum()), legacy_inside=int(b.sum()), both_inside=int((a & b).sum()),
        caveat='Simple marginal optical-support diagnostic, not a density estimate or bandpass matching; strict joint cuts may leave few objects')


def inventory(out):
    import torch
    source = read(WARM/'level0/source_cohort.npz')
    models = {}; reference = None; protected = {}
    for name in MODELS:
        if name.startswith('original'):
            method = name.split('-')[1]
            folder = RUNS/f'qwen-galaxiesml-spec-{method}-specz-e10'
            meta = json.loads((folder/'prepared.json').read_text())
            result = torch.load(folder/method/'result.pt', map_location='cpu', weights_only=False, mmap=True)
            config = result['metadata']['config']; ev = result['val_evaluation']
            test = np.asarray(result['test_evaluation']['object_id'], dtype=np.int64)
            selected = folder/method
            selection = dict(rule='Minimum source validation CE across ten epochs',
                             epoch=min(result['history'], key=lambda x:x['val_cross_entropy'])['epoch'])
            key = meta['cache_key']
        else:
            level = ('A','B','C').index(name);folder = WARM/f'level{level}'
            m = json.loads((folder/'manifest.json').read_text())
            s = json.loads((folder/'summary.json').read_text())
            assert s['phase'] == 'complete'
            ev = torch.load(folder/'best_validation.pt', map_location='cpu', weights_only=False)
            other = read(folder/'source_cohort.npz')
            for k in source:
                np.testing.assert_equal(source[k], other[k])
            config=m['source_config'];selected=folder/'selected';test=reference['test']
            selection=dict(rule='Minimum HSC source validation CE at update 0, every 250 updates, and epoch ends',
                           update=s['best_update'], training_updates=s['global_update'])
            key=reference['key']
        val=np.asarray(ev['object_id'], dtype=np.int64)
        train=source['targetid'][source['split']=='train']
        np.testing.assert_array_equal(np.sort(val), np.sort(source['targetid'][source['split']=='val']))
        assert not np.isin(train,val).any() and not np.isin(test,np.r_[train,val]).any()
        if reference is None:
            reference=dict(val=val, test=test, key=key, edges=ev['redshift_edges'].numpy(), centers=ev['redshift_centers'].numpy())
        else:
            assert key == reference['key']
            np.testing.assert_array_equal(val, reference['val']);np.testing.assert_array_equal(test,reference['test'])
            np.testing.assert_array_equal(ev['redshift_edges'].numpy(),reference['edges'])
            np.testing.assert_array_equal(ev['redshift_centers'].numpy(),reference['centers'])
        files=[selected/'photoz_head.pt', *sorted((selected/'adapter').glob('*'))]
        hashes={str(p):sha(p) for p in files if p.is_file()};protected.update(hashes)
        np.savez_compressed(out/f'{name}_exposure_ids.npz',train=train,val=val,test=test)
        np.savez_compressed(out/f'{name}_replay_reference.npz',targetid=val[:32],
            logits=ev['logits'][:32].numpy(),z=ev['z_spec'][:32].numpy())
        models[name]=dict(directory=str(selected), source_run=str(folder), catalogue='galaxiesml-spec',
            level=0 if name.startswith('original') else ('A','B','C').index(name), config=config,
            cohort_cache_key=key, split_counts=dict(train=len(train),val=len(val),test=len(test)),
            exposure_evidence='Persisted exact warm source train/val IDs; exact archived original validation/test IDs; identical original preparation cache key and split counts; all warm source arrays verified equal',
            selection=selection, sha256=hashes,
            serializer='Exact native HSC/Legacy schema for level 0; physical filter identities for level 1; verified registry pivot/equivalent width for level 2',
            units='AB nanomaggy flux, one-sigma flux error, EBV as catalogued; no extra MW correction')
    absent={}
    for method in ('qlora','dora'):
        folder=RUNS/f'qwen-desi-ls-{method}-specz-e10'
        m=json.loads((folder/'prepared.json').read_text())
        absent[method]=dict(run=str(folder), included=False, completed_result=(folder/method/'result.pt').exists(),
            completed_head=(folder/method/'photoz_head.pt').exists(), completed_adapter=(folder/method/'adapter/adapter_model.safetensors').exists(),
            source_split_counts=m['split_counts'], source_cache_key=m['cache_key'],
            exposure='Exact sampled Legacy source IDs unavailable in removed cohort cache; incomplete models excluded, never presumed unexposed')
        assert not absent[method]['completed_result']
    return source,models,protected,absent,reference


def prepare(args):
    from aion_magnitude.survey_catalogues import flux_features
    from aion_magnitude.filter_photometry import serialize_rows, feature_columns, load_registry, DEFAULT_REGISTRY
    from transformers import AutoTokenizer
    out=args.output_dir
    if (out/'manifest.json').exists():
        raise FileExistsError('Prepared benchmark is immutable; use a new output directory')
    raw=out/'raw-audit';raw_audit=json.loads((raw/'raw_audit.json').read_text())
    h=read(raw/'hdf_metadata.npz');p=read(raw/'tractor_matches.npz');s=read(raw/'redshift_matches.npz')
    source,models,protected,absent,ref=inventory(out)
    # All cuts here precede prediction. Preserve counts at every cumulative cut.
    counts=[];mask=np.ones(len(h['targetid']),bool)
    def cut(name, criterion):
        nonlocal mask
        before=int(mask.sum());mask &= criterion
        counts.append(dict(step=name,before=before,removed=before-int(mask.sum()),remaining=int(mask.sum())))
    def align(table, condition):
        indices=np.flatnonzero(condition);order=np.argsort(table['TARGETID'][indices]);indices=indices[order]
        ids=table['TARGETID'][indices];pos=np.searchsorted(ids,h['targetid'])
        valid=pos<len(ids);valid[valid] &= ids[pos[valid]]==h['targetid'][valid]
        full=np.zeros(len(mask),np.int64);full[valid]=indices[pos[valid]]
        return full,valid
    pi,present=align(p,unique_mask(p['TARGETID']))
    cut('Exactly one original Tractor record per exact TARGETID',present)
    text=lambda k:np.char.strip(s[k].astype('U'))
    quality=(s['ZCAT_PRIMARY']!=0)&(text('SPECTYPE')=='GALAXY')&(text('OBJTYPE')=='TGT')&(s['ZWARN']==0)&(s['COADD_FIBERSTATUS']==0)&np.isfinite(s['DELTACHI2'])&(s['DELTACHI2']>40)&np.isfinite(s['Z'])&(s['Z']>=0)&(s['Z']<=4.5)&np.isfinite(s['ZERR'])&(s['ZERR']>=0)
    # Uniqueness is required among primary quality rows, not among all spectral reductions.
    qindex=np.flatnonzero(quality);q_unique=np.zeros(len(quality),bool);q_unique[qindex]=unique_mask(s['TARGETID'][qindex])
    zi,zpresent=align(s,quality&q_unique)
    cut('Unique primary DESI Iron spectrum passing original uniform galaxy quality and 0<=z<=4.5',zpresent)
    cut('Original Legacy survey restriction: main bright/dark (intersection of source protocols)',(text('SURVEY')[zi]=='main')&np.isin(text('PROGRAM')[zi],['bright','dark']))
    native=np.isfinite(h['z_hdf'])&(h['z_hdf']>=0)&(h['z_hdf']<=4.5)&np.isfinite(h['zerr_hdf'])&(h['zerr_hdf']>=0)
    cut('Valid native HDF redshift and error',native)
    dz=np.abs(h['z_hdf']-s['Z'][zi]);relative=dz/(1+np.abs(s['Z'][zi]))
    label_audit=dict(rows_checked=int(mask.sum()),exact_equal=int(((dz==0)&mask).sum()),
        numerical_agreement_1e6=int(((relative<=1e-6)&mask).sum()),
        original_tolerance_1e3=int(((relative<=.001)&mask).sum()),
        max_absolute_difference=float(dz[mask].max()) if mask.any() else None,
        hdf_reduction='Not encoded in HDF attributes or saved publisher provenance; identical reduction cannot be asserted',
        tractor_label_reduction='DESI DR1 Iron zall-pix-iron.fits',
        rule='Conservative pre-prediction agreement <=1e-6*(1+abs(z_Iron)); exclude ambiguous differences, do not adjudicate by prediction error; retain original HSC native HDF labels for all cells')
    np.savez_compressed(out/'label_discrepancies.npz',targetid=h['targetid'][mask],z_hdf=h['z_hdf'][mask],z_iron=s['Z'][zi][mask],relative_difference=relative[mask])
    cut('HDF/Iron label agreement to predeclared 1e-6 relative tolerance',relative<=1e-6)
    legacy_pairs=[flux_features(p['FLUX_'+b][pi],p['FLUX_IVAR_'+b][pi],p['NOBS_'+b][pi]) for b in ('G','R','Z','W1','W2')]
    lf=np.column_stack([x for pair in legacy_pairs for x in pair[:2]]+[p['EBV'][pi]]).astype(np.float32)
    lv=np.column_stack([x[2] for x in legacy_pairs]);hf=h['hsc_features']
    cut('Original HSC minimum three valid bands',h['hsc_valid'].sum(axis=1)>=3)
    cut('Original Legacy minimum three valid bands, with positive IVAR and NOBS',lv.sum(axis=1)>=3)
    cut('Verified DR9 filter provenance RELEASE in 9010/9011/9012',np.isin(p['RELEASE'][pi],[9010,9011,9012]))
    finite=np.isfinite(h['hsc_ra'])&np.isfinite(h['hsc_dec'])&np.isfinite(p['RA'][pi])&np.isfinite(p['DEC'][pi])
    sep=separation(h['hsc_ra'],h['hsc_dec'],p['RA'][pi],p['DEC'][pi])
    cut('Exact-ID pair positional verification within fixed one arcsecond',finite&(sep<=1.))
    ls_groups=np.asarray([f'ls:{a}:{b}:{c}' for a,b,c in zip(p['RELEASE'][pi],p['BRICKID'][pi],p['OBJID'][pi])])
    cut('Unique HSC imaging identity across raw HDF rows',unique_mask(h['hsc_object_id']))
    # Reject all ambiguous imaging keys among remaining pairs rather than choose by label.
    ix=np.flatnonzero(mask);unambiguous=np.zeros(len(mask),bool);unambiguous[ix]=unique_mask(ls_groups[ix])
    cut('Unique Legacy imaging key among remaining pairs',unambiguous)
    # Reject close pairs too: ambiguous physical associations cannot be independent bootstrap units.
    ix=np.flatnonzero(mask)
    if len(ix)>1:
        near=cKDTree(unit_vectors(h['hsc_ra'][ix],h['hsc_dec'][ix])).query_pairs(2*np.sin(np.deg2rad(1/3600)/2),output_type='ndarray')
        bad=np.unique(near) if near.size else np.array([],int)
        unambiguous=np.ones(len(mask),bool);unambiguous[ix[bad]]=False
        cut('Reject both members of ambiguous sub-arcsecond paired sky neighbours',unambiguous)
    ix=np.flatnonzero(mask)
    assert len(ix)>0,'No eligible cleanly paired galaxies; report audit without inference'
    # Exposure ORs exact TARGETID, same HSC imaging identity and spherical coordinate matches.
    id_to_row={int(t):i for i,t in enumerate(h['targetid'])}
    membership={};train_any=np.zeros(len(ix),bool);val_any=train_any.copy()
    for name in MODELS:
        ids=read(out/f'{name}_exposure_ids.npz')
        for split in ('train','val','test'):
            rows=np.array([id_to_row[int(t)] for t in ids[split]],int)
            exact=np.isin(h['targetid'][ix],ids[split]);group=np.isin(h['hsc_object_id'][ix],h['hsc_object_id'][rows])
            sky=nearby(h['hsc_ra'][ix],h['hsc_dec'][ix],h['hsc_ra'][rows],h['hsc_dec'][rows])
            sky |= nearby(p['RA'][pi][ix],p['DEC'][pi][ix],h['hsc_ra'][rows],h['hsc_dec'][rows])
            # DESI targeting coordinates are an additional independent coordinate representation.
            sky |= nearby(h['hsc_ra'][ix],h['hsc_dec'][ix],h['target_ra'][rows],h['target_dec'][rows])
            membership[f'{name}_{split}_exact']=exact;membership[f'{name}_{split}_group']=group
            membership[f'{name}_{split}_positional']=sky;membership[f'{name}_{split}']=exact|group|sky
        train_any |= membership[f'{name}_train'];val_any |= membership[f'{name}_val']
    category=categories(train_any,val_any)
    support,support_audit=fit_support(source['features'][source['split']=='train'],hf[ix],lf[ix])
    all_data=dict(targetid=h['targetid'][ix],galaxy_id=np.array([f'hsc:{x}' for x in h['hsc_object_id'][ix]]),
        ra=h['hsc_ra'][ix],dec=h['hsc_dec'][ix],legacy_ra=p['RA'][pi][ix],legacy_dec=p['DEC'][pi][ix],
        pair_separation_arcsec=sep[ix],z=h['z_hdf'][ix].astype(np.float32),z_iron=s['Z'][zi][ix],zerr=h['zerr_hdf'][ix],
        hsc_features=hf[ix],legacy_features=lf[ix],hsc_valid=h['hsc_valid'][ix],legacy_valid=lv[ix],
        legacy_group=ls_groups[ix],legacy_release=p['RELEASE'][pi][ix],category=category,common_support=support,**membership)
    np.savez_compressed(out/'eligible_pairs.npz',**all_data)
    # Sampling, if requested, is ID-hash-only and performed after the complete exposure audit.
    selected=[]
    for cat,maximum in [('P',args.n_primary),('V',args.n_validation)]:
        candidates=np.flatnonzero(category==cat)
        order=sorted(candidates,key=lambda i:hashlib.sha256(f'{SEED}:{all_data["galaxy_id"][i]}'.encode()).digest())
        selected.extend(order if maximum==0 else order[:maximum])
    selected=np.asarray(selected,int);d={k:v[selected] for k,v in all_data.items()}
    np.savez_compressed(out/'cohort.npz',**d)
    bin_policy={cat:fixed_bins(d['z'][d['category']==cat]) for cat in ('P','V') if (d['category']==cat).any()}
    measurement={}
    for j,b in enumerate(('G','R','Z','W1','W2')):
        a=h['hdf_legacy_features'][ix,2*j];bflux=p['FLUX_'+b][pi][ix]
        raw_ivar=h['hdf_legacy_features'][ix,2*j+1];ivar=p['FLUX_IVAR_'+b][pi][ix]
        good=np.isfinite(a)&np.isfinite(bflux)
        measurement[b]=dict(flux_equal=int((a[good]==bflux[good]).sum()),rows=int(good.sum()),
            median_abs_flux_difference=float(np.median(np.abs(a[good]-bflux[good]))),
            p95_abs_flux_difference=float(np.quantile(np.abs(a[good]-bflux[good]),.95)),
            ivar_equal=int((raw_ivar==ivar).sum()),nobs_missing_in_hdf=True,
            legacy_valid=int(lv[ix,j].sum()),legacy_nonpositive_valid_flux=int((lv[ix,j]&(bflux<=0)).sum()))
    measurement['EBV']=dict(equal=int((h['hdf_legacy_features'][ix,-1]==p['EBV'][pi][ix]).sum()),rows=len(ix))
    measurement['imaging_key_equal']=int(((h['hdf_release'][ix]==p['RELEASE'][pi][ix])&(h['hdf_brickid'][ix]==p['BRICKID'][pi][ix])&(h['hdf_objid'][ix]==p['OBJID'][pi][ix])).sum())
    audit=dict(created_utc=now(),cutflow=counts,label_audit=label_audit,raw_audit=raw_audit,
        category_counts={c:int((category==c).sum()) for c in ('P','V','T')},
        selected_counts={c:int((d['category']==c).sum()) for c in ('P','V','T')},
        membership_counts={k:int(v.sum()) for k,v in membership.items()},
        measurement_comparison=measurement,support=support_audit,
        release_counts={str(x):int((all_data['legacy_release']==x).sum()) for x in np.unique(all_data['legacy_release'])},
        scope='P is independent of training and validation of the five completed HSC models only. Original held-out HSC test objects are allowed and explicitly flagged. Legacy incomplete models excluded; mutual HSC/Legacy-source independence not established.',
        pair_radius_arcsec=1.,missingness='No fabricated measurements or errors; native source preprocessing with minimum 3/5 valid bands',
        photometry='HSC PDR2 cmodel AB magnitudes converted with original Jacobian; original DR9 Tractor signed fluxes, positive IVAR/NOBS; each uses its own EBV; no extra extinction correction')
    write(out/'audit.json',audit)
    policy=dict(created_utc=now(),inference_started=False,request='notebooks/filter_aware_fallback.md, explicitly requested by user',
        models=list(MODELS),excluded_models=absent,categories='P primary; V exploratory; T never evaluated',
        sample_rule=f'Ascending SHA256({SEED}:hsc_object_id); no z or predictions in sampling; n=0 means entire category',
        n_primary_limit=args.n_primary,n_validation_limit=args.n_validation,selected_counts=audit['selected_counts'],
        unchanged_selection='Checkpoints selected on archived HSC validation CE only; no new model/preprocessing/calibration selection using paired labels',
        label_rule=label_audit['rule'],redshift_bins=bin_policy,
        support=support_audit,bootstrap=dict(replicates=1000,seed=SEED,unit='paired physical galaxy',interval='2.5/97.5 percentile',
        conditional='Existing frozen checkpoints; no training-seed uncertainty'),
        source_test_policy='Original HSC test objects were previously evaluated in completed training reports; now permitted for this predeclared frozen-model matched observation comparison, with no tuning or checkpoint selection. Not a never-inspected external sample.',
        comparisons='HSC vs Legacy within each frozen model; A/B/C penalties on identical IDs; empty Legacy-trained rows, no bidirectional claim')
    write(out/'evaluation_policy.json',policy)
    # Build all complete input strings, including source-only replay, before any new evaluation.
    registry=load_registry(verify_curves=True);tokenizer=AutoTokenizer.from_pretrained('/arc/home/gsm/hf_models/Qwen3.5-4B-Base',local_files_only=True)
    inputs=out/'inputs';inputs.mkdir(exist_ok=True);lengths={};text_hashes={};max_length=512
    for level in range(3):
        for domain,kind,feature,group in [('hsc','galaxiesml-spec','hsc_features',d['galaxy_id']),('legacy','desi-ls','legacy_features',d['legacy_group'])]:
            names=[n for _,v,e in feature_columns(kind) for n in (v,e)]+['MW_EBV']
            texts=serialize_rows(dict(features=d[feature],group=group),dict(feature_names=names),catalogue=kind,level=level,registry=registry)
            key=f'level{level}_{domain}';path=inputs/f'{key}.jsonl'
            with path.open('w') as f:
                for tid,t in zip(d['targetid'],texts):f.write(json.dumps(dict(targetid=str(tid),text=t))+'\n')
            lens=[len(x) for x in tokenizer(texts,truncation=False,padding=False)['input_ids']]
            lengths[key]=dict(min=min(lens),max=max(lens),n=len(lens),would_truncate_at_512=sum(x>512 for x in lens))
            max_length=max(max_length,max(lens));text_hashes[key]=sha(path)
        ids=read(out/'A_replay_reference.npz')['targetid'];lookup={int(t):i for i,t in enumerate(source['targetid'])};ridx=np.array([lookup[int(t)] for t in ids])
        meta=json.loads((RUNS/'qwen-galaxiesml-spec-qlora-specz-e10/prepared.json').read_text())
        replay=serialize_rows(dict(features=source['features'][ridx],group=source['group'][ridx]),meta,catalogue='galaxiesml-spec',level=level,registry=registry)
        # Check byte identity to actual archived warm-start source strings.
        archive={json.loads(line)['object_id']:json.loads(line)['text'] for line in (WARM/f'level{level}/source_inputs.jsonl').open()}
        assert all(t==archive[str(tid)] for tid,t in zip(ids,replay))
        with (inputs/f'level{level}_replay.jsonl').open('w') as f:
            for tid,t in zip(ids,replay):f.write(json.dumps(dict(targetid=str(tid),text=t))+'\n')
    write(out/'manifest.json',dict(created_utc=now(),models=models,unavailable_legacy=absent,protected_sha256=protected,
        base_model='/arc/home/gsm/hf_models/Qwen3.5-4B-Base',
        cohort_sha256=sha(out/'cohort.npz'),eligible_pairs_sha256=sha(out/'eligible_pairs.npz'),
        policy_sha256=sha(out/'evaluation_policy.json'),input_sha256=text_hashes,token_audits=lengths,
        max_length=max_length,sequence_rule='Common limit=max(512, longest complete record), fixed before predictions; no truncation',
        grids_compatible=True,redshift_edges=ref['edges'].tolist(),redshift_centers=ref['centers'].tolist(),
        probability_convention='Softmax over 300 equal-width bins from 0 to 4.5; probability mass per bin',
        registry_sha256=sha(DEFAULT_REGISTRY),training=False,weights_updated=False,target_labels_fitted=False,
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'notebooks/audit_hsc_legacy_pairing.py',ROOT/'aion_magnitude/filter_photometry.py',ROOT/'aion_magnitude/metrics.py']}))
    print(json.dumps(dict(counts=audit['category_counts'],selected=audit['selected_counts'],max_length=max_length,labels=label_audit)),flush=True)


def protected_check(manifest):
    for path,expected in manifest['protected_sha256'].items():
        assert sha(path)==expected, f'Checkpoint changed: {path}'


def run(args):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model_state_dict
    from safetensors.torch import load_file
    from aion_magnitude.FM_Qwen import load_frozen_qwen
    from aion_magnitude.qwen_posttraining import QwenPhotoZModel,qwen_hidden_size
    from aion_magnitude.metrics import summarize_pdf_metrics
    from qwen_zero_shot_sanity import reserve_one_gpu,evaluate,save_predictions
    out=args.output_dir;m=json.loads((out/'manifest.json').read_text());protected_check(m)
    assert sha(out/'cohort.npz')==m['cohort_sha256'] and sha(out/'evaluation_policy.json')==m['policy_sha256']
    d=read(out/'cohort.npz');info=m['models'][args.model];config=info['config'].copy();config['max_length']=m['max_length']
    folder=out/args.model;folder.mkdir(exist_ok=True);started=time.time()
    def texts(key):
        path=out/'inputs'/f'{key}.jsonl'
        if key in m['input_sha256']:assert sha(path)==m['input_sha256'][key]
        return [json.loads(x)['text'] for x in path.open()]
    with reserve_one_gpu() as gpu:
        assert gpu['memory_before_gib']['used']<.25, f'GPU slice in use: {gpu}'
        write(folder/'running.json',dict(started_utc=now(),gpu=gpu,model=args.model))
        torch.manual_seed(SEED)
        base,tokenizer=load_frozen_qwen(m['base_model'],device='cuda',load_in_4bit=True,torch_dtype='bf16',local_files_only=True)
        base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
        selected=Path(info['directory']);qwen=PeftModel.from_pretrained(base,selected/'adapter',is_trainable=False)
        expected=load_file(str(selected/'adapter/adapter_model.safetensors'));actual=get_peft_model_state_dict(qwen)
        assert actual.keys()==expected.keys()
        for k,v in actual.items():torch.testing.assert_close(v.cpu(),expected[k],rtol=0,atol=0,check_dtype=False)
        model=QwenPhotoZModel(qwen,hidden_size=qwen_hidden_size(qwen),n_z_bins=config['n_z_bins'],head_hidden_dim=config['head_hidden_dim'],pooling=config['pooling']).to('cuda')
        head=torch.load(selected/'photoz_head.pt',map_location='cpu',weights_only=True);model.photoz_head.load_state_dict(head,strict=True)
        model.requires_grad_(False);model.eval();assert not any(p.requires_grad for p in model.parameters())
        replay=read(out/f'{args.model}_replay_reference.npz')
        ev=evaluate(model,tokenizer,dict(targetid=replay['targetid'],z=replay['z']),texts(f'level{info["level"]}_replay'),config)
        diff=np.abs(ev['logits'].numpy()-replay['logits'])
        # Small GPU/backend rounding allowed; exact text and adapter/head tensors checked separately.
        assert float(diff.max())<.15, f'Unexpected source replay mismatch: {diff.max()}'
        write(folder/'source_replay.json',dict(n=32,source='archived validation only',max_abs_logit_difference=float(diff.max()),
            mean_abs_logit_difference=float(diff.mean()),metrics=summarize_pdf_metrics(ev),frozen=True,all_adapter_tensors_exact=True))
        print('SOURCE REPLAY OK',args.model,float(diff.max()),flush=True)
        # Incremental domain/chunk outputs survive interruption; immutable completed chunks are verified before reuse.
        for domain in ('hsc','legacy'):
            ts=texts(f'level{info["level"]}_{domain}');feature=d[f'{domain}_features']
            for start in range(0,len(ts),256):
                end=min(start+256,len(ts));path=folder/f'{domain}_{start:06d}_{end:06d}.npz'
                if path.exists():
                    old=read(path);np.testing.assert_array_equal(old['targetid'],d['targetid'][start:end]);continue
                small=dict(targetid=d['targetid'][start:end],z=d['z'][start:end],features=feature[start:end])
                ev=evaluate(model,tokenizer,small,ts[start:end],config)
                temp=path.with_name(path.stem+'.partial.npz');save_predictions(temp,small,ev);temp.replace(path)
                write(folder/'status.json',dict(model=args.model,domain=domain,completed=end,total=len(ts),updated_utc=now(),seconds=time.time()-started))
                print('PROGRESS',args.model,domain,end,'/',len(ts),f'{time.time()-started:.1f}s',flush=True)
        protected_check(m)
        write(folder/'complete.json',dict(model=args.model,finished_utc=now(),seconds=time.time()-started,gpu=gpu,
            training=False,weights_updated=False,target_labels_fitted=False,checkpoint_hashes_unchanged=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run','report']);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--n-primary',type=int,default=1024);p.add_argument('--n-validation',type=int,default=256);p.add_argument('--model',choices=MODELS)
    args=p.parse_args()
    if args.stage=='prepare':prepare(args)
    elif args.stage=='run':
        assert args.model;run(args)
    else:
        from hsc_legacy_paired_report import report
        report(args.output_dir)


if __name__=='__main__':main()
