#!/usr/bin/env python3
"""Reproducible frozen-weight Stage-0; never opens a training entry point.

Support preparation reads photometry only. Evaluation labels are loaded only by
the inference/collection stages, after support assignments have been saved.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'notebooks')]
import numpy as np

STORE = Path('/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints')
BASE = STORE / '1023/hsc-warmstart'
COMMON = STORE / '1023/clauds-common-flux-evaluation'
FAST = ROOT / 'fast-check-runs/1023/hsc-qlora-to-legacy-clauds'
SOURCE = Path('/arc/home/gsm/aion_output/figures/qwen-galaxiesml-spec-qlora-specz-e10')
HDF = Path('/arc/projects/ots/Cosmic_Imprint_of_Time/GalaxiesMLSpec/DESI_HSC_64x64_v2.hdf5')
CAT = Path('/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological_multiband_updated.fits')
BANDS = tuple('grizy')
SEED = 1027
VARIANTS = ('source_exact_na', 'source_exact_omit_ebv', 'target_schema_na')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + '\n')


def npz(p, keys=None):
    with np.load(p, allow_pickle=False) as a:
        return {k: a[k] for k in (keys if keys is not None else a.files)}


def table(p, rows):
    if not rows:
        return
    with Path(p).open('w') as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def protect():
    files = [BASE/'level0/source_cohort.npz', BASE/'level0/source_inputs.jsonl', COMMON/'cohort.npz', COMMON/'manifest.json', FAST/'summary.json', FAST/'clauds_predictions.npz']
    for folder in [FAST/'source_snapshot', *[BASE/f'level{i}/selected' for i in range(3)]]:
        files.extend(p for p in folder.rglob('*') if p.is_file())
    for i in range(3):
        files += [BASE/f'level{i}/manifest.json', BASE/f'level{i}/summary.json', BASE/f'level{i}/clauds_selected_predictions.npz']
    files += list(COMMON.glob('arm-*/*.npz'))
    return {str(p): sha(p) for p in files}


def check(out):
    m = json.loads((out/'manifest.json').read_text())
    for p, digest in m['protected_sha256'].items():
        assert sha(p) == digest, f'Protected artifact changed: {p}'
    for p, digest in m['prepared_sha256'].items():
        assert sha(out/p) == digest, f'Prepared artifact changed: {p}'
    for f in m['base_model_source']['files']:
        st = Path(f['path']).stat()
        assert st.st_size == f['bytes'] and st.st_mtime_ns == f['mtime_ns']
    return m


def feature_raw(flux, sigma, softness, joint):
    """Signed luptitude-like fluxes/colours; source-fixed scales throughout."""
    a = np.arcsinh(flux/softness)
    columns = [a, np.diff(a, axis=1)]
    names = [f'asinh_flux_{b}' for b in BANDS] + [f'asinh_colour_{a}_{b}' for a,b in zip(BANDS[:-1], BANDS[1:])]
    if joint:
        columns += [np.log(sigma/softness), np.arcsinh(flux/sigma), np.log1p(sigma/np.sqrt(flux**2+softness**2))]
        names += [f'log_sigma_{b}' for b in BANDS] + [f'asinh_sn_{b}' for b in BANDS] + [f'log_relative_sigma_{b}' for b in BANDS]
    return np.column_stack(columns), names


def fit_transform(flux, sigma, joint):
    softness = np.nanmedian(sigma, axis=0)
    assert np.all(softness > 0)
    x, names = feature_raw(flux, sigma, softness, joint)
    med = np.nanmedian(x, axis=0)
    scale = (np.nanquantile(x,.75,axis=0)-np.nanquantile(x,.25,axis=0))/1.349
    scale = np.where(scale > 1e-8, scale, 1.)
    return dict(softness=softness.tolist(), median=med.tolist(), scale=scale.tolist(), names=names,
                missing_indicator_weight=1., joint=joint)


def transform(flux, sigma, spec):
    x, _ = feature_raw(flux, sigma, np.array(spec['softness']), spec['joint'])
    missing = ~np.isfinite(x)
    x = np.where(missing, np.array(spec['median']), x)
    x = (x-np.array(spec['median']))/np.array(spec['scale'])
    # Finite signed transforms, no target-based winsorization or statistics.
    return np.column_stack([x, missing.astype(float)])


def support(out, sf, se, sid, vf, ve, vid, tf, te, tid, quality):
    from sklearn.model_selection import KFold
    from sklearn.neighbors import NearestNeighbors
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, log_loss
    folds = list(KFold(5,shuffle=True,random_state=SEED).split(sf))
    np.savez_compressed(out/'source_training_photometry.npz', targetid=sid, flux=sf, sigma=se)
    results, specs, scores = {}, {}, {}
    for name, joint in [('flux_colour',False), ('joint_photometry',True)]:
        oof = np.zeros(len(sf)); target_ratios=[]; val_ratios=[]; fold_info=[]
        for fold,(fit,cal) in enumerate(folds):
            spec = fit_transform(sf[fit],se[fit],joint)
            x = transform(sf[fit],se[fit],spec)
            nn = NearestNeighbors(n_neighbors=10,n_jobs=2).fit(x)
            dist = lambda f,e: nn.kneighbors(transform(f,e,spec),return_distance=True)[0].mean(axis=1)
            dc=dist(sf[cal],se[cal]); dt=dist(tf,te); dv=dist(vf,ve)
            q=np.quantile(dc,.95)
            oof[cal]=dc/q; target_ratios.append(dt/q); val_ratios.append(dv/q)
            fold_info.append(dict(fold=fold,fit_count=len(fit),calibration_count=len(cal),threshold_q95=float(q),preprocessor=spec))
            if fold==0:
                specs[name]=spec
                np.savez_compressed(out/f'{name}_fold0.npz',fit_source_indices=fit,calibration_source_indices=cal,source_distances=dc,target_distances=dt,validation_distances=dv)
        tr=np.median(target_ratios,axis=0); vr=np.median(val_ratios,axis=0)
        scores[name]=tr
        np.savez_compressed(out/f'{name}_support.npz',targetid=tid,target_ratio=tr,target_fold_ratios=target_ratios,source_oof_ratio=oof,validation_id=vid,validation_ratio=vr)
        results[name]=dict(k=10,folds=fold_info,aggregation='median of five distance/q95 ratios <= 1',
            source_oof_pass_fraction=float(np.mean(oof<=1)),source_validation_pass_fraction=float(np.mean(vr<=1)),target_pass_count=int(np.sum(tr<=1)),target_pass_fraction=float(np.mean(tr<=1)),
            target_ratio_quantiles=np.quantile(tr,[0,.25,.5,.75,1]).tolist(),
            sensitivity={str(q):int(np.sum(np.median([target_ratios[j]*fold_info[j]['threshold_q95']/np.quantile(npz(out/f'{name}_fold0.npz')['source_distances'],q) for j in [0]],axis=0)<=1)) for q in (.90,.95,.99)})
        # Sensitivity is explicitly fold 0, not an alternative selected stratum.
        results[name]['sensitivity_description']='fold-0 counts at source-calibration 90/95/99 percentiles; descriptive only'
    complete=np.isfinite(tf).all(axis=1)
    common=complete & (scores['flux_colour']<=1) & (scores['joint_photometry']<=1)
    brightness_edges=np.nanquantile(sf[:,2],[1/3,2/3])
    source_sn=np.nanmedian(sf/se,axis=1); target_sn=np.nanmedian(tf/te,axis=1)
    sn_edges=np.nanquantile(source_sn,[1/3,2/3])
    bright=np.where(~np.isfinite(tf[:,2]),'missing',np.array(['faint','middle','bright'])[np.digitize(np.nan_to_num(tf[:,2],nan=-np.inf),brightness_edges)])
    sn=np.where(~np.isfinite(target_sn),'missing',np.array(['low','middle','high'])[np.digitize(np.nan_to_num(target_sn,nan=-np.inf),sn_edges)])
    rng=np.random.default_rng(SEED); ti=rng.permutation(len(tf)); target_part=np.empty(len(tf),dtype='U10')
    target_part[ti[:128]]='train';target_part[ti[128:192]]='validation';target_part[ti[192:]]='heldout'
    source_part=np.empty(len(sf),dtype='U10'); si=rng.permutation(len(sf)); n=len(sf)
    source_part[si[:int(.6*n)]]='train';source_part[si[int(.6*n):int(.8*n)]]='validation';source_part[si[int(.8*n):]]='heldout'
    # Independent source-only preprocessing for the domain diagnostic.
    domain_spec=fit_transform(sf[source_part=='train'],se[source_part=='train'],True)
    sx=transform(sf,se,domain_spec);tx=transform(tf,te,domain_spec)
    domain={}; domain_prob={}
    for name,clf in [('logistic',LogisticRegression(C=1.,max_iter=2000,class_weight='balanced',random_state=SEED)),
                     ('hist_gradient',HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,l2_regularization=5.,early_stopping=False,class_weight='balanced',random_state=SEED))]:
        a=source_part=='train';b=target_part=='train'
        clf.fit(np.vstack([sx[a],tx[b]]),np.r_[np.zeros(a.sum()),np.ones(b.sum())])
        p=clf.predict_proba(tx)[:,1];domain_prob[name]=p; detail={}
        for part in ('validation','heldout'):
            a=source_part==part;b=target_part==part
            # Balanced evaluation makes the prior explicit and avoids source-count dominated accuracy.
            aidx=np.flatnonzero(a)[:b.sum()]; y=np.r_[np.zeros(len(aidx)),np.ones(b.sum())]
            pred=np.r_[clf.predict_proba(sx[aidx])[:,1],p[b]]
            boot=[];brng=np.random.default_rng(SEED)
            for _ in range(400):
                idx=brng.integers(0,len(y),len(y))
                if len(np.unique(y[idx]))==2:boot.append(roc_auc_score(y[idx],pred[idx]))
            detail[part]=dict(n_source=len(aidx),n_target=int(b.sum()),auc=float(roc_auc_score(y,pred)),auc_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),balanced_accuracy=float(balanced_accuracy_score(y,pred>=.5)),log_loss=float(log_loss(y,pred)))
        domain[name]=dict(parameters=clf.get_params(),**detail)
    write(out/'domain_classifier.json',dict(models=domain,preprocessor=domain_spec,source_train=int(np.sum(source_part=='train')),target_train=128,target_validation=64,target_heldout=64,
        caveat='Domain separability is not proof of zero support; probabilities are not calibrated density ratios. No redshift labels used. Hyperparameters fixed in code.'))
    np.savez_compressed(out/'domain_partition.npz',source_id=sid,source_partition=source_part,targetid=tid,target_partition=target_part,**{f'p_{k}':v for k,v in domain_prob.items()})
    rows=[]
    for i,ident in enumerate(tid):
        rows.append(dict(object_id=str(ident),common_support=bool(common[i]),flux_colour_ratio=float(scores['flux_colour'][i]),joint_photometry_ratio=float(scores['joint_photometry'][i]),complete_bands=int(np.isfinite(tf[i]).sum()),
            brightness_stratum=str(bright[i]),sn_stratum=str(sn[i]),median_sn=float(target_sn[i]) if np.isfinite(target_sn[i]) else None,bad_photometry=bool(quality[i]),domain_partition=target_part[i],
            logistic_target_probability=float(domain_prob['logistic'][i]),hist_target_probability=float(domain_prob['hist_gradient'][i])))
    table(out/'support_assignments.csv',rows)
    np.savez_compressed(out/'support_assignments.npz',targetid=tid,common_support=common,complete=complete,brightness=bright,sn=sn,quality=quality,**scores)
    report=dict(definition='complete grizy AND median five-fold flux/colour ratio <=1 AND median five-fold joint-photometry ratio <=1; q95 fixed on source training folds only',common_count=int(common.sum()),n_target=len(tf),common_fraction=float(common.mean()),
        flux_colour_complete_count=int(np.sum(complete&(scores['flux_colour']<=1))),source_training_count=len(sf),source_validation_count=len(vf),
        brightness_edges_i_flux_nanomaggy=brightness_edges.tolist(),sn_edges=sn_edges.tolist(),diagnostics=results,target_labels_read=False,domain_classifier=domain)
    write(out/'support_summary.json',report)
    plots_features(out,sf,se,tf,te,scores,domain_prob,target_part,common)
    return report


def plots_features(out,sf,se,tf,te,scores,prob,part,common):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,5,figsize=(16,6),constrained_layout=True)
    soft=np.nanmedian(se,axis=0)
    for j,b in enumerate(BANDS):
        for ax,s,t in [(axs[0,j],np.arcsinh(sf[:,j]/soft[j]),np.arcsinh(tf[:,j]/soft[j])),(axs[1,j],np.arcsinh(sf[:,j]/se[:,j]),np.arcsinh(tf[:,j]/te[:,j]))]:
            values=np.r_[s[np.isfinite(s)],t[np.isfinite(t)]];bins=np.linspace(values.min(),values.max(),45)
            ax.hist(s,bins=bins,density=True,histtype='step',label='HSC training')
            ax.hist(t,bins=bins,density=True,histtype='step',label='CLAUDS validation')
        axs[0,j].set_title(f'{b}: asinh flux');axs[1,j].set_title(f'{b}: asinh S/N')
    axs[0,0].legend(fontsize=7)
    for ext in ('png','pdf'):fig.savefig(out/f'feature_distributions.{ext}',dpi=170)
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    axs[0].scatter(scores['flux_colour'],scores['joint_photometry'],c=common,s=14);axs[0].axvline(1,color='grey');axs[0].axhline(1,color='grey');axs[0].set(xscale='log',yscale='log',xlabel='Flux/colour distance / source q95',ylabel='Joint distance / source q95')
    for name,p in prob.items():axs[1].hist(p[part=='heldout'],bins=np.linspace(0,1,20),histtype='step',label=name)
    axs[1].legend();axs[1].set(xlabel='Held-out target domain probability',ylabel='Objects')
    # Positive i only; no logarithm is applied to a negative flux.
    for flux,label in ((sf,'HSC training'),(tf,'CLAUDS')):
        x=flux[:,2];m=22.5-2.5*np.log10(x[x>0]);axs[2].hist(m,bins=np.linspace(14,31,45),density=True,histtype='step',label=label)
    axs[2].set(xlabel='Observed i AB magnitude (positive only)',ylabel='Density');axs[2].legend()
    for ext in ('png','pdf'):fig.savefig(out/f'support_diagnostics.{ext}',dpi=170)
    plt.close(fig)


def prepare(out):
    from transformers import AutoTokenizer
    from aion_magnitude.filter_photometry import CLAUDS_BANDS,load_registry,clauds_flux_to_nanomaggy,serialize_photometry,legacy_config,audit_tokens
    from aion_magnitude.FM_Qwen import serialize_qwen_feature_row
    from aion_magnitude.clauds_bands import ALL_BAND_FLUX_COLUMNS,ALL_BAND_ERROR_COLUMNS
    import fitsio
    import h5py
    if (out/'manifest.json').exists():raise FileExistsError('Use a fresh Stage-0 directory')
    out.mkdir(parents=True,exist_ok=True);(out/'inputs').mkdir(exist_ok=True)
    protected=protect()
    old=json.loads((COMMON/'manifest.json').read_text())
    source=npz(BASE/'level0/source_cohort.npz',('targetid','features','split','group'))
    assert set(source['split'])=={'train','val'}
    assert np.sum(source['split']=='train')==22444
    assert not (set(source['group'][source['split']=='train'])&set(source['group'][source['split']=='val']))
    cohort=npz(COMMON/'cohort.npz',('targetid','source_row','raw_flux_microjy','raw_sigma_microjy','flux_nanomaggy','sigma_nanomaggy','valid','notObserved','isNoData','hasBadPhotometry','isClean'))
    ids=cohort['targetid']; assert [str(x) for x in ids]==old['query_ids'] and len(ids)==256
    cols=['ID']+list(ALL_BAND_FLUX_COLUMNS.values())+list(ALL_BAND_ERROR_COLUMNS.values())
    for b in CLAUDS_BANDS:
        suffix=ALL_BAND_FLUX_COLUMNS[b].removeprefix('FLUX_CMODEL_')
        cols += [prefix+'_'+suffix for prefix in ('notObserved','isNoData','hasBadPhotometry','isClean')]
    with fitsio.FITS(str(CAT)) as f:
        raw=f[1].read(rows=cohort['source_row'],columns=cols)
        (out/'catalogue_header.txt').write_text(str(f[1].read_header()))
    assert np.array_equal(raw['ID'],ids)
    rawf=np.column_stack([raw[ALL_BAND_FLUX_COLUMNS[b]] for b in CLAUDS_BANDS]);rawe=np.column_stack([raw[ALL_BAND_ERROR_COLUMNS[b]] for b in CLAUDS_BANDS])
    np.testing.assert_array_equal(rawf,cohort['raw_flux_microjy']);np.testing.assert_array_equal(rawe,cohort['raw_sigma_microjy'])
    for flag in ('notObserved','isNoData','hasBadPhotometry','isClean'):
        np.testing.assert_array_equal(np.column_stack([raw[flag+'_'+ALL_BAND_FLUX_COLUMNS[b].removeprefix('FLUX_CMODEL_')] for b in CLAUDS_BANDS]),cohort[flag])
    flux,sigma,valid=clauds_flux_to_nanomaggy(rawf,rawe,not_observed=cohort['notObserved'],no_data=cohort['isNoData'])
    np.testing.assert_allclose(flux,cohort['flux_nanomaggy'],rtol=0,atol=0,equal_nan=True)
    np.testing.assert_allclose(sigma,cohort['sigma_nanomaggy'],rtol=0,atol=0,equal_nan=True)
    idx=[CLAUDS_BANDS.index(b) for b in BANDS];tf=flux[:,idx];te=sigma[:,idx]
    sf=source['features'][:,0:10:2];se=source['features'][:,1:10:2]
    train=source['split']=='train';val=source['split']=='val'
    np.savez_compressed(out/'target_photometry.npz',targetid=ids,flux=tf,sigma=te,source_row=cohort['source_row'],notObserved=cohort['notObserved'][:,idx],isNoData=cohort['isNoData'][:,idx],hasBadPhotometry=cohort['hasBadPhotometry'][:,idx],isClean=cohort['isClean'][:,idx])
    # Independently verify original magnitudes/errors for source TRAINING rows only.
    # Root IDs are read for indexing, but no test photometry, labels or pixels.
    with h5py.File(HDF,'r') as f:
        allids=np.asarray([int(x) for x in f['targetid'][:]],dtype=np.int64);lookup={int(x):i for i,x in enumerate(allids)}
        rows=np.array([lookup[int(x)] for x in source['targetid'][train]]);order=np.argsort(rows);inv=np.argsort(order)
        mags=np.column_stack([f[f'HSC_metadata/{b}_cmodel_mag'][rows[order]][inv] for b in BANDS])
        magerr=np.column_stack([f[f'HSC_metadata/{b}_cmodel_magsigma'][rows[order]][inv] for b in BANDS])
        inventory={k:dict(shape=list(f['HSC_metadata'][k].shape),attributes={n:str(v) for n,v in f['HSC_metadata'][k].attrs.items()}) for k in f['HSC_metadata']}
        write(out/'source_hdf_metadata_inventory.json',dict(root_attributes={k:str(v) for k,v in f.attrs.items()},metadata=inventory,photometry_rows_read=len(rows),labels_read=False,images_read=False))
    from aion_magnitude.survey_catalogues import magnitude_features
    for j in range(5):
        f,e,v=magnitude_features(mags[:,j],magerr[:,j])
        np.testing.assert_array_equal(f.astype(np.float32),sf[train,j]);np.testing.assert_array_equal(e.astype(np.float32),se[train,j])
    registry=load_registry(verify_curves=True)
    tokenizer=AutoTokenizer.from_pretrained('/arc/home/gsm/hf_models/Qwen3.5-4B-Base',local_files_only=True)
    names=json.loads((SOURCE/'prepared.json').read_text())['feature_names']
    spec=legacy_config('galaxiesml-spec');texts={v:[] for v in VARIANTS}
    for i in range(256):
        mapping={name:value for b,ff,ee in zip(BANDS,tf[i],te[i]) for name,value in ((f'HSC_PDR2_cmodel_{b}_flux',ff),(f'HSC_PDR2_cmodel_{b}_flux_error',ee))}
        mapping['MW_EBV']=np.nan
        assert list(mapping)==names
        texts['source_exact_na'].append(serialize_qwen_feature_row(mapping,serialization=spec))
        texts['source_exact_omit_ebv'].append(serialize_qwen_feature_row({k:v for k,v in mapping.items() if k!='MW_EBV'},serialization=spec))
        texts['target_schema_na'].append(serialize_qwen_feature_row(mapping,serialization=replace(spec,schema_name='clauds_native_photometry_specz_v1')))
    audits={}
    for variant,strings in texts.items():
        audits[variant]=audit_tokens(tokenizer,strings,512)
        with (out/'inputs'/f'{variant}.jsonl').open('w') as f:
            for ident,text in zip(ids,strings):f.write(json.dumps(dict(object_id=str(ident),text=text))+'\n')
    # Prove original source serializer bytes/order using the actual archive.
    first=json.loads((BASE/'level0/source_inputs.jsonl').open().readline())
    assert serialize_photometry(dict(zip(names,source['features'][0])),catalogue='galaxiesml-spec',level=0)==first['text']
    assert str(source['targetid'][0])==first['object_id']
    quality=cohort['hasBadPhotometry'][:,idx].astype(bool).any(axis=1)
    report=support(out,sf[train],se[train],source['targetid'][train],sf[val],se[val],source['targetid'][val],tf,te,ids,quality)
    # Examples fixed by index and photometric properties, before reference labels.
    bright=int(np.nanargmax(tf[:,2]));faint=int(np.nanargmin(np.where(tf[:,2]>0,tf[:,2],np.nan)));missing=int(np.argmax((~np.isfinite(tf)).sum(axis=1)))
    assignment=npz(out/'support_assignments.npz'); common_idx=np.flatnonzero(assignment['common_support'])
    examples=list(dict.fromkeys([0,64,128,192,bright,faint,missing,*common_idx[:2].tolist()]))
    for variant,strings in texts.items():
        write(out/'inputs'/f'{variant}_examples.json',[dict(index=i,object_id=str(ids[i]),text=strings[i],input_ids=tokenizer(strings[i],truncation=False)['input_ids'],last_token_id=tokenizer(strings[i])['input_ids'][-1],last_token=tokenizer.convert_ids_to_tokens(tokenizer(strings[i])['input_ids'][-1])) for i in examples])
    band_audit=[]
    for j,b in enumerate(BANDS):
        band_audit.append(dict(band=b,filter_id=f'HSC.{b}',source_magnitude=f'HSC_metadata/{b}_cmodel_mag',source_error=f'HSC_metadata/{b}_cmodel_magsigma',target_flux=ALL_BAND_FLUX_COLUMNS[b],target_error=ALL_BAND_ERROR_COLUMNS[b],source_valid=int(np.isfinite(sf[train,j]).sum()),target_valid=int(np.isfinite(tf[:,j]).sum()),target_signed_nonpositive=int(np.sum(np.isfinite(tf[:,j])&(tf[:,j]<=0))),target_sigma_valid=int(np.isfinite(te[:,j]).sum()),target_bad_flag=int(cohort['hasBadPhotometry'][:,idx[j]].sum()),target_isClean_count=int(cohort['isClean'][:,idx[j]].sum()),source_median_sigma=float(np.nanmedian(se[train,j])),target_median_sigma=float(np.nanmedian(te[:,j]))))
    table(out/'band_audit.csv',band_audit)
    write(out/'unit_audit.json',dict(scale=10**(-.56),historical_magnitude_offset=-.9,old_flux_overestimate=10**.36,raw_conversion_verified=True,source_training_HDF_conversion_bit_exact=True,
        original_source_text_exact=True,source_signed_flux_recoverability='Magnitude-only source discards invalid/nonpositive flux; no raw signed source measurement is available to recover.',
        target_extinction='No correction; raw hscPipe fluxes. Exact local calibration history not specified.',source_extinction='As catalogued HSC cmodel magnitudes, MW_EBV supplied; no correction in source loader; local per-column extinction history not specified.',
        band_audit=band_audit,missing_target_EBV=True,unavailable_source_quality_flags=True))
    serializer=dict(source_config=dict(schema_name=spec.schema_name,decimals=spec.decimals,missing_token=spec.missing_token,prefix=spec.prefix),field_order=names,query_token='None is appended by source serializer or collator; last observed token is pooled. With EBV=NA this final token changes; omission also changes pooled field.',
        variants=dict(source_exact_na='Exact source field names/order/precision, MW_EBV=NA. Source survey schema and PDR2 field aliases are COUNTERFACTUAL diagnostic tags, not CLAUDS catalogue provenance.',source_exact_omit_ebv='Same diagnostic mapping, omit unavailable EBV; changes final pooling field.',target_schema_na='Keep all photometry fixed, change only schema survey identifier galaxiesml-spec -> clauds. PDR2 names remain diagnostic aliases.'),
        target_photometry='Actual raw CLAUDS HSC grizy; microJy scaled linearly; measured signed fluxes kept; no invented EBV or uncertainties.',token_audits=audits,example_indices=examples)
    write(out/'serializer_comparison.json',serializer)
    selected={}
    for arm,level in [('A',0),('B',1),('C',2)]:
        s=json.loads((BASE/f'level{level}/summary.json').read_text());selected[arm]=dict(directory=str(BASE/f'level{level}/selected'),update=s['best_update'])
    selected['original']=dict(directory=str(FAST/'source_snapshot'),update='completed original HSC QLoRA')
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),guide_sha256=sha(ROOT/'phys_guide_1027a.md'),seed=SEED,query_ids=[str(x) for x in ids],n=256,label='CLAUDS ZPHOT, not spectroscopic',target_split='validation',
        test_evaluated=False,redshift_model_training=False,weights_updated=False,target_label_calibration=False,source_training_rows=22444,source_validation_rows=5683,selected_checkpoints=selected,source_config=old['source_config'],base_model_source=old['base_model_source'],
        protected_sha256=protected,prepared_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.log'},
        overlap_audit=old['overlap_audit'],source_exclusion=old['source_exclusion'],example_indices=examples,variants=list(VARIANTS),max_length=512,source_support_count=report['common_count'],code_sha256=sha(__file__),versions={p:importlib.metadata.version(p) for p in ['numpy','torch','transformers','peft','bitsandbytes','scikit-learn','scipy','fitsio','h5py']})
    write(out/'manifest.json',manifest);print(json.dumps(dict(phase='prepared',common=report['common_count'],n=256,token_audits=audits)),flush=True)


def run(out,arm):
    import torch
    import qwen_clauds_common_flux_evaluation as runner
    import qwen_zero_shot_sanity as pilot
    from aion_magnitude.filter_photometry import audit_tokens
    from aion_magnitude.qwen_continued_sft import verify_replay
    m=check(out);folder=out/f'arm-{arm}'
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir();write(folder/'status.json',dict(phase='starting',pid=os.getpid()))
    try:
        with pilot.reserve_one_gpu() as gpu:
            assert gpu['memory_before_gib']['used']<.2
            model,tokenizer=runner.load_model(folder,m,arm)
            runner.BASE=BASE
            data=npz(COMMON/'cohort.npz',('targetid','features','z'))
            config=dict(m['source_config'],max_length=512)
            # Replay before diagnostic inputs. Validate runtime against archival weights/PDFs.
            if arm=='original':
                data_replay=npz(FAST/'source_replay.npz')
                metadata=json.loads((SOURCE/'prepared.json').read_text())
                from aion_magnitude.filter_photometry import serialize_rows
                texts=serialize_rows(data_replay,metadata,catalogue='galaxiesml-spec',level=0)
                e=pilot.evaluate(model,tokenizer,dict(data_replay,z=data_replay['z_spec']),texts,config)
                ref={k:torch.from_numpy(data_replay[k]) for k in ('logits','pz','z_spec','z_p16','z_p50','z_p84','redshift_edges','redshift_centers')}
                ref['object_id']=[str(x) for x in data_replay['targetid']]
                replay=verify_replay(e,ref)
            else:
                variant=f'{arm}_native_full';texts=[json.loads(l)['text'] for l in (COMMON/'inputs'/f'{variant}.jsonl').read_text().splitlines()]
                e=pilot.evaluate(model,tokenizer,data,texts,config)
                replay=verify_replay(e,runner.evaluation_from_archive(BASE/f'level{"ABC".index(arm)}/clauds_selected_predictions.npz'))
            write(folder/'native_replay.json',replay)
            for variant in VARIANTS:
                texts=[json.loads(l)['text'] for l in (out/'inputs'/f'{variant}.jsonl').read_text().splitlines()]
                audit_tokens(tokenizer,texts,512)
                write(folder/'status.json',dict(phase='inference',variant=variant,pid=os.getpid()))
                e=pilot.evaluate(model,tokenizer,data,texts,config)
                assert e['object_id']==m['query_ids'] and np.isfinite(e['logits'].numpy()).all()
                pilot.save_predictions(folder/f'{variant}.npz',data,e)
                from aion_magnitude.metrics import summarize_pdf_metrics
                print(arm,variant,json.dumps(summarize_pdf_metrics(e)),flush=True)
            check(out);write(folder/'status.json',dict(phase='complete',gpu=gpu,weights_updated=False,test_evaluated=False))
    except BaseException as e:
        write(folder/'status.json',dict(phase='failed',error=repr(e)));raise


def metrics(a,mask):
    import torch
    from aion_magnitude.metrics import summarize_pdf_metrics
    e={k:torch.from_numpy(a[k][mask] if k not in ('redshift_edges','redshift_centers') else a[k]) for k in ('logits','pz','z_spec','z_p16','z_p50','z_p84','redshift_edges','redshift_centers')}
    result=summarize_pdf_metrics(e);p=a['z_p50'][mask];v,c=np.unique(p,return_counts=True);s=np.sort(p)
    result.update(n=int(mask.sum()),predicted_median=float(np.median(p)),predicted_p05=float(np.quantile(p,.05)),predicted_p95=float(np.quantile(p,.95)),predicted_std=float(np.std(p)),modal_bin_fraction=float(c.max()/len(p)),unique_median_bins=len(v),concentration_width_003=float(max(np.searchsorted(s,x+.0300001,side='right')-i for i,x in enumerate(s))/len(s)))
    return result


def collect(out):
    m=check(out);s=npz(out/'support_assignments.npz');cohort=npz(COMMON/'cohort.npz',('targetid',))
    evaluations={'original_archived_native':npz(FAST/'clauds_predictions.npz')}
    for name in ('A_native_full','B_native_full','B_flux_full','B_flux_hsc','C_native_full','C_flux_full','C_flux_hsc'):
        evaluations[name]=npz(COMMON/f'arm-{name[0]}'/f'{name}.npz')
    for arm in ('original','A','B','C'):
        assert json.loads((out/f'arm-{arm}/status.json').read_text())['phase']=='complete'
        for v in VARIANTS:evaluations[f'{arm}_{v}']=npz(out/f'arm-{arm}/{v}.npz')
    strata={'all':np.ones(256,bool),'common_support':s['common_support'],'out_of_support':~s['common_support'],'complete_grizy':s['complete'],'incomplete_grizy':~s['complete']}
    for name in ('brightness','sn'):
        for value in np.unique(s[name]):strata[f'{name}_{value}']=s[name]==value
    strata['bad_photometry_flag']=s['quality'];strata['no_bad_photometry_flag']=~s['quality']
    rows=[]
    for name,a in evaluations.items():
        assert np.array_equal(a['targetid'],cohort['targetid'])
        for stratum,mask in strata.items():
            if not mask.any():continue
            rows.append(dict(condition=name,stratum=stratum,**metrics(a,mask),median_flux_colour_ratio=float(np.median(s['flux_colour'][mask])),median_joint_ratio=float(np.median(s['joint_photometry'][mask]))))
    table(out/'conditional_performance.csv',rows);table(out/'performance_all.csv',[r for r in rows if r['stratum']=='all'])
    paired=[]
    for arm,baseline in [('original','original_archived_native'),('A','A_native_full'),('B','B_flux_hsc'),('C','C_flux_hsc')]:
        for variant in VARIANTS:
            name=f'{arm}_{variant}';new=evaluations[name]['z_p50'];old=evaluations[baseline]['z_p50'];z=evaluations[name]['z_spec'];en=np.abs((new-z)/(1+z));eo=np.abs((old-z)/(1+z))
            paired.append(dict(condition=name,baseline=baseline,median_prediction_shift=float(np.median(new-old)),mean_abs_prediction_shift=float(np.mean(np.abs(new-old))),rescued_outliers=int(np.sum((eo>.15)&(en<=.15))),new_outliers=int(np.sum((eo<=.15)&(en>.15)))))
    table(out/'paired_changes.csv',paired)
    table(out/'object_predictions.csv',[dict(object_id=str(cohort['targetid'][i]),CLAUDS_ZPHOT=float(evaluations['original_archived_native']['z_spec'][i]),common_support=bool(s['common_support'][i]),**{n:float(e['z_p50'][i]) for n,e in evaluations.items()}) for i in range(256)])
    plot_predictions(out,evaluations,s,m)
    write(out/'summary.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),n=256,conditions=len(evaluations),common_support=int(s['common_support'].sum()),weights_updated=False,test_evaluated=False,target_labels_fitted=False,protected_artifacts_unchanged=True,performance_all=[r for r in rows if r['stratum']=='all'],performance_common=[r for r in rows if r['stratum']=='common_support']))
    print(json.dumps(dict(phase='complete',n_conditions=len(evaluations),common_support=int(s['common_support'].sum()))))


def plot_predictions(out,evals,s,m):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['original_archived_native','original_source_exact_na','original_source_exact_omit_ebv','original_target_schema_na','A_source_exact_na','B_flux_hsc','B_source_exact_na','C_flux_hsc','C_source_exact_na']
    fig,axs=plt.subplots(3,3,figsize=(13,12),constrained_layout=True)
    for ax,name in zip(axs.flat,names):
        e=evals[name]
        for mask,label,color in [(~s['common_support'],'Outside photometric support','tab:orange'),(s['common_support'],'Common photometric support','tab:blue')]:
            ax.scatter(e['z_spec'][mask],e['z_p50'][mask],s=12,alpha=.65,label=label,c=color)
        ax.plot([0,4.5],[0,4.5],'--',color='grey');ax.set(title=name,xlabel='CLAUDS ZPHOT',ylabel='Predicted median',xlim=(0,4.5),ylim=(0,4.5))
    axs.flat[0].legend(fontsize=7)
    for ext in ('png','pdf'):fig.savefig(out/f'conditional_scatter.{ext}',dpi=170)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
    for ax,mask,label in zip(axs,[s['common_support'],~s['common_support']],['Common support','Outside support']):
        for name in names:
            ax.hist(evals[name]['z_p50'][mask],bins=np.linspace(0,4.5,91),histtype='step',label=name)
        ax.hist(evals[names[0]]['z_spec'][mask],bins=np.linspace(0,4.5,91),histtype='step',color='black',ls='--',label='ZPHOT')
        ax.set(title=f'{label} (n={mask.sum()})',xlabel='Redshift',ylabel='Objects')
    axs[0].legend(fontsize=5)
    for ext in ('png','pdf'):fig.savefig(out/f'conditional_redshift_distributions.{ext}',dpi=170)
    plt.close(fig)
    indices=m['example_indices'];fig,axs=plt.subplots(int(np.ceil(len(indices)/3)),3,figsize=(14,3.5*np.ceil(len(indices)/3)),constrained_layout=True)
    for ax,i in zip(axs.flat,indices):
        for name in names:ax.plot(evals[name]['redshift_centers'],evals[name]['pz'][i],label=name,lw=1)
        ax.axvline(evals[names[0]]['z_spec'][i],color='black',ls='--');ax.set(title=f'ID {m["query_ids"][i]}, support={s["common_support"][i]}',xlabel='Redshift',ylabel='Probability per bin',xlim=(0,4.5))
    for ax in list(axs.flat)[len(indices):]:ax.axis('off')
    axs.flat[0].legend(fontsize=5)
    for ext in ('png','pdf'):fig.savefig(out/f'representative_pdfs.{ext}',dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,choices=['prepare','run','collect']);p.add_argument('--output-dir',type=Path,default=ROOT/'reports/stage0/1027a');p.add_argument('--arm',choices=['original','A','B','C']);args=p.parse_args()
    if args.stage=='run':
        if args.arm is None:p.error('--arm required')
        run(args.output_dir,args.arm)
    elif args.stage=='prepare':prepare(args.output_dir)
    else:collect(args.output_dir)


if __name__=='__main__':main()
