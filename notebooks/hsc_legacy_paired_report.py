"""Analysis of the predeclared frozen paired-observation benchmark."""
from pathlib import Path
import csv
import json
import shutil

import numpy as np

from hsc_legacy_paired_benchmark import MODELS, SEED, ROOT, WARM, read, write, sha, now, protected_check


def photometric_summary(features, bands):
    def quantiles(values):
        values=values[np.isfinite(values)]
        return dict(n=len(values),q05_q50_q95=np.quantile(values,[.05,.5,.95]).tolist() if len(values) else [])
    result={};flux=features[:,::2][:,:5];sigma=features[:,1::2]
    for j,band in enumerate(bands):
        f,e=flux[:,j],sigma[:,j];positive=(f>0)&np.isfinite(f)
        with np.errstate(divide='ignore',invalid='ignore'):
            result[band]=dict(flux=quantiles(f),sigma=quantiles(e),snr=quantiles(f/e),
                apparent_ab_magnitude=quantiles(np.where(positive,22.5-2.5*np.log10(f),np.nan)),
                nominal_5sigma_ab_depth=quantiles(22.5-2.5*np.log10(5*e)),
                missing=int((~np.isfinite(f)|~np.isfinite(e)).sum()),nonpositive_measured_flux=int((np.isfinite(f)&(f<=0)).sum()))
    for a,b in [('g','r'),('r','z')]:
        fa,fb=flux[:,bands.index(a)],flux[:,bands.index(b)]
        with np.errstate(divide='ignore',invalid='ignore'):
            result[f'{a}-{b}_AB_colour']=quantiles(np.where((fa>0)&(fb>0),-2.5*np.log10(fa/fb),np.nan))
    valid=np.isfinite(flux)&np.isfinite(sigma)
    patterns=[''.join('1' if v else '0' for v in row) for row in valid]
    patterns,counts=np.unique(patterns,return_counts=True)
    result['missingness_patterns']=dict(zip(patterns,counts.tolist()))
    result['EBV']=quantiles(features[:,-1])
    return result


def lower_median(a):
    return float(np.partition(a, (len(a)-1)//2)[(len(a)-1)//2])


def object_metrics(p):
    """Reuse the repository's per-object PDF definitions before bootstrap."""
    import torch
    from aion_magnitude.metrics import binned_log_score, discrete_crps, pit_values
    t={k:torch.from_numpy(v) for k,v in p.items() if k in ('logits','pz','z_spec','redshift_edges','redshift_centers')}
    return dict(delta=(p['z_p50']-p['z_spec'])/(1+p['z_spec']),
        ce=binned_log_score(t['logits'],t['z_spec'],t['redshift_edges']).numpy(),
        crps=discrete_crps(t['pz'],t['z_spec'],t['redshift_centers']).numpy(),
        coverage=((p['z_spec']>=p['z_p16'])&(p['z_spec']<=p['z_p84'])).astype(float),
        pit=pit_values(t['pz'],t['z_spec'],t['redshift_edges']).numpy())


def metrics(obj, rows):
    delta=obj['delta'][rows];median=lower_median(delta)
    return dict(ce=float(obj['ce'][rows].mean()),nmad=1.48*lower_median(np.abs(delta-median)),
        outlier=float((np.abs(delta)>.15).mean()),bias=float(delta.mean()),median_bias=median,
        crps=float(obj['crps'][rows].mean()),coverage68=float(obj['coverage'][rows].mean()),
        pit_mean=float(obj['pit'][rows].mean()),mean_abs_delta=float(np.abs(delta).mean()))


def paired_summary(h,l,rows,replicates=1000):
    a,b=metrics(h,rows),metrics(l,rows);keys=list(a)
    point={k:b[k]-a[k] for k in keys};rng=np.random.default_rng(SEED)
    source=[];target=[];deltas=[]
    for _ in range(replicates):
        r=rng.choice(rows,size=len(rows),replace=True)
        aa,bb=metrics(h,r),metrics(l,r)
        source.append([aa[k] for k in keys]);target.append([bb[k] for k in keys]);deltas.append([bb[k]-aa[k] for k in keys])
    intervals={}
    for name,values in [('hsc',source),('legacy',target),('delta',deltas)]:
        q=np.quantile(values,[.025,.975],axis=0)
        intervals[name]={k:[float(q[0,j]),float(q[1,j])] for j,k in enumerate(keys)}
    hin=np.abs(h['delta'][rows])>.15;lin=np.abs(l['delta'][rows])>.15
    return dict(n=len(rows),hsc=a,legacy=b,delta=point,ci95=intervals,
        transitions=dict(nonoutlier_to_outlier=int((~hin&lin).sum()),outlier_to_nonoutlier=int((hin&~lin).sum()),
            both_outlier=int((hin&lin).sum()),neither_outlier=int((~hin&~lin).sum())),
        absolute_residual_change=dict(mean=float((np.abs(l['delta'][rows])-np.abs(h['delta'][rows])).mean()),
            median=lower_median(np.abs(l['delta'][rows])-np.abs(h['delta'][rows])),
            fraction_worse=float((np.abs(l['delta'][rows])>np.abs(h['delta'][rows])).mean())))


def load_predictions(out,model,domain,cohort):
    chunks=[read(p) for p in sorted((out/model).glob(f'{domain}_??????_??????.npz'))]
    if not chunks:
        raise FileNotFoundError(f'No predictions: {model}/{domain}')
    pred={k:(chunks[0][k] if k in ('redshift_edges','redshift_centers') else np.concatenate([x[k] for x in chunks])) for k in chunks[0]}
    np.testing.assert_array_equal(pred['targetid'],cohort['targetid'])
    np.testing.assert_array_equal(pred['z_spec'],cohort['z'])
    for chunk in chunks:
        np.testing.assert_array_equal(chunk['redshift_edges'],pred['redshift_edges'])
        np.testing.assert_array_equal(chunk['redshift_centers'],pred['redshift_centers'])
    return pred


def contrast(h1,l1,h2,l2,rows):
    rng=np.random.default_rng(SEED);draws=[]
    keys=list(metrics(h1,rows))
    def diff(r):
        a,b,c,d=metrics(h1,r),metrics(l1,r),metrics(h2,r),metrics(l2,r)
        return {k:(d[k]-c[k])-(b[k]-a[k]) for k in keys}
    point=diff(rows)
    for _ in range(1000):
        r=rng.choice(rows,size=len(rows),replace=True);v=diff(r);draws.append([v[k] for k in keys])
    q=np.quantile(draws,[.025,.975],axis=0)
    return dict(definition='Penalty of second model minus penalty of first, all four cells resampled with identical galaxy indices',
        delta_penalty=point,ci95={k:[float(q[0,j]),float(q[1,j])] for j,k in enumerate(keys)})


def plots(out,cohort,pred,obj,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    p=cohort['category']=='P';z=cohort['z'][p]
    fig,axes=plt.subplots(5,2,figsize=(10,19),sharex=True,sharey=True)
    for row,model in enumerate(MODELS):
        for col,domain in enumerate(('hsc','legacy')):
            ax=axes[row,col];ax.scatter(z,pred[model][domain]['z_p50'][p],s=5,alpha=.4)
            ax.plot([0,4.5],[0,4.5],color='k',lw=.7);ax.set_title(f'{model}: {domain.upper()} (P, n={p.sum()})')
            ax.set(xlim=(0,4.5),ylim=(0,4.5),xlabel='DESI spec-z',ylabel='Predicted median z')
    fig.tight_layout();fig.savefig(figdir/'matched_predictions.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(19,7))
    for j,model in enumerate(MODELS):
        for domain,color in [('hsc','C0'),('legacy','C1')]:
            axes[0,j].hist(pred[model][domain]['z_p50'][p],bins=np.linspace(0,4.5,46),histtype='step',label=domain,color=color,density=True)
            axes[1,j].hist(obj[model][domain]['pit'][p],bins=np.linspace(0,1,21),histtype='step',density=True,label=domain,color=color)
        axes[0,j].hist(z,bins=np.linspace(0,4.5,46),histtype='step',color='k',ls=':',label='spec-z',density=True)
        axes[0,j].set(title=model,xlabel='Redshift',ylabel='Density');axes[1,j].set(xlabel='Binned PIT',ylabel='Density')
        axes[1,j].axhline(1,color='k',lw=.7);axes[0,j].legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'redshift_and_pit.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for index,model in enumerate(MODELS):
        bins=summary['models'][model]['P']['redshift_bins'];x=np.arange(len(bins))+(index-2)*.055
        y=np.array([b['metrics']['delta']['ce'] for b in bins]);ci=np.array([b['metrics']['ci95']['delta']['ce'] for b in bins])
        axes[0].errorbar(x,y,yerr=np.maximum(0,np.vstack((y-ci[:,0],ci[:,1]-y))),marker='o',ms=4,ls='none',capsize=2,label=model)
        vals=[summary['models'][model]['P'][k]['delta']['mean_abs_delta'] if k in summary['models'][model]['P'] else np.nan for k in ['full','support','outside_support']]
        axes[1].plot(range(3),vals,marker='o',label=model)
    axes[0].set(xticks=np.arange(len(bins)),xticklabels=[f"{b['low']:g}–{b['high']:g}\nn={b['metrics']['n']}" for b in bins],
                xlabel='Predefined spec-z bin (95% paired CI)',ylabel='Paired ΔCE (Legacy − HSC)');axes[0].legend(fontsize=8)
    axes[1].set(xticks=range(3),xticklabels=['Full P','Common support','Outside support'],ylabel='Mean Δ|normalised residual|')
    for ax in axes:ax.axhline(0,color='k',lw=.7)
    fig.tight_layout();fig.savefig(figdir/'transfer_by_redshift_and_support.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,10))
    for row,band in enumerate(('g','r','z')):
        jh=[0,1,3][row];jl=[0,1,2][row]
        for domain,index in [('hsc',jh),('legacy',jl)]:
            flux=cohort[f'{domain}_features'][p,2*index];error=cohort[f'{domain}_features'][p,2*index+1]
            values=[np.arcsinh(flux),np.log10(error),np.arcsinh(flux/error)]
            for col,v in enumerate(values):
                v=v[np.isfinite(v)];axes[row,col].hist(v,bins=35,histtype='step',density=True,label=domain)
                axes[row,col].set(xlabel=[f'{band}: asinh(flux nMgy)',f'{band}: log10(sigma nMgy)',f'{band}: asinh(SNR)'][col],ylabel='Density')
        axes[row,0].legend()
    fig.tight_layout();fig.savefig(figdir/'common_optical_photometry.png',dpi=150);plt.close(fig)


def report(out):
    compact=ROOT/'reports/filter-aware/1027'/out.name
    m=json.loads((out/'manifest.json').read_text());policy=json.loads((out/'evaluation_policy.json').read_text());audit=json.loads((out/'audit.json').read_text())
    protected_check(m);assert sha(out/'evaluation_policy.json')==m['policy_sha256'];d=read(out/'cohort.npz')
    assert sha(out/'cohort.npz')==m['cohort_sha256']
    full=read(out/'eligible_pairs.npz');source=read(WARM/'level0/source_cohort.npz')
    photometry={'source_hsc_train':photometric_summary(source['features'][source['split']=='train'],['g','r','i','z','y'])}
    for label,cohort in [('eligible_pool',full),('evaluated_P',{k:v[d['category']=='P'] for k,v in d.items()})]:
        for domain,bands in [('hsc',['g','r','i','z','y']),('legacy',['g','r','z','W1','W2'])]:
            photometry[f'{label}_{domain}']=photometric_summary(cohort[f'{domain}_features'],bands)
    write(out/'photometric_distributions.json',photometry)
    extra=json.loads((out/'supplementary_audit.json').read_text())
    pred={};obj={};result=dict(created_utc=now(),policy_sha256=m['policy_sha256'],models={},contrasts={},
        analysis_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'notebooks/audit_hsc_legacy_measurements.py']},
        confidence_intervals='1000 paired galaxy bootstrap draws, percentile 95%; conditional on these checkpoints; excludes training-seed uncertainty')
    for model in MODELS:
        assert (out/model/'complete.json').exists(),f'Model unfinished: {model}'
        pred[model]={};obj[model]={};result['models'][model]={}
        for domain in ('hsc','legacy'):
            pred[model][domain]=load_predictions(out,model,domain,d)
            np.testing.assert_array_equal(pred[model][domain]['redshift_edges'],np.asarray(m['redshift_edges'],np.float32))
            obj[model][domain]=object_metrics(pred[model][domain])
        h,l=obj[model]['hsc'],obj[model]['legacy']
        for category in ('P','V'):
            rows=np.flatnonzero(d['category']==category)
            if not len(rows):continue
            r={'full':paired_summary(h,l,rows)}
            for label,mask in [('support',d['common_support']),('outside_support',~d['common_support'])]:
                subset=rows[mask[rows]]
                if len(subset):r[label]=paired_summary(h,l,subset)
            edges=policy['redshift_bins'][category]['edges'];r['redshift_bins']=[]
            for j,(low,high) in enumerate(zip(edges[:-1],edges[1:])):
                subset=rows[(d['z'][rows]>=low)&((d['z'][rows]<=high) if j==len(edges)-2 else (d['z'][rows]<high))]
                if len(subset):r['redshift_bins'].append(dict(low=low,high=high,metrics=paired_summary(h,l,subset)))
            bins=r['redshift_bins'];r['redshift_standardized']={domain:{k:float(np.mean([b['metrics'][domain][k] for b in bins])) for k in r['full'][domain]} for domain in ('hsc','legacy','delta')}
            # Equal-bin standardization: independently resample within each fixed bin, preserving pairs.
            rng=np.random.default_rng(SEED);draws=[];keys=list(r['full']['delta'])
            binrows=[rows[(d['z'][rows]>=b['low'])&((d['z'][rows]<=b['high']) if j==len(bins)-1 else (d['z'][rows]<b['high']))] for j,b in enumerate(bins)]
            for _ in range(1000):
                values=[]
                for br in binrows:
                    sample=rng.choice(br,size=len(br),replace=True);a,b=metrics(h,sample),metrics(l,sample)
                    values.append([b[k]-a[k] for k in keys])
                draws.append(np.mean(values,axis=0))
            q=np.quantile(draws,[.025,.975],axis=0);r['redshift_standardized_delta_ci95']={k:[float(q[0,j]),float(q[1,j])] for j,k in enumerate(keys)}
            result['models'][model][category]=r
        # Per-galaxy differences retain IDs, class, support and both observation-system predictions.
        with (out/f'{model}_paired_differences.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['targetid','galaxy_id','category','common_support','z_spec','z_hsc','z_legacy','delta_hsc','delta_legacy','delta_abs_residual','delta_ce','delta_crps','outlier_hsc','outlier_legacy'])
            for i in range(len(d['z'])):
                writer.writerow([d['targetid'][i],d['galaxy_id'][i],d['category'][i],int(d['common_support'][i]),d['z'][i],
                    pred[model]['hsc']['z_p50'][i],pred[model]['legacy']['z_p50'][i],h['delta'][i],l['delta'][i],
                    abs(l['delta'][i])-abs(h['delta'][i]),l['ce'][i]-h['ce'][i],l['crps'][i]-h['crps'][i],
                    int(abs(h['delta'][i])>.15),int(abs(l['delta'][i])>.15)])
        print('ANALYSED',model,flush=True)
    rows=np.flatnonzero(d['category']=='P')
    for a,b in [('A','B'),('B','C'),('original-qlora','original-dora')]:
        result['contrasts'][f'{b}_minus_{a}']=contrast(obj[a]['hsc'],obj[a]['legacy'],obj[b]['hsc'],obj[b]['legacy'],rows)
    write(out/'results.json',result);plots(out,d,pred,obj,result)
    positive=sum(result['models'][n]['P']['full']['delta']['ce']>0 for n in MODELS)
    q=result['models']['original-qlora']['P']['full']
    lines=['# Frozen same-galaxy HSC → Legacy benchmark','',
        'Five completed HSC-trained models were evaluated with both HSC and original Tractor observations of the **same galaxies**. No completed Legacy-trained adapter/head pair exists, so the reverse transfer cells are unavailable. No training, calibration or target-based checkpoint selection was performed.','',
        f"On the independent selected P cohort, Legacy observations increase CE for **{positive}/5 models**. For original QLoRA, NMAD changes from {q['hsc']['nmad']:.4f} to {q['legacy']['nmad']:.4f}, and the catastrophic-outlier fraction from {q['hsc']['outlier']:.2%} to {q['legacy']['outlier']:.2%}. The comparisons below distinguish absolute predictive performance from the same-model observational transfer penalty.",'',
        '## Cohort and independence','',
        f"The complete raw audit found {audit['raw_audit']['hdf_rows']:,} HDF galaxies. The eligible paired pool contains **{audit['category_counts']['P']:,} P**, **{audit['category_counts']['V']:,} V**, and **{audit['category_counts']['T']:,} T** objects. This execution evaluates **{audit['selected_counts']['P']:,} P** and **{audit['selected_counts']['V']:,} V** objects. P excludes training and validation exposure for every included model using TARGETID, HSC imaging identity and one-arcsecond spherical matches against HSC and DESI targeting coordinates. T is never scored. V is explicitly exploratory.",
        '',f"Selection: {policy['sample_rule']}. If a finite size limit was used, the numerical results describe that deterministic first-pass sample, **not every object in the eligible pool**. Whole-pool audit counts are provided separately. Every model/domain cell uses exactly the same selected IDs and labels.",
        '',policy['source_test_policy'],
        '',audit['scope'],'','### Sequential cuts','',
        '| Cut | Before | Removed | Remaining |','|---|---:|---:|---:|']
    for row in audit['cutflow']:lines.append(f"| {row['step']} | {row['before']} | {row['removed']} | {row['remaining']} |")
    lines += ['', '### Measurements and spectroscopy','',
        audit['photometry']+'. HDF fibermap targeting measurements were not assumed equivalent to the original Tractor VAC; the HDF also lacks NOBS. Native signed Legacy fluxes and all genuine missingness are retained. HSC i/y never occupy Legacy W1/W2 fields.',
        '',f"Spectroscopy uses the intersection of the archived source protocols: primary GALAXY/TGT, ZWARN=0, COADD_FIBERSTATUS=0, DELTACHI2>40, finite nonnegative ZERR, 0≤z≤4.5, main bright/dark. Label agreement was fixed at 1e−6×(1+|z_Iron|) before predictions. {audit['label_audit']['exact_equal']:,}/{audit['label_audit']['rows_checked']:,} checked labels are numerically identical; {audit['label_audit']['numerical_agreement_1e6']:,} pass the tighter agreement rule. All cells retain the same native HDF spectroscopic label. The HDF reduction tag is absent: numerical agreement does **not** establish identical DESI reductions.",
        '',f"The HDF lacks ZWARN, DELTACHI2, SPECTYPE, ZCAT_PRIMARY, SURVEY and PROGRAM. Its available OBJTYPE agrees for all {extra['quality']['rows']:,} eligible pairs; its COADD_FIBERSTATUS differs for {extra['quality']['hdf_fiberstatus_nonzero']} pairs despite their DR1 Iron status passing the original source quality cut. These differences are preserved and reported, rather than claiming identical quality metadata. Selection uses the same Iron quality definition as the original source pipeline; no new HDF fiber-status cut was introduced. All 28,127 reconstructed HSC train/validation feature rows match the archived source arrays exactly. See `supplementary_audit.json` for quality flags, NOBS, MASKBITS and saved primary-provenance hashes.",
        '', '## Frozen checkpoint inventory','', '| Model | Selected state | Source train / val / test | Native representation |','|---|---|---|---|']
    for name,info in m['models'].items():
        count=info['split_counts'];sel=info['selection'];selected=f"update {sel['update']}" if 'update' in sel else f"epoch {sel['epoch']} (zero-based)"
        lines.append(f"| {name} | {selected}; minimum HSC val CE | {count['train']} / {count['val']} / {count['test']} | Level {info['level']} |")
    lines += ['', 'All five source cohort membership sets agree; exact original QLoRA/DoRA validation/test IDs, matching preparation cache keys, and identical persisted A/B/C source arrays were verified. Exact ID lists are saved per model. Adapter/head paths, SHA256 hashes, configurations, selection rules and provenance are in `manifest.json`. Every adapter tensor was checked against its saved file, every source text replay checked against archived warm-start text, and a 32-object source-validation logit replay passed before paired inference. Checkpoint hashes were unchanged after inference.',
        '', '## Matched transfer matrix: independent P','',
        'The point estimate is the median PDF-bin centre. CE is negative log probability **mass** in the spec-z bin. All grids were verified exactly equal: 300 equal-width bins over [0,4.5]. NMAD and median bias use the existing lower-median convention. CRPS and binned PIT use the existing evaluator; 68% coverage uses the bin-centre 16th–84th interval. These are frozen-checkpoint descriptive results.',
        '', '| Model | n | HSC CE | Legacy CE | HSC NMAD | Legacy NMAD | HSC outliers | Legacy outliers |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model in MODELS:
        r=result['models'][model]['P']['full'];a,b=r['hsc'],r['legacy']
        lines.append(f"| {model} | {r['n']} | {a['ce']:.4f} | {b['ce']:.4f} | {a['nmad']:.4f} | {b['nmad']:.4f} | {a['outlier']:.2%} | {b['outlier']:.2%} |")
    lines += ['| Legacy QLoRA | — | unavailable | unavailable | — | — | — | — |','| Legacy DoRA | — | unavailable | unavailable | — | — | — | — |',
        '', '### Paired transfer penalties (Legacy minus HSC)','',
        '| Model | ΔCE [95% CI] | ΔNMAD | Δoutlier fraction | Mean Δabs residual | Newly outlier / recovered |','|---|---|---:|---:|---:|---:|']
    for model in MODELS:
        r=result['models'][model]['P']['full'];v=r['delta'];ci=r['ci95']['delta']['ce'];tr=r['transitions']
        lines.append(f"| {model} | {v['ce']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | {v['nmad']:.4f} | {v['outlier']:+.2%} | {v['mean_abs_delta']:.4f} | {tr['nonoutlier_to_outlier']} / {tr['outlier_to_nonoutlier']} |")
    lines += ['',result['confidence_intervals']+'. All source, target and delta metric intervals, biases, CRPS, coverage, PIT, outlier transition counts and per-galaxy residual changes are in `results.json` and `*_paired_differences.csv`. The same paired bootstrap indices are used in every within-model contrast and all four cells of a cross-model penalty contrast.','',
        '## A/B/C and adaptation comparisons','',
        '| Penalty contrast | ΔΔCE [95% CI] | ΔΔNMAD [95% CI] |','|---|---|---|']
    for name,r in result['contrasts'].items():
        vals=[]
        for k in ('ce','nmad'):
            ci=r['ci95'][k];vals.append(f"{r['delta_penalty'][k]:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
        lines.append(f"| {name} | {' | '.join(vals)} |")
    lines += ['', 'Positive values indicate the second model has a larger observational transfer penalty. Absolute HSC and Legacy scores must also be considered: a smaller penalty can arise from worse in-domain predictions. C had worse archived HSC validation performance than B. These matched comparisons cannot establish that wavelength metadata were interpreted physically.','',
        '## Redshift dependence','',
        f"Bins were fixed before inference from the spectroscopic distribution: {policy['redshift_bins']['P']}. All bins retain the same objects in both input conditions. Standardized results equally weight the merged nonempty bins, rather than changing the galaxy selection. Macro-averaged within-bin NMAD is explicitly not a pooled NMAD.",
        '', '| Model | Spec-z range | n | HSC NMAD | Legacy NMAD | HSC outlier | Legacy outlier | ΔCE [95% CI] |','|---|---|---:|---:|---:|---:|---:|---|']
    for model in MODELS:
        for b in result['models'][model]['P']['redshift_bins']:
            r=b['metrics'];a,l=r['hsc'],r['legacy'];ci=r['ci95']['delta']['ce']
            lines.append(f"| {model} | {b['low']:g}–{b['high']:g} | {r['n']} | {a['nmad']:.4f} | {l['nmad']:.4f} | {a['outlier']:.2%} | {l['outlier']:.2%} | {r['delta']['ce']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] |")
    lines += ['', '| Model | Equal-bin HSC CE | Equal-bin Legacy CE | Equal-bin ΔCE [95% CI] |','|---|---:|---:|---|']
    for model in MODELS:
        r=result['models'][model]['P'];v=r['redshift_standardized'];ci=r['redshift_standardized_delta_ci95']['ce']
        lines.append(f"| {model} | {v['hsc']['ce']:.4f} | {v['legacy']['ce']:.4f} | {v['delta']['ce']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] |")
    lines += ['', 'Per-bin bias, median bias, CRPS, coverage and all transfer deltas are saved in `results.json`. This is redshift-stratified transfer, not strict high-z extrapolation: the source training includes these redshifts.','',
        '## Photometric support','',audit['support']['rule']+'. '+audit['support']['fit']+'. The transform uses source-only median g/r/z flux errors as asinh softening scales, log errors, asinh SNRs, and two asinh colours. It accepts signed target fluxes and requires genuine common-band errors. No target redshift enters the estimator.',
        '', '| Model | P subset | n | ΔCE | Mean Δabs residual | Legacy outlier fraction |','|---|---|---:|---:|---:|---:|']
    for model in MODELS:
        for key in ('full','support','outside_support'):
            r=result['models'][model]['P'].get(key)
            if r:lines.append(f"| {model} | {key} | {r['n']} | {r['delta']['ce']:.4f} | {r['delta']['mean_abs_delta']:.4f} | {r['legacy']['outlier']:.2%} |")
    lines += ['', 'The full selected P sample remains the main reference. `photometric_distributions.json` provides source-training, full-eligible-pool and evaluated-P flux, error, signed SNR, positive-flux AB magnitude, nominal 5σ depth, g−r/r−z colour, missingness-pattern and EBV distributions. Colours/magnitudes are undefined for nonpositive fluxes; those measurements remain valid for inference. The support subset is a diagnostic fixed independently of target labels; it is a conservative marginal-box rule and does not establish equality of filter transmission, depth, calibration, or measurement pipelines. A paired observation comparison controls the evaluated physical galaxies and spec-z distribution, but does not causally isolate bandpass effects.','',
        '## Validation-exposed exploratory cohort','', '| Model | n | HSC CE | Legacy CE | ΔCE |','|---|---:|---:|---:|---:|']
    for model in MODELS:
        r=result['models'][model].get('V',{}).get('full')
        if r:lines.append(f"| {model} | {r['n']} | {r['hsc']['ce']:.4f} | {r['legacy']['ce']:.4f} | {r['delta']['ce']:.4f} |")
    lines += ['', 'V is not independent evidence because its HSC labels were used in checkpoint selection. Its full stratified/support results are separately tagged in `results.json`.','',
        '## Figures','', '![Matched predictions](figures/matched_predictions.png)','', '![Redshift distributions and PIT](figures/redshift_and_pit.png)','',
        '![Transfer by redshift and support](figures/transfer_by_redshift_and_support.png)','', '![Common optical photometry](figures/common_optical_photometry.png)','',
        '## Reproduction and remaining bidirectional gap','',
        f"Persistent outputs: `{out.resolve()}`. The compact report is mirrored under `{compact.relative_to(ROOT)}/`. `audit.json` contains raw fingerprints, filtering counts, numerical label comparisons, per-band HDF/Tractor comparison, RELEASE counts and all per-model membership counts. `eligible_pairs.npz` preserves the full audited pool; `cohort.npz` preserves exact evaluated IDs, both measurement matrices, masks, labels and exposure. `inputs/` stores every exact serialized record. `evaluation_policy.json` was written before predictions and its hash was verified by every worker. `manifest.json` contains exact checkpoints, hashes and compatible PDF grids. Prediction chunks are atomic and resumable.",
        '', '```bash',
        f'PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/audit_hsc_legacy_pairing.py --output-dir {out}/raw-audit',
        f"PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/hsc_legacy_paired_benchmark.py prepare --output-dir {out} --n-primary {policy['n_primary_limit']} --n-validation {policy['n_validation_limit']}",
        '# For each of original-qlora original-dora A B C, select one actually idle MIG slice:',
        f'CUDA_VISIBLE_DEVICES=<idle-MIG-UUID> PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/hsc_legacy_paired_benchmark.py run --output-dir {out} --model <model>',
        f'PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/audit_hsc_legacy_measurements.py --output-dir {out}',
        f'PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/hsc_legacy_paired_benchmark.py report --output-dir {out}',
        '```','',
        'Do not rerun prepare over an existing manifest. Frozen inference reuses verified completed chunks. The original Legacy runs currently have no completed adapter/head; their removed sampled-cohort cache prevents reconstructing exact source exposure from preparation metadata alone. To complete a mutually independent bidirectional benchmark: (1) recover or deterministically reconstruct the original Legacy sampled cohort and train/val/test IDs using the archived preparation code/configuration and raw input order, verifying the saved cache key and split counts; (2) obtain genuinely completed compatible Legacy-trained adapter/head states selected solely on source validation (or separately authorize any future training); (3) repeat identifier/group/position exposure OR across both source cohorts, freeze the mutually clean P intersection and the evaluation policy, then evaluate all four cells on exactly those IDs. If that intersection is too small, reserve a genuinely new matched catalogue outside both source exposure sets before any new training. No training was launched here.']
    text='\n'.join(lines)+'\n';(out/'report.md').write_text(text)
    compact.mkdir(parents=True,exist_ok=True)
    for name in ('report.md','results.json','audit.json','evaluation_policy.json','manifest.json','photometric_distributions.json','supplementary_audit.json','base_model_identity.json'):
        shutil.copy2(out/name,compact/name)
    shutil.copytree(out/'figures',compact/'figures',dirs_exist_ok=True)
    write(out/'complete.json',dict(finished_utc=now(),models=list(MODELS),checkpoint_hashes_unchanged=True,report=str(compact/'report.md')))
    print('REPORT COMPLETE',compact/'report.md',flush=True)
