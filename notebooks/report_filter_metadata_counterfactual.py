"""Paired analyses of predeclared numerical filter-metadata interventions."""
from pathlib import Path
import csv
import json
import shutil

import numpy as np

from filter_metadata_counterfactual import ROOT,BINS,read,sha,write,now,validate_run,check_protected
from hsc_legacy_paired_report import object_metrics,metrics


def load_predictions(out,condition,domain,data):
    files=sorted((out/'predictions'/condition/domain).glob('??????_??????.npz'))
    assert len(files)==(len(data['z'])+255)//256,f'Incomplete {condition}/{domain}'
    chunks=[read(p) for p in files]
    result={k:(chunks[0][k] if k.startswith('redshift_') else np.concatenate([c[k] for c in chunks])) for k in chunks[0] if k!='features'}
    np.testing.assert_array_equal(result['targetid'],data['targetid']);np.testing.assert_array_equal(result['z_spec'],data['z'])
    for chunk in chunks:
        for key in ('redshift_edges','redshift_centers'):np.testing.assert_array_equal(chunk[key],result[key])
    return result


def bootstrap_statistics(obj,indices,bin_id):
    keys=list(metrics(obj,np.arange(len(obj['delta']))));all_values=[];bins=[]
    for sample in indices:
        values=metrics(obj,sample);all_values.append([values[k] for k in keys])
        counts=np.bincount(bin_id[sample],minlength=len(BINS)-1)
        bin_values=[]
        for key in ('ce','crps'):
            sums=np.bincount(bin_id[sample],weights=obj[key][sample],minlength=len(BINS)-1)
            bin_values.append(sums/np.maximum(1,counts))
        bins.append(np.asarray(bin_values).T)
    return keys,np.asarray(all_values),np.asarray(bins)


def interval(draws,keys):
    q=np.quantile(draws,[.025,.975],axis=0)
    return {key:[float(q[0,j]),float(q[1,j])] for j,key in enumerate(keys)}


def difference(point,base,draws,base_draws,keys):
    return dict(estimate={key:point[key]-base[key] for key in keys},ci95=interval(draws-base_draws,keys))


def transitions(base,current):
    a=np.abs(base['delta']);b=np.abs(current['delta']);ao=a>.15;bo=b>.15
    return dict(improved_fraction=float((b<a).mean()),worsened_fraction=float((b>a).mean()),unchanged_fraction=float((b==a).mean()),
        newly_catastrophic=int((~ao&bo).sum()),rescued=int((ao&~bo).sum()),both_catastrophic=int((ao&bo).sum()))


def plots(out,result,objects,examples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'figures';folder.mkdir(exist_ok=True)
    conditions=[c for c in result['conditions'] if c!='correct']
    fig,axes=plt.subplots(1,3,figsize=(18,7),sharey=True)
    for ax,key,title in zip(axes,['hsc','legacy','difference_in_sensitivity'],['HSC metadata effect','Legacy metadata effect','Legacy − HSC metadata sensitivity']):
        for i,name in enumerate(conditions):
            value=result['conditions'][name][key] if key=='difference_in_sensitivity' else result['conditions'][name]['metadata_effect'][key]
            x=value['estimate']['ce'];lo,hi=value['ci95']['ce']
            ax.errorbar(x,i,xerr=np.array([[max(0,x-lo)],[max(0,hi-x)]]),fmt='o',capsize=3)
        ax.axvline(0,color='k',lw=.7);ax.set(title=title,xlabel='ΔCE with paired 95% CI',yticks=range(len(conditions)),yticklabels=conditions)
    axes[0].invert_yaxis();fig.tight_layout();fig.savefig(folder/'metadata_ce_effects.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True)
    for ax,domain in zip(axes,['hsc','legacy']):
        values=[objects[name][domain]['delta']-objects['correct'][domain]['delta'] for name in conditions]
        ax.boxplot(values,showfliers=False,tick_labels=conditions);ax.axhline(0,color='k',lw=.6)
        ax.set(ylabel=f'{domain.upper()} Δnormalised residual')
    axes[-1].tick_params(axis='x',rotation=65);fig.tight_layout();fig.savefig(folder/'residual_changes.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    selected=['paired-cycle-forward','paired-cycle-reverse','paired-derangement','wavelength-cycle','width-cycle','placebo-length-structure']
    for domain,ax in zip(['hsc','legacy'],axes):
        for name in selected:
            bins=result['conditions'][name]['redshift_bins']
            ax.plot(range(len(bins)),[b['metadata_effect'][domain]['ce'] for b in bins],marker='o',label=name)
        ax.axhline(0,color='k',lw=.6)
        ax.set(xticks=range(len(BINS)-1),xticklabels=[f'{a:g}–{b:g}' for a,b in zip(BINS[:-1],BINS[1:])],xlabel='Fixed spec-z bin',ylabel=f'{domain.upper()} metadata ΔCE')
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(folder/'redshift_sensitivity.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,8),sharex=True)
    names=['correct','paired-cycle-forward','wavelength-cycle','width-cycle','placebo-length-structure','pivot-plus-05','pivot-minus-05']
    for j,ax in enumerate(axes.flat):
        for name in names:
            example=examples[name]['legacy'];ax.plot(example['centers'],example['pz'][j],label=name,lw=2 if name=='correct' else .8)
        ax.axvline(examples['correct']['legacy']['z'][j],color='k',ls=':',lw=.8)
        ax.set(title=f"Hash-selected TARGETID {examples['correct']['legacy']['targetid'][j]}",xlabel='Redshift',ylabel='Probability mass')
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(folder/'example_legacy_pdfs.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,7))
    for row,domain in enumerate(['hsc','legacy']):
        for name in ['correct','B',*conditions]:
            obj=objects[name][domain]
            axes[row,0].hist(obj['z_pred'],bins=np.linspace(0,4.5,61),histtype='step',density=True,label=name,lw=1.6 if name in ['correct','B'] else .6)
            axes[row,1].hist(obj['pit'],bins=np.linspace(0,1,21),histtype='step',density=True,lw=1.6 if name in ['correct','B'] else .6)
        axes[row,0].set(xlabel='Median predicted z',ylabel=f'{domain.upper()} density');axes[row,1].set(xlabel='Binned PIT',ylabel='Density')
        axes[row,1].axhline(1,color='k',ls=':')
    axes[0,0].legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(folder/'redshift_and_pit.png',dpi=150);plt.close(fig)


def report(out):
    m,d=validate_run(out);assert (out/'gates/full_baseline_replay.json').exists();check_protected(m,full=True)
    policy=json.loads((out/'evaluation_policy_v1.json').read_text());audit=json.loads((out/'tokenization_audit.json').read_text())
    mappings=json.loads((out/'metadata_mappings_v1.json').read_text());bootstrap=np.load(out/'bootstrap_indices.npy',mmap_mode='r')
    assert sha(out/'bootstrap_indices.npy')==policy['bootstrap']['file_sha256']
    rows=np.arange(len(d['z']));bin_id=np.minimum(np.searchsorted(BINS,d['z'],side='right')-1,len(BINS)-2)
    examples_ix=np.load(out/'smoke_indices.npy')[:6]
    result=dict(created_utc=now(),n=len(rows),policy_sha256=m['policy_sha256'],conditions={},B_reference={},redshift_edges=BINS,
        intervals='1000 fixed, saved paired galaxy bootstrap index arrays shared across all conditions and domains; conditional on these checkpoints and population; no training-seed uncertainty',
        analysis_code_sha256=sha(Path(__file__)))
    objects={};draws={};bin_draws={};points={};examples={};baseline_pdfs={}
    # Fixed references first; no ranking or checkpoint selection from target metrics.
    for name in ['correct','B',*[c for c in mappings['conditions'] if c!='correct']]:
        objects[name]={};draws[name]={};bin_draws[name]={};points[name]={};examples[name]={};sensitivity={}
        for domain in ['hsc','legacy']:
            pred=load_predictions(out,name,domain,d)
            np.testing.assert_array_equal(pred['redshift_edges'],np.asarray(m['redshift_edges'],np.float32))
            if name in ('correct','B'):
                ref=read(out/'reference'/f'{"C" if name=="correct" else "B"}_{domain}.npz')
                for key in ref:np.testing.assert_array_equal(pred[key],ref[key])
            obj=object_metrics(pred);obj['z_pred']=pred['z_p50'];obj['z_mean']=(pred['pz']*pred['redshift_centers']).sum(1)
            objects[name][domain]=obj;points[name][domain]=metrics(obj,rows)
            keys,draws[name][domain],bin_draws[name][domain]=bootstrap_statistics(obj,bootstrap,bin_id)
            examples[name][domain]=dict(targetid=d['targetid'][examples_ix],z=d['z'][examples_ix],pz=pred['pz'][examples_ix],centers=pred['redshift_centers'])
            if name=='correct':baseline_pdfs[domain]=pred['pz'].copy()
            if name!='B':
                tv=.5*np.abs(pred['pz']-baseline_pdfs[domain]).sum(1)
                sensitivity[domain]=dict(mean_total_variation=float(tv.mean()),median_total_variation=float(np.median(tv)),
                    p95_total_variation=float(np.quantile(tv,.95)),identical_pdf_fraction=float((tv==0).mean()),
                    median_redshift_changed_fraction=float((obj['z_pred']!=objects['correct'][domain]['z_pred']).mean()))
                np.savez_compressed(out/f'per_object_{name}_{domain}.npz',targetid=d['targetid'],z_spec=d['z'],
                    z_pred=obj['z_pred'],z_mean=obj['z_mean'],delta=obj['delta'],ce=obj['ce'],crps=obj['crps'],coverage=obj['coverage'],pit=obj['pit'],total_variation_from_correct=tv,
                    delta_ce=obj['ce']-objects['correct'][domain]['ce'],delta_crps=obj['crps']-objects['correct'][domain]['crps'],
                    delta_abs_residual=np.abs(obj['delta'])-np.abs(objects['correct'][domain]['delta']))
            del pred
        h,l=points[name]['hsc'],points[name]['legacy']
        absolute={domain:dict(metrics=points[name][domain],ci95=interval(draws[name][domain],keys)) for domain in ['hsc','legacy']}
        transfer=dict(estimate={k:l[k]-h[k] for k in keys},ci95=interval(draws[name]['legacy']-draws[name]['hsc'],keys))
        if name=='B':
            result['B_reference']=dict(absolute=absolute,transfer=transfer,redshift_bins=[dict(low=low,high=high,n=int((bin_id==j).sum()),
                absolute={domain:metrics(objects[name][domain],np.flatnonzero(bin_id==j)) for domain in ['hsc','legacy']})
                for j,(low,high) in enumerate(zip(BINS[:-1],BINS[1:]))]);continue
        effects={domain:difference(points[name][domain],points['correct'][domain],draws[name][domain],draws['correct'][domain],keys) for domain in ['hsc','legacy']}
        difference_draw=(draws[name]['legacy']-draws['correct']['legacy'])-(draws[name]['hsc']-draws['correct']['hsc'])
        r=dict(family=mappings['conditions'][name]['family'],absolute=absolute,transfer_both_domains=transfer,metadata_effect=effects,
            difference_in_sensitivity=dict(estimate={k:effects['legacy']['estimate'][k]-effects['hsc']['estimate'][k] for k in keys},ci95=interval(difference_draw,keys)),
            target_only=dict(definition='Legacy perturbed, HSC stays at correct C; its change in transfer penalty equals Legacy metadata effect, not the both-domain sensitivity difference',
                transfer=dict(estimate={k:l[k]-points['correct']['hsc'][k] for k in keys},ci95=interval(draws[name]['legacy']-draws['correct']['hsc'],keys)),
                transfer_penalty_change=effects['legacy']),
            transitions={domain:transitions(objects['correct'][domain],objects[name][domain]) for domain in ['hsc','legacy']},
            prediction_sensitivity=sensitivity,redshift_bins=[])
        for j,(low,high) in enumerate(zip(BINS[:-1],BINS[1:])):
            ix=np.flatnonzero(bin_id==j);a={domain:metrics(objects[name][domain],ix) for domain in ['hsc','legacy']}
            b={domain:metrics(objects['correct'][domain],ix) for domain in ['hsc','legacy']}
            e={domain:{k:a[domain][k]-b[domain][k] for k in keys} for domain in ['hsc','legacy']}
            ci={domain:interval(bin_draws[name][domain][:,j,:]-bin_draws['correct'][domain][:,j,:],['ce','crps']) for domain in ['hsc','legacy']}
            r['redshift_bins'].append(dict(low=low,high=high,n=len(ix),absolute=a,metadata_effect=e,metadata_effect_ci95=ci,
                sensitivity_difference={k:e['legacy'][k]-e['hsc'][k] for k in keys}))
        result['conditions'][name]=r
        print('ANALYSED',name,flush=True)
    # Show all three paired permutations, not only an extreme result.
    result['paired_permutation_outcomes']={name:{domain:result['conditions'][name]['metadata_effect'][domain] for domain in ['hsc','legacy']}
        for name in mappings['conditions'] if mappings['conditions'][name]['family']=='paired'}
    result['local_response']={}
    for domain in ['hsc','legacy']:
        names=['pivot-minus-05','pivot-minus-02','correct','pivot-plus-02','pivot-plus-05']
        z=np.stack([objects[name][domain]['z_mean'] for name in names]);diff=np.diff(z,axis=0)
        result['local_response'][domain]=dict(fraction_nondecreasing=float(np.all(diff>=0,axis=0).mean()),fraction_nonincreasing=float(np.all(diff<=0,axis=0).mean()),
            mean_shift={name:float((objects[name][domain]['z_mean']-objects['correct'][domain]['z_mean']).mean()) for name in names},
            interpretation='Fixed-checkpoint token-level finite perturbations, not a physically changed galaxy spectrum, physical derivative, or proof of spectral-feature reasoning')
    incorrect=[name for name in result['conditions'] if name!='correct']
    result['hypothesis_summary']={domain:dict(conditions=len(incorrect),incorrect_worse_CE=sum(result['conditions'][n]['metadata_effect'][domain]['estimate']['ce']>0 for n in incorrect),
        incorrect_worse_CE_ci_excludes_zero=sum(result['conditions'][n]['metadata_effect'][domain]['ci95']['ce'][0]>0 for n in incorrect),
        incorrect_better_CE_ci_excludes_zero=sum(result['conditions'][n]['metadata_effect'][domain]['ci95']['ce'][1]<0 for n in incorrect)) for domain in ['hsc','legacy']}
    sensitivity_ranges={domain:dict(mean_total_variation_range=[float(min(result['conditions'][n]['prediction_sensitivity'][domain]['mean_total_variation'] for n in incorrect)),
        float(max(result['conditions'][n]['prediction_sensitivity'][domain]['mean_total_variation'] for n in incorrect))],
        median_changed_fraction_range=[float(min(result['conditions'][n]['prediction_sensitivity'][domain]['median_redshift_changed_fraction'] for n in incorrect)),
        float(max(result['conditions'][n]['prediction_sensitivity'][domain]['median_redshift_changed_fraction'] for n in incorrect))]) for domain in ['hsc','legacy']}
    result['sensitivity_ranges']=sensitivity_ranges
    write(out/'results.json',result);plots(out,result,objects,examples)
    # Flat machine-readable tables complement full PDFs and per-object NPZs.
    with (out/'summary.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['condition','domain','n',*keys,'delta_meta_ce','delta_meta_crps'])
        for name,r in result['conditions'].items():
            for domain in ['hsc','legacy']:writer.writerow([name,domain,len(rows),*[r['absolute'][domain]['metrics'][k] for k in keys],r['metadata_effect'][domain]['estimate']['ce'],r['metadata_effect'][domain]['estimate']['crps']])
    lines=['# Frozen filter-metadata counterfactuals','',
        f"Completed all **13 predetermined C conditions**, each on both observations of the identical **{len(rows):,} independent paired galaxies**. B is an external frozen Level-1 reference. All weights and PDF heads were unchanged. Both original B/C full-cohort metrics and every live baseline logit/PDF were reproduced exactly before full counterfactual inference.",'',
        '## Established findings','',f"Across the 12 incorrect assignments, the CE outcome counts are `{result['hypothesis_summary']}`. The complete table below is reported; no perturbations were selected using target errors.",'',
        '| Condition | HSC CE | Legacy CE | HSC NMAD | Legacy NMAD | HSC outlier | Legacy outlier |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def row(name,r):
        h,l=r['absolute']['hsc']['metrics'],r['absolute']['legacy']['metrics']
        return f"| {name} | {h['ce']:.4f} | {l['ce']:.4f} | {h['nmad']:.4f} | {l['nmad']:.4f} | {h['outlier']:.2%} | {l['outlier']:.2%} |"
    lines.append(row('B (different frozen weights)',result['B_reference']))
    for name,r in result['conditions'].items():lines.append(row(name,r))
    lines+=['','## Paired metadata effects','',
        'All Δ values below change only numerical pivot/width fields in the same C checkpoint. ΔΔ is Legacy metadata sensitivity minus HSC metadata sensitivity when both domains are perturbed. For a target-only perturbation, the transfer-penalty change equals ΔLegacy; it is a separate design.','',
        '| Condition | ΔHSC CE [95% CI] | ΔLegacy CE [95% CI] | ΔΔCE [95% CI] | ΔLegacy CRPS [95% CI] |','|---|---|---|---|---|']
    def formatted(value,key):
        low,high=value['ci95'][key];return f"{value['estimate'][key]:+.4f} [{low:+.4f}, {high:+.4f}]"
    for name in incorrect:
        r=result['conditions'][name]
        lines.append(f"| {name} | {formatted(r['metadata_effect']['hsc'],'ce')} | {formatted(r['metadata_effect']['legacy'],'ce')} | {formatted(r['difference_in_sensitivity'],'ce')} | {formatted(r['metadata_effect']['legacy'],'crps')} |")
    lines+=['',result['intervals']+'. Bootstrap indices are saved and hashed before inference. Per-bin CE/CRPS intervals use the same draws restricted to that bin. All biases, coverage, PIT, paired residual changes and newly catastrophic/rescued counts are retained in `results.json` and per-object NPZs.','',
        '## Interpretation: H1–H4','',
        f"H1 (ignored metadata): the observed ranges are `{sensitivity_ranges}`. Nonzero PDF total variation demonstrates that metadata is not exactly ignored, even when the discretized median stays in the same bin. The size of the response and the corresponding CE/CRPS changes, rather than mere nonzero sensitivity, determine its practical importance.",'',
        f"H2 (numerical but nonphysical sensitivity): on Legacy, {result['hypothesis_summary']['legacy']['incorrect_better_CE_ci_excludes_zero']} of 12 incorrect assignments improve CE with paired intervals entirely below zero; {result['hypothesis_summary']['legacy']['incorrect_worse_CE_ci_excludes_zero']} worsen it with intervals entirely above zero. Changed PDFs establish dependence on numerical tokens. Improved incorrect assignments or mixed directions weaken a uniform physical-correctness claim, while degradation alone cannot rule out learned nonphysical token associations.",'',
        f"H3 (correctness advantage): correct metadata has lower Legacy CE at the point estimate for {result['hypothesis_summary']['legacy']['incorrect_worse_CE']} of 12 comparisons, and lower HSC CE for {result['hypothesis_summary']['hsc']['incorrect_worse_CE']} of 12. Correctness is systematically advantageous across this entire predetermined set only if all comparisons support that direction. Even consistent superiority supports an inference-time association, not a learned physical response model: each HSC profile’s correct constants were effectively fixed throughout training.",'',
        f"H4 (structured local response): fixed ±2% and ±5% results are `{result['local_response']}`. These are finite token interventions on unchanged photometry, not physically self-consistent altered observations. They do not establish redshifting of spectral features.",'',
        'B versus C uses different adapted weights and measures the broader training/representation change. Correct C versus corrupted C isolates inference-time numerical sensitivity. A future matched-training intervention with correct versus shuffled metadata could address a different causal question; no such training was launched.','',
        '## Redshift stratification','',f'Predeclared edges: `{BINS}`. The bin above z=1 contains {policy["redshift_counts"][-1]} objects, and no claim about z>2 generalisation is supported. All conditions use identical objects in each bin. B’s separate absolute per-bin metrics are retained in `B_reference.redshift_bins`; B/C differences are not called within-checkpoint metadata effects.','',
        '| Condition | Bin | n | HSC CE | Legacy CE | ΔHSC CE | ΔLegacy CE | ΔΔCE |','|---|---|---:|---:|---:|---:|---:|---:|']
    for name,r in result['conditions'].items():
        for b in r['redshift_bins']:
            lines.append(f"| {name} | {b['low']:g}–{b['high']:g} | {b['n']} | {b['absolute']['hsc']['ce']:.4f} | {b['absolute']['legacy']['ce']:.4f} | {b['metadata_effect']['hsc']['ce']:+.4f} | {b['metadata_effect']['legacy']['ce']:+.4f} | {b['sensitivity_difference']['ce']:+.4f} |")
    lines+=['','## Input and token audit','',
        'The intervention wraps the original Level-2 serializer and changes only `pivot_nm` and `width_nm`, both to three decimal places in nm. Every original nonmetadata byte, filter identity, survey/instrument name, signed flux, uncertainty, missing value, EBV, object ordering and label is preserved. North/south Legacy optical profiles are distinct; shared WISE identities have one global mapping. Paired cycles and seeded derangements preserve the entire pair collection within each domain’s physical-profile group.','',
        f"The common context limit is {m['max_length']}; no record is truncated. The numerical placebo independently permutes pivot/width assignments within HSC, Legacy optical, and WISE strata. Its observed exact-length fractions are `{audit['placebo_length_matching']}`. Equal lengths do not imply identical token IDs or equal attention. `tokenization_audit.json` records all token-length distributions, positionwise changed-ID frequencies and systematic length differences; `tokens/` preserves actual token IDs for every record.",'',
        '## Figures','', '![Paired CE effects](figures/metadata_ce_effects.png)','',
        '![Redshift sensitivity](figures/redshift_sensitivity.png)','', '![Residual changes](figures/residual_changes.png)','',
        '![Example PDFs](figures/example_legacy_pdfs.png)','', '![Redshift and PIT](figures/redshift_and_pit.png)','',
        '## Reproduction and preserved artifacts','',
        f"Persistent output: `{out.resolve()}`. `manifest.json` records exact saved B/C checkpoint paths, SHA256 identities, the immutable original registry hash and original benchmark artifact hashes. Policy SHA256: `{m['policy_sha256']}`. All original artifacts were verified unchanged after analysis. Raw predictions, exact inputs/token IDs, per-object metrics, bootstrap arrays, replay proofs and code snapshots remain in project storage through the ignored `filter-aware-runs` link. Compact results and figures are mirrored in this report directory.",'',
        '```bash',f'PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/filter_metadata_counterfactual.py prepare --output-dir {out}',
        f'PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/supervise_filter_metadata_counterfactual.py --output-dir {out}',
        f'PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/filter_metadata_counterfactual.py report --output-dir {out}',
        '```','', 'Prepare requires a new directory; completed prediction chunks are reused on an interrupted inference restart. Stop/check an existing supervisor before resuming; the supervisor also holds an exclusive process lock. No target calibration, fitting, condition selection, unrelated test evaluation or SFT occurred.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    compact=ROOT/'reports/filter-aware/1027'/out.name;compact.mkdir(parents=True,exist_ok=True)
    for name in ['report.md','results.json','summary.csv','manifest.json','evaluation_policy_v1.json','metadata_mappings_v1.json','tokenization_audit.json','archived_baseline_reproduction.json']:
        shutil.copy2(out/name,compact/name)
    shutil.copytree(out/'figures',compact/'figures',dirs_exist_ok=True)
    check_protected(m,full=True)
    write(out/'complete.json',dict(finished_utc=now(),n=len(rows),conditions=list(mappings['conditions']),report=str(compact/'report.md'),original_artifacts_unchanged=True,training=False))
    print('REPORT COMPLETE',compact/'report.md',flush=True)
