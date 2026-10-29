#!/usr/bin/env python3
"""Scientific collection of independent Stage-1 probes and joint stage assessment."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import stage1_physics_benchmark as b


def load_csv(path):return list(csv.DictReader(Path(path).open()))


def metrics(pred,mask):
    if 'logits' in pred:return b.d.metrics(pred,mask)
    import torch
    from aion_magnitude.metrics import point_photoz_metrics
    return dict(n=int(mask.sum()),**point_photoz_metrics(torch.from_numpy(pred['z_p50'][mask]),torch.from_numpy(pred['z_spec'][mask])))


def reasoning(out):
    cases=json.loads((out/'reasoning_cases.json').read_text());baseline=[]
    rests={'hydrogen Balmer series limit':3646.,'hydrogen Lyman-alpha':1216.,'hydrogen Lyman series limit':912.,'hydrogen H-alpha':6563.}
    for c in cases:
        if c['kind'] in ('explicit_break_arithmetic','generic_arithmetic','named_feature_without_rest_value'):
            rest=rests[c['feature']] if 'feature' in c else c['rest'];answer=c['observed']/rest-1;index=int(np.argmin([abs(float(x)-answer) for x in c['options']]))
        elif c['kind']=='break_ambiguity':index=c['options'].index('Both interpretations are possible')
        elif c['kind']=='physical_consistency':index=c['options'].index('Yes' if abs(c['observed']/c['rest']-1-c['supplied_z'])<1e-6 else 'No')
        else:index=c['options'].index('Both f_nu values are equal' if c['flat_fnu'] else 'The redder filter has larger f_nu')
        baseline.append(dict(case_id=c['id'],kind=c['kind'],prediction=index,correct=index==c['correct_index']))
    assert all(r['correct'] for r in baseline);b.d.write(out/'reasoning_rule_baseline.json',baseline)
    rows=[];raw={}
    for model in ('base','instruct','random'):
        raw[model]=json.loads((out/f'reasoning-{model}/results.json').read_text())
        for kind in ['all',*sorted({r['kind'] for r in raw[model]})]:
            for few in (False,True):
                rr=[r for r in raw[model] if r['few_examples']==few and (kind=='all' or r['kind']==kind)];n=len(rr);correct=sum(r['correct'] for r in rr);p=correct/n;den=1+1.96**2/n;center=(p+1.96**2/(2*n))/den;half=1.96*np.sqrt(p*(1-p)/n+1.96**2/(4*n*n))/den
                rows.append(dict(model=model,kind=kind,few_examples=few,n=n,correct=correct,accuracy=p,wilson95_low=center-half,wilson95_high=center+half))
    b.d.table(out/'reasoning_comparison.csv',rows)
    return rows


def collect(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from astropy.cosmology import Planck18
    m=b.check(out);assert json.loads((out/'fit_status.json').read_text())['phase']=='complete'
    rec=[json.loads(l) for l in (out/'records.jsonl').read_text().splitlines()];data=b.d.npz(out/'photometry.npz');rrows=reasoning(out)
    prediction_files=sorted((out/'predictions').glob('*.npz'));preds={p.stem:b.d.npz(p) for p in prediction_files};allrows=load_csv(out/'performance.csv')+load_csv(out/'physical_baseline_performance.csv')
    source_fit=[dict(model=name,stratum='source_training_fit_not_independent',**metrics(pred,data['split']=='train')) for name,pred in preds.items()]
    b.d.table(out/'source_fit_metrics.csv',source_fit)
    if (out/'tabicl_performance.csv').exists():allrows+=load_csv(out/'tabicl_performance.csv')
    grouped=defaultdict(list)
    for r in allrows:grouped[r['model'],r['stratum']].append(r)
    aggregate=[]
    for (model,stratum),rr in grouped.items():
        row=dict(model=model,stratum=stratum,n=int(rr[0]['n']),seeds=len(rr))
        for key in ['nmad','catastrophic_outlier_fraction','bias','cross_entropy','mean_crps','p16_p84_coverage']:
            vals=[float(x[key]) for x in rr if x.get(key) not in (None,'')]
            if vals:row[key]=float(np.mean(vals));row[key+'_seed_std']=float(np.std(vals))
        aggregate.append(row)
    b.d.table(out/'aggregate_performance.csv',aggregate)
    conditional=[];paired=[];pair_summary=[]
    correct=np.array([r['variant']=='correct' for r in rec]);lookup={(r['latent_id'],r['system'],r['variant']):r['index'] for r in rec}
    contrast=[('HSC_to_Legacy','HSC','correct','Legacy','correct',0.),('HSC_to_CLAUDS','HSC','correct','CLAUDS','correct',0.),('HSC_to_new_filters','HSC','correct','new_filters','correct',0.),('wrong_metadata','CLAUDS','correct','CLAUDS','wrong_metadata',0.),('missing_bands','CLAUDS','correct','CLAUDS','missing_bands',0.),('extra_noise','CLAUDS','correct','CLAUDS','noisy',0.),('redshift_plus03','HSC','correct','HSC','redshift_plus03',.3)]
    latents=sorted({r['latent_id'] for r in rec if r['split']=='test'})
    for name,pred in preds.items():
        for split in ('test','family_holdout'):
            for system in b.f.SYSTEMS:
                base=np.array([r['split']==split and r['system']==system and r['variant']=='correct' for r in rec]);groups={}
                for fam in sorted({r['family'] for r in rec if r['split']==split}):groups['family_'+fam]=np.array([r['family']==fam for r in rec])
                for lo,hi in [(0,.75),(.75,1.5),(1.5,2.5),(2.5,4.5)]:groups[f'z_{lo}_{hi}']=(data['z']>=lo)&(data['z']<hi)
                sn=np.array([r['sn_nuisance'] for r in rec]);groups['sn_10_30']=(sn>=10)&(sn<30);groups['sn_30_100']=(sn>=30)&(sn<=100)
                for group,mask in groups.items():
                    keep=base&mask
                    if keep.any():conditional.append(dict(model=name,split=split,system=system,condition=group,**metrics(pred,keep)))
        for cname,s1,v1,s2,v2,expected in contrast:
            these=[]
            for latent in latents:
                i=lookup[latent,s1,v1];j=lookup[latent,s2,v2];assert abs((data['z'][j]-data['z'][i])-expected)<1e-6
                delta=float(pred['z_p50'][j]-pred['z_p50'][i]);row=dict(model=name,contrast=cname,latent_id=latent,sed_id=rec[i]['sed_id'],z_initial=float(data['z'][i]),expected_shift=expected,predicted_shift=delta,response_error=delta-expected,source_prediction=float(pred['z_p50'][i]),target_prediction=float(pred['z_p50'][j]));these.append(row);paired.append(row)
            pair_summary.append(dict(model=name,contrast=cname,n=len(these),mean_shift=float(np.mean([r['predicted_shift'] for r in these])),mean_abs_shift=float(np.mean([abs(r['predicted_shift']) for r in these])),mean_abs_response_error=float(np.mean([abs(r['response_error']) for r in these])),positive_shift_fraction=float(np.mean([r['predicted_shift']>0 for r in these]))))
    b.d.table(out/'conditional_physics_metrics.csv',conditional);b.d.table(out/'paired_counterfactuals.csv',paired);b.d.table(out/'paired_response_summary.csv',pair_summary)
    # SED-cluster bootstrap for the primary (predeclared seed 11) contrasts.
    bootrows=[];rng=np.random.default_rng(b.SEED);test_ids=np.unique(data['sed_id'][data['split']=='test'])
    for system in ('HSC','CLAUDS','new_filters'):
        mask=np.array([r['split']=='test' and r['system']==system and r['variant']=='correct' for r in rec]);ix=np.flatnonzero(mask)
        for reference in ('random_last_seed11','numeric_set_mlp_seed11','physical_template'):
            samples=[]
            for _ in range(400):
                sed_sample=rng.choice(test_ids,len(test_ids),replace=True);idx=np.concatenate([ix[data['sed_id'][ix]==ident] for ident in sed_sample]);truth=data['z'][idx]
                a=np.abs((preds['base_last_seed11']['z_p50'][idx]-truth)/(1+truth))>.15;bb=np.abs((preds[reference]['z_p50'][idx]-truth)/(1+truth))>.15;samples.append(float(a.mean()-bb.mean()))
            bootrows.append(dict(system=system,contrast='base_last_seed11 minus '+reference,metric='outlier_fraction_difference',bootstrap_unit='SED identity',median=float(np.median(samples)),ci95_low=float(np.quantile(samples,.025)),ci95_high=float(np.quantile(samples,.975)),n_seds=len(test_ids)))
    b.d.table(out/'cluster_bootstrap.csv',bootrows)
    # Input-length and nuisance audit: a shortcut cannot use source survey/count
    # because both source systems observe every labelled SED/redshift instance.
    lengths=b.d.npz(out/'embeddings-base.npz')['lengths'];length_rows=[]
    for system in b.f.SYSTEMS:
        keep=np.array([r['system']==system and r['variant']=='correct' for r in rec]);length_rows.append(dict(system=system,n=int(keep.sum()),min_tokens=int(lengths[keep].min()),max_tokens=int(lengths[keep].max()),unique_lengths=len(np.unique(lengths[keep])),brightness_rms_min=float(np.sqrt(np.nanmean(data['normalized_flux'][keep]**2,axis=1)).min()),brightness_rms_max=float(np.sqrt(np.nanmean(data['normalized_flux'][keep]**2,axis=1)).max())))
    b.d.table(out/'shortcut_audit.csv',length_rows)
    b.d.write(out/'forward_model_audit.json',dict(wavelength_unit='Angstrom; registry nm multiplied by ten',flux_unit='AB nanomaggy; one nMgy=3.631e-29 erg/s/cm2/Hz',ab_reference_jy=3631.,cosmology={k:str(v) for k,v in Planck18.to_format('mapping').items()},flambda_observed='Llambda(lambda_observed/(1+z))/(4*pi*DL**2*(1+z))',fnu_band='integral flambda*lambda*R dlambda / (c*integral R/lambda dlambda)',filter_grid='actual response knots union 5-Angstrom grid, no pivot approximation',luminosity_nuisance='raw physical independent luminosities retained; supervised features remove amplitude using observed RMS only',shared_empirical_library='Empirical base templates shared by primary instance splits; full emergent spectra have unique dust/slope parameters. This is not an empirical-library-template-disjoint test. Double-break/emission family is entirely held out.',noise='independent Gaussian controlled exposure/SNR, not a calibrated survey/image simulator',igm='simplified line-screen/continuum attenuation approximation, not a validated full population IGM model'))
    # Comparison plots use the predeclared primary seed, not the best seed.
    chosen=['base_last_seed11','random_last_seed11','numeric_set_mlp_seed11','physical_template','tabicl'];chosen=[n for n in chosen if n in preds]
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,system in zip(axes,['HSC','CLAUDS','new_filters']):
        keep=np.array([r['split']=='test' and r['system']==system and r['variant']=='correct' for r in rec])
        for name in chosen:ax.scatter(data['z'][keep],preds[name]['z_p50'][keep],s=13,alpha=.5,label=name)
        ax.plot([0,4.5],[0,4.5],'--',c='grey');ax.set(title=system,xlabel='Synthetic true redshift',ylabel='Predicted median / scalar',xlim=(0,4.5),ylim=(0,4.5))
    axes[0].legend(fontsize=6)
    for ext in ('png','pdf'):fig.savefig(out/f'cross_filter_predictions.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
    for name in chosen:
        rr=[r for r in paired if r['model']==name and r['contrast']=='HSC_to_new_filters'];axes[0].scatter([r['source_prediction'] for r in rr],[r['target_prediction'] for r in rr],s=12,alpha=.5,label=name)
        rr=[r for r in paired if r['model']==name and r['contrast']=='redshift_plus03'];axes[1].hist([r['predicted_shift'] for r in rr],bins=np.linspace(-2,2,50),histtype='step',label=name)
    axes[0].plot([0,4.5],[0,4.5],'--',c='grey');axes[0].set(xlabel='HSC prediction',ylabel='New-only filters prediction',title='Same SED and true redshift');axes[1].axvline(.3,c='black',ls='--');axes[1].set(xlabel='Predicted redshift change',ylabel='Objects',title='True change +0.3');axes[0].legend(fontsize=6)
    for ext in ('png','pdf'):fig.savefig(out/f'paired_invariance_and_response.{ext}',dpi=170)
    plt.close(fig)
    params=json.loads((out/'sed_parameters.json').read_text());spectra=b.d.npz(out/'spectra.npz');fig,axes=plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
    for family in sorted({p['family'] for p in params}):
        i=next(i for i,p in enumerate(params) if p['family']==family);axes[0].plot(spectra['rest_wavelength_angstrom'],spectra['emergent_lnu_shape'][i],label=family)
    axes[0].set(xscale='log',yscale='log',xlim=(600,25000),xlabel='Rest wavelength (Angstrom)',ylabel='Relative emergent Lnu');axes[0].legend(fontsize=7)
    for name,r in b.f.responses(ROOT).items():axes[1].plot(r['wavelength'],r['response']/r['response'].max(),label=name,lw=.8)
    axes[1].set(xscale='log',xlabel='Observed wavelength (Angstrom)',ylabel='Normalized response');axes[1].legend(fontsize=5,ncol=2)
    for ext in ('png','pdf'):fig.savefig(out/f'sed_and_filter_families.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
    for model in ('base','instruct','random'):
        rr=[r for r in rrows if r['model']==model and r['kind']=='all'];axes[0].plot([int(r['few_examples']) for r in rr],[r['accuracy'] for r in rr],'o-',label=model)
    axes[0].axhline(.25,ls='--',c='grey');axes[0].set(xticks=[0,1],xticklabels=['No example','Two examples'],ylabel='Constrained answer accuracy',ylim=(0,1));axes[0].legend()
    for name in ['base_last','random_last','numeric_set_mlp','base_last_shuffled_labels','physical_template','tabicl']:
        rr=[next((r for r in aggregate if r['model']==name and r['stratum']==f'test_{system}_correct'),None) for system in ('HSC','CLAUDS','new_filters')]
        if all(r is not None for r in rr):axes[1].plot([0,1,2],[r['catastrophic_outlier_fraction'] for r in rr],'o-',label=name)
    axes[1].set(xticks=[0,1,2],xticklabels=['HSC','CLAUDS system','New-only filters'],ylabel='Catastrophic outlier fraction',ylim=(0,1));axes[1].legend(fontsize=6)
    for ext in ('png','pdf'):fig.savefig(out/f'knowledge_and_transfer_comparison.{ext}',dpi=170)
    plt.close(fig)
    make_report(out,m,aggregate,rrows,pair_summary)
    original=b.d.check(ROOT/'reports/stage0/1027a')
    b.d.write(out/'final_provenance.json',dict(code_sha256={str(p.relative_to(ROOT)):b.d.sha(p) for p in [ROOT/'scripts/stage1_physics_benchmark.py',ROOT/'scripts/stage1_sed_forward.py',ROOT/'scripts/stage1_tabicl_control.py',Path(__file__)]},stage05_completed_before_stage1=True,stage05_gate=m['stage05_gate'],archived_protected_files_verified=len(original['protected_sha256']),backbones_updated=False,no_real_clauds_labels=True,held_out_real_test_evaluated=False))
    b.d.write(out/'summary.json',dict(stage='1',phase='complete',records=m['records'],sed_count=m['sed_count'],supervised_source_observations=512,trainable_backbone=False,head_seeds=m['config']['probe_seeds'],reasoning=rrows,aggregate_performance=aggregate,real_clauds_success_claimed=False))
    b.d.write(out/'artifact_inventory.json',{str(p.relative_to(out)):dict(sha256=b.d.sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='artifact_inventory.json' and p.suffix!='.log'})
    print(out/'report.md')


def make_report(out,m,aggregate,reasoning_rows,paired):
    by={(r['model'],r['stratum']):r for r in aggregate}
    def table(rows):
        text=['| Model | Evaluation | N | NMAD | Outliers | Bias | CE | CRPS | Coverage |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            val=lambda k:f'{r[k]:.4f}' if k in r else '—'
            text.append(f"| {r['model']} | {r['stratum']} | {r['n']} | {r['nmad']:.4f} | {r['catastrophic_outlier_fraction']:.2%} | {r['bias']:+.4f} | {val('cross_entropy')} | {val('mean_crps')} | {100*r['p16_p84_coverage']:.1f}% |" if 'p16_p84_coverage' in r else f"| {r['model']} | {r['stratum']} | {r['n']} | {r['nmad']:.4f} | {r['catastrophic_outlier_fraction']:.2%} | {r['bias']:+.4f} | — | — | — |")
        return '\n'.join(text)
    main=['base_last','random_last','numeric_set_mlp','base_last_shuffled_labels','physical_template','numeric_extra_trees','tabicl']
    compare=[by[name,f'test_{system}_correct'] for system in ['HSC','CLAUDS','new_filters'] for name in main if (name,f'test_{system}_correct') in by]
    unseen_family=[by[name,f'family_holdout_{system}_correct'] for system in ['CLAUDS','new_filters'] for name in ['base_last','random_last','numeric_set_mlp','physical_template','tabicl'] if (name,f'family_holdout_{system}_correct') in by]
    rr=[r for r in reasoning_rows if r['kind']=='all'];rtable=['| Checkpoint | Examples | Correct / 48 | Accuracy | Wilson 95% interval |','|---|---:|---:|---:|---:|']
    for r in rr:rtable.append(f"| {r['model']} | {2 if r['few_examples'] else 0} | {r['correct']} | {r['accuracy']:.2%} | {r['wilson95_low']:.2%}–{r['wilson95_high']:.2%} |")
    ptable=['| Model (predeclared seed 11 where applicable) | Contrast | Mean absolute prediction shift | Mean absolute response error |','|---|---|---:|---:|']
    for r in paired:
        if r['model'] in ('base_last_seed11','random_last_seed11','numeric_set_mlp_seed11','physical_template','tabicl') and r['contrast'] in ('HSC_to_CLAUDS','HSC_to_new_filters','wrong_metadata','redshift_plus03'):ptable.append(f"| {r['model']} | {r['contrast']} | {r['mean_abs_shift']:.4f} | {r['mean_abs_response_error']:.4f} |")
    base=by['base_last','test_new_filters_correct'];random=by['random_last','test_new_filters_correct'];numeric=by['numeric_set_mlp','test_new_filters_correct'];physical=by['physical_template','test_new_filters_correct']
    source_fit_map={r['model']:r for r in load_csv(out/'source_fit_metrics.csv')}
    text=f'''# Stage 1 — Controlled physics, pretraining and cross-filter transfer

This is a small **synthetic** benchmark. It does not establish success on real
CLAUDS. Stage 0.5 completed and passed its correctness gate before Stage 1 began.
No real CLAUDS redshift labels or held-out real-survey test objects are used.
Only small source-synthetic probes are trained; Qwen backbones stay frozen.

## Physical benchmark and controls

The fixed benchmark has **208 unique emergent SED instances**, 1,312 observations,
and 512 labelled source observations (128 SEDs x two redshifts x HSC/Legacy).
Validation has 24 SEDs; primary test has 32; the double-break/emission family has
24 additional held-out SEDs. An ID split precedes every redshift, noise or filter
realization. No complete emergent spectrum repeats across splits. Primary
empirical instances share CWW base-library templates but have independent
continuous dust/slope parameters; this is not a base-library-template-disjoint
test. The entire double-break/emission combination is unseen in source training.

Families include a 4000-Angstrom break, UV/Lyman continuum, weak-break controls,
four pinned extended CWW empirical galaxy templates, and a held-out family with
Balmer/4000 breaks plus emission lines. Continuum slopes, break strengths,
power-law dust attenuation and independent luminosities are explicit. Mean IGM
attenuation is a documented simplified line/ionizing screen, not a validated
survey population model. The CWW files and source commit/hashes are vendored
unchanged from the [EAZY template repository](https://github.com/gbrammer/eazy-photoz/tree/master/templates/CWW%2BKIN).

Spectra are redshifted with the luminosity-distance factor, then fully integrated
through actual pinned photon-response curves, using original knots plus a
5-Angstrom grid. AB normalization uses 3631 Jy; f_lambda*lambda*R weighting and
the c*integral(R/lambda) denominator preserve constant f_nu. These conventions
follow the [photon-response convolution documentation](https://speclite.readthedocs.io/en/latest/api/speclite.filters.FilterResponse.html)
and [cosmological distance relations](https://ned.ipac.caltech.edu/level5/Hogg/Hogg7.html).
Seven forward tests and four leakage/counterfactual checks pass. A smooth flat
f_nu continuum has no low-z colour-redshift information after amplitude removal;
poor predictions there can reflect genuine non-identifiability.

All raw fluxes/errors, clean measurements, spectra, nuisances, response samples
and seeds are saved. Independent Gaussian measurement errors use a redshift-
independent RMS-S/N nuisance, not a calibrated survey image/exposure model.
Supervised inputs remove overall amplitude using observed RMS, retaining signed
fluxes and relative errors. Physical luminosities are independent of redshift;
survey identities and intrinsic truth/SED IDs never enter model strings. Source
filter systems observe exactly the same SED/redshift instances, so source survey
or record length alone cannot determine the label. Band order is permuted.
`shortcut_audit.csv` documents token-length and amplitude checks. New-only filters
are all absent from source training. Different target band counts/coverage remain
part of the transfer challenge and prevent attribution to bandpass identity alone.

## Probe A — Explicit reasoning before photo-z probe fitting

The interface scores four one-token option completions with the tied LM
unembedding. It is a constrained completion test, not free-form generation.
No-example and fixed two-example conditions are separate. Base and compatible
Instruct checkpoints are separate; instruction tuning is not astrophysical
pretraining. No prompt was tuned on these answers.

{chr(10).join(rtable)}

The explicit arithmetic/rule baseline scores 48/48, using supplied wavelengths,
known named-line rest wavelengths, the redshift relation, ambiguity rules and AB
filter normalization. Task-specific results and every correct/incorrect model
answer, logits, actual input IDs/masks and prompt are saved. Pretraining improves
some constrained completions over the full random architecture, particularly
ambiguity recognition; the named-feature tasks are small and inconsistent.
These results do not establish reliable astrophysical numerical reasoning.
Base-checkpoint prompt sensitivity does not establish absence of information.

## Probe B/C — Frozen representations and held-out filter systems

The full frozen random text architecture has 4.206 billion parameters, matching
the pretrained text stack; the unused vision branch is omitted. Both use NF4
double quantization with BF16 compute, identical tokenizer and 2,560-dimensional
representations. Primary pooling is a fixed ordinary terminal query; mean pooling
is a predeclared secondary condition, not chosen by target performance.

Pretrained/random heads are identical 2560->32->90 MLPs. The numerical control
uses shared per-band MLP/set aggregation and a parameter-matched head (see exact
budgets). Both receive normalized photometry, uncertainties, response summaries
and masks. Each uses the same 512 source observations, 400 AdamW updates, batch
64, learning rate 0.002 and three predeclared initialization seeds. All feature
normalization is fitted on source training only. No target-filter labels choose
hyperparameters, updates, checkpoint or pooling. Label-shuffled pretrained-head,
scalar ExtraTrees and frozen in-context TabICL controls are included.

The source-fit diagnostic matters. At predeclared seed 11, pretrained and random
last-pool heads achieve source-training CE
{float(source_fit_map['base_last_seed11']['cross_entropy']):.4f} and
{float(source_fit_map['random_last_seed11']['cross_entropy']):.4f}, with 0/512
catastrophic outliers in both cases. They fit the supervised examples but fail
to generalize well across independent SED instances. The numerical set MLP has
training CE {float(source_fit_map['numeric_set_mlp_seed11']['cross_entropy']):.4f};
its effective capacity/optimization differs despite matched parameter and update
budgets. Thus parameter matching does not guarantee identical fit quality. This
measures sample/inductive-bias behaviour under a fixed small protocol, not an
optimally tuned comparison of every architecture. No evaluation labels are used
to fix this after seeing results. Training fits are not independent validation.

The physical baseline profiles a nonnegative luminosity scale over 20 fixed
analytic/CWW prototypes and 90 redshift bins, using the actual supplied response
curves. Its prototype grid is independent of test SED identities. Its PDF is an
equal-template profile-likelihood pseudo-posterior, not a calibrated Bayesian
posterior. Scalar baselines do not receive invented CE/CRPS/coverage metrics.

The following values are means over three head seeds where applicable; seed
dispersion is in aggregate_performance.csv, and uncertainty from SED sampling is
addressed by cluster_bootstrap.csv. Two redshifts per test SED are not independent
galaxy-template draws.

{table(compare)}

On the strict new-only filters, primary pretrained outliers are
**{base['catastrophic_outlier_fraction']:.2%}**, compared with
**{random['catastrophic_outlier_fraction']:.2%}** for full random features,
**{numeric['catastrophic_outlier_fraction']:.2%}** for the numerical MLP, and
**{physical['catastrophic_outlier_fraction']:.2%}** for the physical baseline.
These empirical differences support only the tested small protocol; three head
seeds do not cover random-backbone initialization or all possible SED populations.
The one random-backbone initialization and limited labels are explicit limits.
The primary last-pool head has a modest average advantage over random features
on the held-out systems, but the predeclared seed-11 SED-cluster intervals include
zero for those outlier differences. It does not beat the numerical MLP overall,
and the secondary mean-pool comparison does not show the same pretraining gain.
Thus there is no robust, general pretrained-representation advantage established
by this small study. The physical baseline performs substantially better, while
also having access to the complete forward law/response curves; learned probes
receive response summaries, not every response sample in their feature vectors.

The stronger unseen-family test gives:

{table(unseen_family)}

![Spectra and filters](sed_and_filter_families.png)
![Source and unseen filters](cross_filter_predictions.png)
![Reasoning and transfer](knowledge_and_transfer_comparison.png)

## Physical counterfactuals

Paired observations retain the same intrinsic SED, luminosity, dust and redshift;
only responses/noise realizations change. Wrong metadata leaves actual fluxes and
errors identical. Redshift-response pairs instead keep the intrinsic SED fixed
and regenerate photometry at z+0.3. Neither ID, seed nor the answer is provided.

{chr(10).join(ptable)}

![Invariance and redshift response](paired_invariance_and_response.png)

A response to incorrect metadata alone is not evidence of physics. The paired
invariance and known-redshift-change tests, together with template and random
controls, delimit that interpretation. `conditional_physics_metrics.csv`
contains results by SED family, redshift and S/N; the held-out double-break family
is reported independently. All noise/missing-band and secondary-pooling results
remain in the machine-readable tables without error-based filtering.
Even a valid physical estimator can shift its posterior median when the noise
realization, coverage or degeneracy changes. The invariant quantity here is the
true redshift; exact equality of noisy point estimates is not a universal physics
criterion. The template comparison helps expose those information limits.

## Claims and the four requested answers

1. **Usable physical information under an interface:** some constrained explicit
   physical-consistency/ambiguity responses are usable. The complete table and
   cross-filter tests define the narrower numerical limits. This is not a
   reliable general physical solver or proof of real-CLAUDS transfer.
2. **Benefit from pretraining:** pretrained-versus-random head tables directly
   show a limited, pooling-dependent gain on some held-out filters, without a
   robust advantage over numerical controls. Reasoning differs from the random control before any
   supervised photo-z head training. Those comparisons isolate general
   pretraining effects within this protocol, not astrophysical training content.
3. **Transfer to unseen responses:** report the strict new-only and CLAUDS-system
   errors and paired shifts above: roughly 70% primary-head outliers do not
   demonstrate reliable cross-filter redshift transfer. Correct source accuracy
   or sensitivity to shuffled tokens would not by itself establish invariance.
4. **Stage-2 justification:** these tests motivate controlled trials of explicit
   filter integration, masked set-like band representations and stable pooling.
   They do not justify an expensive Qwen-specific physics architecture solely on
   presumed pretrained astrophysical knowledge. Retain the full random-feature,
   numerical, shuffled-label and template controls and a frozen-backbone branch.

**Claim 1 (physical behaviour)** is task/interface dependent and constrained by
the explicit and paired measurements. **Claim 2 (a pretraining benefit)** can be
assessed from matched head differences, with sampling limits. **Claim 3 (specifically
astrophysical pretraining causes the benefit)** is not established: generic
numeric/text representations and ambiguity heuristics remain explanations, and
named-feature retrieval is not consistently strong. Successful synthetic
supervised regression alone would not establish this stronger claim.

## Reproduction

manifest.json fixes all simulation/split/probe choices before model outputs;
the preflight q50 bookkeeping change gave numerical controls the same response
summaries as Qwen before any model probe. Actual commands, environment versions,
template/response hashes and random initialization hashes are saved. Final
provenance verifies archived checkpoints and records the prior-stage gate.
No original adapter, trained PDF head, validation split or historical result is
overwritten. No SFT, QLoRA, DoRA or large-model weight training is performed.
'''
    (out/'report.md').write_text(text)
    stage05=json.loads((ROOT/'reports/stage05/1027/summary.json').read_text());s05={r['condition']:r for r in stage05['performance']}
    joint=f'''# Joint scientific assessment — Stage 0.5 followed by Stage 1

Both experiment stages completed sequentially. Reports:
[Stage 0.5](stage05/1027/report.md) and [Stage 1](stage1/1027/report.md).

**Current real-data failure mechanisms.** On all 5,683 source-validation objects,
native outliers are {s05['full_native']['catastrophic_outlier_fraction']:.2%},
median-EBV outliers {s05['full_ebv_median']['catastrophic_outlier_fraction']:.2%},
NA-EBV outliers {s05['full_ebv_na']['catastrophic_outlier_fraction']:.2%}, and moving
true EBV to the front gives {s05['full_ebv_front']['catastrophic_outlier_fraction']:.2%}.
This pattern is much more consistent with substantial syntax/terminal-state
fragility than a need for each galaxy's exact EBV value, while not uniquely
separating token identity, position and semantics. Dimming at fixed S/N gives
{s05['full_dim7']['catastrophic_outlier_fraction']:.2%} outliers and median
z={s05['full_dim7']['predicted_median']:.4f}; uncertainty-reporting changes alone
are much milder in screening. Actual noisy-faint cases remain separate.
The direction under true EBV is predominantly toward ~0.91, not the archived
CLAUDS ~0.08–0.13 mode. Thus source interventions reproduce severe concentration
but do not uniquely explain the real-domain collapse or its full direction.
These are source-condition sensitivities, not additive causal shares of CLAUDS
outliers. Remaining extinction/calibration, extraction-pipeline and population
differences, and the target's ZPHOT reference quality, remain unresolved.

**Pretrained knowledge and cross-filter tasks.** Stage 1 quantifies constrained
reasoning and the matched frozen-representation advantages without assuming them.
On strictly new filters, pretrained/random/numerical/physical outliers are
{base['catastrophic_outlier_fraction']:.2%}/{random['catastrophic_outlier_fraction']:.2%}/
{numeric['catastrophic_outlier_fraction']:.2%}/{physical['catastrophic_outlier_fraction']:.2%}.
These controlled outcomes do not establish astrophysical pretraining as the
specific cause of any benefit, or success on real CLAUDS. Last-pool gains over
random features are modest and sampling-uncertain, mean pooling does not repeat
them, and the numerical/physical controls remain stronger. Both Qwen feature
heads can memorize the small source training set; zero training outliers do not
translate to physical generalization. Template/sampling/noise
approximations, one random backbone and small independent test-SED counts remain.

**Stage 2 priorities.** First test mask-aware shared band encoders with explicit
response integration and luminosity treated as a nuisance or separately declared
prior, including a rest-frame SED basis with a differentiable band-integration
decoder; compare a numerical set MLP and physical/template baseline. Use a stable
query/set pooling mechanism trained under a matched protocol, with explicit EBV
availability separated from its value. Keep pretrained and matched random frozen
backbones, matched parameter/label budgets, shuffled labels, wrong-metadata and
physically regenerated-response controls, plus held-out SED families and real
survey validation. Do not infer an astrophysical-knowledge advantage from a
supervised fit or shuffled-token degradation. An expensive adapter sweep is not
the next justified default. No Stage-2 model was trained in this execution.
'''
    joint_path=ROOT/'reports/joint_physics_1027.md' if out.resolve()==b.OUT.resolve() else out/'joint_assessment.md'
    joint_path.write_text(joint)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=b.OUT);collect(parser.parse_args().output_dir)
