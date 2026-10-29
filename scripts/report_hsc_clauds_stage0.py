#!/usr/bin/env python3
"""Collect supplementary photometry plots and a concise Stage-0 report."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import diagnose_hsc_clauds_stage0 as d
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT/'reports/stage0/1027a');args=p.parse_args();out=args.output_dir
    manifest=d.check(out);support=json.loads((out/'support_summary.json').read_text());s=d.npz(out/'source_training_photometry.npz');t=d.npz(out/'target_photometry.npz');a=d.npz(out/'support_assignments.npz')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    rows=[];soft=np.nanmedian(s['sigma'],axis=0)
    for name,data in [('source_training',s),('target_validation',t)]:
        for j,b in enumerate(d.BANDS):
            f,e=data['flux'][:,j],data['sigma'][:,j];good=f>0;mag=22.5-2.5*np.log10(f[good])
            rows.append(dict(domain=name,band=b,n=len(f),valid_flux=int(np.isfinite(f).sum()),nonpositive_flux=int(np.sum(np.isfinite(f)&(f<=0))),flux_q05=float(np.nanquantile(f,.05)),flux_median=float(np.nanmedian(f)),flux_q95=float(np.nanquantile(f,.95)),sigma_median=float(np.nanmedian(e)),positive_magnitude_median=float(np.median(mag)),sn_median=float(np.nanmedian(f/e))))
    d.table(out/'photometric_distribution_summary.csv',rows)
    fig,axs=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for j in range(4):
        for label,data in [('HSC training',s),('CLAUDS',t)]:
            colour=np.diff(np.arcsinh(data['flux']/soft),axis=1)[:,j]
            axs[0,j].hist(colour,bins=np.linspace(-5,5,55),density=True,histtype='step',label=label)
            logerr=np.log10(data['sigma'][:,j]);axs[1,j].hist(logerr,bins=np.linspace(-5,2,55),density=True,histtype='step',label=label)
        axs[0,j].set(xlabel=f'asinh colour {d.BANDS[j]}-{d.BANDS[j+1]}',ylabel='Density')
        axs[1,j].set(xlabel=f'log10 sigma {d.BANDS[j]} (nMgy)',ylabel='Density')
    axs[0,0].legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(out/f'colours_and_uncertainties.{ext}',dpi=170)
    plt.close(fig)
    neighbour_rows=[]
    for name in ('flux_colour','joint_photometry'):
        spec=support['diagnostics'][name]['folds'][0]['preprocessor'];fit=d.npz(out/f'{name}_fold0.npz')['fit_source_indices'];sx=d.transform(s['flux'][fit],s['sigma'][fit],spec);tx=d.transform(t['flux'],t['sigma'],spec)
        distances,indices=NearestNeighbors(n_neighbors=10,n_jobs=2).fit(sx).kneighbors(tx)
        for i,ident in enumerate(t['targetid']):
            for rank in range(10):neighbour_rows.append(dict(diagnostic=name,target_id=str(ident),rank=rank+1,source_training_id=str(s['targetid'][fit[indices[i,rank]]]),distance=float(distances[i,rank])))
    d.table(out/'nearest_source_training_neighbours.csv',neighbour_rows)
    counts=[]
    for feature in ('complete','brightness','sn','quality'):
        for value in np.unique(a[feature]):
            mask=a[feature]==value;counts.append(dict(feature=feature,value=str(value),n=int(mask.sum()),common_support=int(a['common_support'][mask].sum()),flux_colour_pass=int(np.sum(a['flux_colour'][mask]<=1)),joint_pass=int(np.sum(a['joint_photometry'][mask]<=1))))
    d.table(out/'support_stratification.csv',counts)
    import csv
    allrows=list(csv.DictReader((out/'performance_all.csv').open()));conditional=list(csv.DictReader((out/'conditional_performance.csv').open()));controls=list(csv.DictReader((out/'source-ebv-control/comparison.csv').open()))
    byname={r['condition']:r for r in allrows}
    def markdown_table(records):
        lines=['| Condition | N | CE | NMAD | Outliers | Bias | CRPS | 68% coverage | Predicted median |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in records:
            lines.append(f"| {r['condition']} | {r['n']} | {float(r['cross_entropy']):.4f} | {float(r['nmad']):.4f} | {100*float(r['catastrophic_outlier_fraction']):.2f}% | {float(r['bias']):+.4f} | {float(r['mean_crps']):.4f} | {100*float(r['p16_p84_coverage']):.2f}% | {float(r['predicted_median']):.4f} |")
        return '\n'.join(lines)
    source_i=next(r for r in rows if r['domain']=='source_training' and r['band']=='i');target_i=next(r for r in rows if r['domain']=='target_validation' and r['band']=='i')
    replay={arm:json.loads((out/f'arm-{arm}/native_replay.json').read_text()) for arm in ('original','A','B','C')}
    audit='''# Input and unit audit

The source loader reads `HSC_metadata/{g,r,i,z,y}_cmodel_mag` and
`*_cmodel_magsigma` from the published GalaxiesML-Spectra v2 HDF5. The package
maps each valid AB magnitude m to `10**((22.5-m)/2.5)` nMgy and its measured
magnitude uncertainty to `f*ln(10)/2.5*sigma_m`. The original selection requires
finite `0<m<40`, `0<sigma_m<10`, at least three valid bands, and DESI quality
cuts. Invalid pairs become NaN. This magnitude-only source cannot preserve or
recover original nonpositive fluxes. Stage 0 independently reconstructs all
22,444 training rows bit exactly against the archived float32 feature matrix.
Source image and spectral arrays, redshift arrays in the HDF5, and held-out test
photometry are never read. Root IDs are used only to locate training rows.

CLAUDS reads the actual `FLUX_CMODEL_HSC-[G,R,I,Z,Y]` and matching `FLUXERR` fields
specified in band_audit.csv. Its measured errors scale linearly with flux:

```
f_nMgy = 10**((22.5-23.9)/2.5) * f_microJy
       = 0.275422870333817 * f_microJy
m_AB   = 23.9 - 2.5*log10(f_microJy)       # positive flux only
m_old  = 23.0 - 2.5*log10(f_microJy) = m_AB - 0.9
```

The historical optional conversion was too large by `10**0.36=2.29087`.
All native archived serializers remain unchanged. Actual finite signed/zero
fluxes survive. Non-finite/-99 flux or `notObserved`/`isNoData` means missing;
nonpositive/non-finite error means missing error. The four raw flag arrays are
verified against the archive. `hasBadPhotometry` and `isClean` are recorded
without introducing a quality cut. No uncertainty floor or imputation is added.

Both sides use HSC grizy nominal system identifiers. The registry acknowledges
unresolved r/r2 and i/i2 mixtures. cModel describes a profile-fitting measurement,
not aperture photometry; matching that name and filter system does not prove
identical extraction, deblending, calibration, selection, or pipeline versions.
The CLAUDS publication documents forced measurements across bands, microJy
cModel fluxes, and an extinction-uncorrected hscPipe/Phosphoros path. The source
HDF5 exposes cModel magnitudes but no definitive per-column pipeline/extinction
attributes or corresponding imaging-quality flags. No new extinction correction
is applied; source `MW_EBV` is as supplied, target EBV remains unavailable.
The exact local processing histories remain unresolved. See
[Desprez et al. (2023), sections 3.1/4.1 and Table E.2](https://arxiv.org/pdf/2301.13750)
and the [GalaxiesML-Spectra publisher record](https://zenodo.org/records/16989593).

The source field order is interleaved g flux/error, r flux/error, i flux/error,
z flux/error, y flux/error, then MW_EBV. Five decimals, `NA`, the prefix and schema
are reproduced by the existing serializer. It appends no prediction/query/EOS
token; the collator applies the existing tokenizer and last nonpadding token is
pooled. Final numeric EBV in source becomes `NA` in the main target diagnostic;
omission also changes the pooled field. Full tokenizer IDs and final tokens for
photometrically preselected examples are in inputs/. All records fit 512 tokens.

`source_exact_*` uses the original source survey identifier embedded in `schema=`
and original PDR2 field names as **counterfactual diagnostic aliases**. These
are not an assertion of CLAUDS provenance. `target_schema_na` changes only the
schema survey name to CLAUDS, leaving the numerical observations fixed.
There is no independent `survey=` field in the original source schema.
These aliases must not be adopted as a scientifically truthful production
representation without explicit provenance. No missing target EBV is filled.
'''
    (out/'input_audit.md').write_text(audit)
    main_names=['original_archived_native','original_source_exact_na','original_source_exact_omit_ebv','original_target_schema_na','A_native_full','A_source_exact_na','B_native_full','B_flux_hsc','B_source_exact_na','C_native_full','C_flux_hsc','C_source_exact_na']
    common_names=['original_archived_native','original_source_exact_na','original_source_exact_omit_ebv','original_target_schema_na','A_source_exact_na','B_flux_hsc','B_source_exact_na','C_flux_hsc','C_source_exact_na']
    domain=support['domain_classifier'];c=support['common_count']
    sensitivity_masks={'flux_colour_complete':a['complete']&(a['flux_colour']<=1)}
    for diagnostic in ('flux_colour','joint_photometry'):
        distances=d.npz(out/f'{diagnostic}_fold0.npz')
        for q in (.90,.95,.99):
            sensitivity_masks[f'{diagnostic}_fold0_q{q}']=a['complete']&(distances['target_distances']<=np.quantile(distances['source_distances'],q))
    sensitivity_rows=[]
    for name,mask in sensitivity_masks.items():
        for condition in byname:
            if condition=='original_archived_native':path=d.FAST/'clauds_predictions.npz'
            elif condition in ('A_native_full','B_native_full','B_flux_full','B_flux_hsc','C_native_full','C_flux_full','C_flux_hsc'):path=d.COMMON/f'arm-{condition[0]}'/f'{condition}.npz'
            else:
                arm=condition.split('_')[0];variant=condition.removeprefix(arm+'_');path=out/f'arm-{arm}'/f'{variant}.npz'
            if mask.any():sensitivity_rows.append(dict(condition=condition,stratum=name,**d.metrics(d.npz(path),mask)))
    d.table(out/'support_sensitivity_performance.csv',sensitivity_rows)
    collapse=['| Condition | Median z | 5–95% predicted medians | Maximum fraction in width 0.03 |','|---|---:|---:|---:|']
    for name in main_names:
        r=byname[name];collapse.append(f"| {name} | {float(r['predicted_median']):.4f} | {float(r['predicted_p05']):.4f}–{float(r['predicted_p95']):.4f} | {100*float(r['concentration_width_003']):.2f}% |")
    def control_sentence(name):
        r=next(r for r in controls if r['condition']==name)
        return f"NMAD {float(r['nmad']):.4f}, outliers {100*float(r['catastrophic_outlier_fraction']):.2f}%"
    report=f'''# Stage 0 — HSC to CLAUDS diagnostic

The corrected common physical units and exact source field format do not establish
successful transfer. The original frozen model with exact source fields and
explicit missing EBV has {100*float(byname['original_source_exact_na']['catastrophic_outlier_fraction']):.2f}% outliers.
Only **{c}/256 ({100*c/256:.2f}%)** pass the prespecified joint photometric support
criterion. This cohort is overwhelmingly outside the bright, high-S/N source
training distribution. There is also direct evidence of input-interface
sensitivity: on unchanged source validation photometry, replacing EBV by NA
increases outliers from 5.47% to 43.75%. The target failure cannot be attributed
uniquely to population shift while this metadata/final-token effect remains.

## Protocol

All 256 IDs are the original overlap-audited CLAUDS **validation** queries, in
their original order. References are **ZPHOT, not spectroscopic redshifts**.
The original completed QLoRA adapter/PDF head and selected continued-SFT A/B/C
states (updates 2500/2500/2806) are frozen. No optimiser, redshift-model training,
target-label calibration or checkpoint selection is performed. Domain classifiers
are fitted solely to photometry and source/target membership. No held-out test
objects are evaluated. The source cohort used for support is 22,444 training and
5,683 validation objects; test membership/photometry is not loaded.

Source/target unit equivalence and catalogue-column checks are in
[input_audit.md](input_audit.md), [band_audit.csv](band_audit.csv), and
[serializer_comparison.json](serializer_comparison.json). The archived native
loader is untouched, including its historical 0.9-mag inconsistency.

## Photometric support

Features use signed asinh fluxes and adjacent asinh-colour differences. The joint
space additionally includes log absolute errors, asinh S/N, and smooth log
relative errors. Per-band softness is the source-training median error; robust
centres/IQR scales and missing-feature fills are fitted on source training only.
Missingness indicators accompany those fills. These analytical fills never enter
model strings. No logarithm of a nonpositive flux is taken. Error/colour/SN terms
are correlated; distance is an operational diagnostic, not a physical density.

Five source-training folds each use 80% as the neighbour reference and 20% for
calibration. The mean distance to k=10 neighbours is divided by that fold's
source-only 95th-percentile calibration distance. Median ratio across folds <=1
defines each pass; **common support requires complete grizy and both passes**.
Assignments were saved and hashed before new inference or reading evaluation
labels. The source validation pass rates are
{100*support['diagnostics']['flux_colour']['source_validation_pass_fraction']:.2f}%
(flux/colour) and {100*support['diagnostics']['joint_photometry']['source_validation_pass_fraction']:.2f}%
(joint). CLAUDS passes are 7/256 and 2/256, respectively. The main joint criterion
retains {c}; this is a finite-sample, feature-dependent estimate, not literal zero
support elsewhere. Fold-0 sensitivity at source 90/95/99 percentiles retains
{support['diagnostics']['flux_colour']['sensitivity']} in flux/colour and
{support['diagnostics']['joint_photometry']['sensitivity']} in joint space. These
descriptive sensitivities do not change the frozen main assignments.

Median positive-flux i magnitude is **{target_i['positive_magnitude_median']:.2f}**
for CLAUDS versus **{source_i['positive_magnitude_median']:.2f}** in source training.
Median i S/N is **{target_i['sn_median']:.1f}** versus **{source_i['sn_median']:.1f}**.
237 objects have all five bands; missingness alone cannot explain the result.
109 have at least one HSC bad-photometry flag, with no quality filtering.
Brightness and S/N strata use source-training terciles, not target errors.
Completeness/quality strata and all conditional metrics are provided separately.

Two independent fixed domain classifiers use 128 target objects for fitting,
64 for validation, and a separate 64 held out from fitting/validation. Source
training objects have a separate 60/20/20 partition; their preprocessing is fitted
only on the source classifier-training partition. Balanced held-out comparisons
have 64 source and 64 target objects. Logistic AUC is
**{domain['logistic']['heldout']['auc']:.4f}**, bootstrap 95% interval
{domain['logistic']['heldout']['auc_bootstrap_95']}; gradient boosting AUC is
**{domain['hist_gradient']['heldout']['auc']:.4f}**, interval
{domain['hist_gradient']['heldout']['auc_bootstrap_95']}.
No density ratios are inferred from these probabilities; near-perfect
classification does not prove literal zero support. Validation results,
partitions, parameters and probabilities are saved in domain_classifier.json
and domain_partition.npz.

![Features](feature_distributions.png)
![Colours and errors](colours_and_uncertainties.png)
![Support](support_diagnostics.png)

## Fixed-weight serialization experiment

`source_exact_na` preserves source field names, units, order and precision with
`MW_EBV=NA`; `source_exact_omit_ebv` omits that unavailable field. Their source
schema identifier and PDR2 field names are explicitly counterfactual aliases.
`target_schema_na` changes only the embedded survey schema tag. No target
measurement or uncertainty is substituted. B/C source-format evaluations are
deliberate cross-schema stress tests, not their trained filter-aware format.

{markdown_table([byname[n] for n in main_names])}

The complete 20-condition table is performance_all.csv, including full-band
common-flux B/C results and every schema variant. NMAD is the archived
1.48*median absolute centred normalized residual; bias is mean normalized
residual; catastrophic outliers have |delta|>0.15. Metrics use the original
300-bin 0–4.5 PDF conventions. With n=2, NMAD can be zero under the archived
lower-median convention and is not evidence of precision. `conditional_performance.csv`
also records median bias, CRPS, coverage, prediction quantiles/std, modal-bin
fraction, concentration in a 0.03-wide interval, and source-distance ratios.

{chr(10).join(collapse)}

The B/C source-format interventions mostly move the concentration to z~0.92;
they do not restore a broad object-sensitive prediction distribution. CE/coverage
and the outlier fraction must be interpreted alongside these concentration data.

## Common-support evaluation

{markdown_table([r for r in conditional if r['stratum']=='common_support' and r['condition'] in common_names])}

The two common-support IDs are {', '.join(str(x) for x in a['targetid'][a['common_support']])}.
This sample is too small for a general claim about conditional transfer. All
other objects remain in the out-of-support stratum; no object is removed on the
basis of its redshift error. Flux/colour-only support is more permissive and
should not be conflated with comparable measurement noise. Support matching
does not establish invariance of redshift conditional on photometry.
`support_sensitivity_performance.csv` evaluates the previously saved source-only
90/95/99-percentile sensitivity rules and the 7-object complete flux/colour
stratum. These are descriptive checks and do not replace the primary definition.
Within that 7-object flux/colour-only stratum, 5/7 are catastrophic outliers for
the original native/source-exact and B/C HSC common-flux conditions. Thus matching
flux/colour alone does not establish noise equivalence or conditional transfer.

![Conditional predictions](conditional_scatter.png)
![Predicted distributions](conditional_redshift_distributions.png)
![Photometrically preselected PDFs](representative_pdfs.png)

## Source EBV/final-token control

A separate intervention uses the first 256 archived source validation objects,
the original frozen model, and unchanged source photometry. It was specified
before this control's inference; no target label determines its configuration.
These source references are the archived DESI spectroscopic redshifts, unlike
the target ZPHOT references.

{markdown_table(controls)}

The native control gives {control_sentence('source_native')}; changing only EBV
to NA gives {control_sentence('source_missing_ebv')}; omission gives
{control_sentence('source_omit_ebv')}. These control rows quantify sensitivity to
unavailable metadata and the changed last token. They do not supply missing
target EBV or justify inventing a target value.
The large degradation on source photometry establishes a genuine input-interface
failure mode. It does not identify whether the model uses astrophysical EBV,
the final numeric-token statistics, or both; the combined intervention changes
all of those. Omission is also harmful and therefore is not a remedy.

## Decisions A–E

**A. Numerical and semantic compatibility:** measured grizy fluxes can be put in
the same AB-nanomaggy units with audited errors and explicit missingness. The
native magnitude interface has a real historical 0.9-mag discrepancy. Filter
system compatibility is supported, but exact measurement-pipeline/extinction
equivalence is not established by the local metadata. Signed target measurements
must remain signed; source magnitude selection has already lost that information.

**B. Serialization:** the original checkpoint is not rescued. For B/C, exact
source fields with NA reduce outliers from ~97% to ~74–77%, but primarily shift
the concentrated mode from ~0.08–0.10 to ~0.92. Coverage remains ~12–14%, so this
is interface sensitivity rather than successful transfer. EBV
omission and schema-tag changes are separate diagnostics and cannot be chosen
as a production interface using these target labels. Last-token and metadata
sensitivity is assessed by the source control above.

**C. Meaningful photometric support:** the main operational estimate is
{c}/256 ({100*c/256:.2f}%), with 7/256 in the less restrictive flux/colour space.
The strong brightness, noise and domain separability shifts are directly
observed; exact support fractions depend on feature/threshold choices.

**D. Common-support failure:** both retained objects avoid catastrophic errors
under all archived native/common-flux conditions and the original source-exact
condition. B source-exact also has 0/2 outliers, while deliberately applying the
foreign source format to C gives 2/2; changing only B's survey schema gives 1/2.
Thus the dominant archived failure is concentrated outside the measured support.
These two galaxies cannot establish robust conditional invariance or success
across CLAUDS; their coverage still depends on the interface.

**E. Interpretation:** a severe photometric population/noise shift is measured,
and missing-EBV/final-token sensitivity independently degrades source performance.
Both contribute plausible failure mechanisms. Together with unresolved processing
differences, they prevent unique attribution to population shift versus learned
representation limitations. This experiment
does not measure all astrophysical knowledge in the foundation model.

**Stage 1/2:** the corrected optional raw-flux interface is physically consistent
for follow-on controlled diagnostics, provided provenance, units, signed fluxes,
real errors and missing metadata stay explicit. An unconditional claim of a
trustworthy cross-survey-equivalent interface is **not yet warranted**. Before
interpreting new training as generalisation, resolve or explicitly model the
extinction/calibration/pipeline history and EBV/final-token differences, and
evaluate adequate photometric overlap using validation-only protocols. No Stage
1/2 training was started by this task.

## Reproducibility and integrity

commands.sh records actual commands; reproduce.sh creates a fresh directory.
requirements-executed.txt records the isolated runtime. manifest.json contains
object IDs, catalogue row indices via hashed inputs, seed, source configuration,
checkpoint paths/updates, base-model size/mtime fingerprints and SHA-256 hashes
for protected adapters, heads, archives and prepared inputs. Protection checks
pass before/after all four inference arms and collection. Native replay maximum
logit differences are { {k:v['max_abs_logit_error'] for k,v in replay.items()} };
maximum PDF differences are { {k:v['max_abs_pdf_error'] for k,v in replay.items()} }.
The source control has a separate replay and protocol. Artifact inventory records
final output hashes. Every example was chosen by fixed index/photometry/support,
not by redshift prediction error. The restored checkpoint symlink is only a local
compatibility link; persistent checkpoint contents are unchanged.
'''
    (out/'report.md').write_text(report)
    d.write(out/'final_provenance.json',dict(code_sha256={str(p.relative_to(ROOT)):d.sha(p) for p in [ROOT/'scripts/diagnose_hsc_clauds_stage0.py',ROOT/'scripts/stage0_source_ebv_control.py',Path(__file__)]},protected_artifacts_unchanged=True,source_reference_checks=replay,catalogue_fingerprints=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in [d.HDF,d.CAT]],report_sha256=d.sha(out/'report.md')))
    d.write(out/'artifact_inventory.json',{str(p.relative_to(out)):dict(sha256=d.sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='artifact_inventory.json' and p.suffix!='.log'})
    print(out/'report.md')


if __name__=='__main__':main()
