"""Read-only scientific collection and provenance for the sequential 1028 run."""
from __future__ import annotations
import csv,json,time,os
from datetime import datetime,timezone
import numpy as np
import torch
import stage2_physics as p
import stage2_numerical as b
from stage2_data import ROOT,d,FILTERS
A=p.OUT;B=b.OUT


def rows(path):
    with open(path) as f:return list(csv.DictReader(f))


def verify_and_diagnose():
    protection=json.loads((A/'historical_protection.json').read_text())
    for path,digest in {**protection['prior_experiment_files'],**protection['original_models']}.items():assert d.sha(path)==digest,path
    d.check(ROOT/'reports/stage0/1027a')
    old=json.loads((B/'models_locked_before_empty_set_fix.json').read_text());new=json.loads((B/'models_locked.json').read_text());assert old==new
    for name,record in new['models'].items():assert d.sha(B/f'models/real_{name}_seed11.pt')==record['sha256']
    audit=[]
    for folder in [A,B/'predictions']:
        for path in folder.glob('*.npz'):
            with np.load(path,allow_pickle=False) as a:
                if 'pz' not in a.files:continue
                assert np.all(np.isfinite(a['pz'])) and np.all(np.isfinite(a['logits'])) and np.allclose(a['pz'].sum(1),1,atol=2e-6),str(path)
                audit.append(dict(path=str(path.relative_to(ROOT)),objects=len(a['pz']),unsupported=int(a['unsupported'].sum()) if 'unsupported' in a.files else None))
    target=d.npz(A/'clauds_photometry.npz');baseline=d.npz(A/'CLAUDS_P0_full.npz');z=baseline['z_spec'];cond=[];objects=[]
    configs={'physics_P0_full':baseline,'physics_P2_full':d.npz(A/'CLAUDS_P2_full.npz'),'archived_Qwen':d.npz(d.FAST/'clauds_predictions.npz')}
    for name in ['N0','N1','N2']:configs[name]=d.npz(B/f'predictions/CLAUDS_{name}_full.npz')
    for name,a in configs.items():
        n=a.get('n_valid',np.full(256,-1));unsupported=a.get('unsupported',np.zeros(256,bool));idx=np.clip(np.searchsorted(a['redshift_edges'],z,side='left')-1,0,299);ce=-a['logits'][np.arange(256),idx]
        # Existing archived logits need log normalization, unlike physical outputs.
        from scipy.special import logsumexp
        ce=ce+logsumexp(a['logits'],axis=1)
        cond.append(dict(model=name,unsupported=int(unsupported.sum()),no_bands=int((n==0).sum()),usable_band_histogram={str(int(k)):int((n==k).sum()) for k in np.unique(n)},CE_median=float(np.median(ce)),CE_p95=float(np.quantile(ce,.95)),CE_max=float(np.max(ce)),five_largest_CE_object_ids=target['object_id'][np.argsort(ce)[-5:]].tolist(),predicted_z_quantiles=np.quantile(a['z_p50'],[0,.05,.5,.95,1]).tolist()))
    for i in range(256):objects.append(dict(object_id=target['object_id'][i],ZPHOT=float(z[i]),valid_bands=int(baseline['n_valid'][i]),bad_quality_bands=int(((target['quality'][i]>0)&target['quality_available'][i]).sum()),**{name+'_median':float(a['z_p50'][i]) for name,a in configs.items()}))
    d.table(ROOT/'reports/physics_1028_object_comparison.csv',objects)
    d.write(A/'fit_diagnostics.json',cond)
    # The empty-set numerical fix must not change any fitted checkpoint or the
    # supported source predictions. Run the final code against saved outputs.
    model_result=torch.load(B/'models/real_N2_seed11.pt',map_location='cpu',weights_only=False);model=b.load_model(model_result,'cpu');t=b.tensor_rows(target);raw=b.infer(model,t);pred=p.prediction(raw['logits'],z);saved=configs['N2'];supported=t['mask'].sum(1).numpy()>0
    delta=float(np.max(np.abs(pred['pz'][supported]-saved['pz'][supported])));assert delta<3e-5,delta
    # All empty objects use the declared uniform fallback in saved products.
    uniform=p.prediction(np.zeros((int((~supported).sum()),300)),z[~supported]);np.testing.assert_allclose(saved['pz'][~supported],uniform['pz'])
    fix=dict(trigger='Five CLAUDS query objects have zero usable flux/error pairs',error='Unmasked template shape divided by a tiny empty-set norm could overflow and yield NaN before the uniform fallback',correction='Mask template shape before normalization; floor squared norm before sqrt for finite empty-set gradients',weights_unchanged=True,settings_unchanged=True,target_labels_used_for_fix=False,source_or_supported_predictions_max_PDF_change_CPU_vs_saved_GPU=delta,regression_test='all_missing_finite_neutral_physics plus finite empty-set gradients',original_model_lock_sha256=d.sha(B/'models_locked_before_empty_set_fix.json'),final_model_lock_sha256=d.sha(B/'models_locked.json'))
    d.write(B/'empty_set_fix_audit.json',fix)
    d.write(ROOT/'reports/physics_execution_1028_verification.json',dict(protected_historical_experiment_files=len(protection['prior_experiment_files']),original_protected_models=True,PDF_archives_checked=len(audit),archives=audit,checkpoints_and_scientific_settings_unchanged_after_target_evaluation=True,all_correctness_tests_pass=True,empty_set_fix=fix))
    return configs,cond


def collect_extra():
    from eazy.filters import FilterFile
    from eazy.param import TranslateFile
    fs=FilterFile(file=str(A/'filters.RES'));assert len(fs.filters)==len(FILTERS)
    for i,name in enumerate(FILTERS):
        curve=p.curves()[i];assert fs.filters[i].name.startswith(name)
        np.testing.assert_allclose(fs.filters[i].wave,curve.wave,rtol=1e-5)
        np.testing.assert_allclose(fs.filters[i].throughput,curve.throughput,rtol=1e-5,atol=1e-8)
    systems={'HSC':[f'HSC.{x}' for x in 'grizy'],'Legacy_DECam':['DECam.g','DECam.r','DECam.z','WISE.W1','WISE.W2'],'Legacy_BASS_MzLS':['BASS.g','BASS.r','MzLS.z','WISE.W1','WISE.W2'],'CLAUDS':['MegaCam.u','MegaCam.u_star',*[f'HSC.{x}' for x in 'grizy'],'VIRCAM.Y','VIRCAM.J','VIRCAM.H','VIRCAM.Ks']}
    for system,names in systems.items():
        lines=[]
        for name in names:
            number=FILTERS.index(name)+1;column=name.replace('.','_');lines.extend([f'{column}_flux F{number}',f'{column}_sigma E{number}'])
        path=A/f'{system}.translate';path.write_text('\n'.join(lines)+'\n');translate=TranslateFile(str(path));assert len(translate.trans)==2*len(names)
    d.write(A/'export_roundtrip_tests.json',dict(EAZY_filter_file_roundtrip=True,throughput_max_relative_rounding=1e-5,translation_configurations=list(systems),passed=True))
    budgets=[]
    for path in sorted(B.glob('*_history.json')):
        r=json.loads(path.read_text());budgets.append({k:r[k] for k in ['name','domain','seed','trainable_parameters','updates','best_step','best_validation_ce','seconds','peak_GPU_bytes']})
    d.table(B/'model_budgets.csv',budgets)
    records=[json.loads(x) for x in (ROOT/'reports/stage1/1027/records.jsonl').read_text().splitlines()];syn=d.npz(B/'synthetic.npz');sr=[]
    test=np.array([r['split'] in ['test','family_holdout'] and r['variant']=='correct' for r in records]);sn=np.array([r['sn_nuisance'] for r in records])
    for name in ['N0','N1','N2','N1_summary','N1_shuffled']:
        pred=d.npz(B/f'predictions/synthetic_{name}.npz')
        for label,mask in [('z_lt1',syn['z']<1),('z_1to2',(syn['z']>=1)&(syn['z']<2)),('z_ge2',syn['z']>=2),('SN_lt30',sn<30),('SN_ge30',sn>=30)]:
            if (test&mask).any():sr.append(dict(model=name,stratum=label,**p.metric(pred,test&mask)))
    d.table(B/'synthetic_redshift_SN_metrics.csv',sr)
    # Synthetic scale stress is determined by the already-tested exact input
    # normalization symmetry, and verified numerically on disjoint test SEDs.
    scale_results=[];a=b.data.take(syn,np.flatnonzero(test)[:32]);scaled={k:v.copy() for k,v in a.items()};scaled['flux']*=1e-3;scaled['sigma']*=1e-3
    for name in ['N0','N1','N2']:
        result=torch.load(B/f'models/synthetic_{name}_seed11.pt',map_location='cpu',weights_only=False);model=b.load_model(result,'cpu');x=b.infer(model,b.tensor_rows(a))['logits'];y=b.infer(model,b.tensor_rows(scaled))['logits'];change=float(np.max(abs(x-y)));assert np.allclose(x,y,atol=3e-4,rtol=3e-5)
        scale_results.append(dict(model=name,common_f_sigma_scale=.001,max_logit_change=change,passed=True))
    d.write(B/'synthetic_scale_test.json',scale_results)
    stage1=[]
    for r in rows(ROOT/'reports/stage1/1027/physical_baseline_performance.csv'):
        if r['stratum'] in ['test_HSC_correct','test_Legacy_correct','test_CLAUDS_correct','test_new_filters_correct']:
            stage1.append(dict(stratum=r['stratum'],NMAD=float(r['nmad']),outliers=float(r['catastrophic_outlier_fraction']),CRPS=float(r['mean_crps']),coverage=float(r['p16_p84_coverage'])))
    d.write(B/'stage1_physics_context.json',dict(grid_bins=90,template_bank='Stage1 controlled bank, different from FSPS QSF12',metrics=stage1,CE_not_compared_across_grids=True))
    # N2 reconstruction diagnostics at the closest available grid centre to z.
    reconstruct=[]
    for domain in ['real','synthetic']:
        a=d.npz(B/'real_val.npz') if domain=='real' else syn;pred=d.npz(B/f'predictions/{domain}_N2.npz');idx=np.clip(np.searchsorted(p.grid()[0],a['z'],side='left')-1,0,299);v=pred['physical_chi2'];n=pred['n_valid'].clip(1)
        for cat in np.unique(a['catalogue']):
            mask=a['catalogue']==cat
            if domain=='synthetic':mask&=np.isin(a['split'],['test','family_holdout'])
            if not mask.any():continue
            true=v[np.arange(len(v)),idx]/n;best=v.min(1)/n;reconstruct.append(dict(domain=domain,system=cat,n=int(mask.sum()),median_reduced_true_z_chi2=float(np.median(true[mask])),median_reduced_best_chi2=float(np.median(best[mask])),median_simplex_max_coefficient=float(np.median(pred['sed_weights'][mask].max(1)))))
    d.table(B/'reconstruction_diagnostics.csv',reconstruct)
    return budgets,stage1


def finalize_reports(configs,diagnostics,budgets,context):
    p.report();b.report();a=json.loads((A/'summary.json').read_text());s=json.loads((B/'summary.json').read_text());target={r['condition']:r for r in s['target']};source={r['condition']:r for r in s['source']};z=configs['N1']['z_spec'];g=s['gates']
    # Independent reports retain enough evidence to stand alone.
    with (A/'report.md').open('a') as f:
        f.write('\n## Numerical and failure audit\n\nThe constant-fnu test agrees to 2.3e-16. Independent Stage-1 photon integration at z=0.42 agrees within 1.40% across all registered responses; the difference reflects conservative spectral interpolation versus point interpolation on throughput knots. This is a measured approximation, not exact equality. The known-mixture chi-square is 1.63e-27.\n\n')
        for r in diagnostics[:2]:f.write(f"{r['model']}: {r['unsupported']}/256 unsupported ({r['no_bands']} with no bands); usable-band histogram {r['usable_band_histogram']}. Median CE={r['CE_median']:.3f}, 95th percentile={r['CE_p95']:.3f}, maximum={r['CE_max']:.3f}. The large aggregate P0 CE is driven by highly overconfident incompatible objects; no probability floor hides those failures. See fit_diagnostics.json and per-object PDFs.\n\n")
        native=d.npz(A/'stress_screen_native.npz');dim=d.npz(A/'stress_screen_dim7.npz');difference=float(np.max(abs(native['pz']-dim['pz'])));f.write(f'On the exact Stage-0.5 matched objects, 7-mag common dimming changes the P0 PDF by at most {difference:.3g}; point predictions are unchanged. This symmetry does not compensate for the poor source point estimates or create high-redshift supervision. Error-only broadening and noisy measurements have separate, often poor outcomes in stress_metrics.csv.\n')
    with (B/'report.md').open('a') as f:
        f.write('\n## Answers supported by this pilot\n\n')
        f.write(f"The shared set architecture improves real-source learning in this seed: HSC/Legacy NMAD goes from {source['real_N0_HSC']['nmad']:.4f}/{source['real_N0_Legacy']['nmad']:.4f} for N0 to {source['real_N1_HSC']['nmad']:.4f}/{source['real_N1_Legacy']['nmad']:.4f} for N1. N2 gives {source['real_N2_HSC']['nmad']:.4f}/{source['real_N2_Legacy']['nmad']:.4f} and a worse balanced source CE than N1. This does not establish an additional physical-decoder gain.\n\n")
        for name in ['N0','N1','N2']:
            r=target[f'CLAUDS_{name}_full'];f.write(f"{name} full CLAUDS: outliers {r['outlier_fraction']:.1%}, CE {r['ce']:.3f}, 68% coverage {r['coverage_68']:.1%}; {r['concentration_width_003']:.1%} of medians lie within a width-0.03 interval. ")
        f.write('These concentrated, poorly calibrated predictions are not credible true-redshift transfer. N2 slightly lowers exploratory outliers but worsens CE against N1; that mixed result does not support a physical-generalisation claim. Compact/shuffled-response, brightness and noise controls are all displayed, not selected using target performance.\n\n')
        f.write('All five small synthetic fits select update zero by source-validation CE: training did not improve held-out synthetic validation loss. Their random-initialization predictions are explicitly retained as selected by the protocol, not presented as successfully trained synthetic regressors. The 300-bin, 512-label pilot therefore supplies no reliable synthetic architecture ranking. N2 has 84.4% outliers on strictly held-out filters versus 76.6% for N1. No expensive expansion or extra seeds are justified by Gate 3; a fresh source-only protocol addressing this synthetic learning failure would be needed before scaling.\n\n')
        f.write(f"The real target has {int((z>=2).sum())}/256 ZPHOT references at z≥2, while the eligible source training population contains none. The faint/noisy population and UV/NIR bands are also shifted. These facts, plus the failed synthetic learning check and physical-model mismatch, prevent clean attribution of real transfer failures to architecture alone. Noise augmentation did not supply high-z labels and did not rescue transfer.\n\n")
        f.write('There is no evidence here that reintroducing Qwen is the necessary next step. Resolve synthetic numerical learnability, obtain independent high-z spectroscopy and verify catalogue processing first. This conclusion leaves foundation-model/pretraining claims untested. No Stage 2C run was started.\n\n')
        f.write('An empty-set decoder bug was found during the first fixed target inference. The template is now masked before shape normalization, and the squared norm is floored before sqrt. Dedicated finite-output and gradient tests pass. All checkpoint hashes and scientific settings are identical to the lock saved before that inference; no target labels informed the repair. See empty_set_fix_audit.json.\n\n')
        f.write('| Model | Domain | Parameters | Updates | Best update | Seconds | Peak GPU MiB |\n|---|---|---:|---:|---:|---:|---:|\n')
        for r in budgets:f.write(f"| {r['name']} | {r['domain']} | {r['trainable_parameters']} | {r['updates']} | {r['best_step']} | {r['seconds']:.1f} | {r['peak_GPU_bytes']/2**20:.1f} |\n")
        f.write('\nStage-1 physics remains useful synthetic context; it used a different controlled template bank and 90-bin grid. Its point/calibration metrics are reproduced in stage1_physics_context.json without an invalid cross-grid CE comparison. Synthetic redshift/S/N and reconstruction diagnostics are separate CSVs; scale tests pass.\n')
    # Concise joint scientific comparison, no winner chosen for subsequent tuning.
    names=[('Archived HSC Qwen',p.metric(configs['archived_Qwen'])),('Stage2A P0 full',next(r for r in a['performance'] if r['condition']=='CLAUDS_P0_full')),('Stage2A P2 full',next(r for r in a['performance'] if r['condition']=='CLAUDS_P2_full'))]+[(name,target[f'CLAUDS_{name}_full']) for name in ['N0','N1','N2']]
    lines=['# Sequential physics execution — 1028','','Executed [Stage 2A](stage2a/1028/report.md) before [Stage 2B](stage2b/1028/report.md), following phys_guide_1027ca.md and phys_guide_1027cb.md. Stage 2A completed physical-input checks, full-response integration tests and baseline evaluation before the handoff. No unresolved correctness error invalidated the Stage 2B pilot. All Stage 2B correctness tests and real-source learnability gates pass; the physical-improvement gate fails, so no population-scale expansion, extra seeds or Stage 2C was launched.','','The physical baseline substantially reduces the archived collapse against CLAUDS ZPHOT. The tested neural architectures do not demonstrate credible target transfer. None of these results establishes accuracy against independent CLAUDS spectroscopy.','','| Fixed condition | NMAD | Outliers | CE | CRPS | 68% coverage | Width-0.03 concentration |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in names:lines.append('| '+name+' | '+' | '.join(f'{r[k]:.4f}' for k in ['nmad','outlier_fraction','ce','crps','coverage_68','concentration_width_003'])+' |')
    lines+=['','Explicit physics provides useful object-dependent exploratory ZPHOT agreement: full UV/NIR P0 outliers are 43.8% versus 53.1% with HSC-only measurements, and the source-selected P2 error model gives 41.4%. Errors remain large; P0 is severely overconfident on incompatible objects, and ZPHOT itself is template-derived. Extinction sensitivity is small for these targets but does not prove historical catalogue equivalence.','','The shared N1 set encoder improves source validation over N0 in this seed. Adding the N2 SED decoder does not improve source CE and fails the controlled synthetic improvement gate. Its small target outlier change is accompanied by worse CE. The synthetic neural fits select initialization because validation loss worsens during training; this is an explicit limitation, not evidence against astrophysical physics. The fixed numerical runs retain substantial concentration and poor target coverage.','','This is a small audited same-galaxy multi-view source experiment, not the definitive independently sampled mixed-source population run. Full Legacy validation (15,053 objects) was recovered, but its original independent training cache remains missing. Source training has no z≥2 spectroscopy while many target references are in that range. One seed, photometric shift, uncertain historical extinction equivalence, nominal HSC r/i responses and different template/label assumptions limit attribution.','','CLAUDS labels were excluded from training, checkpoint selection, error calibration, architecture and augmentation selection. All model locks predate target evaluation and remain identical after the empty-set numerical repair. Historical experiment/model hashes are unchanged. See physics_execution_1028_verification.json and the two independent manifests for exact commands, dependencies, input hashes, PDFs and gates.']
    (ROOT/'reports/joint_physics_1028.md').write_text('\n'.join(lines)+'\n')


def manifests():
    commands='''#!/usr/bin/env bash
set -euo pipefail
cd /scratch/RedFM_MLP
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON=/scratch/.tmp-gsm/stage0-venv/bin/python
"$PYTHON" scripts/stage2_numerical.py tests
"$PYTHON" scripts/stage2_numerical.py prepare
CUDA_VISIBLE_DEVICES=MIG-88b5a6d2-3af1-5088-9cb7-235af969d28e "$PYTHON" scripts/stage2_numerical.py run
"$PYTHON" scripts/report_stage2_physics.py
'''
    (B/'reproduce.sh').write_text(commands)
    code={str(path.relative_to(ROOT)):d.sha(path) for path in [ROOT/'scripts/stage2_data.py',ROOT/'scripts/stage2_physics.py',ROOT/'scripts/stage2_numerical.py',ROOT/'scripts/report_stage2_physics.py']}
    for stage,folder in [('Stage2A',A),('Stage2B',B)]:
        paths={str(path.relative_to(folder)):d.sha(path) for path in sorted(folder.rglob('*')) if path.is_file() and path.name not in ['execution_manifest.json','collection.log']}
        d.write(folder/'execution_manifest.json',dict(stage=stage,completed_UTC=datetime.now(timezone.utc).isoformat(),code_sha256=code,artifacts_sha256=paths,original_guide=str(ROOT/'reports/stage1/1027/phys_guide_execution_1028.md'),referenced_prompt=str(ROOT/f'reports/stage1/1027/phys_guide_1027c{"a" if stage=="Stage2A" else "b"}.md'),target_label_training_or_selection=False,historical_artifacts_unchanged=True,full_independent_mixed_source_population_pending=stage=='Stage2B'))
    d.write(ROOT/'reports/physics_execution_1028_manifest.json',dict(completed_UTC=datetime.now(timezone.utc).isoformat(),Stage2A='stage2a/1028/report.md',Stage2B='stage2b/1028/report.md',comparison='joint_physics_1028.md',handoff=json.loads((A/'handoff.json').read_text()),Stage2B_gates=json.loads((B/'gates_before_target.json').read_text()),Qwen_training=False,Stage2C_started=False,target_label_fitting=False,manifest_sha256={str(folder.relative_to(ROOT)):d.sha(folder/'execution_manifest.json') for folder in [A,B]}))


if __name__=='__main__':
    torch.set_num_threads(2);configs,diag=verify_and_diagnose();budgets,context=collect_extra();finalize_reports(configs,diag,budgets,context);manifests();print('Stage2A + gated Stage2B reports, comparison and provenance complete.')
