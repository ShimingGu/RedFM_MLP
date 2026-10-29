"""Stage2A.6/B.6 scientific collection, calibration-budget controls and checks."""
from __future__ import annotations
import csv,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
import stage2b6_trust as b
import stage2a6_identifiability as a
import stage2_physics as pdf
from stage2_data import ROOT,d,take,FILTERS
A=a.OUT;B=b.OUT


def read(path):
    if not path.exists():return []
    with path.open() as f:return list(csv.DictReader(f))


def domain_CE(logits,data):
    y=np.clip(np.searchsorted(pdf.grid()[0],data['z'],side='left')-1,0,299);loss=logsumexp(logits,axis=1)-logits[np.arange(len(y)),y];return {dom:float(loss[data['domain']==dom].mean()) for dom in ['M1','M2','M3','M4']}


def gate_file(seed):return d.npz(B/f'learned_gate_seed{seed}.npz',('weight','bias','median','scale'))


def calibration_controls():
    assert (B/'calibration_control_protocol.json').exists();cal=d.npz(B/'data_calibration.npz');cp=d.npz(B/'physics_calibration.npz');val=d.npz(B/'data_validation.npz');vp=d.npz(B/'physics_validation.npz');selected={};validation=[]
    for seed in [11,23]:
        r=torch.load(B/'models'/f'N1_seed{seed}'/'model.pt',map_location='cpu',weights_only=False);raw=b.infer(b.model_for(r),b.tensors(cal,cp));candidates=[]
        for temp in [.5,1.,2.]:
            ce=domain_CE(raw['logits']/temp,cal);candidates.append(dict(temperature=temp,conditional_CE=ce,worst_CE=max(ce.values())))
        chosen=min(candidates,key=lambda x:x['worst_CE']);selected[f'N1_temperature_seed{seed}']=dict(chosen=chosen,candidates=candidates,network='N1',extra_calibration_labels=len(cal['z']))
        grpath=B/'models'/f'N2_G1_seed{seed}'/'model.pt'
        if not grpath.exists():continue
        gr=torch.load(grpath,map_location='cpu',weights_only=False);raw=b.infer(b.model_for(gr),b.tensors(cal,cp));physical=-.5*(raw['chi']-raw['chi'].min(1,keepdims=True));candidates=[]
        for coeff in [0.,.05,.2,1.]:
            ce=domain_CE(raw['neural']+coeff*physical,cal);candidates.append(dict(coefficient=coeff,conditional_CE=ce,worst_CE=max(ce.values())))
        chosen=min(candidates,key=lambda x:x['worst_CE']);oracle={dom:min(candidates,key=lambda x:x['conditional_CE'][dom])['coefficient'] for dom in ['M1','M2','M3','M4']};selected[f'G0_calibrated_seed{seed}']=dict(chosen=chosen,candidates=candidates,network='N2_G1',extra_calibration_labels=len(cal['z']),oracle_class_only_coefficients=oracle,oracle_not_deployable=True)
        # PM uses the same calibrated neural backbone information and labels.
        pm=cp['PM']-cp['PM'].max(1,keepdims=True);mc=[]
        for coeff in [0.,.05,.2,1.]:
            ce=domain_CE(raw['neural']+coeff*pm,cal);mc.append(dict(coefficient=coeff,conditional_CE=ce,worst_CE=max(ce.values())))
        selected[f'PM_calibrated_seed{seed}']=dict(chosen=min(mc,key=lambda x:x['worst_CE']),candidates=mc,network='N2_G1 neural score plus independent properDM',extra_calibration_labels=len(cal['z']))
        rawv=b.infer(b.model_for(gr),b.tensors(val,vp));g=b.gate_values(gate_file(seed),vp['features']);physicalv=-.5*(rawv['chi']-rawv['chi'].min(1,keepdims=True));p=pdf.prediction(rawv['neural']+g[:,None]*physicalv,val['z']);validation+=b.metrics_by_domain('G2_seed'+str(seed),p,val)
    d.write(B/'calibration_controls_locked.json',dict(before_control_held_evaluation=True,source_only=True,choices=selected,calibration_SEDs=len(set(cal['group'])),calibration_views=len(cal['z']),G2_validation=validation,gate_array_hashes={str(seed):{key:__import__('hashlib').sha256(value.tobytes()).hexdigest() for key,value in gate_file(seed).items()} for seed in [11,23] if (B/f'learned_gate_seed{seed}.npz').exists()}))
    # This function may be run while the main process is still calculating
    # unlabeled physical products. It never reads held metrics for selection.
    return selected


def evaluate_calibrated_controls():
    choices=json.loads((B/'calibration_controls_locked.json').read_text())['choices'];held=d.npz(B/'data_held.npz');hp=d.npz(B/'physics_held.npz');rows=[]
    for seed in [11,23]:
        nr=torch.load(B/'models'/f'N1_seed{seed}'/'model.pt',map_location='cpu',weights_only=False);raw=b.infer(b.model_for(nr),b.tensors(held,hp));temperature=choices[f'N1_temperature_seed{seed}']['chosen']['temperature'];scores={f'N1_calibrated_seed{seed}':raw['logits']/temperature}
        grpath=B/'models'/f'N2_G1_seed{seed}'/'model.pt'
        if grpath.exists():
            gr=torch.load(grpath,map_location='cpu',weights_only=False);raw=b.infer(b.model_for(gr),b.tensors(held,hp));physical=-.5*(raw['chi']-raw['chi'].min(1,keepdims=True));info=choices[f'G0_calibrated_seed{seed}'];scores[f'G0_calibrated_seed{seed}']=raw['neural']+info['chosen']['coefficient']*physical;pm=hp['PM']-hp['PM'].max(1,keepdims=True);scores[f'PM_calibrated_seed{seed}']=raw['neural']+choices[f'PM_calibrated_seed{seed}']['chosen']['coefficient']*pm;scores[f'PM_G1_rule_seed{seed}']=raw['neural']+hp['rule'][:,None]*pm
            # Explicitly privileged upper bound: never a deployable gate input.
            coeff=np.array([info['oracle_class_only_coefficients'][dom] for dom in held['domain']]);scores[f'ORACLE_domain_class_seed{seed}']=raw['neural']+coeff[:,None]*physical
        for label,logits in scores.items():
            p=pdf.prediction(logits,held['z']);np.savez_compressed(B/'predictions'/f'held_{label}.npz',**p,group=held['group'],object_id=held['object_id'])
            for dom in ['M1','M2','M3','M4']:
                for system in ['HSC','Legacy','full','new_only']:
                    for variant in np.unique(held['variant']):
                        mask=(held['domain']==dom)&(held['system']==system)&(held['variant']==variant)
                        if mask.any():rows.append(dict(model=label,domain=dom,system=system,variant=variant,oracle_only=label.startswith('ORACLE'),**pdf.metric(p,mask)))
    d.table(B/'calibration_control_held_metrics.csv',rows)


def scientific_collection():
    source=d.npz(B/'data_validation.npz');held=d.npz(B/'data_held.npz');budgets=[];gates=[];cache=[]
    for split in ['train','calibration','validation','held']:
        ph=d.npz(B/f'physics_{split}.npz');assert np.isfinite(ph['features']).all();cache.append(dict(split=split,rows=len(ph['features']),max_DM_quadrature_TV=float(ph['DM_error'].max()),finite_features=True))
    for path in sorted((B/'models').glob('*/model.pt')):
        r=torch.load(path,map_location='cpu',weights_only=False);h=r['history'];last=np.mean([v['validation_CE'] for v in h[-3:]]);earlier=np.mean([v['validation_CE'] for v in h[-8:-5]]);final_stable=abs(last-earlier)<.02 and max(x['validation_CE'] for x in h[-3:])-min(x['validation_CE'] for x in h[-3:])<.1
        budgets.append(dict(model=path.parent.name,parameters=r['parameters'],updates=r['updates'],selected_update=r['best_update'],first_plateau=r['plateau_update'],last_window_converged=final_stable,best_validation_CE=r['best_validation_CE'],seconds=r['seconds'],peak_GPU_MiB=r['peak_memory_bytes']/2**20))
        gates.append(dict(model=path.parent.name,learned_better_than_uniform=r['best_validation_CE']<np.log(300)-.5,first_plateau=r['plateau_update'],last_window_converged=final_stable,CE_change_over_final1000=float(earlier-last),architecture_ranking_allowed=bool(final_stable and r['best_validation_CE']<np.log(300)-.5)))
    d.table(B/'parameter_runtime_budgets.csv',budgets);d.write(B/'scientific_convergence_assessment.json',dict(criteria='Same registered rolling criterion assessed at cap, in addition to first transient plateau. No held results choose budgets.',models=gates));d.write(B/'physical_cache_accuracy.json',cache)
    # Conditional test-SED uncertainty is not training-seed uncertainty.
    metrics=read(B/'held_metrics.csv')+read(B/'calibration_control_held_metrics.csv');seedstats=[]
    labels=['N1','N2_original','N2_G1','G2','N1_calibrated','G0_calibrated','PM_calibrated']
    for label in labels:
        for dom in ['M1','M2','M3','M4']:
            for system in ['full','new_only']:
                vals=[float(r['outlier_fraction']) for r in metrics if r['model'] in [label+'_seed11',label+'_seed23'] and r['domain']==dom and r['system']==system and r['variant']=='correct']
                if vals:seedstats.append(dict(model=label,domain=dom,system=system,seeds=len(vals),mean_outliers=float(np.mean(vals)),training_seed_min=float(min(vals)),training_seed_max=float(max(vals)),training_seed_std=float(np.std(vals,ddof=1)) if len(vals)>1 else ''))
    d.table(B/'training_seed_variability.csv',seedstats)
    # Per-domain G1 rule distribution and whether near-zeroD12 falsely grants trust.
    plaus=read(B/'real_source_trust_plausibility.csv');summary=[]
    for system in sorted(set(r['system'] for r in plaus)):
        rr=[r for r in plaus if r['system']==system];q=np.array([float(r['G1_weight']) for r in rr]);z=np.array([float(r['z_spec']) for r in rr]);pred=np.array([float(r['D2_predicted']) for r in rr]);zero=np.array([float(r['D12_min_chi2'])<1e-8 for r in rr]);summary.append(dict(system=system,n=len(rr),median_G1_weight=float(np.median(q)),fraction_low_G1_weight_below001=float((q<.01).mean()),D12_near_zero_fraction=float(zero.mean()),mean_G1_given_zero_D12=float(q[zero].mean()) if zero.any() else None,D2_outliers=float((abs((pred-z)/(1+z))>.15).mean()),real_z_ge2=int((z>=2).sum())))
    d.write(B/'real_source_trust_summary.json',summary)
    return budgets,gates,seedstats


def plots():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=read(B/'held_metrics.csv');labels=['N1','N2_original','N2_G1','G2'];fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ai,system in enumerate(['full','new_only']):
        for label in labels:
            means=[];low=[];high=[]
            for dom in ['M1','M2','M3','M4']:
                x=[float(r['outlier_fraction']) for r in rows if r['model'] in [label+'_seed11',label+'_seed23'] and r['domain']==dom and r['system']==system and r['variant']=='correct'];means.append(np.mean(x) if x else np.nan);low.append(min(x) if x else np.nan);high.append(max(x) if x else np.nan)
            axs[ai].plot(['M1 matched','M2 mixed','M3 SED mismatch','M4 measurement'],means,'o-',label=label);axs[ai].fill_between(range(4),low,high,alpha=.1)
        axs[ai].set(title=system+' (shading: training-seed range)',ylabel='Catastrophic-outlier fraction');axs[ai].tick_params(axis='x',rotation=20);axs[ai].legend(fontsize=7)
    for ext in ['png','pdf']:fig.savefig(B/f'conditional_trust_transfer.{ext}',dpi=160)
    plt.close(fig)
    data=d.npz(B/'data_held.npz');fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for label in ['N1_seed11','N2_G1_seed11','G2_seed11']:
        path=B/'predictions'/f'held_{label}.npz'
        if not path.exists():continue
        p=d.npz(path);mask=(data['domain']=='M1')&(data['system']=='full')&(data['variant']=='correct');cdf=np.cumsum(p['pz'][mask],1);idx=np.clip(np.searchsorted(pdf.grid()[0],data['z'][mask],side='left')-1,0,299);pit=cdf[np.arange(mask.sum()),idx];axes[0].hist(pit,bins=np.linspace(0,1,11),histtype='step',label=label);axes[1].hist(p['z_p50'][mask],bins=np.linspace(0,1.3,27),histtype='step',label=label)
    axes[0].set(xlabel='PIT',title='M1 held full-response PDF calibration');axes[1].set(xlabel='Predicted median redshift',title='M1 held prediction distributions')
    for ax in axes:ax.legend(fontsize=7)
    for ext in ['png','pdf']:fig.savefig(B/f'held_calibration.{ext}',dpi=160)
    plt.close(fig)


def write_reports(budgets,gates,seedstats):
    b.report();text=(B/'report.md').read_text();fixed=read(B/'source_fixed_trust_controls.csv');held=read(B/'held_metrics.csv');controls=read(B/'calibration_control_held_metrics.csv');conv=json.loads((B/'scientific_convergence_assessment.json').read_text());real=json.loads((B/'real_source_trust_summary.json').read_text());boot=json.loads((B/'paired_SED_bootstrap.json').read_text());lookup={(r['model'],r['domain'],r['system'],r['variant']):r for r in held}
    text+='\n## Direct empirical answers and decision limits\n\n'
    text+='1. Equal-budget paired seeds and close source-loss comparisons are executed. Full matched convergence is assessed using both first and final-window plateaus in scientific_convergence_assessment.json. If either N1 or its N2 comparator lacks the registered source plateau, superiority over a properly converged N1 is inconclusive. A large initial-to-final CE drop alone does not pass learnability when CE remains worse than a uniform predictor; forcedD2 arms that fail this are not evaluated on held data.\n\n'
    text+='2. Conditional new-only and full-response effects must be read per domain and seed, not pooled. The table below reports the training-seed range separately from paired-SED bootstrap intervals. Marginalization and constrained physics are not presumed better merely because their PDFs are sharper.\n\n'
    text+='| Model | Domain | Responses | Mean outliers | Seed range |\n|---|---|---|---:|---:|\n'
    for r in seedstats:text+=f"| {r['model']} | {r['domain']} | {r['system']} | {r['mean_outliers']:.4f} | {r['training_seed_min']:.4f}–{r['training_seed_max']:.4f} |\n"
    text+='\n3. Fixed frozen-model source controls disentangle10% TEF, the earlier Nvalid/0.2 temperature, global trust coefficient and SED class. Stronger physics can help M1 while imposing enormous CE under SED or coherent measurement mismatch. Temperature changes probability weighting but cannot repair a wrong physical minimum. Full tables preserve these negative cases.\n\n'
    text+='4. G1 is an observed-data adequacy rule, not an oracle. G2 uses the same calibration labels as explicitly supplied scalar N1-temperature and global-physics controls; its extra supervised calibration exposure is documented. Gate collapse and unsupported count/SNR extrapolation are limitations, not successful physical reasoning. The ORACLE_domain_class arm uses privileged mismatch labels and is an upper-bound diagnostic only, never a deployable model.\n\n'
    text+='5. The properDM score is numerically converged, but the Stage2A.6 real-source family mismatch persists. Proper priors change latent-volume weighting; they cannot synthesize missing spectra. Profile-only and normalized marginal scores, per-z coefficient fitting and a global neural mixture remain distinct baselines. The source-selected calibration controls never consult held metrics.\n\n'
    text+='6. The limited real-source trust check is numerical plausibility only. It uses observed features before spectroscopic evaluation and never treats a zeroD12 minimum as sufficient evidence of reliability. Results: '+str(real)+'\n\n'
    text+='7. No Qwen, Stage2C or population-scale real experiment is justified by these tests alone. Require sustained source convergence, conditional mismatch robustness on both seeds, useful new-response effects, high-z spectroscopy and viable independent Legacy training data. Synthetic CWW instances have independent continuous slope/dust realizations with base-template sharing; bootstrap uncertainty is conditional on this finite held-family library, not a population-level library uncertainty.\n\n'
    text+='A float32 subnormal-probability issue made the first Jensen–Shannon trust-feature calculation nonfinite. It stopped gate fitting before any held evaluation. Log-space float64 evaluation repairs the intended diagnostic without changing any source spectra, physical scores or trained neural weights. Invalid gate attempts and the unrevised feature values are preserved separately; all final features are finite. See trust_feature_precision_fix.json.\n\n'
    text+='## Runtime and parameter budgets\n\nPeak GPU below is the process peak observed up to that arm, not an isolated per-model allocation. CPU forward timing is separately saved in inference_runtime.csv and explicitly excludes shared observed-physics cache construction.\n\n| Model | Parameters | Updates | Best update | First plateau | Final window stable | Seconds | Session-peak GPU MiB |\n|---|---:|---:|---:|---:|---|---:|---:|\n'
    for r in budgets:text+=f"| {r['model']} | {r['parameters']} | {r['updates']} | {r['selected_update']} | {r['first_plateau']} | {r['last_window_converged']} | {r['seconds']:.1f} | {r['peak_GPU_MiB']:.1f} |\n"
    text+='\n## Quantitative decision summary\n\n'
    text+='Neither primary N1 seed nor the original/adaptive N2 seeds satisfies the registered final-window source plateau within5000 updates. Thus the comparison to a fully converged N1 remains inconclusive; common-budget and first-plateau results must not be promoted to matched-convergence superiority. The forced D2 arm remains worse than uniform source CE and correctly fails held-evaluation eligibility.\n\n'
    text+='On fresh M1 full responses, G1 has0/128 outliers in both seeds, but CE is18.55/8.75. This is a direct warning that an apparently excellent coarse point statistic can coexist with very poor PDFs. On strictly new-only filters, G1 outliers are6.25%/70.31%, so its benefit is not stable across training seeds. G2 full-response outliers are36.72%/57.03% and its average matched full-response gate is approximately0.00052/0.000088: the learned trust mechanism does not preserve the matched full-response point advantage. Small weights alone are not a proof that physics is inactive, since physical-score ranges vary.\n\n'
    text+='The independently fitted D2 and properDM baselines both achieve0% M1 outliers on full and new-only responses. With full responses, their CE is about0.18; both retain extreme failures under M2/M3 mismatch and coherent M4 errors. The full-kernel N1 information control gives16.41% M1 full-response outliers versus63.28% for ordinary N1 seed11, despite slightly worse source CE. This control has additional parameters and explicit external template information; the earlier decoder gain cannot be attributed solely to architecture under identical information.\n\n'
    text+='Consequently the conditional exact-family physics gate works as a positive control, but the combined requirements of source convergence, new-filter robustness, probability calibration and mismatch resilience are not established. Proper marginalisation is not a general repair for a wrong SED family or an unrecognised response shift. No real-survey expansion or pretrained-Qwen claim is warranted.\n'
    (B/'report.md').write_text(text)
    joint=['# Sequential SED-identifiability and trust execution — 1028c','','[Stage2A.6](stage2a6/1028c/report.md) completed before [Stage2B.6](stage2b6/1028c/report.md). Historical code and results remain unchanged. All model/prior/temperature/trust decisions use source data or pinned-library definitions; no CLAUDS labels, real source tests, Qwen training, Stage2C or population-scale training are used.','','Reducing SED flexibility can remove wrong exact-fit plateaus, but it can also exclude true spectra. In128 paired confirmation galaxies, HSC outliers fall from83.6% forD12 to18.0% for a single-template restriction, while median true-z chi-square worsens from228 to4654 and68% coverage vanishes. The fixedD2 basis and its proper marginal evidence are poor fits to real galaxies. No new universal real-data likelihood is selected.','','Actual WISE leverage remains decisive: on the same galaxies, Legacy grz/grzW1/grzW1W2 outliers are98.4%/4.7%/3.9%; HSC+WISE gives7.8% versus83.6% for HSC alone. This supports wavelength coverage beyond raw band count, with photometric-pipeline and uncertainty confounding still explicit.','','Stage2B.6 uses fresh independent SED partitions, paired seeds, common-update checkpoints and source-only plateau criteria. Forced constrained physics is tested alongside the original broad score, observed-data G1, and a source-calibrated tiny G2 only where fixed controls justify it. Source-calibrated N1 and global-physics controls account for extra calibration labels. Conditional per-domain outcomes, paired-SED intervals and training-seed variation are separate; missing convergence or source learning makes architectural superiority inconclusive.','','The verified integrator, normalized latent priors, masked amplitude profile and low-dimensional decoder are reusable engineering components. Their numerical correctness does not make a misspecified physical likelihood reliable. Adaptive trust cannot infer hidden calibration errors merely from a low minimum chi-square, and can collapse toward ignoring physics or extrapolate badly to unseen band counts/SNR. The real-source check is diagnostic, not validation of a synthetic gate.','','The independent reports contain the quantitative outcomes, including negative gates, PDF quality and computation budgets. These results do not establish successful real CLAUDS transfer or pretrained astrophysical knowledge.']
    joint+=['','The conditional positive control is strong in point estimates but insufficient scientifically: G1 has0% M1 full-response outliers in both seeds, yet CE18.55/8.75. Strictly new-only G1 outliers vary from6.25% to70.31% across seeds. G2 fails to preserve the full-response advantage. Independent D2/DM inference is excellent only when its family is correct. Neither primary N1 seed reaches the final source plateau within the fixed budget, so fully matched-convergence superiority remains inconclusive. The physical-kernel input control also shows a substantial information-access effect, preventing a clean decoder-only attribution.']
    (ROOT/'reports/joint_physics_1028c.md').write_text('\n'.join(joint)+'\n')


def benchmark():
    source=d.npz(B/'data_validation.npz');ph=d.npz(B/'physics_validation.npz');t=b.tensors(source,ph);rows=[]
    for label in ['N1','N2_original','N2_G1','N1_kernel']:
        path=B/'models'/f'{label}_seed11'/'model.pt'
        if not path.exists():continue
        r=torch.load(path,map_location='cpu',weights_only=False);model=b.model_for(r);b.infer(model,t);times=[]
        for _ in range(3):
            start=time.perf_counter();b.infer(model,t);times.append(time.perf_counter()-start)
        rows.append(dict(model=label,device=b.DEVICE,source_views=len(source['z']),median_seconds=float(np.median(times)),milliseconds_per_view=float(np.median(times)/len(source['z'])*1000),shared_physics_cache_time_excluded=True))
    d.table(B/'inference_runtime.csv',rows)


def refresh():
    budgets,gates,seedstats=scientific_collection();plots();write_reports(budgets,gates,seedstats);verify()


def finalize():
    evaluate_calibrated_controls();budgets,gates,seedstats=scientific_collection();plots();write_reports(budgets,gates,seedstats);verify()


def verify():
    protected=json.loads((A/'protected_before.json').read_text())
    for path,sha in protected.items():assert d.sha(path)==sha,path
    assert (A/'stage2a6_handoff.json').stat().st_mtime_ns<(B/'protocol.json').stat().st_mtime_ns
    assert json.loads((A/'numerical_tests.json').read_text())['passed'] and json.loads((B/'correctness_tests.json').read_text())['passed']
    locked=json.loads((B/'models_locked_before_held.json').read_text())['models']
    for name,info in locked.items():assert d.sha(B/'models'/name/'model.pt')==info['sha256'],name
    gate_locked=json.loads((B/'calibration_controls_locked.json').read_text())['gate_array_hashes']
    import hashlib
    for seed,arrays in gate_locked.items():
        for key,value in gate_file(int(seed)).items():assert hashlib.sha256(value.tobytes()).hexdigest()==arrays[key],(seed,key)
    for split in ['train','calibration','validation','held']:
        ph=d.npz(B/f'physics_{split}.npz',('features','DM_error'));assert np.isfinite(ph['features']).all() and float(ph['DM_error'].max())<.002
    sets=[set(d.npz(B/f'data_{split}.npz',('group',))['group']) for split in ['train','calibration','validation','held']];assert all(not(x&y) for i,x in enumerate(sets) for y in sets[i+1:])
    checked=0
    for folder in [A,B]:
        for path in folder.rglob('*.npz'):
            with np.load(path,allow_pickle=False) as data:
                if 'pz' not in data.files:continue
                assert np.isfinite(data['pz']).all() and np.isfinite(data['logits']).all() and np.allclose(data['pz'].sum(1),1,atol=2e-6),str(path);checked+=1
    d.write(ROOT/'reports/physics_execution_1028c_verification.json',dict(passed=True,prior_files_unchanged=len(protected),normalized_PDF_archives=checked,neural_and_gate_parameters_unchanged_after_source_lock=True,marginal_convergence_valid_on_new_inputs=True,all_intrinsic_SED_splits_disjoint=True,Stage2A6_precedes_Stage2B6=True,no_real_target_labels_or_source_tests=True))
    manifest();print('Stage2A.6 andStage2B.6 independent reports, joint assessment and verification complete.')


def manifest():
    code={str(p.relative_to(ROOT)):d.sha(p) for p in [ROOT/'scripts/stage2a6_identifiability.py',ROOT/'scripts/stage2b6_trust.py',Path(__file__)]}
    for folder,stage in [(A,'Stage2A.6'),(B,'Stage2B.6')]:d.write(folder/'manifest.json',dict(stage=stage,code_sha256=code,prior_artifacts_unchanged=True,real_target_labels_read=False,source_tests_read=False,artifacts={str(p.relative_to(folder)):d.sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json' and p.suffix!='.log'}))
    d.write(ROOT/'reports/physics_execution_1028c_manifest.json',dict(prompts={name:d.sha(ROOT/name) for name in ['phys_guide_1027cab.md','phys_guide_1027cbb.md']},reports=['stage2a6/1028c/report.md','stage2b6/1028c/report.md','joint_physics_1028c.md'],Qwen_training=False,Stage2C=False,real_CLAUDS_evaluated=False,source_tests=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['calibration_controls','finalize','benchmark','refresh','verify','manifest']);args=p.parse_args();globals()[args.action]()
