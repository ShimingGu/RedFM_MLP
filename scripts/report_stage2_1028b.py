"""Controlled held-response diagnostics and separate final 2A.5/2B.5 reports."""
from __future__ import annotations
import json,csv,time
from pathlib import Path
import numpy as np
import torch
import stage2b5_learnability as b
from stage2_data import ROOT,d,FILTERS,take
import stage2_physics as pdf
import stage2a5_audit as a5
import stage2_numerical as numeric
A=a5.OUT;B=b.OUT
FULL=['MegaCam.u','MegaCam.u_star',*[f'HSC.{x}' for x in 'grizy'],'VIRCAM.Y','VIRCAM.J','VIRCAM.H','VIRCAM.Ks'];NEW=[x for x in FULL if not x.startswith('HSC.')]


def read(path):
    if not path.exists():return []
    with path.open() as f:return list(csv.DictReader(f))


def identifiable(split,n):
    path=B/f'identifiable_{split}.npz'
    if path.exists():return d.npz(path)
    rng=np.random.default_rng(431+int(split=='val'));k=d.npz(A/'response_kernel_highres.npz')['kernel'];flux=[];sig=[];bands=[];z=[];groups=[];weights=[];indices=[]
    for i in range(n):
        w=np.zeros(12);w[1]=rng.uniform(0,.05);w[0]=1-w[1];iz=int(rng.integers(7,74))
        for bb in [b.HSC,b.LS]:
            f=k[[FILTERS.index(x) for x in bb],iz]@w;e=np.ones(5)*np.median(f)*.01;flux.append(f);sig.append(e);bands.append(bb);z.append(pdf.grid()[1][iz]);groups.append(f'identifiable:{split}:{i}');weights.append(w);indices.append(iz)
    rows=b.rows_from_flux(flux,sig,bands,z,groups,split);np.savez_compressed(path,**rows);np.savez_compressed(B/f'identifiable_{split}_truth.npz',weights=np.array(weights),redshift_index=np.array(indices));return rows


def controlled_views(domain):
    # Independent source-validation latent identities are reobserved under new
    # response systems only after source learning and all checkpoints are frozen.
    rng=np.random.default_rng(2805);flux=[];sigma=[];bands=[];zs=[];groups=[];systems=[];variants=[]
    if domain=='analytic':
        params=json.loads((B/'L3_val_truth.json').read_text())[::2]
    else:
        inputa=identifiable('val',256) if domain=='identifiable' else b.matched('val',256);truth=d.npz(B/('identifiable_val_truth.npz' if domain=='identifiable' else 'matched_val_256_truth.npz'));params=[dict(z=float(z),group=str(g),weights=w.tolist()) for z,g,w in zip(inputa['z'][::2],inputa['group'][::2],truth['weights'][::2])]
    for par in params:
        for system,bb in [('full',FULL),('new_only',NEW)]:
            for variant in ['correct','wrong_metadata','missing','noisy','redshift_plus01']:
                zz=float(par['z'])+(.1 if variant=='redshift_plus01' else 0.)
                if domain=='analytic':clean=b.break_flux(zz,bb,par['slope'],par['strength'],par['av'])
                else:clean=a5.continuous(zz)[[FILTERS.index(x) for x in bb]]@np.array(par['weights'])
                e=np.ones(len(bb))*np.median(abs(clean))*(1/30 if domain=='analytic' else .01)
                if variant=='noisy':e*=5
                ff=clean+rng.normal(size=len(bb))*e if domain=='analytic' or variant=='noisy' else clean.copy()
                if variant=='missing':ii=rng.choice(len(bb),2,replace=False);ff[ii]=np.nan;e[ii]=np.nan
                metadata=list(np.roll(bb,1)) if variant=='wrong_metadata' else bb
                flux.append(ff);sigma.append(e);bands.append(metadata);zs.append(zz);groups.append(par['group']);systems.append(system);variants.append(variant)
    rows=b.rows_from_flux(flux,sigma,bands,zs,groups,'held_response');rows['system']=np.array(systems);rows['variant']=np.array(variants);np.savez_compressed(B/f'{domain}_held_views.npz',**rows);return rows


def bootstrap(p1,p2,groups,seed=2805):
    rng=np.random.default_rng(seed);unique=np.unique(groups);slices=[np.flatnonzero(groups==x) for x in unique];delta=[]
    err1=np.abs((p1['z_p50']-p1['z_spec'])/(1+p1['z_spec']))>.15;err2=np.abs((p2['z_p50']-p2['z_spec'])/(1+p2['z_spec']))>.15
    for _ in range(300):
        idx=np.concatenate([slices[j] for j in rng.integers(len(unique),size=len(unique))]);delta.append(float(np.mean(err2[idx])-np.mean(err1[idx])))
    return dict(unique_SED_units=len(unique),N2_minus_N1_outlier_change=float(err2.mean()-err1.mean()),paired_SED_bootstrap_CI95=np.quantile(delta,[.025,.975]).tolist(),replicates=300)


def controls():
    from qwen_zero_shot_sanity import reserve_one_gpu
    assert json.loads((B/'gates.json').read_text())['elementary'];selected=json.loads((B/'source_optimization_selection.json').read_text())['selected'];options={k:selected[k] for k in ['lr','smooth','coarse']}
    with reserve_one_gpu():
        tr=identifiable('train',1024);va=identifiable('val',256);rs={};source=[]
        for name in ['N0','N1','N2']:
            r=b.train(tr,va,name,'identifiable_'+name,steps=2000,**options);rs[name]=r;source.append(b.score_row(r,va,'identifiable_'+name))
        d.table(B/'identifiable_source_metrics.csv',source);b.physical_controls(rs['N2'],va,'identifiable');d.write(B/'controlled_models_locked.json',dict(models={name:d.sha(B/'runs'/r['tag']/'model.pt') for name,r in rs.items()},real_target_labels_used=False,source_validation_only=True))
        models={'identifiable':rs}
        for domain,prefix in [('analytic','L3'),('matched','matched')]:
            models[domain]={}
            for name in ['N0','N1','N2']:
                tag='L3_N2_mismatched' if domain=='analytic' and name=='N2' else prefix+'_'+name
                r=torch.load(B/'runs'/tag/'model.pt',map_location='cpu',weights_only=False);models[domain][name]=r
        results=[];uncertainty=[];paired=[]
        for domain,arms in models.items():
            # Gate on source-validation trajectory, never held-response labels.
            if any(r['history'][0]['source_validation_CE']-r['best_validation_CE']<.3 for name,r in arms.items() if name in ['N1','N2']):continue
            views=controlled_views(domain);predictions={}
            for name,r in arms.items():
                pred=b.evaluate(r,views,domain+'_'+name+'_held_responses');predictions[name]=pred
                for system in ['full','new_only']:
                    for variant in ['correct','wrong_metadata','missing','noisy','redshift_plus01']:
                        mask=(views['system']==system)&(views['variant']==variant);results.append(dict(domain=domain,model=name,system=system,variant=variant,**pdf.metric(pred,mask)))
            for system in ['full','new_only']:
                mask=(views['system']==system)&(views['variant']=='correct');s1={k:(v[mask] if k not in ['redshift_edges','redshift_centers'] and np.ndim(v)>0 else v) for k,v in predictions['N1'].items()};s2={k:(v[mask] if k not in ['redshift_edges','redshift_centers'] and np.ndim(v)>0 else v) for k,v in predictions['N2'].items()};uncertainty.append(dict(domain=domain,system=system,**bootstrap(s1,s2,views['group'][mask])))
            for name,pred in predictions.items():
                a=(views['system']=='full')&(views['variant']=='correct');c=(views['system']=='new_only')&(views['variant']=='correct');zplus=(views['system']=='full')&(views['variant']=='redshift_plus01');paired.append(dict(domain=domain,model=name,mean_abs_full_to_new_filter_shift=float(np.mean(abs(pred['z_p50'][a]-pred['z_p50'][c]))),mean_plus01_response=float(np.mean(pred['z_p50'][zplus]-pred['z_p50'][a])),fraction_positive_response=float(np.mean(pred['z_p50'][zplus]>pred['z_p50'][a]))))
            # Independent maintained NNLS on identical full/new-only correct views.
            clean=take(views,views['variant']=='correct');k=d.npz(A/'response_kernel_highres.npz')['kernel'];cs=np.array([a5.profile((f,e,ids,k,0.)) for f,e,ids in zip(clean['flux'],clean['sigma'],clean['band_id'])]);pred=pdf.prediction(-.5*cs,clean['z']);np.savez_compressed(B/f'{domain}_held_physical_baseline.npz',**pred,chi2=cs)
            for system in ['full','new_only']:results.append(dict(domain=domain,model='NNLS_physical_only',system=system,variant='correct',**pdf.metric(pred,clean['system']==system)))
        d.table(B/'controlled_held_response_metrics.csv',results);d.write(B/'paired_SED_uncertainty.json',uncertainty);d.table(B/'paired_counterfactuals.csv',paired)


def independent_identifiability():
    # A known-intrinsic-spectrum likelihood is an explicit upper-information
    # oracle: it diagnoses weak photometric features without blaming training.
    records=[]
    for task in ['L0','L1','L2_noise','L2_missing','L2_weak']:
        a=d.npz(B/f'{task}_val.npz');params=json.loads((B/f'{task}_val_truth.json').read_text());idx=np.arange(0,len(a['z']),max(len(a['z'])//8,1))[:8]
        for i in idx:
            par=params[i];clean=np.array([b.break_flux(float(z),b.HSC,par['slope'],par['strength'],par['av']) for z in pdf.grid()[1]]).T;f=a['flux'][i,:5];e=a['sigma'][i,:5];valid=np.isfinite(f)&np.isfinite(e)&(e>0);t=clean[valid];ff=f[valid];ee=e[valid];amp=np.maximum(0,np.sum(ff[:,None]*t/ee[:,None]**2,axis=0)/np.sum(t*t/ee[:,None]**2,axis=0));cs=np.sum(((ff[:,None]-amp*t)/ee[:,None])**2,axis=0);p=pdf.prediction(-.5*cs[None,:],a['z'][i:i+1]);records.append(dict(task=task,object_id=a['object_id'][i],fraction_z_bins_within_delta1=float(np.mean(cs-cs.min()<1)),P16_P84_width=float(p['z_p84'][0]-p['z_p16'][0]),oracle_median=float(p['z_p50'][0]),true_z=float(a['z'][i]),unknown_SED_more_degenerate_than_this_oracle=True))
    d.table(B/'source_identifiability_oracles.csv',records)


def diagnostics():
    syn=d.npz(ROOT/'reports/stage2b/1028/synthetic.npz');train=syn['split']=='train';val=syn['split']=='validation';yt=np.clip(np.searchsorted(pdf.grid()[0],syn['z'][train],side='left')-1,0,299);yv=np.clip(np.searchsorted(pdf.grid()[0],syn['z'][val],side='left')-1,0,299);occupied=np.unique(yt);miss=~np.isin(yv,occupied)
    d.write(B/'bin_support_audit.json',dict(train_observations=int(train.sum()),train_unique_SEDs=len(set(syn['group'][train])),train_unique_latent_redshifts=len(np.unique(syn['z'][train])),occupied_bins=int(len(occupied)),total_bins=300,source_validation_observations=int(val.sum()),fraction_validation_labels_in_unseen_training_bins=float(miss.mean()),training_z_quantiles=np.quantile(syn['z'][train],[0,.05,.5,.95,1]).tolist(),validation_z_quantiles=np.quantile(syn['z'][val],[0,.05,.5,.95,1]).tolist(),interpretation='Hard independently parameterized 300-bin classes have sparse label coverage; this is not a label-index bug or direct evidence of redshift non-identifiability'))
    loss=[];budgets=[]
    for path in sorted((B/'runs').glob('*/model.pt')):
        r=torch.load(path,map_location='cpu',weights_only=False);budgets.append(dict(tag=r['tag'],parameters=r['parameters'],steps=r['steps'],best_step=r['best_step'],best_validation_CE=r['best_validation_CE'],seconds=r['seconds'],coarse=r['coarse'],smooth=r['smooth'],learning_rate=r['lr']))
        for h in r['history']:
            if h['gradient']:loss.append(dict(experiment=r['tag'],update=h['update'],**h['gradient']))
    d.table(B/'loss_gradient_scales.csv',loss);d.table(B/'model_budgets.csv',budgets)
    # Effective error dominated by TEF is a weighting assumption, not additional
    # measured information. Compare it on actual original synthetic sources.
    t=numeric.tensor_rows(take(syn,train));var=t['e'].square();modelvar=(.1*t['f'].clamp_min(0)).square();valid=t['mask'];fraction=(modelvar/(var+modelvar))[valid].numpy();d.write(B/'effective_error_audit.json',dict(original_source_model_variance_fraction_quantiles=np.quantile(fraction,[.05,.5,.95]).tolist(),fraction_bands_model_error_dominant=float((fraction>.5).mean()),note='Observed-flux-dependent TEF is an approximate score, not a fully specified generative noise law. Noise-free matched controls use nominal1% measured errors; original decoder adds10% TEF. This distinction is explicit in physical-only comparisons.'))
    return budgets


def fair_error_controls():
    rows=read(B/'controlled_held_response_metrics.csv')
    if any(r['model']=='NNLS_same_TEF' for r in rows):return
    k=d.npz(A/'response_kernel_highres.npz')['kernel'];coefs=[]
    for domain in ['identifiable','analytic','matched']:
        path=B/f'{domain}_held_views.npz'
        if not path.exists():continue
        views=d.npz(path);clean=take(views,views['variant']=='correct');cs=np.array([a5.profile((f,e,ids,k,.1)) for f,e,ids in zip(clean['flux'],clean['sigma'],clean['band_id'])]);pred=pdf.prediction(-.5*cs,clean['z']);np.savez_compressed(B/f'{domain}_held_physical_same_TEF.npz',**pred,chi2=cs)
        for system in ['full','new_only']:rows.append(dict(domain=domain,model='NNLS_same_TEF',system=system,variant='correct',**pdf.metric(pred,clean['system']==system)))
        if domain=='analytic':continue
        source=identifiable('val',256) if domain=='identifiable' else b.matched('val',256);truth=d.npz(B/('identifiable_val_truth.npz' if domain=='identifiable' else 'matched_val_256_truth.npz'))['weights'];known=dict(zip(source['group'],truth));tag='identifiable_N2' if domain=='identifiable' else 'matched_N2';sp=d.npz(B/f'predictions/{tag}.npz');hp=d.npz(B/f'predictions/{domain}_N2_held_responses.npz');coefs.append(dict(domain=domain,view='source',median_weight_L1_error=float(np.median(abs(sp['sed_weights']-truth).sum(1))),median_true_component0=float(np.median(truth[:,0])),median_predicted_component0=float(np.median(sp['sed_weights'][:,0]))))
        for system in ['full','new_only']:
            mask=(views['system']==system)&(views['variant']=='correct');ww=np.array([known[g] for g in views['group'][mask]]);coefs.append(dict(domain=domain,view=system,median_weight_L1_error=float(np.median(abs(hp['sed_weights'][mask]-ww).sum(1))),median_true_component0=float(np.median(ww[:,0])),median_predicted_component0=float(np.median(hp['sed_weights'][mask,0]))))
    d.table(B/'controlled_held_response_metrics.csv',rows);d.table(B/'coefficient_recovery.csv',coefs)


def source_score_scales():
    results=[];ranges=[]
    for domain,tag,source in [('identifiable','identifiable_N2',identifiable('val',256)),('analytic','L3_N2_mismatched',d.npz(B/'L3_val.npz')),('matched','matched_N2',b.matched('val',256))]:
        r=torch.load(B/'runs'/tag/'model.pt',map_location='cpu',weights_only=False);model=b.model_for(r);t=numeric.tensor_rows(source,b.DEVICE)
        with torch.no_grad():
            logits,aux=model(t);chi=aux['chi'];full=-.5*(chi-chi.min(1,keepdim=True).values);neural=aux['neural'];tempered=.2*aux['physical_score'];n=t['mask'].sum(1)
            for label,ll in [('neural_only',neural),('physical_only_original_tempering',tempered),('physical_only_untempered',full),('combined_original',logits),('combined_untempered_diagnostic',neural+full)]:results.append(dict(domain=domain,condition=label,**pdf.metric(pdf.prediction(ll.cpu().numpy(),source['z']))))
            ranges.append(dict(domain=domain,median_effective_likelihood_temperature=float((n/.2).float().median()),median_neural_range=float((neural.max(1).values-neural.min(1).values).median()),median_tempered_physical_range=float((tempered.max(1).values-tempered.min(1).values).median()),median_untempered_physical_range=float((full.max(1).values-full.min(1).values).median())))
    d.table(B/'source_score_scale_controls.csv',results);d.write(B/'score_scale_audit.json',dict(source_only_posthoc_mathematical_diagnostic=True,settings_not_changed=True,held_response_or_real_target_not_reevaluated=True,formula='Physical score is original Gaussian profile log likelihood divided by Nvalid/beta: temperature25 for5bands and55 for11bands. This is in addition to the10% TEF.',ranges=ranges))


def source_convergence():
    from qwen_zero_shot_sanity import reserve_one_gpu
    traina=identifiable('train',1024);vala=identifiable('val',256);rows=[]
    d.write(B/'source_convergence_protocol.json',dict(reason='Both source-validation trajectories were still decreasing at the2000-update cutoff; inspect optimization confounding without reusing held-response results',updates=4000,models=['N1','N2'],settings='Same source-selected sigma3 neighborCE, lr0.002, seed11, data and weights initialization',held_response_or_real_target_evaluation=False))
    with reserve_one_gpu():
        for name in ['N1','N2']:
            tag='identifiable_source_convergence_'+name;r=b.train(traina,vala,name,tag,steps=4000,lr=.002,smooth=3.,coarse=False);rows.append(b.score_row(r,vala,tag))
    d.table(B/'source_convergence_metrics.csv',rows)


def plots():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for name in ['N0','N1','N2']:
        r=json.loads((B/'runs'/('replay_'+name)/'history.json').read_text());h=r['history'];axes[0].plot([x['update'] for x in h],[x['train_CE'] for x in h],label=name+' train');axes[0].plot([x['update'] for x in h],[x['source_validation_CE'] for x in h],ls='--',label=name+' val')
        for j,tag in enumerate(['L0_'+name,'identifiable_'+name],start=1):
            path=B/'runs'/tag/'history.json'
            if not path.exists():continue
            h=json.loads(path.read_text())['history'];axes[j].plot([x['update'] for x in h],[x['source_validation_CE'] for x in h],label=name)
    for ax in axes:ax.set(xlabel='Updates',ylabel='CE');ax.legend(fontsize=7)
    axes[0].set_title('Exact original replay');axes[1].set_title('Elementary source validation');axes[2].set_title('Identifiable matched physics')
    for ext in ['png','pdf']:fig.savefig(B/f'learning_audit.{ext}',dpi=150)
    plt.close(fig)


def reports(budgets):
    reproduction=read(B/'update_zero_reproduction.csv');ladder=read(B/'ladder_metrics.csv');control=read(B/'controlled_held_response_metrics.csv');ident=read(B/'identifiable_source_metrics.csv');gates=json.loads((B/'gates.json').read_text());support=json.loads((B/'bin_support_audit.json').read_text());oracle=json.loads((B/'oracle_tests.json').read_text());unc=json.loads((B/'paired_SED_uncertainty.json').read_text())
    lines=['# Stage 2B.5 — synthetic learnability and decoder audit','','Completed Stage2A.5 before this stage. Original scripts, Stage1 truth/splits and all prior reports/checkpoints remain unchanged. No real CLAUDS labels, source test photometry, Qwen training or population-scale real training are used. This is a diagnostic experiment, not a rerun of the registered Stage2B benchmark.','','## Exact update-zero reproduction','','| Arm | Train CE initial → final | Validation CE initial → final | Selected update | Parameter change L2 |','|---|---:|---:|---:|---:|']
    for r in reproduction:lines.append(f"| {r['model']} | {float(r['initial_train_CE']):.3f} → {float(r['final_train_CE']):.3f} | {float(r['initial_validation_CE']):.3f} → {float(r['final_validation_CE']):.3f} | {r['best_update']} | {float(r['parameter_change_L2']):.2f} |")
    lines+=['','All five initial validation CEs and update-zero selections reproduce within1e-4. Training gradients reach the heads/encoders and coefficient predictor; parameter changes are nonzero and losses decrease. The failure is poor validation generalisation while fitting training labels, not an optimizer that failed to run. Histories save gradient norms, reconstruction/CE gradients, parameter/head changes, entropy, occupied-probability bins and initial/final logits. AdamW includes all intended parameters and weight decay; no scheduler was configured in the original experiment. Training/eval mode and label indices are consistent.','',f"The original source training has {support['train_unique_SEDs']} independent SEDs, {support['train_unique_latent_redshifts']} redshift draws, and only {support['occupied_bins']}/300 occupied classes. {support['fraction_validation_labels_in_unseen_training_bins']:.1%} of source-validation labels fall in bins with no direct training label. Randomly initialized independent fine-bin logits can learn empirical class occupancy and memorize SEDs while becoming overconfident on validation. This is contributory evidence; the fixed source-only optimization screen does not establish that sparsity alone explains every failure.",'','## Elementary repairs and learnability','','The original padded N0 receives random band-to-slot assignments. On a fixed tiny spectrum family it collapsed to a constant predictor, whereas N1 and the independent known-family physical oracle learned. A new source-only N0 canonicalizes bands by numerical wavelength before flattening. Failed N0 attempts remain under *_unsorted and elementary_gate_before_N0_canonicalization.json. This restores a stable conventional MLP representation; it does not alter the original replay. N1/N2 retain shared-set invariance. A cache-key collision discovered before downstream experiments was also repaired by including requested object count in matched-dataset filenames; completed exact replay outputs were retained.','', 'L0 has independent tiny-jitter SED parameters, one prominent4000A break, uniform z0.1–1.1, no noise/dust and HSC curves. Positive1% nominal sigma supplies finite numerical weights; it is not a claim that random noise was added. Both N0 and N1 pass the declared validation gate, and an additional seed is reported. Tiny-overfit tests are debugging evidence, not generalisation. N2’s tiny matched task reduces CE but may retain ambiguous/wrong medians; that limitation is preserved.','','| New source experiment | N | NMAD | Outliers | CE | CRPS | Coverage | Concentration |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in ladder+ident:lines.append('| '+r['experiment']+' | '+r['n']+' | '+' | '.join(f'{float(r[k]):.4f}' for k in ['nmad','outlier_fraction','ce','crps','coverage_68','concentration_width_003'])+' |')
    lines+=['','L1 broadens independent slope/break parameters; L2 separately examines amplitude, dust, noise, missingness and weak features; L3 provides actual HSC/Legacy paired views of identical latent instances. Each new source stage uses1024 unique training SEDs and256 independent validation SEDs, not duplicated observations relabelled as independent objects. L3/matched stages have two views per SED. Task-specific seeds differ; cross-task differences are not matched causal effects. The pure amplitude symmetry is tested independently. Known-intrinsic-spectrum likelihoods in source_identifiability_oracles.csv show how weak features can remain ambiguous even with more intrinsic information than the networks possess.','', 'Four fixed source-only optimization candidates test original hard300 CE, lower learning rate, Gaussian-neighbour CE (sigma3 bins), and a30-output head mapped to300 equal-bin masses. The chosen protocol uses lowest original source-validation hard CE. It gains only modestly on the original benchmark; L4 uses this new labelled protocol and does not overwrite the original. The source-validation set is reused for this small diagnostic screen and is not an unbiased confirmation set. Original strict held-out filters/families are evaluated only for L4 models with >0.3 source-validation CE gain and nonzero selected update; skipped arms are not assigned an architectural ranking.','','## Physics audit and controlled unseen responses','',f"The known-mixture oracle recovers the injected grid redshift in {oracle['known_mixture_correct_fraction']:.1%} of128 noiseless HSC/Legacy observations. Both known-mixture and NNLS true-z chi-square approach zero. Free12-component fits with5 observed bands often have a continuum of exact fits across z; an arbitrary minimum on this plateau is not numerical failure. Learned mixtures have less intrinsic information than the privileged known-mixture oracle.",'','Random12-component mixtures test a deliberately underdetermined inverse problem. A second near-single-family FSPS0 +0–5% FSPS1 control makes the matched physics task identifiable, with1024/256 independent mixtures and the same source-selected optimizer. All three models have the same labels and updates. Models are locked before evaluating new responses. This distinguishes physical-decoder learnability from coefficient degeneracy.','', 'N2 keeps the explicit original bounded neural correction plus tempered physical profile score and reconstruction loss. Stage2A.5’s verified kernel is copied into new models; exact replay alone uses the old kernel. Gradient probes quantify CE/reconstruction norms and cosine alignment, and physical/neural logit ranges. Neural-only and learned-physical-only ablations are reported. Effective-error audit explicitly distinguishes1% nominal measurement errors in noiseless matched controls from the decoder’s additional10% observed-flux TEF. The latter often dominates and is a pseudo-likelihood weighting prescription, not a fully specified generative noise distribution. A broadened score or wider prediction distribution is not by itself evidence of physics benefit.','','| Controlled domain | Model | Unseen response system | NMAD | Outliers | CE | Coverage |','|---|---|---|---:|---:|---:|---:|']
    for r in control:
        if r['variant']=='correct':lines.append('| '+' | '.join([r['domain'],r['model'],r['system']]+[f'{float(r[k]):.4f}' for k in ['nmad','outlier_fraction','ce','coverage_68']])+' |')
    lines+=['','These controlled held views use independent source-validation SEDs reobserved through actual CLAUDS-like/full or strictly new UV/NIR curves; they are not real CLAUDS galaxies. Noise, missingness, wrong metadata and +0.1 redshift counterfactuals use locked models. New controlled sets are separately identified from original Stage1 strict evaluation membership. N2 has external FSPS prior information through the full response kernel, while N1 learns a finite response encoding; this information advantage is explicit. Known-mixture oracles have additional privileged latent information.','', 'Paired SED-unit bootstrap results (N2 minus N1 outlier fractions; negative favours N2):','',str(unc),'','No bootstrap treats repeated survey/noise views as independent SEDs. A physical advantage can be claimed only in the particular controlled conditions supported by these intervals, oracle adequacy and source convergence, not as general real-population transfer.','','## Direct conclusions and future components','','1. Original update-zero selection reflects decreasing training loss with worsening validation CE, sparse fine-bin supervision and SED ambiguity/mismatch. It does not mean weights or gradients were absent. The fixed small optimization screen yields limited rescue on the original benchmark; no single cause is proven sufficient.','','2. N0 and N1 can learn an identifiable elementary source problem; N0 needs stable canonical numeric input slots. N1’s shared aggregation passes permutation checks. Later weak-feature failures must be distinguished from optimization failures.','','3. The physical kernel, amplitude profile and NNLS recover the expected true-z fit when spectral assumptions match. Unknown mixtures can be non-identifiable. Empty/masked sets, negative fluxes, scaling symmetry and finite gradients pass. No true redshift enters inference; source z is used only to index a training reconstruction loss or explicitly labelled oracle diagnostic.','','4. The decoder is numerically usable as a tested, conditional physical constraint. Controlled source/held-response results above delimit any benefit. Neither a good elementary fit nor a wider target distribution establishes general physical transfer; mismatched analytic/CWW versus FSPS families and bounded neural corrections remain risks.','','5. Training data quantity and class resolution, intrinsic non-identifiability, stable input architecture, physical misspecification and source-noise assumptions all matter. The new ladder and original benchmark are separate experiments; sample/parameter budgets, convergence and extra template information are reported rather than hidden.','','6. No population-scale real experiment is justified here. Real-source high-z spectroscopy, an independent Legacy training population, trustworthy catalogue/calibration semantics and independently verified target redshifts are still required.','','7. No result demonstrates pretrained Qwen astrophysical knowledge or its necessity. Establish interpretable numerical/physical controls and independent real confirmation before any Stage2C comparison. No Qwen/large-backbone training was started.','','## Reproduction','','Run stage2a5_audit.py replay/kernels/analyze/details first, then stage2b5_learnability.py prepare/run and report_stage2_1028b.py controls/finalize. Original source splits and synthetic truth are immutable. See protocols, per-object PDFs, source/held metrics, gradients, exact runtime dependencies, model budgets, protected hashes and separate manifests.']
    scales=read(B/'source_score_scale_controls.csv')
    lines+=['','## Quantified physical-score attribution','','There is conditional synthetic evidence for a decoder benefit after source learning is valid: in the near-single-family matched test, full held-response outliers decrease from42.97% for N1 to30.08% for N2. The paired256-SED bootstrap gives N2−N1 =−12.89 percentage points (95% interval−21.09 to−4.69). The strictly new-only interval crosses zero, and arbitrary-mixture controls do not establish an improvement. Analytic-domain gains also leave high absolute error. These are conditional-on-checkpoint intervals, not uncertainty over training seeds; there is no general or real-CLAUDS transfer claim.','','The independent full-response NNLS baseline reaches0% outliers in the identifiable full-band matched test, both with measured errors alone and with the same10% TEF. N2 therefore leaves recoverable physical information unused. The measurement-only and same-TEF baselines are distinguished explicitly; no noise-model information advantage is hidden. Coefficient recovery is good for the near-single-family case (median L1 error about0.05), but poor for random12-component mixtures (about1.3–1.4), consistent with underdetermination from five source bands.','','The explicit score is Gaussian profile log likelihood divided by Nvalid/beta: temperature25 for five bands and55 for eleven bands, in addition to TEF broadening. In the identifiable source control its median physical-logit range is3.55 versus6.82 for the bounded neural term; the untempered physical range is88.86. About80.8% of original synthetic source bands have TEF variance larger than measured variance (median model fraction84.5%). This is deliberate score tempering, not a hidden arithmetic bug, but its effect is physically important: on identical frozen source predictions, physical-only outliers are50.0% under original tempering and0.59% without it.','','Source-only fixed scale ablations follow. They were derived from the mathematical score and do not change saved checkpoints or hyperparameters; held-response and real-target evaluations were not rerun under these alternatives. Removing tempering is not declared a general fix, since mismatched physics can then dominate a correct neural prediction.','','| Source domain | Frozen score diagnostic | Outliers | CE | CRPS |','|---|---|---:|---:|---:|']
    for r in scales:lines.append('| '+' | '.join([r['domain'],r['condition']]+[f'{float(r[k]):.4f}' for k in ['outlier_fraction','ce','crps']])+' |')
    convergence=read(B/'source_convergence_metrics.csv')
    if convergence:
        lines+=['','## Source-only convergence limitation','','The2000-update identifiable N1/N2 source curves were still decreasing, so an additional fixed4000-update diagnostic checks optimization confounding on source validation only. No held-response or real-target predictions are reevaluated with these later checkpoints. The earlier held-response gain is therefore conditional on that fixed training budget and does not establish an architecture ranking after matched convergence.','', '| Source-only longer-budget model | NMAD | Outliers | CE |','|---|---:|---:|---:|']
        for r in convergence:lines.append('| '+r['experiment']+' | '+' | '.join(f'{float(r[k]):.4f}' for k in ['nmad','outlier_fraction','ce'])+' |')
    (B/'report.md').write_text('\n'.join(lines)+'\n')
    # Correct an obsolete explanatory sentence from the initial source script.
    text=(A/'report.md').read_text().split('\nAt accurate HSC validation minima,')[0];text=text.replace('The original-configuration zspec residuals here use the accurate convolution as a common physical reference; archived original chi2(z) and minimum residuals retain the original kernel.','Original-configuration continuous-zspec residuals use the original EAZY convolution; refined configurations use independent Gauss convolution. Each retains its own grid-global chi2 and minimum residuals.')
    modes=read(A/'alternative_modes_and_rank.csv');flat=[float(x['fraction_bins_chi2_below1e_8']) for x in modes if x['condition']=='HSC_val_accurate'];text+='\nAt accurate HSC validation minima, median chi-square is effectively zero despite median continuous true-z chi-square311.1. The median fraction of redshift bins fitting to chi-square<1e-8 is '+f'{np.median(flat):.1%}'+'. With12 coefficients and5 measurements, exact-fit plateaus and their redshift extent matter: an arbitrary argmin inside them is not a unique physical solution. Formal i-band relative sigma is about0.07%; small SED/measurement discrepancies can therefore strongly suppress the true-redshift likelihood. Legacy W1/W2 supplies crucial additional wavelength leverage; the matched optical-only comparison supports this explanation, but does not identify a unique calibration or template defect.\n';(A/'report.md').write_text(text)
    joint=['# Sequential diagnostic execution — 1028b','','[Stage2A.5 report](stage2a5/1028b/report.md) and [Stage2B.5 report](stage2b5/1028b/report.md) were completed separately and sequentially.','','The HSC/Legacy discrepancy is not explained by a nanomaggy conversion, column-order or NNLS implementation failure. Refined integration leaves HSC outliers near80%. At spectroscopic redshift the HSC median chi-square is311, but a flexible12-template model can fit five bands almost exactly at wrong high redshifts. HSC’s very small formal errors reject small SED/measurement mismatches; absent priors and broad degenerate high-z solution regions bias the profile pseudo-posterior. On identical galaxies, Legacy has7.0% outliers versus86.7% for HSC; removing W1/W2 raises Legacy outliers to97.7%. Wavelength coverage is therefore a demonstrated factor; population differences alone cannot explain the contrast.','','Exact replay shows neural training loss falls while validation loss worsens. Update zero is a checkpoint-selection outcome, not evidence that training did nothing. Sparse300-bin supervision, model/SED ambiguity and overfitting contribute. The new N0 canonicalization fixes a separately demonstrated unstable-input control; N0/N1 then both learn an identifiable break task with zero validation outliers. The known-mixture physical oracle recovers all injected redshifts; arbitrary12-component mixtures remain underdetermined.','','The controlled ladder, near-single-family matched test, physical/neural ablations and SED bootstrap in Stage2B.5 delimit when these components learn or transfer across synthetic responses. These are source/synthetic diagnostics only. Original difficult-benchmark arms are not ranked unless independent source-validation learning passes; no real CLAUDS labels are used to choose settings or rerun target evaluation.','','Suitable future components are the independently checked union-knot response integrator, signed-flux/uncertainty masks, stable numerical inputs, shared-set encoder and tested nonnegative amplitude/SED decoder. Their empirical usefulness depends on identifiable spectra, adequate training support and an appropriate noise/template model. Unknown foreground histories, HSC response mixtures, cmodel/Tractor conventions, SED mismatch and the uncalibrated10% template-error prescription remain explicit.','','A claim about Qwen’s pretrained astrophysical knowledge remains premature: no controlled foundation-model comparison here separates pretraining from numerical optimization, physical prior information, source coverage or measurement mismatch. Independent target spectroscopy and a viable mixed-source training population remain missing. No Qwen training, Stage2C, source-test evaluation or population-scale real training was launched.']
    joint+=['','A concrete loss-scale issue is now quantified: N2’s physical score has temperature Nvalid/0.2, i.e.25 for five bands and55 for eleven, in addition to10% TEF. In the identifiable source test, frozen physical-only outliers change from50.0% under original tempering to0.59% without it. This source-only diagnosis was not applied to held-response or real-target evaluations. It does not imply that untempered mismatched physics is safe.','','After valid source learning, N2 shows a limited synthetic full-response gain over N1 in the identifiable matched test: outliers43.0%→30.1%, paired SED-bootstrap difference−12.9 points (95% interval−21.1 to−4.7). The new-only interval crosses zero, error remains high, and the independent physical baseline performs much better. N1/N2 source curves were still improving at the cutoff; a separate4000-update source-only convergence check is reported without reevaluating held responses. Thus matched-convergence architectural superiority is not established. This supports further controlled decoder/score investigation, not a population-scale real run or a pretrained-Qwen claim.']
    (ROOT/'reports/joint_physics_1028b.md').write_text('\n'.join(joint)+'\n')


def manifest():
    protected=json.loads((A/'protected_before.json').read_text())
    for path,sha in protected.items():assert d.sha(path)==sha,path
    code={str(path.relative_to(ROOT)):d.sha(path) for path in [ROOT/'scripts/stage2a5_audit.py',ROOT/'scripts/stage2b5_learnability.py',Path(__file__)]}
    for folder,stage in [(A,'Stage2A.5'),(B,'Stage2B.5')]:
        d.write(folder/'manifest.json',dict(stage=stage,code_sha256=code,protected_prior_files=len(protected),originals_unchanged=True,real_target_labels_read=False,source_test_photometry_read=False,artifacts={str(path.relative_to(folder)):d.sha(path) for path in folder.rglob('*') if path.is_file() and path.name!='manifest.json' and path.suffix!='.log'}))
    d.write(ROOT/'reports/physics_execution_1028b_manifest.json',dict(guide_sha256=d.sha(ROOT/'phys_guide_execution_1028b.md'),prompts={name:d.sha(ROOT/name) for name in ['phys_guide_1027caa.md','phys_guide_1027cba.md']},Stage2A5='stage2a5/1028b/report.md',Stage2B5='stage2b5/1028b/report.md',joint='joint_physics_1028b.md',prior_files_unchanged=len(protected),Qwen_training=False,Stage2C=False,real_CLAUDS_evaluated=False,population_training=False))


def finalize():
    fair_error_controls();source_score_scales();independent_identifiability();budgets=diagnostics();plots();reports(budgets);manifest();print('Both diagnostic stages, source-only controls, independent reports and joint assessment complete.')


def refresh():
    budgets=diagnostics();reports(budgets);manifest();print('Reports and manifests refreshed from completed fixed diagnostics.')


def verify():
    protected=json.loads((A/'protected_before.json').read_text())
    for path,sha in protected.items():assert d.sha(path)==sha,path
    assert json.loads((B/'full_replay_trajectory_check.json').read_text())['passed']
    assert json.loads((B/'final_numeric_checks.json').read_text())['passed']
    assert json.loads((A/'NNLS_tests.json').read_text())['passed']
    assert json.loads((B/'elementary_gate.json').read_text())['passed']
    assert (A/'handoff.json').stat().st_mtime_ns<(B/'protocol.json').stat().st_mtime_ns
    checked=0
    for folder in [A,B]:
        for path in folder.rglob('*.npz'):
            with np.load(path,allow_pickle=False) as f:
                if 'pz' not in f.files:continue
                assert np.isfinite(f['pz']).all() and np.isfinite(f['logits']).all() and np.allclose(f['pz'].sum(1),1,atol=2e-6),str(path)
                checked+=1
    records=[]
    for cohort in ['HSC_val','Legacy_val']:
        obs=d.npz(A/f'input_{cohort}.npz');before=d.npz(A/f'{cohort}_original.npz');after=d.npz(A/f'{cohort}_accurate.npz')
        for i in range(32):
            f,e,ids=obs['flux'][i],obs['sigma'][i],obs['band_id'][i];valid=np.isfinite(f)&np.isfinite(e)&(e>0)&(ids>=0);oldmodel=f[valid]-before['residual_zspec'][i,valid]*e[valid];newmodel=a5.continuous(float(obs['z'][i]))[ids[valid]]@before['coeff_zspec'][i];delta=(newmodel-oldmodel)/e[valid];newchi=float(np.sum(((f[valid]-newmodel)/e[valid])**2));records.append(dict(cohort=cohort,object_id=obs['object_id'][i],z_spec=float(obs['z'][i]),max_absolute_model_difference_over_measured_sigma=float(np.max(abs(delta))),RMS_model_difference_over_measured_sigma=float(np.sqrt(np.mean(delta**2))),delta_chi2_at_same_original_coefficients=float(newchi-before['chi2_zspec'][i]),delta_chi2_after_refitting=float(after['chi2_zspec'][i]-before['chi2_zspec'][i])))
    d.table(A/'numerical_significance_at_zspec.csv',records)
    oldmanifest=json.loads((pdf.OUT/'physical_manifest.json').read_text());assert d.sha(ROOT/'configs/filter_registry_v1.json')==oldmanifest['filter_registry_sha256']
    d.write(ROOT/'reports/physics_execution_1028b_verification.json',dict(passed=True,protected_prior_files=len(protected),old_filter_registry_unchanged=True,normalized_finite_PDF_archives_checked=checked,full_original_neural_replay_exact=True,source_only=True,no_real_CLAUDS_label_evaluation=True,Stage2A5_handoff_precedes_Stage2B5_protocol=True))
    manifest();print('Preservation, full replay, sequencing, PDFs and continuous-zspec numerical significance verified.')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['controls','source_convergence','finalize','refresh','manifest','verify']);args=p.parse_args();globals()[args.action]()
