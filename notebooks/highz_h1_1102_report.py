"""H1 fixed-final comparison with paired-galaxy uncertainty, never P selection."""
import json
from pathlib import Path
import numpy as np
from highz_h1_1102 import OUT,ARMS,QUEUE,config,read,write,sha,now,validate,freeze_final
from hsc_legacy_paired_report import object_metrics,metrics,lower_median


def pending_report():
    audit=json.loads((OUT/'isolation_audit/worktree.json').read_text())
    sample=json.loads((OUT/'splits/membership_audit.json').read_text())
    heads=json.loads((OUT/'protocol/head_audit.json').read_text())
    text=['# HSC high-redshift H1 pilot, zcut=0.6','',
        'Six seed-42 Qwen3.5-4B-Base QLoRA arms are prepared for isolated execution. Results are pending. The original matched-transfer and morphology branches retain their five GPU reservations and their unchanged code/configuration hashes.','',
        f"Worktree: `{audit['worktree']}`; branch `{audit['branch']}`; base `{audit['base_sha']}`. All new artifacts are beneath this execution root. Shared pretrained weights and frozen source assets are read-only. See `isolation_audit/` and `manifest.json` for identities, independent reservations and exact modified files.",'',
        '| Condition | Distinct train | z<0.6 | z>=0.6 | Low-z validation |','|---|---:|---:|---:|---:|']
    for c,v in sample['counts'].items():text.append(f"| {c} | {v['total']} | {v['below_cutoff']} | {v['at_or_above_cutoff']} | {v['validation']} |")
    text+=['','Every condition is used by both heads with identical membership and presentation order. Budget per arm: 10 epochs, 3 head-only + 7 joint, 6,560 optimizer updates and 104,890 presentations; microbatch1/accumulation16 including correctly normalized partial windows.','',
        f"Head parameters: {heads['parameters']}. The common hidden trunk is copied exactly, but the head capacities differ. The mixture has three globally truncated Gaussian components and returns integrated native-bin log masses to the unchanged CE and evaluator. Sigma is fixed to the differentiable range[0.0075,4.5], chosen before P predictions. Arithmetic uses float64 log-CDF differences. See `head_tests/` and `protocol/head_audit.json`.",'',
        'Unobserved softmax bins receive CE gradient p_j and therefore suppression pressure; this does not prove extrapolation impossible. Both head initializations, extreme/narrow/broad PDFs, independent numerical integration, gradients, output updates, native grid/loss/quantiles, save/reload, exact interrupted resume and the six-checkpoint P gate are tested with synthetic inputs and source-only records.','',
        '| Arm | Condition | Head | Status |','|---|---|---|---|']
    for a in QUEUE:
        c=ARMS[a];run=OUT/'training'/a
        status='complete' if (run/'complete.json').exists() else ('running/resumable' if (run/'progress.json').exists() else 'queued')
        text.append(f"| {a} | {c['condition']} | {c['head']} | {status} |")
    text+=['','Primary checkpoints are the fixed epoch-10 final states. Minimum low-z source-validation CE checkpoints are retained as secondary diagnostics only. All six primary identities and source-reload replays must freeze before any high-z P scoring.','',
        'The locked P benchmark has been previously analysed in other RedFM studies; it is not a new prospective blind test. It contains53,786 below-cutoff and7,910 at/above-cutoff galaxies. Their predictions are never used for tuning, checkpoint selection, minimum scale, cohort changes or protocol repairs.','',
        'The eventual report includes absolute point/probabilistic/calibration metrics, below/above and fixed-redshift strata, training-prior and boundary baselines, paired bootstrap contrasts, PDF broadening/collapse diagnostics, selection shifts and histories. No extrapolation conclusion or expansion decision is available until training and locked evaluation finish. Natural-tail and enriched-tail arms are high-z-supervision controls, not strict extrapolation. One seed does not establish seed robustness.','',
        'Next scientific questions—unobserved-bin suppression, head effect, few-label effect, recovery versus PDF broadening, expansion value and Legacy1.5 readiness—remain pending. No Legacy, DoRA, extra seed/model, MLP/TabICL, zcut1.5 or z>2 run is launched. Foundation pretraining exposure is unknown; this is supervised-support extrapolation and also a joint galaxy-property/selection shift.']
    (OUT/'report.md').write_text('\n'.join(text)+'\n')


def load_predictions(arm,d):
    folder=OUT/'evaluation'/arm;done=json.loads((folder/'complete.json').read_text())
    for p,h in done['prediction_sha256'].items():assert sha(folder/p)==h
    chunks=[read(p) for p in sorted(folder.glob('??????_??????.npz'))]
    pred={k:chunks[0][k] if k.startswith('redshift_') else np.concatenate([p[k] for p in chunks]) for k in chunks[0]}
    np.testing.assert_array_equal(pred['targetid'],d['targetid']);np.testing.assert_array_equal(pred['z_spec'],d['z'])
    for part in chunks:
        for k in ['redshift_edges','redshift_centers']:np.testing.assert_array_equal(part[k],pred[k])
    np.testing.assert_allclose(pred['pz'].sum(1),1,atol=2e-6,rtol=1e-5)
    assert np.isfinite(pred['logits']).all() and np.isfinite(pred['pz']).all() and (pred['pz']>=0).all()
    return pred


def additional_objects(p):
    o=object_metrics(p);edge=p['redshift_edges'].astype(float)
    frac=np.clip((edge[1:]-np.maximum(edge[:-1],.6))/np.diff(edge),0,1)
    o.update(zphot=p['z_p50'],above_mass=(p['pz']*frac).sum(1),width68=p['z_p84']-p['z_p16'],
        entropy=-(p['pz']*np.log(np.maximum(p['pz'],1e-30))).sum(1))
    return o


def score(o,r):
    m=metrics(o,r);med=o['zphot'][r]
    m.update(mean_mass_above_cutoff=float(o['above_mass'][r].mean()),median_predicted_z=lower_median(med),
        fraction_predicted_below_cutoff=float((med<.6).mean()),mean_entropy=float(o['entropy'][r].mean()),mean_width68=float(o['width68'][r].mean()))
    return m


def estimate(point,draws,keys):
    draws=np.asarray(draws);q=np.nanquantile(draws,[.025,.975],axis=0)
    return {k:dict(value=float(point[j]),conditional_galaxy_ci95=q[:,j].tolist()) for j,k in enumerate(keys)}


def baseline_objects(mass,d):
    import torch
    from aion_magnitude.metrics import predict_photoz_from_logits
    from aion_magnitude.utils import make_redshift_grid
    edge,center=make_redshift_grid(0,4.5,300)
    logits=torch.from_numpy(np.broadcast_to(np.log(mass),(len(d['z']),300)).copy()).float()
    v=predict_photoz_from_logits(logits,center);p={k:x.numpy() for k,x in v.items()}
    p.update(logits=logits.numpy(),z_spec=d['z'],redshift_edges=edge.numpy(),redshift_centers=center.numpy())
    return additional_objects(p)


def analysis(d,objects,policy,master):
    """Same original master draws conditioned on each stratum, paired all arms."""
    z=d['z'];groups={'below_cutoff':np.flatnonzero(z<.6),'above_cutoff':np.flatnonzero(z>=.6)}
    bins=policy['redshift_bins'];groups.update({f'z_{lo:g}_{hi:g}':np.flatnonzero((z>=lo)&(z<hi)) for lo,hi in zip(bins[:-1],bins[1:])})
    names=list(objects);result={};draw_export={}
    keys=list(score(objects[names[0]],np.arange(len(z))))
    for label,rows in groups.items():
        if not len(rows):result[label]=dict(n=0,exploratory=True,cells={},contrasts={});continue
        membership=np.zeros(len(z),bool);membership[rows]=True
        point=np.array([list(score(objects[n],rows).values()) for n in names]);draws=[]
        for i,draw in enumerate(master):
            selected=draw[membership[draw]]
            if not len(selected):continue
            draws.append([list(score(objects[n],selected).values()) for n in names])
        draws=np.asarray(draws);draw_export[label]=draws
        cells={n:estimate(point[j],draws[:,j],keys) for j,n in enumerate(names)}
        contrasts={}
        definitions={f'{b}_minus_{a}':(a,b) for a,b in [('H1-A','H1-D'),('H1-A','H1-B'),('H1-A','H1-C'),('H1-D','H1-E'),('H1-D','H1-F'),('H1-B','H1-C'),('H1-E','H1-F')]}
        for arm in ARMS:
            definitions[f'{arm}_minus_training_prior']=('prior_'+ARMS[arm]['condition'],arm)
            definitions[f'{arm}_minus_boundary']=('boundary',arm)
        for name,(a,b) in definitions.items():
            ia,ib=names.index(a),names.index(b);contrasts[name]=estimate(point[ib]-point[ia],draws[:,ib]-draws[:,ia],keys)
        diagnostics={}
        for n,o in objects.items():
            med=o['zphot'][rows];iqr=float(np.quantile(med,.75)-np.quantile(med,.25));z_iqr=float(np.quantile(z[rows],.75)-np.quantile(z[rows],.25))
            diagnostics[n]=dict(predicted_median_IQR=iqr,true_redshift_IQR=z_iqr,predicted_median_std=float(med.std()),
                boundary_collapse=bool(iqr<.03 and abs(float(np.median(med))-.6)<.03 and z_iqr>.1),
                below_support_failure=bool((med<.6).mean()>=.8 and o['above_mass'][rows].mean()<.1),
                fraction_with_median_above_cutoff=float((med>=.6).mean()),
                pit_counts=np.histogram(o['pit'][rows],np.linspace(0,1,21))[0].tolist(),
                warning='Broad PDFs and input-dependent medians are diagnostics; neither alone establishes useful high-z recovery')
        result[label]=dict(n=len(rows),exploratory=len(rows)<policy.get('exploratory_count_threshold',100),cells=cells,contrasts=contrasts,diagnostics=diagnostics,valid_bootstrap_draws=len(draws))
        print('H1 PAIRED BOOTSTRAP',label,len(rows),flush=True)
    return dict(strata=result,metric_names=keys,cell_names=names),draw_export


def report():
    manifest=validate();gate=freeze_final();policy=json.loads((OUT/'protocol/evaluation_policy.json').read_text())
    for p,h in manifest['P_sha256'].items():assert sha(p)==h
    d=read(OUT/'evaluation/locked_P.npz');objects={};example_pdfs={}
    for arm in ARMS:
        p=load_predictions(arm,d);o=additional_objects(p);objects[arm]=o
        np.savez_compressed(OUT/'evaluation'/arm/'object_metrics.npz',targetid=d['targetid'],z_spec=d['z'],**o)
        # Outcome-independent deterministic examples: smallest TARGETID per fixed z bin.
        examples=[]
        for lo,hi in zip(policy['redshift_bins'][:-1],policy['redshift_bins'][1:]):
            rows=np.flatnonzero((d['z']>=lo)&(d['z']<hi))
            if len(rows):examples.append(int(rows[np.argmin(d['targetid'][rows])]))
        example_pdfs[arm]=p['pz'][examples]
        del p
    baselines=read(OUT/'protocol/baselines.npz')
    for name in ['lowz_only','natural_tail','tail_enriched']:objects['prior_'+name]=baseline_objects(baselines[name],d)
    objects['boundary']=baseline_objects(baselines['boundary'],d)
    bootstrap=policy['bootstrap'];assert sha(bootstrap['master_file'])==bootstrap['master_sha256'];master=np.load(bootstrap['master_file'],mmap_mode='r');assert master.shape==(1000,len(d['z']))
    result,draws=analysis(d,objects,policy,master)
    np.savez_compressed(OUT/'evaluation/bootstrap_metrics.npz',names=np.array(list(objects)),metrics=np.array(result['metric_names']),**draws)
    np.savez_compressed(OUT/'evaluation/example_pdfs.npz',targetid=d['targetid'][examples],z_spec=d['z'][examples],**example_pdfs)
    histories={a:json.loads((OUT/'training'/a/'history.json').read_text()) for a in ARMS}
    resources={a:json.loads((OUT/'training'/a/'resource_cost.json').read_text()) for a in ARMS}
    result.update(created_utc=now(),primary='fixed_final_epoch10',secondary_checkpoints=gate['models'],histories=histories,resources=resources,
        policy_sha256=sha(OUT/'protocol/evaluation_policy.json'),uncertainty='1000 shared conditioned paired-galaxy draws; one training seed42, no training-seed robustness claim')
    write(OUT/'evaluation/results.json',result)
    pending_report();text=(OUT/'report.md').read_text();text=text.replace('Results are pending.','All six fixed-budget training arms and primary locked-P evaluations are complete.')
    text+='\n## Fixed-final extrapolation results\n\n'
    for label in ['below_cutoff','above_cutoff']:
        r=result['strata'][label];text+=f"\n{label}, n={r['n']:,}.\n\n| Model/baseline | CE | CRPS | NMAD | Outliers | Bias | Coverage68 | Mean mass>=0.6 | Median predicted z | Predicted below0.6 | Entropy |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        for name,c in r['cells'].items():
            v={k:x['value'] for k,x in c.items()};text+=f"| {name} | {v['ce']:.5f} | {v['crps']:.5f} | {v['nmad']:.5f} | {v['outlier']:.2%} | {v['bias']:.5f} | {v['coverage68']:.2%} | {v['mean_mass_above_cutoff']:.5f} | {v['median_predicted_z']:.5f} | {v['fraction_predicted_below_cutoff']:.2%} | {v['mean_entropy']:.4f} |\n"
        text+='\n| Paired contrast | Delta CE [95% CI] | Delta outlier fraction [95% CI] |\n|---|---|---|\n'
        for n,c in r['contrasts'].items():
            ce=c['ce'];o=c['outlier'];text+=f"| {n} | {ce['value']:.5f} {ce['conditional_galaxy_ci95']} | {o['value']:.5f} {o['conditional_galaxy_ci95']} |\n"
    text+='\n## Fixed redshift strata and scientific review\n\n| Stratum | N | Exploratory (<100) |\n|---|---:|---|\n'
    for label,r in result['strata'].items():text+=f"| {label} | {r['n']} | {r['exploratory']} |\n"
    high=result['strata']['above_cutoff'];text+='\nInterpretation uses both recovery and PDF broadening; CE improvement alone is insufficient. Low-z-only A/D are strict supervised-support holdouts. B/C/E/F have high-z supervised exposure. Photometric SNR/magnitude shifts are recorded in `protocol/selection_diagnostics.json`.\n'
    a=high['cells']['H1-A'];dd=high['cells']['H1-D']
    def val(c,k):return c[k]['value']
    text+=f"\n1. Softmax suppression evidence: A mean high-z probability mass={val(a,'mean_mass_above_cutoff'):.6f}, fraction high-z medians below0.6={val(a,'fraction_predicted_below_cutoff'):.2%}. Compare its initial source-only mass and training trajectory; suppression is a diagnostic, not an impossibility proof.\n"
    effect=high['contrasts']['H1-D_minus_H1-A'];text+=f"\n2. Continuous-head effect: D-A CE={effect['ce']}, outlier={effect['outlier']}. D has median predicted z={val(dd,'median_predicted_z'):.5f}, mean entropy={val(dd,'mean_entropy'):.5f}, width68={val(dd,'mean_width68'):.5f}.\n"
    for x,y in [('H1-B','H1-A'),('H1-C','H1-A'),('H1-E','H1-D'),('H1-F','H1-D')]:
        c=high['contrasts'][f'{x}_minus_{y}'];text+=f"\n3. High-z supervision {x}-{y}: CE={c['ce']}; outlier={c['outlier']}. Membership counts are in the sample table above; natural-tail is not a few-shot claim without these counts.\n"
    text+='\n4. Recovery versus broadening must be assessed from all displayed point metrics, width/entropy, calibration, both source-only baselines, redshift bins, collapse flags and input-dependent prediction spread. No single metric establishes astrophysical recovery.\n'
    meaningful=effect['ce']['conditional_galaxy_ci95'][1]<0 and effect['outlier']['conditional_galaxy_ci95'][1]<0
    text+=f"\n5. This pilot shows paired improvement in both high-z CE and outlier rate for D versus A: {meaningful}. Even if true, independent training seeds and a scientific review are needed before expanding. No expansion is launched.\n"
    text+='\n6. Before Legacy zcut1.5: independently review DESI1.5–1.6 label reliability and selection, excluded-exposure independence, sufficient tail validation/control construction, final sample/budget/head choices and a new resource reservation. Neither this paired-P experiment nor the separate prepared Legacy design authorizes execution.\n'
    text+='\nComplete per-bin absolute metrics and paired intervals, median bias, PIT histograms, collapse flags, per-object residuals/PDF-mass diagnostics, all six histories, selected-versus-final identities and GPU consumption are in `evaluation/results.json`. Figures show predictions, calibration, high-z mass, fixed-bin scores and source histories. Training-condition loss differences reflect different label populations. Foundation pretraining exposure and causal attribution between representation and head remain unresolved by this architectural comparison.\n'
    (OUT/'report.md').write_text(text);plots(d,objects,result,histories)


def plots(d,objects,result,histories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,9))
    for ax,a in zip(axes.flat,ARMS):
        ax.hexbin(d['z'],objects[a]['zphot'],gridsize=55,mincnt=1,bins='log');ax.axvline(.6,color='r',ls=':');ax.axhline(.6,color='r',ls=':');ax.plot([0,2],[0,2],'k:',lw=.5);ax.set(title=a,xlabel='Spectroscopic z',ylabel='Predicted median z',xlim=(0,2),ylim=(0,2))
    fig.tight_layout();fig.savefig(OUT/'figures/H1_predictions.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for a in ARMS:
        o=objects[a];r=d['z']>=.6
        axes[0,0].hist(o['pit'][r],np.linspace(0,1,21),density=True,histtype='step',label=a)
        axes[0,1].plot(np.arange(1,11),[x['val']['cross_entropy'] for x in histories[a]],label=a)
        bins=[k for k,v in result['strata'].items() if k.startswith('z_') and v['n']]
        ce=[result['strata'][b]['cells'][a]['ce']['value'] for b in bins]
        axes[1,0].plot(np.arange(len(bins)),ce,marker='.',label=a)
        axes[1,1].hist(o['above_mass'][r],np.linspace(0,1,31),density=True,histtype='step',label=a)
    axes[0,0].set(xlabel='High-z binned PIT',ylabel='Density');axes[0,1].set(xlabel='Epoch',ylabel='Low-z validation CE')
    axes[1,0].set(xticks=np.arange(len(bins)),xticklabels=bins,ylabel='Fixed-bin CE');axes[1,0].tick_params(axis='x',rotation=50)
    axes[1,1].set(xlabel='High-z probability mass above0.6',ylabel='Density')
    for ax in axes.flat:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'figures/H1_calibration_and_support.png',dpi=150);plt.close(fig)
