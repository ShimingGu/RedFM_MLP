"""Shared-index historical inference analysis and source-only training gate report."""
from pathlib import Path
import json
import numpy as np
from bidirectional_1029 import OUT,PKG,ROOT,NAMES,BINS,read,write,sha,now,verify,load_manifest
from hsc_legacy_paired_report import object_metrics,metrics

def cell_predictions(name,domain,d):
    folder=OUT/'cells'/name/domain;complete=json.loads((folder/'complete.json').read_text())
    verify({str(folder/k):v for k,v in complete['prediction_sha256'].items()})
    chunks=[read(p) for p in sorted(folder.glob('??????_??????.npz'))]
    p={k:(chunks[0][k] if k.startswith('redshift_') else np.concatenate([c[k] for c in chunks])) for k in chunks[0]}
    np.testing.assert_array_equal(p['targetid'],d['targetid']);np.testing.assert_array_equal(p['z_spec'],d['z'])
    for c in chunks:
        for k in ('redshift_edges','redshift_centers'):np.testing.assert_array_equal(c[k],p[k])
    np.testing.assert_allclose(p['pz'].sum(1),1,rtol=1e-5,atol=1e-6)
    return p

def historical():
    m=load_manifest();verify(m['protected_sha256']);d=read(OUT/'cohort.npz');n=len(d['z']);rows=np.arange(n)
    labels=[f'{name}/{domain}' for name in NAMES for domain in ('hsc','legacy')];objs=[];zphot={};cells={}
    for label in labels:
        name,domain=label.split('/');p=cell_predictions(name,domain,d)
        for k in ('redshift_edges','redshift_centers'):np.testing.assert_array_equal(p[k],np.array(m[k],np.float32))
        o=object_metrics(p);objs.append(o);zphot[label]=p['z_p50'];r=metrics(o,rows)
        r.update(pit_histogram_counts=np.histogram(o['pit'],np.linspace(0,1,21))[0].tolist(),pit_histogram_edges=np.linspace(0,1,21).tolist())
        cells[label]=dict(metrics=r,n=n,interpretation='native' if name.split('-')[0]==domain else 'interface-stress diagnostic',reused=name.startswith('hsc-'))
        np.savez_compressed(OUT/'cells'/name/domain/'object_metrics.npz',targetid=d['targetid'],z_spec=d['z'],z_phot=p['z_p50'],**o)
        del p
    keys=list(metrics(objs[0],rows));brows=[np.flatnonzero((d['z']>=lo)&((d['z']<=hi) if j==4 else (d['z']<hi))) for j,(lo,hi) in enumerate(zip(BINS[:-1],BINS[1:]))]
    assert all(len(r) for r in brows),'Predeclared empty bin requires explicit handling'
    point=np.array([[metrics(o,rows)[k] for k in keys] for o in objs])
    bp=np.array([[[metrics(o,r)[k] for k in keys] for o in objs] for r in brows])
    boot=np.load(OUT/'bootstrap_indices.npy',mmap_mode='r');vals=np.empty((1000,8,len(keys)));bvals=np.empty((1000,5,8,len(keys)))
    for i,r in enumerate(boot):
        samplebins=[r[(d['z'][r]>=lo)&((d['z'][r]<=hi) if j==4 else (d['z'][r]<hi))] for j,(lo,hi) in enumerate(zip(BINS[:-1],BINS[1:]))]
        for c,o in enumerate(objs):
            v=metrics(o,r);vals[i,c]=[v[k] for k in keys]
            for j,br in enumerate(samplebins):
                v=metrics(o,br);bvals[i,j,c]=[v[k] for k in keys]
        if (i+1)%50==0:print('SHARED BOOTSTRAP',i+1,'/1000',flush=True)
    np.savez_compressed(OUT/'bootstrap_metrics.npz',keys=np.array(keys),cells=np.array(labels),full=vals,redshift_bins=bvals,equal_bin_macro=bvals.mean(1))
    def estimate(p,v):
        ci=np.quantile(v,[.025,.975],axis=0)
        return {k:dict(value=float(p[i]),ci95=ci[:,i].tolist()) for i,k in enumerate(keys)}
    binresults=[]
    for j,r in enumerate(brows):binresults.append(dict(low=BINS[j],high=BINS[j+1],n=len(r),cells={label:estimate(bp[j,c],bvals[:,j,c]) for c,label in enumerate(labels)}))
    for c,label in enumerate(labels):
        cells[label]['full_ci95']=estimate(point[c],vals[:,c]);cells[label]['equal_bin_macro']=estimate(bp[:,c].mean(0),bvals[:,:,c].mean(1))
    contrasts={};diffs={'targetid':d['targetid'],'z_spec':d['z']}
    def contrast(label,weights):
        w=np.array([weights.get(l,0) for l in labels]);v=np.einsum('bcK,c->bK',vals,w);p=w@point
        br=np.einsum('bjcK,c->bjK',bvals,w);pp=np.einsum('jcK,c->jK',bp,w)
        contrasts[label]=dict(definition=weights,full=estimate(p,v),redshift_bins=[estimate(pp[j],br[:,j]) for j in range(5)],equal_bin_macro=estimate(pp.mean(0),br.mean(1)))
        diffs[label+'_ce']=sum(w[c]*objs[c]['ce'] for c in range(8))
    for method in ('qlora','dora'):
        hh=f'hsc-{method}/hsc';hl=f'hsc-{method}/legacy';lh=f'legacy-{method}/hsc';ll=f'legacy-{method}/legacy'
        for label,weights in [('H_to_L_penalty',{hl:1,hh:-1}),('L_to_H_penalty',{lh:1,ll:-1}),('H_to_L_target_native_gap',{hl:1,ll:-1}),('L_to_H_target_native_gap',{lh:1,hh:-1}),('source_observation_interaction',{hl:1,hh:-1,lh:1,ll:-1}),('directional_penalty_asymmetry',{hl:1,hh:-1,lh:-1,ll:1})]:contrast(f'{method}_{label}',weights)
        for direction,src,dst in [('H_to_L',hh,hl),('L_to_H',ll,lh)]:
            a,b=objs[labels.index(src)],objs[labels.index(dst)];oa=np.abs(a['delta'])>.15;ob=np.abs(b['delta'])>.15
            contrasts[f'{method}_{direction}_penalty']['outlier_transitions']=dict(newly_catastrophic=int((~oa&ob).sum()),rescued=int((oa&~ob).sum()),both=int((oa&ob).sum()),neither=int((~oa&~ob).sum()))
            diffs[f'{method}_{direction}_delta_abs_residual']=np.abs(b['delta'])-np.abs(a['delta'])
            diffs[f'{method}_{direction}_delta_crps']=b['crps']-a['crps']
    for s in ('hsc','legacy'):
        for domain in ('hsc','legacy'):contrast(f'DoRA_minus_QLoRA_{s}_on_{domain}',{f'{s}-dora/{domain}':1,f'{s}-qlora/{domain}':-1})
    for direction,source,target in [('H_to_L','hsc','legacy'),('L_to_H','legacy','hsc')]:
        contrast(f'DoRA_minus_QLoRA_{direction}_penalty',{f'{source}-dora/{target}':1,f'{source}-dora/{source}':-1,f'{source}-qlora/{target}':-1,f'{source}-qlora/{source}':1})
        contrast(f'DoRA_minus_QLoRA_{direction}_target_native_gap',{f'{source}-dora/{target}':1,f'{target}-dora/{target}':-1,f'{source}-qlora/{target}':-1,f'{target}-qlora/{target}':1})
    np.savez_compressed(OUT/'paired_differences.npz',**diffs)
    result=dict(created_utc=now(),n=n,cells=cells,contrasts=contrasts,redshift_bins=binresults,
                weighting='Full cohort is the empirical native paired population; equal-bin macro gives each nonempty fixed bin 1/5. This is not reweighting to either complete parent survey.',
                bootstrap='1000 identical saved draws across all 8 cells and all contrasts; bin estimates condition each shared draw on fixed spec-z boundaries; percentile CI; conditional fixed-model uncertainty only',
                limitations=['Cross-domain Level-0 schema shift: interface-stress diagnostics, not clean filter-aware transfer.','Original source training cohorts differ in membership and size.','Sparse z>1 and no z>2: no strict high-z extrapolation claim.','No training-seed uncertainty measured.'])
    write(OUT/'results.json',result);plots(d,labels,objs,zphot,contrasts)
    lines=['# Original bidirectional matched transfer','',f'Completed eight cells on exactly **{n:,} P_mutual galaxies**, in prepared order. Four HSC-source cells reuse live-verified archives; four Legacy-source cells are newly inferred. All four selected adapter/head states passed exact source-validation replay.','',
        '**Cross-domain cells are interface-stress diagnostics.** Native Level-0 schemas and physical filter identities differ; actual destination-band fields are preserved, without positional substitution. Original source cohorts also differ, so adapter method and observation system are not the only historical differences.','',
        '| Model | Observations | CE | NMAD | Outliers | CRPS | Coverage 68% |','|---|---|---:|---:|---:|---:|---:|']
    for label in labels:
        r=cells[label]['metrics'];lines.append(f"| {label.split('/')[0]} | {label.split('/')[1]} | {r['ce']:.6f} | {r['nmad']:.5f} | {r['outlier']:.2%} | {r['crps']:.5f} | {r['coverage68']:.2%} |")
    lines+=['','## Transfer comparisons','','All CE values above are absolute probability-mass scores on the same 300-bin [0,4.5] grid. Penalties and target-native gaps below are paired; a smaller penalty alone does not establish a better target predictor.','','| Contrast | ΔCE | 95% paired CI |','|---|---:|---|']
    for label,c in contrasts.items():
        r=c['full']['ce'];lines.append(f"| {label} | {r['value']:.6f} | [{r['ci95'][0]:.6f}, {r['ci95'][1]:.6f}] |")
    lines+=['','## Redshift, calibration and uncertainty','',f'Fixed redshift boundaries: {BINS}. Bin counts: {[len(r) for r in brows]}. Full cohort scores use the empirical paired population. Equal-bin macro results, all per-bin metrics/intervals, PIT histograms, outlier transitions and paired object differences are in `results.json`, `bootstrap_metrics.npz`, `paired_differences.npz` and each cell’s `object_metrics.npz`.',
        '',result['bootstrap']+'. No training-seed uncertainty is estimated. Few objects exceed z=1 and none exceed z=2; these data cannot establish strict high-redshift extrapolation.',
        '','## Provenance and execution','', 'See `manifest.json`, `cohort_identity.json`, `input_compatibility.json`, `replays/`, and per-cell completion/reuse proofs. Native mass-grid CE, discrete CRPS/PIT, lower-median NMAD, inclusive bin-centre 68% intervals follow the existing evaluator. Mean and median bias are normalized residuals. Stage 2 uses the already-frozen protocol and is not tuned using these results.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    verify(m['protected_sha256']);write(OUT/'stage1_complete.json',dict(complete=True,finished_utc=now(),n=61310,cells=8,results_sha256=sha(OUT/'results.json'),report_sha256=sha(OUT/'report.md')))

def plots(d,labels,objs,zphot,contrasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(4,2,figsize=(10,16))
    for ax,label in zip(axes.flat,labels):
        ax.hexbin(d['z'],zphot[label],gridsize=90,mincnt=1,bins='log');ax.plot([0,4.5],[0,4.5],'k:',lw=.6);ax.set(title=label,xlabel='Spectroscopic z',ylabel='PDF median z',xlim=(0,2),ylim=(0,4.5))
    fig.tight_layout();fig.savefig(folder/'predicted_vs_spectroscopic.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(11,15))
    for j,name in enumerate(NAMES):
        for domain in ('hsc','legacy'):
            i=labels.index(name+'/'+domain);o=objs[i]
            axes[j,0].hist(o['delta'],np.linspace(-1,1,101),histtype='step',density=True,label=domain)
            axes[j,1].hist(o['pit'],np.linspace(0,1,21),histtype='step',density=True,label=domain)
        axes[j,0].set(title=name,xlabel='Normalized residual',ylabel='Density');axes[j,1].set(xlabel='Binned PIT',ylabel='Density');axes[j,1].axhline(1,color='k',lw=.6);axes[j,0].legend()
    fig.tight_layout();fig.savefig(folder/'residual_and_calibration.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for j,method in enumerate(('qlora','dora')):
        for k,direction in enumerate(('H_to_L','L_to_H')):
            c=contrasts[f'{method}_{direction}_penalty'];x=np.arange(5)+(j*2+k-1.5)*.05
            y=np.array([r['ce']['value'] for r in c['redshift_bins']]);ci=np.array([r['ce']['ci95'] for r in c['redshift_bins']])
            axes[0].errorbar(x,y,yerr=np.maximum(0,np.array([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=2,label=f'{method} {direction}')
            r=c['full']['ce'];axes[1].errorbar(j*2+k,r['value'],yerr=[[max(0,r['value']-r['ci95'][0])],[max(0,r['ci95'][1]-r['value'])]],fmt='o',capsize=3)
    axes[0].set(xticks=np.arange(5),xticklabels=[f'{a:g}–{b:g}' for a,b in zip(BINS[:-1],BINS[1:])],xlabel='Fixed spec-z bin',ylabel='Transfer penalty ΔCE');axes[0].legend(fontsize=8)
    axes[1].set(xticks=np.arange(4),xticklabels=['Q H→L','Q L→H','D H→L','D L→H'],ylabel='Full-cohort ΔCE')
    for ax in axes:ax.axhline(0,color='k',lw=.6)
    fig.tight_layout();fig.savefig(folder/'transfer_penalties.png',dpi=140);plt.close(fig)

def consolidated(final=False):
    compact=ROOT/'reports/filter-aware/1029/bidirectional-seed42';compact.mkdir(parents=True,exist_ok=True)
    lines=['# Bidirectional transfer and seed-42 execution','',f'Updated {now()}. Persistent execution root: `{OUT}`.','', '## A. Historical bidirectional transfer','']
    if (OUT/'results.json').exists():
        r=json.loads((OUT/'results.json').read_text());lines.append('Eight historical cells completed on 61,310 objects. Cross-domain cells are Level-0 interface-stress diagnostics. Full tables, shared paired confidence intervals, plots and redshift analyses: [historical report]('+str(OUT/'report.md')+').')
        for method in ('qlora','dora'):
            for direction in ('H_to_L','L_to_H'):
                c=r['contrasts'][f'{method}_{direction}_penalty']['full']['ce'];lines.append(f"\n{method} {direction} ΔCE = {c['value']:.6f}, 95% CI {c['ci95']}.")
    else:lines.append('Historical inference is running; training remains gated on its complete report.')
    lines+=['','## B. Seed-42 matched training','','All workers receive only 16,600 training and 4,257 validation objects from their own system. Locked P is excluded from the process. Original prepared configurations, canonical head/adapter initialisation, saved orders, optimiser schedule and fixed budgets are retained. A versioned worker fixes the prepared entrypoint’s full-cohort load and adds source-only diagnostics. Seeds 43/44 and new-model P scoring remain unauthorized.','','| Model | Status | Updates | Presentations | Best source-val CE |','|---|---|---:|---:|---:|']
    complete=[];assessment={}
    for method in ('dora','qlora'):
        for domain in ('hsc','legacy'):
            name=f'{domain}-{method}-seed42';cfg=json.loads((PKG/'training_configs'/f'{name}.json').read_text());run=Path(cfg['output_dir']);p=run/'complete.json'
            if p.exists():
                s=json.loads(p.read_text());assert s['global_update']==10380 and s['examples_seen']==166000 and len(s['history'])==10 and not s['test_evaluated']
                assert (run/'completion_checks.json').exists();checks=json.loads((run/'completion_checks.json').read_text());assert checks['passed']
                complete.append(name);assessment[name]=json.loads((run/'convergence.json').read_text());status='complete'
            elif (run/'progress.json').exists():s=json.loads((run/'progress.json').read_text());status='training'
            else:s={};status='gated / queued'
            lines.append(f"| {name} | {status} | {s.get('global_update',0)} | {s.get('examples_seen',0)} | {s.get('best_source_val_ce','—')} |")
            if p.exists():lines+=['',f"{name}: selected update {s['best_update']}; final update 10380. Source convergence flags: {assessment[name]['flags']}. Full histories, first-backward memory/throughput, gradient norms, parameter changes, selected/final source predictions, replay and optimizer/RNG states are retained in `{run}`."]
    lines+=['','## C. Next scientific gate','','Original cross-domain scores have source-specific interface and unequal-training-cohort limitations. Historical directional asymmetry and adapter contrasts have paired CIs, conditional on these fixed models. The fresh Level-1 experiment is required for a cleaner shared-interface comparison.']
    if len(complete)==4:
        adequate=all(not a['flags'] for a in assessment.values());lines+=['',f'All four fixed-budget runs completed. Automatic source-only adequacy screen: {adequate}. Flags are diagnostic rather than an authorization for protocol changes.', '', 'Review source-only ten-epoch convergence, selected/final differences and measured resource cost before approving seeds 43/44 or locked-P inference. No new-model P evaluation has occurred. Checkpoints and frozen future evaluation configurations are preserved.']
    else:lines+=['','Source convergence, adequacy of ten epochs, extension to seeds 43/44, and readiness of the selected/final pairs remain pending completion of all four runs. No new-model P scores are available.']
    if final:assert len(complete)==4
    value='\n'.join(lines)+'\n';(OUT/'consolidated_report.md').write_text(value);(compact/'report.md').write_text(value)
    write(compact/'status.json',dict(updated_utc=now(),execution_root=str(OUT),historical_complete=(OUT/'stage1_complete.json').exists(),training_complete=complete,new_model_P_evaluated=False,source_only_assessments=assessment))
