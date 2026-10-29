#!/usr/bin/env python3
"""Compare only existing MLP/Qwen predictions; CPU plotting, no inference/training.

Best means lowest saved validation CE among completed Qwen results with exactly
the same validation IDs, labels and PDF grid as an existing numeric MLP.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import gc
import hashlib
import json
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.text import Text

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from aion_magnitude.metrics import summarize_pdf_metrics

RUNS=Path('/arc/home/gsm/aion_output/figures')
WARM=Path('/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1023/hsc-warmstart')
MLPS={
    'hsc':RUNS/'glm52-capacity-galaxiesml-spec-1021/matched-numeric-mlp/result.pt',
    'clauds':RUNS/'glm52-capacity-1021/matched-numeric-mlp/result.pt',
}
TITLES={'hsc':'HSC–DESI','clauds':'CLAUDS'}
BLUE='#2567A4'
ORANGE='#CF6B25'
INK='#203447'

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
    'axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#B9C3CC',
    'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False})


def load(path):
    return torch.load(path,map_location='cpu',weights_only=False,mmap=True)


def arr(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def compatible(base,ev):
    if 'object_id' not in ev:return 'no saved object IDs'
    if list(base['object_id'])!=list(ev['object_id']):return 'different validation IDs/order'
    for key in ['z_spec','redshift_edges','redshift_centers']:
        if not torch.equal(base[key],ev[key]):return f'different {key}'
    return None


def inventory(baselines):
    rows=[];best={}
    # Only completed result artifacts are eligible. Checkpoints alone do not
    # count as completed evaluations. Different cohorts are never CE-ranked.
    candidates=[(p,None) for p in sorted(RUNS.glob('qwen*/*/result.pt'))
                if p.parent.name in ['frozen','qlora','dora','ia3','rlvr','embedding_adapter']]
    candidates += [(p,k) for p in sorted(RUNS.glob('qwen*/results.pt')) for k in ['frozen','qlora']]
    candidates += [(p,'qwen_morphology') for p in sorted(RUNS.glob('qwen*/qwen_mlp_full_results.pt'))]
    candidates += [(WARM/f'level{i}/best_validation.pt','direct') for i in range(3)]
    for path,key in candidates:
        if not path.is_file():continue
        try:
            raw=load(path)
            if key=='direct':result={'val_evaluation':raw}
            elif key is not None:
                if key not in raw:continue
                result=raw[key]
            else:result=raw
            if 'val_evaluation' not in result:continue
            ev=result['val_evaluation']
            match=None;reasons=[]
            for dataset,base in baselines.items():
                why=compatible(base['val_evaluation'],ev)
                if why is None:match=dataset;break
                reasons.append(f'{dataset}: {why}')
            if match is None:
                rows.append(dict(path=str(path),key=key,eligible=False,reason='; '.join(reasons)))
                continue
            # loss is already saved CE for the frozen selected checkpoint.
            ce=float(ev['loss'])
            label=(f'Qwen3.5-4B · warm-start Level {path.parent.name[-1]}'
                   if key=='direct' else f"{path.parents[1].name} / {key or path.parent.name}")
            rows.append(dict(path=str(path),key=key,dataset=match,eligible=True,
                             validation_ce=ce,label=label,n=len(ev['z_spec'])))
            if match not in best or ce<best[match]['ce']:
                best[match]=dict(path=path,key=key,ce=ce,label=label,
                    evaluation={k:v.clone() if torch.is_tensor(v) else list(v) if isinstance(v,list) else v for k,v in ev.items()},
                    metrics=result.get('final_metrics',{}).get('val'))
            del raw,result,ev
            gc.collect()
        except (OSError,RuntimeError,KeyError,ValueError) as exc:
            rows.append(dict(path=str(path),key=key,eligible=False,reason=str(exc)[:300]))
        print(f'Checked {path.parent.parent.name}/{path.parent.name}',flush=True)
    return best,rows


def save(fig,path):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    clipped=[]
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():continue
        b=text.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1>fig.bbox.width+1 or b.y1>fig.bbox.height+1:
            clipped.append(text.get_text())
    assert not clipped,clipped
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'),dpi=220)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def prediction_figure(data,out):
    fig,axs=plt.subplots(2,2,figsize=(14,12),gridspec_kw={'hspace':.43,'wspace':.28})
    fig.subplots_adjust(left=.075,right=.88,bottom=.105,top=.88)
    fig.suptitle('Existing MLP vs. best compatible Qwen: redshift predictions',fontsize=21,y=.976,weight='bold')
    fig.text(.5,.94,'Same validation objects within each dataset · Qwen selected by lowest saved validation CE',ha='center',fontsize=11,color='#536474')
    images=[]
    max_counts=1
    for row,dataset in enumerate(['hsc','clauds']):
        d=data[dataset];limit=2 if dataset=='hsc' else 6
        ref='DESI spectroscopic redshift' if dataset=='hsc' else 'Catalogue photometric redshift (ZPHOT)'
        for col,model in enumerate(['mlp','qwen']):
            ax=axs[row,col];ev=d[model];m=d['metrics'][model]
            z=arr(ev['z_spec']);pred=arr(ev['z_p50'])
            im=ax.hexbin(z,pred,gridsize=75,extent=(0,limit,0,limit),
                         mincnt=1,cmap='viridis',norm=LogNorm(vmin=1),linewidths=0,rasterized=True)
            max_counts=max(max_counts,float(im.get_array().max()));images.append(im)
            ax.plot([0,limit],[0,limit],color='#666666',lw=1,ls='--')
            ax.set(xlim=(0,limit),ylim=(0,limit),xlabel=ref,ylabel='Predicted redshift (PDF median)')
            name='Numeric MLP' if model=='mlp' else 'Qwen3.5-4B · '+('QLoRA + Level-0 warm-start' if dataset=='hsc' else 'QLoRA')
            ax.set_title(f'{TITLES[dataset]} · {name}',loc='left',fontsize=12,weight='bold',pad=10)
            ax.text(.035,.965,f'N = {len(z):,}\nCE = {m["cross_entropy"]:.4f}\nNMAD = {m["nmad"]:.4f}\nOutliers = {100*m["catastrophic_outlier_fraction"]:.2f}%',
                    transform=ax.transAxes,va='top',fontsize=10.5,
                    bbox={'facecolor':'white','alpha':.88,'edgecolor':'none','pad':5})
            ax.grid(alpha=.18)
    for im in images:im.set_clim(1,max_counts)
    cax=fig.add_axes([.909,.19,.013,.60])
    fig.colorbar(images[-1],cax=cax,label='Galaxies per hexagon',
                 ticks=[10**i for i in range(7) if 10**i<=max_counts])
    fig.text(.075,.056,'Validation/model-selection diagnostics, not independent test results. Existing training budgets differ; no new training or inference.',fontsize=10,color='#536474')
    fig.text(.075,.031,'Outliers: |(zpred − zref)/(1 + zref)| > 0.15. HSC axes show 0–2 (model grid: 0–4.5); all objects enter the metrics.',fontsize=10,color='#536474')
    save(fig,out/'mlp_best_qwen_predictions')


def tomo_data(data):
    result={}
    for dataset,d in data.items():
        grid=arr(d['mlp']['redshift_edges']).astype(float)
        # Coarser common histogram bins keep empirical distributions readable.
        hist=grid[::3]
        if hist[-1]!=grid[-1]:hist=np.append(hist,grid[-1])
        bounds=np.array([0.,.2,.4,.6,1.,4.5] if dataset=='hsc' else [0.,.5,1.,1.5,2.,3.,6.])
        entries={}
        for model in ['mlp','qwen']:
            ev=d[model];z=arr(ev['z_spec']);med=arr(ev['z_p50']);pz=arr(ev['pz'])
            for i,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:])):
                mask=(med>=lo)&((med<hi) if i<len(bounds)-2 else (med<=hi))
                n=int(mask.sum())
                posterior=np.zeros(len(hist)-1);reference=posterior.copy()
                if n:
                    mass=pz[mask].sum(axis=0,dtype=np.float64)/n
                    cdf=np.r_[0.,np.cumsum(mass)]
                    # Histogram edges coincide with the original PDF edges.
                    ix=np.searchsorted(grid,hist)
                    posterior=np.diff(cdf[ix])/np.diff(hist)
                    reference=np.histogram(z[mask],bins=hist)[0]/(n*np.diff(hist))
                    assert abs(np.sum(posterior*np.diff(hist))-1)<2e-6
                    assert abs(np.sum(reference*np.diff(hist))-1)<2e-6
                entries[f'{model}_{i}']=dict(n=n,posterior=posterior,reference=reference)
            entries[f'{model}_population']=dict(n=len(z),posterior=np.zeros(len(hist)-1),reference=np.zeros(len(hist)-1))
        assert sum(entries[f'mlp_{i}']['n'] for i in range(len(bounds)-1))==len(z)
        assert sum(entries[f'qwen_{i}']['n'] for i in range(len(bounds)-1))==len(z)
        result[dataset]=dict(edges=hist,bounds=bounds,entries=entries)
    return result


def tomography_figure(tomo,out):
    fig,axs=plt.subplots(2,6,figsize=(19,9),gridspec_kw={'hspace':.56,'wspace':.28})
    fig.subplots_adjust(left=.045,right=.985,bottom=.14,top=.80)
    fig.suptitle('Tomographic n(z): existing MLP vs. best compatible Qwen',fontsize=21,y=.974,weight='bold')
    fig.text(.5,.929,'Galaxies assigned to bins using each model’s predicted PDF median; reference and stacked PDF use the same selected galaxies.',ha='center',fontsize=11)
    handles=[Line2D([],[],color=BLUE,lw=2,label='MLP: stacked PDF'),Line2D([],[],color=BLUE,lw=1.5,ls='--',label='MLP: reference histogram'),
             Line2D([],[],color=ORANGE,lw=2,label='Qwen: stacked PDF'),Line2D([],[],color=ORANGE,lw=1.5,ls='--',label='Qwen: reference histogram')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.905),ncol=4,frameon=False,fontsize=11)
    for row,dataset in enumerate(['hsc','clauds']):
        d=tomo[dataset];bounds=d['bounds'];edges=d['edges'];nb=len(bounds)-1
        fig.text(.045,.835 if row==0 else .444,
                 f'{TITLES[dataset]} · '+('5,683 validation galaxies · DESI spec-z reference' if dataset=='hsc' else '15,000 validation galaxies · catalogue ZPHOT reference'),fontsize=13,weight='bold')
        for i,ax in enumerate(axs[row]):
            if i>=nb:
                ax.axis('off');ax.text(.06,.74,'Within each bin:',transform=ax.transAxes,fontsize=11,weight='bold')
                ax.text(.06,.24,'Each model has its own\nbin membership.\n\nSolid and dashed curves\nshare those exact objects.',transform=ax.transAxes,fontsize=10,linespacing=1.5)
                continue
            lo,hi=bounds[i:i+2]
            ax.set_title(f'{lo:g} ≤ zmed '+('≤' if i==nb-1 else '<')+f' {hi:g}',fontsize=11,pad=8)
            for model,col in [('mlp',BLUE),('qwen',ORANGE)]:
                e=d['entries'][f'{model}_{i}']
                if e['n']:
                    ax.stairs(e['posterior'],edges,color=col,lw=1.65)
                    ax.stairs(e['reference'],edges,color=col,lw=1.15,ls='--',alpha=.85)
            n1=d['entries'][f'mlp_{i}']['n'];n2=d['entries'][f'qwen_{i}']['n']
            ax.text(.96,.96,f'MLP: {n1:,}\nQwen: {n2:,}',ha='right',va='top',transform=ax.transAxes,fontsize=8.5,
                    bbox={'facecolor':'white','alpha':.78,'edgecolor':'none','pad':2})
            ax.set(xlim=(0,2 if dataset=='hsc' else 6),ylim=(0,None),xlabel='Redshift')
            if i==0:ax.set_ylabel('Normalized density')
            ax.tick_params(labelsize=9);ax.grid(alpha=.16)
    fig.text(.045,.078,'Curves are normalized separately within each model/bin. Stacked PDFs are exact sums of saved probabilities; no Monte Carlo sampling or smoothing.',fontsize=10,color='#536474')
    fig.text(.045,.048,'Validation/model-selection diagnostics; training budgets differ. HSC curves are shown at 0–2 without renormalizing the zoom (full PDF support: 0–4.5).',fontsize=10,color='#536474')
    save(fig,out/'mlp_best_qwen_nztomo')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=REPO/'reports/mlp-best-qwen-20261029')
    args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    baseline={d:load(p) for d,p in MLPS.items()}
    selected,audit=inventory(baseline)
    assert set(selected)==set(MLPS)
    data={};metrics_rows=[]
    for dataset in MLPS:
        m=baseline[dataset]['val_evaluation'];q=selected[dataset]['evaluation']
        assert compatible(m,q) is None
        data[dataset]={'mlp':m,'qwen':q,'metrics':{}}
        for model,ev in [('mlp',m),('qwen',q)]:
            metrics=summarize_pdf_metrics(ev)
            assert abs(metrics['cross_entropy']-float(ev['loss']))<2e-5
            data[dataset]['metrics'][model]=metrics
            metrics_rows.append(dict(dataset=dataset,model=model,n=len(ev['z_spec']),**metrics))
        assert abs(data[dataset]['metrics']['qwen']['cross_entropy']-min(r['validation_ce'] for r in audit if r.get('dataset')==dataset))<2e-5
    prediction_figure(data,out)
    tomo=tomo_data(data);tomography_figure(tomo,out)
    arrays={}
    for ds,d in tomo.items():
        arrays[ds+'_histogram_edges']=d['edges'];arrays[ds+'_tomographic_bounds']=d['bounds']
        for k,v in d['entries'].items():
            if k.endswith('population'):continue
            for name,value in v.items():arrays[f'{ds}_{k}_{name}']=value
    np.savez_compressed(out/'tomographic_curves.npz',**arrays)
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics_rows[0]));w.writeheader();w.writerows(metrics_rows)
    (out/'candidate_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    provenance={}
    for ds,p in MLPS.items():provenance[str(p)]=sha(p)
    for d in selected.values():provenance[str(d['path'])]=sha(d['path'])
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),selection='Lowest saved validation CE within exactly matched validation IDs, order, labels and PDF grid.',
        scope='Completed compatible HSC and CLAUDS predictions. No completed survey-specific Legacy MLP result was found; Legacy is not fabricated or substituted with frozen Qwen.',
        split='validation; also used for checkpoint selection, not independent test evidence',
        selected={ds:dict(path=str(d['path']),key=d['key'],label=d['label'],validation_ce=d['ce']) for ds,d in selected.items()},
        mlp_sources={ds:str(p) for ds,p in MLPS.items()},source_sha256=provenance,
        cohort_checks='Exact validation object IDs/order, target redshifts and PDF grids agree between both models within each dataset.',
        target_definitions={'hsc':'DESI spectroscopic redshift','clauds':'Catalogue photometric redshift ZPHOT; not spectroscopy'},
        tomography='Bins use each model median; dashed reference histogram and solid stacked probabilities use exactly the same selected galaxies per model. Histograms use every third saved PDF edge. Exact probability sums, no sampling/smoothing.',
        training='Only existing trained models and saved evaluations; no new fitting, inference or GPU use. Training budgets differ.',
        metrics=metrics_rows,visual_inspection='pending')
    (out/'provenance.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'README.md').write_text('''# Existing MLP vs. best compatible Qwen (2026-10-29)

`mlp_best_qwen_predictions.pdf/png/svg` and `mlp_best_qwen_nztomo.pdf/png/svg` compare the same validation objects in HSC–DESI (5,683) and CLAUDS (15,000). They are checkpoint-selection diagnostics, not held-out test confirmation. Existing MLP and Qwen training budgets differ.

Best Qwen is defined by lowest saved validation CE among completed evaluations with exactly matching object IDs, order, reference redshifts and PDF grid. HSC selects the Qwen3.5-4B Level-0 continued-QLoRA checkpoint at update 2,500. CLAUDS selects Qwen3.5-4B QLoRA. All completed eligible candidates and exclusions are recorded in `candidate_audit.json`. Recent matched-transfer training uses different train/validation cohorts and cannot be ranked on the MLP's saved validation cohort; inference-time corrupted-metadata conditions are not additional trained models. Unfinished 0.8B CLAUDS adapter jobs are not completed candidates. No complete Legacy-specific numeric MLP evaluation was located, so there is no Legacy panel.

HSC uses native photometric fluxes/errors and EBV with DESI spectroscopy. CLAUDS uses 11 catalogue magnitudes and catalogue ZPHOT references (not spec-z). Saved predictions and targets are unchanged. Both MLPs are the existing matched-numeric-MLP baselines from the capacity studies; these are ordinary numeric MLP models, despite the parent GLM experiment name. Matching IDs, labels and grids are asserted. Input representation/training provenance is in the source run metadata; this is a descriptive comparison, not a matched-budget architecture intervention.

Tomography selects objects by each model's PDF median. For each model/bin, solid stacked PDF density and dashed reference histogram contain exactly the same selected objects. Both densities integrate to one on the full model grid. Thus MLP/Qwen bin memberships can differ; the panel reports both counts. This avoids comparing different reference and predicted selections under the same model. Exact sums of saved PDFs avoid Monte Carlo noise. Every third PDF edge defines the coarser display histogram. For CLAUDS, dashed lines show the empirical catalogue ZPHOT distribution; they are not spectroscopic truth or reconstructed reference-PDF stacks.

HSC axes zoom to 0–2; all metrics and normalization use the full 0–4.5 grid. CLAUDS shows 0–6. Point metrics use the saved PDF median, the repository's lower-median NMAD convention, and outlier threshold |(zpred−zref)/(1+zref)| > 0.15.

`metrics.csv` contains plot metrics. `tomographic_curves.npz` preserves all binned curves and counts. `provenance.json` contains exact selected source paths and SHA256 hashes. Rebuild using the existing project environment:

```bash
/scratch/RedFM_MLP/.pixi/envs/default/bin/python reports/plot_mlp_best_qwen.py
```

The script uses CPU-only saved-artifact analysis and plotting; it never runs inference or training.
''')
    print(json.dumps(summary['selected'],indent=2),flush=True)
    print(out,flush=True)


if __name__=='__main__':main()
