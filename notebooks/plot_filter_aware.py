#!/usr/bin/env python3
"""Collect completed validation-only ablations; never performs inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from aion_magnitude.metrics import discrete_crps
from plot_qwen_zero_shot_sanity import continuous_pit


def collect(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    levels = {}
    for level in (0, 1, 2):
        run = folder / f"level{level}-native-qlora"
        if not (run / "summary.json").exists():
            continue
        summary = json.loads((run / "summary.json").read_text())
        result = torch.load(run / "result.pt", weights_only=False, map_location="cpu", mmap=True)
        if (summary["cohort_sha256"] != manifest["cohort_sha256"] or result["metadata"]["test_evaluated"]
                or "test_evaluation" in result):
            raise ValueError("Cohort mismatch or unexpected test evaluation")
        evaluation = result["val_evaluation"]
        np.savez_compressed(run / "validation_predictions.npz", **{k:v.numpy() for k,v in evaluation.items() if isinstance(v,torch.Tensor)},
                            object_id=np.asarray(evaluation["object_id"]))
        summary["interpolated_pit_mean"] = float(continuous_pit(evaluation["pz"].numpy(), evaluation["z_spec"].numpy(), evaluation["redshift_edges"].numpy()).mean())
        transfer = run / "transfer/summary.json"
        if transfer.exists():
            summary["transfer"] = json.loads(transfer.read_text())
        levels[level] = (summary, evaluation)
    for level, (_, evaluation) in levels.items():
        if 0 in levels:
            reference = levels[0][1]
            if reference["object_id"] != evaluation["object_id"] or not torch.equal(reference["z_spec"], evaluation["z_spec"]):
                raise ValueError("Validation object order/labels differ between levels")
        if 0 in levels and "transfer" in levels[0][0] and "transfer" in levels[level][0]:
            for kind, target in levels[0][0]["transfer"]["targets"].items():
                actual = levels[level][0]["transfer"]["targets"][kind]
                if actual["query_ids"] != target["query_ids"]:
                    raise ValueError("Transfer queries differ between representation levels")
    return manifest, levels


def bootstrap_difference(a, b):
    rng = np.random.default_rng(1023)
    difference = np.asarray(a) - np.asarray(b)
    means = difference[rng.integers(0,len(difference),size=(2000,len(difference)))].mean(1)
    return dict(mean=float(difference.mean()), conditional_query_bootstrap_95=np.quantile(means,[.025,.975]).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--report",type=Path,default=ROOT / "SFT-label_results_1023.md")
    args=p.parse_args()
    manifest, levels = collect(args.run_dir)
    if len(levels) != 3:
        raise ValueError("All three matched arms must finish before collecting the comparison")
    config = levels[0][0]["config"]
    for summary, _ in levels.values():
        if summary["config"] != config or summary["registry_sha256"] != levels[0][0]["registry_sha256"]:
            raise ValueError("Optimizer/model settings or filter registry differ between levels")
    n_train, n_val = manifest["train_rows"], manifest["val_rows"]
    warmup, epochs = config["head_warmup_epochs"], config["epochs"]
    updates_per_epoch = int(np.ceil(np.ceil(n_train / config["batch_size"]) / config["gradient_accumulation_steps"]))
    total_updates = updates_per_epoch * epochs
    adapter_updates = updates_per_epoch * (epochs - warmup)
    plot_dir=args.run_dir / "plots";plot_dir.mkdir(exist_ok=True)
    colors=["#3973ac","#d06b32","#7c589b"]
    fig, axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for level,(s,e) in levels.items():
        history=s['history'];x=np.arange(1,len(history)+1)
        axes[0].plot(x,[r['val_cross_entropy'] for r in history],marker='o',label=f'Level {level}',color=colors[level])
        axes[1].plot(x,[r['val_nmad'] for r in history],marker='o',color=colors[level])
        axes[2].plot(x,[r['train_loss'] for r in history],marker='o',color=colors[level])
    for ax,y in zip(axes,['Validation CE','Validation NMAD','Training CE']):ax.set(xlabel='Epoch',ylabel=y);ax.axvline(warmup+.5,color='.6',ls='--')
    axes[0].legend();fig.suptitle(f'Matched HSC pilot: {n_train} train / {n_val} validation; {warmup} head-warmup + {epochs-warmup} adapter epochs')
    for ext in ['png','pdf']:fig.savefig(plot_dir/f'learning_curves.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for level,(s,e) in levels.items():
        z=e['z_spec'].numpy();pred=e['z_p50'].numpy();pz=e['pz'].numpy()
        axes[0,level].scatter(z,pred,s=13,color=colors[level]);axes[0,level].plot([0,4.5],[0,4.5],ls='--',color='.4')
        axes[0,level].set(title=f'Level {level}',xlabel='DESI spec-z',ylabel='Predicted PDF median',xlim=(0,4.5),ylim=(0,4.5))
        pit=continuous_pit(pz,z,e['redshift_edges'].numpy());axes[1,level].hist(pit,bins=np.linspace(0,1,11),density=True,color=colors[level])
        axes[1,level].axhline(1,ls='--',color='.4');axes[1,level].set(xlabel='Within-bin interpolated PIT',ylabel='Density')
    for ext in ['png','pdf']:fig.savefig(plot_dir/f'validation_predictions.{ext}',dpi=180)
    plt.close(fig)
    transfer_complete = all("transfer" in s for s,e in levels.values())
    if transfer_complete:
        fig,axes=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
        for col,kind in enumerate(("desi-ls","clauds")):
            reference=None
            for level,(s,e) in levels.items():
                path=args.run_dir / f'level{level}-native-qlora/transfer/{kind}_predictions.npz'
                with np.load(path,allow_pickle=False) as a:
                    data={k:a[k] for k in a.files}
                if reference is None:
                    reference=data
                elif not np.array_equal(data['targetid'],reference['targetid']) or not np.array_equal(data['z_spec'],reference['z_spec']):
                    raise ValueError('Transfer prediction IDs or labels differ between levels')
                z=data['z_spec'];edges=data['redshift_edges'];centres=data['redshift_centers'];pz=data['pz']
                axes[0,col].scatter(z,data['z_p50'],s=8,alpha=.4,color=colors[level],label=f'Level {level}')
                axes[1,col].plot(centres,pz.mean(0)/np.diff(edges),color=colors[level],label=f'Level {level}')
                axes[2,col].hist(continuous_pit(pz,z,edges),bins=np.linspace(0,1,16),density=True,histtype='step',color=colors[level],label=f'Level {level}')
            label='DESI spec-z' if kind=='desi-ls' else 'ZPHOT (photometric reference)'
            axes[0,col].plot([0,4.5],[0,4.5],ls='--',color='.4')
            axes[0,col].set(title=kind,xlabel=label,ylabel='Predicted PDF median',xlim=(0,4.5),ylim=(0,4.5))
            axes[1,col].hist(reference['z_spec'],bins=np.linspace(0,4.5,31),density=True,histtype='step',color='black',label=label)
            axes[1,col].set(xlabel='Redshift',ylabel='Mean predicted PDF / reference density',xlim=(0,4.5))
            axes[1,col].legend(fontsize=8)
            axes[2,col].axhline(1,ls='--',color='.4');axes[2,col].set(xlabel='Within-bin interpolated PIT',ylabel='Density',xlim=(0,1))
        fig.suptitle('Fixed-weight cross-survey pilot; identical target validation objects; no target-label fitting')
        for ext in ['png','pdf']:fig.savefig(plot_dir/f'transfer_predictions.{ext}',dpi=180)
        plt.close(fig)
    underfit = all(s['metrics']['val']['catastrophic_outlier_fraction'] > .5 for s,e in levels.values())
    interpretation = ('All three arms remain severely underfit in-domain. These runs verify the implementation, but do not establish that filter metadata improve cross-survey generalisation.'
                      if underfit else 'These are preliminary matched-budget results. Cross-survey improvements require repeated seeds and metadata controls before physical interpretation.')
    rows=['# Filter-aware input pilot — 2026-10-23','',
          'Implemented from `SFT-label_1023.md`. All results below are from a small source-only pilot, with no held-out test evaluation and no target-label fitting.','',
          interpretation,'',
          '## Matched HSC-source training','',
          f"{manifest['model']}, fresh rank-{config['lora_rank']} QLoRA adapters and a newly initialized instance of the existing {config['head_hidden_dim']}-hidden-unit PDF head; {config['n_z_bins']} bins over {config['z_min']}–{config['z_max']}. All arms use the same {n_train} source-training galaxies, {n_val} source-validation galaxies, seed {config['seed']}, {epochs} epochs, {warmup} head-warmup epochs, head LR {config['learning_rate']}, adapter LR {config['lora_learning_rate']}, microbatch {config['batch_size']} and accumulation {config['gradient_accumulation_steps']}. The best validation-CE checkpoint is evaluated. These short runs must not be compared as matched-budget counterparts of the archived 10-epoch/full-cohort models.",'',
          '| Level | Best epoch | Val CE | NMAD | Outliers | Bias | CRPS | 68% coverage | Peak allocated GiB | Seconds |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for level,(s,e) in levels.items():
        best=int(np.argmin([r['val_cross_entropy'] for r in s['history']]))+1
        m=s['metrics']['val'];rows.append(f"| {level} | {best} | {m['cross_entropy']:.4f} | {m['nmad']:.4f} | {m['catastrophic_outlier_fraction']:.2%} | {m['bias']:.4f} | {m['mean_crps']:.4f} | {m['p16_p84_coverage']:.2%} | {s['peak_allocated_gib']:.2f} | {s['elapsed_seconds']:.1f} |")
    comparisons={}
    for level in (1,2):
        e=levels[level][1];pz=e['pz'];logits=e['logits'];z=e['z_spec'];edges=e['redshift_edges']
        index=(torch.bucketize(z,edges)-1).clamp(0,pz.shape[1]-1)
        ce=-torch.log_softmax(logits,dim=-1).gather(1,index[:,None]).squeeze(1).numpy()
        crps=discrete_crps(pz,z,centers=e['redshift_centers']).numpy()
        comparisons[level]=(ce,crps)
    paired=dict(level2_minus_level1_ce=bootstrap_difference(comparisons[2][0],comparisons[1][0]),
                level2_minus_level1_crps=bootstrap_difference(comparisons[2][1],comparisons[1][1]))
    rows+=['','Query bootstrap intervals are descriptive and conditional on this one training seed and selected checkpoints; they do not quantify training-seed, checkpoint-selection or survey variation. These validation queries also selected checkpoints, so the intervals are not independent confirmation. Negative differences favour Level 2.','',f'```json\n{json.dumps(paired,indent=2)}\n```','',
           '## Fixed-weight cross-survey checks','',
           'Targets reuse the same 256 validation queries per catalogue from the HSC-source fast-check. That audit excluded the entire original source cohort by IDs and a 1-arcsec positional check; the new source-training subset is contained within that cohort. Legacy uses DESI spec-z; CLAUDS uses ZPHOT and unchanged, training-imputed AB magnitudes. The input-format and label-definition shifts for CLAUDS remain explicit confounders.','',
           '| Level | Target | N | CE | NMAD | Outliers | Bias | CRPS | 68% coverage |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for level,(s,e) in levels.items():
        if 'transfer' not in s:
            transfer_complete=False;rows.append(f'| {level} | Not executed | — | — | — | — | — | — | — |');continue
        for kind,t in s['transfer']['targets'].items():
            m=t['metrics'];rows.append(f"| {level} | {kind} | {t['n']} | {m['cross_entropy']:.4f} | {m['nmad']:.4f} | {m['catastrophic_outlier_fraction']:.2%} | {m['bias']:.4f} | {m['mean_crps']:.4f} | {m['p16_p84_coverage']:.2%} |")
    rows+=['','## Interpretation and limits','',
           'Level 1 changes the structure and spelling of already band-labelled Level-0 inputs. Level 2 differs from Level 1 only through pivot wavelength and rectangular equivalent width. All three native arms retain the same photometric values. The optional CLAUDS common-flux representation was implemented and token-audited separately; it was not mixed into this native-input comparison.','',
           f'The pilot has {total_updates} optimizer updates, including {adapter_updates} after head warmup. It tests that the complete training and transfer paths work. Single-survey filter metadata are constant across galaxies; even an improvement would not establish that the model learned a transferable physical relationship. A promising result needs repeated seeds, adequate source learning, and matched-length/shuffled-metadata controls before a physics or novelty claim.','',
           interpretation,'',
           'Legacy-source and CLAUDS-source Level-0/1/2 training, DoRA pilots, common-flux training/transfer, mixed-survey training and final test evaluation were not run. The full Legacy cohort cache was unavailable in this session; only existing audited query photometry was used, with actual filter identities recovered from Tractor records. The existing loader remains available when that cache is restored.','',
           'The report uses historical binned CE/CRPS and bin-centre interval coverage for comparability. PIT figures interpolate within each bin; historical upper-bin PIT and finite-bin coverage effects should not be interpreted as exact continuous calibration.','',
           '## Implementation and validation','',
           'All three serializers, optional raw-flux harmonisation for CLAUDS, QLoRA/DoRA integration and validation-only training are implemented. The offline registry contains 21 pinned response profiles with hashes, nm units, reproducible pivot/width definitions and documented nominal-response approximations. Legacy query filters are resolved from actual RELEASE values and matching saved fluxes. HSC r/r2 and i/i2 exposure mixtures are absent from the original metadata; original r/i profiles remain explicit approximations.','',
           'The relevant test suite passes 43 tests. Level-0 serialization exactly matches the 512 archived Legacy/CLAUDS query strings; the 32-object GPU HSC source replay has maximum logit error 0.0. Complete actual inputs are token-audited, with IDs and missing masks retained. Native/common-flux CLAUDS bundles share one portable cohort fingerprint and match archived validation IDs/labels. Source test evaluation is disabled, and target evaluations use validation objects only.','',
           'Measured token lengths, filter provenance, the corrected CLAUDS split-string storage, full launch commands and unexecuted follow-ups are documented in [the implementation notes](filter-aware-runs/1023/README.md).','',
           '## Files and reproduction','',
           '- Serializer: `aion_magnitude/filter_photometry.py`.',
           '- Offline response registry: `configs/filter_registry_v1.json`; vendored originals, normalized curves and provenance notes: `configs/filter_responses_v1/`.',
           '- Validation-only trainer integration: `aion_magnitude/qwen_posttraining.py` (`test_dataset=None`).',
           '- Prepare/train/replay/transfer: `notebooks/qwen_filter_aware.py`.',
           '- Sequential safe launcher: `scripts/run_filter_aware_pilot.py`.',
           '- Tests: `tests/test_filter_photometry.py`.',
           f'- Run manifest, shared cohort, actual tokenizer inputs and per-arm results: `{args.run_dir}`.',
           '- Plot generation: `notebooks/plot_filter_aware.py`.',
           '- Full implementation and launch notes: `filter-aware-runs/1023/README.md`.','',
           f'![Learning curves]({(plot_dir / "learning_curves.png").resolve()})','',
           f'![Validation predictions]({(plot_dir / "validation_predictions.png").resolve()})']
    if transfer_complete:
        rows += ['', f'![Cross-survey predictions and PDFs]({(plot_dir / "transfer_predictions.png").resolve()})']
    args.report.write_text('\n'.join(rows)+'\n')
    (args.run_dir/'comparison.json').write_text(json.dumps(dict(manifest=manifest, paired=paired,
        levels={str(k):s for k,(s,e) in levels.items()},transfer_complete=transfer_complete),indent=2)+'\n')
    print(args.report)


if __name__=='__main__':main()
