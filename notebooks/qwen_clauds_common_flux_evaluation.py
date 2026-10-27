#!/usr/bin/env python3
"""Fixed-weight A/B/C CLAUDS interface diagnostic; no training entry point."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np
import torch
from transformers import AutoTokenizer

from aion_magnitude.clauds_bands import ALL_BAND_FLUX_COLUMNS, ALL_BAND_ERROR_COLUMNS
from aion_magnitude.clauds_common_flux import serialize_clauds_common_flux
from aion_magnitude.filter_photometry import (
    CLAUDS_BANDS, DEFAULT_REGISTRY, audit_tokens, clauds_flux_to_nanomaggy,
    load_registry, magnitude_to_nanomaggy, serialize_rows,
)
from aion_magnitude.FM_Qwen import load_frozen_qwen, resolve_qwen_model_path
from aion_magnitude.metrics import summarize_pdf_metrics
from aion_magnitude.qwen_continued_sft import verify_replay, write_json
from aion_magnitude.qwen_posttraining import QwenPhotoZModel, qwen_hidden_size
import qwen_zero_shot_sanity as pilot

BASE = ROOT / "filter-aware-runs/1023/hsc-warmstart"
FAST = ROOT / "fast-check-runs/1023/hsc-qlora-to-legacy-clauds"
CATALOGUE = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological_multiband_updated.fits")
PAPER = "https://arxiv.org/pdf/2301.13750"
CONDITIONS = ("A_native_full", "B_native_full", "B_flux_full", "B_flux_hsc",
              "C_native_full", "C_flux_full", "C_flux_hsc")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as a:
        return {k:a[k] for k in a.files}


def evaluation_from_archive(path):
    a = read_npz(path)
    e = {k:torch.from_numpy(a[k]) for k in ("logits", "pz", "z_spec", "z_p16", "z_p50", "z_p84", "redshift_edges", "redshift_centers")}
    e["object_id"] = [str(x) for x in a["targetid"]]
    return e


def check_protected(out):
    m = json.loads((out / "manifest.json").read_text())
    for path, digest in m["protected_sha256"].items():
        if sha(path) != digest:
            raise ValueError(f"Protected model or archived result changed: {path}")
    for file in m["base_model_source"]["files"]:
        s = Path(file["path"]).stat()
        if s.st_size != file["bytes"] or s.st_mtime_ns != file["mtime_ns"]:
            raise ValueError(f"Base model fingerprint changed: {file['path']}")


def prepare(args):
    import fitsio
    from aion_magnitude.dataset import make_random_split

    out = args.output_dir
    if out.exists():
        raise FileExistsError("Use a new output directory; existing results are immutable")
    fast = json.loads((FAST / "summary.json").read_text())
    archived = read_npz(FAST / "clauds_predictions.npz")
    data = dict(targetid=archived["targetid"], features=archived["features"], z=archived["z_spec"])
    meta = fast["targets"]["clauds"]
    ids = [str(x) for x in data["targetid"]]
    if len(ids) != 256 or len(set(ids)) != 256 or meta["split"] != "validation":
        raise ValueError("Exactly the original 256 unique validation queries are required")
    original = json.loads((FAST / "clauds_inputs.json").read_text())
    assert original["object_id"] == ids
    splits = make_random_split(300000, train_fraction=.2, test_fraction=.75, val_fraction=.05, seed=42)
    assert np.all(splits[np.asarray(meta["selected_indices"]) ] == "val")
    registry = load_registry(verify_curves=True)
    source = json.loads((BASE / "level1/manifest.json").read_text())
    config = source["source_config"]
    assert (config["n_z_bins"], config["z_min"], config["z_max"]) == (300, 0., 4.5)
    tokenizer = AutoTokenizer.from_pretrained(resolve_qwen_model_path(config["model_path"]), local_files_only=True)
    # Read IDs only for row lookup. Labels are read only for the 256 audited rows.
    with fitsio.FITS(str(CATALOGUE)) as f:
        all_ids = f[1].read(columns=["ID"])["ID"]
        order = np.argsort(all_ids)
        rows = order[np.searchsorted(all_ids[order], data["targetid"])]
        assert np.array_equal(all_ids[rows], data["targetid"])
        columns = ["ID", "ZPHOT"] + list(ALL_BAND_FLUX_COLUMNS.values()) + list(ALL_BAND_ERROR_COLUMNS.values())
        for b in CLAUDS_BANDS:
            suffix = ALL_BAND_FLUX_COLUMNS[b].removeprefix("FLUX_CMODEL_")
            columns += [prefix + "_" + suffix for prefix in ("notObserved", "isNoData", "hasBadPhotometry", "isClean")]
        raw = f[1].read(rows=rows, columns=columns)
        header = str(f[1].read_header())
    assert np.array_equal(raw["ID"], data["targetid"])
    assert np.array_equal(raw["ZPHOT"].astype(np.float32), data["z"])
    raw_flux = np.column_stack([raw[ALL_BAND_FLUX_COLUMNS[b]] for b in CLAUDS_BANDS])
    raw_sigma = np.column_stack([raw[ALL_BAND_ERROR_COLUMNS[b]] for b in CLAUDS_BANDS])
    flags = {}
    for prefix in ("notObserved", "isNoData", "hasBadPhotometry", "isClean"):
        flags[prefix] = np.column_stack([raw[prefix + "_" + ALL_BAND_FLUX_COLUMNS[b].removeprefix("FLUX_CMODEL_")] for b in CLAUDS_BANDS])
    flux, sigma, valid = clauds_flux_to_nanomaggy(raw_flux, raw_sigma,
                         not_observed=flags["notObserved"], no_data=flags["isNoData"])
    positive = np.isfinite(raw_flux) & (raw_flux > 0)
    raw_mag = np.full_like(raw_flux, np.nan)
    raw_mag[positive] = 23.9 - 2.5*np.log10(raw_flux[positive])
    expected_f, _ = magnitude_to_nanomaggy(raw_mag)
    np.testing.assert_allclose(flux[positive & valid], expected_f[positive & valid], rtol=1e-12)
    # Demonstrate the inherited native ZP=23 offset without changing native text.
    historical_mag = 23.0 - 2.5*np.log10(raw_flux[positive])
    np.testing.assert_allclose(data["features"][positive], historical_mag, atol=3e-6, rtol=0)
    missing_magnitude = ~positive
    if np.any(np.isfinite(sigma) & (sigma < .5e-5)):
        raise ValueError("An actual uncertainty would round to zero; review numerical precision before inference")
    texts = {}
    protected = {}
    selected = {}
    for arm, level in (("A",0),("B",1),("C",2)):
        folder = BASE / f"level{level}"
        summary = json.loads((folder / "summary.json").read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        assert summary["phase"] == "complete"
        assert manifest["source_config"] == config
        assert manifest["query_manifest_sha256"] == sha(FAST / "summary.json")
        texts[f"{arm}_native_full"] = serialize_rows(data, meta, catalogue="clauds", level=level, registry=registry)
        if level == 0:
            assert texts[f"{arm}_native_full"] == original["texts"]
        else:
            old = [json.loads(line) for line in (folder / "clauds_inputs.jsonl").read_text().splitlines()]
            assert [p["object_id"] for p in old] == ids
            assert [p["text"] for p in old] == texts[f"{arm}_native_full"]
            for bands in ("full", "hsc"):
                texts[f"{arm}_flux_{bands}"] = [serialize_clauds_common_flux(
                    dict(zip(meta["feature_names"], row)), flux[i], sigma[i], level=level,
                    band_configuration=bands, registry=registry) for i,row in enumerate(data["features"])]
        selected[arm] = dict(directory=str(folder/"selected"), update=summary["best_update"], source_manifest_sha256=sha(folder/"manifest.json"))
        for file in list((folder/"selected").rglob("*")) + [folder/"summary.json", folder/"manifest.json", folder/"clauds_selected_predictions.npz"]:
            if file.is_file(): protected[str(file)] = sha(file)
    # The rule is fixed by token completeness, before any new predictions exist.
    raw_lengths = {name:[len(x) for x in tokenizer(t, truncation=False, padding=False)["input_ids"]] for name,t in texts.items()}
    max_length = max(512, max(max(x) for x in raw_lengths.values()))
    token_audits = {name:audit_tokens(tokenizer,t,max_length) for name,t in texts.items()}
    out.mkdir(parents=True)
    (out/"inputs").mkdir()
    np.savez_compressed(out/"cohort.npz", **data, source_row=rows, raw_flux_microjy=raw_flux,
        raw_sigma_microjy=raw_sigma, flux_nanomaggy=flux, sigma_nanomaggy=sigma, valid=valid,
        original_magnitude_missing=missing_magnitude, **flags)
    (out/"catalogue_header.txt").write_text(header)
    examples = list(dict.fromkeys([0, int(np.argmax(missing_magnitude.sum(axis=1))), int(np.argmax((~valid).sum(axis=1)))]))
    for name,t in texts.items():
        with (out/"inputs"/f"{name}.jsonl").open("w") as f:
            for ident,text in zip(ids,t,strict=True): f.write(json.dumps(dict(object_id=ident,text=text))+"\n")
        write_json(out/"inputs"/f"{name}_examples.json", [dict(index=i,object_id=ids[i],text=t[i],
            input_ids=tokenizer(t[i],truncation=False)["input_ids"],original_magnitude_missing=missing_magnitude[i].tolist(),
            flux_valid=valid[i].tolist(),uncertainty_valid=np.isfinite(sigma[i]).tolist()) for i in examples])
    counts = {}
    for j,b in enumerate(CLAUDS_BANDS):
        counts[b] = dict(valid_flux=int(valid[:,j].sum()),missing_flux=int((~valid[:,j]).sum()),
            valid_sigma=int(np.isfinite(sigma[:,j]).sum()),original_imputed_magnitude=int(missing_magnitude[:,j].sum()),
            genuine_nonpositive_flux=int((valid[:,j] & (flux[:,j] <= 0)).sum()),
            flag_excluded_finite_flux=int((np.isfinite(raw_flux[:,j]) & (raw_flux[:,j] != -99) & ~valid[:,j]).sum()),
            bad_photometry_flag=int(flags['hasBadPhotometry'][:,j].sum()))
    write_json(out/"conversion_audit.json", dict(
        raw_unit="microJy in all bands, Desprez et al. 2023 Table E.2",source=PAPER,
        conversion="f_nMgy = f_microJy * 10**((22.5-23.9)/2.5)",scale=10**(-.56),
        magnitude_equivalent="m_AB=23.9-2.5*log10(f_microJy); f_nMgy=10**((22.5-m_AB)/2.5)",
        sigma="Measured FLUXERR scaled linearly; equivalent to magnitude-error Jacobian for positive measurements; no extra error floor or rescaling",
        flags="Missing iff non-finite/-99 flux OR notObserved OR isNoData; preserve signed flux otherwise. hasBadPhotometry/isClean saved, no new quality cut.",
        native_reference="Historical imputed m=23-2.5log10(f_microJy), 0.9 mag brighter than physical AB; unchanged byte for byte",
        previous_optional_flux_scale=10**(-.2),previous_flux_overestimate_factor=10**(.36),
        missingness="Raw fluxes replace imputed magnitude entries only when real measurements exist; no magnitude imputation enters flux inputs",
        changed_fields=["unit and sigma_unit AB_mag -> AB_nanomaggy", "mag -> flux", "measured raw flux uncertainties replace sigma=NA when available", "unavailable imputed magnitude entries -> flux=NA", "hsc configuration deletes six non-HSC band records only"],
        extinction="No extinction correction, EBV lookup, or photometric zero-point fitting applied. Target EBV stays NA. hscPipe/Phosphoros uses uncorrected fluxes (paper section 4.1); SExtractor extinction statement does not describe this catalogue. Source HSC magnitudes are used as catalogued with MW_EBV supplied. Exact local cmodel extinction history is not encoded in its FITS header.",
        near_infrared="VIRCAM raw cmodel fluxes are microJy under the same AB convention, not Vega magnitudes; no Vega offset applied",
        counts=counts,formula_verified=True,raw_ID_ZPHOT_matches=True,
        reference_supported_range_fraction=float(((data['z']>=0)&(data['z']<=4.5)).mean()),
    ))
    protected[str(FAST/"summary.json")] = sha(FAST/"summary.json")
    protected[str(FAST/"clauds_predictions.npz")] = sha(FAST/"clauds_predictions.npz")
    manifest = dict(created_utc=now(),purpose="Fixed-weight CLAUDS numerical-interface diagnostic",conditions=list(CONDITIONS),
        source_config=config,base_model_source=source["model_source"],selected_checkpoints=selected,
        protected_sha256=protected,query_ids=ids,n=256,label="ZPHOT (photometric; not spectroscopy)",
        query_manifest_sha256=sha(FAST/"summary.json"),overlap_audit=meta["overlap_audit"],source_exclusion=fast["source_exclusion"],
        original_selected_indices=meta["selected_indices"],target_split="validation",test_evaluated=False,training=False,
        weights_updated=False,target_labels_fitted=False,target_calibration=False,target_based_selection=False,
        max_length=max_length,sequence_rule="One common limit=max(512,largest complete record) for all conditions, selected before inference",token_audits=token_audits,
        would_truncate_at_512={k:sum(n>512 for n in v) for k,v in raw_lengths.items()},
        input_sha256={name:sha(out/"inputs"/f"{name}.jsonl") for name in texts},cohort_sha256=sha(out/"cohort.npz"),
        catalogue_fingerprint=dict(path=str(CATALOGUE),bytes=CATALOGUE.stat().st_size,mtime_ns=CATALOGUE.stat().st_mtime_ns),
        registry_sha256=sha(DEFAULT_REGISTRY),code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'aion_magnitude/filter_photometry.py',ROOT/'aion_magnitude/clauds_common_flux.py']},
        concentration_definition="Maximum fraction in any closed redshift interval of width 0.03 (two source-grid bins), plus most-populated single median-bin fraction; no labels used",
        example_indices=examples,execution_commands=[],
    )
    write_json(out/"manifest.json",manifest)
    print(json.dumps(dict(output=str(out),n=256,max_length=max_length,token_audits=token_audits,counts=counts)),flush=True)


def load_model(out, manifest, arm):
    from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model_state_dict
    from safetensors.torch import load_file

    c=manifest["source_config"];selected=Path(manifest['selected_checkpoints'][arm]['directory'])
    base,tokenizer=load_frozen_qwen(c['model_path'],device='cuda',load_in_4bit=True,torch_dtype='bf16',local_files_only=True)
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
    qwen=PeftModel.from_pretrained(base,selected/'adapter',is_trainable=False)
    expected=load_file(str(selected/'adapter/adapter_model.safetensors'));actual=get_peft_model_state_dict(qwen)
    assert actual.keys()==expected.keys()
    for k,v in actual.items():torch.testing.assert_close(v.cpu(),expected[k],rtol=0,atol=0,check_dtype=False)
    model=QwenPhotoZModel(qwen,hidden_size=qwen_hidden_size(qwen),n_z_bins=c['n_z_bins'],head_hidden_dim=c['head_hidden_dim'],pooling=c['pooling']).to('cuda')
    head=torch.load(selected/'photoz_head.pt',map_location='cpu',weights_only=True);model.photoz_head.load_state_dict(head,strict=True)
    for k,v in model.photoz_head.state_dict().items():torch.testing.assert_close(v.cpu(),head[k],rtol=0,atol=0)
    model.requires_grad_(False);model.eval();assert not any(p.requires_grad for p in model.parameters())
    return model,tokenizer


def run(args):
    out=args.output_dir;manifest=json.loads((out/'manifest.json').read_text());check_protected(out)
    cohort=read_npz(out/'cohort.npz');assert sha(out/'cohort.npz')==manifest['cohort_sha256']
    data={k:cohort[k] for k in ('targetid','features','z')}
    folder=out/f'arm-{args.arm}'
    if folder.exists():raise FileExistsError(f'Existing arm outputs are immutable: {folder}')
    folder.mkdir();started=time.monotonic()
    write_json(folder/'status.json',dict(phase='starting',utc=now(),pid=os.getpid(),weights_updated=False))
    try:
        with pilot.reserve_one_gpu() as gpu:
            if gpu['memory_before_gib']['used']>.2:raise RuntimeError('Inference requires an idle slice')
            model,tokenizer=load_model(folder,manifest,args.arm)
            config=dict(manifest['source_config'],max_length=manifest['max_length'])
            results={}
            for name in manifest['conditions']:
                if not name.startswith(args.arm+'_'):continue
                path=out/'inputs'/f'{name}.jsonl';assert sha(path)==manifest['input_sha256'][name]
                payload=[json.loads(line) for line in path.read_text().splitlines()]
                assert [p['object_id'] for p in payload]==manifest['query_ids']
                texts=[p['text'] for p in payload];audit_tokens(tokenizer,texts,manifest['max_length'])
                write_json(folder/'status.json',dict(phase='inference',condition=name,utc=now(),pid=os.getpid(),weights_updated=False))
                t=time.monotonic();evaluation=pilot.evaluate(model,tokenizer,data,texts,config)
                assert evaluation['object_id']==manifest['query_ids'];assert evaluation['logits'].shape==(256,300)
                torch.testing.assert_close(evaluation['z_spec'],torch.from_numpy(data['z']),rtol=0,atol=0)
                assert np.isfinite(evaluation['logits'].numpy()).all()
                if '_native_' in name:
                    ref=evaluation_from_archive(BASE/f'level{"ABC".index(args.arm)}/clauds_selected_predictions.npz')
                    replay=verify_replay(evaluation,ref);write_json(folder/'native_replay.json',replay)
                    if args.arm=='A':evaluation=ref # The table's Level-0 control remains archived.
                torch.save(evaluation,folder/f'{name}.pt')
                pilot.save_predictions(folder/f'{name}.npz',data,evaluation)
                results[name]=dict(metrics=summarize_pdf_metrics(evaluation),seconds=time.monotonic()-t,weights_updated=False)
                print(name,json.dumps(results[name]),flush=True)
            check_protected(out)
            write_json(folder/'summary.json',dict(conditions=results,gpu=gpu,seconds=time.monotonic()-started,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,weights_updated=False,test_evaluated=False,
                command=sys.argv,code_sha256=sha(__file__)))
            write_json(folder/'status.json',dict(phase='complete',utc=now(),pid=os.getpid(),conditions=list(results),weights_updated=False))
    except BaseException as error:
        write_json(folder/'status.json',dict(phase='failed',utc=now(),error=repr(error),pid=os.getpid(),weights_updated=False))
        raise


def prediction_diagnostics(e):
    pred=e['z_p50'].numpy();z=e['z_spec'].numpy();sorted_z=np.sort(pred)
    max_n=max(np.searchsorted(sorted_z,x+.0300001,side='right')-i for i,x in enumerate(sorted_z))
    values,counts=np.unique(pred,return_counts=True);mode=values[np.argmax(counts)]
    return dict(predicted_median=float(np.median(pred)),predicted_p05=float(np.quantile(pred,.05)),
        predicted_p95=float(np.quantile(pred,.95)),concentration_width_003=float(max_n/len(pred)),
        modal_median_bin=float(mode),modal_bin_fraction=float(counts.max()/len(pred)),unique_median_bins=len(values),
        predicted_in_grid_fraction=float(((pred>=0)&(pred<=4.5)).mean()),reference_in_grid_fraction=float(((z>=0)&(z<=4.5)).mean()),
        raw_mean_bias=float((pred-z).mean()),raw_median_bias=float(np.median(pred-z)))


def collect(args):
    out=args.output_dir;manifest=json.loads((out/'manifest.json').read_text());check_protected(out)
    if (out/'summary.json').exists():raise FileExistsError('Existing final results are immutable')
    evaluations={};metrics={}
    for name in CONDITIONS:
        arm=name[0];folder=out/f'arm-{arm}';assert json.loads((folder/'status.json').read_text())['phase']=='complete'
        e=torch.load(folder/f'{name}.pt',map_location='cpu',weights_only=False);assert e['object_id']==manifest['query_ids']
        evaluations[name]=e;metrics[name]=dict(**summarize_pdf_metrics(e),**prediction_diagnostics(e))
    with (out/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['condition',*metrics[CONDITIONS[0]]]);w.writeheader()
        for name in CONDITIONS:w.writerow(dict(condition=name,**metrics[name]))
    pairs=[('B_flux_full','B_native_full'),('C_flux_full','C_native_full'),('B_flux_hsc','B_flux_full'),
           ('C_flux_hsc','C_flux_full'),('C_flux_full','B_flux_full'),('C_flux_hsc','B_flux_hsc'),('B_flux_full','A_native_full')]
    paired={}
    for new,old in pairs:
        n,o=evaluations[new],evaluations[old];zn=n['z_p50'].numpy();zo=o['z_p50'].numpy();z=n['z_spec'].numpy()
        en=(zn-z)/(1+z);eo=(zo-z)/(1+z)
        paired[f'{new} minus {old}']=dict(metric_deltas={k:metrics[new][k]-metrics[old][k] for k in summarize_pdf_metrics(n)},
            median_prediction_shift=float(np.median(zn-zo)),mean_abs_prediction_shift=float(np.mean(np.abs(zn-zo))),
            improved_absolute_error=int((np.abs(en)<np.abs(eo)).sum()),rescued_outliers=int(((np.abs(eo)>.15)&(np.abs(en)<=.15)).sum()),
            new_outliers=int(((np.abs(eo)<=.15)&(np.abs(en)>.15)).sum()),query_ids_identical=True)
    write_json(out/'paired_differences.json',paired)
    with (out/'paired_predictions.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['object_id','CLAUDS_ZPHOT',*CONDITIONS])
        for i,ident in enumerate(manifest['query_ids']):w.writerow([ident,float(evaluations[CONDITIONS[0]]['z_spec'][i]),*[float(evaluations[n]['z_p50'][i]) for n in CONDITIONS]])
    plots(out,evaluations,manifest)
    summary=dict(completed_utc=now(),conditions=metrics,paired=paired,weights_updated=False,test_evaluated=False,
                 target_labels_fitted=False,target_based_selection=False,protected_artifacts_unchanged=True,
                 n=256,label='CLAUDS ZPHOT',manifest_sha256=sha(out/'manifest.json'))
    write_json(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


def plots(out,evals,manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={n:plt.get_cmap('tab10')(i) for i,n in enumerate(CONDITIONS)}
    fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for ax,name in zip(axes.flat,CONDITIONS):
        e=evals[name];ax.scatter(e['z_spec'],e['z_p50'],s=10,alpha=.5,color=colors[name]);ax.plot([0,4.5],[0,4.5],ls='--',color='.4')
        ax.set(title=name,xlabel='CLAUDS ZPHOT (photometric)',ylabel='Predicted PDF median',xlim=(0,4.5),ylim=(0,4.5))
    axes.flat[-1].axis('off')
    for ext in ('png','pdf'):fig.savefig(out/f'prediction_scatter.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
    for name,e in evals.items():
        axes[0].hist(e['z_p50'].numpy(),bins=np.linspace(0,4.5,61),histtype='step',label=name,color=colors[name])
        v=np.sort(e['z_p50'].numpy());axes[1].step(v,np.arange(1,257)/256,label=name,color=colors[name])
    axes[0].hist(next(iter(evals.values()))['z_spec'].numpy(),bins=np.linspace(0,4.5,61),histtype='step',label='Reference ZPHOT',color='black',ls='--')
    axes[0].set(xlabel='Redshift',ylabel='Number of queries');axes[1].set(xlabel='Predicted median redshift',ylabel='Empirical CDF',xlim=(0,4.5));axes[0].legend(fontsize=7)
    for ext in ('png','pdf'):fig.savefig(out/f'prediction_distributions.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for ax,i in zip(axes.flat,(0,64,128,192)):
        for name,e in evals.items():ax.plot(e['redshift_centers'],e['pz'][i],label=name,color=colors[name],lw=1)
        z=float(next(iter(evals.values()))['z_spec'][i]);ax.axvline(z,color='black',ls='--',label='ZPHOT')
        ax.set(title=f'Query {i}, ID {manifest["query_ids"][i]}',xlabel='Redshift',ylabel='Probability per source bin',xlim=(0,4.5))
    axes.flat[0].legend(fontsize=6)
    for ext in ('png','pdf'):fig.savefig(out/f'representative_pdfs.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,arm in zip(axes,('B','C')):
        z=evals[f'{arm}_native_full']['z_spec'].numpy()
        for variant in ('native_full','flux_full','flux_hsc'):
            name=f'{arm}_{variant}';ax.scatter(z,evals[name]['z_p50'],s=12,alpha=.5,label=variant,color=colors[name])
        ax.plot([0,4.5],[0,4.5],color='.4',ls='--');ax.set(title=f'Matched queries: {arm}',xlabel='CLAUDS ZPHOT',ylabel='PDF median',xlim=(0,4.5),ylim=(0,4.5));ax.legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(out/f'paired_predictions.{ext}',dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('prepare','run','collect'),required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--arm',choices=('A','B','C'))
    args=p.parse_args()
    if args.stage=='run' and args.arm is None:p.error('--arm is required for inference')
    {'prepare':prepare,'run':run,'collect':collect}[args.stage](args)


if __name__=='__main__':main()
