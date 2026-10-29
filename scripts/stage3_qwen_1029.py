"""Frozen original checkpoint, target label-free inference with native HSC fields."""
from pathlib import Path
import sys,json,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'notebooks'),str(ROOT)]
from stage2_data import d,FILTERS
from aion_magnitude.filter_photometry import legacy_config,audit_tokens,serialize_rows
from aion_magnitude.FM_Qwen import serialize_qwen_feature_row
from aion_magnitude.qwen_continued_sft import verify_replay
import qwen_zero_shot_sanity as pilot
import qwen_clauds_common_flux_evaluation as runner
A=ROOT/'reports/stage3a/1029';OUT=ROOT/'reports/stage3b/1029/qwen_frozen'
def run():
    OUT.mkdir(parents=True,exist_ok=True);assert d.sha(A/'benchmark.json')==json.loads((A/'benchmark_lock.json').read_text())['sha256']
    m=d.check(ROOT/'reports/stage0/1027a');a=d.npz(A/'T2_photometry.npz');names=json.loads((d.SOURCE/'prepared.json').read_text())['feature_names'];config=dict(m['source_config'],max_length=512)
    features=np.full((len(a['object_id']),11),np.nan)
    for j,b in enumerate('grizy'):
        bid=FILTERS.index(f'HSC.{b}');mask=a['band_id']==bid
        features[:,2*j]=np.where(mask,a['flux'],0).sum(1);features[:,2*j+1]=np.where(mask,a['sigma'],0).sum(1)
    texts=[serialize_qwen_feature_row(dict(zip(names,row)),serialization=legacy_config('galaxiesml-spec')) for row in features]
    data=dict(targetid=a['object_id'],features=features,z=np.zeros(len(features),np.float32))
    with pilot.reserve_one_gpu() as gpu:
        start=time.time();model,tokenizer=runner.load_model(OUT,m,'original');tokenaudit=audit_tokens(tokenizer,texts,512)
        replaydata=d.npz(d.FAST/'source_replay.npz');metadata=json.loads((d.SOURCE/'prepared.json').read_text());replaytexts=serialize_rows(replaydata,metadata,catalogue='galaxiesml-spec',level=0)
        e=pilot.evaluate(model,tokenizer,dict(replaydata,z=replaydata['z_spec']),replaytexts,config)
        ref={k:torch.from_numpy(replaydata[k]) for k in ['logits','pz','z_spec','z_p16','z_p50','z_p84','redshift_edges','redshift_centers']};ref['object_id']=[str(x) for x in replaydata['targetid']]
        replay=verify_replay(e,ref);d.write(OUT/'native_replay.json',replay)
        e=pilot.evaluate(model,tokenizer,data,texts,config)
        assert e['object_id']==list(a['object_id']);assert np.isfinite(e['logits'].numpy()).all()
        # No placeholder is misrepresented as target truth.
        np.savez_compressed(OUT/'T2_label_free_predictions.npz',object_id=a['object_id'],features=features,**{k:e[k].numpy() for k in ['logits','pz','z_p16','z_p50','z_p84','redshift_edges','redshift_centers']})
        d.write(OUT/'provenance.json',dict(model='Original frozen HSC QLoRA and PDF head',checkpoint=m['selected_checkpoints']['original'],GPU=gpu,seconds=time.time()-start,target_labels_read=False,weights_updated=False,source_replay=replay,token_audit=tokenaudit,interface='Original HSC source fields/order/precision and actual CLAUDS grizy flux in nanomaggies; MW_EBV=NA (unavailable); no UV/NIR bands accepted by original interface',limitations='Counterfactual source schema alias; historical PDR2 nominal response differs from target exposure mixtures; missing EBV/terminal-token fragility measured in Stage0; no claim of equivalent information to full11-band numerical inference'))
    d.check(ROOT/'reports/stage0/1027a')
if __name__=='__main__':run()
