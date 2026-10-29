"""Frozen BS native interface extraction; never modifies historical files."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'notebooks'),str(ROOT)]
from anecdote_common_1029 import *
import anecdote_features_1029 as feat
from aion_magnitude.filter_photometry import serialize_rows,audit_tokens
from aion_magnitude.qwen_posttraining import QwenPhotoZModel
from aion_magnitude.qwen_continued_sft import verify_replay
from qwen_zero_shot_sanity import reserve_one_gpu,evaluate
from aion_magnitude.models import PhotoZHead
DEST=ROOT/'reports/stage4_5/readout_knn_discrepancy_1029'
def run():
    torch.set_num_threads(2);t0=time.time()
    a=d.npz(OUT/'GMS_photometry.npz'); chosen=np.isin(a['split'],['support_pool','development','query']); obj=a['object_id'][chosen]
    h=d.npz(RAW/'hdf_metadata.npz',('targetid','hsc_features'));lookup={str(v):i for i,v in enumerate(h['targetid'])};ii=np.array([lookup[x] for x in obj]);raw=h['hsc_features'][ii]
    assert np.array_equal(h['targetid'][ii].astype(str),obj)
    assert np.allclose(raw[:,:10:2],a['flux'][chosen,:5],equal_nan=True) and np.allclose(raw[:,1:10:2],a['sigma'][chosen,:5],equal_nan=True)
    assert np.allclose(raw[:,-1],a['ebv'][chosen],equal_nan=True)
    meta=json.loads((d.SOURCE/'prepared.json').read_text());texts=serialize_rows({'features':raw},meta,catalogue='galaxiesml-spec',level=0)
    r=torch.load(BS/'result.pt',map_location='cpu',weights_only=False,mmap=True)['test_evaluation'];idx={str(v):i for i,v in enumerate(r['object_id'])};assert all(str(x) in idx for x in obj)
    ridx=np.array([idx[str(x)] for x in obj]); np.savez_compressed(DEST/'archived_native_all_clean.npz',object_id=obj,split=a['split'][chosen],**{k:r[k][ridx].numpy() for k in ['logits','pz','z_spec','z_p16','z_p50','z_p84']},redshift_edges=r['redshift_edges'].numpy(),redshift_centers=r['redshift_centers'].numpy())
    with reserve_one_gpu() as gpu:
        model,tok=feat.load('BS');sig=feat.parameter_signature(model);replay=feat.replay(model,tok,'BS');audit=audit_tokens(tok,texts,512)
        c=json.loads((OUT/'checkpoint_provenance.json').read_text())['families']['BS']['metadata']['config'];q=np.flatnonzero(a['split'][chosen]=='query')[:32]; wrapper=QwenPhotoZModel(model,hidden_size=2560,n_z_bins=300,head_hidden_dim=256,pooling='last').to('cuda');wrapper.photoz_head.load_state_dict(torch.load(BS/'photoz_head.pt',map_location='cpu',weights_only=True));wrapper.requires_grad_(False);wrapper.eval()
        ev=evaluate(wrapper,tok,dict(targetid=obj[q],z=r['z_spec'][ridx[q]].numpy()),[texts[i] for i in q],c)
        ref={k:r[k][ridx[q]] if k not in ['redshift_edges','redshift_centers'] else r[k] for k in ['logits','pz','z_spec','z_p16','z_p50','z_p84','redshift_edges','redshift_centers']};ref['object_id']=list(obj[q]);query_replay=verify_replay(ev,ref)
        del wrapper
        last=[];mean=[]
        for i in range(0,len(texts),8):
            l,m=feat.hidden(model,tok,texts[i:i+8]);last.append(l);mean.append(m)
            if i%256==0: print('NATIVE',i,'/',len(texts),'seconds',int(time.time()-t0),flush=True)
        last=np.concatenate(last);mean=np.concatenate(mean);assert sig==feat.parameter_signature(model);assert not any(p.requires_grad for p in model.parameters())
        np.savez_compressed(DEST/'BS_native_features.npz',object_id=obj,split=a['split'][chosen],last=last,mean=mean)
        torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
    info=dict(n_clean=len(obj),native_replay=replay,query_32_archived_replay=query_replay,native_token_audit=audit,source_features_match_stage4=True,all_original_test_ids_matched=True,backbone_frozen=True,parameter_signature_unchanged=True,GPU=gpu,peak_GPU_bytes=peak,elapsed_seconds=time.time()-t0,archived_predictions_sha256=d.sha(DEST/'archived_native_all_clean.npz'),native_features_sha256=d.sha(DEST/'BS_native_features.npz'))
    d.write(DEST/'native_extraction.json',info);print('NATIVE COMPLETE',info,flush=True)
if __name__=='__main__':run()
