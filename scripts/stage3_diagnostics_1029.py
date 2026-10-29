"""Small registered physical controls, runtime measurements and final integrity."""
from pathlib import Path
import sys,json,time,hashlib,shutil
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
from stage2_data import d,take
from stage3_evaluate_1029 import A,B,physical_row,c4_data
import stage2_physics as pdf
def physical_controls():
    rows=[];datasets=c4_data()
    for name in ['HSC','Legacy']:
        a=d.npz(A/f'{name}_eligible.npz');a=take(a,a['split']=='val');priority=[hashlib.sha256(g.encode()).hexdigest() for g in a['group']];a=take(a,np.argsort(priority)[:512]);datasets['source_'+name]=a
    (B/'physical_diagnostics').mkdir(exist_ok=True)
    for name,a in datasets.items():
        np.savez_compressed(B/'physical_diagnostics'/f'input_{name}.npz',**a)
        with ProcessPoolExecutor(max_workers=6) as pool:scores=list(pool.map(physical_row,zip(a['flux'],a['sigma'],a['band_id']),chunksize=16))
        for j,arm in enumerate(['D12','D2']):
            pred=pdf.prediction(np.array([x[j] for x in scores]),a['z']);np.savez_compressed(B/'physical_diagnostics'/f'{name}_{arm}.npz',**pred,object_id=a['object_id'],group=a['group']);rows.append(dict(cohort=name,model=arm,**pdf.metric(pred)))
        print('registered physical diagnostic',name,flush=True)
    d.table(B/'physical_diagnostic_metrics.csv',rows)
def inference_runtime():
    from stage3_models_1029 import Model
    import stage2_numerical as base
    from stage3_train_1029 import logits
    torch.set_num_threads(2);a=d.npz(A/'T2_photometry.npz');a=take(a,np.arange(min(256,len(a['object_id']))));t=base.tensor_rows(a);rows=[];replay=[]
    for p in sorted((B/'models').glob('*/model.pt')):
        r=torch.load(p,map_location='cpu',weights_only=False);m=Model(r['arm']);m.load_state_dict(r['state']);logits(m,t);times=[]
        for _ in range(3):
            start=time.perf_counter();logits(m,t);times.append(time.perf_counter()-start)
        rows.append(dict(source=r['source'],arm=r['arm'],seed=r['seed'],batch=256,CPU_threads=2,median_seconds=float(np.median(times)),measure='Prepared tensors -> CPU logits; excludes model loading, physical kernel construction and metrics'))
        if r['arm'] in ['R1','R2']:
            va=d.npz(A/f"{r['source']}_eligible.npz");va=take(va,va['split']=='val');va=take(va,np.arange(32));pred=pdf.prediction(logits(m,base.tensor_rows(va)),va['z']);old=d.npz(p.parent/'validation_predictions.npz',('pz','object_id'));assert np.array_equal(va['object_id'],old['object_id'][:32]);diff=float(np.max(np.abs(pred['pz']-old['pz'][:32])));assert diff<1e-4;replay.append(dict(model=p.parent.name,source_CPU_GPU_max_PDF_difference=diff,n=32))
    d.table(B/'inference_runtime.csv',rows);d.write(B/'source_CPU_GPU_replay.json',dict(passed=True,models=replay,tolerance=1e-4))
def verify():
    protected=json.loads((A/'protected_before.json').read_text());changed=[p for p,h in protected.items() if not Path(p).exists() or d.sha(Path(p))!=h];assert not changed,changed
    cfg=json.loads((A/'benchmark.json').read_text());lock=json.loads((B/'source_gate_and_model_lock.json').read_text());assert lock['passed'] and lock['all_controls_frozen'];assert d.sha(A/'benchmark.json')==json.loads((A/'benchmark_lock.json').read_text())['sha256']==lock['protocol_sha256']
    for name,sha in cfg['dataset_sha256'].items():assert d.sha(A/name)==sha,name
    for name,sha in cfg['implementation_sha256'].items():assert d.sha(ROOT/'scripts'/name)==sha,name
    for name,sha in lock['model_sha256'].items():assert d.sha(B/name)==sha,name
    for folder in [B/'predictions',B/'physical_diagnostics']:
        for p in folder.glob('*.npz'):
            with np.load(p,allow_pickle=False) as f:
                if 'pz' not in f:continue
                assert np.isfinite(f['pz']).all() and np.allclose(f['pz'].sum(1),1,atol=1e-5),p
    a=d.npz(B/'T2_truth_eligible.npz');assert len(set(a['object_id']))==len(a['z']);assert np.all((a['z']>0)&(a['z']<=4.5))
    for name in ['HSC','Legacy','Mixed']:
        s=d.npz(A/f'{name}_eligible.npz');assert not set(s['group'][s['split']=='train'])&set(s['group'][s['split']=='val']);assert not np.any((s['ra']>147)&(s['ra']<153)&(s['dec']>-1)&(s['dec']<5))
    old=d.check(ROOT/'reports/stage0/1027a')
    result=dict(completed=True,protected_files_checked=len(protected),protected_files_unchanged=True,protocol_unchanged=True,all_preregistered_dataset_hashes_match=True,all_preregistered_candidate_implementation_hashes_match=True,all20_frozen_model_hashes_match=True,finite_normalized_prediction_files=len(list((B/'predictions').glob('*.npz'))),source_split_disjoint=True,COSMOS_field_held_out=True,target_IDs_unique=True,target_labels_in_range_without_clipping=True,original_Qwen_artifacts_unchanged=True,source_test_evaluated=False,Qwen_training=False,Stage2C=False,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    d.write(ROOT/'reports/physics_execution_1029_verification.json',result)
    for name,folder in [('stage3a',A),('stage3b',B)]:
        files={str(p.relative_to(folder)):d.sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json'};d.write(folder/'manifest.json',dict(stage=name,complete=True,protocol_sha256=d.sha(A/'benchmark.json'),files_sha256=files,verification=result))
    d.write(ROOT/'reports/physics_execution_1029_manifest.json',dict(stages={'3A':str(A/'report.md'),'3B':str(B/'report.md')},joint=str(ROOT/'reports/joint_physics_1029.md'),benchmark_sha256=d.sha(A/'benchmark.json'),source_model_lock_sha256=d.sha(B/'source_gate_and_model_lock.json'),source_gate_UTC=lock['created_UTC'],target_label_open_UTC=json.loads((B/'target_labels_opened.json').read_text())['UTC'],scripts_sha256={p.name:d.sha(p) for p in (ROOT/'scripts').glob('stage3*1029.py')},verification=result))
    print('FINAL VERIFICATION',result,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
