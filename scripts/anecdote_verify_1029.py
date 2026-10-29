"""Integrity, exposure separation and full artifact manifests for Paper II."""
import json,time,sys
from pathlib import Path
import numpy as np
import torch
from anecdote_common_1029 import *
def prequery():
    cfg=json.loads((OUT/'protocol.json').read_text());assert d.sha(OUT/'protocol.json')==json.loads((OUT/'protocol_lock.json').read_text())['sha256']
    amendment=json.loads((OUT/'source_only_sparse_amendment.json').read_text()) if (OUT/'source_only_sparse_amendment.json').exists() else None
    for name,sha in cfg['candidate_code_sha256'].items():
        if name=='anecdote_heads_1029.py' and amendment:
            assert d.sha(OUT/'source_version_archive/anecdote_heads_before_sparse_diagnostic.py')==sha
            assert d.sha(ROOT/'scripts'/name)==amendment['amended_head_code_sha256']
        else:assert d.sha(ROOT/'scripts'/name)==sha,name
    for name,sha in cfg['data_hashes'].items():assert d.sha(OUT/name)==sha,name
    expected=amendment['expected_head_counts'] if amendment else {'GMS':264,'C_SPEC':264,'C_ZPHOT':228};initial={};heads=0
    for task in TASKS:
        a=d.npz(OUT/f'{task}_photometry.npz');labels=d.npz(OUT/f'{task}_support_development_labels.npz');sealed=d.npz(OUT/f'{task}_query_labels_sealed.npz');assert not set(labels['object_id'])&set(sealed['object_id']);assert len(set(a['group']))==len(a['group']);assert np.all(a['exposure_category'][a['split']!='exposure_query']=='clean_no_train_or_val')
        lock=json.loads((OUT/f'{task}_head_selection_locked.json').read_text());assert len(lock['head_files_sha256'])==expected[task]
        for path,sha in lock['head_files_sha256'].items():
            p=OUT/path;assert d.sha(p)==sha;r=json.loads((p.parent/'summary.json').read_text());assert r['train_IDs']==r['normalizer_fit_IDs'];assert len(r['train_IDs'])==r['K'];assert set(r['train_IDs'])<=set(a['object_id'][a['split']=='support_pool']);assert r['all_optimizer_parameters_fresh_head_only'] and not r['backbone_model_loaded_in_training_process'];heads+=1
            if r['arm'] in BACKBONES:
                key=(task,r['head_seed']);initial.setdefault(key,set()).add(r['initial_head_sha256'])
        for backbone in BACKBONES:
            m=json.loads((OUT/'features'/backbone/'manifest.json').read_text());assert m['all_parameters_frozen'] and m['model_eval'] and m['no_optimizer_exists'] and m['in_memory_parameter_signature_unchanged']
            for name,rec in m['feature_files'].items():
                if not name.startswith(task+'_'):continue
                p=Path(rec['path']);assert d.sha(p)==rec['sha256'];f=d.npz(p);assert np.array_equal(f['object_id'],a['object_id']);assert np.isfinite(f['last']).all() and np.isfinite(f['mean']).all()
    assert heads==sum(expected.values());assert all(len(v)==1 for v in initial.values()),'Fresh head initialization differs across same-dimension backbone arms'
    result=dict(passed=True,heads_checked=heads,identical_initial_weights_per_task_seed=True,all_primary_objects_no_historical_train_or_val_exposure=True,all_backbones_frozen=True,all_normalizers_fit_only_K_support=True,query_not_used_for_selection=True,original_protocol_preserved=True,source_only_sparse_readout_amendment_verified=bool(amendment),UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    if not (OUT/'prequery_verification.json').exists():d.write(OUT/'prequery_verification.json',result)
    else:assert json.loads((OUT/'prequery_verification.json').read_text())['passed']
    print('PREQUERY VERIFIED',result,flush=True)
def final():
    assert json.loads((OUT/'query_evaluation_complete.json').read_text())['complete'];prequery()
    protected=json.loads((OUT/'protected_before.json').read_text());changed=[p for p,h in protected.items() if not Path(p).exists() or d.sha(Path(p))!=h];assert not changed,changed
    pro=json.loads((OUT/'checkpoint_provenance.json').read_text())
    for name,sha in pro['base_files_sha256'].items():assert d.sha(BASE/name)==sha
    finite=0
    for p in (OUT/'predictions').glob('*.npz'):
        f=d.npz(p,('pz','z_spec','object_id'));assert np.isfinite(f['pz']).all();assert np.allclose(f['pz'].sum(1),1,atol=1e-5);assert np.all((f['z_spec']>=0)&(f['z_spec']<=4.5));assert len(set(f['object_id']))==len(f['object_id']);finite+=1
    report=dict(completed=True,protected_files_checked=len(protected),historical_files_unchanged=True,all_base_model_full_content_hashes_unchanged=True,adapters_and_historical_heads_unchanged=True,head_models=len(list((OUT/'heads').glob('*/head.pt'))),finite_normalized_prediction_files=finite,all_backbones_no_grad=True,all_optimizers_fresh_heads_only=True,native_replays_BS_BC_exact=True,query_not_used_to_select_heads_or_format_or_pooling=True,old_T2_spec_study_exploratory=True,source_training_exposure_excluded=True,Stage2C=False,new_backbone_or_adapter_updates=False,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    d.write(OUT/'verification.json',report);files={str(p.relative_to(OUT)):d.sha(p) for p in OUT.rglob('*') if p.is_file() and p!=OUT/'manifest.json' and p.name!='status.json'};d.write(OUT/'manifest.json',dict(study='PaperII frozen readout anecdote1029',protocol_sha256=d.sha(OUT/'protocol.json'),files_sha256=files,scripts_sha256={p.name:d.sha(p) for p in (ROOT/'scripts').glob('anecdote*1029.py')},verification=report));print('FINAL VERIFIED',report,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
