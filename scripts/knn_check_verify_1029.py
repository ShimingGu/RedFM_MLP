"""Validate all final query predictions and protected historical artifacts."""
import json,hashlib,sys,time,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'notebooks'),str(ROOT)]
from anecdote_common_1029 import *
P=ROOT/'reports/stage4_5/readout_knn_discrepancy_1029'
def main():
 start=time.time();m=json.loads((OUT/'manifest.json').read_text());bad=[]
 for path,sha in m['files_sha256'].items():
  p=OUT/path
  if not p.is_file() or d.sha(p)!=sha:bad.append(str(p))
 for path,sha in m['scripts_sha256'].items():
  p=ROOT/'scripts'/path
  if not p.is_file() or d.sha(p)!=sha:bad.append(str(p))
 protected=json.loads((OUT/'protected_before.json').read_text())
 for path,sha in protected.items():
  p=Path(path)
  if not p.is_file() or d.sha(p)!=sha:bad.append(str(p))
 base=json.loads((OUT/'checkpoint_provenance.json').read_text())['base_files_sha256']
 for path,sha in base.items():
  p=BASE/path
  if not p.is_file() or d.sha(p)!=sha:bad.append(str(p))
 assert not bad,('PROTECTED FILES CHANGED',bad[:10],len(bad))
 a=d.npz(OUT/'GMS_photometry.npz');q=a['split']=='query';expected=a['object_id'][q];truth=d.npz(OUT/'GMS_query_labels_sealed.npz');zz={v:z for v,z in zip(truth['object_id'],truth['z'])};z=np.array([zz[v] for v in expected]);maxnorm=0;count=0
 for path in sorted((P/'predictions').glob('*.npz')):
  v=d.npz(path);assert np.array_equal(v['object_id'],expected),path;assert np.array_equal(v['z_spec'],z),path;assert v['pz'].shape==(1024,300) and np.isfinite(v['pz']).all() and (v['pz']>=0).all();delta=float(abs(v['pz'].sum(1)-1).max());maxnorm=max(maxnorm,delta);assert delta<2e-6,(path,delta);assert all(np.isfinite(v[k]).all() for k in ['logits','z_p16','z_p50','z_p84']);count+=1
 replay=json.loads((P/'native_extraction.json').read_text());assert replay['native_replay']['passed'] and replay['query_32_archived_replay']['passed'] and replay['query_32_archived_replay']['max_abs_pdf_error']==0;assert d.sha(P/'BS_native_features.npz')==replay['native_features_sha256'];assert d.sha(P/'archived_native_all_clean.npz')==replay['archived_predictions_sha256'];assert replay['all_original_test_ids_matched'] and replay['backbone_frozen'] and replay['parameter_signature_unchanged']
 init=json.loads((P/'native_head_training.json').read_text());assert len(init['results'])==6 and all(r['head_learned'] for r in init['results']);audit=json.loads((P/'asset_and_identity_audit.json').read_text());assert audit['existing_head_fits_reproduced']==48 and audit['maximum_head_pdf_replay_error']==0;assert not any(audit['known_historical_exposures'].values())
 k=json.loads((P/'knn_reproduction_checks.json').read_text());assert len(k)==3 and all(max(r['pdf_max_difference'],r['query_label_perturbation_pdf_max_difference'],r['query_row_permutation_pdf_max_difference'])<2e-6 for r in k)
 links=[]
 for md in P.glob('*.md'):
  for target in re.findall(r'\]\(([^)]+)\)',md.read_text()):
   if not (target.startswith(('http:','https:','#')) or (md.parent/target).exists()):links.append((str(md),target))
 assert not links,links
 assert d.sha(P/'protocol.json')==(P/'protocol.sha256').read_text().split()[0]
 verification=dict(completed=True,stage4_manifest_files_unchanged=len(m['files_sha256']),stage4_scripts_unchanged=len(m['scripts_sha256']),historical_protected_files_unchanged=len(protected),base_files_unchanged=len(base),historical_file_failures=bad,query_ids_identical=True,query_prediction_files=count,maximum_pdf_normalization_error=maxnorm,reproduced_stage4_heads=48,original_knn_reproductions=3,query_label_and_row_perturbations_passed=True,native_validation_and_query_replays_exact=True,new_native_heads=6,base_or_adapter_updates=False,query_used_for_selection=False,retrospective=True,stage2c=False,links_valid=True,seconds=time.time()-start)
 d.write(P/'verification.json',verification)
 manifest=dict(protocol_sha256=d.sha(P/'protocol.json'),scripts_sha256={p.name:d.sha(p) for p in sorted((ROOT/'scripts').glob('knn_check*1029.py'))},files_sha256={str(p.relative_to(P)):d.sha(p) for p in sorted(P.rglob('*')) if p.is_file() and p.name!='manifest.json'},verification=verification)
 d.write(P/'manifest.json',manifest);print('STAGE4.5 VERIFIED',verification,flush=True)
if __name__=='__main__':main()
