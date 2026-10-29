"""One immutable frozen feature cache per backbone, task and shared format."""
import json,time,os
from pathlib import Path
import numpy as np
import torch
from anecdote_common_1029 import *
def serialize(a,format_name):
    registry=json.loads((ROOT/'configs/filter_registry_v1.json').read_text())['filters'];texts=[]
    for i in range(len(a['object_id'])):
        candidates=np.flatnonzero(a['band_id'][i]>=0);candidates=sorted(candidates,key=lambda j:registry[FILTERS[a['band_id'][i,j]]]['pivot_nm']);blocks=[]
        for j in candidates:
            band=FILTERS[a['band_id'][i,j]];f=a['flux'][i,j];e=a['sigma'][i,j];valid=np.isfinite(f)&np.isfinite(e)&(e>0);fs=f'{f:.7g}' if valid else 'NA';es=f'{e:.7g}' if valid else 'NA';qa=bool(a['quality_available'][i,j]);q=f"{a['quality'][i,j]:.3g}" if qa else 'NA'
            if format_name=='canonical':blocks.append(f'band={band} flux={fs} sigma={es} available={int(valid)} quality={q}')
            else:blocks.append(f'{band}_flux={fs}; {band}_flux_error={es}; {band}_available={int(valid)}; {band}_quality={q}')
        ebv=f"{a['ebv'][i]:.7g}" if a['ebv_available'][i] else 'NA'
        prefix='galaxy observed_photometry_v1 flux_unit=AB_nanomaggy sigma=one_sigma' if format_name=='canonical' else 'galaxy native_observed_photometry schema=shared_flat_flux_v1 flux_unit=AB_nanomaggy errors=one_sigma extinction=as_catalogued'
        texts.append(prefix+' '+' | '.join(blocks)+f' MW_EBV={ebv} PHOTOZ_QUERY')
    return texts
def prepare():
    from transformers import AutoTokenizer
    assert (OUT/'data_ready.json').exists();assert not (OUT/'protocol.json').exists()
    tokenizer=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True);(OUT/'inputs').mkdir(exist_ok=True);audits={}
    for task in TASKS:
        a=d.npz(OUT/f'{task}_photometry.npz')
        for fmt in ['canonical']+(['alternate'] if task in ['GMS','C_SPEC'] else []):
            texts=serialize(a,fmt);tokens=tokenizer(texts,padding=False,truncation=False);lengths=np.array([len(v) for v in tokens['input_ids']]);assert lengths.max()<=1024
            last=np.array([v[-1] for v in tokens['input_ids']]);assert len(set(last))==1
            payload=[dict(object_id=str(x),text=t) for x,t in zip(a['object_id'],texts)];path=OUT/'inputs'/f'{task}_{fmt}.json';path.write_text(json.dumps(payload,ensure_ascii=False));audits[task+'_'+fmt]=dict(rows=len(a['object_id']),max_tokens=int(lengths.max()),min_tokens=int(lengths.min()),terminal_token_id=int(last[0]),sha256=d.sha(path),same_input_strings_all_backbones=True,actual_band_identity=True,physical_signed_fluxes=True,no_object_or_redshift_in_strings=True)
    assert len(set(x['terminal_token_id'] for x in audits.values()))==1
    cfg=dict(study='PaperII frozen-backbone head-only representation transfer; independent of PaperI kernels',backbones=BACKBONES,compatible_base='Qwen3.5-4B-Base; same full content hashes and rank8 QLoRA shapes',tasks=TASKS,task_interpretations={'GMS':'BC cross-survey; BS in-domain control, clean held-out galaxies','C_SPEC':'BS cross-survey; BC in-domain control; exploratory new split of previously inspected Stage3 T2','C_ZPHOT':'BS cross-survey ZPHOT agreement; BC in-domain control, not true-redshift accuracy'},budgets=[16,32,64,128,256,512],primary_budgets=[64,256],subset_seeds=[101,202,303],head_seeds=[11,23],primary=dict(format='canonical',pooling='last',loss='hard_CE'),controls=dict(smooth_sigma3='Fixed Gaussian neighbor-label sigma3 bins atK64/256, all tasks/backbones',mean_pooling='Canonical masked mean atK64/256, all tasks/backbones',formatting='Same alternate flat signed-flux/error strings atK64/256 forGMS andC_SPEC, all backbones; source-native-like structural control, not exact BC median-imputed magnitude native format',negative_adapter='Optional zero-effect initialized LoRA insertion check on B0, no backbone randomization required'),feature_normalization='Fit each feature mean and populationSTD on K support objects only; STD floor5% of median positiveSTD, absolute floor1e-4; clip standardized coordinates to[-10,10]; same rule for all backbones; no transductive/query statistics',head=dict(input_dim=2560,hidden_dim=32,output_dim=300,activation='SiLU',dropout=.1,initialization='Fresh torch seed; identical initial weights across same-dimension backbone heads; never initialize from old head'),optim=dict(algorithm='AdamW',lr=.001,weight_decay=.01,updates=800,batch_size=64,validation_every=25,gradient_clip=5.,selection='Minimum development hard CE, including initialization; same head and subset seeds, batch indices and budget for all backbones'),grid=dict(zmin=0,zmax=4.5,bins=300,outside='Count and exclude, never clip labels'),point_estimator='CDF median on original300 bin centers',sparse_supervision='Main hard CE retained at allK. Separate fixed sigma3 smoothing at primaryK; no query choice of loss/smoothing.',pilot='K64/256, subset101, head11, canonical/last/hard andsmooth, alltasks/allbackbones. No query metrics during pilot. ModerateK256 source/development learning must pass. Full protocol locked before any query evaluation.',learning_gate='At least2/3 backbones per task improve development hardCE by>=0.05 with positive selected update atK256 under at least one preregistered loss; each selected head finite, training loss decreases>=0.3, frozen feature checks pass. Nonlearning arms are reported without interpreting them as representation rankings.',numerical_controls=['Uncertainty-aware target photometric kNN KDE, fixed k=min(10,K), Gaussian redshift bandwidthmax(.03*(1+zi),support_z_STD/sqrtK)','Compact target-only numerical MLP, same hidden32/output300/optimizer/head seeds/batches and hard loss; sigma3 control at primaryK','Affine correction of original frozen source-head median, ridge toward slope1, Gaussian residual PDF with fixed .03 minimum width; same K support; GMS usesBC, CLAUDS usesBS'],uncertainty='Paired galaxy bootstrap1000 replicates(seed1029) conditional on subset/head seed; separate variability across3 subset seeds and2 initialization seeds; no pooling repeated predictions as independent galaxies',metrics=['NMAD1.48','outlier absdz/(1+z)>.15','mean and median normalizedbias','hard CE','CRPS','68%coverage','PIT hist','predictedzdistribution','concentration width.03','failures','redshift/magnitude/SNR/completeness/support strata'],query_policy='All head/checkpoint/normalizer selections completed and hashed before sealed query labels are read. T2 reuse explicitly exploratory. No query-driven pooling, format, optimization orK selection.',exposure_diagnostic='Evaluate primaryK64/256 frozen heads also on separately tagged historically trained objects; non-independent descriptive diagnostic, not matched causal exposure effect',original_heads='Separate frozen original-head references on shared strings; never initialize fresh heads. BC native grid differs from fresh grid, so native-grid CE is not directly compared or plotted as K0 on fresh learning curve.',no_updates=['Qwen base','QLoRA adapters','DoRA adapters'],no_stage2C=True,extraction=dict(quantization='NF4 double quantization, BF16 compute, same kbit preparation forB0/BS/BC',model_mode='eval, allrequires_gradFalse, inference_mode, no KV cache',max_tokens=1024,batch_size=8,primary_pooling='last actual nonpadding token',mean_control='FP32 masked mean from same forward',integrity='Native archived-head replay where executable, exact adapter tensor reload, parameter version counters and sample digests, unchanged on-disk hashes',cache='Once per object per backbone/serialization; both pooling vectors saved; IDs and allinput/model/code hashes bound'))
    cfg['input_definitions']=audits;cfg['data_hashes']={p.name:d.sha(p) for p in OUT.glob('*.npz')};cfg['nested_subset_hashes']={p.name:d.sha(p) for p in OUT.glob('*nested_subsets.json')};cfg['provenance_sha256']=d.sha(OUT/'checkpoint_provenance.json');cfg['guide_sha256']=d.sha(ROOT/'reports/anecdote_header_1029.md');cfg['candidate_code_sha256']={p.name:d.sha(p) for p in (ROOT/'scripts').glob('anecdote*1029.py')}
    d.write(OUT/'protocol.json',cfg);d.write(OUT/'protocol_lock.json',dict(sha256=d.sha(OUT/'protocol.json'),UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),new_query_evaluated=False));print('PROTOCOL LOCKED',d.sha(OUT/'protocol.json'),audits,flush=True)
def parameter_signature(model):
    version={k:int(v._version) for k,v in model.named_parameters()};h=__import__('hashlib').sha256()
    for k,p in model.named_parameters():h.update(k.encode());h.update(p.detach().reshape(-1)[:8].to(dtype=torch.float32,device='cpu').numpy().tobytes())
    return version,h.hexdigest()
def load(family):
    from aion_magnitude.FM_Qwen import load_frozen_qwen
    from peft import PeftModel,prepare_model_for_kbit_training,get_peft_model_state_dict
    from safetensors.torch import load_file
    model,tokenizer=load_frozen_qwen(BASE,device='cuda',load_in_4bit=True,torch_dtype='bf16',local_files_only=True)
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    if family!='B0':
        folder=BS if family=='BS' else BC;model=PeftModel.from_pretrained(model,folder/'adapter',is_trainable=False);expected=load_file(str(folder/'adapter/adapter_model.safetensors'));actual=get_peft_model_state_dict(model);assert actual.keys()==expected.keys()
        for k,v in actual.items():torch.testing.assert_close(v.cpu(),expected[k],rtol=0,atol=0,check_dtype=False)
    model.requires_grad_(False);model.eval();assert not any(p.requires_grad for p in model.parameters());assert not any(m.training for m in model.modules())
    return model,tokenizer
@torch.inference_mode()
def hidden(model,tokenizer,texts):
    tokens=tokenizer(texts,return_tensors='pt',padding=True,truncation=False);assert tokens['input_ids'].shape[1]<=1024;tokens={k:v.to('cuda') for k,v in tokens.items()}
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**tokens,use_cache=False,return_dict=True).last_hidden_state
    mask=tokens['attention_mask'].bool();index=torch.arange(mask.shape[1],device='cuda')[None,:].expand_as(mask).masked_fill(~mask,-1).amax(1);last=out[torch.arange(len(texts),device='cuda'),index].float();mean=(out.float()*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
    assert torch.isfinite(last).all() and torch.isfinite(mean).all();return last.cpu().numpy(),mean.cpu().numpy()
def replay(model,tokenizer,family):
    if family=='B0':return dict(not_applicable='No historical redshift head on foundation-only B0')
    from aion_magnitude.qwen_posttraining import QwenPhotoZModel
    import qwen_zero_shot_sanity as pilot
    from aion_magnitude.qwen_continued_sft import verify_replay
    folder=BS if family=='BS' else BC;r=d.npz(OUT/f'replay/{family}.npz');texts=json.loads((OUT/f'replay/{family}_texts.json').read_text());c=json.loads((OUT/'checkpoint_provenance.json').read_text())['families'][family]['metadata']['config'];wrapper=QwenPhotoZModel(model,hidden_size=2560,n_z_bins=300,head_hidden_dim=256,pooling='last').to('cuda');wrapper.photoz_head.load_state_dict(torch.load(folder/'photoz_head.pt',map_location='cpu',weights_only=True));wrapper.requires_grad_(False);wrapper.eval();ev=pilot.evaluate(wrapper,tokenizer,dict(targetid=r['object_id'],z=r['z_spec']),texts,c);ref={k:torch.tensor(r[k]) for k in ['logits','pz','z_spec','z_p16','z_p50','z_p84','redshift_edges','redshift_centers']};ref['object_id']=list(r['object_id']);v=verify_replay(ev,ref);del wrapper;return v
def extract(family):
    assert family in BACKBONES;cfg=json.loads((OUT/'protocol.json').read_text());assert d.sha(OUT/'protocol.json')==json.loads((OUT/'protocol_lock.json').read_text())['sha256'];folder=OUT/'features'/family;folder.mkdir(parents=True,exist_ok=True);assert not (folder/'manifest.json').exists()
    from qwen_zero_shot_sanity import reserve_one_gpu
    with reserve_one_gpu() as gpu:
        start=time.time();model,tokenizer=load(family);sig=parameter_signature(model)
        # Replay is a loading audit; any execution discrepancy is recorded and diagnosed.
        try:native=replay(model,tokenizer,family)
        except AssertionError as err:
            d.write(folder/'native_replay_failed.json',dict(error=str(err)));raise
        d.write(folder/'native_replay.json',native);man={}
        for task in TASKS:
            a=d.npz(OUT/f'{task}_photometry.npz')
            for fmt in ['canonical']+(['alternate'] if task in ['GMS','C_SPEC'] else []):
                inputfile=OUT/'inputs'/f'{task}_{fmt}.json';assert d.sha(inputfile)==cfg['input_definitions'][task+'_'+fmt]['sha256'];payload=json.loads(inputfile.read_text());assert np.array_equal(np.array([p['object_id'] for p in payload]),a['object_id']);texts=[p['text'] for p in payload];last=[];mean=[];begin=time.time()
                for i in range(0,len(texts),8):
                    l,m=hidden(model,tokenizer,texts[i:i+8]);last.append(l);mean.append(m)
                    if i%512==0:print(family,task,fmt,i,'/',len(texts),flush=True)
                last=np.concatenate(last);mean=np.concatenate(mean);repeat=hidden(model,tokenizer,texts[:8]);np.testing.assert_array_equal(last[:8],repeat[0]);np.testing.assert_array_equal(mean[:8],repeat[1]);assert last.shape==(len(texts),2560)
                path=folder/f'{task}_{fmt}.npz';np.savez_compressed(path,object_id=a['object_id'],split=a['split'],last=last,mean=mean)
                man[task+'_'+fmt]=dict(path=str(path),sha256=d.sha(path),input_sha256=d.sha(inputfile),rows=len(last),dimension=2560,poolings=['last','FP32_masked_mean'],dtype='float32 cached; NF4/BF16 frozen forward',seconds=time.time()-begin,deterministic_repeat_max=0.)
        assert sig==parameter_signature(model),'Frozen parameter version/sample digest changed';assert not any(p.requires_grad for p in model.parameters())
        pro=json.loads((OUT/'checkpoint_provenance.json').read_text())
        for p,sha in (pro['families'][family]['files_sha256'].items() if family!='B0' else []):assert d.sha(Path(p))==sha
        d.write(folder/'manifest.json',dict(family=family,protocol_sha256=d.sha(OUT/'protocol.json'),base_files_sha256=pro['base_files_sha256'],adapter_files_sha256=pro['families'][family]['files_sha256'] if family!='B0' else {},tokenizer_sha256=pro['base_files_sha256']['tokenizer.json'],extraction_code_sha256=d.sha(Path(__file__)),feature_files=man,GPU=gpu,seconds=time.time()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),all_parameters_frozen=True,model_eval=True,no_optimizer_exists=True,native_replay=native,in_memory_parameter_signature_unchanged=True,parameter_sample_sha256=sig[1]))
        print('FEATURE EXTRACTION COMPLETE',family,time.time()-start,flush=True)
if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:extract(sys.argv[1])
