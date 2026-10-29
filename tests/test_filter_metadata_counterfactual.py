"""Integrity tests for the frozen metadata experiment, never using target errors."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'notebooks'))
from aion_magnitude.filter_photometry import load_registry,serialize_photometry,feature_columns
from aion_magnitude.filter_metadata_counterfactual import build_conditions,intervene_text,nonmetadata,derangement


@pytest.fixture(scope='module')
def design():
    registry=load_registry();return registry,build_conditions(registry,[9010,9011,9012])


def example(kind,release=None):
    names=[n for _,v,e in feature_columns(kind) for n in (v,e)]+['MW_EBV']
    values=[-1.,.2,np.nan,np.nan,3.,.3,0.,.4,5.,.5,.03]
    return serialize_photometry(dict(zip(names,values)),catalogue=kind,level=2,legacy_release=release)


def test_design_is_fixed_and_label_blind(design):
    registry,d=design
    assert d==build_conditions(registry,[9010,9011,9012])
    assert len(d['conditions'])==13
    assert len(set(d['groups']['legacy']))==8
    assert 'BASS.g' in d['groups']['legacy'] and 'DECam.g' in d['groups']['legacy']


@pytest.mark.parametrize('kind,release',[('galaxiesml-spec',None),('desi-ls',9010),('desi-ls',9011)])
def test_every_intervention_preserves_all_other_bytes(design,kind,release):
    registry,d=design;baseline=example(kind,release)
    for name,condition in d['conditions'].items():
        text=intervene_text(baseline,condition['mapping'])
        assert nonmetadata(text)==nonmetadata(baseline)
        assert 'flux=-1.00000 sigma=0.20000' in text
        assert 'flux=NA sigma=NA' in text
        if name=='correct':assert text==baseline


def test_pair_permutations_preserve_collections_and_have_no_fixed_identity(design):
    registry,d=design
    for name in ['paired-cycle-forward','paired-cycle-reverse','paired-derangement']:
        condition=d['conditions'][name]
        for keys in d['groups'].values():
            expected=sorted(tuple(d['original'][k][f] for f in ['pivot_nm','width_nm']) for k in keys)
            actual=sorted(tuple(condition['mapping'][k][f] for f in ['pivot_nm','width_nm']) for k in keys)
            assert actual==expected
            assert all(condition['mapping'][k]!=d['original'][k] for k in keys)


def test_wavelength_and_width_interventions_are_separate(design):
    registry,d=design
    for name,condition in d['conditions'].items():
        if condition['family'] in ['wavelength','local-wavelength']:
            assert all(v['width_nm']==d['original'][k]['width_nm'] for k,v in condition['mapping'].items())
        if condition['family']=='width':
            assert all(v['pivot_nm']==d['original'][k]['pivot_nm'] for k,v in condition['mapping'].items())
    for key in d['original']:
        assert d['conditions']['pivot-plus-02']['mapping'][key]['pivot_nm']==1.02*d['original'][key]['pivot_nm']


def test_intervention_rejects_extra_fields_or_invalid_units(design):
    registry,d=design;text=example('galaxiesml-spec');mapping={k:v.copy() for k,v in d['original'].items()}
    mapping['HSC.g']['flux']=4.
    with pytest.raises(ValueError):intervene_text(text,mapping)
    mapping['HSC.g']=dict(pivot_nm=-1.,width_nm=1.)
    with pytest.raises(ValueError):intervene_text(text,mapping)
    with pytest.raises(ValueError):intervene_text(nonmetadata(text),d['original'])


def test_placebo_actual_token_lengths_and_shared_wise_mapping(design):
    from transformers import AutoTokenizer
    registry,d=design;mapping=d['conditions']['placebo-length-structure']['mapping']
    tokenizer=AutoTokenizer.from_pretrained('/arc/home/gsm/hf_models/Qwen3.5-4B-Base',local_files_only=True)
    for kind,release in [('galaxiesml-spec',None),('desi-ls',9010),('desi-ls',9011)]:
        text=example(kind,release);counter=intervene_text(text,mapping)
        assert len(tokenizer(text)['input_ids'])==len(tokenizer(counter)['input_ids'])
        assert tokenizer(text)['input_ids']!=tokenizer(counter)['input_ids']
    north=intervene_text(example('desi-ls',9011),mapping).splitlines()
    south=intervene_text(example('desi-ls',9010),mapping).splitlines()
    assert north[-3:]==south[-3:]


def test_exact_replay_rejects_small_prediction_drift():
    import torch
    from filter_metadata_counterfactual import exact_replay
    ref={k:np.ones((4,3),np.float32) if k in ['logits','pz'] else np.ones(4,np.float32) for k in ['logits','pz','z_spec','z_p16','z_p50','z_p84']}
    ref.update(redshift_edges=np.arange(4,dtype=np.float32),redshift_centers=np.arange(3,dtype=np.float32))
    ev={k:torch.from_numpy(v[:2].copy() if not k.startswith('redshift_') else v.copy()) for k,v in ref.items()}
    exact_replay(ev,ref,np.arange(2))
    ev['logits'][0,0]+=.000001
    with pytest.raises(AssertionError):exact_replay(ev,ref,np.arange(2))


def test_identical_fixed_bootstrap_draws_cancel():
    from report_filter_metadata_counterfactual import bootstrap_statistics,difference
    from hsc_legacy_paired_report import metrics
    obj=dict(delta=np.array([-.2,-.1,.01,.2]),ce=np.array([1.,2.,3.,4.]),crps=np.array([.1,.2,.3,.4]),coverage=np.array([0.,1.,1.,0.]),pit=np.array([.1,.3,.6,.9]))
    indices=np.random.default_rng(1).integers(0,4,size=(100,4),dtype=np.uint32)
    keys,draws,bins=bootstrap_statistics(obj,indices,np.array([0,0,1,4]))
    point=metrics(obj,np.arange(4));contrast=difference(point,point,draws,draws,keys)
    assert all(v==0 for v in contrast['estimate'].values())
    assert all(v==[0.,0.] for v in contrast['ci95'].values())


@pytest.mark.parametrize('model,condition,reason',[('C','paired-cycle-forward','Full baseline replay'),('B','paired-cycle-forward','external Level-1')])
def test_full_inference_cannot_bypass_replay_or_apply_interventions_to_B(tmp_path,monkeypatch,model,condition,reason):
    import argparse,json
    import filter_metadata_counterfactual as experiment
    monkeypatch.setattr(experiment,'validate_run',lambda out:({},{}))
    (tmp_path/'metadata_mappings_v1.json').write_text(json.dumps(dict(conditions={'paired-cycle-forward':{}})))
    with pytest.raises(AssertionError,match=reason):
        experiment.infer(argparse.Namespace(output_dir=tmp_path,model=model,condition=condition,domain='both'))
