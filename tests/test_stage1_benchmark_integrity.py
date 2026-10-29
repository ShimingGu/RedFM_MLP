"""Leakage and paired-counterfactual checks on the frozen synthetic manifest."""
from pathlib import Path
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import stage1_sed_forward as f

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/stage1/1027'


def records():
    return [json.loads(l) for l in (OUT/'records.jsonl').read_text().splitlines()]


def test_no_sed_identity_or_target_filter_supervision_leakage():
    rec=records();groups={s:{r['sed_id'] for r in rec if r['split']==s} for s in ('train','validation','test','family_holdout')}
    for s,ids in groups.items():
        for t,other in groups.items():
            if s!=t:assert not (ids&other)
    assert {r['system'] for r in rec if r['split']=='train'}=={'HSC','Legacy'}
    assert all(r['family']!='double_break_emission' for r in rec if r['split']=='train')
    source={b for r in rec if r['split']=='train' for b in r['bands']}
    assert not (source&set(f.SYSTEMS['new_filters']))


def test_wrong_metadata_has_identical_measurements_redshift_and_order():
    rec=records();a=np.load(OUT/'photometry.npz');lookup={(r['latent_id'],r['system'],r['variant']):r for r in rec}
    for wrong in [r for r in rec if r['variant']=='wrong_metadata']:
        true=lookup[wrong['latent_id'],wrong['system'],'correct'];i=wrong['index'];j=true['index']
        np.testing.assert_array_equal(a['raw_flux_nanomaggy'][i],a['raw_flux_nanomaggy'][j]);np.testing.assert_array_equal(a['raw_sigma_nanomaggy'][i],a['raw_sigma_nanomaggy'][j])
        assert wrong['z']==true['z'] and wrong['sed_id']==true['sed_id'] and wrong['order_seed']==true['order_seed']
        assert wrong['metadata_bands']!=true['metadata_bands']


def test_all_supervised_input_brightness_is_normalized_and_labels_absent():
    a=np.load(OUT/'photometry.npz');rms=np.sqrt(np.nanmean(a['normalized_flux']**2,axis=1));np.testing.assert_allclose(rms,1.,rtol=1e-13)
    for line in (OUT/'inputs.jsonl').read_text().splitlines():
        text=json.loads(line)['text'];assert 'sed_id' not in text and 'family=' not in text and 'z=' not in text and 'seed=' not in text
    assert np.nanmin(a['raw_flux_nanomaggy'])<0


def test_redshift_counterfactual_is_recomputed_through_response_curves():
    rec=records();shift=next(r for r in rec if r['variant']=='redshift_plus03');params=json.loads((OUT/'sed_parameters.json').read_text());p=params[shift['sed_id']]
    expected=f.observe_clean(p,shift['z'],shift['bands'],f.responses(ROOT),OUT/'templates');a=np.load(OUT/'photometry.npz')
    np.testing.assert_allclose(expected,a['clean_flux_nanomaggy'][shift['index'],:len(expected)],rtol=1e-13)
