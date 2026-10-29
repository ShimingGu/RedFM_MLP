"""Physical invariants of the Stage-0.5 interventions; no model training."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage05_interface_diagnostics import texts_for,configurations

NAMES=[f'HSC_PDR2_cmodel_{b}_{suffix}' for b in 'grizy' for suffix in ('flux','flux_error')]+['MW_EBV']


def sample():
    return dict(features=np.array([[1.,.1,2.,.2,-.2,.3,np.nan,np.nan,4.,.4,.035]]))


def test_dimming_preserves_signed_flux_ratios_sn_ebv_and_missingness():
    data=sample();a=10**(-2.8)
    _,x=texts_for(data,NAMES,dict(kind='brightness',a=a),.03,[0,.2],np.array([.04]))
    np.testing.assert_allclose(x[0,:10],a*data['features'][0,:10],equal_nan=True)
    np.testing.assert_allclose(x[:,0:10:2]/x[:,1:10:2],data['features'][:,0:10:2]/data['features'][:,1:10:2],equal_nan=True)
    assert x[0,4]<0 and x[0,-1]==data['features'][0,-1]


def test_uncertainty_reporting_keeps_measurements_and_ebv_fixed():
    data=sample();_,x=texts_for(data,NAMES,dict(kind='uncertainty',b=30),.03,[0,.2],np.array([.04]))
    np.testing.assert_array_equal(x[:,0:10:2],data['features'][:,0:10:2]);assert x[0,-1]==.035
    np.testing.assert_allclose(x[:,1:10:2],30*data['features'][:,1:10:2],equal_nan=True)


def test_noisy_reobservation_adds_only_incremental_variance():
    data=sample();a=.01;b=3;seed=17
    _,x=texts_for(data,NAMES,dict(kind='noisy',a=a,b=b,seed=seed),.03,[0,.2],np.array([.04]))
    orig=data['features'];old=a*orig[:,1:10:2];new=b*old
    expected=a*orig[:,0:10:2]+np.random.default_rng(seed).normal(size=(1,5))*np.sqrt(new**2-old**2)
    np.testing.assert_allclose(x[:,0:10:2],expected,equal_nan=True);np.testing.assert_allclose(x[:,1:10:2],new,equal_nan=True)
    _,identical=texts_for(data,NAMES,dict(kind='noisy',a=a,b=1,seed=seed),.03,[0,.2],np.array([.04]))
    np.testing.assert_allclose(identical[:,:10],a*orig[:,:10],equal_nan=True)


def test_position_marker_and_missing_ebv_do_not_change_photometry():
    data=sample()
    for c in [dict(kind='front'),dict(kind='query',value='true'),dict(kind='ebv',value='missing'),dict(kind='omit')]:
        _,x=texts_for(data,NAMES,c,.03,[0,.2],np.array([.04]));np.testing.assert_array_equal(x[:,:10],data['features'][:,:10])
    config=configurations(.001)
    assert len({c['name'] for c in config})==len(config)
