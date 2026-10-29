"""Scientific invariants for paired-cohort selection and exposure checks."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('paired', Path(__file__).parents[1]/'notebooks/hsc_legacy_paired_benchmark.py')
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)


def test_positional_exposure_wraps_ra_and_handles_poles():
    assert paired.nearby(np.array([359.99995]), np.array([0.]), np.array([.00005]), np.array([0.]))[0]
    assert paired.nearby(np.array([0.]), np.array([89.99999]), np.array([180.]), np.array([89.99999]))[0]
    assert not paired.nearby(np.array([0.]), np.array([0.]), np.array([.001]), np.array([0.]))[0]


def test_training_exposure_takes_precedence_and_ambiguity_rejects_all():
    assert paired.categories(np.array([True,False,False]), np.array([True,True,False])).tolist()==['T','V','P']
    assert paired.unique_mask(np.array([4,4,5])).tolist()==[False,False,True]


def test_redshift_merges_preserve_all_objects_including_upper_edge():
    z=np.r_[np.full(60,.1),np.full(5,.3),np.full(45,.5),np.full(2,4.5)]
    bins=paired.fixed_bins(z)
    assert bins['edges']==[0.,.2,4.5]
    assert sum(bins['counts'])==len(z)
    assert min(bins['counts'])>=40


def test_support_uses_source_only_and_requires_real_uncertainties():
    rng=np.random.default_rng(7)
    source=np.zeros((1000,11),np.float32)
    for j in range(5):
        source[:,2*j]=np.exp(rng.normal(1,.4,1000))
        source[:,2*j+1]=np.exp(rng.normal(-2,.1,1000))
    hsc=source[:50].copy();legacy=hsc.copy();legacy[:,4:6]=hsc[:,6:8]
    inside,audit=paired.fit_support(source,hsc,legacy)
    assert inside.any()
    legacy[inside,1]=np.nan
    outside,audit2=paired.fit_support(source,hsc,legacy)
    assert not outside[inside].any()
    assert audit['lower']==audit2['lower'] and audit['upper']==audit2['upper']


def test_paired_bootstrap_preserves_zero_difference_for_identical_observations():
    from hsc_legacy_paired_report import paired_summary
    obj=dict(delta=np.array([-.3,0.,.01,.2]),ce=np.array([2.,1.,3.,4.]),crps=np.array([.2,.1,.3,.4]),
             coverage=np.array([0.,1.,1.,0.]),pit=np.array([.1,.4,.5,.9]))
    result=paired_summary(obj,obj,np.arange(4),replicates=100)
    assert all(v==0 for v in result['delta'].values())
    assert all(v==[0.,0.] for v in result['ci95']['delta'].values())
    assert result['transitions']['nonoutlier_to_outlier']==0
    assert result['transitions']['outlier_to_nonoutlier']==0


def test_analysis_reproduces_existing_metrics_at_grid_boundaries():
    import torch
    from aion_magnitude.metrics import predict_photoz_from_logits,summarize_pdf_metrics
    from hsc_legacy_paired_report import object_metrics,metrics
    logits=torch.tensor([[1.,2.,3.],[0.,-1.,2.],[3.,2.,1.],[1.,0.,2.]])
    edges=torch.tensor([0.,.1,.2,.3]);centers=(edges[:-1]+edges[1:])/2
    ev=dict(logits=logits,z_spec=torch.tensor([0.,.1,.2,.3]),redshift_edges=edges,redshift_centers=centers,
            **predict_photoz_from_logits(logits,centers))
    expected=summarize_pdf_metrics(ev)
    actual=metrics(object_metrics({k:v.numpy() for k,v in ev.items()}),np.arange(4))
    mapping=dict(ce='cross_entropy',nmad='nmad',outlier='catastrophic_outlier_fraction',bias='bias',
                 median_bias='median_bias',crps='mean_crps',coverage68='p16_p84_coverage',pit_mean='pit_mean')
    for a,b in mapping.items():
        np.testing.assert_allclose(actual[a],expected[b],rtol=1e-6,atol=1e-7)


def test_photometric_diagnostics_preserve_negative_flux_without_inventing_magnitude():
    from hsc_legacy_paired_report import photometric_summary
    features=np.array([[-1.,.2,-1.,.2,-1.,.2,-1.,.2,-1.,.2,.03]])
    summary=photometric_summary(features,['g','r','z','W1','W2'])
    assert summary['g']['flux']['n']==1
    assert summary['g']['missing']==0
    assert summary['g']['apparent_ab_magnitude']['n']==0
    assert summary['g']['nominal_5sigma_ab_depth']['q05_q50_q95']==[22.5]*3
