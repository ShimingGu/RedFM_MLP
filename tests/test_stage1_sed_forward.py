"""Independent physical/unit checks; no language model or training is needed."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import stage1_sed_forward as f


def test_flat_fnu_all_filters_and_throughput_normalization():
    curves=f.responses(Path(__file__).resolve().parents[1])
    for r in curves.values():
        w=r['wavelength'];fl=1e-28*f.C_AA_S/w**2
        np.testing.assert_allclose(f.band_fnu(w,fl,r['response']),1e-28,rtol=2e-14)
        np.testing.assert_allclose(f.band_fnu(w,fl,7*r['response']),1e-28,rtol=2e-14)
    np.testing.assert_allclose(1e-28/f.NMAGGY_CGS,2.754062241806664,rtol=1e-10)


def test_cosmological_flambda_energy_and_fnu_factors():
    w=np.linspace(1000,15000,20001);llambda=np.exp(-.5*((w-5000)/1000)**2);lnu=llambda*w*w/f.C_AA_S
    z=1.7;distance=2e27;wo=w*(1+z);fo=f.redshift_flambda(wo,w,lnu,z,distance)
    np.testing.assert_allclose(np.trapezoid(fo,wo),np.trapezoid(llambda,w)/(4*np.pi*distance**2),rtol=1e-14)
    const=np.ones_like(w)*1e28;fo=f.redshift_flambda(wo,w,const,z,distance)
    np.testing.assert_allclose(fo*wo**2/f.C_AA_S,(1+z)*1e28/(4*np.pi*distance**2),rtol=1e-14)


def test_top_hat_break_is_integrated_not_pivot_sampled():
    w=np.linspace(5000,7000,200001);r=np.ones_like(w);fnu=np.where(w<6200,1.,2.)
    measured=f.band_fnu(w,fnu*f.C_AA_S/w**2,r)
    expected=(np.log(6200/5000)+2*np.log(7000/6200))/np.log(7000/5000)
    np.testing.assert_allclose(measured,expected,rtol=3e-6)
    assert abs(measured-1)>0.2 and abs(measured-2)>0.2


def test_normalization_removes_only_overall_amplitude_and_preserves_signs():
    flux=np.array([1.,-.1,2.,np.nan]);sigma=np.array([.1,.2,.1,np.nan]);a,b,_=f.normalize_observation(flux,sigma);c,d,_=f.normalize_observation(1e-3*flux,1e-3*sigma)
    np.testing.assert_allclose(a,c,equal_nan=True);np.testing.assert_allclose(b,d,equal_nan=True);assert a[1]<0


def test_igm_is_transparent_redward_of_lyman_and_attenuates_high_z_uv():
    w=np.array([800.,1000.,1200.,1500.]);t=f.igm_transmission(w,4)
    assert np.all((t>=0)&(t<=1));assert t[-1]==1;assert t[0]<t[1]<1


def test_sed_parameters_and_redshifting_preserve_known_break_location():
    p=dict(family='break4000',slope=0.,break_strength=2.,av=0.,equivalent_width=0,log_lnu=28)
    shape=f.sed_shape(p);low=np.interp(3900,f.REST_GRID,shape);high=np.interp(4100,f.REST_GRID,shape)
    assert high/low>1.7


def test_smooth_flat_fnu_has_no_low_z_shape_information():
    p=dict(family='lyman_continuum',slope=0.,break_strength=1.,av=0.,equivalent_width=0.,log_lnu=28.)
    curves=f.responses(Path(__file__).resolve().parents[1]);bands=f.SYSTEMS['HSC']
    a=f.observe_clean(p,.2,bands,curves,None);b=f.observe_clean(p,1.1,bands,curves,None)
    np.testing.assert_allclose(a/np.linalg.norm(a),b/np.linalg.norm(b),rtol=1e-13)
