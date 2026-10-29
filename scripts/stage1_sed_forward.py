"""Controlled SED/photometry forward model; all wavelength arguments are Angstrom.

Photon throughput R uses <fnu> = int(flambda*lambda*R dlambda) /
(c*int(R/lambda dlambda)). Toy attenuation is explicitly not a fitted survey model.
"""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from scipy.special import expit

C_AA_S=2.99792458e18
NMAGGY_CGS=3.631e-29
REST_GRID=np.geomspace(500.,150000.,7000)
SYSTEMS={
    'HSC':tuple(f'HSC.{b}' for b in 'grizy'),
    'Legacy':('DECam.g','DECam.r','DECam.z','WISE.W1','WISE.W2'),
    'CLAUDS':('MegaCam.u','MegaCam.u_star',*[f'HSC.{b}' for b in 'grizy'],'VIRCAM.Y','VIRCAM.J','VIRCAM.H','VIRCAM.Ks'),
    'new_filters':('MegaCam.u','MegaCam.u_star','VIRCAM.Y','VIRCAM.J','VIRCAM.H','VIRCAM.Ks'),
}


def responses(root):
    reg=json.loads((Path(root)/'configs/filter_registry_v1.json').read_text());out={}
    for name in sorted({b for bands in SYSTEMS.values() for b in bands}):
        entry=reg['filters'][name];raw=np.loadtxt(Path(root)/'configs'/entry['curve_path'],delimiter=',',skiprows=1)
        # Include actual knots and an oversampling grid; narrow line sigma is 20 A.
        lam=np.unique(np.r_[raw[:,0]*10,np.arange(raw[0,0]*10,raw[-1,0]*10,5.)]);r=np.interp(lam,raw[:,0]*10,raw[:,1])
        norm=np.trapezoid(r/lam,lam);assert norm>0
        cdf=np.cumsum((r/lam)*np.gradient(lam));cdf/=cdf[-1]
        out[name]=dict(wavelength=lam,response=r,normalization=norm,pivot=entry['pivot_nm']*10,width=entry['width_nm']*10,quantiles=np.interp([.1,.5,.9],cdf,lam))
    return out


def band_fnu(wavelength,flambda,response):
    w=np.asarray(wavelength,float);f=np.asarray(flambda,float);r=np.asarray(response,float)
    assert w.shape==f.shape==r.shape and np.all(np.diff(w)>0)
    return np.trapezoid(f*w*r,w)/(C_AA_S*np.trapezoid(r/w,w))


def igm_transmission(rest_wave,z):
    """Limited mean-screen toy: Lyman-series-like opacity, plus ionizing suppression.

    The coefficients follow the familiar power-law mean-line-opacity form, but
    the continuum prescription is deliberately simplified, not Madau/Inoue code.
    """
    rest=np.asarray(rest_wave,float);obs=rest*(1+z);tau=np.zeros_like(rest)
    if z>1.5:
        for lam,coefficient in [(1216.,.0036),(1026.,.0017),(973.,.0012),(950.,.00093)]:
            tau+=np.where((obs>=lam)&(rest<lam),coefficient*(obs/lam)**3.46,0.)
    t=np.exp(-tau)
    t*=np.where(rest<912.,.01*np.exp(-((1+z)/3)**3),1.)
    return t


def sed_shape(p,wave=REST_GRID,template_dir=None):
    """Emergent Lnu shape, normalized at 5500 A after controlled attenuation."""
    wave=np.asarray(wave,float);family=p['family'];slope=p['slope'];strength=p['break_strength']
    if family=='empirical':
        raw=np.loadtxt(Path(template_dir)/p['template']);x=raw[:,0];y=np.maximum(raw[:,1]*x*x,1e-40)
        # The vendored spectra are extended CWW templates. Any remaining endpoint
        # extension uses a documented clipped power-law slope, never a target fit.
        logx=np.log(x);logy=np.log(y);value=np.interp(np.log(wave),logx,logy)
        left=np.clip((logy[1]-logy[0])/(logx[1]-logx[0]),-3,3);right=np.clip((logy[-1]-logy[-2])/(logx[-1]-logx[-2]),-3,3)
        value=np.where(wave<x[0],logy[0]+left*(np.log(wave)-logx[0]),value)
        value=np.where(wave>x[-1],logy[-1]+right*(np.log(wave)-logx[-1]),value)
        value=np.exp(value)*np.power(wave/5500.,slope*.3)
    else:
        value=np.power(wave/5500.,slope)
        if family in ('break4000','weak_break','double_break_emission'):
            value*=1+(strength-1)*expit((wave-4000.)/40.)
        if family=='double_break_emission':
            value*=1+.7*expit((wave-3646.)/30.)
            for lam,ratio in [(3727.,1),(4861.,.7),(5007.,1.4),(6563.,1.5)]:
                value*=1+p['equivalent_width']*ratio/(np.sqrt(2*np.pi)*20)*np.exp(-.5*((wave-lam)/20)**2)
    value*=np.exp(-(p['av']/1.086)*np.power(wave/5500.,-.7))
    return value/np.interp(5500.,wave,value)


@lru_cache(maxsize=4096)
def luminosity_distance(z):
    from astropy.cosmology import Planck18
    return Planck18.luminosity_distance(z).to_value('cm')


def redshift_flambda(observed_wave,rest_wave,rest_lnu,z,distance_cm):
    """Lnu -> observed flambda, with consistent luminosity-distance factors."""
    rest=observed_wave/(1+z)
    lnu=np.interp(rest,rest_wave,rest_lnu)
    llambda=lnu*C_AA_S/rest**2
    return llambda/(4*np.pi*distance_cm**2*(1+z))


def observe_clean(p,z,bands,curves,template_dir,log_lnu=None):
    lnu=sed_shape(p,template_dir=template_dir)*10**(p['log_lnu'] if log_lnu is None else log_lnu);distance=luminosity_distance(float(z));values=[]
    for b in bands:
        r=curves[b];w=r['wavelength'];assert w.min()/(1+z)>=REST_GRID.min() and w.max()/(1+z)<=REST_GRID.max()
        f=redshift_flambda(w,REST_GRID,lnu,z,distance)*igm_transmission(w/(1+z),z)
        values.append(band_fnu(w,f,r['response'])/NMAGGY_CGS)
    return np.array(values)


def add_noise(clean,sn,rng):
    """Independent Gaussian sky-equivalent errors, exposure set by a nuisance S/N.

    The same RMS S/N distribution is used at every redshift. This is a controlled
    observation model, not a calibrated survey exposure/image-noise simulator.
    """
    sigma=np.full_like(clean,np.sqrt(np.mean(clean**2))/sn)
    return clean+rng.normal(size=len(clean))*sigma,sigma


def normalize_observation(flux,sigma):
    scale=np.sqrt(np.nanmean(flux**2));assert np.isfinite(scale) and scale>0
    return flux/scale,sigma/scale,scale
