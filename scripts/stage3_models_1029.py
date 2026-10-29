"""Predeclared compact numerical controls, all sharing the signed-flux contract."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import stage2_numerical as base
import stage2_physics as pdf
from stage2_data import FILTERS,d
OUT=ROOT/'reports/stage3a/1029'
KPATH=ROOT/'reports/stage2a5/1028b/response_kernel_highres.npz'
KSH='fc0db74392c18fc1423d85bdbb93ea0c95b4da331049fb18c3abf7f68c923fd8'
def representations():
    path=OUT/'representations.npz'
    if path.exists():return d.npz(path)
    assert d.sha(KPATH)==KSH
    k=d.npz(KPATH)['kernel'];full,_,wave=base.response_features();curves=pdf.curves()
    ww=np.geomspace(min(f.wave.min() for f in curves),max(f.wave.max() for f in curves),3600)
    high=[];generic=[];centers=np.linspace(np.log(600),np.log(60000),12);width=1.5*(centers[1]-centers[0]);z=pdf.grid()[1]
    nodes,weights=np.polynomial.legendre.leggauss(3)
    for fil in curves:
        r=np.interp(ww,fil.wave,fil.throughput,left=0,right=0);r/=np.trapezoid(r,np.log(ww));high.append(r)
        lo=fil.wave[:-1];hi=fil.wave[1:];q=((hi+lo)[:,None]/2+(hi-lo)[:,None]/2*nodes).ravel();qw=((hi-lo)[:,None]/2*weights).ravel();rr=np.interp(q,fil.wave,fil.throughput);wt=qw*rr/q;wt/=wt.sum()
        g=np.exp(-.5*((np.log(q[None,:,None]/(1+z[:,None,None]))-centers[None,None,:])/width)**2)
        generic.append(np.einsum('n,znk->zk',wt,g))
    reps=dict(R1=full,R2=k.reshape(21,-1),C2=np.array(high),C3=np.array(generic).reshape(21,-1))
    np.savez_compressed(path,**{k:v.astype(np.float32) for k,v in reps.items()})
    d.write(OUT/'representation_definition.json',dict(FSPS_kernel_sha256=KSH,FSPS_shape=list(k.shape),C2='3600 log-wavelength full photon-response samples over all registered curve support; area normalized',C3='12 fixed positive Gaussian functions in log rest wavelength; centers logspace600..60000 Angstrom; sigma=1.5 center spacings; actual response quadrature; no IGM or external SEDs',C3_centers_A=np.exp(centers).tolist(),generic_integration='Three-point Gaussian quadrature on every actual passband interval',information_caveat='C3 matches dimensions and trainable parameters, not information content; R2 additionally knows FSPS and IGM',encoder_optimization='Encode each of21 fixed filter representations once before gathering valid bands: exactly equivalent to per-observation encoding under masked aggregation'))
    return d.npz(path)
class Model(nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm;reps=representations();kind='R1' if arm in ['R0','C1','R1_brightness'] else arm
        self.register_buffer('responses',torch.tensor(reps[kind]));self.register_buffer('pivots',torch.tensor([pdf.REGISTRY['filters'][b]['pivot_nm'] for b in FILTERS]))
        if arm=='R0':self.mlp=nn.Sequential(nn.Linear(11*(256+6)+1,32),nn.SiLU(),nn.Linear(32,64),nn.SiLU(),nn.Linear(64,300))
        else:
            self.response_encoder=nn.Sequential(nn.Linear(self.responses.shape[1],16),nn.SiLU(),nn.Linear(16,16),nn.SiLU())
            self.band_encoder=nn.Sequential(nn.Linear(22,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
            width=400 if arm=='C1' else 256
            self.head=nn.Sequential(nn.Linear(65,width),nn.SiLU(),nn.Linear(width,300))
    def forward(self,a):
        mask=a['mask'];ids=a['ids'].clamp_min(0)
        x=torch.stack([torch.asinh(a['f']),torch.log(a['e']).clamp(-20,20),torch.asinh(a['f']/a['e']).clamp(-20,20),mask.float(),a['q'],a['qa'].float()],-1)*mask[:,:,None]
        bright=a['brightness'][:,None] if self.arm=='R1_brightness' else torch.zeros_like(a['brightness'][:,None])
        if self.arm=='R0':
            x=torch.cat([x,self.responses[ids]*mask[:,:,None]],-1)
            order=torch.argsort(torch.where(mask,self.pivots[ids],torch.full_like(a['f'],1e10)),dim=1,stable=True)
            x=x.gather(1,order[:,:,None].expand(-1,-1,x.shape[2]));raw=self.mlp(torch.cat([x.flatten(1),bright],1))
        else:
            r=self.response_encoder(self.responses)[ids]
            h=self.band_encoder(torch.cat([x,r],-1));h=(h*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
            raw=self.head(torch.cat([h,bright],1))
        out=4*torch.tanh(raw/4)
        return torch.where((mask.sum(1)>=3)[:,None],out,torch.zeros_like(out))
def tests():
    a=d.npz(ROOT/'reports/stage2a/1028/hsc_trainval.npz');a={k:v[:16] for k,v in a.items()};t=base.tensor_rows(a);results=[]
    for arm in ['R0','R1','R2','C1','C2','C3','R1_brightness']:
        torch.manual_seed(1029);m=Model(arm);p=torch.tensor([3,8,4,1,0,9,2,5,10,6,7]);perm={k:(v[:,p] if v.ndim==2 and v.shape[1]==11 else v) for k,v in t.items()}
        v=m(t);vp=m(perm);assert torch.allclose(v,vp,atol=1e-5),arm
        empty={k:v.clone() for k,v in t.items()};empty['mask'][:]=False;assert torch.all(m(empty)==0)
        scaled={k:v.copy() for k,v in a.items()};scaled['flux']*=1e-5;scaled['sigma']*=1e-5
        scale=float((m(base.tensor_rows(scaled))-v).abs().max().detach())
        if arm!='R1_brightness':assert scale<1e-5
        modified={k:v.clone() for k,v in t.items()};bad=~modified['mask'];modified['f'][bad]=1e9;modified['e'][bad]=1e-9;modified['q'][bad]=99
        assert torch.allclose(m(modified),v,atol=1e-5)
        v.sum().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        results.append(dict(arm=arm,parameters=sum(p.numel() for p in m.parameters()),permutation_max=float((v-vp).abs().max().detach()),scale_max=scale,finite_gradients=True,empty_uniform=True,masked_values_irrelevant=True))
    d.write(OUT/'model_contract_tests.json',dict(passed=True,models=results,input_keys=list(t),kernel_sha256=d.sha(KPATH)))
if __name__=='__main__':tests()
