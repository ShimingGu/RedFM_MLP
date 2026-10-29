"""Paper II frozen-backbone readout study: shared paths and exact identities."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'notebooks'),str(ROOT)]
from stage2_data import d,FILTERS,RAW,make_rows,take
OUT=ROOT/'reports/stage4/1029'
BASE=Path('/arc/home/gsm/hf_models/Qwen3.5-4B-Base')
HISTORY=Path('/arc/home/gsm/aion_output/figures')
BS=HISTORY/'qwen-galaxiesml-spec-qlora-specz-e10/qlora'
BC=HISTORY/'qwen-qwen_posttraining_comparison-e10/qlora'
OLD_CAT=Path('/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological_multiband.fits')
TASKS=['GMS','C_SPEC','C_ZPHOT'];BACKBONES=['B0','BS','BC']
def order(ids,tag):
    return np.argsort([hashlib.sha256(f'{tag}:{x}'.encode()).hexdigest() for x in ids],kind='stable')
def xyz(ra,dec):
    r=np.deg2rad(ra);de=np.deg2rad(dec)
    return np.c_[np.cos(de)*np.cos(r),np.cos(de)*np.sin(r),np.sin(de)]
def nearest(a,b,k=1):
    from scipy.spatial import cKDTree
    dd,ix=cKDTree(xyz(b['ra'],b['dec'])).query(xyz(a['ra'],a['dec']),k=k,workers=2)
    return np.rad2deg(2*np.arcsin(np.clip(dd/2,0,1)))*3600,ix
def ids(x):return np.asarray(x).astype('U30')
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()
