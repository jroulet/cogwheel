#!/usr/bin/env python3
from __future__ import annotations
import csv
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from chang_refsdal_global_tracking import GlobalChannelTracker
from chang_refsdal_geometry import critical_point
from chang_refsdal_exact_partition import interpolate_complex_linear
from exact_gauge_partition import reconstructed_total

def err(a,t):
    floor=.15*np.max(np.abs(t)); return np.abs(a-t)/np.maximum(np.abs(t),floor)
def rec(fid,cand,idx):
    r=cand.kernels/fid.kernels
    ri=interpolate_complex_linear(fid.w[idx],r[idx],fid.w)
    return reconstructed_total(fid.w,cand.delays,fid.kernels*ri)
def direct(fid,cand,idx):
    r=cand.exact_total/fid.exact_total
    return fid.exact_total*interpolate_complex_linear(fid.w[idx],r[idx],fid.w)
def adapt(fid,cand,mode,tol=1e-3):
    nodes={0,len(fid.w)-1}
    while len(nodes)<len(fid.w):
        idx=np.array(sorted(nodes)); a=rec(fid,cand,idx) if mode=='channels' else direct(fid,cand,idx)
        ee=err(a,cand.exact_total); ee[idx]=-1
        if np.max(err(a,cand.exact_total))<=tol:return len(idx)
        nodes.add(int(np.argmax(ee)))
    return len(fid.w)

def main():
    w=np.linspace(5,40,41); gamma=.12; beta=.18; theta=3.7
    k0=.25
    _,yc,_,es,_=critical_point(gamma,theta,beta,k0)
    y0=yc+.006*es
    tr=GlobalChannelTracker(w,operator_dps=75,operator_max_order=44)
    fid=tr.evaluate(gamma=gamma,kappa=k0,beta=beta,y=y0)
    rows=[]
    for k in [.20,.225,.24,.26,.275,.30]:
        # Keep the same dimensionless position relative to the corresponding
        # caustic under exact mass-sheet scaling.
        lam0=1-k0; lam=1-k
        y=y0*np.sqrt(lam/lam0)
        cand=tr.evaluate(gamma=gamma*lam/lam0,kappa=k,beta=beta,y=y)
        rows.append(dict(kappa=k,gamma=gamma*lam/lam0,n_channels=adapt(fid,cand,'channels'),n_direct=adapt(fid,cand,'direct'),reconstruction_error=cand.reconstruction_error))
    out=ROOT/'data'/'external_convergence_benchmark.csv'
    with out.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print(rows)
if __name__=='__main__':main()
