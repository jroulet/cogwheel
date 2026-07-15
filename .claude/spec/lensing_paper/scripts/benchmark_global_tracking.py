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
    r=cand.kernels/fid.kernels;ri=interpolate_complex_linear(fid.w[idx],r[idx],fid.w)
    return reconstructed_total(fid.w,cand.delays,fid.kernels*ri)
def direct(fid,cand,idx):
    r=cand.exact_total/fid.exact_total;return fid.exact_total*interpolate_complex_linear(fid.w[idx],r[idx],fid.w)
def adapt(fid,cand,mode,tol=1e-3):
    nodes={0,len(fid.w)-1}
    while len(nodes)<len(fid.w):
        idx=np.array(sorted(nodes));a=rec(fid,cand,idx) if mode=='channels' else direct(fid,cand,idx)
        ee=err(a,cand.exact_total);ee[idx]=-1
        if np.max(err(a,cand.exact_total))<=tol:return len(idx)
        nodes.add(int(np.argmax(ee)))
    return len(fid.w)
def main():
    w=np.linspace(5,40,31); ts=np.linspace(0,2*np.pi,49)
    path=[]
    for t in ts:
        g=.2+.012*np.sin(t); b=.035*np.sin(2*t)
        _,yc,_,_,_=critical_point(g,t,b)
        # Follow just outside the moving caustic, with a small transverse wobble.
        y=(1.0+0.08*np.sin(2*t))*yc + .0015*np.array([np.cos(3*t),np.sin(3*t)])
        path.append(dict(gamma=g,beta=b,y=y))
    tr=GlobalChannelTracker(w,cusp_angle=.16,operator_dps=65,operator_max_order=42)
    parts=tr.evaluate_path(path);rows=[]
    for i,(a,b) in enumerate(zip(parts[:-1],parts[1:])):
        rows.append(dict(step=i,nreal_a=int(a.real_mask.sum()),nreal_b=int(b.real_mask.sum()),n_channels=adapt(a,b,'channels'),n_direct=adapt(a,b,'direct'),
          reconstruction_error=b.reconstruction_error,max_delay_step=float(np.max(np.abs(b.delays-a.delays))),
          max_marker_step=float(np.max(np.linalg.norm(b.slot_positions-a.slot_positions,axis=1)))))
    out=ROOT/'data'/'global_tracking_benchmark.csv'
    with out.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print('steps',len(rows),'switches',sum(r['nreal_a']!=r['nreal_b'] for r in rows))
    print('max reconstruction',max(r['reconstruction_error'] for r in rows))
    print('channel nodes median/max',np.median([r['n_channels'] for r in rows]),max(r['n_channels'] for r in rows))
    print('direct nodes median/max',np.median([r['n_direct'] for r in rows]),max(r['n_direct'] for r in rows))
    print('max delay step',max(r['max_delay_step'] for r in rows))
    print('max marker step',max(r['max_marker_step'] for r in rows))
if __name__=='__main__':main()
