#!/usr/bin/env python3
"""Focused topology-crossing benchmark using only analytic runtime components."""
from __future__ import annotations
import csv
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from chang_refsdal_topology_stable import build_fold_crossing_partition, build_cusp_crossing_partition
from chang_refsdal_exact_partition import interpolate_complex_linear
from exact_gauge_partition import reconstructed_total


def profile(a,t):
    floor=.15*np.max(np.abs(t))
    return np.abs(a-t)/np.maximum(np.abs(t),floor)

def rms(a,t):
    return float(np.sqrt(np.sum(np.abs(a-t)**2)/np.sum(np.abs(t)**2)))

def channel_reconstruct(fid,cand,idx):
    w=fid.w; wn=w[idx]; out=np.zeros_like(cand.exact_total)
    rp=cand.persistent_kernels/fid.persistent_kernels
    rip=interpolate_complex_linear(wn,rp[idx],w)
    out+=reconstructed_total(w,cand.persistent_delays,fid.persistent_kernels*rip)
    rc=cand.cluster_kernels/fid.cluster_kernels
    ric=interpolate_complex_linear(wn,rc[idx],w)
    out+=reconstructed_total(w,cand.cluster_delays,fid.cluster_kernels*ric)
    return out

def direct_reconstruct(fid,cand,idx):
    r=cand.exact_total/fid.exact_total
    ri=interpolate_complex_linear(fid.w[idx],r[idx],fid.w)
    return fid.exact_total*ri

def adapt(fid,cand,tol,mode):
    nodes={0,len(fid.w)-1}
    while len(nodes)<len(fid.w):
        idx=np.array(sorted(nodes))
        a=channel_reconstruct(fid,cand,idx) if mode=='channels' else direct_reconstruct(fid,cand,idx)
        e=profile(a,cand.exact_total)
        e[idx]=-1.0
        k=int(np.argmax(e))
        full=profile(a,cand.exact_total)
        if np.max(full)<=tol:
            return idx,a,float(np.max(full))
        nodes.add(k)
    idx=np.arange(len(fid.w));a=channel_reconstruct(fid,cand,idx) if mode=='channels' else direct_reconstruct(fid,cand,idx)
    return idx,a,float(np.max(profile(a,cand.exact_total)))

def main():
    root=ROOT/'data';w=np.linspace(5,40,71);tol=1e-3
    configs={
      'fold': (lambda e:build_fold_crossing_partition(w,gamma=.2,theta_c=4.,eta_s=e,operator_max_order=42,operator_dps=70)),
      'cusp': (lambda e:build_cusp_crossing_partition(w,gamma=.2,theta_c=np.pi,eta_h=e,operator_max_order=42,operator_dps=70)),
    }
    eta_values=[-.02,-.01,-.005,-.001,0.,.001,.005,.01,.02]
    cache={}
    for kind,builder in configs.items():
      for eta in eta_values:
        print('build',kind,eta,flush=True);cache[(kind,eta)]=builder(eta)
    rows=[]
    pairs=[]
    for kind in configs:
      for fid in (-.01,.01):
        for cand in eta_values:
          if cand==fid:continue
          pairs.append((kind,fid,cand))
    for kind,fid_eta,cand_eta in pairs:
      fid=cache[(kind,fid_eta)];cand=cache[(kind,cand_eta)]
      for mode in ('channels','direct'):
        idx,a,m=adapt(fid,cand,tol,mode)
        rows.append({
          'kind':kind,'eta_fid':fid_eta,'eta_candidate':cand_eta,
          'sides':('in' if fid.geometry.inside else 'out')+'->'+('in' if cand.geometry.inside else 'out'),
          'method':mode,'target':tol,'n_nodes':len(idx),'achieved_max':m,'rms':rms(a,cand.exact_total),
          'fid_exact_error':fid.reconstruction_error,'cand_exact_error':cand.reconstruction_error,
          'min_fid_cluster_abs':float(np.min(np.abs(fid.cluster_kernels))),
          'min_cand_cluster_abs':float(np.min(np.abs(cand.cluster_kernels))),
          'all_operator_converged':bool(np.all(fid.operator_converged) and np.all(cand.operator_converged)),
        })
    csvp=root/'topology_crossing_benchmark.csv'
    with csvp.open('w',newline='') as f:
      wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print(csvp)
if __name__=='__main__':main()
