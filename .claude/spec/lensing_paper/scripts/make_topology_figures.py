#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from chang_refsdal_geometry import critical_point, macro_matrix
from chang_refsdal_topology_stable import build_fold_crossing_partition, build_cusp_crossing_partition
from chang_refsdal_exact_partition import interpolate_complex_linear
from exact_gauge_partition import reconstructed_total

OUT=ROOT/'figures'
OUT.mkdir(parents=True,exist_ok=True)

def profile(a,t):
    floor=.15*np.max(np.abs(t))
    return np.abs(a-t)/np.maximum(np.abs(t),floor)

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

def greedy_curve(fid,cand,mode,max_nodes=20):
    nodes={0,len(fid.w)-1}; ns=[]; errs=[]; snapshots={}
    for _ in range(max_nodes-1):
        idx=np.array(sorted(nodes))
        a=channel_reconstruct(fid,cand,idx) if mode=='channels' else direct_reconstruct(fid,cand,idx)
        e=profile(a,cand.exact_total)
        ns.append(len(idx));errs.append(float(np.max(e)));snapshots[len(idx)]=(idx,a)
        e[idx]=-1
        nodes.add(int(np.argmax(e)))
    idx=np.array(sorted(nodes));a=channel_reconstruct(fid,cand,idx) if mode=='channels' else direct_reconstruct(fid,cand,idx)
    ns.append(len(idx));errs.append(float(np.max(profile(a,cand.exact_total))));snapshots[len(idx)]=(idx,a)
    return np.array(ns),np.array(errs),snapshots

# Figure 1: caustic and selected crossing directions.
gamma=.2
th=np.linspace(0,2*np.pi,1200,endpoint=False)
ys=[]
for t in th:
    _,yc,_,_,_=critical_point(gamma,float(t));ys.append(yc)
ys=np.array(ys)
fig,ax=plt.subplots(figsize=(5.2,4.8))
ax.plot(ys[:,0],ys[:,1],lw=1.6,label='caustic')
for kind,theta,direction_label in [('fold',4.0,'fold path'),('cusp',np.pi,'cusp path')]:
    xc,yc,eh,es,_=critical_point(gamma,theta)
    if kind=='fold': direction=es
    else:
        if xc@eh<0:eh=-eh
        direction=eh
    et=np.linspace(-.025,.025,60)
    path=yc[None,:]+et[:,None]*direction[None,:]
    ax.plot(path[:,0],path[:,1],lw=1.5,label=direction_label)
    ax.scatter([yc[0]],[yc[1]],s=28)
ax.set_xlabel(r'$y_1$')
ax.set_ylabel(r'$y_2$')
ax.set_aspect('equal')
ax.legend(frameon=False)
ax.set_title(r'Chang--Refsdal caustic, $\gamma=0.2$')
fig.tight_layout()
fig.savefig(OUT/'caustic_paths.pdf')
fig.savefig(OUT/'caustic_paths.png',dpi=200)
plt.close(fig)

# Expensive objects reused below.
w=np.linspace(5,40,141)
fold_fid=build_fold_crossing_partition(w,gamma=.2,theta_c=4.,eta_s=-.01,operator_max_order=42,operator_dps=70)
fold_can=build_fold_crossing_partition(w,gamma=.2,theta_c=4.,eta_s=.01,operator_max_order=42,operator_dps=70)
cusp_fid=build_cusp_crossing_partition(w,gamma=.2,theta_c=np.pi,eta_h=-.01,operator_max_order=42,operator_dps=70)
cusp_can=build_cusp_crossing_partition(w,gamma=.2,theta_c=np.pi,eta_h=.01,operator_max_order=42,operator_dps=70)

# Figure 2: ratios are smooth channel-by-channel.
fig,axes=plt.subplots(2,2,figsize=(8.0,5.8),sharex=True)
for row,(name,fid,can) in enumerate([('fold',fold_fid,fold_can),('cusp',cusp_fid,cusp_can)]):
    ratios=[]
    labels=[]
    rp=can.persistent_kernels/fid.persistent_kernels
    rc=can.cluster_kernels/fid.cluster_kernels
    for j in range(rp.shape[1]):ratios.append(rp[:,j]);labels.append(f'persistent {j+1}')
    for j in range(rc.shape[1]):ratios.append(rc[:,j]);labels.append(f'cluster {j+1}')
    for r,lbl in zip(ratios,labels):
        axes[row,0].plot(w,np.abs(r),label=lbl)
        axes[row,1].plot(w,np.unwrap(np.angle(r)))
    axes[row,0].set_ylabel(f'{name}: ratio amplitude')
    axes[row,1].set_ylabel(f'{name}: ratio phase')
axes[1,0].set_xlabel(r'$w$');axes[1,1].set_xlabel(r'$w$')
axes[0,0].legend(frameon=False,ncol=2,fontsize=8)
fig.tight_layout()
fig.savefig(OUT/'channel_ratios.pdf')
fig.savefig(OUT/'channel_ratios.png',dpi=200)
plt.close(fig)

# Figure 3: adaptive error versus node count.
fig,axes=plt.subplots(1,2,figsize=(8.0,3.35),sharey=True)
for ax,(name,fid,can) in zip(axes,[('fold crossing',fold_fid,fold_can),('cusp crossing',cusp_fid,cusp_can)]):
    nc,ec,_=greedy_curve(fid,can,'channels',20)
    nd,ed,_=greedy_curve(fid,can,'direct',20)
    ax.semilogy(nc,ec,marker='o',ms=3,label='channel ratios')
    ax.semilogy(nd,ed,marker='s',ms=3,label='direct total ratio')
    ax.axhline(1e-3,ls='--',lw=1)
    ax.set_xlabel('number of frequency nodes')
    ax.set_title(name)
axes[0].set_ylabel('maximum normalized waveform error')
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT/'adaptive_nodes.pdf')
fig.savefig(OUT/'adaptive_nodes.png',dpi=200)
plt.close(fig)

# Save representative exactness and operator diagnostics.
with (ROOT/'data'/'figure_metrics.txt').open('w') as f:
    for name,p in [('fold fid',fold_fid),('fold candidate',fold_can),('cusp fid',cusp_fid),('cusp candidate',cusp_can)]:
        f.write(f'{name}: reconstruction={p.reconstruction_error:.16e}, max_order={p.operator_orders.max()}, all_converged={bool(np.all(p.operator_converged))}\n')
