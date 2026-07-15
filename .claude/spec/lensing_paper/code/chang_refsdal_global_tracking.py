#!/usr/bin/env python3
"""Global four-channel Chang--Refsdal partition with no topology selection.

All parameter points use the same four computational channels.  Real images are
continued between neighboring parameter points by minimum-cost matching.  Empty
channels sit at the nearest critical point and become the newly created images
when a caustic is crossed.  An exact residual projection makes the channel sum
identical to the analytic operator total for every frequency.

No contour integral is used anywhere in this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Sequence
import numpy as np
from scipy.optimize import minimize_scalar

from chang_refsdal_geometry import critical_point, delay, find_images, image_kernel, macro_matrix
from chang_refsdal_topology_stable import _operator_total, _validate_w
from exact_gauge_partition import reconstructed_total, smootherstep


def nearest_caustic_point(gamma: float, beta: float, y: np.ndarray, *, kappa: float = 0.0, n_grid: int = 256):
    y=np.asarray(y,float); grid=np.linspace(0,2*np.pi,n_grid,endpoint=False); step=2*np.pi/n_grid
    vals=[]
    for th in grid:
        _,yc,_,_,_=critical_point(gamma,th,beta,kappa); vals.append(float(np.sum((yc-y)**2)))
    best=None
    for i in np.argsort(vals)[:4]:
        c=grid[i]
        def f(t):
            _,yc,_,_,_=critical_point(gamma,float(t%(2*np.pi)),beta,kappa)
            return float(np.sum((yc-y)**2))
        r=minimize_scalar(f,bounds=(c-step,c+step),method='bounded',options={'xatol':1e-12})
        if best is None or r.fun<best.fun: best=r
    th=float(best.x%(2*np.pi)); xc,yc,eh,es,lam=critical_point(gamma,th,beta,kappa)
    return th,xc,yc,eh,es,lam,float(np.sqrt(best.fun))


def _assign_real_images(prev_markers: np.ndarray | None, images: list[np.ndarray], xc: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Return image index per global channel (-1 for virtual) and markers."""
    n=len(images); X=np.asarray(images,float)
    if prev_markers is None:
        # Initial real labels are sorted by polar angle. Empty labels follow.
        order=np.lexsort((np.linalg.norm(X,axis=1),np.arctan2(X[:,1],X[:,0])))
        assignment=np.full(4,-1,int); assignment[:n]=order
    else:
        scale=max(float(np.median(np.linalg.norm(prev_markers,axis=1))),.3)
        best_cost=np.inf; assignment=None
        # Choose n distinct labels and a permutation of the n images.
        for labels in permutations(range(4),n):
            cost=sum(float(np.sum((X[j]-prev_markers[labels[j]])**2))/scale**2 for j in range(n))
            if cost<best_cost:
                best_cost=cost; assignment=np.full(4,-1,int)
                for j,l in enumerate(labels): assignment[l]=j
    markers=np.tile(xc,(4,1))
    for c,j in enumerate(assignment):
        if j>=0: markers[c]=X[j]
    return assignment,markers


def _exact_four_channel_projection(w,total,delays,physical,switch,weights=None):
    w=np.asarray(w,float); total=np.asarray(total,complex); delays=np.asarray(delays,float)
    H=np.asarray(physical,complex); S=np.asarray(switch,float)
    if weights is None: weights=np.full(4,.25)
    weights=np.asarray(weights,float);weights/=weights.sum()
    tau_ref=float(np.mean(delays))
    Kref=np.exp(-1j*w*tau_ref)*total
    L=np.exp(-1j*np.multiply.outer(w,delays-tau_ref))*weights[None,:]*Kref[:,None]
    trial=(1-S)*L+S*H
    E=np.exp(1j*np.multiply.outer(w,delays))
    residual=total-np.sum(E*trial,axis=1)
    return trial+weights[None,:]*np.conj(E)*residual[:,None]


@dataclass
class GlobalPartition:
    w: np.ndarray; y: np.ndarray; gamma: float; beta: float; kappa: float
    theta_critical: float; x_critical: np.ndarray; y_critical: np.ndarray; caustic_distance: float
    delays: np.ndarray; kernels: np.ndarray; slot_positions: np.ndarray; real_mask: np.ndarray
    exact_total: np.ndarray; operator_orders: np.ndarray; operator_converged: np.ndarray
    @property
    def reconstructed(self): return reconstructed_total(self.w,self.delays,self.kernels)
    @property
    def reconstruction_error(self): return float(np.max(np.abs(self.reconstructed-self.exact_total)))


class GlobalChannelTracker:
    """Continue a universal four-channel gauge along any continuous path."""
    def __init__(self,w:Sequence[float],*,rho_start=.5,rho_end=4.,operator_tolerance=2e-12,
                 operator_max_order=42,operator_dps=80,**_ignored):
        self.w=_validate_w(w);self.rho_start=float(rho_start);self.rho_end=float(rho_end)
        self.operator_tolerance=operator_tolerance;self.operator_max_order=operator_max_order;self.operator_dps=operator_dps
        self._markers=None
    def reset(self): self._markers=None
    def evaluate(self,*,gamma:float,y:Sequence[float],beta:float=0.,kappa:float=0.)->GlobalPartition:
        y=np.asarray(y,float);A=macro_matrix(gamma,beta,kappa)
        th,xc,yc,eh,es,lam,dc=nearest_caustic_point(gamma,beta,y,kappa=kappa)
        imgs=find_images(y,A);tabs=np.array([delay(x,y,A) for x in imgs]);tmin=float(tabs.min());trel=tabs-tmin
        assign,markers=_assign_real_images(self._markers,imgs,xc);self._markers=markers.copy()
        tau_c=float(delay(xc,y,A)-tmin); delays=np.full(4,tau_c); real=np.zeros(4,bool)
        physical=np.zeros((len(self.w),4),complex)
        for c,j in enumerate(assign):
            if j>=0:
                real[c]=True;delays[c]=trel[j];physical[:,c]=image_kernel(self.w,imgs[j],A)
        exact,orders,conv=_operator_total(self.w,y,gamma,tmin,kappa=kappa,beta=beta,operator_tolerance=self.operator_tolerance,
            operator_max_order=self.operator_max_order,operator_dps=self.operator_dps)
        # A real saddle is used as a physical target only when separated in delay
        # from its nearest real neighbor. Newly born/coalescing images therefore
        # remain in the exact artificial gauge automatically.
        switch=np.zeros((len(self.w),4))
        ids=np.flatnonzero(real)
        for c in ids:
            others=ids[ids!=c]
            sep=float(np.min(np.abs(delays[c]-delays[others]))) if len(others) else 0.
            switch[:,c]=smootherstep(self.w*sep,self.rho_start,self.rho_end)
        kernels=_exact_four_channel_projection(self.w,exact,delays,physical,switch)
        return GlobalPartition(self.w,y,gamma,beta,kappa,th,xc,yc,dc,delays,kernels,markers,real,exact,orders,conv)
    def evaluate_path(self,path:Iterable[dict])->list[GlobalPartition]:
        self.reset();return [self.evaluate(**p) for p in path]
