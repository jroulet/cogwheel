#!/usr/bin/env python3
"""Exact gauge partitions for unresolved Chang--Refsdal image clusters.

The functions in this module never approximate the supplied total amplification.
They only choose a computational decomposition

    F = sum_a exp(i w tau_a) K_a

whose individual channels can be made smooth while a fold/cusp cluster is
unresolved.  The total amplification may be supplied by the analytic
quadratic-field operator representation.  A contour integral is not required.
"""
from __future__ import annotations

import numpy as np


def smootherstep(x: np.ndarray | float, x0: float, x1: float):
    """C2 switch from zero at x<=x0 to one at x>=x1."""
    x = np.asarray(x, dtype=float)
    u = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    return u**3 * (10.0 - 15.0*u + 6.0*u*u)


def exact_cluster_kernel(
    w,
    total_amplification,
    persistent_total,
    tau_cluster,
):
    """Demodulated exact residual associated with an unresolved cluster.

    By definition,

        total = persistent_total + exp(i w tau_cluster) K_cluster.
    """
    w = np.asarray(w, dtype=float)
    total_amplification = np.asarray(total_amplification, dtype=complex)
    persistent_total = np.asarray(persistent_total, dtype=complex)
    return np.exp(-1j*w*tau_cluster) * (total_amplification - persistent_total)


def unresolved_member_channels(
    w,
    cluster_kernel,
    tau_cluster,
    member_delays,
    weights=None,
):
    """Artificially split one exact unresolved cluster among member carriers.

    Returns K_j satisfying

        sum_j exp(i w tau_j) K_j = exp(i w tau_cluster) K_cluster

    identically.  Equal weights are the safest fold/cusp gauge choice.
    """
    w = np.asarray(w, dtype=float)
    cluster_kernel = np.asarray(cluster_kernel, dtype=complex)
    tau = np.asarray(member_delays, dtype=float)
    n = tau.size
    if weights is None:
        weights = np.full(n, 1.0/n)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    phase = np.exp(-1j*np.multiply.outer(w, tau - tau_cluster))
    if w.ndim == 0:
        phase = np.exp(-1j*w*(tau-tau_cluster))
        return weights * phase * cluster_kernel
    return phase * weights[None, :] * cluster_kernel[:, None]


def exact_transition_channels(
    w,
    total_cluster,
    tau_cluster,
    member_delays,
    physical_kernels,
    switch,
    weights=None,
):
    """Blend artificial unresolved channels into physical image channels exactly.

    ``physical_kernels[j]`` is the saddle/asymptotic target H_j.  A residual
    projection is added so that the sum of channels remains equal to the
    supplied exact cluster total at every w, including through the transition.
    The correction vanishes when the physical approximation becomes exact.
    """
    w = np.asarray(w, dtype=float)
    total_cluster = np.asarray(total_cluster, dtype=complex)
    tau = np.asarray(member_delays, dtype=float)
    H = np.asarray(physical_kernels, dtype=complex)
    S = np.asarray(switch, dtype=float)
    n = tau.size
    if weights is None:
        weights = np.full(n, 1.0/n)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)

    Kc = np.exp(-1j*w*tau_cluster) * total_cluster
    L = unresolved_member_channels(w, Kc, tau_cluster, tau, weights)

    if w.ndim == 0:
        trial = (1.0-S)*L + S*H
        E = np.exp(1j*w*tau)
        residual = total_cluster - np.sum(E*trial)
        return trial + weights*np.exp(-1j*w*tau)*residual

    trial = (1.0-S[:, None])*L + S[:, None]*H
    E = np.exp(1j*np.multiply.outer(w, tau))
    residual = total_cluster - np.sum(E*trial, axis=1)
    return trial + weights[None, :]*np.conj(E)*residual[:, None]


def reconstructed_total(w, member_delays, kernels):
    """Reconstruct sum_j exp(i w tau_j) K_j."""
    w = np.asarray(w, dtype=float)
    tau = np.asarray(member_delays, dtype=float)
    K = np.asarray(kernels, dtype=complex)
    if w.ndim == 0:
        return np.sum(np.exp(1j*w*tau)*K)
    return np.sum(np.exp(1j*np.multiply.outer(w, tau))*K, axis=1)
