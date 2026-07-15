#!/usr/bin/env python3
"""Topology-stable Chang--Refsdal channel partitions.

This module contains the production evaluation path for crossing a selected
fold or axis cusp.  It uses only

* the analytic quadratic-field operator total,
* real-image geometry and analytic saddle kernels, and
* algebraic exact residual projections.

The number of computational channels is fixed across the caustic.  Virtual
cluster carriers coincide at the caustic and on the unresolved side, then
smoothly approach the real member-image carriers on the resolved side.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from chang_refsdal_operator import amplification_grid
from chang_refsdal_geometry import (
    critical_point,
    delay,
    find_images,
    image_kernel,
    macro_matrix,
)
from exact_gauge_partition import (
    exact_transition_channels,
    reconstructed_total,
    smootherstep,
)


def _validate_w(w: Sequence[float]) -> np.ndarray:
    arr = np.asarray(w, dtype=float)
    if arr.ndim != 1 or arr.size < 2 or np.any(arr <= 0.0):
        raise ValueError("w must be a positive one-dimensional grid")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError("w must be strictly increasing")
    return arr


def _carrier_free_image_kernels(
    w: np.ndarray,
    images: list[np.ndarray],
    indices: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    if len(indices) == 0:
        return np.empty((len(w), 0), dtype=complex)
    return np.column_stack([image_kernel(w, images[int(j)], matrix) for j in indices])


def _phase_shift_targets(
    w: np.ndarray,
    actual_delays: np.ndarray,
    virtual_delays: np.ndarray,
    actual_kernels: np.ndarray,
) -> np.ndarray:
    """Express physical targets under virtual rather than actual carriers."""
    phase = np.exp(1j * np.multiply.outer(w, actual_delays - virtual_delays))
    return phase * actual_kernels


def _virtual_delays(
    tau_critical: float,
    actual_delays: np.ndarray | None,
    *,
    w_max: float,
    carrier_rho_start: float,
    carrier_rho_end: float,
    n_channels: int,
) -> tuple[np.ndarray, float, float]:
    """Return topology-stable real carriers and their geometry blend.

    Near or outside the caustic all virtual carriers equal ``tau_critical``.
    When the member delays are resolved somewhere in the requested band, they
    approach the actual real-image delays.  The approach is deliberately flatter
    than the physical |distance|^(3/2) splitting at the caustic.
    """
    if actual_delays is None:
        return np.full(n_channels, tau_critical), 0.0, 0.0
    actual = np.asarray(actual_delays, dtype=float)
    spread = float(np.ptp(actual))
    blend = float(smootherstep(w_max * spread, carrier_rho_start, carrier_rho_end))
    center = float(np.mean(actual))
    virtual_center = (1.0 - blend) * tau_critical + blend * center
    virtual = virtual_center + blend * (actual - center)
    return virtual, blend, spread


@dataclass
class TopologyStableGeometry:
    kind: str
    y: np.ndarray
    x_critical: np.ndarray
    tmin: float
    images: list[np.ndarray]
    image_delays: np.ndarray
    persistent_indices: np.ndarray
    physical_cluster_indices: np.ndarray
    tau_critical: float
    virtual_cluster_delays: np.ndarray
    actual_cluster_delays: np.ndarray | None
    inside: bool
    carrier_blend: float
    delay_spread: float


@dataclass
class TopologyStablePartition:
    w: np.ndarray
    exact_total: np.ndarray
    persistent_kernels: np.ndarray
    persistent_delays: np.ndarray
    cluster_kernels: np.ndarray
    cluster_delays: np.ndarray
    physical_switch: np.ndarray
    geometry: TopologyStableGeometry
    operator_orders: np.ndarray
    operator_converged: np.ndarray

    @property
    def reconstructed(self) -> np.ndarray:
        total = np.zeros_like(self.exact_total)
        if self.persistent_kernels.shape[1]:
            total += reconstructed_total(
                self.w, self.persistent_delays, self.persistent_kernels
            )
        total += reconstructed_total(self.w, self.cluster_delays, self.cluster_kernels)
        return total

    @property
    def reconstruction_error(self) -> float:
        return float(np.max(np.abs(self.reconstructed - self.exact_total)))


def _operator_total(
    w: np.ndarray,
    y: np.ndarray,
    gamma: float,
    tmin: float,
    *,
    kappa: float = 0.0,
    beta: float = 0.0,
    operator_tolerance: float,
    operator_max_order: int,
    operator_dps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw, diagnostics = amplification_grid(
        w,
        y,
        gamma,
        kappa=kappa,
        beta=beta,
        tolerance=operator_tolerance,
        max_order=operator_max_order,
        dps=operator_dps,
    )
    total = raw * np.exp(-1j * w * tmin)
    orders = np.array([d.order_used for d in diagnostics], dtype=int)
    converged = np.array([d.converged for d in diagnostics], dtype=bool)
    return total, orders, converged


def build_fold_crossing_partition(
    w: Sequence[float],
    *,
    gamma: float,
    theta_c: float,
    eta_s: float,
    rho_start: float = 0.5,
    rho_end: float = 4.0,
    carrier_rho_start: float = 0.2,
    carrier_rho_end: float = 2.0,
    operator_tolerance: float = 2e-12,
    operator_max_order: int = 42,
    operator_dps: int = 80,
) -> TopologyStablePartition:
    """Fixed two-persistent plus two-cluster partition across a fold.

    ``eta_s`` is the displacement along the soft eigenvector returned for the
    selected critical point.  Either sign is accepted; the image count decides
    which side of the caustic is being evaluated.
    """
    w = _validate_w(w)
    matrix = macro_matrix(gamma)
    xc, yc, _, es, _ = critical_point(gamma, theta_c)
    y = yc + float(eta_s) * es
    images = find_images(y, matrix)
    n_images = len(images)
    if n_images not in (2, 3, 4):
        raise RuntimeError(f"expected 2--4 local fold images, found {n_images}")

    times = np.array([delay(x, y, matrix) for x in images], dtype=float)
    tmin = float(times.min())
    rel_times = times - tmin
    distances = np.array([np.linalg.norm(x - xc) for x in images])

    if n_images == 4:
        cluster_idx = np.argsort(distances)[:2]
        cluster_idx = cluster_idx[np.argsort(rel_times[cluster_idx])]
        persistent_idx = np.array(
            [j for j in range(n_images) if j not in set(cluster_idx)], dtype=int
        )
        persistent_idx = persistent_idx[np.argsort(rel_times[persistent_idx])]
        actual_delays: np.ndarray | None = rel_times[cluster_idx]
        inside = True
    elif n_images == 3:
        # At the fold, the merged critical image is retained in the exact block.
        merged = int(np.argmin(distances))
        cluster_idx = np.array([], dtype=int)
        persistent_idx = np.array([j for j in range(n_images) if j != merged], dtype=int)
        persistent_idx = persistent_idx[np.argsort(rel_times[persistent_idx])]
        actual_delays = None
        inside = False
    else:
        cluster_idx = np.array([], dtype=int)
        persistent_idx = np.argsort(rel_times).astype(int)
        actual_delays = None
        inside = False

    if len(persistent_idx) != 2:
        raise RuntimeError("fold partition must retain exactly two persistent images")

    tau_critical = float(delay(xc, y, matrix) - tmin)
    virtual_delays, carrier_blend, spread = _virtual_delays(
        tau_critical,
        actual_delays,
        w_max=float(w[-1]),
        carrier_rho_start=carrier_rho_start,
        carrier_rho_end=carrier_rho_end,
        n_channels=2,
    )

    exact_total, orders, converged = _operator_total(
        w,
        y,
        gamma,
        tmin,
        operator_tolerance=operator_tolerance,
        operator_max_order=operator_max_order,
        operator_dps=operator_dps,
    )
    persistent_kernels = _carrier_free_image_kernels(
        w, images, persistent_idx, matrix
    )
    persistent_delays = rel_times[persistent_idx]
    persistent_total = reconstructed_total(w, persistent_delays, persistent_kernels)
    cluster_total = exact_total - persistent_total

    if inside and actual_delays is not None:
        actual_kernels = _carrier_free_image_kernels(w, images, cluster_idx, matrix)
        physical_targets = _phase_shift_targets(
            w, actual_delays, virtual_delays, actual_kernels
        )
        physical_switch = smootherstep(w * spread, rho_start, rho_end)
    else:
        physical_targets = np.zeros((len(w), 2), dtype=complex)
        physical_switch = np.zeros(len(w), dtype=float)

    cluster_kernels = exact_transition_channels(
        w,
        cluster_total,
        tau_critical,
        virtual_delays,
        physical_targets,
        physical_switch,
    )

    geometry = TopologyStableGeometry(
        kind="fold",
        y=y,
        x_critical=xc,
        tmin=tmin,
        images=images,
        image_delays=rel_times,
        persistent_indices=persistent_idx,
        physical_cluster_indices=cluster_idx,
        tau_critical=tau_critical,
        virtual_cluster_delays=virtual_delays,
        actual_cluster_delays=actual_delays,
        inside=inside,
        carrier_blend=carrier_blend,
        delay_spread=spread,
    )
    return TopologyStablePartition(
        w=w,
        exact_total=exact_total,
        persistent_kernels=persistent_kernels,
        persistent_delays=persistent_delays,
        cluster_kernels=cluster_kernels,
        cluster_delays=virtual_delays,
        physical_switch=np.asarray(physical_switch, dtype=float),
        geometry=geometry,
        operator_orders=orders,
        operator_converged=converged,
    )


def _orient_cusp_basis(xc: np.ndarray, eh: np.ndarray, es: np.ndarray):
    if float(xc @ eh) < 0.0:
        eh = -eh
    if np.linalg.det(np.column_stack([eh, es])) < 0.0:
        es = -es
    return eh, es


def build_cusp_crossing_partition(
    w: Sequence[float],
    *,
    gamma: float,
    theta_c: float,
    eta_h: float,
    eta_s: float = 0.0,
    rho_start: float = 0.5,
    rho_end: float = 4.0,
    carrier_rho_start: float = 0.2,
    carrier_rho_end: float = 2.0,
    operator_tolerance: float = 2e-12,
    operator_max_order: int = 42,
    operator_dps: int = 80,
) -> TopologyStablePartition:
    """Fixed one-persistent plus three-cluster partition across an axis cusp."""
    w = _validate_w(w)
    matrix = macro_matrix(gamma)
    xc, yc, eh, es, _ = critical_point(gamma, theta_c)
    eh, es = _orient_cusp_basis(xc, eh, es)
    y = yc + float(eta_h) * eh + float(eta_s) * es
    images = find_images(y, matrix)
    n_images = len(images)
    if n_images not in (2, 3, 4):
        raise RuntimeError(f"expected 2--4 local cusp images, found {n_images}")

    times = np.array([delay(x, y, matrix) for x in images], dtype=float)
    tmin = float(times.min())
    rel_times = times - tmin
    distances = np.array([np.linalg.norm(x - xc) for x in images])

    # The image farthest from the cusp is the single persistent image on both
    # sides.  All remaining exact field is assigned to the topology-stable
    # three-channel cusp block.
    persistent_idx = np.array([int(np.argmax(distances))], dtype=int)
    if n_images == 4:
        cluster_idx = np.array(
            [j for j in range(n_images) if j != persistent_idx[0]], dtype=int
        )
        cluster_idx = cluster_idx[np.argsort(rel_times[cluster_idx])]
        actual_delays: np.ndarray | None = rel_times[cluster_idx]
        inside = True
    else:
        cluster_idx = np.array([], dtype=int)
        actual_delays = None
        inside = False

    tau_critical = float(delay(xc, y, matrix) - tmin)
    virtual_delays, carrier_blend, spread = _virtual_delays(
        tau_critical,
        actual_delays,
        w_max=float(w[-1]),
        carrier_rho_start=carrier_rho_start,
        carrier_rho_end=carrier_rho_end,
        n_channels=3,
    )

    exact_total, orders, converged = _operator_total(
        w,
        y,
        gamma,
        tmin,
        operator_tolerance=operator_tolerance,
        operator_max_order=operator_max_order,
        operator_dps=operator_dps,
    )
    persistent_kernels = _carrier_free_image_kernels(
        w, images, persistent_idx, matrix
    )
    persistent_delays = rel_times[persistent_idx]
    persistent_total = reconstructed_total(w, persistent_delays, persistent_kernels)
    cluster_total = exact_total - persistent_total

    if inside and actual_delays is not None:
        actual_kernels = _carrier_free_image_kernels(w, images, cluster_idx, matrix)
        physical_targets = _phase_shift_targets(
            w, actual_delays, virtual_delays, actual_kernels
        )
        physical_switch = smootherstep(w * spread, rho_start, rho_end)
    else:
        physical_targets = np.zeros((len(w), 3), dtype=complex)
        physical_switch = np.zeros(len(w), dtype=float)

    cluster_kernels = exact_transition_channels(
        w,
        cluster_total,
        tau_critical,
        virtual_delays,
        physical_targets,
        physical_switch,
    )

    geometry = TopologyStableGeometry(
        kind="cusp",
        y=y,
        x_critical=xc,
        tmin=tmin,
        images=images,
        image_delays=rel_times,
        persistent_indices=persistent_idx,
        physical_cluster_indices=cluster_idx,
        tau_critical=tau_critical,
        virtual_cluster_delays=virtual_delays,
        actual_cluster_delays=actual_delays,
        inside=inside,
        carrier_blend=carrier_blend,
        delay_spread=spread,
    )
    return TopologyStablePartition(
        w=w,
        exact_total=exact_total,
        persistent_kernels=persistent_kernels,
        persistent_delays=persistent_delays,
        cluster_kernels=cluster_kernels,
        cluster_delays=virtual_delays,
        physical_switch=np.asarray(physical_switch, dtype=float),
        geometry=geometry,
        operator_orders=orders,
        operator_converged=converged,
    )
