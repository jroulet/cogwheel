#!/usr/bin/env python3
"""Exact gauge partition for a Chang--Refsdal fold cluster.

The full amplification is evaluated by the contour-free shear operator.  The
persistent images are assigned ordinary saddle kernels.  Their difference
from the exact total defines an exact residual fold block.  That block is
artificially partitioned among the fold-pair carriers while unresolved, then
blended toward physical saddle kernels with an algebraic residual projection
that preserves the exact total at every frequency.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from chang_refsdal_operator import amplification_grid
from exact_gauge_partition import (
    exact_cluster_kernel,
    exact_transition_channels,
    reconstructed_total,
    smootherstep,
)
from chang_refsdal_geometry import (
    critical_point, delay, find_images, image_kernel, macro_matrix,
)


@dataclass
class FoldGeometry:
    y: np.ndarray
    x_critical: np.ndarray
    tmin: float
    images: list[np.ndarray]
    delays: np.ndarray
    critical_indices: np.ndarray
    persistent_indices: np.ndarray
    tau_cluster: float


@dataclass
class FoldPartition:
    w: np.ndarray
    exact_total: np.ndarray
    persistent_kernels: np.ndarray
    persistent_delays: np.ndarray
    cluster_kernels: np.ndarray
    cluster_delays: np.ndarray
    switch: np.ndarray
    geometry: FoldGeometry
    operator_orders: np.ndarray

    @property
    def reconstructed(self) -> np.ndarray:
        total = np.zeros_like(self.exact_total)
        if self.persistent_kernels.shape[1]:
            total += reconstructed_total(
                self.w, self.persistent_delays, self.persistent_kernels
            )
        total += reconstructed_total(self.w, self.cluster_delays, self.cluster_kernels)
        return total


def _kernel_without_carrier(
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    matrix: np.ndarray,
    tmin: float,
) -> np.ndarray:
    del y, tmin  # the carrier-free kernel depends only on x and the Hessian
    return image_kernel(w, x, matrix)


def fold_geometry(gamma: float, theta_c: float, eta_s: float) -> FoldGeometry:
    matrix = macro_matrix(gamma)
    xc, yc, _, es, _ = critical_point(gamma, theta_c)
    y = yc + eta_s * es
    images = find_images(y, matrix)
    if len(images) < 4:
        raise ValueError(
            "the member-channel fold partition currently requires a source "
            "on the four-image side of the selected fold"
        )
    times = np.array([delay(x, y, matrix) for x in images])
    tmin = float(times.min())
    distances = np.array([np.linalg.norm(x - xc) for x in images])
    critical = np.argsort(distances)[:2]
    critical = critical[np.argsort(times[critical])]
    persistent = np.array([j for j in range(len(images)) if j not in set(critical)], dtype=int)
    persistent = persistent[np.argsort(times[persistent])]
    member_delays = times[critical] - tmin
    tau_cluster = float(np.mean(member_delays))
    return FoldGeometry(
        y=y,
        x_critical=xc,
        tmin=tmin,
        images=images,
        delays=times - tmin,
        critical_indices=critical,
        persistent_indices=persistent,
        tau_cluster=tau_cluster,
    )


def build_fold_partition(
    w,
    *,
    gamma: float,
    theta_c: float,
    eta_s: float,
    rho_start: float = 0.5,
    rho_end: float = 4.0,
    operator_tolerance: float = 2e-12,
    operator_max_order: int = 36,
    operator_dps: int = 70,
) -> FoldPartition:
    """Build exact persistent plus fold-pair channels on a frequency grid."""
    w = np.asarray(w, dtype=float)
    if w.ndim != 1 or np.any(np.diff(w) <= 0):
        raise ValueError("w must be a strictly increasing one-dimensional grid")
    geometry = fold_geometry(gamma, theta_c, eta_s)
    matrix = macro_matrix(gamma)

    raw_total, diagnostics = amplification_grid(
        w,
        geometry.y,
        gamma,
        tolerance=operator_tolerance,
        max_order=operator_max_order,
        dps=operator_dps,
    )
    exact_total = raw_total * np.exp(-1j * w * geometry.tmin)

    pidx = geometry.persistent_indices
    persistent_delays = geometry.delays[pidx]
    persistent_kernels = np.column_stack([
        _kernel_without_carrier(w, geometry.images[j], geometry.y, matrix, geometry.tmin)
        for j in pidx
    ]) if len(pidx) else np.empty((len(w), 0), dtype=complex)
    persistent_total = (
        reconstructed_total(w, persistent_delays, persistent_kernels)
        if len(pidx) else np.zeros_like(exact_total)
    )

    cidx = geometry.critical_indices
    cluster_delays = geometry.delays[cidx]
    exact_cluster_total = exact_total - persistent_total
    physical_kernels = np.column_stack([
        _kernel_without_carrier(w, geometry.images[j], geometry.y, matrix, geometry.tmin)
        for j in cidx
    ])

    delta_tau = float(abs(cluster_delays[1] - cluster_delays[0]))
    rho = w * delta_tau
    switch = smootherstep(rho, rho_start, rho_end)
    cluster_kernels = exact_transition_channels(
        w,
        exact_cluster_total,
        geometry.tau_cluster,
        cluster_delays,
        physical_kernels,
        switch,
    )

    return FoldPartition(
        w=w,
        exact_total=exact_total,
        persistent_kernels=persistent_kernels,
        persistent_delays=persistent_delays,
        cluster_kernels=cluster_kernels,
        cluster_delays=cluster_delays,
        switch=switch,
        geometry=geometry,
        operator_orders=np.array([d.order_used for d in diagnostics], dtype=int),
    )


def interpolate_complex_linear(nodes_w, nodes_value, target_w):
    """Piecewise-linear interpolation of one or more complex channels."""
    nodes_w = np.asarray(nodes_w, dtype=float)
    values = np.asarray(nodes_value, dtype=complex)
    target_w = np.asarray(target_w, dtype=float)
    if values.ndim == 1:
        return np.interp(target_w, nodes_w, values.real) + 1j*np.interp(
            target_w, nodes_w, values.imag
        )
    columns = [interpolate_complex_linear(nodes_w, values[:, j], target_w)
               for j in range(values.shape[1])]
    return np.column_stack(columns)


def reconstruct_from_interpolated_channels(partition: FoldPartition, node_indices) -> np.ndarray:
    """Interpolate channel kernels at selected nodes and reconstruct F."""
    node_indices = np.asarray(node_indices, dtype=int)
    wn = partition.w[node_indices]
    pk = interpolate_complex_linear(
        wn, partition.persistent_kernels[node_indices], partition.w
    ) if partition.persistent_kernels.shape[1] else partition.persistent_kernels
    ck = interpolate_complex_linear(
        wn, partition.cluster_kernels[node_indices], partition.w
    )
    total = np.zeros_like(partition.exact_total)
    if pk.shape[1]:
        total += reconstructed_total(partition.w, partition.persistent_delays, pk)
    total += reconstructed_total(partition.w, partition.cluster_delays, ck)
    return total


def interpolate_logamp_phase(nodes_w, nodes_value, target_w, amplitude_floor=1e-300):
    """Piecewise-linear interpolation of log amplitude and unwrapped phase."""
    nodes_w = np.asarray(nodes_w, dtype=float)
    values = np.asarray(nodes_value, dtype=complex)
    target_w = np.asarray(target_w, dtype=float)
    if values.ndim == 1:
        logamp = np.log(np.maximum(np.abs(values), amplitude_floor))
        phase = np.unwrap(np.angle(values))
        return np.exp(
            np.interp(target_w, nodes_w, logamp)
            + 1j * np.interp(target_w, nodes_w, phase)
        )
    return np.column_stack([
        interpolate_logamp_phase(nodes_w, values[:, j], target_w, amplitude_floor)
        for j in range(values.shape[1])
    ])


def reconstruct_from_interpolated_channels_polar(partition: FoldPartition, node_indices) -> np.ndarray:
    """Log-amplitude/phase interpolation of channel kernels."""
    node_indices = np.asarray(node_indices, dtype=int)
    wn = partition.w[node_indices]
    pk = interpolate_logamp_phase(
        wn, partition.persistent_kernels[node_indices], partition.w
    ) if partition.persistent_kernels.shape[1] else partition.persistent_kernels
    ck = interpolate_logamp_phase(
        wn, partition.cluster_kernels[node_indices], partition.w
    )
    total = np.zeros_like(partition.exact_total)
    if pk.shape[1]:
        total += reconstructed_total(partition.w, partition.persistent_delays, pk)
    total += reconstructed_total(partition.w, partition.cluster_delays, ck)
    return total


@dataclass
class CuspGeometry:
    y: np.ndarray
    x_critical: np.ndarray
    tmin: float
    images: list[np.ndarray]
    delays: np.ndarray
    critical_indices: np.ndarray
    persistent_indices: np.ndarray
    tau_cluster: float


@dataclass
class CuspPartition:
    w: np.ndarray
    exact_total: np.ndarray
    persistent_kernels: np.ndarray
    persistent_delays: np.ndarray
    cluster_kernels: np.ndarray
    cluster_delays: np.ndarray
    switch: np.ndarray
    geometry: CuspGeometry
    operator_orders: np.ndarray

    @property
    def reconstructed(self) -> np.ndarray:
        total = np.zeros_like(self.exact_total)
        if self.persistent_kernels.shape[1]:
            total += reconstructed_total(self.w, self.persistent_delays, self.persistent_kernels)
        total += reconstructed_total(self.w, self.cluster_delays, self.cluster_kernels)
        return total


def cusp_geometry(
    gamma: float,
    theta_c: float,
    eta_h: float,
    eta_s: float = 0.0,
) -> CuspGeometry:
    """Geometry of a three-image cusp cluster on the four-image side."""
    matrix = macro_matrix(gamma)
    xc, yc, eh, es, _ = critical_point(gamma, theta_c)
    # Fix the otherwise arbitrary eigenvector sign deterministically.
    if float(xc @ eh) < 0.0:
        eh = -eh
    if np.linalg.det(np.column_stack([eh, es])) < 0.0:
        es = -es
    y = yc + eta_h * eh + eta_s * es
    images = find_images(y, matrix)
    if len(images) < 4:
        raise ValueError(
            "the member-channel cusp partition currently requires a source "
            "on the four-image side of the selected cusp"
        )
    times = np.array([delay(x, y, matrix) for x in images])
    tmin = float(times.min())
    distances = np.array([np.linalg.norm(x - xc) for x in images])
    critical = np.argsort(distances)[:3]
    critical = critical[np.argsort(times[critical])]
    persistent = np.array([j for j in range(len(images)) if j not in set(critical)], dtype=int)
    persistent = persistent[np.argsort(times[persistent])]
    member_delays = times[critical] - tmin
    return CuspGeometry(
        y=y,
        x_critical=xc,
        tmin=tmin,
        images=images,
        delays=times - tmin,
        critical_indices=critical,
        persistent_indices=persistent,
        tau_cluster=float(np.mean(member_delays)),
    )


def build_cusp_partition(
    w,
    *,
    gamma: float,
    theta_c: float,
    eta_h: float,
    eta_s: float = 0.0,
    rho_start: float = 0.5,
    rho_end: float = 4.0,
    operator_tolerance: float = 2e-12,
    operator_max_order: int = 36,
    operator_dps: int = 70,
) -> CuspPartition:
    """Build exact persistent plus three-member cusp channels."""
    w = np.asarray(w, dtype=float)
    if w.ndim != 1 or np.any(np.diff(w) <= 0):
        raise ValueError("w must be a strictly increasing one-dimensional grid")
    geometry = cusp_geometry(gamma, theta_c, eta_h, eta_s)
    matrix = macro_matrix(gamma)
    raw_total, diagnostics = amplification_grid(
        w, geometry.y, gamma,
        tolerance=operator_tolerance,
        max_order=operator_max_order,
        dps=operator_dps,
    )
    exact_total = raw_total * np.exp(-1j * w * geometry.tmin)

    pidx = geometry.persistent_indices
    persistent_delays = geometry.delays[pidx]
    persistent_kernels = np.column_stack([
        _kernel_without_carrier(w, geometry.images[j], geometry.y, matrix, geometry.tmin)
        for j in pidx
    ]) if len(pidx) else np.empty((len(w), 0), dtype=complex)
    persistent_total = (
        reconstructed_total(w, persistent_delays, persistent_kernels)
        if len(pidx) else np.zeros_like(exact_total)
    )

    cidx = geometry.critical_indices
    cluster_delays = geometry.delays[cidx]
    exact_cluster_total = exact_total - persistent_total
    physical_kernels = np.column_stack([
        _kernel_without_carrier(w, geometry.images[j], geometry.y, matrix, geometry.tmin)
        for j in cidx
    ])
    delay_spread = float(np.ptp(cluster_delays))
    switch = smootherstep(w * delay_spread, rho_start, rho_end)
    cluster_kernels = exact_transition_channels(
        w, exact_cluster_total, geometry.tau_cluster,
        cluster_delays, physical_kernels, switch,
    )
    return CuspPartition(
        w=w,
        exact_total=exact_total,
        persistent_kernels=persistent_kernels,
        persistent_delays=persistent_delays,
        cluster_kernels=cluster_kernels,
        cluster_delays=cluster_delays,
        switch=switch,
        geometry=geometry,
        operator_orders=np.array([d.order_used for d in diagnostics], dtype=int),
    )
