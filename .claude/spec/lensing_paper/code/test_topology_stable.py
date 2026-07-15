from __future__ import annotations
import numpy as np

from chang_refsdal_topology_stable import (
    build_fold_crossing_partition,
    build_cusp_crossing_partition,
)


def test_fold_fixed_channel_count_and_exactness_both_sides():
    w = np.linspace(5.0, 20.0, 9)
    for eta in (-1e-2, 0.0, 1e-2):
        part = build_fold_crossing_partition(
            w, gamma=.2, theta_c=4.0, eta_s=eta,
            operator_max_order=42, operator_dps=70,
        )
        assert part.persistent_kernels.shape == (len(w), 2)
        assert part.cluster_kernels.shape == (len(w), 2)
        assert part.reconstruction_error < 2e-14
        assert np.all(part.operator_converged)


def test_cusp_fixed_channel_count_and_exactness_both_sides():
    w = np.linspace(5.0, 20.0, 9)
    for eta in (-1e-2, 0.0, 1e-2):
        part = build_cusp_crossing_partition(
            w, gamma=.2, theta_c=np.pi, eta_h=eta,
            operator_max_order=42, operator_dps=70,
        )
        assert part.persistent_kernels.shape == (len(w), 1)
        assert part.cluster_kernels.shape == (len(w), 3)
        assert part.reconstruction_error < 2e-14
        assert np.all(part.operator_converged)


def test_channels_are_continuous_at_fold_crossing():
    w = np.array([5.0, 10.0, 20.0, 40.0])
    parts = [
        build_fold_crossing_partition(
            w, gamma=.2, theta_c=4.0, eta_s=eta,
            operator_max_order=42, operator_dps=70,
        )
        for eta in (-1e-5, 0.0, 1e-5)
    ]
    for left, right in zip(parts[:-1], parts[1:]):
        rel = np.max(
            np.abs(right.cluster_kernels-left.cluster_kernels)
            / np.maximum(np.abs(left.cluster_kernels), 1e-12)
        )
        assert rel < 2e-4


def test_channels_are_continuous_at_cusp_crossing():
    w = np.array([5.0, 10.0, 20.0, 40.0])
    parts = [
        build_cusp_crossing_partition(
            w, gamma=.2, theta_c=np.pi, eta_h=eta,
            operator_max_order=42, operator_dps=70,
        )
        for eta in (-1e-5, 0.0, 1e-5)
    ]
    for left, right in zip(parts[:-1], parts[1:]):
        rel = np.max(
            np.abs(right.cluster_kernels-left.cluster_kernels)
            / np.maximum(np.abs(left.cluster_kernels), 1e-12)
        )
        assert rel < 1e-4

from chang_refsdal_geometry import (
    find_images_quartic,
    lens_residual,
    macro_matrix,
)


def test_quartic_solver_has_small_residual_on_general_sources():
    rng = np.random.default_rng(20260714)
    for _ in range(30):
        gamma = float(rng.uniform(0.02, 0.65))
        beta = float(rng.uniform(-np.pi / 2, np.pi / 2))
        matrix = macro_matrix(gamma, beta)
        radius = float(10 ** rng.uniform(-2.5, 0.0))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        y = radius * np.array([np.cos(theta), np.sin(theta)])
        images = find_images_quartic(y, matrix)
        assert len(images) in (2, 4)
        assert max(np.linalg.norm(lens_residual(x, y, matrix)) for x in images) < 1e-9


def test_quartic_solver_centered_source_has_four_images_for_nonzero_shear():
    matrix = macro_matrix(0.2, 0.37)
    images = find_images_quartic(np.zeros(2), matrix)
    assert len(images) == 4
    assert max(np.linalg.norm(lens_residual(x, np.zeros(2), matrix)) for x in images) < 1e-12

from chang_refsdal_global_tracking import GlobalChannelTracker


def test_global_tracker_exact_on_path_crossing_fold_and_cusp_regions():
    w = np.array([5.0, 10.0, 20.0])
    tracker = GlobalChannelTracker(w, operator_dps=65, operator_max_order=42)
    # A closed source-space path that passes near a cusp and generic folds.
    phi = np.linspace(0, 2*np.pi, 17)
    path = [dict(gamma=.2, beta=.0, y=.18*np.array([np.cos(t), .75*np.sin(t)])) for t in phi]
    parts = tracker.evaluate_path(path)
    assert all(p.kernels.shape == (len(w),4) for p in parts)
    assert max(p.reconstruction_error for p in parts) < 3e-13
    assert all(np.all(p.operator_converged) for p in parts)


def test_global_tracker_labels_are_continuous_under_small_parameter_steps():
    w = np.array([5.0, 10.0, 20.0])
    tracker = GlobalChannelTracker(w, operator_dps=65, operator_max_order=42)
    path = [dict(gamma=.2+2e-4*k, beta=.002*k, y=np.array([.12+.001*k,.035-.0005*k])) for k in range(8)]
    parts = tracker.evaluate_path(path)
    for a,b in zip(parts[:-1],parts[1:]):
        # Carrier motion is continuous and no label makes an order-unity jump.
        assert np.max(np.abs(b.delays-a.delays)) < .03
        assert np.max(np.linalg.norm(b.slot_positions-a.slot_positions,axis=1)) < .15
