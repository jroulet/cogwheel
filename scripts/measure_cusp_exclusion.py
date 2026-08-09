#!/usr/bin/env python
"""Measure the FARFIELD_KERNEL_SUM envelope turn-on distance from cusp vertices.

For each (gamma, parity, cusp angle) configuration, sweeps source-plane
positions along cusp rays outward from the cusp vertex and measures at what
physical distance the normalised envelope magnitude |E|/max|F| first drops
below a 1e-4 threshold (10× tighter than the 1e-3 accuracy bar, providing
margin for spline error + reconstruction floor).  The conservative maximum
across all configs becomes the calibrated ``_CUSP_EXCLUSION_DISTANCE``.

Coverage:
- gamma values: 0.2, 0.4, 0.6, 0.8, 0.92 (astroid, parity==1)
- gamma values: 1.1, 1.5, 2.0 (deltoid, parity==-1)
- w = w_floor (worst-case cancellation, computed per-config)
- cusp rays: the physical source-plane directions of each cusp vertex

Usage
-----
    python scripts/measure_cusp_exclusion.py
"""
from __future__ import annotations

import math

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels,
    farfield_envelope_from_partition,
    farfield_w_floor,
    FARFIELD_KERNEL_SUM,
)

_GAMMA_VALUES_ASTROID = [0.2, 0.4, 0.6, 0.8, 0.92]
_GAMMA_VALUES_DELTOID = [1.1, 1.5, 2.0]

_N_CUSP_SWEEP = 200

_START_DISTANCE = 0.02
_END_DISTANCE = 1.5
_N_SWEEP = 100
_TURN_ON_THRESHOLD = 1e-4


def _astroid_cusp_angles(gamma: float) -> list[float]:
    """Astroid cusp source-plane angles measured from origin (rad)."""
    from cogwheel.lensing.surrogate_training import _cusp_source_angles
    return _cusp_source_angles(gamma, _N_CUSP_SWEEP)


def _deltoid_cusp_positions(gamma: float) -> list[tuple[float, float]]:
    """Deltoid cusp physical source-plane positions for one gamma.

    Returns the physical (y1, y2) positions of all deltoid cusps,
    computed by sweeping both saddle branches around lens-plane 0 and π.
    """
    from cogwheel.lensing.surrogate_training import (
        _branch_speed_profile, _find_cusps,
        _SADDLE_CUSP_WIDTH_SAFETY, _SADDLE_CUSP_MIN_HALFWIDTH,
    )
    theta_max = 0.5 * np.arcsin(1.0 / abs(gamma))
    positions: list[tuple[float, float]] = []
    for lens_center in (0.0, math.pi):
        for branch in (1, -1):
            lo = lens_center - theta_max
            hi = lens_center + theta_max
            thetas, speed = _branch_speed_profile(
                gamma, branch, lo, hi, _N_CUSP_SWEEP, periodic=False)
            for theta_lens, _delta in _find_cusps(
                    thetas, speed, periodic=False, gamma=gamma, branch=branch,
                    width_safety=_SADDLE_CUSP_WIDTH_SAFETY,
                    min_halfwidth=_SADDLE_CUSP_MIN_HALFWIDTH):
                try:
                    src = geometry.critical_point(
                        gamma, float(theta_lens), 0.0, 0.0, branch).source
                except geometry.LensDomainError:
                    continue
                positions.append((float(src[0]), float(src[1])))
    return positions


def _measure_envelope(gamma: float, parity: int, y1: float, y2: float,
                      w: float) -> float | None:
    """Evaluate |E|/max|F| at one source-plane position.

    Returns None if the engine refuses the configuration.
    """
    try:
        channels = ChangRefsdalChannels(np.array([w, w * 1.001]))
        partition = channels.evaluate(
            gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
    except (geometry.LensDomainError, RuntimeError):
        return None
    w_floor_val = farfield_w_floor(partition.delays, partition.real_mask)
    if w_floor_val is None or w < w_floor_val:
        return None
    e_ff = farfield_envelope_from_partition(
        partition, definition=FARFIELD_KERNEL_SUM)
    env_mag = float(np.max(np.abs(e_ff)))
    f_mag = float(np.max(np.abs(partition.exact_total)))
    if f_mag == 0.0:
        return None
    return env_mag / f_mag


def _sweep_cusp_ray(gamma: float, parity: int,
                    cusp_pos: tuple[float, float],
                    direction: tuple[float, float],
                    w: float) -> float:
    """Sweep outward from cusp_pos along direction, return turn-on distance.

    Returns the physical source-plane distance at which |E|/max|F| first
    drops below ``_TURN_ON_THRESHOLD``.  If the ratio is already below the
    threshold at the start distance, returns ``_START_DISTANCE``.  If it
    never drops below, returns ``_END_DISTANCE``.
    """
    cx, cy = cusp_pos
    dx, dy = direction
    dists = np.linspace(_START_DISTANCE, _END_DISTANCE, _N_SWEEP)
    for dist in dists:
        y1 = float(cx + dist * dx)
        y2 = float(cy + dist * dy)
        ratio = _measure_envelope(gamma, parity, y1, y2, w)
        if ratio is not None and ratio < _TURN_ON_THRESHOLD:
            return float(dist)
    return float(_END_DISTANCE)


def main() -> None:
    print("gamma,parity,cusp_angle,position,turn_on_distance")
    results: dict[str, list[float]] = {}

    # -- Astroid (positive parity) --
    def _w_floor(gamma: float) -> float:
        """farfield_w_floor at a near-caustic source (delays, real_mask)."""
        try:
            partition = ChangRefsdalChannels(
                np.array([10.0, 10.01])).evaluate(
                    gamma=gamma, y=(0.5, 0.0), beta=0.0, kappa=0.0)
            return float(farfield_w_floor(
                partition.delays, partition.real_mask) or 10.0)
        except (geometry.LensDomainError, RuntimeError):
            return 10.0

    for gamma in _GAMMA_VALUES_ASTROID:
        cusp_angles = _astroid_cusp_angles(gamma)
        if not cusp_angles:
            print(f"# gamma={gamma}: no cusps found", flush=True)
            continue
        w_floor = _w_floor(gamma)
        for angle in cusp_angles:
            try:
                r = geometry.r_caustic(gamma, angle)
            except geometry.LensDomainError:
                continue
            cusp_pos = (float(r * math.cos(angle)),
                        float(r * math.sin(angle)))
            direction = (math.cos(angle), math.sin(angle))
            dist = _sweep_cusp_ray(
                gamma, 1, cusp_pos, direction, w_floor)
            key = f"gamma={gamma:.2f},parity=1,angle={angle:.4f}"
            results.setdefault(key, []).append(dist)
            print(f"{gamma:.2f},1,{angle:.4f},{cusp_pos},{dist:.6f}")

    # -- Deltoid (saddle parity) --
    for gamma in _GAMMA_VALUES_DELTOID:
        cusp_positions = _deltoid_cusp_positions(gamma)
        if not cusp_positions:
            print(f"# gamma={gamma}: no deltoid cusps found", flush=True)
            continue
        w_floor = _w_floor(gamma)
        for pos in cusp_positions:
            y1, y2 = pos
            angle = math.atan2(y2, y1)
            mag = math.hypot(y1, y2)
            if mag < 1e-12:
                continue
            direction = (y1 / mag, y2 / mag)
            dist = _sweep_cusp_ray(
                gamma, -1, pos, direction, w_floor)
            print(f"{gamma:.2f},-1,{angle:.4f},{pos},{dist:.6f}")

    if results:
        max_dist = max(
            max(vals) for vals in results.values())
        print(f"\n# Conservative maximum turn-on distance: {max_dist:.6f}")
    else:
        print("\n# No results collected.")


if __name__ == '__main__':
    main()
