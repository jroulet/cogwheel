#!/usr/bin/env python
"""Measure the minimum angular reach of the Pearcey cusp arm.

Sweeps representative (gamma, w) pairs and, at each, finds the smallest
angular offset delta_theta from the cusp vertex at which the Pearcey
scaled radius R = hypot(x, y) reaches R_min (the refusal boundary).
The MINIMUM reach across the sweep is the coverage constant to set in
surrogate.py as ``_CUSP_ARM_COVERAGE``.

Also performs a boundary verification: at the worst-case point, confirms
that cusp_amplification serves (non-None) and agrees with F_op to within
the F016 bar (5%).

Usage
-----
    python scripts/measure_cusp_arm_reach.py

Prints the measured minimum angular reach and the worst-case (gamma, w)
pair.

References
----------
- cogwheel/lensing/chang_refsdal/_pearcey_cusp.py (the cusp arm)
- cogwheel/lensing/surrogate.py (_CUSP_ARM_COVERAGE)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal._pearcey_cusp import (
    _soft_normal_form,
    cusp_amplification,
    use_pearcey_table,
    _UNIFORM_ERROR_CONST,
    _DEFAULT_ENVELOPE_BAR,
)
from cogwheel.lensing.chang_refsdal.operator import F_op

# R_min: the arm refuses when R < R_min.
_R_MIN = (_UNIFORM_ERROR_CONST / _DEFAULT_ENVELOPE_BAR) ** (2.0 / 3.0)

# Sweep grid.
_GAMMA_VALUES = [0.1, 0.3, 0.5, 1.2, 1.5]
_W_VALUES = [5.0, 10.0, 20.0, 40.0, 60.0]


def _cusp_theta(gamma: float) -> float:
    """Return the cusp angle for beta=0, kappa=0.

    Positive parity (gamma < 1): astroid cusps at theta=0, pi/2, pi, 3pi/2.
    Saddle (gamma > 1): wedge-tip cusp at theta=0.
    Both: use theta=0.
    """
    return 0.0


def _get_cusp_frame(gamma: float):
    """Get the cusp vertex, its local frame, and C4.

    Returns (vertex: CriticalPoint, c4: float, abs_c4: float) or None
    on any geometry refusal.
    """
    theta_cusp = _cusp_theta(gamma)
    branch = 1
    try:
        vertex = geometry.critical_point(gamma, theta_cusp, beta=0.0,
                                         kappa=0.0, branch=branch)
    except geometry.LensDomainError:
        return None

    matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
    normal_form = _soft_normal_form(vertex.image, matrix, vertex.soft_axis,
                                    vertex.hard_axis, vertex.hard_eigenvalue)
    if normal_form is None:
        return None
    c4, _phi_ssr = normal_form
    return vertex, c4, abs(c4)


def _radius_at_delta_theta(gamma: float, w: float, abs_c4: float,
                           vertex, delta_theta: float) -> float:
    """Compute the scaled Pearcey radius R at angular offset delta_theta.

    Steps delta_theta away from the cusp along the critical curve, maps
    the caustic-point offset onto the soft/hard axes, and computes the
    Pearcey controls (x, y) -> R.
    """
    theta_cusp = _cusp_theta(gamma)
    branch = 1
    try:
        pt = geometry.critical_point(gamma, theta_cusp + delta_theta,
                                     beta=0.0, kappa=0.0, branch=branch)
    except geometry.LensDomainError:
        return 0.0  # Outside domain -> arm cannot serve -> R=0

    offset = pt.source - vertex.source
    delta_parallel = float(offset @ vertex.soft_axis)
    delta_perp = float(offset @ vertex.hard_axis)

    x = delta_parallel * math.sqrt(w) / math.sqrt(abs_c4)
    y = delta_perp * w ** 0.75 / abs_c4 ** 0.25
    return math.hypot(x, y)


def _find_reach(gamma: float, w: float) -> float | None:
    """Find the minimum delta_theta at which R >= R_min.

    Uses bisection on delta_theta in [0, pi/4] (positive parity) or
    [0, half_wedge] (saddle).  Returns the delta_theta reach, or None
    if the cusp frame is degenerate.
    """
    frame = _get_cusp_frame(gamma)
    if frame is None:
        return None
    vertex, c4, abs_c4 = frame

    # Upper bound on the search: for positive parity, pi/4 is well
    # short of the next cusp.  For saddle, use the half-wedge width.
    lam = 1.0
    if abs(gamma) >= lam:
        # Saddle: half-wedge width.
        try:
            theta_max = 0.5 * math.asin(lam / abs(gamma))
        except ValueError:
            return None
        upper = 0.9 * theta_max  # Stay inside the wedge.
    else:
        upper = math.pi / 4.0

    # Verify that R at upper > R_min (the arm can actually serve there).
    r_upper = _radius_at_delta_theta(gamma, w, abs_c4, vertex, upper)
    if r_upper < _R_MIN:
        # At the upper bound the arm still can't serve; the reach
        # exceeds our search range.  Return the upper bound as a
        # conservative (too large) estimate.
        return upper

    # Verify that R at delta_theta=0 < R_min (the arm refuses at cusp).
    r_zero = _radius_at_delta_theta(gamma, w, abs_c4, vertex, 1e-10)
    if r_zero >= _R_MIN:
        # Already above threshold at the cusp — reach is ~0.
        return 0.0

    # Bisect to find the crossing.
    def objective(dt):
        return _radius_at_delta_theta(gamma, w, abs_c4, vertex, dt) - _R_MIN

    reach = brentq(objective, 1e-10, upper, xtol=1e-8, rtol=1e-8)
    return reach


def _verify_boundary(gamma: float, w: float, reach: float) -> dict:
    """Verify the arm serves at the boundary and agrees with F_op.

    Returns a dict with verification results.
    """
    results = {
        'gamma': gamma,
        'w': w,
        'reach': reach,
        'arm_serves_at_boundary': False,
        'arm_refuses_at_cusp': False,
        'relative_error': float('nan'),
        'passes_f016': False,
        'verification_partial': False,
    }

    frame = _get_cusp_frame(gamma)
    if frame is None:
        results['verification_partial'] = True
        return results
    vertex, c4, abs_c4 = frame

    # Source at the boundary: step by `reach` from the cusp.
    theta_cusp = _cusp_theta(gamma)
    try:
        pt_boundary = geometry.critical_point(
            gamma, theta_cusp + reach, beta=0.0, kappa=0.0, branch=1)
    except geometry.LensDomainError:
        results['verification_partial'] = True
        return results
    source_at_boundary = pt_boundary.source

    # Load the Pearcey table.
    table_ok = use_pearcey_table()
    if not table_ok:
        results['verification_partial'] = True

    # Verify arm REFUSES at cusp (delta_theta ~ 0).
    arm_at_cusp = cusp_amplification(w, vertex.source, gamma)
    results['arm_refuses_at_cusp'] = (arm_at_cusp is None)

    # Verify arm SERVES at boundary.
    arm_at_boundary = cusp_amplification(w, source_at_boundary, gamma)
    if arm_at_boundary is None:
        # Try slightly beyond the boundary.
        for extra in (0.001, 0.005, 0.01):
            try:
                pt_extra = geometry.critical_point(
                    gamma, theta_cusp + reach + extra,
                    beta=0.0, kappa=0.0, branch=1)
            except geometry.LensDomainError:
                continue
            arm_at_boundary = cusp_amplification(w, pt_extra.source, gamma)
            if arm_at_boundary is not None:
                source_at_boundary = pt_extra.source
                break
        if arm_at_boundary is None:
            results['verification_partial'] = True
            return results

    results['arm_serves_at_boundary'] = True

    # Compare with F_op.
    try:
        f_op_value, _diag = F_op(w, source_at_boundary, gamma)
    except Exception:
        results['verification_partial'] = True
        return results

    if abs(f_op_value) > 0.0:
        rel_err = abs(arm_at_boundary - f_op_value) / abs(f_op_value)
        results['relative_error'] = rel_err
        results['passes_f016'] = (rel_err < 0.05)

    return results


def main() -> None:
    """Run the full measurement and verification."""
    print(f'R_min = ({_UNIFORM_ERROR_CONST} / {_DEFAULT_ENVELOPE_BAR})'
          f'^(2/3) = {_R_MIN:.6f}')
    print(f'Gamma values: {_GAMMA_VALUES}')
    print(f'w values: {_W_VALUES}')
    print()

    min_reach = float('inf')
    worst_gamma = None
    worst_w = None
    all_reaches: list[tuple[float, float, float]] = []

    for gamma in _GAMMA_VALUES:
        for w in _W_VALUES:
            reach = _find_reach(gamma, w)
            if reach is None:
                print(f'  gamma={gamma:.2f}, w={w:5.1f}: '
                      f'SKIPPED (degenerate frame)')
                continue
            all_reaches.append((gamma, w, reach))
            if reach < min_reach:
                min_reach = reach
                worst_gamma = gamma
                worst_w = w
            print(f'  gamma={gamma:.2f}, w={w:5.1f}: '
                  f'reach = {reach:.6f} rad')

    print()
    if worst_gamma is None:
        print('ERROR: No valid (gamma, w) pairs found.')
        return

    print(f'Minimum angular reach: {min_reach:.6f} rad '
          f'at gamma={worst_gamma}, w={worst_w}')
    print()

    # Part B: Boundary verification at the worst-case point.
    print('--- Boundary verification ---')
    results = _verify_boundary(worst_gamma, worst_w, min_reach)
    print(f'  Arm refuses at cusp (R<R_min): {results["arm_refuses_at_cusp"]}')
    print(f'  Arm serves at boundary: {results["arm_serves_at_boundary"]}')
    print(f'  Relative error vs F_op: {results["relative_error"]:.4e}')
    print(f'  Passes F016 (<5%): {results["passes_f016"]}')
    if results['verification_partial']:
        print('  NOTE: Verification was partial (some checks skipped).')
    print()
    print(f'==> Set _CUSP_ARM_COVERAGE = {min_reach:.6f} in surrogate.py')


if __name__ == '__main__':
    main()
