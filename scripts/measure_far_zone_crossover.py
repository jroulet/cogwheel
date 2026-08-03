#!/usr/bin/env python
"""Step 4: geometric tiling-coverage verification at the C8 boundary.

Verifies that tube + far-field tiling has no angular or radial gap at the
exclusion_rho boundary for all representative gammas (both parities).

For each gamma + parity, sweeps angular directions and checks whether
each direction is covered by (a) the tube chart (arc coverage) or
(b) the far-field chart (nearest-caustic distance >= eta_max). Gaps
near cusps are expected (handled by exact-engine fallthrough) and are
measured and reported.

Output: per-gamma coverage summary with explicit gap reporting.

Usage:
    conda run -n $SDK_CONDA_ENV python scripts/measure_far_zone_crossover.py
"""
from __future__ import annotations

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cogwheel.lensing.surrogate_training import (
    _coordinate_radius_bounds,
    _min_curvature_radius,
    _astroid_arcs,
    _saddle_arcs,
)
from cogwheel.lensing.chang_refsdal import geometry

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Representative gamma values (positive parity: gamma < 1).
GAMMAS_POSITIVE: list[float] = [0.05, 0.1, 0.2, 0.4, 0.7]

#: Representative gamma values (macro-saddle parity: gamma > 1).
GAMMAS_SADDLE: list[float] = [1.1, 1.3, 1.5, 2.0]

#: Fraction of curvature radius defining tube coverage depth (from
#: TrainingConfig default).
F_MAX: float = 0.40

#: Number of angular samples in the caustic profile scan.
N_CAUSTIC_SAMPLES: int = 200

#: Number of uniform sweep angles in [-pi, pi].
N_SWEEP_ANGLES: int = 36

#: Band half-width for gamma sweeps (matches TrainingConfig default).
GAMMA_BAND_HW: float = 0.01


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _is_in_cusp_window(theta: float, cusp_windows: tuple) -> bool:
    """Check whether theta falls within any cusp exclusion window."""
    for tc, hw in cusp_windows:
        delta = _wrap_angle(theta - tc)
        if abs(delta) < hw:
            return True
    return False


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _theta_in_arc(theta: float, arc) -> bool:
    """Check whether theta is within the arc's angular span.

    Uses modular arithmetic to handle wrap-around for periodic arcs
    (positive parity astroid) where theta_hi can exceed theta_lo by
    more than 2*pi.
    """
    lo = arc.theta_lo
    hi = arc.theta_hi
    # Map theta into the arc's frame: find the representative of theta
    # in [lo, lo + 2*pi) and check if it's <= hi.
    theta_in_frame = lo + (theta - lo) % (2.0 * math.pi)
    return theta_in_frame <= hi


def _tube_covers(theta: float, arcs: list) -> bool:
    """Check if the tube chart covers this angular direction.

    The tube covers theta if it falls within any arc's angular span
    AND is not inside a cusp exclusion window of that arc.
    """
    for arc in arcs:
        if _theta_in_arc(theta, arc):
            if not _is_in_cusp_window(theta, arc.cusp_windows):
                return True
    return False


def _source_at_boundary(
    theta_source: float, exclusion_rho: float, coord_radius_min: float
) -> np.ndarray:
    """Compute the source-plane position at rho = exclusion_rho.

    The far-field chart coordinate is additive:
        rho = 1 + |y| - coord_radius_min
    so |y| = (exclusion_rho - 1) + coord_radius_min.
    This is angle-independent (coord_radius_min is the band's worst-case).
    """
    y_mag = (exclusion_rho - 1.0) + coord_radius_min
    return np.array([y_mag * math.cos(theta_source),
                     y_mag * math.sin(theta_source)])


def _farfield_covers(
    gamma: float, theta_source: float, exclusion_rho: float,
    coord_radius_min: float, eta_max: float
) -> bool:
    """Check whether the far-field chart admits at the boundary rho.

    The far-field admits when the nearest-caustic distance exceeds eta_max.
    Uses geometry.nearest_caustic_point for an exact check.
    """
    source = _source_at_boundary(theta_source, exclusion_rho, coord_radius_min)
    try:
        ncp = geometry.nearest_caustic_point(gamma, 0.0, source)
        return ncp.distance >= eta_max
    except geometry.LensDomainError:
        # If geometry refuses, the far-field cannot confirm coverage
        return False


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    """Run geometric coverage verification."""
    print("=" * 70)
    print("GEOMETRIC TILING-COVERAGE VERIFICATION (Step 4)")
    print("=" * 70)
    print(f"  Positive gammas: {GAMMAS_POSITIVE}")
    print(f"  Saddle gammas:   {GAMMAS_SADDLE}")
    print(f"  f_max:           {F_MAX}")
    print(f"  Sweep angles:    {N_SWEEP_ANGLES} uniform + cusp angles")
    print(flush=True)

    all_results: list[dict] = []

    # Process both parities
    for parity, gammas in [(1, GAMMAS_POSITIVE), (-1, GAMMAS_SADDLE)]:
        parity_label = "positive" if parity == 1 else "saddle"
        print(f"\n{'─' * 70}")
        print(f"  PARITY: {parity_label}")
        print(f"{'─' * 70}")

        for gamma in gammas:
            print(f"\n  gamma = {gamma:.2f} ({parity_label}):")

            # Step a: get arcs and reach
            if parity == 1:
                cusps, arcs, _reach = _astroid_arcs(gamma, N_CAUSTIC_SAMPLES)
            else:
                cusps, arcs, _reach = _saddle_arcs(gamma, N_CAUSTIC_SAMPLES)

            if not arcs:
                print(f"    WARNING: no arcs found for gamma={gamma}")
                continue

            # Step b: compute R_c per arc
            band = (gamma - GAMMA_BAND_HW, gamma + GAMMA_BAND_HW)
            r_c_values = []
            for arc in arcs:
                r_c = _min_curvature_radius(band, arc, N_CAUSTIC_SAMPLES)
                r_c_values.append(r_c)

            r_c_max = max(r_c_values)
            r_c_min = min(r_c_values)

            # Step c: tube coverage depth (max eta across all arcs)
            eta_max_max = F_MAX * r_c_max

            # Step d: far-field floor / exclusion_rho
            coord_radius_min, reach_max = _coordinate_radius_bounds(band, parity)
            exclusion_rho = 1.0 + (reach_max + eta_max_max) - coord_radius_min

            print(f"    arcs: {len(arcs)}, cusps: {len(cusps)}")
            print(f"    R_c range: [{r_c_min:.4f}, {r_c_max:.4f}]")
            print(f"    eta_max (max across arcs): {eta_max_max:.4f}")
            print(f"    coord_radius_min: {coord_radius_min:.4f}")
            print(f"    reach_max: {reach_max:.4f}")
            print(f"    exclusion_rho: {exclusion_rho:.4f}")

            # Step e: angular sweep
            # Uniform angles plus cusp angles for fine probing
            uniform_thetas = np.linspace(
                -math.pi, math.pi, N_SWEEP_ANGLES, endpoint=False)
            cusp_thetas = np.array([tc for tc, _hw in cusps])
            # Also probe just outside cusp windows
            cusp_edge_thetas = []
            for tc, hw in cusps:
                cusp_edge_thetas.append(tc + hw + 0.01)
                cusp_edge_thetas.append(tc - hw - 0.01)
            all_thetas = np.unique(np.concatenate([
                uniform_thetas, cusp_thetas, np.array(cusp_edge_thetas)]))

            gaps: list[tuple[float, str]] = []
            for theta in all_thetas:
                tube_ok = _tube_covers(theta, arcs)
                ff_ok = _farfield_covers(
                    gamma, theta, exclusion_rho, coord_radius_min,
                    eta_max_max)

                if not tube_ok and not ff_ok:
                    # Determine reason
                    in_cusp = any(
                        abs(_wrap_angle(theta - tc)) < hw
                        for tc, hw in cusps)
                    reason = "cusp_window" if in_cusp else "gap"
                    gaps.append((theta, reason))

            # Report
            n_cusp_gaps = sum(1 for _, r in gaps if r == "cusp_window")
            n_true_gaps = sum(1 for _, r in gaps if r == "gap")

            if gaps:
                print(f"    GAPS: {len(gaps)} "
                      f"({n_cusp_gaps} cusp-window, {n_true_gaps} other)")
                for theta, reason in gaps:
                    print(f"      theta={theta:+.4f} rad "
                          f"({math.degrees(theta):+.1f} deg) [{reason}]")
            else:
                print("    COVERED: no gap at C8 boundary")

            all_results.append({
                'gamma': gamma,
                'parity': parity_label,
                'exclusion_rho': exclusion_rho,
                'n_gaps': len(gaps),
                'n_cusp_gaps': n_cusp_gaps,
                'n_true_gaps': n_true_gaps,
                'gaps': gaps,
            })

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_cusp_gaps = sum(r['n_cusp_gaps'] for r in all_results)
    total_true_gaps = sum(r['n_true_gaps'] for r in all_results)

    for r in all_results:
        status = "OK" if r['n_gaps'] == 0 else f"{r['n_gaps']} gaps"
        print(f"  gamma={r['gamma']:.2f} ({r['parity']:>8s}): "
              f"rho*={r['exclusion_rho']:.3f}  {status}")

    print()
    if total_true_gaps == 0 and total_cusp_gaps == 0:
        print("COVERAGE VERIFIED: no gamma/theta gap at the C8 boundary.")
    elif total_true_gaps == 0:
        print(f"COVERAGE VERIFIED (modulo {total_cusp_gaps} cusp-window gaps "
              f"handled by exact-engine fallthrough).")
    else:
        print(f"WARNING: {total_true_gaps} non-cusp gaps detected. "
              f"Tiling may have angular holes at the C8 boundary.")
        print(f"  (Additionally {total_cusp_gaps} cusp-window gaps, "
              f"expected and handled by exact-engine fallthrough.)")


if __name__ == "__main__":
    main()
