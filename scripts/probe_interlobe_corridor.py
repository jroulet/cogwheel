#!/usr/bin/env python
"""Geometric measurement of the inter-lobe corridor for saddle-parity configs.

Computes the inter-lobe corridor width — the thin slab around the lobe-
equidistance (perpendicular-bisector) line excluded by the ``_lobe_serves``
corridor test — for representative macro-saddle (gamma > 1) configurations and
reports whether it is negligible.

The corridor is the region where, for BOTH lobes,
``|p - centroid_this| + corridor_half > |p - centroid_other|`` fails,
i.e. the source ``p`` is NOT clearly nearer either lobe's centroid by the
margin ``corridor_half = _INTERLOBE_CORRIDOR_ETA_SCALE * eta_max``.

Output columns:
    gamma        -- external shear parameter
    centroid_sep -- distance between the two lobe centroids (dimensionless y)
    R_c_min      -- min caustic curvature radius over all arcs (dimensionless)
    eta_max      -- _DEFAULT_F_MAX * R_c_min  (tube half-width)
    corr_width   -- 2 * _INTERLOBE_CORRIDOR_ETA_SCALE * eta_max
    width/sep    -- corridor_width / centroid_separation ratio
    reach_A/B    -- scalar lobe extent for each lobe
    area_frac    -- Monte Carlo fraction of lobe-interior sources in corridor

Verdict: CLOSED if max area_frac < 0.01 across all gammas (professor estimate:
O(1-5%) near gamma=1); OPEN otherwise (follow-up needed).

Usage:
    conda run -n $SDK_CONDA_ENV python scripts/probe_interlobe_corridor.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cogwheel.lensing.surrogate_training import (
    _DEFAULT_F_MAX,
    _INTERLOBE_CORRIDOR_ETA_SCALE,
    _SADDLE_LOBE_CENTERS,
    _directional_lobe_boundary,
    _lobe_caustic_points,
    _min_curvature_radius,
    _saddle_arcs,
    FoldArc,
)
from cogwheel.lensing.chang_refsdal import geometry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Representative saddle gamma values to probe.
GAMMAS: tuple[float, ...] = (1.1, 1.3, 1.5, 2.0)

#: Number of caustic samples per arc (and per lobe).
N_CAUSTIC_SAMPLES: int = 500

#: Number of Monte Carlo source draws for area-fraction estimate.
N_MC_SAMPLES: int = 10_000

#: Half-width of the gamma band around each representative gamma.
GAMMA_BAND_HW: float = 0.02

#: Random seed for reproducible Monte Carlo.
_RNG_SEED: int = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lobe_centroid(gamma: float, lens_center: float, n: int) -> np.ndarray:
    """Source-plane centroid of one saddle deltoid lobe.

    Parameters
    ----------
    gamma : float
        External shear parameter (> 1).
    lens_center : float
        Lens-plane angular centre of the lobe (0 or pi rad).
    n : int
        Number of caustic samples per branch.

    Returns
    -------
    np.ndarray
        Shape ``(2,)`` centroid as the mean of the ``(k, 2)`` lobe caustic
        points.  Returns the zero vector when no points are generated.
    """
    pts = _lobe_caustic_points(gamma, lens_center, n)
    if pts.shape[0] == 0:
        return np.zeros(2)
    return pts.mean(axis=0)


def _corridor_geometry(gamma: float) -> dict:
    """Compute inter-lobe corridor geometry for a given saddle gamma.

    Parameters
    ----------
    gamma : float
        External shear parameter (> 1).

    Returns
    -------
    dict with keys:
        centroid_A, centroid_B : np.ndarray  -- lobe centroids
        centroid_sep           : float        -- |centroid_A - centroid_B|
        R_c_min                : float        -- min curvature radius over arcs
        eta_max                : float        -- _DEFAULT_F_MAX * R_c_min
        corridor_half          : float        -- corridor half-width
        corridor_width         : float        -- 2 * corridor_half
        width_sep_ratio        : float        -- corridor_width / centroid_sep
        lobe_reach_A           : float        -- max caustic dist from centroid_A
        lobe_reach_B           : float        -- max caustic dist from centroid_B
    """
    # Lobe centroids at gamma midpoint (representative)
    centroid_a = _lobe_centroid(gamma, _SADDLE_LOBE_CENTERS[0],
                                N_CAUSTIC_SAMPLES)
    centroid_b = _lobe_centroid(gamma, _SADDLE_LOBE_CENTERS[1],
                                N_CAUSTIC_SAMPLES)
    centroid_sep = float(np.hypot(centroid_a[0] - centroid_b[0],
                                  centroid_a[1] - centroid_b[1]))

    # Curvature radius: min over all arcs at worst band gamma
    band = (gamma - GAMMA_BAND_HW, gamma + GAMMA_BAND_HW)
    _cusps, arcs, _reach = _saddle_arcs(gamma, N_CAUSTIC_SAMPLES)
    if arcs:
        r_c_min = min(
            _min_curvature_radius(band, arc, N_CAUSTIC_SAMPLES)
            for arc in arcs
        )
    else:
        r_c_min = float('inf')

    eta_max = _DEFAULT_F_MAX * r_c_min
    corridor_half = _INTERLOBE_CORRIDOR_ETA_SCALE * eta_max
    corridor_width = 2.0 * corridor_half

    # Scalar lobe reaches (max |caustic - centroid| over all lobe points)
    pts_a = _lobe_caustic_points(gamma, _SADDLE_LOBE_CENTERS[0],
                                 N_CAUSTIC_SAMPLES)
    pts_b = _lobe_caustic_points(gamma, _SADDLE_LOBE_CENTERS[1],
                                 N_CAUSTIC_SAMPLES)
    lobe_reach_a = (
        float(np.hypot(pts_a[:, 0] - centroid_a[0],
                       pts_a[:, 1] - centroid_a[1]).max())
        if pts_a.shape[0] > 0 else 0.0
    )
    lobe_reach_b = (
        float(np.hypot(pts_b[:, 0] - centroid_b[0],
                       pts_b[:, 1] - centroid_b[1]).max())
        if pts_b.shape[0] > 0 else 0.0
    )

    return {
        'centroid_A': centroid_a,
        'centroid_B': centroid_b,
        'centroid_sep': centroid_sep,
        'R_c_min': r_c_min,
        'eta_max': eta_max,
        'corridor_half': corridor_half,
        'corridor_width': corridor_width,
        'width_sep_ratio': (corridor_width / centroid_sep
                            if centroid_sep > 0.0 else float('nan')),
        'lobe_reach_A': lobe_reach_a,
        'lobe_reach_B': lobe_reach_b,
    }


def _in_lobe(point: np.ndarray, centroid: np.ndarray,
             boundary_theta: np.ndarray, boundary_r: np.ndarray) -> bool:
    """Check whether a source point is inside the lobe boundary.

    Uses the directional boundary radius: a point is inside when its
    lobe-local radial coordinate ``rho_lobe = |p - centroid| /
    r_deltoid(theta_local)`` is less than 1.

    Parameters
    ----------
    point : np.ndarray  shape (2,)
    centroid : np.ndarray  shape (2,)
    boundary_theta, boundary_r : np.ndarray
        Lobe directional boundary from ``_directional_lobe_boundary``.
    """
    rel = point - centroid
    dist = math.hypot(rel[0], rel[1])
    if dist == 0.0:
        return True  # exactly at centroid -- inside
    theta_local = math.atan2(rel[1], rel[0])
    # Periodic linear interpolation matching ``_lobe_boundary_radius``
    r_deltoid = float(np.interp(theta_local, boundary_theta, boundary_r,
                                period=2.0 * math.pi))
    return dist < r_deltoid  # rho_lobe < 1


def _in_corridor(point: np.ndarray, centroid_this: np.ndarray,
                 centroid_other: np.ndarray, corridor_half: float) -> bool:
    """Check whether a point falls in the inter-lobe corridor for THIS lobe.

    The corridor test from ``_SaddleLobeAdmission.admits``:
    ``near_this + corridor_half > near_other`` means the point is NOT
    clearly nearer this lobe's centroid, so THIS lobe rejects it.  A
    point is in the BILATERAL corridor when BOTH lobes reject it.

    Parameters
    ----------
    point : np.ndarray  shape (2,)
    centroid_this : np.ndarray
    centroid_other : np.ndarray
    corridor_half : float  half-width of the corridor
    """
    near_this = math.hypot(point[0] - centroid_this[0],
                           point[1] - centroid_this[1])
    near_other = math.hypot(point[0] - centroid_other[0],
                            point[1] - centroid_other[1])
    return near_this + corridor_half > near_other


def _corridor_area_fraction(gamma: float, n_mc: int) -> float:
    """Monte Carlo estimate of bilateral corridor fraction inside either lobe.

    Uniform samples are drawn inside the bounding box of lobe A, keeping
    those that fall inside lobe A OR lobe B (via directional boundary test),
    then counting those that ALSO fail the corridor test for BOTH lobes (i.e.
    lie in the bilateral corridor that neither lobe's tile covers).

    Parameters
    ----------
    gamma : float  external shear (> 1)
    n_mc : int     number of Monte Carlo draws

    Returns
    -------
    float
        Fraction of lobe-interior sources that fall in the bilateral corridor.
        Returns 0.0 when no sample falls inside either lobe.
    """
    # Build directional boundaries at gamma (representative)
    pts_a = _lobe_caustic_points(gamma, _SADDLE_LOBE_CENTERS[0],
                                 N_CAUSTIC_SAMPLES)
    pts_b = _lobe_caustic_points(gamma, _SADDLE_LOBE_CENTERS[1],
                                 N_CAUSTIC_SAMPLES)
    centroid_a = (pts_a.mean(axis=0)
                  if pts_a.shape[0] > 0 else np.zeros(2))
    centroid_b = (pts_b.mean(axis=0)
                  if pts_b.shape[0] > 0 else np.zeros(2))

    b_theta_a, b_r_a = _directional_lobe_boundary(pts_a, centroid_a)
    b_theta_b, b_r_b = _directional_lobe_boundary(pts_b, centroid_b)

    # Corridor half-width
    band = (gamma - GAMMA_BAND_HW, gamma + GAMMA_BAND_HW)
    _cusps, arcs, _reach = _saddle_arcs(gamma, N_CAUSTIC_SAMPLES)
    if arcs:
        r_c_min = min(
            _min_curvature_radius(band, arc, N_CAUSTIC_SAMPLES)
            for arc in arcs
        )
    else:
        r_c_min = float('inf')
    corridor_half = _INTERLOBE_CORRIDOR_ETA_SCALE * _DEFAULT_F_MAX * r_c_min

    # Bounding box: span of both lobes' caustic points
    if pts_a.shape[0] == 0 and pts_b.shape[0] == 0:
        return 0.0
    all_pts = np.vstack([p for p in (pts_a, pts_b) if p.shape[0] > 0])
    reach_a = (float(np.hypot(pts_a[:, 0] - centroid_a[0],
                               pts_a[:, 1] - centroid_a[1]).max())
               if pts_a.shape[0] > 0 else 0.0)
    reach_b = (float(np.hypot(pts_b[:, 0] - centroid_b[0],
                               pts_b[:, 1] - centroid_b[1]).max())
               if pts_b.shape[0] > 0 else 0.0)

    # Bounding box covers BOTH lobes with some margin
    x_min = float(all_pts[:, 0].min()) - max(reach_a, reach_b)
    x_max = float(all_pts[:, 0].max()) + max(reach_a, reach_b)
    y_min = float(all_pts[:, 1].min()) - max(reach_a, reach_b)
    y_max = float(all_pts[:, 1].max()) + max(reach_a, reach_b)

    rng = np.random.default_rng(_RNG_SEED)
    px = rng.uniform(x_min, x_max, n_mc)
    py = rng.uniform(y_min, y_max, n_mc)

    inside_count = 0
    corridor_count = 0

    for idx in range(n_mc):
        pt = np.array([px[idx], py[idx]])
        in_a = _in_lobe(pt, centroid_a, b_theta_a, b_r_a)
        in_b = _in_lobe(pt, centroid_b, b_theta_b, b_r_b)
        if not (in_a or in_b):
            continue
        inside_count += 1
        # Bilateral corridor: BOTH lobes reject this point
        rej_a = _in_corridor(pt, centroid_a, centroid_b, corridor_half)
        rej_b = _in_corridor(pt, centroid_b, centroid_a, corridor_half)
        if rej_a and rej_b:
            corridor_count += 1

    return corridor_count / inside_count if inside_count > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run inter-lobe corridor geometry measurement and print summary table."""
    print("=" * 78)
    print("INTER-LOBE CORRIDOR GEOMETRY PROBE (saddle-parity, gamma > 1)")
    print("=" * 78)
    print(f"  gammas:            {GAMMAS}")
    print(f"  N_CAUSTIC_SAMPLES: {N_CAUSTIC_SAMPLES}")
    print(f"  N_MC_SAMPLES:      {N_MC_SAMPLES}")
    print(f"  GAMMA_BAND_HW:     {GAMMA_BAND_HW}")
    print(f"  _DEFAULT_F_MAX:    {_DEFAULT_F_MAX}")
    print(f"  _INTERLOBE_CORRIDOR_ETA_SCALE: {_INTERLOBE_CORRIDOR_ETA_SCALE}")
    print(flush=True)

    # Header for the summary table
    header = (
        f"{'gamma':>6}  {'centroid_sep':>12}  {'R_c_min':>8}  "
        f"{'eta_max':>8}  {'corr_width':>10}  {'width/sep':>9}  "
        f"{'reach_A':>7}  {'reach_B':>7}  {'area_frac':>9}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    results: list[dict] = []

    for gamma in GAMMAS:
        print(f"  Computing gamma={gamma:.1f} ... ", end="", flush=True)
        geom = _corridor_geometry(gamma)
        area_frac = _corridor_area_fraction(gamma, N_MC_SAMPLES)
        geom['area_frac'] = area_frac
        results.append(geom | {'gamma': gamma})
        print("done", flush=True)

        print(
            f"{gamma:>6.2f}  "
            f"{geom['centroid_sep']:>12.4f}  "
            f"{geom['R_c_min']:>8.4f}  "
            f"{geom['eta_max']:>8.4f}  "
            f"{geom['corridor_width']:>10.4f}  "
            f"{geom['width_sep_ratio']:>9.4f}  "
            f"{geom['lobe_reach_A']:>7.4f}  "
            f"{geom['lobe_reach_B']:>7.4f}  "
            f"{area_frac:>9.4f}"
        )

    print(sep)

    # Per-gamma detail block
    print("\nPer-gamma detail:")
    for res in results:
        gamma = res['gamma']
        print(f"\n  gamma = {gamma:.2f}:")
        print(f"    centroid_A      = {res['centroid_A']}")
        print(f"    centroid_B      = {res['centroid_B']}")
        print(f"    centroid_sep    = {res['centroid_sep']:.6f}")
        print(f"    R_c_min         = {res['R_c_min']:.6f}")
        print(f"    eta_max         = {res['eta_max']:.6f}")
        print(f"    corridor_half   = {res['corridor_half']:.6f}")
        print(f"    corridor_width  = {res['corridor_width']:.6f}")
        print(f"    width/sep       = {res['width_sep_ratio']:.4%}")
        print(f"    lobe_reach_A    = {res['lobe_reach_A']:.6f}")
        print(f"    lobe_reach_B    = {res['lobe_reach_B']:.6f}")
        print(f"    area_fraction   = {res['area_frac']:.4%}")

    # Verdict
    max_frac = max(res['area_frac'] for res in results)
    max_ratio = max(res['width_sep_ratio'] for res in results)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  Max corridor/sep ratio across gammas : {max_ratio:.4%}")
    print(f"  Max area fraction across gammas      : {max_frac:.4%}")
    print()
    if max_frac < 0.01:
        print(
            f"REGION 10 CLOSED: inter-lobe corridor is negligible "
            f"(max area fraction = {max_frac:.4%}, "
            f"max width/sep = {max_ratio:.4%}).  "
            f"No accuracy concern — exact-engine fallback handles the corridor."
        )
    else:
        print(
            f"OPEN: corridor captures > 1% of interior sources "
            f"(max area fraction = {max_frac:.4%}).  Follow-up needed."
        )


if __name__ == "__main__":
    main()
