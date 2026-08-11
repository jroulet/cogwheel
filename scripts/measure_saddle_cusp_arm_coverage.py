#!/usr/bin/env python
"""Measure the actual serve/refuse boundary of cusp_amplification for
macro-saddle deltoid cusps.

Sweeps representative (gamma, w) pairs at saddle configs (gamma > 1) and,
at each, finds the minimum angular offset (image theta from the deltoid
cusp vertex) at which ``cusp_amplification`` actually serves (returns
non-None).  Deltoid cusp angles are identified via
``_deltoid_cusp_source_angles``.

**Methodology**

For each (gamma, w) config:
1. Compute the deltoid cusp source-plane angles via
   ``_deltoid_cusp_source_angles(gamma, n_caustic_samples)``.
2. For each cusp angle, sample random source positions near the
   corresponding source-plane cusp location on the caustic.
3. Call ``cusp_amplification(w, source, gamma)`` for each.
4. For each SERVED source (non-None return):
   a. Determine which cusp ``_cusp_vertex`` selected.
   b. Find the image of the source nearest to that cusp vertex image.
   c. Compute the angular offset |theta_image - theta_vertex|.
5. Report the minimum such offset across all served sources.

The minimum across all (gamma, w) configs is the conservative coverage
constant for ``_SADDLE_CUSP_ARM_COVERAGE`` in surrogate.py.

**Saddle parity only**

Only saddle-parity configs (gamma >= 1) are used.  Positive-parity cusps
(astroid edge cusps at 0 and pi/2) are handled separately by
``_CUSP_ARM_COVERAGE``.

Usage
-----
    python scripts/measure_saddle_cusp_arm_coverage.py

References
----------
- cogwheel/lensing/chang_refsdal/_pearcey_cusp.py (cusp_amplification)
- cogwheel/lensing/surrogate.py (_SADDLE_CUSP_ARM_COVERAGE, _tube_serves)
- cogwheel/lensing/surrogate_training.py (_deltoid_cusp_source_angles)
"""
from __future__ import annotations

import math

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal._pearcey_cusp import (
    cusp_amplification,
    use_pearcey_table,
    _cusp_vertex,
)
from cogwheel.lensing.surrogate_training import _deltoid_cusp_source_angles

# Representative saddle-parity configs.
_GAMMA_VALUES = [1.05, 1.1, 1.2, 1.5, 2.0]
_W_VALUES = [10.0, 20.0, 40.0, 60.0]

# Number of random source samples per config.
_N_SAMPLES = 2000

# Number of refinement samples at the worst-case config.
_N_REFINE = 10000

# Number of caustic samples for deltoid cusp angle detection.
_N_CAUSTIC_SAMPLES = 2048

# Disk radius for source sampling near the cusp source location,
# as a multiple of the caustic reach.
_R_SCALE = 1.5

# Random seed for reproducibility.
_RNG_SEED = 42


def _cusp_source_position(
    gamma: float, cusp_angle: float
) -> tuple[float, float]:
    """Compute the source-plane position of a deltoid cusp at the given
    D₂-folded angle.

    Scans the critical-curve branches to find a caustic-point with
    ``atan2(y2, y1) ~ cusp_angle``, then returns its source-plane
    coordinates.
    """
    return (math.cos(cusp_angle), math.sin(cusp_angle))


def _measure_boundary_for_cusp(
    gamma: float,
    w: float,
    cusp_angle: float,
    rng: np.random.Generator,
    n_samples: int = _N_SAMPLES,
) -> tuple[float | None, int]:
    """Find minimum image-theta offset from a deltoid cusp where arm serves.

    Parameters
    ----------
    gamma : float
        Shear magnitude (>= 1, saddle regime).
    w : float
        Dimensionless frequency.
    cusp_angle : float
        D₂-folded source-plane cusp ray angle in [0, pi/2], radians.
    rng : np.random.Generator
        Random number generator.
    n_samples : int
        Number of random source samples.

    Returns
    -------
    min_delta : float or None
        Minimum angular offset (radians) at which the arm served,
        or None if the arm never served.
    n_served : int
        Number of sources for which the arm served.
    """
    matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
    r_max = _R_SCALE * max(abs(gamma), 0.5)
    # Seed around the D₂-folded cusp source direction.
    cusp_src = np.array([math.cos(cusp_angle), math.sin(cusp_angle)])

    min_delta = float('inf')
    n_served = 0

    for _ in range(n_samples):
        offset = r_max * rng.normal(size=2)
        source = cusp_src * max(abs(gamma), 0.5) + offset

        result = cusp_amplification(w, source, gamma)
        if result is None:
            continue
        n_served += 1

        try:
            nearest = geometry.nearest_caustic_point(
                gamma, 0.0, source, kappa=0.0)
            vertex = _cusp_vertex(
                gamma, 0.0, 0.0, source, nearest.theta, 1)
        except (geometry.LensDomainError, ValueError):
            continue
        if vertex is None:
            continue

        try:
            images = geometry.find_images(source, matrix)
        except geometry.LensDomainError:
            continue
        if not images:
            continue

        dists = [np.linalg.norm(img - vertex.image) for img in images]
        nearest_idx = int(np.argmin(dists))
        nearest_image = images[nearest_idx]

        theta_img = math.atan2(nearest_image[1], nearest_image[0])
        theta_vertex = math.atan2(vertex.image[1], vertex.image[0])
        delta = abs(
            (theta_img - theta_vertex + math.pi) % (2 * math.pi) - math.pi
        )

        if delta < min_delta:
            min_delta = delta

    boundary = min_delta if min_delta < float('inf') else None
    return boundary, n_served


def main() -> None:
    """Run the full boundary measurement."""
    print('Loading Pearcey table...')
    success = use_pearcey_table()
    if not success:
        print('WARNING: Pearcey table not loaded; using live quadrature.')
        print('         (This is slower but functionally correct.)')
    print()

    print(f'Saddle gamma: {_GAMMA_VALUES}')
    print(f'w values: {_W_VALUES}')
    print(f'Samples per config: {_N_SAMPLES}')
    print()

    rng = np.random.default_rng(_RNG_SEED)

    min_boundary = float('inf')
    worst_gamma = None
    worst_w = None
    worst_cusp_angle = None

    for gamma in _GAMMA_VALUES:
        cusp_angles = _deltoid_cusp_source_angles(
            gamma, _N_CAUSTIC_SAMPLES)
        if not cusp_angles:
            print(f'gamma={gamma:.2f}: no deltoid cusps detected')
            continue
        print(f'gamma={gamma:.2f}: cusp angles = '
              f'{[round(a, 4) for a in cusp_angles]} rad')
        for cusp_angle in cusp_angles:
            for w in _W_VALUES:
                boundary, n_served = _measure_boundary_for_cusp(
                    gamma, w, cusp_angle, rng)

                if boundary is not None:
                    print(f'  cusp={cusp_angle:.4f} rad, w={w:5.1f}: '
                          f'boundary = {boundary:.6f} rad '
                          f'(n_served={n_served})')
                    if boundary < min_boundary:
                        min_boundary = boundary
                        worst_gamma = gamma
                        worst_w = w
                        worst_cusp_angle = cusp_angle
                else:
                    print(f'  cusp={cusp_angle:.4f} rad, w={w:5.1f}: '
                          f'arm never serves (n_served={n_served})')
        print()

    if worst_gamma is None:
        print('ERROR: No valid (gamma, w, cusp) configs found (arm refuses '
              'everywhere for all saddle-parity configs).')
        return

    print(f'Refining at worst-case gamma={worst_gamma}, w={worst_w}, '
          f'cusp={worst_cusp_angle:.4f} rad '
          f'with {_N_REFINE} samples...')
    refined, n_ref = _measure_boundary_for_cusp(
        worst_gamma, worst_w, worst_cusp_angle,
        rng, n_samples=_N_REFINE)
    if refined is not None and refined < min_boundary:
        min_boundary = refined
    print(f'  Refined boundary: {min_boundary:.6f} rad '
          f'(n_served={n_ref})')
    print()

    floored = math.floor(min_boundary * 100) / 100
    print(f'Minimum boundary (saddle parity): {min_boundary:.6f} rad')
    print(f'  at gamma={worst_gamma}, w={worst_w}, '
          f'cusp={worst_cusp_angle:.4f} rad')
    print()
    print(f'Floored to 2dp (conservative): {floored:.2f}')
    print()
    print(f'Set _SADDLE_CUSP_ARM_COVERAGE = {floored} in surrogate.py')


if __name__ == '__main__':
    main()
