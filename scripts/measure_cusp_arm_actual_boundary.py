#!/usr/bin/env python
"""Measure the actual serve/refuse boundary of cusp_amplification.

Sweeps representative (gamma, w) pairs and, at each, finds the minimum
angular offset (image theta from the cusp vertex) at which
``cusp_amplification`` actually serves (returns non-None).

Unlike ``measure_cusp_arm_reach.py`` (which estimates the analytic R-gate
boundary), this script calls ``cusp_amplification`` directly on random
source positions inside the caustic and measures the empirical boundary.

**Methodology**

For each (gamma, w) config:
1. Sample random source positions in a disk of radius 2 * max(gamma, 0.5).
2. Call ``cusp_amplification(w, source, gamma)`` for each.
3. For each SERVED source (non-None return):
   a. Determine which cusp ``_cusp_vertex`` selected.
   b. Find the image of the source nearest to that cusp vertex image.
   c. Compute the angular offset |theta_image - theta_vertex|.
4. Report the minimum such offset across all served sources.

The minimum across all (gamma, w) configs is the conservative coverage
constant for ``_CUSP_ARM_COVERAGE`` in surrogate.py.

**Positive parity only**

Only positive-parity configs (gamma < 1) are used for the final minimum.
For saddle parity (gamma >= 1), the measurement converges toward zero
because deep-interior sources can have images arbitrarily close to the
cusp angle; these sources are operationally excluded by the tube's
``eta_floor`` and are not relevant to the cusp window decision.  Saddle
cusp coverage is handled separately by ``_SADDLE_CUSP_MIN_HALFWIDTH``.

**Why not scan along the critical curve?**

Sources on the critical curve lie ON the caustic (the fold/cusp boundary).
The arm's calibration certificate systematically fails for on-caustic
sources because the uniform approximation's matched-delay requirement
cannot be satisfied at the catastrophe surface.  The arm serves sources
*near* (inside) the caustic.  The operationally relevant measurement
is: at what image-theta offset from a cusp does the arm first serve?
That offset is what ``_tube_serves`` checks.

Usage
-----
    python scripts/measure_cusp_arm_actual_boundary.py

References
----------
- cogwheel/lensing/chang_refsdal/_pearcey_cusp.py (cusp_amplification)
- cogwheel/lensing/surrogate.py (_CUSP_ARM_COVERAGE, _tube_serves)
- scripts/measure_cusp_arm_reach.py (companion analytic R-gate measurement)
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

# Representative configs: positive-parity only for the binding minimum.
# Saddle configs reported for reference but excluded from the floor.
_GAMMA_VALUES_POSITIVE = [0.1, 0.2, 0.3, 0.5]
_GAMMA_VALUES_SADDLE = [1.2, 1.5]
_W_VALUES = [10.0, 20.0, 40.0, 60.0]

# Number of random source samples per config (coarse pass).
_N_SAMPLES = 2000

# Number of refinement samples at the worst-case config.
_N_REFINE = 10000

# Disk radius for source sampling, as a multiple of max(gamma, 0.5).
_R_SCALE = 2.0

# Random seed for reproducibility.
_RNG_SEED = 42


def _measure_boundary_for_config(
    gamma: float,
    w: float,
    rng: np.random.Generator,
    n_samples: int = _N_SAMPLES,
) -> tuple[float | None, int]:
    """Find minimum image-theta offset from cusp where arm serves.

    For each served source, finds the image nearest the cusp vertex
    and reports its angular offset.  The minimum across all served
    sources is the boundary for this config.

    Returns (min_delta_theta, n_served).
    """
    matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
    r_max = _R_SCALE * max(abs(gamma), 0.5)

    min_delta = float('inf')
    n_served = 0

    for _ in range(n_samples):
        r = r_max * math.sqrt(rng.random())
        angle = 2 * math.pi * rng.random()
        source = r * np.array([math.cos(angle), math.sin(angle)])

        result = cusp_amplification(w, source, gamma)
        if result is None:
            continue
        n_served += 1

        # Determine which cusp the arm selected.
        try:
            nearest = geometry.nearest_caustic_point(
                gamma, 0.0, source, kappa=0.0)
            vertex = _cusp_vertex(
                gamma, 0.0, 0.0, source, nearest.theta, 1)
        except (geometry.LensDomainError, ValueError):
            continue
        if vertex is None:
            continue

        # Find the image of source nearest to the cusp vertex image.
        try:
            images = geometry.find_images(source, matrix)
        except geometry.LensDomainError:
            continue
        if not images:
            continue

        dists = [np.linalg.norm(img - vertex.image) for img in images]
        nearest_idx = int(np.argmin(dists))
        nearest_image = images[nearest_idx]

        # Angular offset of this image from the cusp vertex image.
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

    print(f'Positive gamma: {_GAMMA_VALUES_POSITIVE}')
    print(f'Saddle gamma (reference): {_GAMMA_VALUES_SADDLE}')
    print(f'w values: {_W_VALUES}')
    print(f'Samples per config: {_N_SAMPLES}')
    print()

    rng = np.random.default_rng(_RNG_SEED)

    # --- Positive parity (binding for the final constant) ---
    print('--- Positive parity (gamma < 1) ---')
    min_boundary_pos = float('inf')
    worst_gamma = None
    worst_w = None

    for gamma in _GAMMA_VALUES_POSITIVE:
        for w in _W_VALUES:
            boundary, n_served = _measure_boundary_for_config(
                gamma, w, rng)

            if boundary is not None:
                print(f'  gamma={gamma:.2f}, w={w:5.1f}: '
                      f'boundary = {boundary:.6f} rad '
                      f'(n_served={n_served})')
                if boundary < min_boundary_pos:
                    min_boundary_pos = boundary
                    worst_gamma = gamma
                    worst_w = w
            else:
                print(f'  gamma={gamma:.2f}, w={w:5.1f}: '
                      f'arm never serves (n_served={n_served})')

    print()

    # --- Saddle parity (reference only) ---
    print('--- Saddle parity (gamma >= 1, reference) ---')
    for gamma in _GAMMA_VALUES_SADDLE:
        for w in _W_VALUES:
            boundary, n_served = _measure_boundary_for_config(
                gamma, w, rng)
            if boundary is not None:
                print(f'  gamma={gamma:.2f}, w={w:5.1f}: '
                      f'boundary = {boundary:.6f} rad '
                      f'(n_served={n_served})')
            else:
                print(f'  gamma={gamma:.2f}, w={w:5.1f}: '
                      f'arm never serves (n_served={n_served})')
    print('  (Saddle excluded from floor; see module docstring.)')
    print()

    if worst_gamma is None:
        print('ERROR: No valid (gamma, w) pairs found (arm refuses '
              'everywhere for all positive-parity configs).')
        return

    # Refine measurement at the worst-case positive-parity config.
    print(f'Refining at worst-case gamma={worst_gamma}, w={worst_w} '
          f'with {_N_REFINE} samples...')
    refined, n_ref = _measure_boundary_for_config(
        worst_gamma, worst_w, rng, n_samples=_N_REFINE)
    if refined is not None and refined < min_boundary_pos:
        min_boundary_pos = refined
    print(f'  Refined boundary: {min_boundary_pos:.6f} rad '
          f'(n_served={n_ref})')
    print()

    # Floor to 2 decimal places (conservative: claiming LESS coverage).
    floored = math.floor(min_boundary_pos * 100) / 100
    print(f'Minimum boundary (positive parity): {min_boundary_pos:.6f} rad')
    print(f'  at gamma={worst_gamma}, w={worst_w}')
    print()
    print(f'Floored to 2dp (conservative): {floored:.2f}')
    print()
    print(f'Set _CUSP_ARM_COVERAGE = {floored} in surrogate.py')


if __name__ == '__main__':
    main()
