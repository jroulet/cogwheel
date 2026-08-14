#!/usr/bin/env python
"""Measure the directional nearest-caustic distance floor ``_SADDLE_ETA_FLOOR``.

The tier-1 macro-saddle analytic rung
(:func:`cogwheel.lensing.likelihood._saddle_farfield_analytic`) serves the
far-from-caustic macro saddle with a ZERO residual envelope.  Its error is
governed by DIRECTIONAL caustic proximity ``eta = |source - nearest caustic
point|`` (``geometry.nearest_caustic_point(...).distance``, exposed on the
partition as ``geom.caustic_distance``), NOT by the retired isotropic scalar
``rho = |y| / caustic_reach``.  The deltoids are two DISCONNECTED lobes, so a
scalar-reach test wrongly refused the transverse (hard-axis) cone that sits
far from both lobes; the directional distance serves it.

This script scans a population of non-interior (two real image) resolvable
saddle sources across a range of measured ``eta`` and ``gamma``, evaluates the
zero-envelope ``FARFIELD_KERNEL_SUM`` serve against the exact Schwinger engine
oracle (the SAME recipe as ``test_lensing_saddle_tier1_accuracy.py``:
``ChangRefsdalChannels.evaluate().exact_total`` reached through the mass-sheet
identity), and locates, per ``gamma``, the worst (largest) ``eta`` at which the
serve still FAILS a ``1e-4`` relative-``|F|`` bar at the band maximum.

The applied floor follows a DETERMINISTIC rule (Professor asymmetry: a
false-admit is a silent lnL bias, a false-refuse only costs engine time)::

    boundary        = max over gamma of the worst failing eta
    _SADDLE_ETA_FLOOR = min(0.5, boundary * 2.0)

If the scan cannot establish a clean boundary at or below ``0.25`` (so the
``x2`` guard band would exceed ``0.5``), the rule DEFAULTS to ``0.5`` and the
whole unmeasured ``(0.05, 0.5)`` band is refused.

Runs entirely inside the double-double Schwinger domain (``w <= 60``) so every
oracle evaluation is cheap (~0.2 s).  Emits a paste-ready provenance block.

Usage
-----
    python scripts/measure_saddle_eta_floor.py
    python scripts/measure_saddle_eta_floor.py --gammas 1.2 1.5 2.0 --n-w 16
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels, FARFIELD_KERNEL_SUM, reconstruct_farfield)
from cogwheel.lensing.chang_refsdal.operator import RHO_END
from cogwheel.lensing.ppgo_map import caustic_geometry

#: Cheap-band dimensionless frequency floor; the served w band is
#: [W_FLOOR, W_CEIL], entirely inside the double-double engine domain.
W_FLOOR = 8.0

#: Cheap-band ceiling: kept at/below the double-double Schwinger ceiling so
#: every exact oracle eval avoids the expensive mpmath QD path.
W_CEIL = 60.0

#: Gamma slices to profile (macro saddle, gamma > 1).
DEFAULT_GAMMAS = (1.2, 1.5, 2.0)

#: Relative-|F| accuracy bar the zero-envelope serve must clear at band max.
ACCURACY_BAR = 1e-4

#: Absolute delay-gap tie tolerance (mirrors likelihood._SADDLE_TIE_EPS):
#: symmetry-tied mirror pairs sit at the ~1e-15 machine floor and never
#: resolve; they are excluded from the resolvability min.
_TIE_EPS = 1e-12

#: Boundary above which the deterministic x2 guard band would exceed the
#: hard 0.5 cap (Professor asymmetry default).
_BOUNDARY_CAP = 0.25


def _resolvable(real_delays: np.ndarray, w_lo: float) -> bool:
    """Whether the real image pair resolves at the band floor.

    Mirrors leg B of ``_saddle_farfield_analytic_serves``: the narrowest
    delay gap surviving the symmetry-tie tolerance must satisfy
    ``w_lo * min_delta_tau >= RHO_END``.
    """
    real = np.sort(np.asarray(real_delays, dtype=float))
    if len(real) < 2:
        return False
    gaps = np.diff(real)
    surviving = gaps[gaps > _TIE_EPS]
    if len(surviving) == 0:
        return False
    return w_lo * float(np.min(surviving)) >= RHO_END


def _serve_vs_oracle(w_grid: np.ndarray, gamma: float,
                     source: np.ndarray) -> tuple[float, int, float, np.ndarray]:
    """Evaluate the zero-envelope serve against the exact engine oracle.

    Returns ``(eta, image_count, rel_err_band_max, real_delays)`` where
    ``rel_err_band_max`` is ``|F_serve - F_oracle| / max|F_oracle|`` at the
    highest w node (the band maximum, where the zero-envelope error is
    largest).
    """
    y = (float(source[0]), float(source[1]))

    # Exact oracle (independent of the serve path).
    oracle_channels = ChangRefsdalChannels(w_grid)
    oracle_channels.reset()
    oracle_partition = oracle_channels.evaluate(
        gamma=gamma, y=y, beta=0.0, kappa=0.0)
    f_oracle = np.asarray(oracle_partition.exact_total)

    # Zero-envelope FARFIELD_KERNEL_SUM serve (mirrors the live rung).
    geom = ChangRefsdalChannels(w_grid).geometry_partition(
        gamma=gamma, y=y, beta=0.0, kappa=0.0)
    envelope = np.zeros(w_grid.shape, dtype=complex)
    _kernels, f_serve = reconstruct_farfield(
        w_grid, envelope, geom.delays, geom.saddle_kernels, geom.real_mask,
        FARFIELD_KERNEL_SUM, geom.t_min)
    f_serve = np.asarray(f_serve)

    eta = float(geom.caustic_distance)
    image_count = int(np.asarray(geom.real_mask, dtype=bool).sum())
    real_delays = np.asarray(geom.delays)[np.asarray(geom.real_mask, dtype=bool)]

    scale = float(np.max(np.abs(f_oracle)))
    if scale == 0.0 or not np.isfinite(scale):
        rel_err = math.inf
    else:
        rel_err = float(np.abs(f_serve[-1] - f_oracle[-1]) / scale)
    return eta, image_count, rel_err, real_delays


def _profile_gamma(gamma: float, w_grid: np.ndarray, *, n_angles: int,
                   n_radii: int) -> dict:
    """Sweep sources across measured eta at one gamma; find the failing edge.

    Places sources on a polar grid scaled by the caustic reach, keeps only
    the gate-eligible population (two real images == non-interior, resolvable
    at the band floor), and records ``(eta, rel_err)`` for each.  The
    per-gamma boundary is the largest ``eta`` among the FAILING admitted
    sources (worst case the floor must clear).
    """
    reach, _direction = caustic_geometry(gamma, kappa=0.0)
    w_lo = float(w_grid.min())
    # Scalar radii from just outside the caustic to well beyond it; the
    # measured directional eta is read back per source (never assumed).
    radius_scales = np.linspace(0.15, 3.0, n_radii)
    angles = np.linspace(0.0, math.pi, n_angles, endpoint=False)

    samples: list[tuple[float, float]] = []  # (eta, rel_err)
    for scale in radius_scales:
        radius = scale * reach
        for angle in angles:
            source = radius * np.array([math.cos(angle), math.sin(angle)])
            try:
                eta, image_count, rel_err, real_delays = _serve_vs_oracle(
                    w_grid, gamma, source)
            except geometry.LensDomainError:
                continue
            # Gate-eligible population only: non-interior (2 real images) AND
            # resolvable.  Interior (>= 4) and unresolvable sources are
            # refused by the other gate legs regardless of eta.
            if image_count != 2:
                continue
            if not _resolvable(real_delays, w_lo):
                continue
            if not np.isfinite(eta):
                continue
            samples.append((float(eta), float(rel_err)))

    failing = [eta for eta, err in samples if err > ACCURACY_BAR]
    passing = [eta for eta, err in samples if err <= ACCURACY_BAR]
    boundary = max(failing) if failing else 0.0
    return {
        'gamma': gamma,
        'n_samples': len(samples),
        'n_failing': len(failing),
        'boundary': boundary,
        'min_eta': min((e for e, _ in samples), default=math.nan),
        'max_eta': max((e for e, _ in samples), default=math.nan),
        'smallest_passing_eta': min(passing) if passing else math.nan,
        'largest_failing_eta': boundary if failing else math.nan,
    }


def main() -> None:
    """Run the eta-floor scan and print a paste-ready provenance block."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gammas', type=float, nargs='*',
                        default=list(DEFAULT_GAMMAS),
                        help='Gamma slices to profile (macro saddle > 1).')
    parser.add_argument('--n-w', type=int, default=16,
                        help='Number of log-spaced w nodes in [W_FLOOR, W_CEIL].')
    parser.add_argument('--n-angles', type=int, default=12,
                        help='Polar angles per radius scale.')
    parser.add_argument('--n-radii', type=int, default=12,
                        help='Radius scales per angle.')
    args = parser.parse_args()

    w_grid = np.geomspace(W_FLOOR, W_CEIL, args.n_w)

    results = [_profile_gamma(g, w_grid, n_angles=args.n_angles,
                              n_radii=args.n_radii)
               for g in args.gammas]

    boundary = max((r['boundary'] for r in results), default=0.0)
    if boundary > _BOUNDARY_CAP:
        applied_floor = 0.5
        clean = False
    else:
        applied_floor = min(0.5, boundary * 2.0)
        clean = True

    print('=' * 70)
    print('SADDLE ETA FLOOR MEASUREMENT')
    print('=' * 70)
    print(f'w band            : [{W_FLOOR}, {W_CEIL}]  ({args.n_w} nodes)')
    print(f'accuracy bar      : {ACCURACY_BAR:g} (relative |F| at band max)')
    print(f'grid per gamma    : {args.n_angles} angles x {args.n_radii} radii')
    print('-' * 70)
    print(f'{"gamma":>7} {"n_adm":>6} {"n_fail":>7} {"eta_min":>9} '
          f'{"eta_max":>9} {"worst_fail":>11} {"first_pass":>11}')
    for r in results:
        print(f'{r["gamma"]:>7.3f} {r["n_samples"]:>6d} {r["n_failing"]:>7d} '
              f'{r["min_eta"]:>9.4f} {r["max_eta"]:>9.4f} '
              f'{r["largest_failing_eta"]:>11.4f} '
              f'{r["smallest_passing_eta"]:>11.4f}')
    print('-' * 70)
    print(f'measured boundary (max worst-fail over gamma) : {boundary:.4f}')
    print(f'clean boundary at/below {_BOUNDARY_CAP}            : {clean}')
    print(f'APPLIED _SADDLE_ETA_FLOOR = min(0.5, {boundary:.4f}*2) = '
          f'{applied_floor:.4f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
