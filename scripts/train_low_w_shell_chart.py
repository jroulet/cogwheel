#!/usr/bin/env python
"""Train the low-w near-fold-shell macro-lead residual chart.

WHAT
----
Bakes the ``LowWShellChart`` NPZ artifact (loaded and hash-verified by
``cogwheel.lensing.low_w_shell_chart.LowWShellChart.load``) that the
positive-parity low-w diffractive serve
(``likelihood.LensedRelativeBinningLikelihood._low_w_shell_chart_serve``)
consults instead of the exact Schwinger engine for the near-fold shell.

The stored object is the SMOOTH macro-lead demodulated-DIFFERENCE residual,
never a quotient::

    R(w; gamma', rho, theta) = f_pure(w) - carrier(w)

with ``f_pure = f_schwinger(w, y_eig, gamma')`` the exact pure-shear
(reduced-frame) engine value and ``carrier = born_lead_carrier(w, y1, y2,
gamma')`` the macro lead (``sqrt(mu_macro) * exp(1j w phi_geo)``, imported
from ``_born``, NOT re-implemented).  Both sides are evaluated in the SAME
REDUCED eigenframe (``kappa = beta = 0``), so the RAW difference needs NO
``t_min`` demodulation (Professor Q1); a difference has no poles -- the
carrier's beating zeros cancel identically because ``f_pure`` carries the
same beat.  This is the settled BornResidualChart representation; the
quotient form it replaces produced 5800x poles.  The serve re-adds the
reduced-frame macro lead and re-modulates with the mass-sheet phase:
``F_abs = mass_sheet_phase * (carrier + R) / lam``.

GRID
----
Tensor-product over ``(gamma', rho, theta, log w)`` in the REDUCED /
caustic-relative coordinates the chart stores:

* ``gamma'``  -- reduced shear, in ``[GAMMA_LO, 1 - DELTA_GAMMA_P]`` (the
  positive-parity range below the wall; denser near the wall where the
  macro amplitude ``sqrt(mu_macro)`` grows).
* ``rho``     -- caustic-relative distance, ``np.linspace(RHO_LO, RHO_HI)``
  (the near-fold shell ``[0.6, 1.4]`` -- the chart owns only this band).
* ``theta``   -- eigenframe source angle, folded to the D2 fundamental
  domain ``[0, pi/2]`` (Nyquist-resolved: 8+ nodes resolve the even-
  harmonic basis ``cos(2k theta)`` up to ``k = 7``).
* ``log w``   -- ``np.geomspace(W_LO, W_HI)`` = ``[0.02, 1.0]`` (the smooth
  low-w shell: ``w * delta_min < 1``, where the fold/cusp structure has not
  yet developed and ``F -> sqrt(mu_macro)`` regardless of ``rho``).

The node source is reconstructed by inverting the fence discriminator via
the single-sourced `reduced_source(gamma', rho, theta)`:
``|y'| = rho * |caustic_point(gamma', theta)|`` -- the same
`geometry.caustic_point` the serve uses, never a numerical root-find.

ORACLE
------
``f_schwinger`` is the OFFLINE oracle ONLY (never called on the serve path).
The shell residual is smooth and O(1) throughout (measured ``|r| ~ 0.61-1.6``
at the shell witnesses), so there is NO de-rate and NO declined mask -- unlike
the rho-partitioned low-w diffractive chart this one replaces.

Usage::

    python scripts/train_low_w_shell_chart.py --scale smoke  # in-build
    python scripts/train_low_w_shell_chart.py --scale full   # driver bake

The SMOKE scale is a tiny non-collinear subset (a few hundred engine calls)
that proves the pipeline end-to-end and emits a smoke artifact under the
system temp dir (never the shipped path).  The FULL scale bakes the shipped
``cogwheel/data/low_w_shell_chart.npz`` -- a DRIVER post-build step.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Ensure the project root is importable (mirrors scripts/train_born_residual.py).
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from cogwheel.lensing.chang_refsdal._born import (
    DELTA_GAMMA_P, born_lead_carrier)
from cogwheel.lensing.chang_refsdal._schwinger import (
    SchwingerCertificationError, f_schwinger)
from cogwheel.lensing.low_w_shell_chart import (
    _SCHEMA, LowWShellChart, RHO_HI, RHO_LO, _content_hash, reduced_source)

#: Lower bound of the reduced-shear ``gamma'`` grid.  Mirrors the old
#: ``train_low_w_diffractive_chart`` grid: below it the caustic is too small
#: to resolve the near-fold shell and ``w_low_fit`` is still calibrated, so
#: the chart cedes the sub-``GAMMA_LO`` band to it.
GAMMA_LO = 0.05

#: Lower bound of the dimensionless-frequency grid.  Slightly below the
#: production minimum ``w`` (10 Msun at 20 Hz ~ 0.025), so a real band never
#: extrapolates below the trained frequency axis.
W_LO = 0.02

#: Upper bound of the dimensionless-frequency grid (the smooth low-w shell:
#: above ``w ~ 1`` the fold/cusp structure develops and the chart declines).
W_HI = 1.0

#: Full-scale grid node counts (the driver bake).  ``n_w`` dominates the
#: cost: ``f_schwinger`` is ~30-350 ms/call, so
#: ``n_gamma * n_rho * n_theta * n_w`` ~ 35k calls is a tens-of-minutes sweep.
_FULL_N_GAMMA = 14
_FULL_N_RHO = 10
_FULL_N_THETA = 16
_FULL_N_W = 16

#: Smoke-scale grid node counts (the in-build pipeline proof).  Every axis
#: must have >= 4 nodes for the cubic interpolator; ``theta`` keeps 8 (the
#: even-harmonic Nyquist floor), the rest are deliberately tiny.
_SMOKE_N_GAMMA = 4
_SMOKE_N_RHO = 4
_SMOKE_N_THETA = 8
_SMOKE_N_W = 6


def _provenance_sha() -> str:
    """Short ``git`` SHA of HEAD, or ``'unknown'`` outside a repo."""
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'], check=True,
            capture_output=True, text=True)
        return out.stdout.strip()
    except Exception:
        return 'unknown'


def _gamma_prime_grid(scale: str) -> np.ndarray:
    """Reduced-shear grid, ascending, covering ``[GAMMA_LO, 1 - DELTA_GAMMA_P]``."""
    if scale == 'smoke':
        return np.linspace(GAMMA_LO, 1.0 - DELTA_GAMMA_P, _SMOKE_N_GAMMA)
    # Full: denser near the parity wall (small ``1 - gamma'``), where the
    # macro amplitude ``sqrt(mu_macro)`` grows.  ``geomspace(1 - GAMMA_LO,
    # DELTA_GAMMA_P)`` descends, so ``1 - that`` ascends from ``GAMMA_LO`` to
    # ``1 - DELTA_GAMMA_P`` with the wall end pinned exactly.
    return 1.0 - np.geomspace(1.0 - GAMMA_LO, DELTA_GAMMA_P, _FULL_N_GAMMA)


def _theta_grid(scale: str) -> np.ndarray:
    """Eigenframe angle grid over the D2 fundamental domain ``[0, pi/2]``."""
    n = _SMOKE_N_THETA if scale == 'smoke' else _FULL_N_THETA
    return np.linspace(0.0, math.pi / 2.0, n)


def _rho_grid(scale: str) -> np.ndarray:
    """Caustic-relative distance grid over the near-fold shell ``[RHO_LO, RHO_HI]``."""
    n = _SMOKE_N_RHO if scale == 'smoke' else _FULL_N_RHO
    return np.linspace(RHO_LO, RHO_HI, n)


def _w_grid(scale: str, n_w: int | None) -> np.ndarray:
    """Dimensionless-frequency grid, log-spaced over ``[W_LO, W_HI]``.

    The shell residual is smooth (no Airy-fold ``w**(1/6)`` variation yet --
    that is the whole point of the low-w shell), so a log-spaced axis is
    adequate; the chart stores ``log_w_grid = np.log(w_grid)``.
    """
    if n_w is None:
        n_w = _SMOKE_N_W if scale == 'smoke' else _FULL_N_W
    return np.geomspace(W_LO, W_HI, n_w)


def _fill_coefficients(gamma_prime_grid: np.ndarray, rho_grid: np.ndarray,
                       theta_grid: np.ndarray,
                       w_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the macro-lead residual ``R = f_pure - carrier`` at every node.

    For each ``(gamma', rho, theta)`` cell the reduced eigenframe source is
    reconstructed via `reduced_source` (single-sourced with the serve), then
    for each ``w`` the pure-shear engine value ``f_schwinger(w, source,
    gamma')`` and the macro-lead carrier ``born_lead_carrier(w, source[0],
    source[1], gamma')`` are subtracted RAW (no ``t_min`` demodulation --
    both are in the same reduced frame).  An engine refusal
    (``SchwingerCertificationError`` / ``ValueError``) is a grid-design bug,
    never a silent NaN -- it aborts with the offending node named.
    """
    n_gp, n_rho, n_theta, n_w = (len(gamma_prime_grid), len(rho_grid),
                                 len(theta_grid), len(w_grid))
    real = np.empty((n_gp, n_rho, n_theta, n_w), dtype=float)
    imag = np.empty((n_gp, n_rho, n_theta, n_w), dtype=float)
    t0 = time.time()
    for i_gp, gp in enumerate(gamma_prime_grid):
        for i_rho, rho in enumerate(rho_grid):
            for i_theta, theta in enumerate(theta_grid):
                source = reduced_source(float(gp), float(rho), float(theta))
                for i_w, w in enumerate(w_grid):
                    try:
                        f_pure = f_schwinger(float(w), source, float(gp))
                    except (SchwingerCertificationError, ValueError) as exc:
                        raise SystemExit(
                            f'f_schwinger refused at gamma_prime={gp}, '
                            f'rho={rho}, theta={theta}, w={w}: {exc}'
                        ) from exc
                    carrier = born_lead_carrier(
                        float(w), source[0], source[1], float(gp))
                    residual = f_pure - carrier
                    real[i_gp, i_rho, i_theta, i_w] = residual.real
                    imag[i_gp, i_rho, i_theta, i_w] = residual.imag
        print(f'  gamma_prime={float(gp):.5f} done '
              f'({i_gp + 1}/{n_gp}, {time.time() - t0:.1f} s)')
    print(f'# {n_gp * n_rho * n_theta * n_w} nodes evaluated, '
          f'{time.time() - t0:.1f} s')
    return real, imag


def _output_path(scale: str, output: str | None) -> Path:
    """Resolve the artifact output path for the requested scale."""
    if output:
        return Path(output)
    if scale == 'full':
        return LowWShellChart._default_artifact_path()
    return Path(tempfile.gettempdir()) / 'low_w_shell_chart_smoke.npz'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--n-w', type=int, default=None,
                        help='dimensionless-frequency node count '
                             '(default: scale-dependent)')
    parser.add_argument('--output', type=str, default=None,
                        help='artifact output path (default: shipped path for '
                             'full, a temp-dir smoke artifact for smoke)')
    args = parser.parse_args()

    t0 = time.time()

    gamma_prime_grid = _gamma_prime_grid(args.scale)
    rho_grid = _rho_grid(args.scale)
    theta_grid = _theta_grid(args.scale)
    w_grid = _w_grid(args.scale, args.n_w)
    log_w_grid = np.log(w_grid)
    print(f'# grid ({args.scale}): '
          f'{len(gamma_prime_grid)} gamma_prime x {len(rho_grid)} rho x '
          f'{len(theta_grid)} theta x {len(w_grid)} w = '
          f'{len(gamma_prime_grid) * len(rho_grid) * len(theta_grid) * len(w_grid)} nodes')
    print(f'#   gamma_prime in [{gamma_prime_grid[0]:.4f}, '
          f'{gamma_prime_grid[-1]:.4f}]')
    print(f'#   rho         in [{rho_grid[0]:.4f}, {rho_grid[-1]:.4f}]')
    print(f'#   w           in [{w_grid[0]:.4f}, {w_grid[-1]:.3f}]')

    real, imag = _fill_coefficients(
        gamma_prime_grid, rho_grid, theta_grid, w_grid)

    provenance = {
        'driver': 'scripts/train_low_w_shell_chart.py',
        'sha': _provenance_sha(),
        'date': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'scale': args.scale,
        'grid_shape': [len(gamma_prime_grid), len(rho_grid),
                       len(theta_grid), len(w_grid)],
        'gamma_prime_range': [float(gamma_prime_grid[0]),
                              float(gamma_prime_grid[-1])],
        'rho_range': [float(rho_grid[0]), float(rho_grid[-1])],
        'theta_range': [float(theta_grid[0]), float(theta_grid[-1])],
        'w_range': [float(w_grid[0]), float(w_grid[-1])],
        'max_abs_residual': float(np.max(np.abs(real + 1j * imag))),
    }

    chart = LowWShellChart(
        gamma_prime_grid=gamma_prime_grid, rho_grid=rho_grid,
        theta_grid=theta_grid, log_w_grid=log_w_grid,
        real_coeffs=real, imag_coeffs=imag, provenance=provenance)

    # Self-check: cubic interpolation is functional at a corner cell.
    test_R = chart.evaluate(w_grid, float(gamma_prime_grid[0]),
                            float(rho_grid[0]), float(theta_grid[0]))
    print(f'Self-check: chart.evaluate at corner -> |R|_max = '
          f'{np.max(np.abs(test_R)):.4e}')

    out_path = _output_path(args.scale, args.output)
    content_hash = _content_hash(
        chart.gamma_prime_grid, chart.rho_grid, chart.theta_grid,
        chart.log_w_grid, chart.real_coeffs, chart.imag_coeffs)
    np.savez(
        out_path,
        gamma_prime_grid=chart.gamma_prime_grid,
        rho_grid=chart.rho_grid,
        theta_grid=chart.theta_grid,
        log_w_grid=chart.log_w_grid,
        real_coeffs=chart.real_coeffs,
        imag_coeffs=chart.imag_coeffs,
        provenance=np.array(json.dumps(chart.provenance)),
        content_hash=np.array(content_hash),
        schema=np.array(_SCHEMA),
    )
    print(f'\nSaved chart to {out_path}')
    print(f'  File size: {out_path.stat().st_size} bytes')

    # Round-trip self-check: the shipped loader must hash-verify the artifact
    # we just wrote (schema + content hash).
    loaded = LowWShellChart.load(out_path)
    assert np.array_equal(loaded.gamma_prime_grid, gamma_prime_grid)
    assert np.array_equal(loaded.rho_grid, rho_grid)
    assert np.array_equal(loaded.theta_grid, theta_grid)
    assert np.array_equal(loaded.log_w_grid, log_w_grid)
    assert np.array_equal(loaded.real_coeffs, real)
    assert np.array_equal(loaded.imag_coeffs, imag)
    print(f'  Round-trip: LowWShellChart.load() verified '
          f'(schema {_SCHEMA!r}, content hash matched)')
    print(f'  Total wall time: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
