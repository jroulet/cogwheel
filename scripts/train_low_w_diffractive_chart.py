#!/usr/bin/env python
"""Train the low-w near-fold / wall-band diffractive residual chart.

WHAT
----
Bakes the ``LowWDiffractiveChart`` NPZ artifact (loaded and hash-verified by
``cogwheel.lensing.low_w_diffractive_chart.LowWDiffractiveChart.load``) that
the positive-parity low-w diffractive serve
(``likelihood.LensedRelativeBinningLikelihood._low_w_diffractive_chart_serve``)
consults instead of the exact Schwinger engine for the near-fold shell and the
wall-approach band.

The stored object is the SMOOTH residual, never the raw amplification::

    r_new(w; gamma', rho, theta) = f_pure * sqrt(1 - gamma'**2) / F_ref(w)

with ``f_pure = f_schwinger(w, y_eig, gamma')`` the exact pure-shear engine
value and ``F_ref(w) = airy_fold_reference(w_grid, gamma', y_eig)`` the
non-vanishing uniform Airy fold reference (the q=p Wronskian form
``|F_ref|^2 ~ w**(1/3) Ai^2 + w**(-1/3) Ai'^2`` never vanishes).  ``F_ref``
replaces the exact point-mass prefactor ``C(w)`` ONLY; the
``sqrt(1 - gamma'**2)`` factor (the macro amplitude ``1 / sqrt(mu_pure)``
that diverges at the parity wall) STAYS in the residual.  Stripping ``F_ref``
leaves a smooth, bounded residual; the serve re-modulates it by ``F_ref`` and
the mass-sheet phase, which reproduces ``f_pure`` at ``kappa = 0`` exactly.

ORACLE
------
``f_schwinger`` is the OFFLINE oracle ONLY (never called on the serve path).
Each grid node is evaluated in the REDUCED / caustic-relative coordinates the
chart stores:

* ``gamma'``  -- reduced shear, in ``[GAMMA_LO, 1 - DELTA_GAMMA_P]``
  (the full positive-parity range below the wall; the near-fold shell exists
  at low ``gamma'`` too, so the grid spans it, not just the wall band).
* ``rho``     -- caustic-relative distance ``|y'| / |y_c(theta)|``, the SAME
  discriminator the near-fold fence uses (`_diffractive._caustic_rho` /
  `geometry.caustic_point`).  The grid edges extend beyond the shell
  ``[RHO_LO, RHO_HI]`` by MEASUREMENT of the wall-band rho spread (see
  `_measure_wall_rho_spread`), never a blanket literal.
* ``theta``   -- eigenframe source angle, folded to the D2 fundamental domain
  ``[0, pi/2]`` (the point-mass + shear potential is even in each source
  component, so the residual is D2-symmetric).
* ``w**(2/3)`` -- the two-thirds power of the dimensionless frequency, in
  ``[W_LO**(2/3), W_CEILING_SCHWINGER**(2/3)]``.  The ``w**(2/3)`` map
  densifies ``w`` at the low end, where the residual ``r ~ w**(1/6)`` varies
  fastest, so the frequency axis is UNIFORM in ``w**(2/3)``, never log-spaced.

The node source is reconstructed by inverting the fence discriminator via the
single-sourced `reduced_source(gamma', rho, theta)`:
``|y'| = rho * |caustic_point(gamma', theta)|`` and
``y_eig = |y'| (cos theta, sin theta)`` -- the same `geometry.caustic_point`
the serve uses, never a numerical root-find.

DE-RATE
-------
The interpolated residual is DE-RATED by a single scalar so the served
amplitude never over-serves on the calibration grid: the overshoot
``|r_interp| / |r_engine|`` is measured (sup over ``w``) on the FULL
calibration grid (every gamma', rho, theta node) AND the off-grid theta
MIDPOINTS (the points cubic interpolation is least constrained at), and the
de-rate is ``min(1.0, 1 / max_overshoot)`` -- never a hard ceiling, so a grid
with no overprediction bakes ``derate = 1.0`` and the served two-sided error
equals the raw interpolation error.  Cells whose SERVED two-sided error
``|derate * r_interp - r_engine| / |r_engine|`` still exceeds
``CERTIFICATION_BAR`` (the near-fold resonance-limited cells) are baked as a
per-cell ``declined_mask`` on the chart; the serve falls through to the exact
engine for a covered draw in a declined cell -- NEVER an amplitude scale.
Cells whose ``F_ref`` is UNBUILDABLE (no merging fold pair / a degenerate
soft-axis cubic / fold amplitude -- expected across the exterior wall band)
are ALSO baked as declined, distinct from engine refusals (which still
raise); see `_fill_coefficients`.  The 1e-4 bar is the real acceptance gate,
enforced on the de-rated served value.

Usage::

    python scripts/train_low_w_diffractive_chart.py --scale smoke  # in-build
    python scripts/train_low_w_diffractive_chart.py --scale full   # driver bake

The SMOKE scale is a small non-collinear subset (a few minutes) covering both
the near-fold shell at low ``gamma'`` (the ``gamma' = 0.3`` shell witness) and
the wall band (``gamma' = 0.8-0.9``); it proves the pipeline end-to-end and
emits a smoke artifact under the system temp dir (never the shipped path).
Its ``w`` axis is deliberately coarse (8 uniform-``w**(2/3)`` nodes), so its
de-rate / margin reflect the under-resolved frequency interpolation -- NOT a
shipped value.  With the ``w**(2/3)`` axis densifying the low-``w`` Airy-fold
variation, the smoke de-rate is expected far above the old log-spaced ``0.055``
(and its served error within ``1e-4``); the FULL bake's dense grid is what
drives the de-rate toward ``1.0``.
The FULL scale bakes the shipped ``cogwheel/data/low_w_diffractive_chart.npz``
-- a DRIVER post-build step (its grid is dense enough that the calibration is
a tens-of-minutes Schwinger sweep).
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

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal._born import DELTA_GAMMA_P
from cogwheel.lensing.chang_refsdal._diffractive import _caustic_rho
from cogwheel.lensing.chang_refsdal._schwinger import (
    W_CEILING_SCHWINGER, SchwingerCertificationError, f_schwinger)
from cogwheel.lensing.low_w_diffractive_chart import (
    _SCHEMA, LowWDiffractiveChart, RHO_HI, RHO_LO, _WALL_GAMMA_PRIME,
    _content_hash, airy_fold_reference, reduced_source)
from cogwheel.lensing.ppgo_map import CERTIFICATION_BAR

#: Lower bound of the reduced-shear ``gamma'`` grid.  Mirrors the lower gamma
#: of the ``w_low_fit`` calibration grid (``np.linspace(0.05, 0.5, 6)`` in
#: ``scripts/fit_diffractive_certificate.py``): below it the caustic is too
#: small to resolve the near-fold shell and ``w_low_fit`` is still calibrated,
#: so the chart cedes the sub-``GAMMA_LO`` band to it.
GAMMA_LO = 0.05

#: Lower bound of the dimensionless-frequency grid.  Slightly below the
#: production minimum ``w`` (10 Msun at 20 Hz ~ 0.025, via
#: ``lensing.waveform.dimensionless_frequency``), so a real band never
#: extrapolates below the trained frequency axis.
W_LO = 0.02

#: Source-position cap (mirrors ``lensing.prior._Y_SCALE_CAP``): the source
#: offset ``|y|`` is bounded by 3 Einstein radii in each component, bounding
#: the wall-band rho spread measured by `_measure_wall_rho_spread`.  Re-typed
#: (not imported) so this offline script does not pull the likelihood chain
#: through ``lensing.prior``.
_Y_SCALE_CAP = 3.0

#: Number of Monte-Carlo draws in the wall-band rho-spread measurement
#: (geometry-only ``find_images`` calls; no engine).
_RHO_SPREAD_SAMPLES = 20_000

#: Wall-band rho grid-edge margins (fractions of the measured spread): the
#: grid floor sits BELOW the measured minimum and the ceiling ABOVE the
#: measured maximum, so the wall band's all-rho box covers every measured
#: exterior source with margin.
_RHO_LO_MARGIN = 0.8
_RHO_HI_MARGIN = 1.1

#: Full-scale grid node counts (the driver bake).  ``n_w`` dominates the cost:
#: ``f_schwinger`` is ~30-350 ms/call on the double-double path, so
#: ``n_gamma * n_rho * n_theta * n_w`` ~ 30k calls is a tens-of-minutes sweep.
_FULL_N_GAMMA = 14
_FULL_N_RHO = 10
_FULL_N_THETA = 16
#: ``16`` w-nodes resolve the low-``w`` Airy-fold variation ``r ~ w**(1/6)``
#: on the uniform-``w**(2/3)`` axis (the map densifies ``w`` at the low end).
_FULL_N_W = 16

#: Smoke-scale dimensionless-frequency node count (spatial grids are the
#: hand-picked `_SMOKE_GAMMA_PRIMES` / `_SMOKE_RHOS` / `_SMOKE_THETAS`).
_SMOKE_N_W = 8

#: Smoke-scale reduced-shear nodes: the near-fold shell witness at
#: ``gamma' = 0.3`` (cf. ``NEAR_FOLD_DECLINED_WITNESSES``) plus the wall band
#: ``gamma' = 0.8-0.9`` and the wall boundary ``0.5``.
_SMOKE_GAMMA_PRIMES = (0.3, 0.5, 0.8, 0.9)

#: Smoke-scale rho nodes: the near-fold shell ``[RHO_LO, RHO_HI]`` plus one
#: wall-band exterior node (``2.0``) beyond the outer fence.
_SMOKE_RHOS = (RHO_LO, 1.0, RHO_HI, 2.0)

#: Smoke-scale theta node count over the D2 domain ``[0, pi/2]``.  8 nodes
#: resolve the even-harmonic basis ``cos(2k theta)`` up to ``k = 7`` (the
#: ``_DIFFRACTIVE_FIT_N_HARM`` ceiling) over the folded domain -- fewer than
#: 8 ALIASES the harmonics (the ``_N_THETAS = 32`` lesson from
#: ``scripts/fit_diffractive_certificate.py``).
_SMOKE_N_THETA = 8


def _provenance_sha() -> str:
    """Short ``git`` SHA of HEAD, or ``'unknown'`` outside a repo."""
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'], check=True,
            capture_output=True, text=True)
        return out.stdout.strip()
    except Exception:
        return 'unknown'


def _measure_wall_rho_spread(seed: int) -> tuple[float, float]:
    """Measured ``[rho_lo, rho_hi]`` of the wall-band exterior source set.

    Samples reduced shear ``gamma'`` uniformly over the wall band
    (``_WALL_GAMMA_PRIME`` to ``1 - DELTA_GAMMA_P``) and the source position
    ``y`` uniformly over ``[-_Y_SCALE_CAP, _Y_SCALE_CAP]^2`` (the reduced
    prior box), keeps the TWO-real-image (far-field exterior) draws -- the
    population the chart serve is gated on -- and returns the minimum and
    maximum ``rho = |y| / |caustic_point(gamma', theta_source)|`` (the SAME
    `_caustic_rho` discriminator the fence and serve use).

    The result is the wall band's "all-rho" spread: the miscalibrated
    caustic-relative discriminator spans well beyond the shell
    ``[RHO_LO, RHO_HI]`` (measured ~0.07 to ~7.7), because a genuinely
    exterior source near a caustic cusp direction can have ``rho << 1`` while
    a far source in the waist direction has ``rho >> 1.4``.  The full-scale
    rho grid is built over this measured spread (with `_RHO_LO_MARGIN` /
    `_RHO_HI_MARGIN` margin), never over a blanket literal.
    """
    rng = np.random.default_rng(seed)
    rho_lo = math.inf
    rho_hi = 0.0
    for _ in range(_RHO_SPREAD_SAMPLES):
        gamma_prime = float(
            rng.uniform(_WALL_GAMMA_PRIME, 1.0 - DELTA_GAMMA_P))
        y = rng.uniform(-_Y_SCALE_CAP, _Y_SCALE_CAP, 2)
        s = float(y[0] * y[0] + y[1] * y[1])
        if not s > 0.0:
            continue
        try:
            matrix = geometry.macro_matrix(gamma_prime, 0.0, 0.0)
            images = geometry.find_images(y, matrix)
        except geometry.LensDomainError:
            continue
        if len(images) != 2:
            continue
        theta = math.atan2(y[1], y[0])
        rho = _caustic_rho(abs(gamma_prime), s, theta)
        rho_lo = min(rho_lo, rho)
        rho_hi = max(rho_hi, rho)
    if not math.isfinite(rho_lo):
        raise SystemExit('wall-band rho-spread measurement found no '
                         'two-image exterior sources; the sample is degenerate')
    return rho_lo, rho_hi


def _gamma_prime_grid(scale: str) -> np.ndarray:
    """Reduced-shear grid, ascending, covering ``[GAMMA_LO, 1 - DELTA_GAMMA_P]``."""
    if scale == 'smoke':
        return np.asarray(_SMOKE_GAMMA_PRIMES, dtype=float)
    # Full: denser near the parity wall (small ``1 - gamma'``), where the
    # residual varies fastest.  ``geomspace(1 - GAMMA_LO, DELTA_GAMMA_P)``
    # descends, so ``1 - that`` ascends from ``GAMMA_LO`` to
    # ``1 - DELTA_GAMMA_P`` with the wall end pinned exactly.
    return 1.0 - np.geomspace(1.0 - GAMMA_LO, DELTA_GAMMA_P, _FULL_N_GAMMA)


def _theta_grid(scale: str) -> np.ndarray:
    """Eigenframe angle grid over the D2 fundamental domain ``[0, pi/2]``."""
    n = _SMOKE_N_THETA if scale == 'smoke' else _FULL_N_THETA
    return np.linspace(0.0, math.pi / 2.0, n)


def _w_grid(scale: str, n_w: int | None) -> np.ndarray:
    """Dimensionless-frequency grid, uniform in ``w**(2/3)``.

    ``w = linspace(W_LO**(2/3), W_CEILING_SCHWINGER**(2/3), n)**(3/2)``.
    The ``w**(2/3)`` map densifies ``w`` at the low end, where the residual
    ``r ~ w**(1/6)`` varies fastest (the Airy fold
    ``|F_ref|^2 ~ w**(1/3) Ai^2 + w**(-1/3) Ai'^2``), so a uniform-``w**(2/3)``
    axis resolves that variation far better than a log-spaced one.  The axis
    is stored on the chart as ``w23_grid = w**(2/3)`` and ``evaluate`` maps a
    query ``w`` to ``w**(2/3)`` before interpolating.
    """
    if n_w is None:
        n_w = _SMOKE_N_W if scale == 'smoke' else _FULL_N_W
    w23_lo = W_LO ** (2.0 / 3.0)
    w23_hi = W_CEILING_SCHWINGER ** (2.0 / 3.0)
    return np.linspace(w23_lo, w23_hi, n_w) ** (3.0 / 2.0)


def _rho_grid(scale: str, rho_lo_meas: float, rho_hi_meas: float) -> np.ndarray:
    """Caustic-relative distance grid.

    The near-fold shell ``[RHO_LO, RHO_HI]`` is always inside the grid; the
    full scale EXTENDS it to the measured wall-band spread (with margin), the
    smoke scale to a fixed shell-plus-one-exterior-node subset.
    """
    if scale == 'smoke':
        return np.asarray(_SMOKE_RHOS, dtype=float)
    rho_lo = min(RHO_LO, _RHO_LO_MARGIN * rho_lo_meas)
    rho_hi = max(RHO_HI, _RHO_HI_MARGIN * rho_hi_meas)
    return np.linspace(rho_lo, rho_hi, _FULL_N_RHO)


def _residual_at(w_grid: np.ndarray, gamma_prime: float, rho: float,
                 theta: float) -> np.ndarray | None:
    """Airy-anchored residual ``r = f_pure * sqrt(1 - gamma'^2) / F_ref``.

    Reconstructs the reduced eigenframe source ``y_eig`` via
    `reduced_source` (the single-sourced fence inversion, never re-inlined)
    and builds the non-vanishing uniform Airy fold reference ``F_ref`` via
    `airy_fold_reference` -- ``F_ref`` replaces ``prefactor_c`` ONLY, the
    ``sqrt(1 - gamma'^2)`` factor stays in the residual.  ``F_ref`` is sampled
    on the FULL ``w_grid``: its buildability (geometry / merging-fold-pair /
    soft-axis-cubic / fold-amplitude refusal) is ``w``-INDEPENDENT, its VALUE
    is ``w``-dependent.

    Returns the complex residual sampled on ``w_grid``, or ``None`` when
    ``F_ref`` is unbuildable (a sentinel the caller treats as a DECLINED cell,
    never an engine refusal).  An engine refusal (`SchwingerCertificationError`
    / ``ValueError`` from `f_schwinger`) is NOT caught here -- it propagates so
    the caller can count it toward ``n_refused`` (a grid-design bug) rather
    than a declined cell.
    """
    try:
        y_eig = reduced_source(gamma_prime, rho, theta)
        f_ref = airy_fold_reference(w_grid, gamma_prime, y_eig)
    except ValueError:
        # A geometry / source-reconstruction refusal (``LensDomainError`` or a
        # domain error in the caustic solve): F_ref is UNBUILDABLE at this
        # source -- a declined cell, NOT an engine refusal.
        return None
    if f_ref is None:
        return None
    residual = np.empty(w_grid.size, dtype=complex)
    for i_w, w in enumerate(w_grid):
        f_pure = f_schwinger(float(w), y_eig, gamma_prime)
        residual[i_w] = (f_pure * math.sqrt(1.0 - gamma_prime * gamma_prime)
                         / f_ref[i_w])
    return residual


def _fill_coefficients(gamma_prime_grid: np.ndarray, rho_grid: np.ndarray,
                       theta_grid: np.ndarray, w_grid: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Evaluate the engine residual at every grid node.

    Returns ``(real_coeffs, imag_coeffs, n_refused, unbuildable_mask)``.
    ``F_ref`` is built ONCE per ``(gamma', rho, theta)`` cell (its buildability
    is ``w``-independent); a cell whose ``F_ref`` is UNBUILDABLE is recorded in
    ``unbuildable_mask`` and skipped (no ``f_schwinger`` call, NOT counted in
    ``n_refused``) -- the exterior wall band is expected to be unbuildable.  A
    cell whose ``F_ref`` IS buildable but whose ``f_schwinger`` engine refuses
    (a ``SchwingerCertificationError`` / ``ValueError``) increments
    ``n_refused`` -- a nonzero ``n_refused`` is a grid design problem the
    caller raises on.
    """
    n_gp, n_rho, n_theta, n_w = (len(gamma_prime_grid), len(rho_grid),
                                 len(theta_grid), len(w_grid))
    real = np.empty((n_gp, n_rho, n_theta, n_w), dtype=float)
    imag = np.empty((n_gp, n_rho, n_theta, n_w), dtype=float)
    unbuildable = np.zeros((n_gp, n_rho, n_theta), dtype=bool)
    n_refused = 0
    t0 = time.time()
    n_nodes = n_gp * n_rho * n_theta * n_w
    for i_gp, gp in enumerate(gamma_prime_grid):
        for i_rho, rho in enumerate(rho_grid):
            for i_theta, theta in enumerate(theta_grid):
                try:
                    residual = _residual_at(w_grid, float(gp), float(rho),
                                            float(theta))
                except (SchwingerCertificationError, ValueError):
                    n_refused += 1
                    # Finite sentinel, NOT NaN: the cubic spline must stay
                    # finite.  The cell is declined (never served), so the
                    # fill value is irrelevant to served values -- it only
                    # keeps the metric probe's interpolator from NaNing on
                    # a neighbor.
                    real[i_gp, i_rho, i_theta] = 1.0
                    imag[i_gp, i_rho, i_theta] = 0.0
                    continue
                if residual is None:
                    unbuildable[i_gp, i_rho, i_theta] = True
                    # Finite sentinel (see above): the cell is declined, the
                    # fill never served, but the spline must be finite.
                    real[i_gp, i_rho, i_theta] = 1.0
                    imag[i_gp, i_rho, i_theta] = 0.0
                    continue
                real[i_gp, i_rho, i_theta] = residual.real
                imag[i_gp, i_rho, i_theta] = residual.imag
        print(f'  gamma_prime={float(gp):.5f} done '
              f'({i_gp + 1}/{n_gp}, {time.time() - t0:.1f} s)')
    n_unbuildable = int(np.sum(unbuildable))
    print(f'# {n_nodes - n_refused - n_unbuildable * n_w}/{n_nodes} nodes '
          f'evaluated, {n_refused} engine-refused, {n_unbuildable} F_ref-'
          f'unbuildable cells, {time.time() - t0:.1f} s')
    return real, imag, n_refused, unbuildable


def _off_grid_engine(gamma_prime_grid: np.ndarray, rho_grid: np.ndarray,
                     theta_grid: np.ndarray, w_grid: np.ndarray
                     ) -> tuple[list[tuple[float, float, float]],
                                list[np.ndarray], np.ndarray]:
    """Engine residuals at the off-grid theta MIDPOINTS (held-out witnesses).

    For each ``(gamma', rho)`` cell, probes the midpoint of every consecutive
    theta-node pair -- the angles cubic interpolation is least constrained at.
    An ``F_ref``-unbuildable midpoint is SKIPPED (recorded in the returned
    ``off_unbuildable`` mask, no ``f_schwinger`` call): it has no valid
    residual and cannot be served, so it contributes no interpolation-error
    witness.  Computed ONCE here so the de-rate and the margin report share
    the same engine values (no redundant ``f_schwinger`` sweep).
    """
    midpoints = 0.5 * (theta_grid[:-1] + theta_grid[1:])
    points: list[tuple[float, float, float]] = []
    values: list[np.ndarray] = []
    off_unbuildable = np.zeros((len(gamma_prime_grid), len(rho_grid),
                                len(midpoints)), dtype=bool)
    t0 = time.time()
    for i_gp, gp in enumerate(gamma_prime_grid):
        for i_rho, rho in enumerate(rho_grid):
            for i_mid, theta_mid in enumerate(midpoints):
                r_engine = _residual_at(w_grid, float(gp), float(rho),
                                        float(theta_mid))
                if r_engine is None:
                    off_unbuildable[i_gp, i_rho, i_mid] = True
                    continue
                points.append((float(gp), float(rho), float(theta_mid)))
                values.append(r_engine)
    print(f'# off-grid theta midpoints: {len(points)} buildable / '
          f'{off_unbuildable.size} total, {time.time() - t0:.1f} s')
    return points, values, off_unbuildable


def _iter_points(chart: LowWDiffractiveChart, gamma_prime_grid: np.ndarray,
                 rho_grid: np.ndarray, theta_grid: np.ndarray,
                 w_grid: np.ndarray, real: np.ndarray, imag: np.ndarray,
                 off_points: list[tuple[float, float, float]],
                 off_values: list[np.ndarray]):
    """Yield ``(r_interp, r_engine)`` complex arrays per calibration point.

    Only non-declined points are yielded: on-grid cells whose coefficients
    are the finite sentinel (an ``F_ref``-unbuildable or engine-refused cell)
    and off-grid midpoints in a declined neighborhood are SKIPPED -- they
    contribute no interpolation-error witness.  ``chart.evaluate`` is cubic
    interpolation only -- never an engine call.
    """
    unbuildable = ~np.all(np.isfinite(real), axis=-1)
    # A cell whose coefficients are the 1.0+0j sentinel is declined; the
    # sentinel is finite, so detect it by the unbuildable mask OR'd with a
    # value check (the sentinel is never a genuine residual).
    sentinel = np.all(np.isclose(real, 1.0, atol=1e-12)
                      & np.isclose(imag, 0.0, atol=1e-12), axis=-1)
    declined = unbuildable | sentinel  # (n_gp, n_rho, n_theta)

    n_gp, n_rho, n_theta = real.shape[:3]
    midpoints = 0.5 * (theta_grid[:-1] + theta_grid[1:])

    for i_gp, gp in enumerate(gamma_prime_grid):
        for i_rho, rho in enumerate(rho_grid):
            for i_theta, theta in enumerate(theta_grid):
                if declined[i_gp, i_rho, i_theta]:
                    continue
                r_engine = (real[i_gp, i_rho, i_theta]
                            + 1j * imag[i_gp, i_rho, i_theta])
                r_interp = chart.evaluate(w_grid, float(gp), float(rho),
                                          float(theta))
                yield r_interp, r_engine
    # Off-grid midpoints: evaluate ALL buildable ones.  The finite sentinel
    # for declined cells keeps the cubic spline finite everywhere, so an
    # off-grid midpoint's spline support never NaNs; a midpoint that is
    # itself unbuildable never appears in `off_points`/`off_values` (the
    # engine probe skips it).  The caller maps one served error per yielded
    # point onto the midpoint grid via `off_unbuildable`.
    for (gp, rho, theta_mid), r_engine in zip(off_points, off_values):
        r_interp = chart.evaluate(w_grid, gp, rho, theta_mid)
        yield r_interp, r_engine


def _node_metrics(chart: LowWDiffractiveChart, gamma_prime_grid: np.ndarray,
                  rho_grid: np.ndarray, theta_grid: np.ndarray,
                  w_grid: np.ndarray, real: np.ndarray, imag: np.ndarray,
                  off_points: list[tuple[float, float, float]],
                  off_values: list[np.ndarray]
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Per-point ``(overshoots, rel_errs)``, each sup over ``w``.

    ``overshoots`` is ``|r_interp| / |r_engine|`` (the un-de-rated magnitude
    ratio feeding the de-rate); ``rel_errs`` is
    ``|r_interp - r_engine| / |r_engine|`` (the raw interpolation accuracy).
    On-grid points read the already-stored coefficients; off-grid points read
    the precomputed ``off_values``.
    """
    overshoots: list[float] = []
    rel_errs: list[float] = []
    for r_interp, r_engine in _iter_points(
            chart, gamma_prime_grid, rho_grid, theta_grid, w_grid, real,
            imag, off_points, off_values):
        denom = np.abs(r_engine)
        ok = denom > 0.0
        overshoots.append(float(np.max(np.abs(r_interp[ok]) / denom[ok])))
        rel_errs.append(float(np.max(
            np.abs(r_interp[ok] - r_engine[ok]) / denom[ok])))
    return np.asarray(overshoots), np.asarray(rel_errs)


def _served_errors(chart: LowWDiffractiveChart, gamma_prime_grid: np.ndarray,
                   rho_grid: np.ndarray, theta_grid: np.ndarray,
                   w_grid: np.ndarray, real: np.ndarray, imag: np.ndarray,
                   off_points: list[tuple[float, float, float]],
                   off_values: list[np.ndarray], derate: float) -> np.ndarray:
    """Per-point SERVED two-sided error, sup over ``w``.

    ``|derate * r_interp - r_engine| / |r_engine|`` -- the value the
    acceptance gate (``<= CERTIFICATION_BAR``) is enforced on, measured on
    the de-rated value the serve actually returns.  ``derate`` must already
    be known (it depends on the worst overshoot, computed separately).
    """
    served: list[float] = []
    for r_interp, r_engine in _iter_points(
            chart, gamma_prime_grid, rho_grid, theta_grid, w_grid, real,
            imag, off_points, off_values):
        denom = np.abs(r_engine)
        ok = denom > 0.0
        served.append(float(np.max(
            np.abs(derate * r_interp[ok] - r_engine[ok]) / denom[ok])))
    return np.asarray(served)


def _margin_report(label: str, ratios: np.ndarray, raw_errs: np.ndarray,
                   served_errs: np.ndarray) -> dict:
    """Print the conservative/tight margin and the SERVED accuracy.

    ``ratios`` is the de-rated magnitude ratio ``|F_serve| / |F_engine|``
    (``= overshoot * derate``); ``raw_errs`` the raw interpolation two-sided
    error ``|r_interp - r_engine| / |r_engine|``; ``served_errs`` the SERVED
    two-sided error ``|derate * r_interp - r_engine| / |r_engine|`` -- the
    value ``n_within_bar`` is counted against ``CERTIFICATION_BAR`` (the real
    acceptance gate, on the de-rated value the serve returns).  Conservative =
    never over-serve (``ratio <= 1``); tight = within a factor 2 of the engine
    (``ratio >= 0.5``).  Returns the summary stats for the provenance block.
    """
    n = len(ratios)
    n_conservative = int(np.sum(ratios <= 1.0))
    n_tight = int(np.sum(ratios >= 0.5))
    n_within_bar = int(np.sum(served_errs <= CERTIFICATION_BAR))
    print(f'# {label}: {n_conservative}/{n} conservative (ratio <= 1), '
          f'{n_tight}/{n} tight (ratio >= 0.5); worst ratio '
          f'{ratios.max():.4f}, median {np.median(ratios):.4f}')
    print(f'# {label} SERVED rel-err: {n_within_bar}/{n} <= '
          f'{CERTIFICATION_BAR:.0e}; worst {served_errs.max():.3e}, '
          f'median {np.median(served_errs):.3e}, p90 '
          f'{np.percentile(served_errs, 90):.3e} (raw worst '
          f'{raw_errs.max():.3e})')
    return {
        'n': n,
        'n_conservative': n_conservative,
        'n_tight': n_tight,
        'n_within_bar': n_within_bar,
        'worst_ratio': float(ratios.max()),
        'worst_served_rel_err': float(served_errs.max()),
        'worst_raw_rel_err': float(raw_errs.max()),
    }


def _output_path(scale: str, output: str | None) -> Path:
    """Resolve the artifact output path for the requested scale."""
    if output:
        return Path(output)
    if scale == 'full':
        return LowWDiffractiveChart._default_artifact_path()
    return Path(tempfile.gettempdir()) / 'low_w_diffractive_chart_smoke.npz'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-w', type=int, default=None,
                        help='dimensionless-frequency node count '
                             '(default: scale-dependent)')
    parser.add_argument('--output', type=str, default=None,
                        help='artifact output path (default: shipped path for '
                             'full, a temp-dir smoke artifact for smoke)')
    args = parser.parse_args()

    t0 = time.time()

    rho_lo_meas, rho_hi_meas = _measure_wall_rho_spread(args.seed)
    print(f'# wall-band rho spread (measured): [{rho_lo_meas:.4f}, '
          f'{rho_hi_meas:.4f}] (shell fence [{RHO_LO}, {RHO_HI}])')

    gamma_prime_grid = _gamma_prime_grid(args.scale)
    rho_grid = _rho_grid(args.scale, rho_lo_meas, rho_hi_meas)
    theta_grid = _theta_grid(args.scale)
    w_grid = _w_grid(args.scale, args.n_w)
    w23_grid = w_grid ** (2.0 / 3.0)
    print(f'# grid ({args.scale}): {len(gamma_prime_grid)} gamma_prime x '
          f'{len(rho_grid)} rho x {len(theta_grid)} theta x {len(w_grid)} w '
          f'= {len(gamma_prime_grid) * len(rho_grid) * len(theta_grid) * len(w_grid)} nodes')
    print(f'#   gamma_prime in [{gamma_prime_grid[0]:.4f}, '
          f'{gamma_prime_grid[-1]:.4f}]')
    print(f'#   rho         in [{rho_grid[0]:.4f}, {rho_grid[-1]:.4f}]')
    print(f'#   w           in [{w_grid[0]:.4f}, {w_grid[-1]:.1f}] '
          f'(w^(2/3) in [{w23_grid[0]:.4f}, {w23_grid[-1]:.4f}])')

    real, imag, n_refused, unbuildable_mask = _fill_coefficients(
        gamma_prime_grid, rho_grid, theta_grid, w_grid)
    if n_refused:
        raise SystemExit(
            f'{n_refused} grid nodes refused the Schwinger engine; the grid '
            f'is degenerate. Adjust the scale/seed (e.g. back off the '
            f'gamma-prime ceiling below {1.0 - DELTA_GAMMA_P}) and re-run.')

    # Probe chart at unit de-rate to measure the overshoot / interpolation
    # accuracy (``evaluate`` is de-rate-independent).
    probe = LowWDiffractiveChart(
        gamma_prime_grid=gamma_prime_grid, rho_grid=rho_grid,
        theta_grid=theta_grid, w23_grid=w23_grid,
        real_coeffs=real, imag_coeffs=imag, derate=1.0)

    off_points, off_values, off_unbuildable = _off_grid_engine(
        gamma_prime_grid, rho_grid, theta_grid, w_grid)
    overshoots, rel_errs = _node_metrics(
        probe, gamma_prime_grid, rho_grid, theta_grid, w_grid, real, imag,
        off_points, off_values)

    max_overshoot = float(overshoots.max())
    # Never-over-serve scalar de-rate: reciprocal of the worst measured
    # overprediction, capped at 1.0 (no hard ceiling -- a grid with no
    # overprediction bakes derate = 1.0, so the served two-sided error
    # equals the raw interpolation error).
    derate = min(1.0, 1.0 / max_overshoot)
    note = ('' if args.scale == 'full' else
            ' (smoke grid is deliberately coarse; the full bake drives '
            'this toward 1.0)')
    print(f'# max un-de-rated overshoot = {max_overshoot:.4f} -> '
          f'de-rate = {derate:.4f}{note}')

    ratios = overshoots * derate
    served_errs = _served_errors(
        probe, gamma_prime_grid, rho_grid, theta_grid, w_grid, real, imag,
        off_points, off_values, derate)
    grid_n = int(np.sum(~unbuildable_mask))
    full_stats = _margin_report('full-grid', ratios, rel_errs, served_errs)
    grid_stats = _margin_report('grid', ratios[:grid_n], rel_errs[:grid_n],
                                served_errs[:grid_n])
    off_stats = _margin_report('off-grid', ratios[grid_n:],
                               rel_errs[grid_n:], served_errs[grid_n:])

    # Per-cell decline mask: a spatial (gamma', rho, theta) node is declined
    # when the SERVED two-sided error cannot meet CERTIFICATION_BAR in its
    # neighborhood -- the sup over the off-grid theta midpoints bracketing it
    # (the points cubic interpolation is least constrained at), plus the
    # uniform de-rate magnitude bias ``1 - derate`` (an on-grid node served
    # at derate < 1 carries that bias exactly).  A measured oracle decision,
    # never a hardcoded band.
    n_gp, n_rho, n_theta = (len(gamma_prime_grid), len(rho_grid),
                            len(theta_grid))
    # Map the buildable off-grid served errors back onto the full
    # (n_gp, n_rho, n_theta - 1) midpoint grid; an F_ref-unbuildable midpoint
    # stays 0.0 (no error contribution -- its adjacent cells are declined via
    # the F_ref-unbuildable grid-node mask instead).
    off_served = np.zeros((n_gp, n_rho, n_theta - 1), dtype=float)
    off_served[~off_unbuildable] = served_errs[grid_n:]
    uniform_bias = 1.0 - derate
    declined_by_error = np.zeros((n_gp, n_rho, n_theta), dtype=bool)
    for i_theta in range(n_theta):
        if i_theta == 0:
            cell_err = off_served[:, :, 0]
        elif i_theta == n_theta - 1:
            cell_err = off_served[:, :, n_theta - 2]
        else:
            cell_err = np.maximum(off_served[:, :, i_theta - 1],
                                  off_served[:, :, i_theta])
        declined_by_error[:, :, i_theta] = (
            np.maximum(cell_err, uniform_bias) > CERTIFICATION_BAR)
    declined_mask = unbuildable_mask | declined_by_error
    n_unbuildable_cells = int(np.sum(unbuildable_mask))
    n_declined_by_error = int(np.sum(declined_by_error))
    n_declined = int(np.sum(declined_mask))
    print(f'# decline mask: {n_declined}/{n_gp * n_rho * n_theta} spatial '
          f'cells declined ({n_unbuildable_cells} F_ref-unbuildable, '
          f'{n_declined_by_error} by served error > {CERTIFICATION_BAR:.0e})')

    provenance = {
        'driver': 'scripts/train_low_w_diffractive_chart.py',
        'sha': _provenance_sha(),
        'date': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'scale': args.scale,
        'seed': args.seed,
        'grid_shape': [len(gamma_prime_grid), len(rho_grid),
                       len(theta_grid), len(w_grid)],
        'gamma_prime_range': [float(gamma_prime_grid[0]),
                              float(gamma_prime_grid[-1])],
        'rho_range': [float(rho_grid[0]), float(rho_grid[-1])],
        'theta_range': [float(theta_grid[0]), float(theta_grid[-1])],
        'w_range': [float(w_grid[0]), float(w_grid[-1])],
        'measured_rho_spread': [rho_lo_meas, rho_hi_meas],
        'derate': derate,
        'max_overshoot': max_overshoot,
        'full_grid_margin': full_stats,
        'grid_margin': grid_stats,
        'off_grid_margin': off_stats,
        'n_declined_cells': n_declined,
        'n_unbuildable_cells': n_unbuildable_cells,
        'n_declined_by_error': n_declined_by_error,
        'declined_mask_shape': [n_gp, n_rho, n_theta],
    }

    chart = LowWDiffractiveChart(
        gamma_prime_grid=gamma_prime_grid, rho_grid=rho_grid,
        theta_grid=theta_grid, w23_grid=w23_grid,
        real_coeffs=real, imag_coeffs=imag, derate=derate,
        declined_mask=declined_mask, provenance=provenance)

    out_path = _output_path(args.scale, args.output)
    content_hash = _content_hash(
        chart.gamma_prime_grid, chart.rho_grid, chart.theta_grid,
        chart.w23_grid, chart.real_coeffs, chart.imag_coeffs, chart.derate,
        chart.declined_mask)
    np.savez(
        out_path,
        gamma_prime_grid=chart.gamma_prime_grid,
        rho_grid=chart.rho_grid,
        theta_grid=chart.theta_grid,
        w23_grid=chart.w23_grid,
        real_coeffs=chart.real_coeffs,
        imag_coeffs=chart.imag_coeffs,
        derate=np.array(chart.derate),
        declined_mask=chart.declined_mask,
        provenance=np.array(json.dumps(chart.provenance)),
        content_hash=np.array(content_hash),
        schema=np.array(_SCHEMA),
    )
    print(f'\nSaved chart to {out_path}')
    print(f'  File size: {out_path.stat().st_size} bytes')

    # Round-trip self-check: the shipped loader must hash-verify the artifact
    # we just wrote (schema + content hash).
    loaded = LowWDiffractiveChart.load(out_path)
    assert loaded.derate == derate
    assert np.array_equal(loaded.gamma_prime_grid, gamma_prime_grid)
    assert np.array_equal(loaded.w23_grid, w23_grid)
    assert np.array_equal(loaded.real_coeffs, real)
    assert np.array_equal(loaded.declined_mask, declined_mask)
    print(f'  Round-trip: LowWDiffractiveChart.load() verified '
          f'(schema {_SCHEMA!r}, content hash matched)')
    print(f'  Total wall time: {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
