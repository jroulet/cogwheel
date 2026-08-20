"""Measure and fit the engine-honest diffractive truncation certificate.

`_diffractive.w_low_fit` is an O(1) FITTED surface replacing the per-proposal
truncation-certificate SCAN.  The certificate is a SMOOTH function of the lens
parameters, so it is calibrated ONCE against the exact engine and baked as
module constants.  This script measures the engine-honest ceiling and fits the
surface.

Oracle
------
The honest ceiling is the largest ``w`` whose order-16 operator series stays
within `CERTIFICATION_BAR` of the EXACT `_schwinger.f_schwinger` engine::

    w_low_true = largest w in (0, W_CEILING_SCHWINGER] s.t.
        |diffractive_amplification(w, y, gamma, beta, kappa)
         - F_full_engine(w, y, gamma, beta, kappa)|
        / |F_full_engine(w, y, gamma, beta, kappa)| <= CERTIFICATION_BAR

measured with sup-over-w semantics (saturating to the ceiling when still
honest there).  `F_full_engine` is `f_schwinger` evaluated at the REDUCED
parameters and reconstructed back through the mass-sheet map
(``exp(0.5j*w*(log lam - kappa*s))/lam`` times the pure-shear value), so it
carries the SAME reconstruction prefactor as `diffractive_amplification` and
the relative error is well-defined at ``kappa != 0``.  The prefactor does NOT
cancel at ``kappa != 0``: comparing the full object directly against the
pure-shear `f_schwinger` would bias the ratio by the ``1/lam`` amplitude
(~43% at ``kappa = 0.3``), silently skipping every ``kappa != 0`` row.

Fit
---
``log w_low = P(log gamma', log s, log(1 - gamma')) + sum_k a_k cos(2k theta)
+ a_c * log(|y'| / |y_c(theta)|) + (1/(M+1)) * log(lam * sqrt_mu)`` with ``P``
a degree-2 polynomial, fitted by linear least squares on the three
log-features, the even harmonics ``cos(2 k theta)`` (``k = 1 .. 7``) and the
parametric-caustic feature ``log(|y'| / |y_c(theta)|)`` (the log ratio of the
reduced source offset to the reduced caustic radius in the same direction).
The ``(1/(M+1))`` amplitude feature is held FIXED (not fitted); only ``P``,
the harmonics and the caustic coefficient are regressed.  The exponentiated
surface is then DE-RATED by the reciprocal of the worst un-de-rated
over-prediction (clamped to ~0.85) so the shipped `w_low_fit` never
over-serves on the calibration grid.

The calibration grid is FENCED: rows whose reduced caustic ratio
``rho = |y'| / |y_c(theta)|`` (see `_diffractive._caustic_rho`) falls inside
the near-fold shell ``[RHO_LO, 1 + DELTA]`` are dropped from both
`_grid_points` and `_off_grid_points` (`_fence_excluded`), so the fit, the
de-rate and the margin report all operate on the fenced domain (probe domain
== training domain).  The deep interior (``rho < RHO_LO``) and the smooth
exterior (``rho > 1 + DELTA``) remain; the near-fold shell is declined by
`w_low_fit` (returns ``None`` -> the fold arm / exact engine).

The feature basis MUST match `_diffractive._fit_features` / the enumeration in
`_diffractive._fit_poly_exponents`; this script imports both from shipping code
(single source of truth) rather than re-deriving them.

Usage::

    python scripts/fit_diffractive_certificate.py --scale smoke   # in-build
    python scripts/fit_diffractive_certificate.py --scale full    # in-build

The SMOKE scale is a reduced-subset run (a few minutes) proving the pipeline
end to end on a non-collinear basis; the FULL scale (hundreds of grid points
plus ~240 off-grid midpoint probes, ~40 min serial) is the IN-BUILD FINAL
BAKE that finalizes the
coefficients -- the baked state MUST be the full emission block, never the
smoke fit.  The emission block is pasted verbatim into ``_diffractive.py``'s
module constants.
"""
from __future__ import annotations

import argparse
import cmath
import math
import subprocess
import time

import numpy as np

from cogwheel.lensing.chang_refsdal._diffractive import (
    _DIFFRACTIVE_FIT_FENCE_DELTA, _DIFFRACTIVE_FIT_FENCE_RHO_LO,
    _DIFFRACTIVE_FIT_LIP, _DIFFRACTIVE_FIT_N_HARM, _caustic_rho,
    _fit_poly_exponents, _fit_poly_features, _fit_features, _reduced_shear,
    diffractive_amplification, w_low_fit)
from cogwheel.lensing.chang_refsdal._schwinger import (
    W_CEILING_SCHWINGER, f_schwinger)
from cogwheel.lensing.ppgo_map import CERTIFICATION_BAR

#: Lower bound of the frequency grid probing the honest ceiling.  Small enough
#: that the series is essentially always honest at the floor (so the crossing
#: is genuinely found rather than assumed below the grid).
_W_MIN = 0.2

#: Hard ceiling on the de-rating factor: the fitted certificate is always
#: de-rated to at most this fraction of its raw value (>= 15% headroom),
#: even when the calibration grid shows no over-prediction anywhere.
_HARD_DERATE_CEILING = 0.85

#: Number of theta samples per (gamma, r) cell of the calibration grid.
#: The k-th even harmonic ``cos(2 k theta)`` has ``2k`` full cycles over
#: ``[0, 2 pi)``, so it needs ``> 4k`` samples to resolve (Nyquist).  With
#: `_DIFFRACTIVE_FIT_N_HARM = 7` (k = 1..7, up to 14 cycles), 32 samples
#: resolve the largest harmonic -- unlike the 8-theta grid that ALIASED the
#: harmonic basis (every ``k >= 2`` collapsed onto a low-order pattern) and
#: produced a degenerate fit.
_N_THETAS = 32

#: Number of theta-midpoint probes per (gamma, r) cell of the off-grid set
#: (a stride over the `_N_THETAS` midpoints).  8 keeps the full-scale
#: off-grid set at ~240 points while still exercising off-node thetas.
_OFF_GRID_PROBES_PER_CELL = 8


def _rot_minus_beta(beta: float) -> np.ndarray:
    """Eigenframe rotation ``R(-beta)`` (2x2)."""
    cos_b, sin_b = math.cos(beta), math.sin(beta)
    return np.array([[cos_b, sin_b], [-sin_b, cos_b]])


def _unreduced_source(r: float, theta: float, gamma: float, beta: float,
                      kappa: float) -> tuple[float, float]:
    """Physical lens-plane source for eigenframe polar ``(r, theta)``.

    ``y_eig = r (cos theta, sin theta)`` is the eigenframe offset; inverting
    ``y_eig = R(-beta) y / sqrt(lam)`` gives ``y = sqrt(lam) R(beta) y_eig``.
    """
    lam, _gamma_prime = _reduced_shear(gamma, kappa)
    root = math.sqrt(lam)
    cos_b, sin_b = math.cos(beta), math.sin(beta)
    y_eig1, y_eig2 = r * math.cos(theta), r * math.sin(theta)
    y1 = root * (cos_b * y_eig1 - sin_b * y_eig2)
    y2 = root * (sin_b * y_eig1 + cos_b * y_eig2)
    return y1, y2


def _fence_excluded(gamma: float, beta: float, kappa: float, r: float,
                    theta: float) -> bool:
    """True if the row's reduced caustic ratio ``rho`` is inside the near-fold shell.

    The fence discriminator ``rho = |y'| / |y_c(theta)| = _caustic_rho(
    abs(gamma'), r**2, theta)`` is 1.0 on the caustic, > 1 outside, < 1
    inside.  Rows in ``[RHO_LO, 1 + DELTA]`` are fenced OUT of the
    calibration grid (single-sourced from `_diffractive._caustic_rho` and the
    `_DIFFRACTIVE_FIT_FENCE_*` constants, never re-derived).
    """
    _lam, gamma_prime = _reduced_shear(gamma, kappa)
    rho = _caustic_rho(abs(gamma_prime), r * r, theta)
    return (_DIFFRACTIVE_FIT_FENCE_RHO_LO <= rho
            <= 1.0 + _DIFFRACTIVE_FIT_FENCE_DELTA)


def _unfenced_grid_points(scale: str, seed: int
                          ) -> list[tuple[float, float, float, float, float]]:
    """Raw calibration grid (no fence): ``(gamma, beta, kappa, r, theta)`` rows.

    ``r = sqrt(s)`` is the reduced source magnitude and ``theta`` the
    eigenframe angle.  ``full`` spans hundreds of points and ``smoke`` a
    deterministic reduced-subset run; both feed the IN-BUILD FINAL BAKE (the
    shipped constants are always the full-scale emission block).

    Each ``(gamma, r)`` cell samples ``_N_THETAS = 32`` equally-spaced thetas
    over ``[0, 2 pi)``.  The density matters: the k-th even harmonic
    ``cos(2 k theta)`` has ``2k`` full cycles, needing ``> 4k`` samples to
    resolve (Nyquist); 32 thetas resolve ``k <= 7`` (the shipped
    `_DIFFRACTIVE_FIT_N_HARM`).  The earlier 8-theta grid ALIASED every
    harmonic beyond ``k = 1`` and produced a degenerate fit.

    The ``smoke`` grid spans the FENCED domain: smooth-region anchor cells
    (``gamma in {0.1, 0.2, 0.3}`` x ``r in {0.5, 0.9}``), one deep-interior
    cell (``gamma = 0.5`` x ``r = 0.3``, ``rho < RHO_LO`` throughout -- so
    the fit is calibrated on the deep interior, whose engine-honest ceiling
    is ~4-6, NOT the DD ceiling), and one near-exterior high-gamma cell
    (``gamma = 0.5`` x ``r = 0.9``, ``rho > 1 + DELTA`` near the diagonal) --
    the region the parametric caustic feature is meant to capture, now
    sampled OUTSIDE the fence rather than at the (fenced-out) corner.
    """
    if scale == 'smoke':
        gammas = (0.1, 0.2, 0.3)
        radii = (0.5, 0.9)
        interior_gamma = 0.5
        interior_r = 0.3
        near_exterior_gamma = 0.5
        near_exterior_r = 0.9
        thetas = np.linspace(0.0, 2.0 * math.pi, _N_THETAS, endpoint=False)
        rows = [(g, 0.0, 0.0, r, float(t))
                for g in gammas for r in radii for t in thetas]
        rows += [(interior_gamma, 0.0, 0.0, interior_r, float(t))
                 for t in thetas]
        rows += [(near_exterior_gamma, 0.0, 0.0, near_exterior_r, float(t))
                 for t in thetas]
        rows += [(0.2, 0.7, 0.0, 0.9, 1.1),
                 (0.1, 0.0, 0.3, 0.5, 0.0),
                 (0.3, 0.7, 0.3, 0.9, 2.0)]
        return rows
    rng = np.random.default_rng(seed)
    rows: list[tuple[float, float, float, float, float]] = []
    gammas = np.linspace(0.05, 0.5, 6)
    radii = np.linspace(0.3, 1.3, 5)
    for gamma in gammas:
        for r in radii:
            for theta in np.linspace(0.0, 2.0 * math.pi, _N_THETAS,
                                     endpoint=False):
                rows.append((float(gamma), 0.0, 0.0, float(r), float(theta)))
    for _ in range(12):
        rows.append((float(rng.uniform(0.05, 0.4)), float(rng.uniform(-1.0, 1.0)),
                     float(rng.uniform(0.0, 0.4)), float(rng.uniform(0.3, 1.3)),
                     float(rng.uniform(0.0, 2.0 * math.pi))))
    return rows


def _grid_points(scale: str, seed: int
                 ) -> list[tuple[float, float, float, float, float]]:
    """Fenced calibration grid: `_unfenced_grid_points` minus the near-fold shell.

    Rows whose reduced caustic ratio ``rho`` falls in
    ``[RHO_LO, 1 + DELTA]`` are dropped (see `_fence_excluded`), so the fit,
    the de-rate and the margin report all operate on the fenced domain
    automatically (probe domain == training domain).
    """
    return [row for row in _unfenced_grid_points(scale, seed)
            if not _fence_excluded(*row)]


def _off_grid_points(scale: str, seed: int) -> list[tuple[float, float, float, float, float]]:
    """Fenced theta-midpoint probes, a theta-offset of `_unfenced_grid_points`.

    For each ``(gamma, r)`` cell of `_unfenced_grid_points`, the grid samples
    ``_N_THETAS`` thetas ``theta_j = 2 pi j / _N_THETAS``; these probes sit at
    the MIDPOINTS ``theta_j + pi / _N_THETAS`` between consecutive nodes --
    the points a harmonic fit is LEAST constrained at, so they are the honest
    out-of-sample witnesses.  A stride keeps the probe count at
    `_OFF_GRID_PROBES_PER_CELL` per cell (~240 total at full scale).  The
    probes are then FENCED (`_fence_excluded`), so the off-grid witness set
    lives on the fenced domain too.

    Derived as a theta-offset of `_unfenced_grid_points` output (single source
    of truth) rather than a hand-rolled second grid, so the two stay coupled.
    The off-grid rows are used for de-rating and the margin report ONLY,
    never the least-squares fit -- they remain a genuine held-out set.
    """
    cells: dict[tuple[float, float], list[float]] = {}
    for gamma, beta, kappa, r, theta in _unfenced_grid_points(scale, seed):
        if beta == 0.0 and kappa == 0.0:
            cells.setdefault((gamma, r), []).append(theta)
    offset = math.pi / _N_THETAS
    probes: list[tuple[float, float, float, float, float]] = []
    for (gamma, r), thetas in sorted(cells.items()):
        thetas = sorted(thetas)
        stride = max(1, len(thetas) // _OFF_GRID_PROBES_PER_CELL)
        for j in range(0, len(thetas), stride):
            probes.append((gamma, 0.0, 0.0, r, thetas[j] + offset))
    return [p for p in probes if not _fence_excluded(*p)]


def _engine_full(w: float, y_eig: np.ndarray, lam: float,
                 gamma_prime: float, kappa: float, s_reduced: float) -> complex:
    """Mass-sheet-reconstructed engine amplitude (the ``kappa >= 0`` oracle).

    `f_schwinger` evaluates the PURE-SHEAR amplification at reduced parameters
    (source ``y_eig``, shear ``gamma_prime``); `diffractive_amplification`
    evaluates the FULL lens-plane amplitude, which carries the mass-sheet
    reconstruction prefactor ``exp(0.5j*w*(log lam - kappa*s))/lam`` on top of
    that same pure-shear object (``s = |y_eig|**2`` is the reduced squared
    offset).  Reconstructing the engine value through the same map makes the
    relative-error oracle well-defined at ``kappa != 0``.  At ``kappa = 0``
    (``lam = 1``) the prefactor is the identity and this collapses to
    `f_schwinger` itself.
    """
    f_pure = f_schwinger(w, y_eig, gamma_prime)
    mass_sheet_phase = cmath.exp(
        0.5j * w * math.log(lam) - 0.5j * w * kappa * s_reduced)
    return mass_sheet_phase * f_pure / lam


def _measure_w_low_true(gamma: float, beta: float, kappa: float, y1: float,
                        y2: float, n_w: int) -> float | None:
    """Largest honest ``w`` up to the engine ceiling (sup-over-w semantics).

    ``None`` when the series is already beyond the bar at the grid floor.

    Notes
    -----
    ``n_w``-SENSITIVE near the fold: when the source sits OUTSIDE the
    caustic, ``rel(w)`` has narrow MARGINAL resonances (~0.1-wide ``w``
    windows where ``rel`` barely exceeds `CERTIFICATION_BAR`, ~1.1-1.2e-4
    vs 1e-4).  The coarse ``n_w`` scan samples those resonances
    INCONSISTENTLY, so the returned ceiling can jump between the ~3.5
    resonance floor and the ~6.9 smooth level for the same source at
    slightly different ``n_w`` (or theta) -- a spurious sharp angular step
    the fitted surface cannot follow (INS-1-001).  A robust measurement
    needs a dense ``w`` scan (spacing << 0.014 in log-w) to catch the
    resonances reliably; that is cost-prohibitive for the calibration grid
    and is left as a separate work package.
    """
    lam, gamma_prime = _reduced_shear(gamma, kappa)
    root = math.sqrt(lam)
    yp0, yp1 = y1 / root, y2 / root
    s_reduced = yp0 * yp0 + yp1 * yp1
    y_eig = _rot_minus_beta(beta) @ np.array([yp0, yp1], dtype=float)

    ws = np.logspace(math.log10(_W_MIN), math.log10(W_CEILING_SCHWINGER), n_w)
    for w in ws:
        wf = float(w)
        try:
            f_diff = diffractive_amplification(wf, (y1, y2), gamma, beta, kappa)
            f_schw = _engine_full(wf, y_eig, lam, gamma_prime, kappa, s_reduced)
        except Exception:
            return None
        denom = abs(f_schw)
        if not denom > 0.0:
            return None
        rel = abs(f_diff - f_schw) / denom
        if not math.isfinite(rel):
            return None
        if rel > CERTIFICATION_BAR:
            if wf == ws[0]:
                return None
            break
    else:
        return W_CEILING_SCHWINGER

    lo, hi = _W_MIN, wf
    for _ in range(24):
        mid = math.sqrt(lo * hi)
        try:
            f_diff = diffractive_amplification(mid, (y1, y2), gamma, beta, kappa)
            f_schw = _engine_full(mid, y_eig, lam, gamma_prime, kappa,
                                  s_reduced)
        except Exception:
            hi = mid
            continue
        denom = abs(f_schw)
        if not denom > 0.0:
            return None
        rel = abs(f_diff - f_schw) / denom
        if not math.isfinite(rel):
            return None
        if rel <= CERTIFICATION_BAR:
            lo = mid
        else:
            hi = mid
    return lo


def _fit_model(rows: list[tuple[float, float, float, float, float]],
               w_low_true: list[float], degree: int):
    """Linear-least-squares fit of the polynomial + harmonic + caustic coeffs."""
    n = len(rows)
    n_poly = len(_fit_poly_exponents(degree))
    n_harm = _DIFFRACTIVE_FIT_N_HARM
    design = np.empty((n, n_poly + n_harm + 1))
    target = np.empty(n)
    for i, (gamma, beta, kappa, r, theta) in enumerate(rows):
        lam, gamma_prime = _reduced_shear(gamma, kappa)
        sqrt_mu = 1.0 / math.sqrt(abs(lam * lam - gamma * gamma))
        s = r * r
        features = _fit_features(abs(gamma_prime), s, theta, lam, sqrt_mu)
        design[i] = features
        target[i] = (math.log(w_low_true[i])
                     - _DIFFRACTIVE_FIT_LIP * math.log(lam * sqrt_mu))
    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(design, target, rcond=None)
    return coeffs, n_poly, n_harm


def _evaluate_fit(coeffs, n_poly, n_harm, gamma: float, beta: float,
                  kappa: float, r: float, theta: float) -> float:
    """Undecorated (pre-derate) fitted log-``w_low``."""
    lam, gamma_prime = _reduced_shear(gamma, kappa)
    sqrt_mu = 1.0 / math.sqrt(abs(lam * lam - gamma * gamma))
    features = _fit_features(abs(gamma_prime), r * r, theta, lam, sqrt_mu)
    fitted = float(np.dot(coeffs, np.asarray(features)))
    fitted += _DIFFRACTIVE_FIT_LIP * math.log(lam * sqrt_mu)
    return math.exp(fitted)


def _provenance_sha() -> str:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'], check=True,
            capture_output=True, text=True)
        return out.stdout.strip()
    except Exception:
        return 'unknown'


def _measure_rows(rows: list[tuple[float, float, float, float, float]],
                  n_w: int, label: str):
    """Measure `_measure_w_low_true` over ``rows``.

    Returns ``(measured_rows, w_low_true, n_skipped)``: the subset of
    ``rows`` with a finite honest ceiling, the ceilings themselves, and the
    number of rows that refused to measure.
    """
    t0 = time.time()
    measured_rows: list[tuple[float, float, float, float, float]] = []
    w_low_true: list[float] = []
    n_skipped = 0
    for i, (gamma, beta, kappa, r, theta) in enumerate(rows):
        y1, y2 = _unreduced_source(r, theta, gamma, beta, kappa)
        w_true = _measure_w_low_true(gamma, beta, kappa, y1, y2, n_w)
        if w_true is None or not math.isfinite(w_true):
            n_skipped += 1
            continue
        measured_rows.append((gamma, beta, kappa, r, theta))
        w_low_true.append(w_true)
        if (i + 1) % 25 == 0:
            print(f'  ... {i + 1}/{len(rows)} {label} measured '
                  f'({time.time() - t0:.1f} s)')
    return measured_rows, w_low_true, n_skipped


def _margin_report(label: str, rows: list[tuple[float, float, float, float, float]],
                   w_low_true: list[float], coeffs, n_poly: int, n_harm: int,
                   derate: float) -> None:
    """Print the de-rated conservative/tight margin of ``coeffs`` on ``rows``."""
    if not rows:
        print(f'# {label} margin: (no measurable points)')
        return
    ratios = []
    for (gamma, beta, kappa, r, theta), w_true in zip(rows, w_low_true):
        w_fit = derate * _evaluate_fit(coeffs, n_poly, n_harm, gamma, beta,
                                       kappa, r, theta)
        w_fit = min(w_fit, W_CEILING_SCHWINGER)
        ratios.append(w_fit / w_true)
    ratios = np.asarray(ratios)
    n = len(ratios)
    n_served = int(np.sum(ratios <= 1.0))
    n_tight = int(np.sum(ratios >= 0.5))
    print(f'# {label} margin: {n_served}/{n} conservative (fit <= true), '
          f'{n_tight}/{n} tight (fit >= 0.5 * true), '
          f'worst ratio {ratios.max():.4f}, median {np.median(ratios):.4f}, '
          f'p90 {np.percentile(ratios, 90):.4f}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-w', type=int, default=16,
                        help='coarse log-spaced frequency grid points per source')
    parser.add_argument('--degree', type=int, default=2)
    args = parser.parse_args()

    t0 = time.time()
    raw_rows = _unfenced_grid_points(args.scale, args.seed)
    rows = _grid_points(args.scale, args.seed)
    off_rows = _off_grid_points(args.scale, args.seed)
    n_fenced = len(raw_rows) - len(rows)
    frac = n_fenced / len(raw_rows) if raw_rows else 0.0
    print(f'# grid: {len(rows)} points ({args.scale}), off-grid: '
          f'{len(off_rows)} points, n_w={args.n_w}, degree={args.degree}')
    print(f'# excluded-shell: {n_fenced}/{len(raw_rows)} grid rows fenced out '
          f'({frac:.1%}) -- prior-mass proxy for the declined near-fold shell')

    measured_rows, w_low_true, n_skipped = _measure_rows(rows, args.n_w, 'grid')
    off_measured, off_w_low_true, off_skipped = _measure_rows(
        off_rows, args.n_w, 'off-grid')
    print(f'# skipped: {n_skipped} grid, {off_skipped} off-grid')

    if not measured_rows:
        raise SystemExit('no measurable grid points; the grid is degenerate')

    coeffs, n_poly, n_harm = _fit_model(measured_rows, w_low_true, args.degree)
    poly_coeffs = tuple(float(c) for c in coeffs[:n_poly])
    harmonic_coeffs = tuple(float(c) for c in coeffs[n_poly:n_poly + n_harm])
    caustic_coeff = float(coeffs[n_poly + n_harm])

    # De-rating factor: the worst-residual reciprocal 1/max_overpred is the
    # natural de-rate, but it is always clamped to <= 0.85 (>= 15% headroom)
    # as a deliberate conservative margin against grid-sparsity / out-of-sample
    # over-prediction -- the fitted certificate is never served un-de-rated,
    # even when the calibration grid shows no over-prediction anywhere.  The
    # worst case is taken over BOTH the calibration grid AND the off-grid
    # midpoint probes, so the off-grid points participate in the de-rate.
    grid_overpreds = [
        _evaluate_fit(coeffs, n_poly, n_harm, gamma, beta, kappa, r, theta)
        / w_true
        for (gamma, beta, kappa, r, theta), w_true
        in zip(measured_rows, w_low_true)]
    off_overpreds = [
        _evaluate_fit(coeffs, n_poly, n_harm, gamma, beta, kappa, r, theta)
        / w_true
        for (gamma, beta, kappa, r, theta), w_true
        in zip(off_measured, off_w_low_true)]
    max_overpred = max(grid_overpreds + off_overpreds)
    derate = min(_HARD_DERATE_CEILING, 1.0 / max_overpred)
    print(f'# max un-de-rated over-prediction = {max_overpred:.4f} '
          f'(grid {max(grid_overpreds):.4f}, off-grid '
          f'{max(off_overpreds) if off_overpreds else 0.0:.4f}) '
          f'-> de-rate = {derate:.4f}')

    # Margin report on the de-rated fit (conservative / tight distribution),
    # printed separately for the calibration grid and the held-out off-grid
    # midpoint probes.  Uses the freshly-fitted surface (with the derate
    # applied), NOT the module's currently-baked `w_low_fit` constants, so
    # the reported margin reflects THIS fit.
    _margin_report('grid', measured_rows, w_low_true, coeffs, n_poly, n_harm,
                   derate)
    _margin_report('off-grid', off_measured, off_w_low_true, coeffs, n_poly,
                   n_harm, derate)

    print()
    print('# PASTE INTO _diffractive.py --------------------------------')
    print('_DIFFRACTIVE_FIT_DEGREE =', args.degree)
    print('_DIFFRACTIVE_FIT_POLY_COEFFS = (')
    for line in _chunk_floats(poly_coeffs, 4):
        print('   ', line)
    print(')')
    print('_DIFFRACTIVE_FIT_HARMONIC_COEFFS = (')
    for line in _chunk_floats(harmonic_coeffs, 4):
        print('   ', line)
    print(')')
    print('_DIFFRACTIVE_FIT_CAUSTIC_COEFF =', repr(caustic_coeff))
    print('_DIFFRACTIVE_FIT_DERATE =', repr(round(derate, 6)))
    print('# provenance: SHA', _provenance_sha(), f'{time.time() - t0:.1f} s')
    print('# -----------------------------------------------------------')


def _chunk_floats(values: tuple[float, ...], per_line: int) -> list[str]:
    lines = []
    for i in range(0, len(values), per_line):
        chunk = values[i:i + per_line]
        lines.append(', '.join(repr(v) for v in chunk) + ',')
    return lines


if __name__ == '__main__':
    main()
