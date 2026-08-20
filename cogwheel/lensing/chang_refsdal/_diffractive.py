"""Positive-parity low-w diffractive amplification and its truncation certificate.

WHAT
----
Analytic serve object for the Chang-Refsdal wave-optics amplification ``F(w)``
in the POSITIVE-PARITY, reduced-shear regime (reduced shear
``gamma' = gamma / (1 - kappa) < 1``).  Two public entry points:

* ``diffractive_amplification`` evaluates ``F_P(w)`` as the
  mass-sheet-reconstructed, truncated eigenframe shear-operator series
  ``F = exp[i gamma' D_0 / (2 w)] G_PM`` acting on the point-mass kernel
  ``G_PM(w, s)``, where ``D_0 = d_u**2 - d_v**2`` is the shear operator in the
  eigenframe and ``s = |y'|**2`` is the reduced squared source offset.
* ``w_low_fit`` returns the per-draw frequency ``w_low`` up to which the
  operator truncation is certified within the certification bar -- the
  truncation certificate.  It is an O(1) parametric surface fitted to the
  ENGINE-HONEST ceiling (the largest ``w`` whose order-``M`` series stays
  within the bar of the exact Schwinger engine), with coefficients baked by
  ``scripts/fit_diffractive_certificate.py``.  It is FENCED: it declines the
  near-fold shell (returning ``None`` so the draw falls through to the fold
  arm / exact engine) and serves the deep interior and the smooth exterior
  via the fit (see `_caustic_rho` and the `_DIFFRACTIVE_FIT_FENCE_*`
  constants).

WHY
---
This rung serves the band bottom for closures #1 and #3 (positive/Born
band-splits).  It computes VALUES ONLY; wiring it into the likelihood is WP2.

Conventions
-----------
* Reduced-shear (mass-sheet) map: ``lam = 1 - kappa``, ``y' = y / sqrt(lam)``,
  ``gamma' = gamma / lam`` -- the same rescaling as
  ``operator._mass_sheet_map``, mirrored here on the scalar path so this module
  does not pull a NumPy object onto the value path.
* F009-S Morse convention: the positive-parity macro image has Morse index 0,
  so the DC limit is ``F_P(w -> 0) = sqrt(mu_macro) = 1 / sqrt((1 - kappa)**2 -
  gamma**2)``, NOT 1.  ``F -> 1`` is the pure point-mass (``gamma = kappa = 0``)
  special case only; the resummed operator tail supplies the shear factor
  ``(1 - gamma'**2)**(-1/2)`` and the ``1/lam`` reconstruction prefactor
  supplies the rest.
* The exact ``w*ln(w)`` diffraction phase ``C(w)`` is carried INSIDE the kernel
  derivatives (``point_mass_g_derivatives`` bakes ``prefactor_c`` into every
  ``values[k]``); it is never bounded by the truncation certificate and is
  never re-applied here -- re-multiplying by ``prefactor_c`` would double-count
  it.

The operator series is the float64 truncation of the INDEPENDENT mpmath
``_oracle_fop`` contraction (``tests/test_lensing_fast_path.py``); the two share
no accumulation path (F002).  See ``_operator_step`` for the eigenframe shear
operator, transcribed verbatim from that oracle.
"""
from __future__ import annotations

import cmath
import math

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal._born import DELTA_GAMMA_P, _born_factors
from cogwheel.lensing.chang_refsdal._hyp1f1 import (
    HypergeometricDomainError, point_mass_g_derivatives)
from cogwheel.lensing.chang_refsdal._schwinger import W_CEILING_SCHWINGER

#: Default operator-series truncation order ``M``.  The tail decays like
#: ``(gamma' s w / 2)**n / n!``, so sixteen orders are ample across the
#: certified low-w band; the certificate reads the first omitted term at
#: order ``M + 1``.
_DEFAULT_MAX_ORDER = 16

#: ---- Calibration provenance ------------------------------------------
#: The fit constants below are PROVISIONAL -- baked by
#: `scripts/fit_diffractive_certificate.py` at SMOKE scale on the FENCED
#: domain (the near-fold shell ``rho`` in ``[RHO_LO, 1 + DELTA]`` is
#: excluded; see `_DIFFRACTIVE_FIT_FENCE_*`), and MUST be re-baked at FULL
#: scale by the driver before release.  Smoke margins (interior-inclusive
#: grid, radii down to r=0.1): de-rate 0.85, grid 178/178 conservative /
#: 178/178 tight, off-grid 44/44 conservative / 44/44 tight, excluded-shell
#: 49/227 grid rows (21.6%).  Provenance SHA 362c58e (526.9 s).  Final bake:
#: ``python scripts/fit_diffractive_certificate.py --scale full`` and paste
#: the emission block verbatim.
#:
#: NEAR-FOLD FENCE: the corner (high-gamma / small-r) region -- where the
#: honest ceiling collapses steeply toward the positive-parity wall and
#: narrow MARGINAL resonances (INS-1-001) make the ceiling
#: measurement-unstable -- is FENCED OUT of the fit's domain.  `w_low_fit`
#: serves the deep interior (``rho < RHO_LO``) AND the smooth exterior
#: (``rho > 1 + DELTA``) via the fit; the near-fold shell
#: (``RHO_LO <= rho <= 1 + DELTA``) is DECLINED (returns ``None`` -> the draw
#: falls through to the fold arm / exact engine).  Fencing the shell lets
#: the de-rate return toward the 0.85 hard floor instead of paying for a
#: corner the diffractive rung should not own.

#: Degree of the log-log polynomial ``P`` in the fitted certificate
#: `w_low_fit` (see `_fit_poly_features`).  The three log-features are
#: ``(log gamma', log s, log(1 - gamma'))``; degree 2 is a low-order fit that
#: still captures the curvature of the engine-honest ceiling.
_DIFFRACTIVE_FIT_DEGREE = 2

#: Polynomial coefficients of `w_low_fit`, one per monomial of the
#: `_DIFFRACTIVE_FIT_DEGREE` log-feature basis (same enumeration as
#: `_fit_poly_exponents`).  Baked by `scripts/fit_diffractive_certificate.py`.
#: FINAL -- full bake at 5adb029 (2293.8 s); grid 950/950 + off-grid 234/234
#: conservative AND tight, de-rate 0.716, median 0.71
_DIFFRACTIVE_FIT_POLY_COEFFS = (
    -110.87645612139974, -40.98597354907916, -0.3094278602205789, -52.07121542457959,
    -2.4344788160006625, 0.061930791573770294, -0.05702555068850229, 106.46881823090128,
    -0.11829682797626047, -6.468749406722436,
)

#: Number of even harmonics in `w_low_fit` (``cos(2 k theta)`` for
#: ``k = 1 .. _DIFFRACTIVE_FIT_N_HARM``).
#:
#: A FIT property (the angular basis size the calibration grid resolves),
#: DECOUPLED from the series truncation order `_DEFAULT_MAX_ORDER` (16).
#: The k-th even harmonic ``cos(2 k theta)`` has ``2k`` full cycles over
#: ``[0, 2 pi)``, needing ``> 4k`` theta samples to resolve (Nyquist); the
#: calibration grid samples 32 thetas per cell, so ``k <= 7`` is the
#: largest alias-free harmonic set.  The earlier 8-theta grid ALIASED
#: every harmonic beyond ``k = 1`` onto a low-order pattern, producing a
#: degenerate fit that oscillated catastrophically off-grid.
_DIFFRACTIVE_FIT_N_HARM = 7

#: Even-harmonic coefficients ``a_k`` of `w_low_fit`, for
#: ``k = 1 .. _DIFFRACTIVE_FIT_N_HARM``.  Baked by
#: `scripts/fit_diffractive_certificate.py`.
#: FINAL -- full bake at 5adb029
_DIFFRACTIVE_FIT_HARMONIC_COEFFS = (
    0.036741585752783266, -0.3060162587978441, -0.020324773538662955, 0.0756720902975168,
    -0.0034289910592988446, -0.02932422939136277, 0.005841489227116231,
)

#: Coefficient of the parametric-caustic feature
#: ``log(|y'| / |y_c(theta)|)`` (see `_fit_features`).  Fitted by
#: `scripts/fit_diffractive_certificate.py`; expected NEGATIVE -- the
#: honest ceiling dips as the source approaches / crosses the fold, so the
#: caustic log-ratio is anti-correlated with ``log w_low``.
#: FINAL -- full bake at 5adb029
_DIFFRACTIVE_FIT_CAUSTIC_COEFF = -0.6250079394041849

#: De-rating factor applied to the exponentiated fit so the fitted ceiling is
#: CONSERVATIVE on the calibration grid (never above the engine-honest
#: ceiling).  Baked by `scripts/fit_diffractive_certificate.py` as the
#: reciprocal of the worst un-de-rated over-prediction (clamped to <= 0.85).
#: FINAL -- full bake at 5adb029
_DIFFRACTIVE_FIT_DERATE = 0.715964

#: Coefficient of the ``log(lam * sqrt_mu)`` feature, held at ``1 / (M + 1)``
#: by construction (the amplitude-space normalisation the fitted surface
#: inherits from the certificate's relative-error currency).
_DIFFRACTIVE_FIT_LIP = 1.0 / (_DEFAULT_MAX_ORDER + 1)

#: Hard ceiling of the fitted certificate: the double-double Schwinger engine
#: domain (`W_CEILING_SCHWINGER`), imported -- never re-typed.
_DIFFRACTIVE_FIT_CEILING = W_CEILING_SCHWINGER

#: Near-fold fence: the fit is NOT certified inside a shell around the
#: directional caustic.  ``rho = |y'| / |y_c(theta)|`` (see `_caustic_rho`)
#: is the reduced-caustic distance ratio -- 1.0 ON the caustic, > 1 outside
#: (smooth region), < 1 inside.  ``w_low_fit`` declines ONLY the near-fold
#: shell (``RHO_LO <= rho <= 1 + DELTA`` -> returns ``None`` so the draw
#: falls through to the fold arm / exact engine); the deep interior
#: (``rho < _DIFFRACTIVE_FIT_FENCE_RHO_LO``) and the smooth exterior
#: (``rho > 1 + DELTA``) are BOTH served by the fit.  Fencing the shell lets
#: the de-rate return toward the 0.85 hard floor instead of paying for a
#: corner the diffractive rung should not own.
#:
#: PROVISIONAL -- tuned at the driver full bake.  ``rho`` is a
#: monotone-but-miscalibrated distance-to-fold discriminator: it is the ratio
#: of the source offset to the CRITICAL-CURVE-derived caustic radius (a
#: critical-curve vs source-angle proxy) and under-estimates the true
#: distance-to-fold by up to ~1.75x, so the fence boundary is a conservative
#: guard, not an exact fold map.
_DIFFRACTIVE_FIT_FENCE_RHO_LO = 0.6

#: Outer shell boundary ``RHO_HI = 1.0 + DELTA ~ 1.4``.  ``DELTA = 0.4`` sits
#: in the Professor's 0.35-0.5 range, lower-bounded by the corner defect at
#: ``rho ~ 1.34`` (the marginal-resonance fold dip, INS-1-001).
_DIFFRACTIVE_FIT_FENCE_DELTA = 0.4

#: Calibration-domain gamma ceiling.  The fitted surface is calibrated on
#: ``gamma in [0.05, 0.5]``; above it the ``log(1 - gamma')`` feature
#: EXTRAPOLATES (its calibrated range ends at ``log(1 - 0.5) ~ -0.7``, and
#: at ``gamma' -> 1`` it runs to ``-inf``, blowing the fitted value up to a
#: ``min(w_fit, CEILING) = 60`` clip).  The order-16 series has a
#: convergence-radius collapse at the parity wall (gamma' -> 1): the
#: ``sqrt(mu_macro) = 1/sqrt(1 - gamma'^2)`` divergence is a square-root
#: branch point not representable at any practical order (40% error at
#: M=16, 10% even at M=64, at gamma'=0.98).  So above this ceiling the
#: diffractive rung DECLINES (returns None) and the draw routes to the
#: exact Schwinger engine, which is the correct serve there.  The wall
#: band is ~6-12% of shear prior mass; engine-serving it is a performance
#: cost, never a correctness loss.
_DIFFRACTIVE_FIT_GAMMA_MAX = 0.5


class DiffractiveDomainError(geometry.LensDomainError):
    """Lens parameters fall outside the positive-parity diffractive regime.

    Raised when the reduced shear reaches the parity wall
    (``gamma' >= 1 - DELTA_GAMMA_P``) or the convergence is non-physical
    (``1 - kappa <= 0``).  Positive parity ONLY lives on this rung; macro
    saddles are out of scope (mirrors ``_born.BornDomainError``).
    """


def _reduced_shear(gamma: float, kappa: float) -> tuple[float, float]:
    """Reduced (mass-sheet) shear ``gamma' = gamma / (1 - kappa)``.

    Parameters
    ----------
    gamma, kappa : float
        External shear magnitude and convergence.

    Returns
    -------
    lam : float
        ``1 - kappa``.
    gamma_prime : float
        Reduced shear ``gamma / lam`` (signed).

    Raises
    ------
    DiffractiveDomainError
        If ``lam <= 0`` or ``abs(gamma') >= 1 - DELTA_GAMMA_P``.  Refusing at
        the wall keeps near-critical draws from returning an optimistically
        small value instead of declining.
    """
    lam = 1.0 - float(kappa)
    if not lam > 0.0:
        raise DiffractiveDomainError(
            f'Cannot reduce the mass sheet for kappa = {kappa}: '
            f'1 - kappa = {lam} must be positive.')
    gamma_prime = float(gamma) / lam
    wall = 1.0 - DELTA_GAMMA_P
    if not abs(gamma_prime) < wall:
        raise DiffractiveDomainError(
            f"Reduced shear |gamma'| = {abs(gamma_prime)} reaches the "
            f'positive-parity wall 1 - DELTA_GAMMA_P = {wall}; macro saddles '
            f'are out of scope for the diffractive rung.')
    return lam, gamma_prime


def _operator_step(state: dict[tuple[int, int], int]
                   ) -> dict[tuple[int, int], int]:
    """Apply the eigenframe shear operator ``D_0 = d_u**2 - d_v**2`` once.

    ``state`` maps ``(a, b) -> int`` coefficient of ``u**a v**b G^(k)``.
    Transcribed VERBATIM from ``_oracle_operator_step`` in
    ``tests/test_lensing_fast_path.py``; coefficients stay exact Python ints
    (no floating-point spent here).
    """
    new: dict[tuple[int, int], int] = {}

    def add(key: tuple[int, int], value: int) -> None:
        new[key] = new.get(key, 0) + value

    for (a, b), coeff in state.items():
        if a >= 2:
            add((a - 2, b), coeff * a * (a - 1))
        add((a, b), coeff * (4 * a + 2))
        add((a + 2, b), coeff * 4)
        if b >= 2:
            add((a, b - 2), -coeff * b * (b - 1))
        add((a, b), -coeff * (4 * b + 2))
        add((a, b + 2), -coeff * 4)
    return {key: value for key, value in new.items() if value}


def _kernel_length(w: float, s: float) -> int:
    """Adaptive term count for ``point_mass_g_derivatives``.

    Mirrors ``operator._series_length``: ``w*sqrt(s) + 8 sqrt(w*sqrt(s)) + 20``
    terms, enough that the kernel's reported per-k truncation tail is
    negligible across the whole derivative ladder.
    """
    product = w * math.sqrt(s)
    return int(math.ceil(product + 8.0 * math.sqrt(product) + 20.0))


def _operator_terms(w: float, u0: float, v0: float, s: float,
                    gamma_prime: float, max_order: int) -> list[complex]:
    """Per-order operator-series terms ``t_0, ..., t_{max_order}``.

    ``t_n = alpha**n / n! * <D_0**n G_PM>`` with ``alpha = i gamma' / (2 w)``,
    evaluated at the eigenframe source ``(u0, v0)`` and reduced offset ``s``.
    The order-``n`` contraction reaches derivative ``2 n`` of ``G_PM`` on its
    leading monomials, so the kernel ladder is built to ``2 * max_order`` and
    the monomial-power tables to ``2 * max_order + 3``.  Float64 truncation of
    the mpmath ``_oracle_fop`` contraction; the returned list carries the terms
    BEFORE the ``(1/lam)`` mass-sheet reconstruction.
    """
    n_terms = _kernel_length(w, s)
    values = point_mass_g_derivatives(w, s, 2 * max_order, n_terms)[0]

    n_powers = 2 * max_order + 3
    u_pow = [1.0] * n_powers
    v_pow = [1.0] * n_powers
    for i in range(1, n_powers):
        u_pow[i] = u_pow[i - 1] * u0
        v_pow[i] = v_pow[i - 1] * v0

    def evaluate(state: dict[tuple[int, int], int], order: int) -> complex:
        return sum(coeff * u_pow[a] * v_pow[b] * values[(a + b) // 2 + order]
                   for (a, b), coeff in state.items())

    alpha = 1j * gamma_prime / (2.0 * w)
    terms: list[complex] = []
    state: dict[tuple[int, int], int] = {(0, 0): 1}
    factorial = 1.0
    for n in range(max_order + 1):
        if n:
            factorial *= n
            state = _operator_step(state)
        terms.append(alpha ** n / factorial * evaluate(state, n))
    return terms


def diffractive_amplification(w: float, y, gamma: float, beta: float = 0.0,
                              kappa: float = 0.0,
                              max_order: int = _DEFAULT_MAX_ORDER) -> complex:
    """Positive-parity diffractive amplification ``F_P(w)``.

    Parameters
    ----------
    w : float
        Dimensionless frequency, ``0 < w <= W_MAX_CERTIFIED``.
    y : array_like, shape (2,)
        Source position in the (unreduced) lens plane.
    gamma, beta, kappa : float
        External shear magnitude, shear orientation (radians), and convergence.
    max_order : int
        Operator-series truncation order ``M``.

    Returns
    -------
    complex
        The reconstructed amplification.  As ``w -> 0`` this tends to
        ``sqrt(mu_macro)`` (F009-S), NOT 1.

    Raises
    ------
    DiffractiveDomainError
        At the positive-parity wall (via `_reduced_shear`) or for ``w <= 0``.
    HypergeometricDomainError
        If ``(w, s)`` leaves the certified kernel domain (propagated from
        `point_mass_g_derivatives`).
    ValueError
        If ``y`` does not have shape ``(2,)``.
    """
    w = float(w)
    if not w > 0.0:
        raise DiffractiveDomainError(
            f'Diffractive amplification requires w > 0, got {w}.')
    lam, gamma_prime = _reduced_shear(gamma, kappa)
    y = np.asarray(y, dtype=float)
    if y.shape != (2,):
        raise ValueError(f'Source position must have shape (2,), got {y.shape}.')

    root = math.sqrt(lam)
    yp0, yp1 = float(y[0]) / root, float(y[1]) / root
    s = yp0 * yp0 + yp1 * yp1
    z_eig = cmath.exp(-1j * float(beta)) * complex(yp0, yp1)

    terms = _operator_terms(w, z_eig.real, z_eig.imag, s, gamma_prime,
                            int(max_order))
    total = sum(terms)
    recon = cmath.exp(0.5j * w * math.log(lam)
                      - 0.5j * w * float(kappa) * s
                      + 0.5j * w * s) / lam
    return complex(recon * total)


def _fit_poly_exponents(degree: int) -> tuple[tuple[int, int, int], ...]:
    """Monomial exponent triples ``(i, j, k)`` for the log-feature polynomial.

    Enumerated in non-decreasing total degree, then lexicographically.  The
    log-features are ordered ``(log gamma', log s, log(1 - gamma'))``, so
    monomial ``(i, j, k)`` contributes ``log(gamma')**i * log(s)**j *
    log(1 - gamma')**k``.  Shared by `w_low_fit` and
    `scripts/fit_diffractive_certificate.py` -- the two MUST agree or the
    baked coefficients evaluate against the wrong basis.
    """
    exponents: list[tuple[int, int, int]] = []
    for total in range(degree + 1):
        for i in range(total + 1):
            for j in range(total - i + 1):
                exponents.append((i, j, total - i - j))
    return tuple(exponents)


def _fit_poly_features(log_gamma_prime: float, log_s: float,
                       log_one_minus: float,
                       exponents: tuple[tuple[int, int, int], ...]
                       ) -> tuple[float, ...]:
    """Evaluate the `_DIFFRACTIVE_FIT_DEGREE` polynomial basis.

    ``0**0`` is 1, so the constant monomial ``(0, 0, 0)`` yields 1; every
    other monomial is a plain product of the three (positive) log-features.
    """
    features = [log_gamma_prime, log_s, log_one_minus]
    return tuple(
        math.prod(features[e] ** p for e, p in zip((0, 1, 2), exps))
        for exps in exponents)


def _caustic_rho(gamma_prime: float, s: float, theta: float) -> float:
    """Reduced caustic distance ratio ``rho = sqrt(s) / |y_c(theta)|``.

    The SINGLE source of the near-fold fence discriminator (shared by
    `w_low_fit`, `_fit_features`, and
    `scripts/fit_diffractive_certificate.py`).  ``sqrt(s) = |y'|`` is the
    reduced source offset and ``|y_c(theta)| = |geometry.caustic_point(
    gamma_prime, theta)|`` the reduced caustic radius in the same eigenframe
    direction; ``rho`` is 1.0 on the caustic, > 1 outside, < 1 inside.  Pure
    python, O(1) -- no numba and no ndarray on the `w_low_fit` value path.
    """
    caustic = geometry.caustic_point(gamma_prime, theta)
    return math.sqrt(s) / math.hypot(caustic[0], caustic[1])


def _fit_features(gamma_prime: float, s: float, theta: float, lam: float,
                  sqrt_mu: float) -> tuple[float, ...]:
    """Feature vector of the fitted certificate (poly + harmonics + caustic).

    The ``_DIFFRACTIVE_FIT_N_HARM`` entries after the polynomial are
    ``cos(2 k theta)`` for ``k = 1 .. _DIFFRACTIVE_FIT_N_HARM``; the final
    trailing entry is the parametric-caustic feature
    ``log(|y'| / |y_c(theta)|)``, the log ratio of the reduced source
    offset ``|y'| = sqrt(s)`` to the reduced caustic radius
    ``|y_c(theta)| = |geometry.caustic_point(gamma_prime, theta)|`` in the
    same eigenframe direction.  It is negative well inside the caustic,
    positive outside, and passes through zero at the fold -- capturing the
    steep ceiling collapse toward the positive-parity wall.  The
    ``log(lam * sqrt_mu)`` feature is NOT part of the fitted vector -- its
    coefficient is held at `_DIFFRACTIVE_FIT_LIP` by construction (added by
    `w_low_fit`, not fitted).
    """
    poly = _fit_poly_features(
        math.log(abs(gamma_prime)), math.log(s),
        math.log(1.0 - abs(gamma_prime)),
        _fit_poly_exponents(_DIFFRACTIVE_FIT_DEGREE))
    harmonics = tuple(math.cos(2.0 * k * theta)
                      for k in range(1, _DIFFRACTIVE_FIT_N_HARM + 1))
    caustic_feature = math.log(_caustic_rho(gamma_prime, s, theta))
    return poly + harmonics + (caustic_feature,)


def w_low_fit(y, gamma: float, beta: float = 0.0, kappa: float = 0.0, *,
              w_hi: float | None = None) -> float | None:
    """Fitted, conservative truncation-certificate boundary ``w_low``.

    O(1) evaluation of a parametric surface fitted to the ENGINE-HONEST
    ceiling (the largest ``w`` whose order-`_DEFAULT_MAX_ORDER` series stays
    within `CERTIFICATION_BAR` of the exact `_schwinger.f_schwinger`
    engine).  It replaces the per-proposal `diffractive_w_low` SCAN in the
    serve hot path (`scripts/fit_diffractive_certificate.py` measures the
    ceiling and bakes the coefficients).

    The surface is

        log w_low = P(log gamma', log s, log(1 - gamma'))
                  + sum_{k=1.._DIFFRACTIVE_FIT_N_HARM} a_k cos(2 k theta)
                  + a_c * log(|y'| / |y_c(theta)|)
                  + (1 / (M + 1)) * log(lam * sqrt_mu),

    with ``P`` a degree-`_DIFFRACTIVE_FIT_DEGREE` polynomial, ``lam =
    1 - kappa``, ``sqrt_mu`` the macro amplitude, ``s = |y'|**2`` the reduced
    squared offset, ``theta`` the eigenframe angle, and ``|y_c(theta)|`` the
    reduced caustic radius in that direction (see `_fit_features`).  The
    exponentiated surface is de-rated by `_DIFFRACTIVE_FIT_DERATE` -- the SOLE
    conservativeness margin (the de-rate alone guarantees the result never
    over-serves on the calibration grid and its held-out off-grid midpoint
    probes; extrapolated off-grid points can over-serve); the
    ``min(., _DIFFRACTIVE_FIT_CEILING)`` clip is a hard
    ORACLE-DOMAIN cap (``W_CEILING_SCHWINGER``, no oracle above 60) and is a
    no-op wherever the fit is calibrated.

    The surface is FENCED around the directional caustic (see `_caustic_rho`
    and the `_DIFFRACTIVE_FIT_FENCE_*` constants): the near-fold shell
    (``rho = |y'| / |y_c(theta)|`` in
    ``[_DIFFRACTIVE_FIT_FENCE_RHO_LO, 1 + _DIFFRACTIVE_FIT_FENCE_DELTA]``)
    is NOT certified and is DECLINED (``None``), so the draw falls through
    to the fold arm / exact engine.  Both the deep interior
    (``rho < RHO_LO``) and the smooth exterior (``rho > 1 + DELTA``) are
    served by the fit.  The wall refusal from `_reduced_shear` is unchanged;
    the fence ``None`` is a DISTINCT near-caustic decline, not a domain
    error.

    Parameters
    ----------
    y : array_like, shape (2,)
        Source position in the (unreduced) lens plane.
    gamma, beta, kappa : float
        External shear magnitude, shear orientation (radians), and
        convergence.
    w_hi : float, optional
        Band cap: the returned ceiling is ``min(fitted, w_hi)``.

    Returns
    -------
    float or None
        The de-rated, clipped fitted ceiling (``w_hi``-capped) in the served
        domain (``rho < RHO_LO`` deep interior or ``rho > 1 + DELTA`` smooth
        exterior); ``0.0`` at ``gamma' == 0`` (series exact); ``None`` in the
        near-fold shell (``RHO_LO <= rho <= 1 + DELTA``) and on a degenerate
        (non-finite) ``sqrt_mu`` or non-finite fitted value.

    Raises
    ------
    DiffractiveDomainError
        At the positive-parity wall (via `_reduced_shear`).
    ValueError
        If ``y`` does not have shape ``(2,)``.

    Notes
    -----
    There is no band floor: the fit is evaluated at a single point and has
    no bracketing scan, so the ``w_lo``-style floor of the retired scan is
    gone.  The nested-split null handling (the whole band below or above
    the certificate boundary) is owned by the CALL SITES via
    ``_band_split_mask(dense_w, w_low)`` plus the ``w_low >=
    dense_w.max()`` whole-band branch, not by this function.

    The fence declines ONLY the near-fold shell: the deep interior
    (``rho < RHO_LO``) is served by the SAME fit as the smooth exterior (the
    fit is calibrated on the interior cells, so it serves them conservatively --
    NOT at the ceiling, whose engine-honest value deep inside the caustic is
    ~4-41, not ``W_CEILING_SCHWINGER``).  The fence discriminator ``rho`` is
    monotone but miscalibrated (it under-estimates the distance-to-fold by
    up to ~1.75x), so the boundaries are conservative guards tuned at the
    driver full bake.
    """
    lam, gamma_prime = _reduced_shear(gamma, kappa)
    if gamma_prime == 0.0:
        return 0.0
    if abs(gamma_prime) > _DIFFRACTIVE_FIT_GAMMA_MAX:
        # Calibration-domain fence: the fit is calibrated only for
        # gamma' <= 0.5, and the order-16 series cannot serve the
        # convergence-radius collapse toward the wall (gamma' -> 1).  Decline
        # so the draw routes to the exact Schwinger engine, the correct serve
        # there.
        return None
    y = np.asarray(y, dtype=float)
    if y.shape != (2,):
        raise ValueError(f'Source position must have shape (2,), got {y.shape}.')

    sqrt_mu = _born_factors(float(y[0]), float(y[1]), float(gamma),
                            float(beta), float(kappa))[0]
    if not math.isfinite(sqrt_mu):
        return None

    root = math.sqrt(lam)
    yp0, yp1 = float(y[0]) / root, float(y[1]) / root
    s = yp0 * yp0 + yp1 * yp1
    if not s > 0.0:
        return None
    z_eig = cmath.exp(-1j * float(beta)) * complex(yp0, yp1)
    theta = math.atan2(z_eig.imag, z_eig.real)

    rho = _caustic_rho(abs(gamma_prime), s, theta)
    if _DIFFRACTIVE_FIT_FENCE_RHO_LO <= rho <= 1.0 + _DIFFRACTIVE_FIT_FENCE_DELTA:
        return None

    features = _fit_features(abs(gamma_prime), s, theta, lam, sqrt_mu)
    n_poly = len(_fit_poly_exponents(_DIFFRACTIVE_FIT_DEGREE))
    n_harm = _DIFFRACTIVE_FIT_N_HARM
    fitted = sum(coeff * feat for coeff, feat in zip(
        _DIFFRACTIVE_FIT_POLY_COEFFS, features[:n_poly]))
    fitted += sum(coeff * feat for coeff, feat in zip(
        _DIFFRACTIVE_FIT_HARMONIC_COEFFS, features[n_poly:n_poly + n_harm]))
    fitted += _DIFFRACTIVE_FIT_CAUSTIC_COEFF * features[n_poly + n_harm]
    fitted += _DIFFRACTIVE_FIT_LIP * math.log(lam * sqrt_mu)

    w_fit = _DIFFRACTIVE_FIT_DERATE * math.exp(fitted)
    if not math.isfinite(w_fit):
        return None
    w_fit = min(w_fit, _DIFFRACTIVE_FIT_CEILING)
    if w_hi is not None:
        w_fit = min(w_fit, w_hi)
    return w_fit
