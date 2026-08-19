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
  ``scripts/fit_diffractive_certificate.py``.

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
#: The fit constants below were baked by
#: `scripts/fit_diffractive_certificate.py` at FULL scale (252-point
#: engine grid, 236/236 measurable) and pasted verbatim: de-rate 0.7452,
#: margin worst ratio 1.0000 / median 0.7506 / p90 0.8852, 236/236
#: conservative (fit <= true) and 236/236 tight (fit >= 0.5 * true).
#: Provenance SHA 7eeedee (440.3 s).  Re-bake with
#: ``python scripts/fit_diffractive_certificate.py --scale full`` and
#: paste the emission block verbatim.

#: Degree of the log-log polynomial ``P`` in the fitted certificate
#: `w_low_fit` (see `_fit_poly_features`).  The three log-features are
#: ``(log gamma', log s, log(1 - gamma'))``; degree 2 is a low-order fit that
#: still captures the curvature of the engine-honest ceiling.
_DIFFRACTIVE_FIT_DEGREE = 2

#: Polynomial coefficients of `w_low_fit`, one per monomial of the
#: `_DIFFRACTIVE_FIT_DEGREE` log-feature basis (same enumeration as
#: `_fit_poly_exponents`).  Baked by `scripts/fit_diffractive_certificate.py`.
_DIFFRACTIVE_FIT_POLY_COEFFS = (
    9.605021772432762, -11.422747999010443, -0.6228709189768966,
    0.17920524341265792, -13.418797194240739, 0.14513967710558642,
    -0.06572995920967026, -24.068104751521076, -0.14143727808347514,
    -0.3425759016661124,
)

#: Number of 4-fold harmonics in `w_low_fit` (``cos(4 k theta)`` for
#: ``k = 1 .. _DIFFRACTIVE_FIT_N_HARM``).  Held at `_DEFAULT_MAX_ORDER`.
_DIFFRACTIVE_FIT_N_HARM = _DEFAULT_MAX_ORDER

#: 4-fold harmonic coefficients ``a_k`` of `w_low_fit`, for
#: ``k = 1 .. _DIFFRACTIVE_FIT_N_HARM``.  Baked by
#: `scripts/fit_diffractive_certificate.py`.
_DIFFRACTIVE_FIT_HARMONIC_COEFFS = (
    -2.194815891133371, -0.6922675418142367, -0.7393624023476256,
    0.10328680455790058, 1.0975367672404386, -0.31667311164922346,
    0.8107119967838469, 1.0622755228096508, 0.9829421223842966,
    1.1823402102995237, -0.013882208004386968, 0.6444597210945835,
    -0.3551444588230961, -1.042554026273715, 0.27864372489563183,
    0.19118490622650733,
)

#: De-rating factor applied to the exponentiated fit so the fitted ceiling is
#: CONSERVATIVE on the calibration grid (never above the engine-honest
#: ceiling).  Baked by `scripts/fit_diffractive_certificate.py` as the
#: reciprocal of the worst un-de-rated over-prediction (clamped to <= 0.85;
#: 0.745168 is the un-clamped reciprocal, below the ceiling).
_DIFFRACTIVE_FIT_DERATE = 0.745168

#: Coefficient of the ``log(lam * sqrt_mu)`` feature, held at ``1 / (M + 1)``
#: by construction (the amplitude-space normalisation the fitted surface
#: inherits from the certificate's relative-error currency).
_DIFFRACTIVE_FIT_LIP = 1.0 / (_DEFAULT_MAX_ORDER + 1)

#: Hard ceiling of the fitted certificate: the double-double Schwinger engine
#: domain (`W_CEILING_SCHWINGER`), imported -- never re-typed.
_DIFFRACTIVE_FIT_CEILING = W_CEILING_SCHWINGER


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


def _fit_features(gamma_prime: float, s: float, theta: float, lam: float,
                  sqrt_mu: float) -> tuple[float, ...]:
    """Feature vector of the fitted certificate (poly terms + 4-fold harmonics).

    The last `_DIFFRACTIVE_FIT_N_HARM` entries are ``cos(4 k theta)`` for
    ``k = 1 .. _DIFFRACTIVE_FIT_N_HARM``.  The ``log(lam * sqrt_mu)`` feature
    is NOT part of the fitted vector -- its coefficient is held at
    `_DIFFRACTIVE_FIT_LIP` by construction (added by `w_low_fit`, not
    fitted).
    """
    poly = _fit_poly_features(
        math.log(abs(gamma_prime)), math.log(s),
        math.log(1.0 - abs(gamma_prime)),
        _fit_poly_exponents(_DIFFRACTIVE_FIT_DEGREE))
    harmonics = tuple(math.cos(4.0 * k * theta)
                      for k in range(1, _DIFFRACTIVE_FIT_N_HARM + 1))
    return poly + harmonics


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
                  + sum_{k=1.._DIFFRACTIVE_FIT_N_HARM} a_k cos(4 k theta)
                  + (1 / (M + 1)) * log(lam * sqrt_mu),

    with ``P`` a degree-`_DIFFRACTIVE_FIT_DEGREE` polynomial, ``lam =
    1 - kappa``, ``sqrt_mu`` the macro amplitude, ``s = |y'|**2`` the reduced
    squared offset, and ``theta`` the eigenframe angle.  The exponentiated
    surface is de-rated by `_DIFFRACTIVE_FIT_DERATE` and clipped to
    `_DIFFRACTIVE_FIT_CEILING` so the result never over-serves.  There is no
    deep-optimistic ``None`` branch: the fitted surface IS the certificate,
    and `_reduced_shear`'s wall refusal is the only domain refusal.

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
        The fitted ceiling; ``0.0`` at ``gamma' == 0`` (series exact); ``None``
        on a degenerate (non-finite) ``sqrt_mu`` or a non-finite fitted
        value; ``w_hi`` when the fitted value reaches ``w_hi``.

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
    """
    lam, gamma_prime = _reduced_shear(gamma, kappa)
    if gamma_prime == 0.0:
        return 0.0
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

    features = _fit_features(abs(gamma_prime), s, theta, lam, sqrt_mu)
    n_poly = len(_fit_poly_exponents(_DIFFRACTIVE_FIT_DEGREE))
    fitted = sum(coeff * feat for coeff, feat in zip(
        _DIFFRACTIVE_FIT_POLY_COEFFS, features[:n_poly]))
    fitted += sum(coeff * feat for coeff, feat in zip(
        _DIFFRACTIVE_FIT_HARMONIC_COEFFS, features[n_poly:]))
    fitted += _DIFFRACTIVE_FIT_LIP * math.log(lam * sqrt_mu)

    w_fit = _DIFFRACTIVE_FIT_DERATE * math.exp(fitted)
    if not math.isfinite(w_fit):
        return None
    w_fit = min(w_fit, _DIFFRACTIVE_FIT_CEILING)
    if w_hi is not None:
        w_fit = min(w_fit, w_hi)
    return w_fit
