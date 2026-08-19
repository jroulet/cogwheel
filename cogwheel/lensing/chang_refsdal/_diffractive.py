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
* ``diffractive_w_low`` returns the per-draw frequency ``w_low`` up to which
  the operator truncation is provably within the certification bar -- the
  truncation certificate.  A closed-form candidate is used as the lower-bracket
  SEED and then bracket-searched UP to the honest ceiling (the largest ``w``
  whose N/2N omitted-tail ratio clears the certification bar).

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

#: Default operator-series truncation order ``M``.  The tail decays like
#: ``(gamma' s w / 2)**n / n!`` (Notes in `diffractive_w_low`), so eight orders
#: are ample across the certified low-w band; the certificate reads the first
#: omitted term at order ``M + 1``.
_DEFAULT_MAX_ORDER = 8

#: Safety factor tightening the certification bar, matching the high-w c3
#: certificate precedent (`likelihood._saddle_c3_split_point` uses 20.0).  No
#: new safety policy is invented here.
_DIFFRACTIVE_CERT_SAFETY = 20.0

#: Reference frequency at which the geometry-only omitted-term magnitude
#: ``R_{M+1}`` is measured (the c3 certificate likewise fixes ``w_ref = 1.0``).
_CERT_REFERENCE_W = 1.0


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


def _rootfind_w_low(relerr, upper: float, target: float,
                    max_iter: int = 100) -> float | None:
    """Largest ``w`` in ``(0, upper]`` whose true relative error is ``<= target``.

    ``relerr`` is the honest relative truncation error of the order-``M``
    operator series, which is monotone non-decreasing in ``w`` on the low-w band
    (the omitted tail grows like ``(gamma' s w / 2)**(M + 1) / (M + 1)!``).  The
    boundary is bracketed by halving ``upper`` until the error drops under
    ``target`` and then refined by bisection; ``None`` is returned only when no
    positive ``w`` satisfies the target.
    """
    if relerr(upper) <= target:
        return upper
    lo = upper
    for _ in range(max_iter):
        lo *= 0.5
        if not lo > 0.0:
            return None
        if relerr(lo) <= target:
            break
    else:
        return None
    hi = upper
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if relerr(mid) <= target:
            lo = mid
        else:
            hi = mid
    return lo


def _honest_tail_ratio(w: float, u0: float, v0: float, s: float,
                       gamma_prime: float, order: int) -> float:
    """Honest N/2N truncation error of the order-``order`` operator series.

    Returns the magnitude of the full omitted tail ``sum(t_{M+1..2M})`` over
    the truncated total ``|sum(t_0..M)|`` at frequency ``w``, using operator
    terms up to order ``2M``.  ``math.inf`` is returned on a domain/overflow
    error or when the truncated total vanishes or either sum is non-finite --
    the hardened sentinel keeps the up-search doubling loop from ever computing
    ``inf / inf = nan`` (whose comparison against the certification bar is
    False, which would otherwise send the loop to ``max_iter`` on garbage).
    """
    try:
        tail = _operator_terms(w, u0, v0, s, gamma_prime, 2 * order)
    except (HypergeometricDomainError, OverflowError, ValueError):
        return math.inf
    total_trunc = sum(tail[:order + 1])
    tail_sum = sum(tail[order + 1:])
    if not (abs(total_trunc) > 0.0):
        return math.inf
    if not (math.isfinite(total_trunc.real) and math.isfinite(total_trunc.imag)
            and math.isfinite(tail_sum.real) and math.isfinite(tail_sum.imag)):
        return math.inf
    return abs(tail_sum) / abs(total_trunc)


def _rootfind_w_high(relerr, lower: float, target: float,
                     w_lo: float | None, w_hi: float | None,
                     max_iter: int = 100) -> float | None:
    """Largest ``w`` in ``[lower, w_hi]`` whose RUNNING-MAX relerr ``<= target``.

    First-breach-aware up-search.  The honest N/2N tail ratio is NOT monotone
    in ``w`` near the first crossing (breach -> dip -> re-breach; the omitted
    tail's phase oscillates as ``w`` grows toward the divergence), so a raw
    doubling+bisection can stride over the first breach and certify a band
    that contains an interior breach.  Instead this search tracks the RUNNING
    MAXIMUM ``rmax(w) = max_{[lower, w]} relerr``, monotone non-decreasing by
    construction, and returns the largest ``w`` whose running max still
    clears ``target`` (the genuine first breach).  ``w_lo`` floors the band (a
    floor that already fails returns ``None``); ``w_hi`` caps it (a whole band
    under ``target`` returns ``w_hi``).

    The scan is TWO-TIER: coarse (5%) while ``relerr`` is far under the bar
    (where it is monotone and feature-free), then fine (0.2%) once ``relerr``
    nears the bar -- the oscillation's width shrinks as the crossing
    approaches (measured ~1% at the lowest gamma demand), so the fine tier is
    sized to the narrowest observed feature, never coarser.  The running max
    is checked at every scan point, so no breach, however narrow, is stridden
    over.

    A note on ``w_hi`` and the ``whole-band`` case: ``rmax(w_hi) <= target``
    is checked, NOT ``relerr(w_hi) <= target`` -- a pointwise top check is
    unsafe because the tail ratio can breach, dip back under the bar, and
    re-breach inside the band (the non-monotone witness).  Only the running
    maximum decides.
    """
    if w_lo is not None and relerr(w_lo) > target:
        return None
    lo = max(lower, w_lo) if w_lo is not None else lower
    if not math.isfinite(relerr(lo)):
        return None

    # Two-tier scan tracking the running maximum.  rmax is monotone by
    # construction, so the first point whose running max exceeds ``target``
    # brackets the genuine first breach; the fine tier starts once relerr is
    # within a factor ``fine_near_bar`` of the bar (the oscillation is only
    # present near the crossing, well above ``target / fine_near_bar``).
    scan_step_coarse = 1.05
    scan_step_fine = 1.002
    fine_near_bar = 100.0
    w = lo
    rmax = relerr(lo)
    while True:
        if w_hi is not None and w >= w_hi:
            return w_hi
        if rmax > target / fine_near_bar:
            scan_step = scan_step_fine
        else:
            scan_step = scan_step_coarse
        w_next = w * scan_step
        if not math.isfinite(w_next):
            break
        r = relerr(w_next)
        if not math.isfinite(r):
            r = math.inf
        rmax = max(rmax, r)
        if rmax > target:
            break
        w = w_next
    if rmax <= target:
        return w_hi if w_hi is not None else w

    # First breach bracketed in (w, w_next].  The fine scan ensures the
    # bracket is narrower than one oscillation lobe, so ``relerr`` is
    # monotone within it: ``relerr(w) <= target`` (running max at w clears)
    # and ``relerr(w_next) > target`` (the scan point that broke the loop).
    # Bisect the CONTINUOUS relerr to float64 convergence -- grid-independent.
    lo_e, hi_e = w, w_next
    for _ in range(max_iter):
        mid = 0.5 * (lo_e + hi_e)
        if mid <= lo_e or mid >= hi_e:
            break
        if relerr(mid) <= target:
            lo_e = mid
        else:
            hi_e = mid
    return lo_e


def diffractive_w_low(y, gamma: float, beta: float = 0.0, kappa: float = 0.0,
                      max_order: int = _DEFAULT_MAX_ORDER, *,
                      w_lo: float | None = None,
                      w_hi: float | None = None) -> float | None:
    """Truncation certificate ``w_low`` for `diffractive_amplification`.

    Returns the frequency up to which the order-``M`` operator truncation is
    provably within `CERTIFICATION_BAR` (relative) of the exact amplification.
    A closed-form CANDIDATE is proposed as a lower-bracket SEED and then VERIFIED
    against the actual truncated series; when the candidate clears the bar the
    boundary is bracket-searched UP to the honest ceiling (capped at ``w_hi``),
    and when it over-reaches it is root-found down to a certified-smaller
    boundary rather than refused outright.

    Candidate
    ---------
    The closed form solves ``est(w_low) = bar`` for the reference-frequency
    error model::

        w_low = (|gamma'| / 2) * [ lam * sqrt_mu * R_{M+1} / bar ]**(1 / (M + 1))

    where ``R_{M+1}`` is the geometry-only magnitude of the leading omitted
    (order ``M + 1``) term measured from `point_mass_g_derivatives` at the
    reference frequency `_CERT_REFERENCE_W`, ``lam = 1 - kappa``,
    ``sqrt_mu = 1 / sqrt(|(1 - kappa)**2 - gamma**2|)`` is the macro amplitude,
    and ``bar = CERTIFICATION_BAR / _DIFFRACTIVE_CERT_SAFETY``.  The
    ``lam * sqrt_mu`` product is the magnitude of the pre-reconstruction series
    total (``|F| = sqrt_mu`` and the reconstruction supplies ``1 / lam``), so
    normalising the omitted term by it makes the model RELATIVE (INS-4-001); the
    factor is a no-op at ``kappa == 0`` (``lam == 1``).  The safety factor and
    the ``k``-th-root inversion mirror the shipped high-w c3 certificate
    (`likelihood._saddle_c3_split_point`); no new measured constant is
    introduced.

    Verification (INS-1-001 / INS-4-001)
    ------------------------------------
    The reference-frequency model is OPTIMISTIC: the true omitted tail scales
    like ``(gamma' s w / 2)**(M + 1) / (M + 1)!`` -- the kernel derivatives
    ``g(k)`` carry their own ``w`` dependence that ``R_{M+1}`` (held at
    ``_CERT_REFERENCE_W``) does not model -- so the true error GROWS with ``w``
    and the closed form over-certifies beyond reduced shear ~1/3.  Moreover the
    leading omitted term alone under-reads the true error near the parity wall,
    where higher tail orders contribute comparably.  The candidate is therefore
    checked with the HONEST relative error -- the magnitude of the FULL omitted
    tail ``sum(t_{M+1..2M})`` over the truncated total ``|sum(t_0..t_M)|``,
    evaluated at the worst (largest-``w``) point of the served band:

    * within ``CERTIFICATION_BAR`` at the candidate -> bracket-search UP to the
      honest ceiling (the largest ``w <= w_hi`` whose honest error clears
      `CERTIFICATION_BAR`);
    * beyond ``_DIFFRACTIVE_CERT_SAFETY * CERTIFICATION_BAR`` -> self-refuse
      (the deep-optimistic regime; consumers fall through to the exact engine);
    * in between -> ROOT-FIND the largest ``w <= candidate`` whose honest error
      is within ``bar_inner`` (a certified-smaller band, per the INS-4-001
      ruling: refusal at a recoverable config is wrong).

    w_lo, w_hi (keyword-only)
    -------------------------
    Optional band bounds that cap the up-search.  ``w_lo`` floors the served
    band (``None`` -> unbounded below, seeded at the candidate); a floor whose
    honest error already exceeds `CERTIFICATION_BAR` self-refuses.  ``w_hi``
    ceilings the served band (``None`` -> unbounded above); a whole band under
    `CERTIFICATION_BAR` returns ``w_hi``.  Both default to ``None``, preserving
    the pre-band behaviour up to the candidate-clears bracket-up.

    Returns
    -------
    float or None
        ``w_low``; ``0.0`` when there is no shear (``gamma' == 0``, series
        exact); ``w_hi`` when the whole band clears the bar; ``None``
        (self-refuse) for degenerate geometry -- non-finite ``sqrt_mu`` or
        omitted term, a vanishing omitted magnitude, an out-of-domain reference
        kernel, a band floor ``w_lo`` whose honest truncation error already
        exceeds the bar, or a candidate whose honest truncation error lies in
        the deep-optimistic (unrecoverable) regime.

    Raises
    ------
    DiffractiveDomainError
        At the positive-parity wall (via `_reduced_shear`).
    ValueError
        If ``y`` does not have shape ``(2,)``.
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

    order = int(max_order)
    root = math.sqrt(lam)
    yp0, yp1 = float(y[0]) / root, float(y[1]) / root
    s = yp0 * yp0 + yp1 * yp1
    z_eig = cmath.exp(-1j * float(beta)) * complex(yp0, yp1)

    try:
        terms = _operator_terms(_CERT_REFERENCE_W, z_eig.real, z_eig.imag, s,
                                gamma_prime, order + 1)
    except HypergeometricDomainError:
        return None

    alpha_ref = abs(gamma_prime) / (2.0 * _CERT_REFERENCE_W)
    if not alpha_ref > 0.0:
        return None
    r_next = abs(terms[order + 1]) / alpha_ref ** (order + 1)
    if not (math.isfinite(r_next) and r_next > 0.0):
        return None

    from cogwheel.lensing.ppgo_map import CERTIFICATION_BAR
    bar_inner = CERTIFICATION_BAR / _DIFFRACTIVE_CERT_SAFETY
    candidate = (abs(gamma_prime) / 2.0) * (
        lam * sqrt_mu * r_next / bar_inner) ** (1.0 / (order + 1))

    # Honest verification (INS-1-001 / INS-4-001).  The closed-form candidate
    # sits under an OPTIMISTIC reference-frequency model; verify it against the
    # ACTUAL truncated series before serving.  ``relerr`` is the magnitude of
    # the full omitted tail (orders ``M + 1 .. 2 M``) relative to the truncated
    # total, evaluated at the band top -- the worst point of ``[w_lo, w]``.
    def relerr(w: float) -> float:
        return _honest_tail_ratio(w, z_eig.real, z_eig.imag, s, gamma_prime,
                                  order)

    honest_error = relerr(candidate)
    if not math.isfinite(honest_error):
        return None
    if honest_error <= CERTIFICATION_BAR:
        return _rootfind_w_high(relerr, candidate, CERTIFICATION_BAR, w_lo,
                                w_hi)
    if honest_error > _DIFFRACTIVE_CERT_SAFETY * CERTIFICATION_BAR:
        return None
    return _rootfind_w_low(relerr, candidate, bar_inner)
