"""
Point-mass Chang-Refsdal kernel: dd 1F1 and its shared-numerator
s-derivative ladder.

WHAT
----
`point_mass_g_derivatives` evaluates the point-mass amplification
kernel and its derivatives with respect to ``s``,

    G_PM(w, s) = C(w) * 1F1(1 - i*w/2; 1; -i*w*s/2),
    values[k]  = d^k G_PM / ds^k    for k = 0, ..., max_derivative,

together with a MEASURED truncation-tail estimate for each k.
`prefactor_c` evaluates the scalar prefactor C(w) alone.

This module is pure special-function evaluation: it knows nothing
about shear, rotation, images, or branch selection.

WHY: THE SHARED NUMERATOR
-------------------------
Differentiating the k = 0 kernel k times gives

    values[k] = C(w) * base**k * (a)_k / k! * 1F1(a + k; 1 + k; z),

with ``a = 1 - i*w/2``, ``z = -i*w*s/2`` and ``base = -i*w/2``.  The
reference prototype evaluates a FRESH mpmath ``hyp1f1`` at every k,
which is far too slow to sit inside a likelihood.  Kummer's
transformation ``1F1(A; B; Z) = e**Z * 1F1(B - A; B; -Z)`` gives

    1F1(a + k; 1 + k; z) = e**z * 1F1(a'; 1 + k; zz),
    a' = 1 - a = i*w/2,   zz = -z = +i*w*s/2,

and ``a'`` is INDEPENDENT of k.  So a single series numerator serves
the whole ladder.  This is a STRUCTURAL win, not a precision one: z is
purely imaginary, so ``|e**z| = 1`` exactly and both forms carry the
same maximum term.

The shared object is the k = 0 TERM, not the Pochhammer ratio:

    P_n        = (a')_n * zz**n / (n!)**2       max|P_n| ~ e**(w*Y)
    beta_{k,n} = 1 / C(n + k, n) = n!k!/(n+k)!  always in (0, 1]
    1F1(a'; 1 + k; zz) = sum_n P_n * beta_{k,n}

Sharing ``t_n = (a')_n * zz**n / n!`` instead and dividing by
``(1 + k)_n`` is the obvious reading and is numerically UNUSABLE:
``|t_n|`` peaks at ``e**((w*Y)**2/4) ~ 1e391`` at ``w*Y = 60`` and
OVERFLOWS float64, while ``1/(1+k)_n = k!/(k+n)! ~ 1e-449`` at
``k = 84, n = 200`` UNDERFLOWS to zero.  Both factors blow up while
their product is perfectly tame -- and double-double extends the
MANTISSA, not the exponent, so it cannot rescue an overflow.  Both
recurrences below are exact and cheap:

    P_n    = P_{n-1} * (a' + n - 1) * zz / n**2,  P_0    = 1
    beta_n = beta_{n-1} * n / (n + k),            beta_0 = 1

The cost is therefore N dd-complex multiplies for the k-independent
numerator plus (K + 1) * N REAL dd scalings: the shared work does not
grow with ``max_derivative``, and the dd-multiply count is linear in
it.  There is deliberately NO k-recurrence on 1F1 (forward recurrence
in k is unstable; the shared numerator sidesteps it) and NO large-|z|
asymptotic branch (physically unreachable inside the gates below).

WHY: DOUBLE-DOUBLE
------------------
The series is alternating with terms exponentially larger than their
sum: the partial terms reach ``e**(w*Y)`` while the total is O(1), so
the relative error follows the cancellation law
``~ eps * e**(w*Y) / |1F1|``.  Kahan summation does not rescue this --
its error bound also carries ``sum|term_i|``.  Only a smaller eps
does, so both the terms AND their accumulation go through `_dd`.  At
``w*Y = 60`` this leaves ``eps_dd * e**60 ~ 1e-6``; the tighter 1e-10
target holds out to ``w*Y ~ 50``, which is exactly where the caller's
geometric branch takes over.

WHY: THE PREFACTOR IS float64 AND POLAR
---------------------------------------
C(w) is a common multiplicative factor, so its relative error factors
out of every k and dd is not needed.  It is NEVER formed as written,
``exp(pi*w/4) * Gamma(1 - i*w/2)``: that route builds a 5.8e238
magnitude at w = 700 against an underflowing Gamma, and overflows
outright near w ~ 905.  Magnitude and phase are built separately; see
`prefactor_c`.

References
----------
Abramowitz, M. and Stegun, I. A. (1964), Handbook of Mathematical
Functions, chapter 13 (confluent hypergeometric functions); Kummer's
transformation is 13.1.27 and the reflection formula used by
`prefactor_c` follows from 6.1.31.
"""
from __future__ import annotations

import numpy as np
from scipy.special import loggamma

from cogwheel.lensing.chang_refsdal._dd import (dd_complex_add,
                                                dd_complex_mul,
                                                dd_complex_to_complex128,
                                                dd_div, dd_mul, dd_sub)

#: Largest ``w`` this module is certified for.
#:
#: `prefactor_c` is certified to rtol 1e-14 on ``[1e-3, 500]`` against
#: a 60-dps oracle.  The ladder magnitude grows as roughly
#: ``(w/2)**(2k) / k!``, which is ~1e92 at ``w = 40, k = 84`` and
#: reaches the float64 overflow rail near ``w ~ 700``; 500 is
#: certified with ~25 orders of margin.  Double-double extends the
#: MANTISSA, not the exponent, so it cannot lift this ceiling.
W_MAX_CERTIFIED = 500.0

#: Largest ``w * Y`` (with ``Y = sqrt(s) = |y'|``) this module accepts.
#:
#: ``w * Y`` is the series' cancellation exponent: the partial terms
#: reach ``e**(w*Y)`` while the sum is O(1).  Double-double holds ~31.9
#: decimal digits, so it degrades to ~1e-6 at ``w*Y = 60`` and float64
#: (~15.95 digits) would already be there at ``w*Y = 22``.  The ceiling
#: is set at 60 precisely so that the wave branch overlaps the
#: geometric branch's ``w*Y >= 50`` onset -- a gap that float64 cannot
#: bridge.  This is the ceiling at ~1e-6; the 1e-10 target holds out to
#: ``w*Y ~ 50``.
DD_PRODUCT_CEILING = 60.0

#: Row indices into the shared-numerator dd table (see
#: `_shared_numerator`).  A dd complex is four flat float64 limbs; a
#: (4, n_terms) array keeps that layout numba-friendly.
_P_RE_HI = 0
_P_RE_LO = 1
_P_IM_HI = 2
_P_IM_LO = 3

#: ``2 * pi`` as a double-double, for the phase reduction in
#: `_reduced_phase`.  The low limb is twice the well-known residual
#: ``pi - fl(pi) = 1.2246467991473532e-16``, which is checkable in one
#: line: it is the value ``np.sin(np.pi)`` returns, since
#: ``sin(fl(pi)) = sin(pi - delta) ~ delta``.  Doubling a float64 is
#: exact, so both limbs below are exact.
_TWO_PI_HI = 6.283185307179586
_TWO_PI_LO = 2.4492935982947064e-16


class HypergeometricDomainError(ValueError):
    """Arguments fall outside this kernel's certified domain."""


def _validate_w(w: float) -> None:
    """Raise `HypergeometricDomainError` unless ``0 < w <= 500``."""
    if not w > 0.0:
        raise HypergeometricDomainError(
            f'Cannot evaluate the point-mass kernel at w = {w}: the '
            f'dimensionless frequency must be strictly positive.')
    if w > W_MAX_CERTIFIED:
        raise HypergeometricDomainError(
            f'Cannot evaluate the point-mass kernel at w = {w}: it '
            f'exceeds the certified ceiling W_MAX_CERTIFIED = '
            f'{W_MAX_CERTIFIED}, beyond which the derivative ladder '
            f'approaches the float64 overflow rail. Double-double '
            f'arithmetic extends the mantissa, not the exponent, so '
            f'it cannot lift this ceiling; restrict w to '
            f'(0, {W_MAX_CERTIFIED}].')


def _validate_domain(w: float, s: float) -> None:
    """Raise `HypergeometricDomainError` unless ``(w, s)`` is
    certified: ``0 < w <= 500``, ``s >= 0`` and ``w * sqrt(s) <= 60``.
    """
    _validate_w(w)
    if not s >= 0.0:
        raise HypergeometricDomainError(
            f'Cannot evaluate the point-mass kernel at s = {s}: the '
            f'squared source offset must be nonnegative.')
    product = w * np.sqrt(s)
    if product > DD_PRODUCT_CEILING:
        raise HypergeometricDomainError(
            f'Cannot evaluate the point-mass kernel at (w, s) = '
            f'({w}, {s}): w * sqrt(s) = {product} exceeds the '
            f'double-double ceiling DD_PRODUCT_CEILING = '
            f'{DD_PRODUCT_CEILING}. That product is the series\' '
            f'cancellation exponent, and beyond it double-double '
            f'precision no longer resolves the sum. Use the '
            f'geometric-optics branch, which is valid from '
            f'w * sqrt(s) ~ 50.')


def prefactor_c(w: float) -> complex:
    """
    Point-mass prefactor ``C(w)``, built in polar form.

    ``C(w) = exp(pi*w/4 + i*(w/2)*ln(w/2)) * Gamma(1 - i*w/2)`` is
    never formed as written; magnitude and phase are built separately.

    The squared magnitude has an EXACT closed form, from
    ``Gamma(1 + ix) * Gamma(1 - ix) = pi*x / sinh(pi*x)`` at
    ``x = w/2``::

        |C(w)|**2 = e**(pi*w/2) * pi*(w/2) / sinh(pi*w/2)
                  = pi*w / (1 - e**(-pi*w))
                  = -pi*w / expm1(-pi*w),

    which `np.expm1` evaluates without losing the small-w limb.
    Because the magnitude comes from here, loggamma's REAL part -- the
    part that underflows -- is never needed at all, and both overflow
    traps disappear with it.

    The phase is summed directly.  Cancelling the ``(w/2)*ln(w/2)``
    term against loggamma's asymptotics is deliberately NOT done: that
    cancellation is Stirling's series, i.e. asymptotic rather than an
    algebraic identity, so it would need a Bernoulli tail plus a
    small-x convergence switch -- real correctness risk for no gain.
    The direct sum costs ~3 digits at w = 700 (the phase stays O(1)
    while ``(w/2)*ln(w/2)`` reaches 2050) and still lands at 3.4e-13
    against a 60-dps oracle, ~300x inside the 1e-10 target.
    ``exp(1j*theta)`` is 2*pi-periodic, so wrap in the unreduced sum is
    irrelevant.

    Parameters
    ----------
    w : float
        Dimensionless frequency, ``0 < w <= W_MAX_CERTIFIED``.

    Returns
    -------
    complex
        ``C(w)``.  ``|C|**2 -> 1`` as ``w -> 0`` and ``-> pi*w`` as
        ``w -> infinity``.

    Raises
    ------
    HypergeometricDomainError
        If ``w <= 0`` or ``w > W_MAX_CERTIFIED``.
    """
    w = float(w)
    _validate_w(w)
    magnitude_squared = -np.pi * w / np.expm1(-np.pi * w)
    phase = (0.5 * w * np.log(0.5 * w)
             + loggamma(complex(1.0, -0.5 * w)).imag)
    return complex(np.sqrt(magnitude_squared)
                   * complex(np.cos(phase), np.sin(phase)))


def _dd_half_product(w: float, s: float) -> tuple[float, float]:
    """Return ``Z = w*s/2`` as a dd real.  ``dd_mul`` of two exact
    float64s is the error-free product, so ``Z`` is exact to dd
    precision -- which matters because ``w*s`` spans many orders and
    its float64 rounding would otherwise leak into both the series and
    the carrier phase.
    """
    hi, lo = dd_mul(w, 0.0, s, 0.0)
    return dd_mul(hi, lo, 0.5, 0.0)


def _reduced_phase(hi: float, lo: float) -> float:
    """
    Return the dd angle ``hi + lo`` reduced modulo ``2*pi`` into
    roughly ``[-pi, pi]``, as a float64.

    The reduction is done in dd against a dd ``2*pi``, which keeps the
    result accurate to ~``|angle| * eps_dd`` instead of
    ``|angle| * eps``.  This is load-bearing rather than fussy: the
    gates permit ``w*s/2`` up to ``1800/w``, so at ``w = 1e-3, w*Y =
    20`` the angle reaches 2e5 and a plain float64 ``exp(1j*angle)``
    would carry ~4.4e-11 of phase error -- dominating the 1e-10 target
    while the dd series beside it is accurate to 1e-24.  The reduction
    stays exact to dd precision while ``|angle|`` is below ~1e16, far
    outside any physically reachable configuration.
    """
    quotient = np.rint(hi / _TWO_PI_HI)
    turns_hi, turns_lo = dd_mul(quotient, 0.0, _TWO_PI_HI, _TWO_PI_LO)
    residual_hi, residual_lo = dd_sub(hi, lo, turns_hi, turns_lo)
    return residual_hi + residual_lo


def _carrier(w: float, z_hi: float, z_lo: float) -> complex:
    """Return the k-independent common factor ``C(w) * e**z``, where
    ``z = -i*w*s/2`` and ``(z_hi, z_lo)`` is the dd ``w*s/2``.  Since z
    is purely imaginary, ``|e**z| = 1`` exactly and the factor is a
    pure rotation of `prefactor_c`.
    """
    phase = _reduced_phase(-z_hi, -z_lo)
    return prefactor_c(w) * complex(np.cos(phase), np.sin(phase))


def _shared_numerator(w: float, z_hi: float, z_lo: float,
                      n_terms: int) -> np.ndarray:
    """
    Return the dd table of ``P_n = (a')_n * zz**n / (n!)**2`` for
    ``n = 0, ..., n_terms - 1``.

    This is the k-INDEPENDENT half of the ladder: it is computed once
    per ``(w, s)`` and reused for every derivative order.

    Parameters
    ----------
    w : float
        Dimensionless frequency.
    z_hi, z_lo : float
        The dd real ``Z = w*s/2``, from `_dd_half_product`.
    n_terms : int
        Number of series terms.

    Returns
    -------
    np.ndarray
        Shape ``(4, n_terms)``, the ``(re_hi, re_lo, im_hi, im_lo)``
        limbs of each ``P_n``; see the ``_P_*`` row indices.
    """
    table = np.zeros((4, n_terms))
    table[_P_RE_HI, 0] = 1.0  # P_0 = 1.

    # With a' = i*w/2 and zz = i*Z both purely imaginary, the factor
    # (a' + n - 1) * zz expands to -(w/2)*Z + i*(n - 1)*Z.  So its real
    # limb does not depend on n at all, and its imaginary limb is a
    # single real scaling -- one dd-complex multiply per term, with no
    # complex arithmetic wasted on structural zeros.
    const_re_hi, const_re_lo = dd_mul(-0.5 * w, 0.0, z_hi, z_lo)

    for n in range(1, n_terms):
        # n**2 <= 262144 for the caller's n_terms, so it is exact.
        n_squared = float(n * n)
        factor_re_hi, factor_re_lo = dd_div(const_re_hi, const_re_lo,
                                            n_squared, 0.0)
        raised_hi, raised_lo = dd_mul(float(n - 1), 0.0, z_hi, z_lo)
        factor_im_hi, factor_im_lo = dd_div(raised_hi, raised_lo,
                                            n_squared, 0.0)
        re_hi, re_lo, im_hi, im_lo = dd_complex_mul(
            table[_P_RE_HI, n - 1], table[_P_RE_LO, n - 1],
            table[_P_IM_HI, n - 1], table[_P_IM_LO, n - 1],
            factor_re_hi, factor_re_lo, factor_im_hi, factor_im_lo)
        table[_P_RE_HI, n] = re_hi
        table[_P_RE_LO, n] = re_lo
        table[_P_IM_HI, n] = im_hi
        table[_P_IM_LO, n] = im_lo
    return table


def _ladder_sum(table: np.ndarray, k: int) -> tuple[complex, float]:
    """
    Return ``(1F1(a'; 1 + k; zz), relative_tail)`` for one ``k``.

    The reciprocal binomial ``beta_{k,n} = 1/C(n+k, n)`` is folded onto
    the shared ``P_n`` by a REAL dd scaling, and the sum is accumulated
    in dd because the terms are exponentially larger than the total.

    Parameters
    ----------
    table : np.ndarray
        Shape ``(4, n_terms)``, from `_shared_numerator`.
    k : int
        Derivative order; the series denominator is ``1 + k``.

    Returns
    -------
    total : complex
        The truncated series value.
    relative_tail : float
        ``|last retained term| / |total|``, a MEASURED truncation
        estimate rather than a heuristic bound.  It is ``inf`` if
        ``total`` is zero, so a caller can never read a vanished
        denominator as convergence.
    """
    n_terms = table.shape[1]
    beta_hi, beta_lo = 1.0, 0.0
    sum_re_hi = sum_re_lo = sum_im_hi = sum_im_lo = 0.0
    last_term = 0.0
    for n in range(n_terms):
        if n:
            beta_hi, beta_lo = dd_mul(beta_hi, beta_lo, float(n), 0.0)
            beta_hi, beta_lo = dd_div(beta_hi, beta_lo,
                                      float(n + k), 0.0)
        term_re_hi, term_re_lo = dd_mul(table[_P_RE_HI, n],
                                        table[_P_RE_LO, n],
                                        beta_hi, beta_lo)
        term_im_hi, term_im_lo = dd_mul(table[_P_IM_HI, n],
                                        table[_P_IM_LO, n],
                                        beta_hi, beta_lo)
        (sum_re_hi, sum_re_lo,
         sum_im_hi, sum_im_lo) = dd_complex_add(
             sum_re_hi, sum_re_lo, sum_im_hi, sum_im_lo,
             term_re_hi, term_re_lo, term_im_hi, term_im_lo)
        # Magnitude only: the diagnostic needs no dd precision.
        last_term = float(np.hypot(term_re_hi, term_im_hi))
    total = dd_complex_to_complex128(sum_re_hi, sum_re_lo,
                                     sum_im_hi, sum_im_lo)
    magnitude = abs(total)
    if magnitude == 0.0:
        return total, np.inf
    return total, last_term / magnitude


def point_mass_g_derivatives(w: float, s: float, max_derivative: int,
                             n_terms: int
                             ) -> tuple[np.ndarray, np.ndarray]:
    """
    Point-mass kernel ``G_PM(w, s)`` and its ``s``-derivatives.

    Evaluates ``values[k] = d^k G_PM / ds^k`` for
    ``k = 0, ..., max_derivative`` from a single shared numerator; see
    the module docstring for the Kummer reparametrization and the
    double-double rationale.

    Parameters
    ----------
    w : float
        Dimensionless frequency, ``0 < w <= W_MAX_CERTIFIED``.
    s : float
        Squared source offset ``s = |y'|**2 >= 0``, constrained by
        ``w * sqrt(s) <= DD_PRODUCT_CEILING``.
    max_derivative : int
        Highest derivative order, ``>= 0``.  The caller's operator
        raises the radial index by up to 2 per application, so this
        reaches ``2 * max_order`` -- 84 at ``max_order = 42``.
    n_terms : int
        Number of series terms.  The caller owns the adaptive rule;
        check the returned tail rather than trusting it.

    Returns
    -------
    values : np.ndarray
        Shape ``(max_derivative + 1,)``, complex.  ``values[0]`` is
        ``G_PM`` itself.
    relative_tail : np.ndarray
        Shape ``(max_derivative + 1,)``, float.  Per-k MEASURED
        truncation estimate: the magnitude of the last retained term
        relative to the series total (see `_ladder_sum`).  Correctness
        never rests on ``n_terms`` having been chosen tightly, because
        this reports what the truncation actually cost.

    Raises
    ------
    HypergeometricDomainError
        If ``(w, s)`` is outside the certified domain.
    ValueError
        If ``max_derivative`` is negative or ``n_terms`` is not
        positive.

    Notes
    -----
    The relative accuracy follows the cancellation law
    ``~ eps_dd * e**(w*sqrt(s)) / |1F1|``: ~1e-10 out to
    ``w*sqrt(s) ~ 50`` and ~1e-6 at the ceiling of 60.
    """
    w = float(w)
    s = float(s)
    _validate_domain(w, s)
    max_derivative = int(max_derivative)
    n_terms = int(n_terms)
    if max_derivative < 0:
        raise ValueError(
            f'Cannot build a derivative ladder of length '
            f'{max_derivative}: max_derivative must be nonnegative.')
    if n_terms < 1:
        raise ValueError(
            f'Cannot sum a series of {n_terms} terms: n_terms must be '
            f'positive.')

    z_hi, z_lo = _dd_half_product(w, s)
    table = _shared_numerator(w, z_hi, z_lo, n_terms)
    carrier = _carrier(w, z_hi, z_lo)

    values = np.empty(max_derivative + 1, dtype=complex)
    relative_tail = np.empty(max_derivative + 1, dtype=float)

    # Q_k = base**k * (a)_k / k!.  Plain float64: every step is a
    # complex multiply, so magnitudes multiply exactly and there is no
    # cancellation to amplify -- the error grows only as ~k*eps ~ 2e-14
    # at k = 84.  Like C(w), Q_k is a per-k common factor whose
    # relative error factors out of the series it multiplies.
    ladder = complex(1.0, 0.0)
    base = complex(0.0, -0.5 * w)
    a_parameter = complex(1.0, -0.5 * w)
    for k in range(max_derivative + 1):
        if k:
            ladder *= base * (a_parameter + (k - 1)) / k
        total, tail = _ladder_sum(table, k)
        values[k] = carrier * ladder * total
        relative_tail[k] = tail
    return values, relative_tail
