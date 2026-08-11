"""
Exact 1D Schwinger-parameter wave branch for the pure-shear saddle.

WHAT
----
`f_schwinger(w, y_eig, gamma_prime)` evaluates the Chang-Refsdal
amplification ``F_{0, gamma'}(w, y_eig)`` for a MACRO SADDLE host
(pure external shear ``gamma' > 1``, convergence removed to the
mass-sheet gauge), in the SHEAR EIGENFRAME.  The two shear eigenvalues
are ``a = 1 - gamma' < 0`` (soft axis ``e1``) and ``b = 1 + gamma' > 0``
(hard axis ``e2``).  Inputs are pure-eigenframe: a scalar ``w``, an
eigenframe source position ``y_eig`` of shape ``(2,)``, and the reduced
shear ``gamma_prime``.  The return value is the complex pure-shear
amplification.

WHY A SEPARATE BRANCH
---------------------
The positive-parity operator power series in `operator.py` DIVERGES for
``gamma' > 1`` (the parity boundary ``|gamma'| = lam`` is a branch point
of that series), so it cannot reach the saddle domain at any truncation.
The exact one-dimensional Schwinger / heat-kernel representation used
here is valid at BOTH parities.  Inserting the identity
``r^{-iw} = (1/Gamma(iw/2)) Int_0^inf dt t^{iw/2-1} e^{-t r^2}`` into the
diffraction integral turns the two lens-plane ``x`` integrals into exact
Gaussians for any signature, leaving a single ``t`` integral::

    F(w, y) = (w / (2 pi i)) e^{i w |y|^2 / 2} (pi / Gamma(iw/2))
              Int_0^inf dt t^{iw/2 - 1} h(t),
    h(t)    = (t - i w a / 2)^{-1/2} (t - i w b / 2)^{-1/2}
              exp[ -w^2 y1^2 / (4 (t - i w a / 2))
                   -w^2 y2^2 / (4 (t - i w b / 2)) ],

with principal square roots of each factor separately (``Re(t - iw./2)``
stays positive on the real-``t`` contour for the saddle, since ``a < 0``
puts its branch point in the lower half plane and ``b > 0`` puts its in
the upper half plane).

THE ``t -> 0`` ENDPOINT
-----------------------
``t^{iw/2 - 1}`` is not integrable in modulus at ``t = 0``; one
integration by parts (continuation in ``s = iw/2``) regularises it::

    Int_0^T t^{s-1} h dt := T^s h(T) / s - (1/s) Int_0^T t^s h'(t) dt,

the ``t = 0`` boundary term vanishing by the analytic continuation that
defines the identity at ``Re s = 0``.  ``h'(t)`` is the ANALYTIC closed
form ``h(t) * G(t)`` with

    G(t) = A1/(t-ca)^2 + A2/(t-cb)^2 - 1/2/(t-ca) - 1/2/(t-cb),

never a finite difference.  The remaining ``[0, T]`` integral and the
tail ``Int_T^inf t^{s-1} h dt`` are both absolutely convergent after the
substitution ``t = e^u`` (the tail integrand decays as ``t^{-2}``); the
split point ``T = w (|a| + |b| + 2) / 2`` sits past both branch points.

THE CANCELLATION LAW (F001-S) AND THE CEILING
---------------------------------------------
The prefactor ``1/Gamma(iw/2)`` grows as ``e^{+pi w/4}`` while the
oscillatory ``t``-integral supplies the compensating ``e^{-pi w/4}``:
the raw ``t``-integral is exponentially smaller than the O(1) GL-node
contributions that build it, so their running sum loses
``L_S = pi w / (4 ln 10)`` decimal digits to cancellation.  This is a
SINGLE, ``y``-INDEPENDENT channel -- there is no ``L = w |y'|`` 1F1
ladder and no ``L_op = w gamma'/2`` operator channel on the saddle
branch.  Double-double carries the GL-node accumulation, giving
~31.9 - L_S digits; requiring 10 surviving digits pins the certified
frequency ceiling at ``w ~ 64``, rounded down to
``W_CEILING_SCHWINGER = 60`` with margin.  ``w > 60`` is an
UNCONDITIONAL hard refuse raised inside the evaluator; it never returns
there.

CERTIFICATION
-------------
Every quadrature panel is evaluated with an ``N``-point and a
``2N``-point Gauss-Legendre rule on the SAME panels.  The two RAW
``t``-integral estimates ``I_N`` and ``I_2N`` (endpoint term plus the
IBP-corrected integral plus the tail, all in double-double, BEFORE any
prefactor) are compared: if
``|I_N - I_2N| / |I_2N| > 3e-10`` the evaluator REFUSES with
`SchwingerCertificationError`.  The difference is measured on the raw
``t``-integral, never on the reconstructed ``F`` -- reconstructing
first would fold the ``e^{pi w/4}`` cancellation that is being certified
back into the estimate.  Domain truncation error is identical in the
``N`` and ``2N`` rules (same ``u`` range) and so cancels in that ratio;
it is instead bounded analytically by the generous ``u`` margins.

Naive float64 quadrature above ``w ~ 20`` silently fabricates a
plausible-but-wrong amplification (the ``e^{pi w/4}`` law is
unforgiving); double-double plus the paired-rule refusal is what makes
this branch certified-or-refuse rather than silently wrong.
"""
from __future__ import annotations

import cmath
import functools
import math

import numba
import numpy as np
from scipy.special import loggamma

from cogwheel.lensing.chang_refsdal._dd import (dd_add, dd_complex_add,
                                                dd_complex_div,
                                                dd_complex_mul,
                                                dd_complex_sub, dd_div,
                                                dd_mul, dd_sub)

__all__ = ['f_schwinger', 'SchwingerCertificationError',
           'W_CEILING_SCHWINGER', 'W_CEILING_SCHWINGER_QD']

#: ``2 * pi`` as a double-double, for the mod-2*pi phase reduction
#: (identical limbs to `_hyp1f1._reduced_phase`; the low limb is twice
#: the residual ``pi - fl(pi)``).
_TWO_PI_HI = 6.283185307179586
_TWO_PI_LO = 2.4492935982947064e-16

#: Hard, unconditional frequency ceiling for the saddle wave branch.
#: Above it the ``e^{pi w/4}`` cancellation (F001-S) outruns the
#: double-double mantissa and the evaluator refuses rather than return.
W_CEILING_SCHWINGER = 60.0

#: Performance ceiling for the mpmath (arbitrary-precision) quadrature path.
#: ``dps = 30 + ceil(w)`` never runs out of PRECISION — the ceiling is
#: RUNTIME: cost scales as ``O(w * dps^2)`` and exceeds the per-node
#: training budget above ``w ~ 150``.  ``w > 150`` is an unconditional
#: hard refuse from `f_schwinger`.
W_CEILING_SCHWINGER_QD = 150.0

#: Paired-rule relative-difference threshold on the RAW ``t``-integral.
_CERTIFICATION_TOL = 3e-10

#: Gauss-Legendre order per composite panel for the mpmath QD path
#: (`60 < w <= 150`).  Higher than the DD path's `_PANEL_ORDER` (24):
#: order-24 under-resolves the upper band (measured N/2N disagreement
#: ~3e-4 > `_CERTIFICATION_TOL` at w=100), which would REGRESS serving
#: coverage in the cusp-exterior windows where the exact engine is the
#: last rung (the surrogate declines those by design).  Order-32 certifies
#: across the band (measured) and is still cheap on this fallback-only
#: path.
_MP_PANEL_ORDER = 32

#: Fixed Gauss-Legendre order per composite panel.  The (node, weight)
#: pairs are Newton-refined to double-double ONCE (`_dd_gl_rule`) and
#: reused across every panel: the raw ``t``-integral is exponentially
#: smaller than the O(1) node contributions that build it (F001-S), so
#: float64 leggauss pairs (rel error ``1e-16``) would be amplified by
#: ``e^{+pi w/4}`` into the reconstructed ``F`` -- exactly the failure
#: mode that made a float64 quadrature silently fabricate garbage above
#: ``w ~ 20`` (the saddle branch's form of the F005 lesson).
_PANEL_ORDER = 24

#: Oscillations ``e^{i (w/2) u}`` per composite panel for the ``N``
#: (coarse) rule; the ``2N`` rule halves it (doubles the panel count).
#: An order-``_PANEL_ORDER`` GL rule resolves a few oscillations per
#: panel spectrally, so a handful of nodes per wavelength drives the
#: paired-rule truncation far below `_CERTIFICATION_TOL` up to the
#: ceiling.
_WAVELENGTHS_PER_PANEL = 2.0

#: Minimum composite-panel count per side for the coarse (``N``) rule.
#: At small ``w`` the ``t^s`` oscillation is slow, so the wavelength rule
#: alone gives too few panels to resolve the branch-point / Gaussian
#: amplitude structure of ``h`` near ``t ~ w gamma'`` (which does not
#: shrink with ``w``); this floor keeps the low-``w`` quadrature resolved.
_MIN_PANELS = 16

#: The SINGLE, ``y``-INDEPENDENT cancellation channel of the saddle wave
#: branch (F001-S): the prefactor ``1/Gamma(iw/2)`` grows as
#: ``e^{+pi w/4}`` while the raw ``t``-integral supplies the compensating
#: ``e^{-pi w/4}``, so the cancellation depth in ``u = ln t`` is
#: ``L_S = _CANCEL_SCALE * w = pi w / 4``.  There is NO ``L = w |y'|``
#: 1F1 ladder and NO ``L_op = w gamma'/2`` operator channel here, so no
#: ``y``-dependent term enters the quadrature-range (certification) scale.
_CANCEL_SCALE = 0.25 * math.pi

#: Additive slack (in ``u``) on each quadrature end, on top of the
#: ``_CANCEL_SCALE * w = pi w / 4`` cancellation depth.  ``e^{-34}``
#: absolute truncation sits far below ``3e-10 * |I|`` across the whole
#: certified band.
_U_MARGIN_CONST = 34.0

# Lazy-loaded mpmath module reference.  The import fires only when the
# QD path (w > W_CEILING_SCHWINGER) is actually invoked, so the core
# lensing package incurs no mpmath import overhead.
_mpmath = None


class SchwingerCertificationError(RuntimeError):
    """
    The saddle wave branch could not be certified at this point.

    Raised for the refusals the evaluator owns: the unconditional
    ``w > W_CEILING_SCHWINGER_QD`` (= 150) hard ceiling, and a
    paired-rule quadrature relative difference exceeding
    `_CERTIFICATION_TOL` -- on the raw ``t``-integral (DD path,
    ``w <= 60``) or on the reconstructed ``F`` (mpmath path,
    ``60 < w <= 150``).
    """


@numba.njit(cache=True, fastmath=False)
def _reduce_phase(hi: float, lo: float) -> float:
    """
    Return the dd angle ``hi + lo`` reduced modulo ``2*pi`` into roughly
    ``[-pi, pi]``, as a float64 (the `_hyp1f1._reduced_phase` idiom).

    Reducing in double-double against a double-double ``2*pi`` keeps the
    result accurate to ``~|angle| * eps_dd`` rather than ``~|angle| *
    eps``, which matters because the ``t^s`` phase ``(w/2) ln t`` reaches
    O(300) (~46 wraps at ``w = 60``) across the quadrature range.
    """
    quotient = np.rint(hi / _TWO_PI_HI)
    turns_hi, turns_lo = dd_mul(quotient, 0.0, _TWO_PI_HI, _TWO_PI_LO)
    residual_hi, residual_lo = dd_sub(hi, lo, turns_hi, turns_lo)
    return residual_hi + residual_lo


# ----------------------------------------------------------------------
# Double-double transcendental helpers (local to the saddle branch).
#
# `_dd.py` deliberately ships no dd sqrt / exp / trig -- its consumers
# (the 1F1 ladder, the operator series) never need them.  The Schwinger
# integrand does: each Gauss-Legendre node value must carry ~dd relative
# accuracy, because the raw t-integral is exponentially SMALLER than the
# O(1/w) node magnitudes that build it (``|I| ~ e^{-pi w/4}``, F001-S),
# so a float64 node error ``~1e-16`` would be amplified by ``e^{+pi w/4}``
# (~1e20 at w = 60) into the reconstructed F.  A float64-node quadrature
# therefore silently fabricates garbage above ``w ~ 20`` (increasing the
# node density makes it WORSE, the signature of node round-off rather
# than under-resolution) -- the saddle branch's form of the F005 lesson.
# These helpers are scoped HERE rather than widening the shared
# substrate; all are njit and expose ``.py_func`` (F010).
# ----------------------------------------------------------------------

#: ``ln 2`` as a double-double, for `_dd_exp` range reduction.
_LN2_HI = 0.6931471805599453
_LN2_LO = 2.3190468138462996e-17

#: ``pi / 2`` as a double-double, for `_dd_cos_sin_of_phase` reduction.
_HALFPI_HI = 1.5707963267948966
_HALFPI_LO = 6.123233995736766e-17

#: Taylor lengths on the reduced arguments ``|r| <= ln2/2`` (exp) and
#: ``|r| <= pi/4`` (sin/cos); both drive the tail below ``2**-105``.
_EXP_TERMS = 24
_TRIG_TERMS = 24


@numba.njit(cache=True, fastmath=False)
def _dd_sqrt(x_hi: float, x_lo: float) -> tuple[float, float]:
    """
    Return the dd square root of a non-negative dd real ``x``.

    One Newton-Heron step ``y <- (y + x/y)/2`` from the float64 seed
    ``sqrt(x_hi)`` doubles the ~16 correct digits to the full dd ~32.
    """
    s = math.sqrt(x_hi)
    if s == 0.0:
        return 0.0, 0.0
    q_hi, q_lo = dd_div(x_hi, x_lo, s, 0.0)
    sum_hi, sum_lo = dd_add(s, 0.0, q_hi, q_lo)
    return 0.5 * sum_hi, 0.5 * sum_lo


@numba.njit(cache=True, fastmath=False)
def _dd_exp(x_hi: float, x_lo: float) -> tuple[float, float]:
    """
    Return the dd exponential ``exp(x_hi + x_lo)``.

    Range-reduce ``x = k ln2 + r`` with ``|r| <= ln2/2`` (``k`` an
    integer), sum the Taylor series of ``exp(r)`` in double-double, then
    scale by ``2**k`` with `math.ldexp` (an exact power-of-two scaling).
    """
    k = math.floor(x_hi / _LN2_HI + 0.5)
    kln2_hi, kln2_lo = dd_mul(k, 0.0, _LN2_HI, _LN2_LO)
    r_hi, r_lo = dd_sub(x_hi, x_lo, kln2_hi, kln2_lo)

    term_hi, term_lo = 1.0, 0.0
    sum_hi, sum_lo = 1.0, 0.0
    for n in range(1, _EXP_TERMS + 1):
        term_hi, term_lo = dd_mul(term_hi, term_lo, r_hi, r_lo)
        term_hi, term_lo = dd_div(term_hi, term_lo, float(n), 0.0)
        sum_hi, sum_lo = dd_add(sum_hi, sum_lo, term_hi, term_lo)

    k_int = int(k)
    return math.ldexp(sum_hi, k_int), math.ldexp(sum_lo, k_int)


@numba.njit(cache=True, fastmath=False)
def _dd_log(x_hi: float, x_lo: float) -> tuple[float, float]:
    """
    Return the dd natural logarithm ``log(x_hi + x_lo)`` for ``x > 0``.

    Two Newton steps ``y <- y - 1 + x e^{-y}`` (for ``e^y = x``) from the
    float64 seed ``log(x_hi)`` reach full dd accuracy.  The endpoint term
    ``T^s h(T)/s`` carries the phase ``(w/2) ln T``, which must be
    dd-accurate for the same F001-S reason the nodes are: it nearly
    cancels against the two integrals down to the ``e^{-pi w/4}`` result.
    """
    y_hi = math.log(x_hi)
    y_lo = 0.0
    for _ in range(2):
        em_hi, em_lo = _dd_exp(-y_hi, -y_lo)
        r_hi, r_lo = dd_mul(x_hi, x_lo, em_hi, em_lo)
        d_hi, d_lo = dd_sub(r_hi, r_lo, 1.0, 0.0)
        y_hi, y_lo = dd_add(y_hi, y_lo, d_hi, d_lo)
    return y_hi, y_lo


@numba.njit(cache=True, fastmath=False)
def _dd_cos_sin_of_phase(
        theta_hi: float, theta_lo: float
) -> tuple[float, float, float, float]:
    """
    Return ``(cos_hi, cos_lo, sin_hi, sin_lo)`` of a dd angle ``theta``.

    ``theta`` may be large -- the oscillatory ``t^s`` phase reaches
    O(2500) (~400 wraps) across the quadrature range -- so it is reduced
    modulo ``pi/2`` in DOUBLE-DOUBLE (against a dd ``pi/2``) before the
    ``|r| <= pi/4`` Taylor series, and the quadrant is applied exactly.
    Collapsing ``theta`` to float64 before reduction would reintroduce
    the ``1e-16`` node error this branch exists to avoid.
    """
    q = math.floor(theta_hi / _HALFPI_HI + 0.5)
    qh_hi, qh_lo = dd_mul(q, 0.0, _HALFPI_HI, _HALFPI_LO)
    r_hi, r_lo = dd_sub(theta_hi, theta_lo, qh_hi, qh_lo)
    r2_hi, r2_lo = dd_mul(r_hi, r_lo, r_hi, r_lo)

    sin_hi, sin_lo = r_hi, r_lo
    sterm_hi, sterm_lo = r_hi, r_lo
    cos_hi, cos_lo = 1.0, 0.0
    cterm_hi, cterm_lo = 1.0, 0.0
    for k in range(1, _TRIG_TERMS + 1):
        denom_s = -float((2 * k) * (2 * k + 1))
        sterm_hi, sterm_lo = dd_mul(sterm_hi, sterm_lo, r2_hi, r2_lo)
        sterm_hi, sterm_lo = dd_div(sterm_hi, sterm_lo, denom_s, 0.0)
        sin_hi, sin_lo = dd_add(sin_hi, sin_lo, sterm_hi, sterm_lo)

        denom_c = -float((2 * k - 1) * (2 * k))
        cterm_hi, cterm_lo = dd_mul(cterm_hi, cterm_lo, r2_hi, r2_lo)
        cterm_hi, cterm_lo = dd_div(cterm_hi, cterm_lo, denom_c, 0.0)
        cos_hi, cos_lo = dd_add(cos_hi, cos_lo, cterm_hi, cterm_lo)

    qi = int(q) % 4
    if qi < 0:
        qi += 4
    if qi == 0:
        return cos_hi, cos_lo, sin_hi, sin_lo
    if qi == 1:
        return -sin_hi, -sin_lo, cos_hi, cos_lo
    if qi == 2:
        return -cos_hi, -cos_lo, -sin_hi, -sin_lo
    return sin_hi, sin_lo, -cos_hi, -cos_lo


@numba.njit(cache=True, fastmath=False)
def _ddc_sqrt(re_hi: float, re_lo: float, im_hi: float, im_lo: float
              ) -> tuple[float, float, float, float]:
    """
    Return the principal dd-complex square root ``sqrt(z)``.

    One complex Newton step ``s <- (s + z/s)/2`` from the float64 seed
    ``cmath.sqrt(z_hi)`` lifts the seed to full dd accuracy.  On the
    real-``t`` saddle contour the arguments ``da``, ``db`` stay off the
    branch cut, so the principal branch is unambiguous.
    """
    s0 = cmath.sqrt(complex(re_hi, im_hi))
    s0_re = s0.real
    s0_im = s0.imag
    if s0_re == 0.0 and s0_im == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    q_re_hi, q_re_lo, q_im_hi, q_im_lo = dd_complex_div(
        re_hi, re_lo, im_hi, im_lo, s0_re, 0.0, s0_im, 0.0)
    sum_re_hi, sum_re_lo, sum_im_hi, sum_im_lo = dd_complex_add(
        s0_re, 0.0, s0_im, 0.0, q_re_hi, q_re_lo, q_im_hi, q_im_lo)
    return (0.5 * sum_re_hi, 0.5 * sum_re_lo,
            0.5 * sum_im_hi, 0.5 * sum_im_lo)


@numba.njit(cache=True, fastmath=False)
def _ddc_exp(re_hi: float, re_lo: float, im_hi: float, im_lo: float
             ) -> tuple[float, float, float, float]:
    """Return the dd-complex exponential ``exp(re + i im)``."""
    er_hi, er_lo = _dd_exp(re_hi, re_lo)
    cos_hi, cos_lo, sin_hi, sin_lo = _dd_cos_sin_of_phase(im_hi, im_lo)
    rr_hi, rr_lo = dd_mul(er_hi, er_lo, cos_hi, cos_lo)
    ri_hi, ri_lo = dd_mul(er_hi, er_lo, sin_hi, sin_lo)
    return rr_hi, rr_lo, ri_hi, ri_lo


@numba.njit(cache=True, fastmath=False)
def _h_dd(t_hi: float, t_lo: float, da_im: float, db_im: float,
          amp1: float, amp2: float) -> tuple[float, float, float, float]:
    """
    Return the Schwinger kernel ``h(t)`` as a dd-complex.

    ``da = t - i w a / 2 = (t, da_im)`` and ``db = (t, db_im)`` are the
    branch-point-shifted arguments (``da_im = -w a / 2 > 0`` since
    ``a < 0``; ``db_im = -w b / 2 < 0``).  Every factor is carried in
    double-double: the two principal square roots, the reciprocal
    ``p = 1/(sqrt(da) sqrt(db))``, and the Gaussian
    ``exp(-amp1/da - amp2/db)`` with ``amp. = w^2 y.^2 / 4``.
    """
    sda_re_hi, sda_re_lo, sda_im_hi, sda_im_lo = _ddc_sqrt(
        t_hi, t_lo, da_im, 0.0)
    sdb_re_hi, sdb_re_lo, sdb_im_hi, sdb_im_lo = _ddc_sqrt(
        t_hi, t_lo, db_im, 0.0)
    den_re_hi, den_re_lo, den_im_hi, den_im_lo = dd_complex_mul(
        sda_re_hi, sda_re_lo, sda_im_hi, sda_im_lo,
        sdb_re_hi, sdb_re_lo, sdb_im_hi, sdb_im_lo)
    p_re_hi, p_re_lo, p_im_hi, p_im_lo = dd_complex_div(
        1.0, 0.0, 0.0, 0.0,
        den_re_hi, den_re_lo, den_im_hi, den_im_lo)

    invda_re_hi, invda_re_lo, invda_im_hi, invda_im_lo = dd_complex_div(
        1.0, 0.0, 0.0, 0.0, t_hi, t_lo, da_im, 0.0)
    invdb_re_hi, invdb_re_lo, invdb_im_hi, invdb_im_lo = dd_complex_div(
        1.0, 0.0, 0.0, 0.0, t_hi, t_lo, db_im, 0.0)

    # earg = -(amp1 * invda + amp2 * invdb)
    a1_re_hi, a1_re_lo = dd_mul(amp1, 0.0, invda_re_hi, invda_re_lo)
    a1_im_hi, a1_im_lo = dd_mul(amp1, 0.0, invda_im_hi, invda_im_lo)
    a2_re_hi, a2_re_lo = dd_mul(amp2, 0.0, invdb_re_hi, invdb_re_lo)
    a2_im_hi, a2_im_lo = dd_mul(amp2, 0.0, invdb_im_hi, invdb_im_lo)
    s_re_hi, s_re_lo, s_im_hi, s_im_lo = dd_complex_add(
        a1_re_hi, a1_re_lo, a1_im_hi, a1_im_lo,
        a2_re_hi, a2_re_lo, a2_im_hi, a2_im_lo)
    e_re_hi, e_re_lo, e_im_hi, e_im_lo = _ddc_exp(
        -s_re_hi, -s_re_lo, -s_im_hi, -s_im_lo)

    return dd_complex_mul(
        p_re_hi, p_re_lo, p_im_hi, p_im_lo,
        e_re_hi, e_re_lo, e_im_hi, e_im_lo)


@numba.njit(cache=True, fastmath=False)
def _g_dd(t_hi: float, t_lo: float, da_im: float, db_im: float,
          amp1: float, amp2: float) -> tuple[float, float, float, float]:
    """
    Return the analytic logarithmic derivative ``G(t) = h'(t)/h(t)``
    as a dd-complex, in closed form (never a finite difference):
    ``G = amp1/da^2 + amp2/db^2 - 1/2/da - 1/2/db``.
    """
    invda_re_hi, invda_re_lo, invda_im_hi, invda_im_lo = dd_complex_div(
        1.0, 0.0, 0.0, 0.0, t_hi, t_lo, da_im, 0.0)
    invdb_re_hi, invdb_re_lo, invdb_im_hi, invdb_im_lo = dd_complex_div(
        1.0, 0.0, 0.0, 0.0, t_hi, t_lo, db_im, 0.0)
    invda2_re_hi, invda2_re_lo, invda2_im_hi, invda2_im_lo = dd_complex_mul(
        invda_re_hi, invda_re_lo, invda_im_hi, invda_im_lo,
        invda_re_hi, invda_re_lo, invda_im_hi, invda_im_lo)
    invdb2_re_hi, invdb2_re_lo, invdb2_im_hi, invdb2_im_lo = dd_complex_mul(
        invdb_re_hi, invdb_re_lo, invdb_im_hi, invdb_im_lo,
        invdb_re_hi, invdb_re_lo, invdb_im_hi, invdb_im_lo)

    t1_re_hi, t1_re_lo = dd_mul(amp1, 0.0, invda2_re_hi, invda2_re_lo)
    t1_im_hi, t1_im_lo = dd_mul(amp1, 0.0, invda2_im_hi, invda2_im_lo)
    t2_re_hi, t2_re_lo = dd_mul(amp2, 0.0, invdb2_re_hi, invdb2_re_lo)
    t2_im_hi, t2_im_lo = dd_mul(amp2, 0.0, invdb2_im_hi, invdb2_im_lo)
    g_re_hi, g_re_lo, g_im_hi, g_im_lo = dd_complex_add(
        t1_re_hi, t1_re_lo, t1_im_hi, t1_im_lo,
        t2_re_hi, t2_re_lo, t2_im_hi, t2_im_lo)

    h1_re_hi, h1_re_lo = dd_mul(-0.5, 0.0, invda_re_hi, invda_re_lo)
    h1_im_hi, h1_im_lo = dd_mul(-0.5, 0.0, invda_im_hi, invda_im_lo)
    h2_re_hi, h2_re_lo = dd_mul(-0.5, 0.0, invdb_re_hi, invdb_re_lo)
    h2_im_hi, h2_im_lo = dd_mul(-0.5, 0.0, invdb_im_hi, invdb_im_lo)
    g_re_hi, g_re_lo, g_im_hi, g_im_lo = dd_complex_add(
        g_re_hi, g_re_lo, g_im_hi, g_im_lo,
        h1_re_hi, h1_re_lo, h1_im_hi, h1_im_lo)
    g_re_hi, g_re_lo, g_im_hi, g_im_lo = dd_complex_add(
        g_re_hi, g_re_lo, g_im_hi, g_im_lo,
        h2_re_hi, h2_re_lo, h2_im_hi, h2_im_lo)
    return g_re_hi, g_re_lo, g_im_hi, g_im_lo


@numba.njit(cache=True, fastmath=False)
def _raw_t_integral_core(
        w: float, a: float, b: float, y1: float, y2: float,
        u_lo: float, u_mid: float, u_hi: float, n_panels: int,
        xk_hi: np.ndarray, xk_lo: np.ndarray,
        wk_hi: np.ndarray, wk_lo: np.ndarray
) -> tuple[float, float, float, float]:
    """
    Return the RAW ``t``-integral ``Int_0^inf t^{s-1} h dt`` as a
    dd-complex ``(re_hi, re_lo, im_hi, im_lo)``, BEFORE any prefactor.

    Value ``= endpoint - (1/s) A + B`` with ``endpoint = T^s h(T) / s``
    evaluated at the actual split point ``T = e^{u_mid}``,
    ``A = Int_0^T t^s h'(t) dt`` (integrand in ``u = ln t`` is
    ``t^{s+1} h G``) and ``B = Int_T^inf t^{s-1} h dt`` (integrand
    ``t^s h``).  ``A`` is quadratured over ``n_panels`` composite panels
    tiling ``[u_lo, u_mid]`` (``u_mid = ln T``) and ``B`` over
    ``n_panels`` panels tiling ``[u_mid, u_hi]``, both using the single
    Newton-refined double-double Gauss-Legendre rule ``(xk, wk)`` on
    ``[-1, 1]``.

    The panel step, centre, and half-width are formed in DOUBLE-DOUBLE
    from the float64 range ends, so each node ``u = centre + halfwidth *
    x_k`` and its weight ``halfwidth * w_k`` are fully double-double.
    This matters because the oscillatory ``t^s`` phase ``(w/2) u`` reaches
    O(w * |u|) ~ 1700: a float64 panel centre would inject an
    uncorrelated ``~1e-16`` phase error per panel that does NOT cancel,
    flooring the raw integral at float64 precision -- fatal once the
    O(1/w) node terms cancel down to the ``e^{-pi w/4}`` result (F001-S)
    and reconstruction multiplies by ``e^{+pi w/4}``.  Only the shared
    float64 range end ``u_lo``/``u_mid`` survives as a global ``u`` shift,
    which is a harmless global rotation of the whole integral.
    """
    half_w = 0.5 * w
    da_im = -half_w * a           # > 0 (a < 0): da = t - i w a / 2
    db_im = -half_w * b           # < 0
    amp1 = 0.25 * w * w * y1 * y1
    amp2 = 0.25 * w * w * y2 * y2
    # 1/|s| = 1/half_w in DOUBLE-DOUBLE (s = i half_w, so
    # 1/s = -i/half_w and -1/s = +i/half_w).  A float64 fl(1.0/half_w)
    # carries an O(eps64) relative error shared by the endpoint and A
    # pieces but NOT by B (which has no 1/s factor), so it does not
    # cancel in the IBP combination, is bit-identical in the N and 2N
    # rules, and is amplified by e^{+pi w/4} on reconstruction (F001-S)
    # -- the same silent-certification failure mode as a float64 split
    # point.
    inv_hw_hi, inv_hw_lo = dd_div(1.0, 0.0, half_w, 0.0)
    n_nodes = xk_hi.shape[0]
    n_float = float(n_panels)

    # Endpoint T^s h(T) / s  (T real, |T^s| = 1), evaluated at the
    # ACTUAL quadrature split point T = e^{u_mid} (u_mid = fl(ln t_cap)):
    # deriving BOTH the phase (w/2) ln T = half_w * u_mid and the kernel
    # argument T = exp(u_mid) from the same u_mid keeps the IBP boundary
    # exactly consistent with the [u_lo, u_mid] / [u_mid, u_hi] domains.
    # Evaluating at t_cap instead leaves an O(eps64) T-inconsistency that
    # is bit-identical in the N and 2N rules (same u_mid), invisible to
    # the paired-rule certification, and amplified by e^{+pi w/4} into
    # relative error ~eps64 * e^{pi w/4} in F (F001-S).
    tcap_hi, tcap_lo = _dd_exp(u_mid, 0.0)
    theta_hi, theta_lo = dd_mul(half_w, 0.0, u_mid, 0.0)
    cc_hi, cc_lo, cs_hi, cs_lo = _dd_cos_sin_of_phase(theta_hi, theta_lo)
    h_re_hi, h_re_lo, h_im_hi, h_im_lo = _h_dd(
        tcap_hi, tcap_lo, da_im, db_im, amp1, amp2)
    th_re_hi, th_re_lo, th_im_hi, th_im_lo = dd_complex_mul(
        cc_hi, cc_lo, cs_hi, cs_lo, h_re_hi, h_re_lo, h_im_hi, h_im_lo)
    acc_re_hi, acc_re_lo, acc_im_hi, acc_im_lo = dd_complex_mul(
        th_re_hi, th_re_lo, th_im_hi, th_im_lo,
        0.0, 0.0, -inv_hw_hi, -inv_hw_lo)

    # A = Int_{u_lo}^{u_mid} t^{s+1} h G du; contributes -(1/s) A.
    span_hi, span_lo = dd_sub(u_mid, 0.0, u_lo, 0.0)
    step_hi, step_lo = dd_div(span_hi, span_lo, n_float, 0.0)
    half_hi, half_lo = dd_mul(0.5, 0.0, step_hi, step_lo)
    for p in range(n_panels):
        pc_hi, pc_lo = dd_mul(float(p) + 0.5, 0.0, step_hi, step_lo)
        center_hi, center_lo = dd_add(u_lo, 0.0, pc_hi, pc_lo)
        for k in range(n_nodes):
            offs_hi, offs_lo = dd_mul(half_hi, half_lo, xk_hi[k], xk_lo[k])
            u_node_hi, u_node_lo = dd_add(
                center_hi, center_lo, offs_hi, offs_lo)
            wt_hi, wt_lo = dd_mul(half_hi, half_lo, wk_hi[k], wk_lo[k])
            t_hi, t_lo = _dd_exp(u_node_hi, u_node_lo)
            ph_hi, ph_lo = dd_mul(half_w, 0.0, u_node_hi, u_node_lo)
            co_hi, co_lo, si_hi, si_lo = _dd_cos_sin_of_phase(ph_hi, ph_lo)
            # t^{s+1} = t * (cos + i sin)
            tps_re_hi, tps_re_lo = dd_mul(t_hi, t_lo, co_hi, co_lo)
            tps_im_hi, tps_im_lo = dd_mul(t_hi, t_lo, si_hi, si_lo)
            hh_re_hi, hh_re_lo, hh_im_hi, hh_im_lo = _h_dd(
                t_hi, t_lo, da_im, db_im, amp1, amp2)
            gg_re_hi, gg_re_lo, gg_im_hi, gg_im_lo = _g_dd(
                t_hi, t_lo, da_im, db_im, amp1, amp2)
            f_re_hi, f_re_lo, f_im_hi, f_im_lo = dd_complex_mul(
                tps_re_hi, tps_re_lo, tps_im_hi, tps_im_lo,
                hh_re_hi, hh_re_lo, hh_im_hi, hh_im_lo)
            f_re_hi, f_re_lo, f_im_hi, f_im_lo = dd_complex_mul(
                f_re_hi, f_re_lo, f_im_hi, f_im_lo,
                gg_re_hi, gg_re_lo, gg_im_hi, gg_im_lo)
            # * (-1/s) = (0, +1/half_w), the reciprocal in dd
            f_re_hi, f_re_lo, f_im_hi, f_im_lo = dd_complex_mul(
                f_re_hi, f_re_lo, f_im_hi, f_im_lo,
                0.0, 0.0, inv_hw_hi, inv_hw_lo)
            c_re_hi, c_re_lo = dd_mul(wt_hi, wt_lo, f_re_hi, f_re_lo)
            c_im_hi, c_im_lo = dd_mul(wt_hi, wt_lo, f_im_hi, f_im_lo)
            acc_re_hi, acc_re_lo, acc_im_hi, acc_im_lo = dd_complex_add(
                acc_re_hi, acc_re_lo, acc_im_hi, acc_im_lo,
                c_re_hi, c_re_lo, c_im_hi, c_im_lo)

    # B = Int_{u_mid}^{u_hi} t^s h du.
    span_hi, span_lo = dd_sub(u_hi, 0.0, u_mid, 0.0)
    step_hi, step_lo = dd_div(span_hi, span_lo, n_float, 0.0)
    half_hi, half_lo = dd_mul(0.5, 0.0, step_hi, step_lo)
    for p in range(n_panels):
        pc_hi, pc_lo = dd_mul(float(p) + 0.5, 0.0, step_hi, step_lo)
        center_hi, center_lo = dd_add(u_mid, 0.0, pc_hi, pc_lo)
        for k in range(n_nodes):
            offs_hi, offs_lo = dd_mul(half_hi, half_lo, xk_hi[k], xk_lo[k])
            u_node_hi, u_node_lo = dd_add(
                center_hi, center_lo, offs_hi, offs_lo)
            wt_hi, wt_lo = dd_mul(half_hi, half_lo, wk_hi[k], wk_lo[k])
            t_hi, t_lo = _dd_exp(u_node_hi, u_node_lo)
            ph_hi, ph_lo = dd_mul(half_w, 0.0, u_node_hi, u_node_lo)
            co_hi, co_lo, si_hi, si_lo = _dd_cos_sin_of_phase(ph_hi, ph_lo)
            hh_re_hi, hh_re_lo, hh_im_hi, hh_im_lo = _h_dd(
                t_hi, t_lo, da_im, db_im, amp1, amp2)
            # f = t^s * h
            f_re_hi, f_re_lo, f_im_hi, f_im_lo = dd_complex_mul(
                co_hi, co_lo, si_hi, si_lo,
                hh_re_hi, hh_re_lo, hh_im_hi, hh_im_lo)
            c_re_hi, c_re_lo = dd_mul(wt_hi, wt_lo, f_re_hi, f_re_lo)
            c_im_hi, c_im_lo = dd_mul(wt_hi, wt_lo, f_im_hi, f_im_lo)
            acc_re_hi, acc_re_lo, acc_im_hi, acc_im_lo = dd_complex_add(
                acc_re_hi, acc_re_lo, acc_im_hi, acc_im_lo,
                c_re_hi, c_re_lo, c_im_hi, c_im_lo)

    return acc_re_hi, acc_re_lo, acc_im_hi, acc_im_lo


@numba.njit(cache=True, fastmath=False)
def _legendre_p_and_deriv_dd(
        order: int, x_hi: float, x_lo: float
) -> tuple[float, float, float, float]:
    """
    Return ``(P_n, P_n')`` (each dd) of the Legendre polynomial of degree
    ``order`` at the dd argument ``x``, via the three-term recurrence.

    ``P_n' = n (x P_n - P_{n-1}) / (x^2 - 1)`` uses the ``P_{n-1}`` left
    in ``p_prev`` by the recurrence.  Used only to Newton-refine the
    leggauss seed to double-double in `_dd_gl_rule_core`.
    """
    if order == 0:
        return 1.0, 0.0, 0.0, 0.0
    p_prev_hi, p_prev_lo = 1.0, 0.0                 # P_0
    p_cur_hi, p_cur_lo = x_hi, x_lo                 # P_1
    for m in range(2, order + 1):
        xp_hi, xp_lo = dd_mul(x_hi, x_lo, p_cur_hi, p_cur_lo)
        t1_hi, t1_lo = dd_mul(float(2 * m - 1), 0.0, xp_hi, xp_lo)
        t2_hi, t2_lo = dd_mul(float(m - 1), 0.0, p_prev_hi, p_prev_lo)
        num_hi, num_lo = dd_sub(t1_hi, t1_lo, t2_hi, t2_lo)
        pm_hi, pm_lo = dd_div(num_hi, num_lo, float(m), 0.0)
        p_prev_hi, p_prev_lo = p_cur_hi, p_cur_lo
        p_cur_hi, p_cur_lo = pm_hi, pm_lo

    xpn_hi, xpn_lo = dd_mul(x_hi, x_lo, p_cur_hi, p_cur_lo)
    dnum_hi, dnum_lo = dd_sub(xpn_hi, xpn_lo, p_prev_hi, p_prev_lo)
    dnum_hi, dnum_lo = dd_mul(float(order), 0.0, dnum_hi, dnum_lo)
    x2_hi, x2_lo = dd_mul(x_hi, x_lo, x_hi, x_lo)
    den_hi, den_lo = dd_sub(x2_hi, x2_lo, 1.0, 0.0)
    dp_hi, dp_lo = dd_div(dnum_hi, dnum_lo, den_hi, den_lo)
    return p_cur_hi, p_cur_lo, dp_hi, dp_lo


@numba.njit(cache=True, fastmath=False)
def _dd_gl_rule_core(
        order: int, seed_nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Refine float64 leggauss nodes to double-double and build dd weights.

    Each node gets three dd Newton steps ``x <- x - P_n(x)/P_n'(x)`` from
    the float64 seed (quadratic convergence: 16 -> 32 correct digits);
    the weight is ``w = 2 / ((1 - x^2) P_n'(x)^2)`` evaluated in dd.
    Returns ``(x_hi, x_lo, w_hi, w_lo)`` on ``[-1, 1]``.
    """
    m = seed_nodes.shape[0]
    x_hi = np.empty(m)
    x_lo = np.empty(m)
    w_hi = np.empty(m)
    w_lo = np.empty(m)
    for i in range(m):
        xi_hi = seed_nodes[i]
        xi_lo = 0.0
        for _ in range(3):
            p_hi, p_lo, dp_hi, dp_lo = _legendre_p_and_deriv_dd(
                order, xi_hi, xi_lo)
            corr_hi, corr_lo = dd_div(p_hi, p_lo, dp_hi, dp_lo)
            xi_hi, xi_lo = dd_sub(xi_hi, xi_lo, corr_hi, corr_lo)
        _, _, dp_hi, dp_lo = _legendre_p_and_deriv_dd(order, xi_hi, xi_lo)
        x2_hi, x2_lo = dd_mul(xi_hi, xi_lo, xi_hi, xi_lo)
        omx2_hi, omx2_lo = dd_sub(1.0, 0.0, x2_hi, x2_lo)
        dp2_hi, dp2_lo = dd_mul(dp_hi, dp_lo, dp_hi, dp_lo)
        den_hi, den_lo = dd_mul(omx2_hi, omx2_lo, dp2_hi, dp2_lo)
        wi_hi, wi_lo = dd_div(2.0, 0.0, den_hi, den_lo)
        x_hi[i] = xi_hi
        x_lo[i] = xi_lo
        w_hi[i] = wi_hi
        w_lo[i] = wi_lo
    return x_hi, x_lo, w_hi, w_lo


@functools.lru_cache(maxsize=None)
def _dd_gl_rule(
        order: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the cached double-double GL rule ``(x_hi, x_lo, w_hi, w_lo)``."""
    seed_nodes, _ = np.polynomial.legendre.leggauss(order)
    return _dd_gl_rule_core(order, np.ascontiguousarray(seed_nodes))


def _panel_count(margin: float, w: float) -> int:
    """
    Return the composite-panel count per side for the coarse (``N``) rule.

    The oscillation period of the ``t^s`` phase ``(w/2) u`` is ``4 pi /
    w`` in ``u``; each panel spans `_WAVELENGTHS_PER_PANEL` of them,
    resolved spectrally by the fixed order-`_PANEL_ORDER` rule.  The
    ``2N`` rule doubles the count; the paired-rule certification refines
    the PANEL COUNT, not the per-panel order.
    """
    wavelength = 4.0 * math.pi / w
    return max(
        _MIN_PANELS,
        int(np.ceil(margin / (_WAVELENGTHS_PER_PANEL * wavelength))))


def _validate_inputs(w: float, y_eig: np.ndarray,
                     gamma_prime: float) -> None:
    """
    Guard the pure-eigenframe saddle domain (raises `ValueError`).

    The ``w > W_CEILING_SCHWINGER`` ceiling and the paired-rule
    quadrature refusal are NOT domain errors; they raise
    `SchwingerCertificationError` from `f_schwinger`.
    """
    if not w > 0.0:
        raise ValueError(
            f'The dimensionless frequency must be strictly positive; '
            f'got w = {w}.')
    if not gamma_prime > 0.0:
        raise ValueError(
            f'The Schwinger branch requires reduced shear '
            f'gamma_prime > 0 (det A != 0); got gamma_prime = '
            f'{gamma_prime}. Positive parity (gamma_prime < 1) and the '
            f'macro saddle (gamma_prime > 1) are both valid; '
            f'gamma_prime == 1 is the degenerate parity boundary '
            f'(det A = 0), caught by the paired-rule certification '
            f'pinch rather than here.')
    if y_eig.shape != (2,):
        raise ValueError(
            f'y_eig must be an eigenframe position of shape (2,); got '
            f'shape {y_eig.shape}.')
    if not np.all(np.isfinite(y_eig)):
        raise ValueError(f'y_eig must be finite; got {y_eig!r}.')


def _reconstruct(w: float, y_eig: np.ndarray, integral: complex) -> complex:
    """
    Apply the outer prefactors to the raw ``t``-integral to form ``F``.

    ``F = (w / (2 pi i)) e^{i w |y|^2 / 2} (pi / Gamma(iw/2)) I``.  The
    scalar factor ``(w / 2 pi i) pi = -i w / 2`` is exact; the two large
    phases (``w |y|^2 / 2`` and the imaginary part of ``ln Gamma``) are
    reduced modulo ``2*pi`` before exponentiation, and the ``1/Gamma``
    magnitude (``~e^{+pi w/4}``) is taken from ``-Re ln Gamma`` so it
    never overflows.
    """
    prefactor = complex(0.0, -0.5 * w)

    phase_y = 0.5 * w * (y_eig[0] * y_eig[0] + y_eig[1] * y_eig[1])
    reduced_y = _reduce_phase(phase_y, 0.0)
    exp_y = complex(math.cos(reduced_y), math.sin(reduced_y))

    log_gamma = loggamma(complex(0.0, 0.5 * w))
    inv_gamma_magnitude = math.exp(-log_gamma.real)
    reduced_gamma_phase = _reduce_phase(-log_gamma.imag, 0.0)
    inv_gamma = inv_gamma_magnitude * complex(
        math.cos(reduced_gamma_phase), math.sin(reduced_gamma_phase))

    return prefactor * exp_y * inv_gamma * integral


@functools.lru_cache(maxsize=None)
def _mp_gl_rule(order: int, dps: int) -> tuple[list, list]:
    """Return the cached mpmath Gauss-Legendre ``(nodes, weights)``.

    Cache key is ``(order, dps)`` because ``mp.gauss_quadrature``
    snapshots the global mpmath precision.  Called only from the
    mpmath QD path where ``mpmath`` is already imported.
    """
    global _mpmath
    if _mpmath is None:
        import mpmath as _mpmath
    nodes, weights = _mpmath.gauss_quadrature(order, 'legendre')
    return list(nodes), list(weights)


def _f_schwinger_mpmath(w: float, y_eig: np.ndarray,
                        gamma_prime: float) -> complex:
    """
    Evaluate the 1D Schwinger integral using mpmath arbitrary-precision
    quadrature for ``w > W_CEILING_SCHWINGER``.

    Follows the IDENTICAL IBP structure as the double-double path:
    split at ``T = w (|a| + |b| + 2) / 2``, integration by parts on
    ``[0, T]`` (removing the ``t^{s-1}`` singularity), direct tail on
    ``[T, inf)``, both in ``u = ln t``.  Each panel is evaluated with a
    fixed-order ``_MP_PANEL_ORDER`` Gauss-Legendre rule.

    Certification is N/2N paired-rule on the RECONSTRUCTED ``F``
    (computed in mpmath): if
    ``|F_N - F_2N| / |F_2N| > _CERTIFICATION_TOL``, raises
    `SchwingerCertificationError`.

    Parameters
    ----------
    w, y_eig, gamma_prime : same semantics as `f_schwinger`.

    Returns
    -------
    complex
        The pure-shear amplification ``F_{0, gamma'}(w, y_eig)``.

    Raises
    ------
    SchwingerCertificationError
        If the paired N/2N rules disagree above `_CERTIFICATION_TOL`.
    ImportError
        If mpmath is not installed.
    """
    global _mpmath
    if _mpmath is None:
        try:
            import mpmath as _mpmath
        except ImportError:
            raise ImportError(
                'mpmath is required for Schwinger evaluation at w > 60; '
                'install with: pip install cogwheel[training]') from None

    mp = _mpmath

    # Set working precision: 30 base digits + ceil(w) for the
    # e^{pi w/4} cancellation (each unit of w costs ~0.8 digits).
    mp.mp.dps = 30 + int(math.ceil(w))

    a = 1.0 - gamma_prime
    b = 1.0 + gamma_prime

    # mpmath constants for the kernel
    w_ = mp.mpf(w)
    s = mp.mpc(0, w_ / 2)
    branch_a = mp.mpc(0, w_ * mp.mpf(a) / 2)
    branch_b = mp.mpc(0, w_ * mp.mpf(b) / 2)
    amp1 = (w_ * mp.mpf(y_eig[0])) ** 2 / 4
    amp2 = (w_ * mp.mpf(y_eig[1])) ** 2 / 4

    def kernel(t):
        """h(t) = exp(-amp1/da - amp2/db) / (sqrt(da) * sqrt(db))."""
        da = t - branch_a
        db = t - branch_b
        return (mp.exp(-amp1 / da - amp2 / db)
                / (mp.sqrt(da) * mp.sqrt(db)))

    def kernel_derivative(t):
        """h'(t) = h(t) * G(t) with G = amp1/da^2 + amp2/db^2 - 1/2/da - 1/2/db."""
        da = t - branch_a
        db = t - branch_b
        return kernel(t) * (amp1 / da ** 2 + amp2 / db ** 2
                            - 1 / (2 * da) - 1 / (2 * db))

    # IBP split point and u-range (identical formulas to the DD path)
    t_cap = w_ * (abs(mp.mpf(a)) + abs(mp.mpf(b)) + 2) / 2
    u_mid = mp.log(t_cap)
    margin = mp.pi * w_ / 4 + _U_MARGIN_CONST

    # Panel count: same formula as _panel_count
    wavelength = 4 * mp.pi / w_
    n_panels = max(
        _MIN_PANELS,
        int(mp.ceil(margin / (_WAVELENGTHS_PER_PANEL * wavelength))))

    # Cached Gauss-Legendre rule for this order and precision
    gl_nodes, gl_weights = _mp_gl_rule(_MP_PANEL_ORDER, mp.mp.dps)

    def _raw_integral_mp(n_side):
        """Compute the raw t-integral with ``n_side`` panels per side
        using a fixed-order ``_MP_PANEL_ORDER`` Gauss-Legendre rule."""
        panel_width = margin / n_side
        hw = mp.mpf(panel_width) / 2

        # Part A: Int_{u_lo}^{u_mid} t^{s+1} h'(t) du  (IBP piece)
        u_lo = u_mid - margin
        u_a = u_lo + hw  # first panel centre
        part_a = mp.mpf(0)
        for i in range(n_side):
            panel_centre = u_a + mp.mpf(i) * mp.mpf(panel_width)
            for x_k, w_k in zip(gl_nodes, gl_weights):
                u_k = panel_centre + hw * x_k
                part_a += w_k * mp.exp((s + 1) * u_k) * kernel_derivative(mp.exp(u_k))
        part_a *= hw

        # Tail B: Int_{u_mid}^{u_hi} t^s h(t) du
        u_hi = u_mid + margin
        u_b = u_mid + hw  # first panel centre
        tail = mp.mpf(0)
        for i in range(n_side):
            panel_centre = u_b + mp.mpf(i) * mp.mpf(panel_width)
            for x_k, w_k in zip(gl_nodes, gl_weights):
                u_k = panel_centre + hw * x_k
                tail += w_k * mp.exp(s * u_k) * kernel(mp.exp(u_k))
        tail *= hw

        # Endpoint: T^s h(T) / s
        endpoint = t_cap ** s * kernel(t_cap) / s

        # IBP combination: endpoint - (1/s)*A + B
        raw = endpoint - part_a / s + tail
        return raw

    # Paired N/2N certification (reconstructed F compared in mpmath)
    raw_n = _raw_integral_mp(n_panels)
    raw_2n = _raw_integral_mp(2 * n_panels)

    y1_mp = mp.mpf(y_eig[0])
    y2_mp = mp.mpf(y_eig[1])

    def _mp_reconstruct(integral_mp):
        """Reconstruct F from the raw t-integral using mpmath arithmetic."""
        prefactor = mp.mpc(0, -0.5 * w_)
        phase_y = 0.5 * w_ * (y1_mp ** 2 + y2_mp ** 2)
        exp_y = mp.exp(mp.mpc(0, phase_y))
        log_gamma = mp.loggamma(mp.mpc(0, 0.5 * w_))
        inv_gamma_mag = mp.exp(-log_gamma.real)
        inv_gamma_phase = mp.exp(mp.mpc(0, -log_gamma.imag))
        return prefactor * exp_y * inv_gamma_mag * inv_gamma_phase * integral_mp

    f_n_mp = _mp_reconstruct(raw_n)
    f_2n_mp = _mp_reconstruct(raw_2n)

    ref_mag = abs(f_2n_mp)
    diff_mag = abs(f_n_mp - f_2n_mp)

    if ref_mag == 0.0 or diff_mag > _CERTIFICATION_TOL * ref_mag:
        if ref_mag == 0.0:
            relative = math.inf
        else:
            relative_mag = diff_mag / ref_mag
            relative = float(relative_mag)
        raise SchwingerCertificationError(
            f'Saddle wave branch (mpmath QD path) refused at w = {w}, '
            f'y_eig = ({y_eig[0]}, {y_eig[1]}), '
            f'gamma_prime = {gamma_prime}: paired N/2N rules disagree by '
            f'{relative:.3e} on the reconstructed F, above the '
            f'{_CERTIFICATION_TOL:.0e} threshold.')

    return complex(f_2n_mp)


def _dd_complex_magnitude(value: tuple[float, float, float, float]) -> float:
    """Return the float64 magnitude of a dd-complex ``(re_hi, re_lo, ...)``."""
    return math.hypot(value[0] + value[1], value[2] + value[3])


def f_schwinger(w: float, y_eig, gamma_prime: float) -> complex:
    """
    Evaluate the pure-shear saddle amplification ``F_{0, gamma'}(w, y)``.

    Parameters
    ----------
    w : float
        Dimensionless lens frequency ``w = 8 pi G M_L (1 + z_L) f / c^3``,
        strictly positive.  ``w <= 60`` uses the double-double path;
        ``60 < w <= 150`` dispatches to the mpmath arbitrary-precision
        path; ``w > 150`` (`W_CEILING_SCHWINGER_QD`) is an unconditional
        hard refuse.
    y_eig : array_like, shape (2,)
        Source position in the SHEAR EIGENFRAME (``e1`` = soft axis,
        ``e2`` = hard axis), dimensionless.
    gamma_prime : float
        Reduced external shear ``gamma' > 0`` (positive parity or macro
        saddle).

    Returns
    -------
    complex
        The pure-shear amplification ``F_{0, gamma'}(w, y_eig)``.

    Raises
    ------
    ValueError
        If ``w <= 0``, ``gamma_prime <= 0``, or ``y_eig`` is not a finite
        shape-``(2,)`` position.
    SchwingerCertificationError
        If ``w > W_CEILING_SCHWINGER_QD``, or the paired ``N``/``2N``
        rules disagree by more than `_CERTIFICATION_TOL` (relative):
        on the raw ``t``-integral (DD path, ``w <= 60``) or on the
        reconstructed ``F`` (mpmath path, ``60 < w <= 150``).
    """
    w = float(w)
    gamma_prime = float(gamma_prime)
    y_eig = np.asarray(y_eig, dtype=np.float64)
    _validate_inputs(w, y_eig, gamma_prime)

    if w > W_CEILING_SCHWINGER_QD:
        raise SchwingerCertificationError(
            f'Saddle wave branch refused: w = {w} exceeds the QD '
            f'ceiling W_CEILING_SCHWINGER_QD = {W_CEILING_SCHWINGER_QD} '
            f'(mpmath runtime O(w * dps^2) exceeds training budget).')
    if w > W_CEILING_SCHWINGER:
        return _f_schwinger_mpmath(w, y_eig, gamma_prime)

    a = 1.0 - gamma_prime
    b = 1.0 + gamma_prime
    t_cap = 0.5 * w * (abs(a) + abs(b) + 2.0)
    log_t_cap = math.log(t_cap)

    # u = ln t range: the y-independent cancellation depth pi w / 4
    # (`_CANCEL_SCALE`) plus a fixed slack so the (N-and-2N-identical,
    # hence uncertified) domain truncation sits far below the tolerance
    # on the tiny raw integral.
    margin = _CANCEL_SCALE * w + _U_MARGIN_CONST
    u_lo = log_t_cap - margin
    u_hi = log_t_cap + margin

    n_panels = _panel_count(margin, w)
    xk_hi, xk_lo, wk_hi, wk_lo = _dd_gl_rule(_PANEL_ORDER)

    y1 = y_eig[0]
    y2 = y_eig[1]
    estimates = []
    for n_side in (n_panels, 2 * n_panels):
        estimates.append(_raw_t_integral_core(
            w, a, b, y1, y2, u_lo, log_t_cap, u_hi, n_side,
            xk_hi, xk_lo, wk_hi, wk_lo))
    integral_n, integral_2n = estimates

    difference = dd_complex_sub(
        integral_n[0], integral_n[1], integral_n[2], integral_n[3],
        integral_2n[0], integral_2n[1], integral_2n[2], integral_2n[3])
    reference_magnitude = _dd_complex_magnitude(integral_2n)
    difference_magnitude = _dd_complex_magnitude(difference)

    if (reference_magnitude == 0.0
            or difference_magnitude
            > _CERTIFICATION_TOL * reference_magnitude):
        relative = (math.inf if reference_magnitude == 0.0
                    else difference_magnitude / reference_magnitude)
        raise SchwingerCertificationError(
            f'Saddle wave branch refused at w = {w}, '
            f'y_eig = ({y1}, {y2}), gamma_prime = {gamma_prime}: paired '
            f'Gauss-Legendre rules disagree by {relative:.3e} on the raw '
            f't-integral, above the {_CERTIFICATION_TOL:.0e} threshold '
            f'(quadrature under-resolved or the |a| -> 0 parity boundary '
            f'is pinching the contour).')

    integral = complex(integral_2n[0] + integral_2n[1],
                       integral_2n[2] + integral_2n[3])
    return _reconstruct(w, y_eig, integral)


def _measure_warm_cost(
        w_values: tuple[float, ...] = (5.0, 10.0, 20.0),
        gamma_values: tuple[float, ...] = (1.05, 1.3),
        y_values: tuple[tuple[float, float], ...] = ((0.2, 0.1), (0.6, 0.4)),
        repeats: int = 5) -> dict[str, float]:
    """
    Measure the warm per-point cost of `f_schwinger` over a small grid.

    This is a MEASUREMENT only -- it prices the envelope-surrogate
    decision and never touches evaluator control flow.  Returns a
    summary dict (mean / min / max ms per point) and prints one line.
    The grid deliberately stops well below the ``w = 60`` ceiling so the
    measurement stays bounded.
    """
    import time  # local: keep the timing dependency out of the hot module

    points = [(w, np.array(y, dtype=np.float64), gamma)
              for w in w_values
              for gamma in gamma_values
              for y in y_values]

    # Warm up: trigger numba compilation and lru_cache population once.
    for w, y_eig, gamma in points:
        try:
            f_schwinger(w, y_eig, gamma)
        except SchwingerCertificationError:
            pass

    per_point_ms = []
    for w, y_eig, gamma in points:
        best = math.inf
        for _ in range(repeats):
            start = time.perf_counter()
            try:
                f_schwinger(w, y_eig, gamma)
            except SchwingerCertificationError:
                pass
            best = min(best, time.perf_counter() - start)
        per_point_ms.append(1e3 * best)

    summary = {
        'n_points': float(len(per_point_ms)),
        'mean_ms': float(np.mean(per_point_ms)),
        'min_ms': float(np.min(per_point_ms)),
        'max_ms': float(np.max(per_point_ms)),
    }
    print(
        f'[_schwinger warm cost] {summary["n_points"]:.0f} points | '
        f'mean {summary["mean_ms"]:.2f} ms/point | '
        f'min {summary["min_ms"]:.2f} | max {summary["max_ms"]:.2f}')
    return summary


if __name__ == '__main__':
    _measure_warm_cost()
