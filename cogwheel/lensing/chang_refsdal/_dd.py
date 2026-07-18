"""
Double-double (two-double) real and complex arithmetic primitives.

WHY THIS EXISTS
---------------
The Chang-Refsdal amplification evaluates alternating series whose
intermediate terms are exponentially larger than the result, losing
roughly ``0.434 * L`` decimal digits to cancellation. There are TWO
such cancellation channels; they are INDEPENDENT, live at SEPARATE
code sites, and do NOT compound into a single summed exponent:

    L_1F1 = w * |y'|        (the 1F1 kernel series, in _hyp1f1.py)
    L_op  = w * gamma'/2    (the operator series, in operator.py)

Double-double is required for the FORMER only. The 1F1 partial terms
reach ``e**(w*|y'|)`` while the sum is O(1), so the series' relative
error follows ``~ eps * e**(w*|y'|)``; Kahan summation does not rescue
it (its bound also carries ``sum|term_i|``), only a smaller eps does,
so the terms AND their accumulation go through this module. The
operator channel is handled elsewhere and differently: `operator.py`
MEASURES its own cancellation ratio ``max_partial_term / |total|`` and
REFUSES (raising ``CancellationError``) past ~1e13 rather than leaning
on dd, so ``L_op`` never demands extended precision here.

Plain float64 (eps = 2.2e-16, ~15.95 digits) absorbs only
``w*|y'| ~ 22``. A double-double number is an unevaluated sum
``hi + lo`` of two non-overlapping float64s, giving eps ~ 1.2e-32
(~31.9 digits) and holding the 1e-10 target out to ``w*|y'| ~ 50``,
degrading to ~1e-6 at the ceiling ``w*|y'| = 60`` (with
``Y = sqrt(s) = |y'|``). That 60 is chosen deliberately: it overlaps
the geometric branch's ``w*|y'| >= 50`` onset, closing a gap that
float64's 22 cannot bridge. Double-double extends the MANTISSA, not
the exponent, so it cannot rescue an overflow -- which is why the
``w <= 500`` frequency ceiling is a hard gate rather than a precision
knob (the kernel ladder magnitude reaches the float64 overflow rail
near ``w ~ 700``). The obvious alternative, mpmath, is ~1e6 times too
slow to sit inside a likelihood loop; it is retained as a test-time
oracle only.

ARITHMETIC
----------
Dekker's (1971) error-free transformations: TwoSum for addition and
TwoProduct for multiplication, the latter built on the Veltkamp split
with the splitting constant ``2**27 + 1 = 134217729.0``. The split is
used deliberately in place of a fused multiply-add: numba exposes no
reliable FMA intrinsic, so ``math.fma`` is not an option here. Division
follows the Hida-Li-Bailey (QD library) accurate-division iteration.

REPRESENTATION
--------------
A dd real is two float64 scalars ``(hi, lo)``; a dd complex is FOUR
separate float64 scalars ``(re_hi, re_lo, im_hi, im_lo)``, passed and
returned individually as flat scalars -- never nested tuples, dataclass
instances, or object arrays. This is what keeps the module in numba's
nopython mode: plain scalars, plain loops, plain arrays.

HOT PATH
--------
The intended consumers are the 1F1 k-ladder recurrence and the operator
series accumulation, which call `dd_complex_mul` and `dd_complex_add`
once per series term. Every function here is written ``@njit``-shaped
(no closures, no containers, homogeneous scalar returns) and is now
decorated with ``@numba.njit(cache=True, fastmath=False)`` so it
inlines into the njit kernels in `_hyp1f1` and `operator`. ``fastmath``
is False deliberately: the two-sum / two-product error-free transforms
depend on strict IEEE-754 semantics, which ``fastmath`` reassociation
would destroy. Compilation happens once at first call (a warm cost, not
a per-evaluation one) and is a no-op for semantics; the pure-Python
bodies remain reachable through each function's ``py_func`` attribute
for the sensitivity tests.

Deliberately NOT provided: dd sqrt/exp/log/trig. Nothing in the hot
path needs them -- the operator prefactor is evaluated once, in plain
double, from its closed form.

References
----------
Dekker, T. J. (1971), "A floating-point technique for extending the
available precision", Numer. Math. 18, 224-242.
Hida, Y., Li, X. S., Bailey, D. H. (2001), "Algorithms for quad-double
precision floating point arithmetic" (the QD library).
"""
from __future__ import annotations

import numba

# The four-scalar dd-complex representation is a numba-compatibility
# requirement (see module docstring), so the binary complex operations
# unavoidably take 8 positional scalars.
# pylint: disable=too-many-arguments, too-many-locals

#: Veltkamp splitting constant, 2**27 + 1 (Dekker 1971).
#:
#: The load-bearing property is that `_split` must return halves of at
#: most 26 significant bits each, so that every cross product in
#: `_two_prod` fits in float64's 53-bit significand and the residual is
#: exact. For float64 that requires a splitter of 2**27 (+/- a small
#: constant); 2**26 + 1 leaves 27 bits in the high word and 2**28 + 1
#: leaves 27 in the low word, and either silently destroys the
#: error-free product. `SplitterSensitivityTestCase` in
#: `cogwheel/tests/test_lensing_dd.py` pins this invariant directly.
_SPLITTER = 134217729.0

#: Magnitude above which a naive Veltkamp split would overflow, 2**996.
_SPLIT_THRESHOLD = 6.69692879491417e+299

#: Down/up scalings used to split safely near the overflow rail.
_SCALE_DOWN = 3.7252902984619140625e-09  # 2**-28
_SCALE_UP = 268435456.0  # 2**28


@numba.njit(cache=True, fastmath=False)
def _two_sum(a: float, b: float) -> tuple[float, float]:
    """
    Return ``(s, e)`` with ``s = fl(a + b)`` and ``a + b == s + e``
    exactly. Dekker's TwoSum; no ordering assumption on the inputs.
    """
    s = a + b
    b_virtual = s - a
    a_virtual = s - b_virtual
    return s, (a - a_virtual) + (b - b_virtual)


@numba.njit(cache=True, fastmath=False)
def _quick_two_sum(a: float, b: float) -> tuple[float, float]:
    """
    Return ``(s, e)`` with ``s = fl(a + b)`` and ``a + b == s + e``
    exactly, assuming ``abs(a) >= abs(b)``. Cheaper than `_two_sum`;
    the caller owns the ordering precondition.
    """
    s = a + b
    return s, b - (s - a)


@numba.njit(cache=True, fastmath=False)
def _split(a: float) -> tuple[float, float]:
    """
    Split ``a`` into non-overlapping halves ``(hi, lo)`` of at most 26
    significant bits each, with ``a == hi + lo`` exactly.

    Values beyond ``2**996`` are scaled down before splitting and back
    up afterwards, so that the internal ``_SPLITTER * a`` cannot
    overflow.
    """
    if a > _SPLIT_THRESHOLD or a < -_SPLIT_THRESHOLD:
        scaled = a * _SCALE_DOWN
        temp = _SPLITTER * scaled
        hi = temp - (temp - scaled)
        lo = scaled - hi
        return hi * _SCALE_UP, lo * _SCALE_UP

    temp = _SPLITTER * a
    hi = temp - (temp - a)
    return hi, a - hi


@numba.njit(cache=True, fastmath=False)
def _two_prod(a: float, b: float) -> tuple[float, float]:
    """
    Return ``(p, e)`` with ``p = fl(a * b)`` and ``a * b == p + e``
    exactly. Dekker's TwoProduct via the Veltkamp split (no FMA).
    """
    p = a * b
    a_hi, a_lo = _split(a)
    b_hi, b_lo = _split(b)
    err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, err


@numba.njit(cache=True, fastmath=False)
def dd_from_float(a: float) -> tuple[float, float]:
    """Return the dd representation of a float64, exactly."""
    return a, 0.0


@numba.njit(cache=True, fastmath=False)
def dd_to_float(a_hi: float, a_lo: float) -> float:
    """Return the nearest float64 to the dd number ``a_hi + a_lo``."""
    return a_hi + a_lo


@numba.njit(cache=True, fastmath=False)
def dd_add(a_hi: float, a_lo: float,
           b_hi: float, b_lo: float) -> tuple[float, float]:
    """
    Return the dd sum ``(a_hi + a_lo) + (b_hi + b_lo)``.

    Uses the QD accurate (as opposed to "sloppy") addition, which keeps
    both low words and is correct to ~1 ulp of the dd format for
    arbitrary operands, including catastrophically cancelling ones.
    """
    s_hi, s_lo = _two_sum(a_hi, b_hi)
    t_hi, t_lo = _two_sum(a_lo, b_lo)
    s_lo += t_hi
    s_hi, s_lo = _quick_two_sum(s_hi, s_lo)
    s_lo += t_lo
    return _quick_two_sum(s_hi, s_lo)


@numba.njit(cache=True, fastmath=False)
def dd_neg(a_hi: float, a_lo: float) -> tuple[float, float]:
    """Return the dd negation ``-(a_hi + a_lo)``, exactly."""
    return -a_hi, -a_lo


@numba.njit(cache=True, fastmath=False)
def dd_sub(a_hi: float, a_lo: float,
           b_hi: float, b_lo: float) -> tuple[float, float]:
    """Return the dd difference ``(a_hi + a_lo) - (b_hi + b_lo)``."""
    return dd_add(a_hi, a_lo, -b_hi, -b_lo)


@numba.njit(cache=True, fastmath=False)
def dd_mul(a_hi: float, a_lo: float,
           b_hi: float, b_lo: float) -> tuple[float, float]:
    """
    Return the dd product ``(a_hi + a_lo) * (b_hi + b_lo)``.

    The ``a_lo * b_lo`` term is below the dd epsilon and is dropped,
    following QD.
    """
    p_hi, p_lo = _two_prod(a_hi, b_hi)
    p_lo += a_hi * b_lo + a_lo * b_hi
    return _quick_two_sum(p_hi, p_lo)


@numba.njit(cache=True, fastmath=False)
def dd_div(a_hi: float, a_lo: float,
           b_hi: float, b_lo: float) -> tuple[float, float]:
    """
    Return the dd quotient ``(a_hi + a_lo) / (b_hi + b_lo)``.

    QD accurate division: three Newton-style correction terms, each
    obtained from a float64 quotient of the running dd remainder.

    Raises
    ------
    ZeroDivisionError
        If ``b_hi`` is zero. This propagates from the underlying
        float64 division rather than being raised explicitly, and is
        the behaviour under both the interpreter and `numba.njit` with
        its default ``error_model='python'``. Callers that instead want
        an inf/nan to propagate must guard the divisor themselves.
    """
    q_1 = a_hi / b_hi

    # r = a - q_1 * b
    p_hi, p_lo = dd_mul(b_hi, b_lo, q_1, 0.0)
    r_hi, r_lo = dd_sub(a_hi, a_lo, p_hi, p_lo)

    q_2 = r_hi / b_hi

    # r -= q_2 * b
    p_hi, p_lo = dd_mul(b_hi, b_lo, q_2, 0.0)
    r_hi, r_lo = dd_sub(r_hi, r_lo, p_hi, p_lo)

    q_3 = r_hi / b_hi

    q_1, q_2 = _quick_two_sum(q_1, q_2)
    return dd_add(q_1, q_2, q_3, 0.0)


@numba.njit(cache=True, fastmath=False)
def dd_complex_from_complex128(
        z: complex) -> tuple[float, float, float, float]:
    """Return the dd-complex representation of a complex128, exactly."""
    return z.real, 0.0, z.imag, 0.0


@numba.njit(cache=True, fastmath=False)
def dd_complex_to_complex128(re_hi: float, re_lo: float,
                             im_hi: float, im_lo: float) -> complex:
    """Return the nearest complex128 to a dd-complex number."""
    return complex(re_hi + re_lo, im_hi + im_lo)


@numba.njit(cache=True, fastmath=False)
def dd_complex_add(
        a_re_hi: float, a_re_lo: float, a_im_hi: float, a_im_lo: float,
        b_re_hi: float, b_re_lo: float, b_im_hi: float, b_im_lo: float
) -> tuple[float, float, float, float]:
    """Return the dd-complex sum ``a + b``, componentwise."""
    re_hi, re_lo = dd_add(a_re_hi, a_re_lo, b_re_hi, b_re_lo)
    im_hi, im_lo = dd_add(a_im_hi, a_im_lo, b_im_hi, b_im_lo)
    return re_hi, re_lo, im_hi, im_lo


@numba.njit(cache=True, fastmath=False)
def dd_complex_sub(
        a_re_hi: float, a_re_lo: float, a_im_hi: float, a_im_lo: float,
        b_re_hi: float, b_re_lo: float, b_im_hi: float, b_im_lo: float
) -> tuple[float, float, float, float]:
    """Return the dd-complex difference ``a - b``, componentwise."""
    re_hi, re_lo = dd_sub(a_re_hi, a_re_lo, b_re_hi, b_re_lo)
    im_hi, im_lo = dd_sub(a_im_hi, a_im_lo, b_im_hi, b_im_lo)
    return re_hi, re_lo, im_hi, im_lo


@numba.njit(cache=True, fastmath=False)
def dd_complex_mul(
        a_re_hi: float, a_re_lo: float, a_im_hi: float, a_im_lo: float,
        b_re_hi: float, b_re_lo: float, b_im_hi: float, b_im_lo: float
) -> tuple[float, float, float, float]:
    """
    Return the dd-complex product ``a * b``.

    Schoolbook four-multiply form: ``(ac - bd) + i(ad + bc)``. The
    three-multiply Karatsuba variant is not used -- it trades a
    multiply for extra additions and loses accuracy when ``ac`` and
    ``bd`` cancel, which is exactly the regime this module exists for.
    """
    ac_hi, ac_lo = dd_mul(a_re_hi, a_re_lo, b_re_hi, b_re_lo)
    bd_hi, bd_lo = dd_mul(a_im_hi, a_im_lo, b_im_hi, b_im_lo)
    ad_hi, ad_lo = dd_mul(a_re_hi, a_re_lo, b_im_hi, b_im_lo)
    bc_hi, bc_lo = dd_mul(a_im_hi, a_im_lo, b_re_hi, b_re_lo)

    re_hi, re_lo = dd_sub(ac_hi, ac_lo, bd_hi, bd_lo)
    im_hi, im_lo = dd_add(ad_hi, ad_lo, bc_hi, bc_lo)
    return re_hi, re_lo, im_hi, im_lo


@numba.njit(cache=True, fastmath=False)
def dd_complex_div(
        a_re_hi: float, a_re_lo: float, a_im_hi: float, a_im_lo: float,
        b_re_hi: float, b_re_lo: float, b_im_hi: float, b_im_lo: float
) -> tuple[float, float, float, float]:
    """
    Return the dd-complex quotient ``a / b``.

    Smith's (1962) scaled algorithm: dividing through by the larger of
    the two denominator components keeps the intermediate ``|b|**2``
    from overflowing or underflowing when the operands sit near the
    float64 rails, which the naive conjugate-multiply form does not.

    Raises
    ------
    ZeroDivisionError
        If ``b`` is zero; see `dd_div`.
    """
    if abs(b_re_hi) >= abs(b_im_hi):
        # ratio = b_im / b_re; den = b_re + b_im * ratio
        ra_hi, ra_lo = dd_div(b_im_hi, b_im_lo, b_re_hi, b_re_lo)
        t_hi, t_lo = dd_mul(b_im_hi, b_im_lo, ra_hi, ra_lo)
        den_hi, den_lo = dd_add(b_re_hi, b_re_lo, t_hi, t_lo)

        # re = (a_re + a_im * ratio) / den
        t_hi, t_lo = dd_mul(a_im_hi, a_im_lo, ra_hi, ra_lo)
        n_hi, n_lo = dd_add(a_re_hi, a_re_lo, t_hi, t_lo)
        re_hi, re_lo = dd_div(n_hi, n_lo, den_hi, den_lo)

        # im = (a_im - a_re * ratio) / den
        t_hi, t_lo = dd_mul(a_re_hi, a_re_lo, ra_hi, ra_lo)
        n_hi, n_lo = dd_sub(a_im_hi, a_im_lo, t_hi, t_lo)
        im_hi, im_lo = dd_div(n_hi, n_lo, den_hi, den_lo)
        return re_hi, re_lo, im_hi, im_lo

    # ratio = b_re / b_im; den = b_re * ratio + b_im
    ra_hi, ra_lo = dd_div(b_re_hi, b_re_lo, b_im_hi, b_im_lo)
    t_hi, t_lo = dd_mul(b_re_hi, b_re_lo, ra_hi, ra_lo)
    den_hi, den_lo = dd_add(t_hi, t_lo, b_im_hi, b_im_lo)

    # re = (a_re * ratio + a_im) / den
    t_hi, t_lo = dd_mul(a_re_hi, a_re_lo, ra_hi, ra_lo)
    n_hi, n_lo = dd_add(t_hi, t_lo, a_im_hi, a_im_lo)
    re_hi, re_lo = dd_div(n_hi, n_lo, den_hi, den_lo)

    # im = (a_im * ratio - a_re) / den
    t_hi, t_lo = dd_mul(a_im_hi, a_im_lo, ra_hi, ra_lo)
    n_hi, n_lo = dd_sub(t_hi, t_lo, a_re_hi, a_re_lo)
    im_hi, im_lo = dd_div(n_hi, n_lo, den_hi, den_lo)
    return re_hi, re_lo, im_hi, im_lo
