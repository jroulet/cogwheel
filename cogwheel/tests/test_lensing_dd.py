"""
Tests for the `lensing.chang_refsdal._dd` double-double primitives.

This is domain test T0. Every primitive is compared DIRECTLY against an
mpmath oracle at 60 decimal digits -- not against a reference
implementation, and not only against float64, both of which would hide
the failure mode this module exists to prevent. A double-double bug is
SILENT: it degrades the operator's accuracy without raising, and would
be misdiagnosed downstream as a 1F1 or series bug.

The adversarial inputs are chosen to break the error-free
transformations if they are wrong:
  - catastrophic cancellation, ``a - b`` with ``b = a * (1 + delta)``
    down to ``delta ~ 1e-33``, where the answer lives entirely in the
    low words;
  - products spanning many binary exponents, which exercise the
    Veltkamp split's scaled branch;
  - operands near the float64 overflow and underflow rails;
  - full-53-bit significands (all-ones, alternating, power-of-two
    boundaries). Uniform random draws are NOT sufficient: they leave a
    broken splitter looking healthy. See `_adversarial_significands`.

The accuracy bound is 1e-30, ~80x above the dd epsilon of 1.2e-32, and
is asserted only where mpmath says the exact answer is representable as
a dd number (see `_DD_MIN_NORMAL` / `_MAX_FLOAT`). `DdTestCase.tearDown`
guards against a sweep that skips every comparison and so asserts
nothing.

`SplitterSensitivityTestCase` closes the loop the rest of the suite
cannot: it corrupts the splitting constant and asserts the invariants
above go red, so that "the suite is green" is evidence rather than
decoration.
"""

import itertools
import struct
from unittest import TestCase, main, mock

import mpmath
import numpy as np

from cogwheel.lensing.chang_refsdal import _dd


mpmath.mp.dps = 60

#: Most significant bits either half of a Veltkamp split may carry if
#: every cross product in `_dd._two_prod` is to be exact in float64.
_MAX_HALF_BITS = 26

#: Below this magnitude a dd number's low word is subnormal, so the
#: format cannot deliver its nominal ~32 digits. Equal to 2**-969.
_DD_MIN_NORMAL = mpmath.mpf(2) ** -969

#: Largest finite float64; above this an exact answer is not
#: representable and the comparison is skipped.
_MAX_FLOAT = mpmath.mpf(np.finfo(np.float64).max)

#: Accuracy required of every primitive on representable answers.
TOL = mpmath.mpf('1e-30')

#: Binary exponents used to span the float64 range in sweeps.
EXPONENTS = [-300, -100, -20, -1, 0, 1, 20, 100, 300]


def _dd_to_mpf(hi, lo):
    """Return the exact value of a dd number as an mpmath mpf."""
    return mpmath.mpf(hi) + mpmath.mpf(lo)


def _dd_to_mpc(re_hi, re_lo, im_hi, im_lo):
    """Return the exact value of a dd-complex number as an mpmath mpc."""
    return mpmath.mpc(_dd_to_mpf(re_hi, re_lo), _dd_to_mpf(im_hi, im_lo))


def _is_representable(exact):
    """
    Return whether `exact` (mpf or mpc) is a value a dd number can hold
    at full precision: nonzero, above the subnormal-low-word floor, and
    below the float64 overflow rail.
    """
    magnitude = abs(exact)
    return _DD_MIN_NORMAL < magnitude < _MAX_FLOAT


def _py(func):
    """
    Return the pure-Python body of `func`.

    `_dd`'s primitives are written ``@njit``-shaped with the decorator
    deferred. If they are later decorated, numba freezes module globals
    at compile time and `_dd._SPLITTER` stops being patchable;
    resolving `py_func` keeps `SplitterSensitivityTestCase` meaningful
    either way. Note that this reaches only one level: once `_two_prod`
    itself is jitted, its call to `_split` binds the compiled version,
    and the mutation test would need `_dd` reloaded under the patch.
    """
    return getattr(func, 'py_func', func)


def _significand_bits(value):
    """
    Return the number of significant bits in `value`'s significand.

    Uses exact integer arithmetic on the IEEE-754 fields rather than
    `numpy.log2`, whose float exponent arithmetic silently underflows
    to zero for the low words of tiny operands and would make the
    width check vacuously true.
    """
    if value == 0. or not np.isfinite(value):
        return 0
    significand, _ = np.frexp(abs(value))
    # `significand` is in [0.5, 1); scaling by 2**53 makes it integral.
    integral = int(significand * 2.**53)
    if integral == 0:
        return 0
    # Trailing zeros are not significant.
    return 53 - (integral & -integral).bit_length() + 1


def _adversarial_significands():
    """
    Return floats whose significands stress a Veltkamp split.

    Uniform random draws are NOT adequate here: a splitter that leaves
    27 bits in a half only produces an inexact `_two_prod` for specific
    bit patterns, so a thin uniform sample reports a broken splitter as
    healthy (that is exactly what this suite used to do). These are
    all-ones, alternating, and power-of-two-boundary patterns, plus a
    dense sweep of full 53-bit significands.
    """
    # Trailing nibbles are chosen so every pattern's low bit is set,
    # i.e. each is the full 53 bits wide (hence ...b, not ...a, for the
    # alternating pattern).
    values = [float.fromhex(h) for h in (
        '0x1.fffffffffffffp0', '0x1.5555555555555p0',
        '0x1.aaaaaaaaaaaabp0', '0x1.0000000000001p0',
        '0x1.7ffffffffffffp0', '0x1.8000000000001p0')]
    rng = np.random.default_rng(1234)
    for _ in range(500):
        # Randomize every significand bit, not just the leading ones,
        # and force the trailing bit so each value is the full 53 bits
        # wide -- a trailing zero would leave one bit of slack that a
        # too-wide split half could hide in.
        bits = int(rng.integers(0, 1 << 52)) | 1
        values.append(
            struct.unpack('d', struct.pack('Q', (1023 << 52) | bits))[0])
    return values


def _random_dd(rng, exponent):
    """
    Return a normalized random dd number ``(hi, lo)`` with magnitude of
    order ``2**exponent`` and a low word ~2**-53 below the high word.
    """
    hi = rng.uniform(-1., 1.) * 2.**exponent
    lo = rng.uniform(-1., 1.) * hi * 2.**-53
    return _dd.dd_add(hi, 0., lo, 0.)


def _random_dd_complex(rng, exponent):
    """Return a random dd-complex number as four float64 scalars."""
    return _random_dd(rng, exponent) + _random_dd(rng, exponent)


def _scaled_dd(a_hi, a_lo, factor):
    """Return the dd number ``(a_hi + a_lo) * factor``."""
    return _dd.dd_mul(a_hi, a_lo, factor, 0.)


class DdTestCase(TestCase):
    """Base class providing the mpmath comparison."""

    def setUp(self):
        """Reset the per-test comparison tally used by `tearDown`."""
        self.n_compared = 0
        self.n_skipped = 0

    def tearDown(self):
        """
        Fail a test whose every comparison was skipped.

        `assert_dd_close` silently returns for values a dd number
        cannot hold, so a sweep whose inputs all fell outside the
        representable range would otherwise pass without asserting
        anything. Tests that never call the helper are unaffected.
        """
        if self.n_skipped and not self.n_compared:
            self.fail(f'all {self.n_skipped} comparisons were skipped as '
                      'unrepresentable; the test asserted nothing')

    def assert_dd_close(self, words, exact, msg=''):
        """
        Assert that the dd number `words` matches the mpmath `exact`
        value to within `TOL` relative error.

        Skips the assertion when `exact` is not representable as a dd
        number, per the module docstring. `words` is a 2-tuple for a dd
        real or a 4-tuple for a dd complex.
        """
        if not _is_representable(exact):
            self.n_skipped += 1
            return
        self.n_compared += 1

        if len(words) == 2:
            got = _dd_to_mpf(*words)
        else:
            got = _dd_to_mpc(*words)

        rel_error = abs(got - exact) / abs(exact)
        self.assertLessEqual(
            rel_error, TOL,
            f'{msg}: relative error {mpmath.nstr(rel_error, 5)} > {TOL}\n'
            f'  got   {mpmath.nstr(got, 40)}\n'
            f'  exact {mpmath.nstr(exact, 40)}')


class ErrorFreeTransformTestCase(DdTestCase):
    """
    Test that the Dekker transformations are exactly error-free.

    These underpin everything else, so they are checked for EXACT
    equality in mpmath, not to a tolerance.
    """

    def test_two_sum_is_exact(self):
        """`_two_sum` residual reproduces ``a + b`` with no error."""
        rng = np.random.default_rng(0)
        for exp_a, exp_b in itertools.product(EXPONENTS, repeat=2):
            a = rng.uniform(-1., 1.) * 2.**exp_a
            b = rng.uniform(-1., 1.) * 2.**exp_b
            s, err = _dd._two_sum(a, b)
            self.assertEqual(_dd_to_mpf(s, err),
                             mpmath.mpf(a) + mpmath.mpf(b),
                             f'_two_sum({a}, {b}) is not error-free')

    def test_two_prod_is_exact(self):
        """`_two_prod` residual reproduces ``a * b`` with no error."""
        rng = np.random.default_rng(1)
        for exp_a, exp_b in itertools.product(EXPONENTS, repeat=2):
            a = rng.uniform(-1., 1.) * 2.**exp_a
            b = rng.uniform(-1., 1.) * 2.**exp_b
            p, err = _dd._two_prod(a, b)
            exact = mpmath.mpf(a) * mpmath.mpf(b)
            if not _is_representable(exact):
                continue
            self.assertEqual(_dd_to_mpf(p, err), exact,
                             f'_two_prod({a}, {b}) is not error-free')

    def test_two_prod_is_exact_on_adversarial_significands(self):
        """
        `_two_prod` is error-free for full-53-bit significands.

        `test_two_prod_is_exact` draws uniformly and therefore passes
        even against a splitter that leaves 27 bits in a half; these
        inputs are what actually discriminate. See
        `_adversarial_significands` and `SplitterSensitivityTestCase`.
        """
        values = _adversarial_significands()
        for a, b in itertools.product(values[:6], values):
            p, err = _dd._two_prod(a, b)
            self.assertEqual(_dd_to_mpf(p, err),
                             mpmath.mpf(a) * mpmath.mpf(b),
                             f'_two_prod({a.hex()}, {b.hex()}) is not '
                             'error-free')

    def test_split_is_exact_and_halves_are_narrow(self):
        """
        `_split` reconstructs its input exactly and yields halves of at
        most 26 significant bits, including for adversarial
        significands and on the scaled branch used near the overflow
        rail.

        The width bound is the property `_two_prod` rests on: 26 bits
        per half means every cross product fits float64's 53-bit
        significand exactly.
        """
        rng = np.random.default_rng(2)
        values = [rng.uniform(-1., 1.) * 2.**exp for exp in EXPONENTS]
        values += [1.7e308, -1.7e308, 1e300, 0., 1., -1.]
        values += _adversarial_significands()
        values += [value * 2.**300 for value in _adversarial_significands()]
        for value in values:
            hi, lo = _dd._split(value)
            self.assertEqual(_dd_to_mpf(hi, lo), mpmath.mpf(value),
                             f'_split({value.hex()}) does not reconstruct')
            for name, half in (('hi', hi), ('lo', lo)):
                self.assertLessEqual(
                    _significand_bits(half), _MAX_HALF_BITS,
                    f'_split({value.hex()}) {name} word {half.hex()} '
                    f'carries more than {_MAX_HALF_BITS} bits')

    def test_split_does_not_overflow_near_rail(self):
        """
        The scaled branch keeps `_two_prod` finite for operands above
        2**996, where a naive splitter would overflow to inf.
        """
        p, err = _dd._two_prod(1.5e300, 2.5e-300)
        self.assertTrue(np.isfinite(p) and np.isfinite(err))
        self.assert_dd_close(
            (p, err), mpmath.mpf(1.5e300) * mpmath.mpf(2.5e-300),
            'product across the overflow rail')


class DdAddTestCase(DdTestCase):
    """Test `dd_add` and `dd_sub`."""

    def test_random_operands(self):
        """`dd_add` matches mpmath across the float64 exponent range."""
        rng = np.random.default_rng(3)
        for exp_a, exp_b in itertools.product(EXPONENTS, repeat=2):
            a_hi, a_lo = _random_dd(rng, exp_a)
            b_hi, b_lo = _random_dd(rng, exp_b)
            got = _dd.dd_add(a_hi, a_lo, b_hi, b_lo)
            exact = _dd_to_mpf(a_hi, a_lo) + _dd_to_mpf(b_hi, b_lo)
            self.assert_dd_close(got, exact, f'dd_add 2^{exp_a}+2^{exp_b}')

    def test_catastrophic_cancellation(self):
        """
        `dd_sub` of nearly-equal operands stays accurate RELATIVE TO THE
        DIFFERENCE, down to a relative separation of 1e-33 where the
        answer lives entirely in the low words. A sloppy (single-low-
        word) add fails this; float64 fails it by construction.
        """
        rng = np.random.default_rng(4)
        deltas = [1e-1, 1e-5, 1e-10, 1e-16, 1e-20, 1e-25, 1e-30, 1e-33]
        for exp, delta in itertools.product([-100, 0, 100], deltas):
            a_hi, a_lo = _random_dd(rng, exp)
            b_hi, b_lo = _scaled_dd(a_hi, a_lo, 1. + delta)
            got = _dd.dd_sub(a_hi, a_lo, b_hi, b_lo)
            exact = _dd_to_mpf(a_hi, a_lo) - _dd_to_mpf(b_hi, b_lo)
            self.assert_dd_close(
                got, exact, f'dd_sub cancelling at delta={delta}')

    def test_beats_float64_under_cancellation(self):
        """
        The dd path is decisively more accurate than float64 on a
        cancelling sum -- a guard that the low words carry information
        rather than being decorative.
        """
        a_hi, a_lo = _dd.dd_add(1., 0., 1e-25, 0.)
        got = _dd.dd_sub(a_hi, a_lo, 1., 0.)
        exact = _dd_to_mpf(a_hi, a_lo) - mpmath.mpf(1.)

        dd_error = abs(_dd_to_mpf(*got) - exact) / abs(exact)
        float64_error = abs(mpmath.mpf((a_hi + a_lo) - 1.) - exact) \
            / abs(exact)
        self.assertLessEqual(dd_error, TOL)
        self.assertGreater(float64_error, 1e-10)

    def test_identity_and_negation(self):
        """Adding zero is exact; negation round-trips to zero."""
        rng = np.random.default_rng(5)
        for exp in EXPONENTS:
            a_hi, a_lo = _random_dd(rng, exp)
            self.assertEqual(_dd.dd_add(a_hi, a_lo, 0., 0.), (a_hi, a_lo))
            self.assertEqual(_dd.dd_sub(a_hi, a_lo, a_hi, a_lo), (0., 0.))


class DdMulTestCase(DdTestCase):
    """Test `dd_mul`."""

    def test_random_operands(self):
        """`dd_mul` matches mpmath across many exponent combinations."""
        rng = np.random.default_rng(6)
        for exp_a, exp_b in itertools.product(EXPONENTS, repeat=2):
            a_hi, a_lo = _random_dd(rng, exp_a)
            b_hi, b_lo = _random_dd(rng, exp_b)
            got = _dd.dd_mul(a_hi, a_lo, b_hi, b_lo)
            exact = _dd_to_mpf(a_hi, a_lo) * _dd_to_mpf(b_hi, b_lo)
            self.assert_dd_close(got, exact, f'dd_mul 2^{exp_a}*2^{exp_b}')

    def test_spans_extreme_exponents(self):
        """
        Products of a huge and a tiny operand -- the case that drives
        the split onto its scaled branch -- stay accurate.
        """
        pairs = [(1.5e300, 2.5e-300), (1e-300, 1e300), (1.7e308, 1e-308),
                 (6.7e299, 1.), (-1.5e300, 3.3e-299)]
        for a_hi, b_hi in pairs:
            got = _dd.dd_mul(a_hi, 0., b_hi, 0.)
            exact = mpmath.mpf(a_hi) * mpmath.mpf(b_hi)
            self.assert_dd_close(got, exact, f'dd_mul({a_hi}, {b_hi})')

    def test_squaring_is_accurate(self):
        """Squaring reproduces mpmath's square (a hot-path pattern)."""
        rng = np.random.default_rng(7)
        for exp in EXPONENTS:
            a_hi, a_lo = _random_dd(rng, exp)
            got = _dd.dd_mul(a_hi, a_lo, a_hi, a_lo)
            self.assert_dd_close(got, _dd_to_mpf(a_hi, a_lo) ** 2,
                                 f'dd_mul square at 2^{exp}')


class DdDivTestCase(DdTestCase):
    """Test `dd_div`."""

    def test_random_operands(self):
        """`dd_div` matches mpmath across many exponent combinations."""
        rng = np.random.default_rng(8)
        for exp_a, exp_b in itertools.product(EXPONENTS, repeat=2):
            a_hi, a_lo = _random_dd(rng, exp_a)
            b_hi, b_lo = _random_dd(rng, exp_b)
            got = _dd.dd_div(a_hi, a_lo, b_hi, b_lo)
            exact = _dd_to_mpf(a_hi, a_lo) / _dd_to_mpf(b_hi, b_lo)
            self.assert_dd_close(got, exact, f'dd_div 2^{exp_a}/2^{exp_b}')

    def test_inverse_round_trip(self):
        """``a / b * b`` recovers ``a`` to dd accuracy."""
        rng = np.random.default_rng(9)
        for exp_a, exp_b in itertools.product([-100, 0, 100], repeat=2):
            a_hi, a_lo = _random_dd(rng, exp_a)
            b_hi, b_lo = _random_dd(rng, exp_b)
            q_hi, q_lo = _dd.dd_div(a_hi, a_lo, b_hi, b_lo)
            got = _dd.dd_mul(q_hi, q_lo, b_hi, b_lo)
            self.assert_dd_close(got, _dd_to_mpf(a_hi, a_lo),
                                 'dd_div round trip')

    def test_reciprocal_of_three(self):
        """
        1/3 is not representable in binary; the dd quotient must still
        agree with mpmath's 1/3 to dd precision.
        """
        got = _dd.dd_div(1., 0., 3., 0.)
        self.assert_dd_close(got, mpmath.mpf(1) / 3, 'dd_div 1/3')

    def test_raises_on_zero_divisor(self):
        """
        Dividing by a dd zero raises rather than propagating an
        inf/nan. Pinned because the operator series must guard its own
        divisors; see `dd_div`'s docstring.
        """
        with self.assertRaises(ZeroDivisionError):
            _dd.dd_div(1., 0., 0., 0.)


class DdComplexTestCase(DdTestCase):
    """Test the dd-complex operations."""

    def test_add_matches_mpmath(self):
        """`dd_complex_add` matches mpmath componentwise."""
        rng = np.random.default_rng(10)
        for exp_a, exp_b in itertools.product(EXPONENTS, repeat=2):
            a = _random_dd_complex(rng, exp_a)
            b = _random_dd_complex(rng, exp_b)
            got = _dd.dd_complex_add(*a, *b)
            exact = _dd_to_mpc(*a) + _dd_to_mpc(*b)
            self.assert_dd_close(got, exact, 'dd_complex_add')

    def test_sub_matches_mpmath(self):
        """`dd_complex_sub` matches mpmath componentwise."""
        rng = np.random.default_rng(11)
        for exp in EXPONENTS:
            a = _random_dd_complex(rng, exp)
            b = _random_dd_complex(rng, exp)
            got = _dd.dd_complex_sub(*a, *b)
            self.assert_dd_close(got, _dd_to_mpc(*a) - _dd_to_mpc(*b),
                                 'dd_complex_sub')

    def test_mul_matches_mpmath(self):
        """`dd_complex_mul` matches mpmath across exponents."""
        rng = np.random.default_rng(12)
        for exp_a, exp_b in itertools.product(EXPONENTS, repeat=2):
            a = _random_dd_complex(rng, exp_a)
            b = _random_dd_complex(rng, exp_b)
            got = _dd.dd_complex_mul(*a, *b)
            exact = _dd_to_mpc(*a) * _dd_to_mpc(*b)
            self.assert_dd_close(got, exact, 'dd_complex_mul')

    def test_mul_by_unit_phase_preserves_modulus(self):
        """
        Multiplying by ``i`` -- the operator series' recurring factor --
        is exact, and by a generic unit-modulus number is accurate.
        """
        rng = np.random.default_rng(13)
        for exp in EXPONENTS:
            a = _random_dd_complex(rng, exp)
            got = _dd.dd_complex_mul(*a, 0., 0., 1., 0.)
            self.assert_dd_close(got, _dd_to_mpc(*a) * mpmath.mpc(0, 1),
                                 'dd_complex_mul by i')

    def test_div_matches_mpmath(self):
        """`dd_complex_div` matches mpmath across exponents."""
        rng = np.random.default_rng(14)
        for exp_a, exp_b in itertools.product(EXPONENTS, repeat=2):
            a = _random_dd_complex(rng, exp_a)
            b = _random_dd_complex(rng, exp_b)
            got = _dd.dd_complex_div(*a, *b)
            exact = _dd_to_mpc(*a) / _dd_to_mpc(*b)
            self.assert_dd_close(got, exact, 'dd_complex_div')

    def test_div_exercises_both_smith_branches(self):
        """
        Smith's algorithm branches on ``|re| >= |im|``; both branches
        must be accurate, including when the components themselves span
        extreme exponents.
        """
        cases = [(3., 1e-300), (1e-300, 3.), (1e300, 1.), (1., 1e300),
                 (1e200, 1e-200), (1e-200, 1e200)]
        for b_re, b_im in cases:
            got = _dd.dd_complex_div(1.5, 0., -2.5, 0., b_re, 0., b_im, 0.)
            exact = mpmath.mpc(1.5, -2.5) / mpmath.mpc(b_re, b_im)
            self.assert_dd_close(got, exact,
                                 f'dd_complex_div by ({b_re}, {b_im})')

    def test_div_does_not_overflow_on_large_denominator(self):
        """
        The naive conjugate-multiply form squares the denominator and
        overflows here; Smith's scaled form must stay finite AND exact.

        Numerator and denominator are both huge so that the quotient is
        of order one and therefore fully representable -- a small
        numerator would push the answer below `_DD_MIN_NORMAL` and get
        the accuracy assertion skipped, testing nothing.
        """
        got = _dd.dd_complex_div(3e300, 0., -1e300, 0., 1e300, 0., 2e300, 0.)
        self.assertTrue(all(np.isfinite(word) for word in got))
        exact = mpmath.mpc(3e300, -1e300) / mpmath.mpc(1e300, 2e300)
        self.assert_dd_close(got, exact, 'dd_complex_div, huge operands')

    def test_div_stays_finite_below_the_dd_normal_floor(self):
        """
        A quotient beneath `_DD_MIN_NORMAL` cannot carry dd precision,
        so this asserts only what is true there: Smith's form returns a
        finite result of the right magnitude instead of overflowing the
        intermediate ``|b|**2`` to inf and collapsing to zero or nan.
        """
        got = _dd.dd_complex_div(1., 0., 1., 0., 1e300, 0., 1e300, 0.)
        self.assertTrue(all(np.isfinite(word) for word in got))
        exact = mpmath.mpc(1, 1) / mpmath.mpc(1e300, 1e300)
        rel_error = abs(_dd_to_mpc(*got) - exact) / abs(exact)
        # float64 accuracy only: the low words are subnormal here.
        self.assertLessEqual(rel_error, mpmath.mpf('1e-15'))

    def test_div_round_trip(self):
        """``a / b * b`` recovers the dd-complex ``a``."""
        rng = np.random.default_rng(15)
        for exp in EXPONENTS:
            a = _random_dd_complex(rng, exp)
            b = _random_dd_complex(rng, 0)
            quotient = _dd.dd_complex_div(*a, *b)
            got = _dd.dd_complex_mul(*quotient, *b)
            self.assert_dd_close(got, _dd_to_mpc(*a),
                                 'dd_complex_div round trip')

    def test_div_raises_on_zero_divisor(self):
        """Dividing by a dd-complex zero raises; see `dd_complex_div`."""
        with self.assertRaises(ZeroDivisionError):
            _dd.dd_complex_div(1., 0., 1., 0., 0., 0., 0., 0.)


class SignificandBitsTestCase(TestCase):
    """
    Test the `_significand_bits` helper.

    It is load-bearing test infrastructure: the split-width invariant
    and the whole of `SplitterSensitivityTestCase` are built on it, and
    a version that under-reported (returning 0, say) would make them
    pass vacuously.
    """

    def test_known_widths(self):
        """Widths match hand-computed values, subnormals included."""
        cases = [(1., 1), (2., 1), (0.5, 1), (3., 2), (-3., 2), (5., 3),
                 (7., 3), (8., 1), (0., 0),
                 (float.fromhex('0x1.fffffffffffffp0'), 53),
                 (float.fromhex('0x1.5555555555555p0'), 53),
                 (1. + 2.**-52, 53),
                 (2.**-1074, 1)]  # smallest subnormal
        for value, expected in cases:
            self.assertEqual(_significand_bits(value), expected,
                             f'_significand_bits({value!r})')

    def test_adversarial_significands_are_full_width(self):
        """
        The adversarial inputs really do carry 53 bits -- otherwise
        they would not stress the split at all.
        """
        widths = {_significand_bits(value)
                  for value in _adversarial_significands()}
        self.assertEqual(widths, {53})


class SplitterSensitivityTestCase(TestCase):
    """
    Prove that this suite is able to FAIL.

    A double-double bug is silent -- it degrades accuracy without
    raising -- so a green suite is worth only as much as its ability to
    go red. These tests corrupt the splitting constant and assert that
    the invariants above actually catch it.

    Not every perturbation is a bug, and the distinction matters: a
    splitter of 2**27 or 2**27 + 2 still yields 26-bit halves and is a
    functionally valid splitter, so the suite staying green for those
    is correct rather than a gap. What must be caught is a splitter
    that moves the split POINT, leaving a half 27 bits wide.
    """

    #: Splitters that move the split point and silently destroy the
    #: error-free product.
    BROKEN_SPLITTERS = (67108865.0,   # 2**26 + 1: 27-bit high word
                        268435457.0)  # 2**28 + 1: 27-bit low word

    #: Splitters that keep both halves at 26 bits and remain correct.
    VALID_SPLITTERS = (134217729.0,   # 2**27 + 1, the canonical choice
                       134217728.0,   # 2**27
                       134217730.0)   # 2**27 + 2

    @staticmethod
    def _widest_half(splitter):
        """Return the most significand bits either half carries."""
        with mock.patch.object(_dd, '_SPLITTER', splitter):
            split = _py(_dd._split)
            return max(max(_significand_bits(half) for half in split(value))
                       for value in _adversarial_significands())

    def test_canonical_splitter_is_in_use(self):
        """The shipped constant is Dekker's 2**27 + 1."""
        self.assertEqual(_dd._SPLITTER, 2.**27 + 1)

    def test_broken_splitter_widens_a_half(self):
        """A moved split point pushes a half past 26 bits."""
        for splitter in self.BROKEN_SPLITTERS:
            self.assertGreater(
                self._widest_half(splitter), _MAX_HALF_BITS,
                f'splitter {splitter} left both halves narrow, so the '
                'width invariant would not detect it')

    def test_broken_splitter_breaks_two_prod(self):
        """
        A moved split point destroys `_two_prod`'s error-free residual
        on adversarial significands -- the silent corruption that would
        otherwise surface downstream as a phantom 1F1 or series bug.
        """
        values = _adversarial_significands()
        for splitter in self.BROKEN_SPLITTERS:
            # Patch `_split` to its pure-Python body too: `_two_prod`'s
            # py_func resolves `_split` from module globals, and the njit
            # dispatcher froze `_SPLITTER` at compile time -- without this
            # the patched splitter never reaches the split and the
            # falsification is vacuous.
            with mock.patch.object(_dd, '_SPLITTER', splitter), \
                    mock.patch.object(_dd, '_split', _py(_dd._split)):
                two_prod = _py(_dd._two_prod)
                inexact = [
                    (a, b)
                    for a, b in itertools.product(values[:6], values)
                    if _dd_to_mpf(*two_prod(a, b))
                    != mpmath.mpf(a) * mpmath.mpf(b)]
            self.assertTrue(
                inexact, f'splitter {splitter} kept _two_prod exact, so '
                'the exactness test cannot detect a broken split')

    def test_valid_splitters_keep_halves_narrow(self):
        """
        Splitters that preserve the split point keep both halves at 26
        bits, documenting why the suite is right not to flag them.
        """
        for splitter in self.VALID_SPLITTERS:
            self.assertLessEqual(self._widest_half(splitter),
                                 _MAX_HALF_BITS,
                                 f'splitter {splitter} widened a half')


class ConversionTestCase(DdTestCase):
    """Test conversion to and from the plain float64/complex128 types."""

    def test_float_round_trip(self):
        """`dd_from_float` then `dd_to_float` is the identity."""
        rng = np.random.default_rng(16)
        values = [rng.uniform(-1., 1.) * 2.**exp for exp in EXPONENTS]
        for value in values + [0., 1., -1., np.inf]:
            self.assertEqual(_dd.dd_to_float(*_dd.dd_from_float(value)),
                             value)

    def test_from_float_is_exact(self):
        """`dd_from_float` introduces no low word."""
        self.assertEqual(_dd.dd_from_float(0.1), (0.1, 0.))

    def test_complex_round_trip(self):
        """The complex128 conversions are mutual inverses."""
        rng = np.random.default_rng(17)
        for exp in EXPONENTS:
            value = complex(rng.uniform(-1., 1.) * 2.**exp,
                            rng.uniform(-1., 1.) * 2.**exp)
            words = _dd.dd_complex_from_complex128(value)
            self.assertEqual(_dd.dd_complex_to_complex128(*words), value)

    def test_to_complex128_folds_low_words(self):
        """
        `dd_complex_to_complex128` returns the nearest complex128,
        i.e. it adds the low word rather than discarding it.
        """
        words = _dd.dd_add(1., 0., 1e-20, 0.) + _dd.dd_add(2., 0., 1e-20,
                                                           0.)
        got = _dd.dd_complex_to_complex128(*words)
        self.assertEqual(got, complex(1. + 1e-20, 2. + 1e-20))


class NumbaCompatibilityTestCase(TestCase):
    """
    Guard the representation contract the module's numba-compatibility
    rests on: flat float64 scalars in, flat float64 scalars out.
    """

    def test_returns_are_flat_float_tuples(self):
        """
        Every dd operation returns a flat tuple of plain floats -- no
        nesting, no numpy scalars, no objects -- so that the intended
        `numba.njit` decorator can infer a homogeneous UniTuple.
        """
        real_ops = [_dd.dd_add, _dd.dd_sub, _dd.dd_mul, _dd.dd_div]
        for func in real_ops:
            got = func(3., 1e-20, 7., 1e-20)
            self.assertEqual(len(got), 2, func.__name__)
            for word in got:
                self.assertIs(type(word), float, func.__name__)

        complex_ops = [_dd.dd_complex_add, _dd.dd_complex_sub,
                       _dd.dd_complex_mul, _dd.dd_complex_div]
        for func in complex_ops:
            got = func(3., 1e-20, 5., 1e-20, 7., 1e-20, 11., 1e-20)
            self.assertEqual(len(got), 4, func.__name__)
            for word in got:
                self.assertIs(type(word), float, func.__name__)


if __name__ == '__main__':
    main()
