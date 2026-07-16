"""
Tests for `lensing.chang_refsdal._hyp1f1`, the point-mass Chang--Refsdal
kernel: the dd-accumulated complex ``1F1`` and its shared-numerator
``s``-derivative ladder.

WHY THESE ORACLES ARE INDEPENDENT
---------------------------------
Every gate here is judged against something the module does NOT derive
for itself, at >= 60 decimal digits in mpmath (kept test-side only; it
is ~1e6x too slow for the likelihood and is never importable from
production):

* the PREFACTOR test evaluates the DEFINITION
  ``|exp(pi*w/4 + i*(w/2)*ln(w/2)) * Gamma(1 - i*w/2)|**2`` directly and
  gates the production closed form ``-pi*w/expm1(-pi*w)`` against it.
  Comparing the closed form to itself at higher precision would be a
  tautology that passes even if the identity were wrong; the flatness
  of the residual in ``w`` is the real content, since a systematic
  disagreement would show as a trend rather than roundoff hash.
* the K-LADDER test uses the reference PROTOTYPE as its oracle -- a
  FRESH ``mp.hyp1f1`` per ``k`` with only the Pochhammer laddered. That
  prototype is the ANTI-TEMPLATE production replaced (too slow for a
  likelihood), so it shares none of production's Kummer reparametrized,
  shared-numerator, double-double derivation.
* the COMPLEXITY test asserts only MEASURED call counts obtained by
  wrapping the ``_dd`` primitives, never a big-O claim taken on faith.

TOLERANCES
----------
The prefactor closed form lands at ~7e-16 against the definition, so
the ``1e-14`` gate is ~14x of roundoff headroom and the flatness slope
(~5e-17 per decade, measured) is gated well below any real trend. For
the ladder the module certifies ``~eps_dd * e**(w*Y)`` relative error:
the ``1e-10`` target holds out to ``w*Y ~ 50`` and degrades to ~1e-6 at
the ceiling ``w*Y = 60``. This suite therefore gates ``1e-10`` only
where the module certifies it (``w*Y <= 50``) and gates the
``eps_dd * e**(w*Y)`` cancellation CONTOUR across the whole certified
domain INCLUDING ``w*Y = 60`` -- asserting the module's stated 1e-6 at
the ceiling, not a 1e-10 it never claims. The mid-range knee band
``12 <= w*Y <= 22`` carries a tight ``1e-11`` gate: a float64
accumulation leak (the defect dd exists to prevent) would climb to
``eps * e**(w*Y) ~ 1e-10`` there, so staying far below it is the
signature that the dd path is intact.

`Hyp1f1TestCase.tearDown` fails a test that made zero comparisons, so a
sweep whose every configuration was skipped cannot read as green, and
`SelfFalsificationTestCase` proves each gate can actually go red.
"""
from __future__ import annotations

import pathlib
import warnings
from unittest import TestCase, main

import mpmath
import numpy as np

from cogwheel.lensing.chang_refsdal import _hyp1f1, geometry
from cogwheel.lensing.chang_refsdal._hyp1f1 import (
    DD_PRODUCT_CEILING, HypergeometricDomainError, W_MAX_CERTIFIED,
    point_mass_g_derivatives, prefactor_c)


try:  # Diagnostics only; never gate a test on plotting being present.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_MPL = False

#: Oracle precision. 70 dps is ~2x the ~32 dd digits, so the mpmath
#: reference is exact relative to anything the kernel can resolve.
ORACLE_DPS = 70
mpmath.mp.dps = ORACLE_DPS

#: Directory for diagnostic figures (shared with the geometry suite).
_OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'

#: Prefactor gate against the independent definition. The closed form
#: lands at ~7e-16, so this is ~14x roundoff headroom.
RTOL_PREFACTOR = 1e-14

#: Number of log-spaced prefactor samples on ``[1e-3, 500]``.
N_PREFACTOR = 40

#: Flatness gate: |slope of signed rel error vs log10(w)| must sit
#: below this. Measured slope is ~5e-17 per decade (pure hash); a real
#: closed-form/definition disagreement would trend orders larger.
PREFACTOR_SLOPE_GATE = 5e-16

#: Double-double machine epsilon, ``2**-106 ~ 1.23e-32`` (~32 digits).
DD_EPS = 2.0 ** -106

#: The ``1e-10`` target and the ``w*Y`` below which the module
#: certifies it. Above it, only the cancellation contour is asserted.
LADDER_RTOL = 1e-10
LADDER_WY_STRICT = 50.0

#: Cancellation-contour envelope ``CONTOUR_FLOOR + SAFETY*eps_dd*e^wY``.
#: FLOOR is the float64 rounding floor of the result and the plain-
#: double Q_k ladder (~k*eps at k=84); SAFETY covers the ``1/|1F1|``
#: inflation of the pessimistic law. Both are measured, not guessed.
CONTOUR_FLOOR = 5e-13
CONTOUR_SAFETY = 100.0

#: Mid-range "no float64 knee" band and its gate. A float64 leak would
#: reach ``eps * e^wY ~ 1e-10`` across this band; the dd path stays
#: ~1e-14, so 1e-11 catches the knee with margin to spare.
KNEE_WY_LO = 12.0
KNEE_WY_HI = 22.0
KNEE_GATE = 1e-11

#: Highest derivative order swept (the caller reaches ``2*max_order``).
MAX_DERIVATIVE = 84

#: Series terms for the ladder sweep -- ample past the ``w*Y = 60``
#: peak so truncation never masquerades as cancellation.
N_TERMS = 400

#: Certified ``(w, s)`` sweep. ``w*Y = w*sqrt(s)`` spans 2 -> 60 and
#: includes the two worst-cancellation corners at the ceiling
#: (``(60, 1)`` and ``(500, 0.0144)``) plus the knee band.
LADDER_SWEEP = ((2.0, 1.0), (5.0, 1.0), (7.0, 4.0), (20.0, 0.49),
                (10.0, 4.0), (300.0, 0.01), (40.0, 1.0),
                (100.0, 0.25), (500.0, 0.01), (60.0, 1.0),
                (500.0, 0.0144))

#: Derivative orders swept by the complexity test (spans > a decade).
COMPLEXITY_ORDERS = (4, 8, 16, 32, 64, 84)

#: Fixed ``(w, s)`` and term count for the complexity sweep.
COMPLEXITY_W = 3.0
COMPLEXITY_S = 2.0
COMPLEXITY_TERMS = 60

#: Reject-the-quadratic gates: the linear fit's max relative residual
#: and the curvature metric ``|c2| * max_order / c1``. Counts are
#: exactly linear, so both land near float roundoff.
COMPLEXITY_LINEAR_RESID_TOL = 1e-9
COMPLEXITY_CURVATURE_TOL = 1e-6


def _oracle_abs_c_squared(w):
    """
    Return ``|C(w)|**2`` from the DEFINITION at `ORACLE_DPS`,
    INDEPENDENT of production's closed form: the literal
    ``|exp(pi*w/4 + i*(w/2)*ln(w/2)) * Gamma(1 - i*w/2)|**2``.
    """
    wm = mpmath.mpf(w)
    carrier = mpmath.exp(mpmath.pi * wm / 4
                         + 1j * (wm / 2) * mpmath.log(wm / 2))
    return abs(carrier * mpmath.gamma(1 - 1j * wm / 2)) ** 2


def _oracle_ladder(w, s, max_derivative):
    """
    Return the reference-PROTOTYPE ladder ``values[k]`` at `ORACLE_DPS`.

    ``values[k] = C(w) * base**k * (a)_k / k! * hyp1f1(a + k, 1 + k, z)``
    with ``a = 1 - i*w/2``, ``base = -i*w/2``, ``z = -i*w*s/2``, and a
    FRESH ``mp.hyp1f1`` at every ``k`` -- the slow prototype production
    replaced, so it shares none of production's derivation.
    """
    wm = mpmath.mpf(w)
    sm = mpmath.mpf(s)
    a = 1 - 1j * wm / 2
    base = -1j * wm / 2
    z = -1j * wm * sm / 2
    carrier = (mpmath.exp(mpmath.pi * wm / 4
                          + 1j * (wm / 2) * mpmath.log(wm / 2))
               * mpmath.gamma(1 - 1j * wm / 2))
    values = []
    for k in range(max_derivative + 1):
        term = base ** k * mpmath.rf(a, k) / mpmath.factorial(k)
        values.append(carrier * term * mpmath.hyp1f1(a + k, 1 + k, z))
    return values


def _savefig(fig, name):
    """Save a diagnostic figure, swallowing any backend error."""
    if not _HAVE_MPL:
        return
    try:
        _OUTPUT_DIR.mkdir(exist_ok=True)
        fig.savefig(_OUTPUT_DIR / name, dpi=80, bbox_inches='tight')
    except Exception:  # pragma: no cover - environment dependent
        pass
    finally:
        plt.close(fig)


class Hyp1f1TestCase(TestCase):
    """
    Base class carrying the mpmath comparison and the anti-vacuity
    tally.

    `assert_close` is the domain assertion: it bumps `n_checks` and
    gates a relative error against an mpmath oracle. `tearDown` fails a
    test that asserted nothing, so a fully-skipped sweep cannot read as
    green.
    """

    _expect_checks = True

    def setUp(self):
        self.n_checks = 0

    def tearDown(self):
        if self._expect_checks and self.n_checks == 0:
            self.fail('vacuous test: no comparison ran, so nothing was '
                      'asserted')

    def assert_close(self, got, exact, tol, msg=''):
        """
        Assert the value `got` matches the mpmath `exact` to relative
        `tol`; bump `n_checks` and return the float relative error.
        """
        got_mp = (mpmath.mpc(got) if isinstance(got, complex)
                  else mpmath.mpf(got))
        rel = abs(got_mp - exact) / abs(exact)
        self.n_checks += 1
        self.assertLessEqual(
            rel, mpmath.mpf(tol),
            f'{msg}: relative error {mpmath.nstr(rel, 5)} > {tol}')
        return float(rel)


class PrefactorTestCase(Hyp1f1TestCase):
    """
    `prefactor_c` against an INDEPENDENT evaluation of the definition.
    """

    def test_closed_form_matches_definition_and_is_flat(self):
        """
        ``|C(w)|**2`` from the closed form ``-pi*w/expm1(-pi*w)`` tracks
        the definition to `RTOL_PREFACTOR`, and the signed residual is
        FLAT in ``w`` -- a fitted slope consistent with zero. The
        flatness is the real content: a trend would mean the closed
        form and the definition disagree systematically, not by
        roundoff.
        """
        ws = np.geomspace(1e-3, 500.0, N_PREFACTOR)
        signed = []
        for w in ws:
            got = abs(prefactor_c(w)) ** 2
            exact = _oracle_abs_c_squared(w)
            signed.append(float((mpmath.mpf(got) - exact) / exact))
            self.assert_close(got, exact, RTOL_PREFACTOR,
                              f'|C(w={w:.4g})|^2')
        signed = np.array(signed)
        slope = np.polyfit(np.log10(ws), signed, 1)[0]
        self.assertLessEqual(
            abs(slope), PREFACTOR_SLOPE_GATE,
            f'signed residual trends with w (slope {slope:.2e} per '
            f'decade > {PREFACTOR_SLOPE_GATE}); the closed form and the '
            'definition disagree systematically')
        self._plot(ws, signed, slope)

    def test_analytic_limits(self):
        """
        ``|C|**2 -> 1`` as ``w -> 0`` (approach ``1 + pi*w/2``, so the
        deviation is bounded by ``pi*w``) and ``|C|**2 -> pi*w`` as
        ``w -> infinity`` (the correction is ``e**(-pi*w)``, so at
        ``w >= 50`` the ratio is 1 to full double precision).
        """
        for w in (1e-4, 1e-6, 1e-8):
            deviation = abs(abs(prefactor_c(w)) ** 2 - 1.0)
            self.assertLessEqual(
                deviation, np.pi * w,
                f'|C|^2 does not approach 1 as w->0 at w={w}')
            self.n_checks += 1
        for w in (50.0, 100.0, 500.0):
            ratio = abs(prefactor_c(w)) ** 2 / (np.pi * w)
            self.assertLessEqual(
                abs(ratio - 1.0), 1e-12,
                f'|C|^2 does not approach pi*w as w->inf at w={w}')
            self.n_checks += 1

    def _plot(self, ws, signed, slope):
        if not _HAVE_MPL:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogx(ws, np.abs(signed) + 1e-18, 'o', label='|rel err|')
        ax.plot(ws, np.full_like(ws, RTOL_PREFACTOR), 'k--',
                label='1e-14 gate')
        ax.set_yscale('log')
        ax.set_xlabel('w')
        ax.set_ylabel('|rel err| of |C|^2')
        ax.set_title(f'prefactor vs definition (slope {slope:.1e}/dec)')
        ax.legend()
        _savefig(fig, 'hyp1f1_prefactor.png')


class LadderTestCase(Hyp1f1TestCase):
    """
    The k-ladder against the reference-prototype mpmath oracle.
    """

    def test_matches_oracle_and_follows_cancellation_law(self):
        """
        For each certified ``(w, s)`` and all ``k <= MAX_DERIVATIVE``:
        the production ladder matches the fresh-``hyp1f1`` oracle. The
        error stays below the ``eps_dd * e**(w*Y)`` contour everywhere
        including ``w*Y = 60`` (worst cancellation), meets ``1e-10``
        where the module certifies it (``w*Y <= 50``), and shows NO
        float64 knee in the ``12 <= w*Y <= 22`` band.
        """
        curve = []
        for w, s in LADDER_SWEEP:
            with self.subTest(w=w, s=s):
                w_y = w * np.sqrt(s)
                values, _ = point_mass_g_derivatives(
                    w, s, MAX_DERIVATIVE, N_TERMS)
                exact = _oracle_ladder(w, s, MAX_DERIVATIVE)
                worst = 0.0
                for k in range(MAX_DERIVATIVE + 1):
                    if abs(exact[k]) == 0:
                        continue
                    rel = float(abs(mpmath.mpc(values[k]) - exact[k])
                                / abs(exact[k]))
                    worst = max(worst, rel)
                    self.n_checks += 1
                contour = (CONTOUR_FLOOR
                           + CONTOUR_SAFETY * DD_EPS * np.exp(w_y))
                self.assertLessEqual(
                    worst, contour,
                    f'w*Y={w_y:.2f}: rel err {worst:.3e} exceeds the '
                    f'cancellation contour {contour:.3e}')
                if w_y <= LADDER_WY_STRICT:
                    self.assertLessEqual(
                        worst, LADDER_RTOL,
                        f'w*Y={w_y:.2f}: rel err {worst:.3e} exceeds '
                        f'the certified 1e-10 target')
                if KNEE_WY_LO <= w_y <= KNEE_WY_HI:
                    self.assertLessEqual(
                        worst, KNEE_GATE,
                        f'w*Y={w_y:.2f} sits in the knee band and rel '
                        f'err {worst:.3e} > {KNEE_GATE}: a float64 '
                        'accumulation leak is climbing the eps*e^wY '
                        'contour where the dd path should be flat')
                curve.append((w_y, worst))
        self._plot(curve)

    def _plot(self, curve):
        if not _HAVE_MPL or not curve:
            return
        curve = sorted(curve)
        w_y = np.array([c[0] for c in curve])
        err = np.array([c[1] for c in curve])
        grid = np.linspace(w_y.min(), w_y.max(), 200)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(w_y, err + 1e-32, 'o', label='rel err')
        ax.plot(grid, DD_EPS * np.exp(grid), 'C2-',
                label='eps_dd e^{wY}')
        ax.plot(grid, np.finfo(float).eps * np.exp(grid), 'C3--',
                label='eps_f64 e^{wY} (knee)')
        ax.axhline(LADDER_RTOL, color='k', ls=':', label='1e-10')
        ax.set_ylim(1e-18, 1.0)
        ax.set_xlabel('w * Y')
        ax.set_ylabel('rel err')
        ax.legend()
        _savefig(fig, 'hyp1f1_ladder_contour.png')


class LadderComplexityTestCase(Hyp1f1TestCase):
    """
    Substantiated ladder-cost claims, from MEASURED ``_dd`` call counts.
    """

    def test_shared_numerator_constant_dd_multiplies_linear(self):
        """
        Wrapping the ``_dd`` primitives: (a) the shared-numerator
        evaluation count (``dd_complex_mul`` calls, one per ``P_n``) is
        INDEPENDENT of ``max_derivative`` -- if it scaled, the numerator
        would be recomputed per ``k`` and the design rationale would be
        void; (b) the total dd-multiply count (``dd_mul``) is LINEAR in
        ``max_derivative`` -- the quadratic coefficient is consistent
        with zero and the linear fit leaves no curvature.
        """
        orig_mul = _hyp1f1.dd_mul
        orig_cmul = _hyp1f1.dd_complex_mul
        orders, mul_counts, cmul_counts = [], [], []
        try:
            for order in COMPLEXITY_ORDERS:
                tally = {'mul': 0, 'cmul': 0}

                def counting_mul(*args, _t=tally):
                    _t['mul'] += 1
                    return orig_mul(*args)

                def counting_cmul(*args, _t=tally):
                    _t['cmul'] += 1
                    return orig_cmul(*args)

                _hyp1f1.dd_mul = counting_mul
                _hyp1f1.dd_complex_mul = counting_cmul
                point_mass_g_derivatives(
                    COMPLEXITY_W, COMPLEXITY_S, order, COMPLEXITY_TERMS)
                orders.append(order)
                mul_counts.append(tally['mul'])
                cmul_counts.append(tally['cmul'])
                self.n_checks += 1
        finally:
            _hyp1f1.dd_mul = orig_mul
            _hyp1f1.dd_complex_mul = orig_cmul

        # (a) shared-numerator work does not grow with max_derivative.
        self.assertEqual(
            len(set(cmul_counts)), 1,
            f'shared-numerator P_n count varies with max_derivative: '
            f'{dict(zip(orders, cmul_counts))} -- the numerator is '
            'being recomputed per k')
        self.assertEqual(
            cmul_counts[0], COMPLEXITY_TERMS - 1,
            'shared-numerator evaluated a wrong number of P_n')

        # (b) total dd-multiplies are linear: fit both, reject quadratic.
        orders_a = np.array(orders, dtype=float)
        muls = np.array(mul_counts, dtype=float)
        linear = np.polyfit(orders_a, muls, 1)
        residual = np.max(np.abs(muls - np.polyval(linear, orders_a)))
        max_resid = residual / np.max(muls)
        quad_c2 = np.polyfit(orders_a, muls, 2)[0]
        curvature = abs(quad_c2) * np.max(orders_a) / linear[0]
        self.assertGreater(linear[0], 0.0,
                           'dd-multiply count does not grow with order')
        self.assertLess(
            max_resid, COMPLEXITY_LINEAR_RESID_TOL,
            f'linear fit leaves curvature (max rel residual '
            f'{max_resid:.2e}): the dd-multiply count is not linear')
        self.assertLess(
            curvature, COMPLEXITY_CURVATURE_TOL,
            f'quadratic coefficient is not negligible (curvature metric '
            f'{curvature:.2e}): a quadratic model is not rejected')
        self._plot(orders_a, muls, cmul_counts, linear)

    def _plot(self, orders, muls, cmul_counts, linear):
        if not _HAVE_MPL:
            return
        grid = np.linspace(orders.min(), orders.max(), 100)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(orders, muls, 'o', label='dd_mul (total)')
        ax.plot(grid, np.polyval(linear, grid), 'C0-', label='linear')
        quad = np.polyfit(orders, muls, 2)
        ax.plot(grid, np.polyval(quad, grid), 'C3--',
                label='quadratic (rejected)')
        ax.plot(orders, cmul_counts, 's',
                label='dd_complex_mul (shared, const)')
        ax.set_xlabel('max_derivative')
        ax.set_ylabel('call count')
        ax.legend()
        _savefig(fig, 'hyp1f1_complexity.png')


class DomainCeilingTestCase(Hyp1f1TestCase):
    """
    Certified-domain ceilings raise the NAMED
    `HypergeometricDomainError`.
    """

    def _assert_accepts(self, w, s):
        """The kernel returns finite values with no warning."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            values, _ = point_mass_g_derivatives(w, s, 4, 200)
        self.assertTrue(np.all(np.isfinite(values)),
                        f'non-finite values inside the domain at '
                        f'(w, s) = ({w}, {s})')
        self.n_checks += 1

    def _assert_rejects(self, w, s, *tokens):
        """
        The kernel raises `HypergeometricDomainError` -- the named
        error, asserted by exact type: NOT a bare ``ValueError`` and NOT
        ``geometry.LensDomainError`` (which would mean the numeric
        primitive wrongly depends on the physics layer). The message
        must name each token.
        """
        with self.assertRaises(HypergeometricDomainError) as ctx:
            point_mass_g_derivatives(w, s, 4, 200)
        exc = ctx.exception
        self.assertIs(type(exc), HypergeometricDomainError,
                      'raised something other than the named error')
        self.assertNotIsInstance(exc, geometry.LensDomainError,
                                 'leaked the physics-layer error type')
        message = str(exc)
        for token in tokens:
            self.assertIn(token, message,
                          f'message does not name {token!r}: {message}')
        self.n_checks += 1

    def test_frequency_ceiling(self):
        """
        ``w = W_MAX_CERTIFIED`` succeeds (boundary not off-by-one);
        ``w`` just above it and ``w <= 0`` raise, naming ``w`` and the
        ceiling.
        """
        self._assert_accepts(W_MAX_CERTIFIED, 0.0)
        self._assert_accepts(np.nextafter(W_MAX_CERTIFIED, 0.0), 0.0)
        above = np.nextafter(W_MAX_CERTIFIED, np.inf)
        self._assert_rejects(above, 1.0, str(above),
                             str(W_MAX_CERTIFIED))
        self._assert_rejects(600.0, 1.0, str(W_MAX_CERTIFIED))

    def test_dd_product_ceiling(self):
        """
        ``w*sqrt(s) = DD_PRODUCT_CEILING`` succeeds exactly (boundary
        not off-by-one); just above it raises, naming the ceiling.
        Checked at both worst-cancellation corners.
        """
        self._assert_accepts(60.0, 1.0)           # product == 60.0
        self._assert_accepts(500.0, 0.0144)       # product == 60.0
        self._assert_accepts(60.0, 0.9996)        # product < 60.0
        self._assert_rejects(60.0, 1.01, str(DD_PRODUCT_CEILING))
        self._assert_rejects(500.0, 0.015, str(DD_PRODUCT_CEILING))

    def test_invalid_arguments(self):
        """``w <= 0`` and ``s < 0`` raise the named error."""
        self._assert_rejects(0.0, 1.0)
        self._assert_rejects(-1.0, 1.0)
        self._assert_rejects(10.0, -0.1)

    def test_prefactor_shares_the_frequency_ceiling(self):
        """
        `prefactor_c` enforces the same named ceiling: it succeeds at
        ``w = W_MAX_CERTIFIED`` and raises just above and at ``w <= 0``.
        """
        self.assertTrue(np.isfinite(prefactor_c(W_MAX_CERTIFIED)))
        self.n_checks += 1
        for bad in (np.nextafter(W_MAX_CERTIFIED, np.inf), 0.0, -3.0):
            with self.assertRaises(HypergeometricDomainError) as ctx:
                prefactor_c(bad)
            self.assertIs(type(ctx.exception),
                          HypergeometricDomainError)
            self.n_checks += 1

    def test_domain_map_diagnostic(self):
        """Diagnostic only: a ``(w, w*sqrt(s))`` raised/returned map."""
        if not _HAVE_MPL:
            self.n_checks += 1  # keep tearDown satisfied
            return
        raised, returned = [], []
        for w in np.linspace(50.0, 560.0, 24):
            for s in np.linspace(0.0, 0.02, 24):
                point = (w, w * np.sqrt(s))
                try:
                    point_mass_g_derivatives(w, s, 2, 80)
                    returned.append(point)
                except HypergeometricDomainError:
                    raised.append(point)
        self.n_checks += 1
        fig, ax = plt.subplots(figsize=(6, 4))
        for data, color, label in ((returned, 'C0', 'returned'),
                                    (raised, 'C3', 'raised')):
            if data:
                arr = np.array(data)
                ax.scatter(arr[:, 0], arr[:, 1], s=8, c=color,
                           label=label)
        ax.axhline(DD_PRODUCT_CEILING, color='k', ls='--')
        ax.axvline(W_MAX_CERTIFIED, color='k', ls='--')
        ax.set_xlabel('w')
        ax.set_ylabel('w * sqrt(s)')
        ax.legend()
        _savefig(fig, 'hyp1f1_domain_map.png')


class SelfFalsificationTestCase(Hyp1f1TestCase):
    """
    Prove the gates above can actually fail.

    A double-double / special-function bug is silent -- it degrades
    accuracy without raising -- so a green suite is worth only as much
    as its ability to go red. Each method shows a gate catching a
    deliberately corrupted input.
    """

    _expect_checks = False

    def test_flatness_gate_detects_a_trend(self):
        """A signed residual with an injected ``w``-trend fails the
        flatness slope gate; roundoff hash alone would not."""
        ws = np.geomspace(1e-3, 500.0, N_PREFACTOR)
        rng = np.random.default_rng(0)
        residual = 1e-13 * np.log10(ws) + 1e-16 * rng.standard_normal(
            ws.size)
        slope = np.polyfit(np.log10(ws), residual, 1)[0]
        self.assertGreater(
            abs(slope), PREFACTOR_SLOPE_GATE,
            'the flatness gate would not discriminate a real trend')

    def test_ladder_gate_detects_a_perturbation(self):
        """A 1e-8 perturbation of a ladder value breaks the 1e-10
        gate, so the gate is not vacuous."""
        values, _ = point_mass_g_derivatives(10.0, 1.0, 4, 200)
        exact = _oracle_ladder(10.0, 1.0, 4)
        perturbed = values[0] * (1.0 + 1e-8)
        rel = float(abs(mpmath.mpc(perturbed) - exact[0])
                    / abs(exact[0]))
        self.assertGreater(
            rel, LADDER_RTOL,
            'the 1e-10 ladder gate would not catch a 1e-8 error')

    def test_curvature_gate_rejects_quadratic_counts(self):
        """Purely quadratic synthetic counts trip the curvature metric,
        so the reject-the-quadratic logic can fire."""
        orders = np.array(COMPLEXITY_ORDERS, dtype=float)
        counts = orders ** 2
        linear = np.polyfit(orders, counts, 1)
        quad_c2 = np.polyfit(orders, counts, 2)[0]
        curvature = abs(quad_c2) * np.max(orders) / linear[0]
        max_resid = np.max(np.abs(
            counts - np.polyval(linear, orders))) / np.max(counts)
        self.assertGreater(
            curvature, COMPLEXITY_CURVATURE_TOL,
            'the curvature metric would not reject a quadratic count')
        self.assertGreater(
            max_resid, COMPLEXITY_LINEAR_RESID_TOL,
            'the linear-residual gate would not reject a quadratic')

    def test_domain_gate_rejects_beyond_ceiling(self):
        """Just past each ceiling the named error is raised, so the
        boundary gate is not vacuous."""
        for w, s in ((np.nextafter(W_MAX_CERTIFIED, np.inf), 0.0),
                     (60.0, 1.01)):
            with self.assertRaises(HypergeometricDomainError):
                point_mass_g_derivatives(w, s, 2, 80)


if __name__ == '__main__':
    main()
