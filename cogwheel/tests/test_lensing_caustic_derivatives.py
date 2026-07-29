"""
Tests for the analytic caustic derivatives in
``lensing.chang_refsdal.geometry`` (WP1 / finding F038).

The Chang--Refsdal caustic is an exact closed-form parametric curve, so
``geometry.caustic_derivatives`` returns the *symbolic* first and second
theta-derivatives of it, not a finite-difference or sampled-arc estimate.
The failure mode this suite exists to catch is therefore a WRONG closed
form -- a transcribed sign, a dropped chain-rule term, or a normalization
factor -- which is silent: it produces a smooth, finite curve that is
simply the derivative of the wrong thing.

The oracle is INDEPENDENT of the code under test.  It reconstructs only
the curve DEFINITION -- ``u(theta)``, ``r(theta)``, ``p_i(theta)`` and
``y_i = p_i r T_i`` -- which is the shared specification of *what the
caustic is* (Professor ruling 7: reusing the definition is required, not
circular), and then differentiates it NUMERICALLY with ``mpmath.diff`` at
40 decimal digits.  It never touches the module's ``u_p``, ``u_pp``,
``r_p``, ``r_pp`` cascade, and never imports ``caustic_derivatives`` /
``caustic_curvature_radius``; ``OracleIndependenceTestCase`` enforces that
by AST inspection.  A high-precision numerical derivative of the shared
definition is thus a genuinely independent check of the module's analytic
derivative.

Tolerances.  The comparison is MIXED, ``|value - expected| <= atol +
rtol * |expected|`` with ``atol = 5e-13`` and ``rtol = 1e-11`` per
component.  A pure relative test would false-fail because near-axial
``theta = 0.02`` and the saddle branch ``-1`` push individual y-components
through zero; a pure absolute test would false-fail near the parity wall
``gamma = 0.99`` where the second derivative and the curvature radius
(``R_c ~ 1145``) are large and the rtol term must dominate.  ``atol`` sits
just above the driver's measured float64 residual (~4.4e-13; this suite
measures worst d2 ~1.4e-12 absolute, covered by rtol on the large values,
and worst d1 ~1.6e-13 absolute).

The astroid-limit gate is a SCALE-AND-SIGN pin, not a 1e-12 gate: at
``gamma = 1e-3`` the curvature radius must approach ``3 gamma |sin 2
theta|`` to within 3e-3 (Professor ruling 4 -- the admitted O(gamma^2)
correction, do NOT tighten).  It pins the leading coefficient 3, the power
of gamma, and the sign; a factor-2 or factor-1.5 convention error fails it.

``_CausticDerivativeTestCase.tearDown`` fails if a sweep skipped every
comparison, so a green run is evidence and not an artefact of an empty
loop.  ``SelfFalsificationTestCase`` corrupts the analytic value and the
astroid normalization and asserts the gates go red, proving they have
teeth.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import pathlib
import warnings
from unittest import TestCase, main, mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpmath
import numpy as np

from cogwheel.lensing.chang_refsdal import geometry


mpmath.mp.dps = 40

#: Absolute floor of the mixed tolerance (dimensionless).  Just above the
#: measured float64 residual of the analytic cascade; lets near-zero
#: y-components pass without a pure-relative false failure.
ATOL = 5e-13

#: Relative term of the mixed tolerance.  Dominates for the large second
#: derivatives and the ``R_c ~ 1145`` radii near the parity wall.
RTOL = 1e-11

#: Shear magnitudes spanning positive parity (|gamma| < 1 - kappa),
#: the parity approach (gamma = 0.99), and the macro saddle
#: (|gamma| > 1 - kappa).  The F038 case set.
GAMMAS = (0.05, 0.3, 0.9, 0.99, 1.02, 1.3)

#: Both square-root branches of ``u(theta)``.  ``+1`` is the only real
#: branch at positive parity; the saddle uses both.
BRANCHES = (1, -1)

#: External convergences: vacuum (lam = 1) and a convergent screen
#: (lam = 0.7), which turns gamma in {0.9, 0.99} into macro saddles.
KAPPAS = (0.0, 0.3)

#: Polar angles, radians: near-axial, mid-arc, and past ``pi``.  None is
#: an exact astroid cusp (0, pi/2, pi, 3pi/2), so every one is a valid
#: comparison point for the derivative gate.
THETAS = (0.02, 0.17, 0.5, 1.0, 1.3, 2.2, 3.9)

#: Oracle speed below which a point is treated as a cusp (|y'| -> 0),
#: where the curvature radius 0/0 is ill-conditioned and excluded from
#: the curvature gate.  No grid point reaches this (min |y'| ~ 6e-3).
CUSP_SPEED_FLOOR = 1e-9

#: Astroid-limit shear: small enough that the geometry is well
#: conditioned and the O(gamma^2) correction is tiny (ruling 4).
ASTROID_GAMMA = 1e-3

#: Mid-arc angles for the astroid pin (away from its cusps).
ASTROID_THETAS = (0.5, 1.0, 1.3)

#: Fractional tolerance on the astroid scale-and-sign pin.
ASTROID_RTOL = 3e-3

# ---------------------------------------------------------------------------
# Constants for the STAGE-1 curve-definition pin.  The oracle reconstructs
# the SAME parametric curve as the shipping ``critical_point`` in the
# ``y_i = p_i r T_i`` form; ``critical_point`` builds it in the algebraically
# identical ``A x - x / |x|**2`` form (the two agree to ~1e-50 in mpmath, so
# the oracle genuinely pins the shipped curve).  The only gap is float64
# roundoff, which the two-part gate below bounds.
# ---------------------------------------------------------------------------

#: Absolute floor of the STAGE-1 mixed tolerance.  Every real F038 point
#: has ``|oracle - shipped| <= 8.53e-14`` (driver-independent measurement),
#: so this covers the near-axial ``theta = 0.02`` component whose value
#: ``~8.2e-7`` is a difference of larger numbers and loses ~5e-12 in
#: RELATIVE precision to cancellation while its ABSOLUTE error is only
#: ~4.3e-18.  A wrong curve (e.g. the historical ``lam*u`` sign bug) gives
#: an O(value) absolute error there, orders of magnitude above this floor.
CURVE_STAGE1_ATOL = 1e-13

#: Relative term of the STAGE-1 mixed tolerance.  Loose enough that the
#: small-magnitude near-axial components pass on the absolute floor, tight
#: enough that any real curve error at an O(1) component fails.
CURVE_STAGE1_RTOL = 1e-12

#: The spec's headline relative gate.  Asserted on the relative-DOMINATED
#: subset (see ``CURVE_HEADLINE_FLOOR``); measured worst there is 6.52e-14.
CURVE_HEADLINE_RTOL = 1e-13

#: Crossover ``|expected| = ATOL / RTOL`` above which the relative term of
#: the mixed tolerance dominates, so the pure relative error is
#: well-defined (not an artefact of float64 cancellation of a near-zero
#: component).  This is a principled threshold, NOT a value hand-picked to
#: exclude the one failing near-axial point.
CURVE_HEADLINE_FLOOR = CURVE_STAGE1_ATOL / CURVE_STAGE1_RTOL

#: Positive-parity configurations ``(gamma, kappa)`` with ``|gamma| <
#: 1 - kappa``, used for the branch = -1 no-nan / no-warning gate.
POSITIVE_PARITY_CONFIGS = ((0.3, 0.0), (0.05, 0.3))

#: Angles for the branch-invariance sweep, away from the astroid cusps.
POSITIVE_PARITY_THETAS = (0.17, 0.5, 1.0, 1.3, 2.2)

#: Shear / convergence grid for the fold-opening-direction gate.  The
#: combination ``(0.9, 0.3)`` is a macro SADDLE (``|gamma| = 0.9 > 1 -
#: kappa = 0.7``), not a positive-parity astroid, and is filtered out at
#: setup so every tested point is a genuine astroid fold.
FOLD_GAMMAS = (0.3, 0.9)
FOLD_KAPPAS = (0.0, 0.3)

#: Fold-test polar angles, chosen away from the astroid cusps ``theta in
#: {0, pi/2, pi, 3pi/2}`` so the extra merging image pair separates
#: cleanly.  None sits on an axis where the pair would be marginally
#: resolvable.
FOLD_THETAS = (0.35, 0.7, 1.1, 1.9, 2.5, 3.4)

#: Finite source-plane step for the fold-side image count.  At ``1e-3`` the
#: merging pair is comfortably resolved (measured ``n_+ = 4`` vs ``n_- =
#: 2`` at every tested point); it is far above the ``~6e-7`` scale at which
#: F039 saw ``find_images_quartic`` fail to separate a merged pair.
FOLD_EPS = 1e-3

#: Names from the module's derivative cascade that the independent oracle
#: must NOT reference (would make it circular): the public entry points,
#: the ``geometry`` module itself, and the intermediate cascade helpers
#: ``u_p, u_pp, r_p, r_pp`` (and their shared-``p`` variants).  ``y_prime``
#: / ``y_double_prime`` are deliberately NOT forbidden -- they are the
#: oracle's OWN output accumulators (the derivatives it must produce), not
#: symbols reused from the module, so listing them would reject a
#: legitimately independent oracle rather than catch a circular one.
_FORBIDDEN_ORACLE_NAMES = frozenset({
    'caustic_derivatives', 'caustic_speed', 'caustic_curvature_radius',
    'geometry', 'u_p', 'u_pp', 'r_p', 'r_pp',
    'p_shared_p', 'p_shared_pp'})

#: Directory for diagnostic plots.
_OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'


def _oracle_y_component(theta, gamma, kappa, branch, comp):
    """Caustic y-component as an mpmath function of ``theta`` alone.

    Reconstructs ONLY the closed-form curve definition (the shared
    specification): the shear-aligned mass-sheet reduction ``lam = 1 -
    kappa``, effective shear ``eff = gamma / lam``,

        u(theta) = eff cos 2theta + branch sqrt(1 - eff**2 sin**2 2theta),
        r(theta) = 1 / sqrt(lam u),
        p_i      = (lam -+ gamma) - lam u,
        y_i      = p_i r T_i,   T_1 = cos theta, T_2 = sin theta.

    No derivative of u, r or p is used here; differentiation is done
    numerically by :func:`mpmath.diff`.
    """
    lam = mpmath.mpf(1) - mpmath.mpf(kappa)
    eff = mpmath.mpf(gamma) / lam
    two_theta = 2 * theta
    # Mirror critical_point's ``discriminant = max(discriminant, 0.0)``
    # so a slightly-negative saddle discriminant at the wedge edge is
    # clamped to zero rather than turning ``u`` complex.  On the interior
    # of the wedge the clamp is a no-op, so this does not perturb the
    # derivative oracle at any tested point.
    discriminant = 1 - eff**2 * mpmath.sin(two_theta)**2
    if discriminant < 0:
        discriminant = mpmath.mpf(0)
    u = eff * mpmath.cos(two_theta) + branch * mpmath.sqrt(discriminant)
    r = 1 / mpmath.sqrt(lam * u)
    if comp == 0:
        p = (lam - mpmath.mpf(gamma)) - lam * u
        tangent = mpmath.cos(theta)
    else:
        p = (lam + mpmath.mpf(gamma)) - lam * u
        tangent = mpmath.sin(theta)
    return p * r * tangent


def oracle_derivatives(gamma, kappa, branch, theta):
    """Return ``(y', y'')`` at ``theta`` from the independent oracle.

    Each is a two-element list of :class:`mpmath.mpf`, obtained by
    numerically differentiating :func:`_oracle_y_component` at 40 dps.
    """
    theta_mp = mpmath.mpf(theta)
    y_prime, y_double_prime = [], []
    for comp in (0, 1):
        func = lambda th, _c=comp: _oracle_y_component(
            th, gamma, kappa, branch, _c)
        y_prime.append(mpmath.diff(func, theta_mp, 1))
        y_double_prime.append(mpmath.diff(func, theta_mp, 2))
    return y_prime, y_double_prime


def caustic_point_is_real(gamma, kappa, branch, theta):
    """Whether ``theta`` yields a real, positive-radius caustic point.

    Off-wedge saddle angles (negative discriminant) and branches with a
    non-positive ``u`` (imaginary radius) are not on the curve; the
    module refuses or produces ``nan`` there, so they are skipped rather
    than compared.
    """
    lam = 1.0 - kappa
    eff = gamma / lam
    discriminant = 1.0 - eff**2 * np.sin(2.0 * theta)**2
    if discriminant < 0.0:
        return False
    u = eff * np.cos(2.0 * theta) + branch * np.sqrt(max(discriminant, 0.0))
    return u > 0.0


def real_cases():
    """Yield ``(gamma, kappa, branch, theta)`` on the caustic curve."""
    for gamma, kappa, branch, theta in itertools.product(
            GAMMAS, KAPPAS, BRANCHES, THETAS):
        if caustic_point_is_real(gamma, kappa, branch, theta):
            yield gamma, kappa, branch, theta


class _CausticDerivativeTestCase(TestCase):
    """Base carrying the mixed-tolerance assertion and anti-vacuity guard."""

    def setUp(self):
        self._comparisons = 0

    def assert_mixed(self, value, expected, msg):
        """Assert ``|value - expected| <= ATOL + RTOL |expected|``.

        Counts one comparison against the anti-vacuity tally.
        """
        self._comparisons += 1
        tol = ATOL + RTOL * abs(expected)
        error = abs(value - expected)
        self.assertLessEqual(
            error, tol,
            f'{msg}: |{value!r} - {expected!r}| = {error:.3e} > '
            f'{tol:.3e} (atol {ATOL:.0e} + rtol {RTOL:.0e} * '
            f'|expected|)')

    def tearDown(self):
        # Anti-vacuity: a sweep that compared nothing must not read green.
        self.assertGreater(
            self._comparisons, 0,
            'no comparisons ran -- the case sweep was vacuous')


class PrimaryDerivativeTestCase(_CausticDerivativeTestCase):
    """``caustic_derivatives`` matches the mpmath oracle componentwise."""

    def test_first_and_second_derivatives_match_oracle(self):
        # Every real F038 case: both components of y' and y'' against a
        # 40-dps numerical derivative of the shared curve definition.
        for gamma, kappa, branch, theta in real_cases():
            analytic_p, analytic_pp = geometry.caustic_derivatives(
                gamma, theta, kappa=kappa, branch=branch)
            oracle_p, oracle_pp = oracle_derivatives(
                gamma, kappa, branch, theta)
            for comp in (0, 1):
                with self.subTest(gamma=gamma, kappa=kappa, branch=branch,
                                  theta=theta, comp=comp, order=1):
                    self.assert_mixed(
                        float(analytic_p[comp]), float(oracle_p[comp]),
                        f'y_prime[{comp}]')
                with self.subTest(gamma=gamma, kappa=kappa, branch=branch,
                                  theta=theta, comp=comp, order=2):
                    self.assert_mixed(
                        float(analytic_pp[comp]), float(oracle_pp[comp]),
                        f'y_double_prime[{comp}]')

    def test_caustic_speed_delegates_to_norm_of_derivative(self):
        # caustic_speed is a pure delegate: |y'| from the oracle.
        for gamma, kappa, branch, theta in real_cases():
            oracle_p, _ = oracle_derivatives(gamma, kappa, branch, theta)
            expected = float(mpmath.sqrt(oracle_p[0]**2 + oracle_p[1]**2))
            speed = float(geometry.caustic_speed(
                gamma, theta, kappa=kappa, branch=branch))
            with self.subTest(gamma=gamma, kappa=kappa, branch=branch,
                              theta=theta):
                self.assert_mixed(speed, expected, 'caustic_speed')


class CurvatureRadiusTestCase(_CausticDerivativeTestCase):
    """``caustic_curvature_radius`` matches |y'|^3 / |y1' y2'' - y2' y1''|."""

    def test_curvature_radius_matches_oracle(self):
        for gamma, kappa, branch, theta in real_cases():
            oracle_p, oracle_pp = oracle_derivatives(
                gamma, kappa, branch, theta)
            speed = mpmath.sqrt(oracle_p[0]**2 + oracle_p[1]**2)
            cross = oracle_p[0] * oracle_pp[1] - oracle_p[1] * oracle_pp[0]
            # Exclude genuine cusps / inflections where R_c = 0/0.
            if float(speed) < CUSP_SPEED_FLOOR or cross == 0:
                continue
            expected = float(speed**3 / abs(cross))
            radius = float(geometry.caustic_curvature_radius(
                gamma, theta, kappa=kappa, branch=branch))
            with self.subTest(gamma=gamma, kappa=kappa, branch=branch,
                              theta=theta):
                self.assert_mixed(radius, expected, 'R_c')


class AstroidLimitTestCase(_CausticDerivativeTestCase):
    """Small-shear limit pins the leading coefficient, power and sign."""

    def test_astroid_curvature_radius_scale_and_sign(self):
        # R_c -> 3 gamma |sin 2 theta| as gamma -> 0 (positive parity).
        for theta in ASTROID_THETAS:
            radius = float(geometry.caustic_curvature_radius(
                ASTROID_GAMMA, theta, kappa=0.0, branch=1))
            reference = 3.0 * ASTROID_GAMMA * abs(np.sin(2.0 * theta))
            ratio = radius / reference
            self._comparisons += 1
            with self.subTest(theta=theta):
                self.assertLessEqual(
                    abs(ratio - 1.0), ASTROID_RTOL,
                    f'astroid ratio at theta={theta}: {ratio!r} deviates '
                    f'from 1 by more than {ASTROID_RTOL} -- a systematic '
                    f'factor exposes a convention/normalization error')


class OracleIndependenceTestCase(TestCase):
    """Optional hardening: the oracle must not reuse the module's cascade."""

    def test_oracle_source_forbids_cascade_names(self):
        # Walk the AST of the oracle helpers and assert none of the
        # module-under-test's derivative-cascade names appear as a used
        # name or attribute (a source-substring check would false-trip on
        # 'r_p' inside 'branch').
        used = set()
        for func in (_oracle_y_component, oracle_derivatives):
            tree = ast.parse(inspect.getsource(func))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used.add(node.id)
                elif isinstance(node, ast.Attribute):
                    used.add(node.attr)
        leaked = _FORBIDDEN_ORACLE_NAMES & used
        self.assertEqual(
            leaked, set(),
            f'oracle leaked cascade names {leaked}: it would no longer be '
            f'independent of the code under test')

    def test_oracle_does_not_call_module_under_test(self):
        # The oracle uses only mpmath + the curve definition; prove it
        # reproduces the derivatives without ever calling the module, on
        # one representative case.
        gamma, kappa, branch, theta = 0.3, 0.0, 1, 1.0
        with mock.patch.object(
                geometry, 'caustic_derivatives',
                side_effect=AssertionError('oracle called the module!')):
            oracle_p, oracle_pp = oracle_derivatives(
                gamma, kappa, branch, theta)
        self.assertTrue(np.isfinite(float(oracle_p[0])))
        self.assertTrue(np.isfinite(float(oracle_pp[1])))


class SelfFalsificationTestCase(TestCase):
    """Prove the gates can go red: corrupt values / normalization."""

    def test_perturbed_first_derivative_exceeds_tolerance(self):
        # A 1e-6 relative error in y' is far above the mixed tolerance.
        oracle_p, _ = oracle_derivatives(0.3, 0.0, 1, 1.0)
        expected = float(oracle_p[0])
        corrupted = expected * (1.0 + 1e-6)
        tol = ATOL + RTOL * abs(expected)
        self.assertGreater(abs(corrupted - expected), tol,
                           'perturbation must exceed the mixed tolerance')

    def test_patched_module_derivative_fails_primary_gate(self):
        # Swap the analytic derivative for a scaled copy and assert the
        # primary gate detects it (the gate has teeth end to end).
        real = geometry.caustic_derivatives

        def scaled(gamma, theta, *, kappa=0.0, branch=1):
            y_prime, y_double_prime = real(
                gamma, theta, kappa=kappa, branch=branch)
            return y_prime * 1.01, y_double_prime

        case = PrimaryDerivativeTestCase(
            'test_first_and_second_derivatives_match_oracle')
        case.setUp()
        with mock.patch.object(geometry, 'caustic_derivatives', scaled):
            with self.assertRaises(AssertionError):
                case.test_first_and_second_derivatives_match_oracle()

    def test_wrong_astroid_normalization_is_rejected(self):
        # A factor-2 convention error (coefficient 6 instead of 3) must
        # push the astroid ratio outside the 3e-3 pin.
        theta = 1.0
        radius = float(geometry.caustic_curvature_radius(
            ASTROID_GAMMA, theta, kappa=0.0, branch=1))
        wrong_reference = 6.0 * ASTROID_GAMMA * abs(np.sin(2.0 * theta))
        wrong_ratio = radius / wrong_reference
        self.assertGreater(abs(wrong_ratio - 1.0), ASTROID_RTOL,
                           'a factor-2 normalization error must fail the pin')


class DiagnosticPlotTestCase(TestCase):
    """Generate diagnostic plots referenced by the specifications."""

    @classmethod
    def setUpClass(cls):
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_derivative_error_scatter(self):
        # |analytic - oracle| vs theta: a whole-branch offset reads as a
        # systematic convention error; a lone near-zero point as an
        # artefact.
        thetas, errors = [], []
        for gamma, kappa, branch, theta in real_cases():
            analytic_p, _ = geometry.caustic_derivatives(
                gamma, theta, kappa=kappa, branch=branch)
            oracle_p, _ = oracle_derivatives(gamma, kappa, branch, theta)
            for comp in (0, 1):
                thetas.append(theta)
                errors.append(abs(float(analytic_p[comp])
                                  - float(oracle_p[comp])))
        figure, axis = plt.subplots()
        axis.semilogy(thetas, np.maximum(errors, 1e-18), '.')
        axis.axhline(ATOL, color='r', linestyle='--', label='atol')
        axis.set_xlabel('theta [rad]')
        axis.set_ylabel("|analytic - oracle| of y'")
        axis.set_title('caustic first-derivative error')
        axis.legend()
        path = _OUTPUT_DIR / 'caustic_derivatives_error_scatter.png'
        figure.savefig(path)
        plt.close(figure)
        self.assertTrue(path.exists())

    def test_curvature_radius_curve_gamma099(self):
        # R_c(theta) at the parity wall gamma=0.99 must show the large
        # radii, not a clipped/flat curve.
        theta_grid = np.linspace(0.05, np.pi - 0.05, 200)
        radius = geometry.caustic_curvature_radius(
            0.99, theta_grid, kappa=0.0, branch=1)
        figure, axis = plt.subplots()
        axis.semilogy(theta_grid, radius)
        axis.set_xlabel('theta [rad]')
        axis.set_ylabel('R_c')
        axis.set_title('caustic curvature radius, gamma=0.99')
        path = _OUTPUT_DIR / 'caustic_curvature_gamma099.png'
        figure.savefig(path)
        plt.close(figure)
        self.assertTrue(path.exists())
        self.assertGreater(float(np.nanmax(radius)), 100.0)

    def test_astroid_ratio_table(self):
        # ratio vs theta; a flat line near 1 confirms the leading term.
        theta_grid = np.linspace(0.3, np.pi / 2 - 0.3, 50)
        radius = geometry.caustic_curvature_radius(
            ASTROID_GAMMA, theta_grid, kappa=0.0, branch=1)
        reference = 3.0 * ASTROID_GAMMA * np.abs(np.sin(2.0 * theta_grid))
        ratio = radius / reference
        figure, axis = plt.subplots()
        axis.plot(theta_grid, ratio)
        axis.axhline(1.0, color='k', linestyle=':')
        axis.set_xlabel('theta [rad]')
        axis.set_ylabel('R_c / (3 gamma |sin 2theta|)')
        axis.set_title('astroid limit ratio, gamma=1e-3')
        path = _OUTPUT_DIR / 'caustic_astroid_ratio.png'
        figure.savefig(path)
        plt.close(figure)
        self.assertTrue(path.exists())


class CurveDefinitionStageOneTestCase(_CausticDerivativeTestCase):
    """STAGE 1: the oracle curve matches the shipping ``critical_point``.

    The derivative gate (:class:`PrimaryDerivativeTestCase`) only checks
    that the module differentiates *some* curve correctly; it cannot see a
    wrong curve that is differentiated consistently (exactly how the
    historical ``lam*u`` bug hid for a full round).  This gate closes that
    hole: it pins the SHARED curve definition -- the ``y_i = p_i r T_i``
    form the oracle reconstructs -- to ``critical_point(...).source[i]``,
    which builds the algebraically identical ``A x - x / |x|**2`` form.

    Two-part gate, both from a driver-independent float64 measurement over
    the 68 real F038 cases (136 component comparisons):

    * per point, the MIXED tolerance ``|oracle - shipped| <= ATOL + RTOL
      |shipped|`` (measured worst absolute 8.53e-14), which bounds the
      near-axial ``theta = 0.02`` component (relative error 5.29e-12 but
      absolute error only 4.33e-18 -- pure float64 cancellation of a
      ``~8.2e-7`` value, not a curve error);
    * the headline PURE relative error over the relative-dominated subset
      ``|shipped| > ATOL/RTOL`` must be ``<= 1e-13`` (measured worst
      6.52e-14).
    """

    def test_oracle_curve_matches_shipping_critical_point(self):
        # Both source components of every real F038 point, oracle (mpmath
        # cast to float64) vs the shipping critical_point.source.
        worst_relative = 0.0
        worst_relative_case = None
        headline_comparisons = 0
        for gamma, kappa, branch, theta in real_cases():
            point = geometry.critical_point(
                gamma, theta, kappa=kappa, branch=branch)
            for comp in (0, 1):
                shipped = float(point.source[comp])
                oracle = float(_oracle_y_component(
                    mpmath.mpf(theta), gamma, kappa, branch, comp))
                error = abs(oracle - shipped)
                with self.subTest(gamma=gamma, kappa=kappa, branch=branch,
                                  theta=theta, comp=comp):
                    # Mixed tolerance: covers every point, incl. near-axial
                    # cancellation.  Counts against the anti-vacuity tally.
                    self.assert_mixed_curve(oracle, shipped,
                                            f'source[{comp}]')
                if abs(shipped) > CURVE_HEADLINE_FLOOR:
                    headline_comparisons += 1
                    relative = error / abs(shipped)
                    if relative > worst_relative:
                        worst_relative = relative
                        worst_relative_case = (gamma, kappa, branch, theta,
                                                comp, shipped)
        # Anti-vacuity for the headline subset specifically.
        self.assertGreater(
            headline_comparisons, 0,
            'no relative-dominated comparisons -- headline gate vacuous')
        # Spec headline: worst relative error on well-conditioned points.
        print(f'\n[STAGE-1] worst relative error over {headline_comparisons}'
              f' relative-dominated points = {worst_relative:.3e} '
              f'(case {worst_relative_case})')
        self.assertLessEqual(
            worst_relative, CURVE_HEADLINE_RTOL,
            f'worst relative curve error {worst_relative:.3e} exceeds the '
            f'headline gate {CURVE_HEADLINE_RTOL:.0e}; the oracle curve and '
            f'the shipping critical_point no longer agree')

    def assert_mixed_curve(self, value, expected, msg):
        """Mixed-tolerance curve assertion; counts one comparison."""
        self._comparisons += 1
        tol = CURVE_STAGE1_ATOL + CURVE_STAGE1_RTOL * abs(expected)
        error = abs(value - expected)
        self.assertLessEqual(
            error, tol,
            f'{msg}: |{value!r} - {expected!r}| = {error:.3e} > {tol:.3e} '
            f'(atol {CURVE_STAGE1_ATOL:.0e} + rtol {CURVE_STAGE1_RTOL:.0e} '
            f'* |expected|)')

    def test_a_wrong_curve_component_fails_the_gate(self):
        # SELF-FALSIFICATION: a 1e-6 relative corruption of one shipped
        # component is far above both parts of the gate, proving teeth.
        # Use a large, well-conditioned component near the parity wall
        # (|source[1]| ~ 17.7 >> the headline floor) so the relative gate
        # unambiguously applies.
        gamma, kappa, branch, theta = 0.99, 0.0, 1, 1.3
        point = geometry.critical_point(gamma, theta, kappa=kappa,
                                        branch=branch)
        shipped = float(point.source[1])
        self.assertGreater(abs(shipped), CURVE_HEADLINE_FLOOR,
                           'chosen falsification point must be well '
                           'conditioned so a relative gate applies')
        corrupted = shipped * (1.0 + 1e-6)
        with self.assertRaises(AssertionError):
            self.assert_mixed_curve(corrupted, shipped, 'corrupted source')
        # And it also breaks the headline relative gate.
        self.assertGreater(abs(corrupted - shipped) / abs(shipped),
                           CURVE_HEADLINE_RTOL)


class PositiveParityBranchInvarianceTestCase(TestCase):
    """At positive parity ``branch`` is ignored and no ``sqrt`` goes NaN.

    ``critical_point`` uses only the ``+`` root at positive parity
    (``abs(gamma) < 1 - kappa``); the derivative cascade must mirror that
    (Professor Q5).  A first pass shipped ``sqrt(negative) -> nan`` with an
    ``invalid value encountered in sqrt`` RuntimeWarning when called with
    ``branch = -1`` here.  This gate makes that regression loud: with
    RuntimeWarnings promoted to errors, all three public entry points must
    (a) not warn, (b) return only finite values, and (c) give
    bit-for-bit identical output for ``branch = -1`` and ``branch = +1``
    (the code forces the branch flag to ``+1`` internally, so equality is
    exact, not merely within roundoff).
    """

    def setUp(self):
        self._checks = 0

    def tearDown(self):
        self.assertGreater(self._checks, 0,
                           'no positive-parity checks ran -- vacuous sweep')

    def test_branch_minus_one_no_warning_no_nan_matches_plus(self):
        for gamma, kappa in POSITIVE_PARITY_CONFIGS:
            self.assertLess(abs(gamma), 1.0 - kappa,
                            f'({gamma}, {kappa}) must be positive parity')
            for theta in POSITIVE_PARITY_THETAS:
                with self.subTest(gamma=gamma, kappa=kappa, theta=theta):
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', RuntimeWarning)
                        minus = self._evaluate(gamma, theta, kappa, -1)
                        plus = self._evaluate(gamma, theta, kappa, +1)
                    for name, value in minus.items():
                        self.assertTrue(
                            np.all(np.isfinite(value)),
                            f'{name} produced a non-finite value at '
                            f'branch=-1, gamma={gamma}, kappa={kappa}, '
                            f'theta={theta}')
                        np.testing.assert_array_equal(
                            value, plus[name],
                            err_msg=f'{name} differs between branch=-1 and '
                                    f'branch=+1 at positive parity')
                    self._checks += 1

    @staticmethod
    def _evaluate(gamma, theta, kappa, branch):
        """All three public quantities as a name -> array dict."""
        derivatives = geometry.caustic_derivatives(
            gamma, theta, kappa=kappa, branch=branch)
        return {
            'y_prime': np.asarray(derivatives[0]),
            'y_double_prime': np.asarray(derivatives[1]),
            'speed': np.asarray(geometry.caustic_speed(
                gamma, theta, kappa=kappa, branch=branch)),
            'radius': np.asarray(geometry.caustic_curvature_radius(
                gamma, theta, kappa=kappa, branch=branch))}

    def test_runtime_warning_filter_is_armed(self):
        # POSITIVE CONTROL: prove the no-warning gate has teeth, i.e. the
        # promoted RuntimeWarning would actually be caught if the code
        # were to emit an 'invalid value encountered in sqrt'.
        with self.assertRaises(RuntimeWarning):
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                # A genuine invalid-sqrt, the exact regression signature.
                np.sqrt(np.array([-1.0]))
        self._checks += 1


class FoldOpeningDirectionTestCase(TestCase):
    """``fold_opening_direction`` points to the fold's two-image side.

    At a fold caustic point ``y_c`` both signs of a displacement along the
    caustic's soft axis map to the SAME side -- the side carrying the
    extra merging image pair (Professor Q2).  ``fold_opening_direction``
    returns the unit vector toward that side.  This gate confirms the
    convention operationally: stepping ``+FOLD_EPS`` along ``d`` must land
    on the two-image side (more images) and ``-FOLD_EPS`` on the fewer-
    image side.

    Points are the positive-parity astroid folds only: ``(0.9, 0.3)`` has
    ``|gamma| = 0.9 > 1 - kappa = 0.7`` -- a macro SADDLE (deltoid), not an
    astroid -- and is filtered out at setup, along with any angle outside
    a saddle wedge.  ``FOLD_THETAS`` sit away from the astroid cusps so the
    pair separates cleanly (measured ``n_+ = 4`` vs ``n_- = 2``); F039's
    single miss was ``find_images_quartic`` failing to split a merged pair
    at ``eps ~ 6e-7``, three orders below ``FOLD_EPS = 1e-3``.
    """

    @classmethod
    def setUpClass(cls):
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self._points = 0

    def tearDown(self):
        self.assertGreater(self._points, 0,
                           'no fold points evaluated -- vacuous sweep')

    @staticmethod
    def _positive_parity_configs():
        for gamma, kappa in itertools.product(FOLD_GAMMAS, FOLD_KAPPAS):
            if abs(gamma) < 1.0 - kappa:
                yield gamma, kappa

    @staticmethod
    def _image_counts(gamma, kappa, theta, direction):
        """Return ``(n_plus, n_minus)`` across the fold at ``+-eps d``."""
        source = geometry.critical_point(
            gamma, theta, kappa=kappa).source
        matrix = geometry.macro_matrix(gamma, 0.0, kappa)
        n_plus = len(geometry.find_images(
            source + FOLD_EPS * direction, matrix))
        n_minus = len(geometry.find_images(
            source - FOLD_EPS * direction, matrix))
        return n_plus, n_minus

    def test_direction_points_to_two_image_side(self):
        for gamma, kappa in self._positive_parity_configs():
            for theta in FOLD_THETAS:
                direction = geometry.fold_opening_direction(
                    gamma, theta, kappa=kappa)
                with self.subTest(gamma=gamma, kappa=kappa, theta=theta):
                    # Unit vector.
                    norm = float(np.linalg.norm(direction))
                    self.assertLessEqual(
                        abs(norm - 1.0), 1e-12,
                        f'fold direction is not a unit vector: |d| = {norm}')
                    n_plus, n_minus = self._image_counts(
                        gamma, kappa, theta, direction)
                    self.assertGreater(
                        n_plus, n_minus,
                        f'the +d side must carry the extra merging pair: '
                        f'n_+ = {n_plus}, n_- = {n_minus} (a lone tie means '
                        f'the pair did not resolve; a whole-branch flip '
                        f'means the convention is reversed)')
                self._points += 1

    def test_direction_invariant_under_soft_axis_sign_flip(self):
        # The closed form depends on soft_axis only through xe**2 and
        # 4 xe e (both even in e), so flipping the eigenvector's sign must
        # leave the result unchanged (bit-for-bit, measured 0.0).
        real_critical_point = geometry.critical_point

        def sign_flipped(*args, **kwargs):
            point = real_critical_point(*args, **kwargs)
            return point._replace(soft_axis=-point.soft_axis)

        for gamma, kappa in self._positive_parity_configs():
            for theta in FOLD_THETAS:
                baseline = geometry.fold_opening_direction(
                    gamma, theta, kappa=kappa)
                with mock.patch.object(geometry, 'critical_point',
                                       sign_flipped):
                    flipped = geometry.fold_opening_direction(
                        gamma, theta, kappa=kappa)
                with self.subTest(gamma=gamma, kappa=kappa, theta=theta):
                    np.testing.assert_array_equal(
                        baseline, flipped,
                        err_msg='fold direction changed under a soft_axis '
                                'sign flip -- the sign ambiguity leaked')
                self._points += 1

    def test_reversed_convention_would_fail_the_gate(self):
        # SELF-FALSIFICATION: a reversed direction (-d) must land the
        # n_+ > n_- test on the WRONG side, proving the gate detects a
        # whole-branch convention flip.
        gamma, kappa, theta = 0.3, 0.0, 1.1
        direction = geometry.fold_opening_direction(gamma, theta, kappa=kappa)
        n_plus, n_minus = self._image_counts(
            gamma, kappa, theta, -direction)
        self.assertLess(
            n_plus, n_minus,
            'reversing the direction must flip which side carries the '
            'extra pair; if it does not, the gate cannot see a convention '
            'error')
        self._points += 1

    def test_fold_side_image_count_scatter(self):
        # DIAGNOSTIC: (n_+ - n_-) vs theta.  A flat line at +2 confirms the
        # convention; a whole branch at -2 would be a reversed convention;
        # a lone 0 an unresolvable pair.
        thetas, deltas = [], []
        for gamma, kappa in self._positive_parity_configs():
            for theta in FOLD_THETAS:
                direction = geometry.fold_opening_direction(
                    gamma, theta, kappa=kappa)
                n_plus, n_minus = self._image_counts(
                    gamma, kappa, theta, direction)
                thetas.append(theta)
                deltas.append(n_plus - n_minus)
                self._points += 1
        figure, axis = plt.subplots()
        axis.plot(thetas, deltas, 'o')
        axis.axhline(0.0, color='k', linestyle=':')
        axis.set_xlabel('theta [rad]')
        axis.set_ylabel('n_+ - n_-')
        axis.set_title('fold_opening_direction: image-count asymmetry')
        axis.set_ylim(-3, 3)
        path = _OUTPUT_DIR / 'fold_opening_direction_image_counts.png'
        figure.savefig(path)
        plt.close(figure)
        self.assertTrue(path.exists())


if __name__ == '__main__':
    main()
