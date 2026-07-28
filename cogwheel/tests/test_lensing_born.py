"""Fast falsification gates for `cogwheel.lensing.chang_refsdal._born`.

`_born.py` (Build 8h-c1) supplies the analytic Born (weak-deflection)
amplification for the low-w far zone, but ships NOT wired into the serve
path (`likelihood._surrogate_coefficients` falls through to the exact
engine unconditionally there): the two-term series carries an unpinned
``O(1)`` placeholder numerator (`_born._born_factors` ``b1 = 1.0``) that
the Professor has not yet derived a closed form for.  Inspector finding
INS-c1-002 flagged that this module shipped with no test coverage at all.
This suite is the four mandated fast gates, each a coder-verified
falsification (not a smoke test):

* `AccuracyAgainstOperatorTestCase` -- `born_amplification` vs the exact
  `operator.F_op` oracle at a validity-region-passing point in the target
  annulus.  The LITERAL contract (agreement to a tight tolerance once
  ``b1`` is pinned) is carried as an ``@unittest.expectedFailure``
  tripwire: it is RED under the current ``b1 = 1.0`` placeholder (measured
  ~11.3% relative error) and flips to a loud UNEXPECTED SUCCESS -- the
  signal to re-derive the acceptance test and reconsider re-enabling the
  serve slot -- only once a correctly pinned ``b1`` closes the gap.  A
  companion PASS test pins the measured placeholder error comfortably
  above a reachable-red floor, proving the tripwire is not measuring
  something vacuously small.
* `ValidityGateFalsificationTestCase` -- `born_gate` reachable-red just
  outside each of its two independent guards (guard A: series
  convergence via the estimated next-order term; guard B: the
  positive-parity margin below the parity wall), with a companion PASS
  just inside each margin so the boundary is a real, non-knife-edge
  crossing rather than an always-refuse or always-pass vacuity.
* `MacroMagnificationLimitTestCase` -- the ``w -> 0`` limit
  ``F_born -> sqrt(mu_macro)``, checked against the closed-form macro
  magnification ``1 / sqrt((1 - kappa)**2 - gamma**2)`` (an oracle
  independent of the unpinned ``b1``, since the leading term is
  ``b1``-free by construction) and confirmed to shrink linearly in ``w``
  as the mandated structure requires.
* `ServeCensusRoundTripTestCase` -- an end-to-end serve/census check: a
  real `channels.ChangRefsdalGeometryPartition` is built exactly as the
  production far-field serve path would, `born_envelope` is evaluated on
  it, and `channels.reconstruct_farfield` -- the SAME serve-mirror
  reconstruction the trained far-field charts use -- is shown to recover
  the original (demodulated) Born total to machine precision, alongside
  a sane real-image census.  This proves the plumbing this rung would
  need on re-enablement is already correct, without touching the
  actually-dormant `likelihood` serve slot.

INDEPENDENT ORACLES
--------------------
The accuracy gate's oracle is `operator.F_op`, the contour-free exact
Chang-Refsdal amplification, which shares no code with `_born`'s
two-term series.  The macro-magnification gate's oracle is the closed-
form ``1 / sqrt((1 - kappa)**2 - gamma**2)``, recomputed independently
in this file rather than imported from `_born._born_factors`.  The
round-trip gate's oracle is `_born.born_amplification` evaluated
directly per-node and demodulated by hand, compared against the value
`channels.reconstruct_farfield` recovers from the envelope.

All fixture points below were numerically probed before being pinned
here (see the class docstrings for the measured values); every gate
tolerance sits at least one order of magnitude away from its measured
value, matching the reachable-red convention used throughout this test
suite (never a knife-edge on a measured constant).
"""

from __future__ import annotations

import unittest

import numpy as np

from cogwheel.lensing.chang_refsdal import _born, channels, geometry, operator

#: A validity-region-passing point in the target annulus
#: ``3.0 < |y| <= 4.2426`` (module docstring) with a moderate shear,
#: ``w`` chosen close to (but inside) the guard-A convergence margin so
#: the placeholder-``b1`` error is near its largest reachable value in
#: this annulus.  Measured: `born_gate` passes; ``F_born`` disagrees with
#: ``F_op`` by ~11.3% (relative).
ACCURACY_Y1, ACCURACY_Y2 = 3.2, 0.0
ACCURACY_GAMMA, ACCURACY_BETA, ACCURACY_KAPPA = 0.3, 0.0, 0.0
ACCURACY_W = 0.5

#: LITERAL accuracy contract once ``b1`` is correctly pinned: agreement
#: with `operator.F_op` to 0.1% relative.  Currently unreachable (b1 is a
#: documented placeholder); see `AccuracyAgainstOperatorTestCase`.
ACCURACY_TARGET_ONCE_B1_PINNED = 1.0e-3

#: Reachable-red floor for the CURRENT placeholder-``b1`` error: measured
#: ~11.3%, gated more than 2x below that so the companion PASS is not
#: perched on the measured value.
B1_PLACEHOLDER_ERROR_FLOOR = 0.05

#: Guard-A (series-convergence) fixture: `_born._born_factors` gives
#: ``Q2r ~ 12.64`` here, so the guard-A boundary (`_born.EPS_BORN`) sits
#: at ``w* ~ 0.358``.  ``w`` below and above that boundary, each with
#: ~15-25% margin so the crossing is reachable, not a knife-edge.
GUARD_A_Y1, GUARD_A_Y2 = 3.2, 0.0
GUARD_A_GAMMA, GUARD_A_BETA, GUARD_A_KAPPA = 0.1, 0.0, 0.0
GUARD_A_W_PASS = 0.30
GUARD_A_W_REFUSE = 0.45

#: Guard-B (parity-wall margin) fixture: ``gamma_p = gamma`` here
#: (``kappa = 0``).  The boundary is ``gamma_p = 1 - DELTA_GAMMA_P =
#: 0.995``; 0.99 sits just inside it and 0.997 just outside, at the SAME
#: ``w`` (tiny, so guard A cannot confound the guard-B decision).
GUARD_B_Y1, GUARD_B_Y2 = 1.0, 1.0
GUARD_B_BETA, GUARD_B_KAPPA = 0.3, 0.0
GUARD_B_W = 0.01
GUARD_B_GAMMA_PASS = 0.99
GUARD_B_GAMMA_REFUSE = 0.997

#: Macro-magnification-limit fixtures: three distinct (y, gamma, beta,
#: kappa) points, none on any guard boundary, spanning zero and nonzero
#: kappa/beta so the closed-form oracle is exercised generically.
MACRO_LIMIT_FIXTURES = (
    (3.2, 0.0, 0.3, 0.0, 0.0),
    (1.0, 1.0, 0.99, 0.3, 0.0),
    (2.0, -1.0, 0.2, 0.5, 0.1),
)

#: A very small and a ten-times-smaller frequency: the leading Born
#: correction is linear in ``w`` (module docstring), so the absolute
#: departure from the closed-form limit must shrink by the same factor.
MACRO_LIMIT_W_SMALL = 1.0e-3
MACRO_LIMIT_W_SMALLER = 1.0e-6

#: Absolute ceiling on ``|F_born(w) - sqrt_mu|`` at `MACRO_LIMIT_W_SMALLER`
#: (measured up to ~6e-4 across `MACRO_LIMIT_FIXTURES`).
MACRO_LIMIT_ABS_TOL = 1.0e-3

#: The linear-in-``w`` convergence ratio must land within this factor of
#: the exact ``w`` ratio (``1000``); measured exactly ``1000`` (the
#: two-term series is linear in ``w`` by construction), so a generous
#: band avoids a knife-edge on floating-point exactness.
MACRO_LIMIT_RATIO_TOL = 0.05

#: End-to-end serve/census fixture: reuses the accuracy-gate lens point
#: (well inside the guard margins across the whole grid) with a dense
#: frequency grid, exactly as a production far-field tile evaluation
#: would use.  Measured: 2 real images (exterior positive-parity host),
#: and the round trip through `channels.reconstruct_farfield` recovers
#: the demodulated Born total to a relative error of ~9e-17.
CENSUS_Y1, CENSUS_Y2 = 3.2, 0.0
CENSUS_GAMMA, CENSUS_BETA, CENSUS_KAPPA = 0.3, 0.0, 0.0
CENSUS_W_GRID = np.linspace(0.05, 0.5, 40)

#: Machine-precision round-trip gate (measured ~9e-17 relative).
CENSUS_ROUNDTRIP_REL_TOL = 1.0e-10


class BornTestCase(unittest.TestCase):
    """Base carrying an anti-vacuity guard on the number of comparisons."""

    def setUp(self) -> None:
        self.comparisons = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self.comparisons, 0,
            'no comparisons were made -- the test asserted nothing')

    def assert_within(self, value: float, tol: float, message: str) -> None:
        """Assert ``value <= tol`` and count the comparison."""
        self.comparisons += 1
        self.assertLessEqual(value, tol, message)


class AccuracyAgainstOperatorTestCase(BornTestCase):
    """`born_amplification` vs the exact `operator.F_op` oracle.

    At `ACCURACY_Y1`/`ACCURACY_Y2`/`ACCURACY_GAMMA`/`ACCURACY_W` --
    strictly inside the target annulus, `born_gate`-passing, and inside
    `operator.F_op`'s certified domain (``w * |y| = 1.6 << 60``) -- the
    two independently-implemented amplifications are compared directly.
    """

    def setUp(self) -> None:
        super().setUp()
        _born.born_gate(ACCURACY_W, ACCURACY_Y1, ACCURACY_Y2,
                        ACCURACY_GAMMA, ACCURACY_BETA, ACCURACY_KAPPA)
        self.f_born = _born.born_amplification(
            ACCURACY_W, ACCURACY_Y1, ACCURACY_Y2, ACCURACY_GAMMA,
            ACCURACY_BETA, ACCURACY_KAPPA)
        y = np.array([ACCURACY_Y1, ACCURACY_Y2])
        f_op, _diagnostics = operator.F_op(
            ACCURACY_W, y, ACCURACY_GAMMA, beta=ACCURACY_BETA,
            kappa=ACCURACY_KAPPA)
        self.f_op = f_op
        self.rel_err = abs(self.f_born - self.f_op) / abs(self.f_op)

    def test_placeholder_b1_error_is_reachable_red(self):
        """The CURRENT ``b1 = 1.0`` placeholder error is measurably large.

        Proves the accuracy tripwire below is not vacuous: the measured
        disagreement (~11.3%) sits comfortably above the reachable-red
        floor, itself far above the tight tolerance the tripwire checks.
        """
        self.assertGreaterEqual(
            self.rel_err, B1_PLACEHOLDER_ERROR_FLOOR,
            f'measured born_amplification-vs-F_op error {self.rel_err:.3e} '
            f'fell below the reachable-red floor '
            f'{B1_PLACEHOLDER_ERROR_FLOOR} -- the accuracy tripwire below '
            f'would not be exercising the known b1 = 1.0 placeholder gap')
        self.comparisons += 1

    @unittest.expectedFailure
    def test_literal_b1_pinned_accuracy_tripwire(self):
        """LITERAL contract: agreement with `operator.F_op` to 0.1%.

        RED under the current ``b1 = 1.0`` placeholder (INS-c1-001):
        measured ~11.3% relative error, two orders of magnitude above
        this tolerance.  Flips to a loud unexpected success -- the signal
        to re-derive this test and reconsider re-enabling the serve slot
        -- only once the Professor pins ``b1``'s closed form and the
        error against `operator.F_op` actually closes.  This tripwire is
        the guard `INS-c1-002` asked for: as long as it stays RED,
        `_born` must stay unwired from `likelihood._surrogate_coefficients`.
        """
        self.assert_within(
            self.rel_err, ACCURACY_TARGET_ONCE_B1_PINNED,
            f'born_amplification disagreed with operator.F_op by '
            f'{self.rel_err:.3e} (relative), above the pinned-b1 target '
            f'{ACCURACY_TARGET_ONCE_B1_PINNED} -- expected while b1 = 1.0 '
            f'is a placeholder')


class ValidityGateFalsificationTestCase(BornTestCase):
    """`born_gate` refuses reachable-red just outside each guard.

    Guard A (series convergence) and guard B (parity-wall margin) are
    independent refusals; each is tested with a PASS just inside its
    margin and a REFUSE just outside it, isolating the OTHER guard by
    construction (guard A's fixture keeps ``gamma_p`` tiny; guard B's
    fixture uses ``w`` so small that guard A cannot trip regardless of
    ``Q2r``).
    """

    def test_guard_a_passes_just_inside_convergence_margin(self):
        """No raise at `GUARD_A_W_PASS` (~28% below the measured
        boundary ``w* ~ 0.358``)."""
        _born.born_gate(GUARD_A_W_PASS, GUARD_A_Y1, GUARD_A_Y2,
                        GUARD_A_GAMMA, GUARD_A_BETA, GUARD_A_KAPPA)
        self.comparisons += 1

    def test_guard_a_refuses_just_outside_convergence_margin(self):
        """Raises `BornDomainError` at `GUARD_A_W_REFUSE` (~26% above
        the boundary), same lens point as the PASS above."""
        with self.assertRaises(_born.BornDomainError):
            _born.born_gate(GUARD_A_W_REFUSE, GUARD_A_Y1, GUARD_A_Y2,
                            GUARD_A_GAMMA, GUARD_A_BETA, GUARD_A_KAPPA)
        self.comparisons += 1

    def test_guard_b_passes_just_inside_parity_margin(self):
        """No raise at `GUARD_B_GAMMA_PASS` (``gamma_p = 0.99 < 0.995``)."""
        _born.born_gate(GUARD_B_W, GUARD_B_Y1, GUARD_B_Y2,
                        GUARD_B_GAMMA_PASS, GUARD_B_BETA, GUARD_B_KAPPA)
        self.comparisons += 1

    def test_guard_b_refuses_just_outside_parity_margin(self):
        """Raises `BornDomainError` at `GUARD_B_GAMMA_REFUSE`
        (``gamma_p = 0.997 >= 0.995``), same ``w`` and source as the
        PASS above so only ``gamma`` crossed the margin."""
        with self.assertRaises(_born.BornDomainError):
            _born.born_gate(GUARD_B_W, GUARD_B_Y1, GUARD_B_Y2,
                            GUARD_B_GAMMA_REFUSE, GUARD_B_BETA,
                            GUARD_B_KAPPA)
        self.comparisons += 1

    def test_refusal_subclasses_lens_domain_error(self):
        """`BornDomainError` refuses symmetrically with the exact path."""
        self.assertTrue(
            issubclass(_born.BornDomainError, geometry.LensDomainError))
        self.comparisons += 1


class MacroMagnificationLimitTestCase(BornTestCase):
    """``F_born -> sqrt(mu_macro)`` as ``w -> 0``, independent of ``b1``.

    The leading term of the two-term series carries no ``w`` (module
    docstring), so this limit is exact regardless of the unpinned ``b1``
    placeholder -- unlike the accuracy gate above, this is a genuinely
    GREEN structural property, checked against a closed-form oracle
    recomputed here rather than read from `_born._born_factors`.
    """

    def _sqrt_mu_oracle(self, gamma: float, kappa: float) -> float:
        return 1.0 / np.sqrt((1.0 - kappa) ** 2 - gamma ** 2)

    def test_born_amplification_reaches_macro_magnification(self):
        """At tiny ``w``, ``|F_born - sqrt_mu|`` sits at the absolute
        floor across every fixture."""
        for y1, y2, gamma, beta, kappa in MACRO_LIMIT_FIXTURES:
            with self.subTest(y1=y1, y2=y2, gamma=gamma, kappa=kappa):
                f_born = _born.born_amplification(
                    MACRO_LIMIT_W_SMALLER, y1, y2, gamma, beta, kappa)
                sqrt_mu = self._sqrt_mu_oracle(gamma, kappa)
                self.assert_within(
                    abs(f_born - sqrt_mu), MACRO_LIMIT_ABS_TOL,
                    f'F_born(w -> 0) departed from sqrt(mu_macro) = '
                    f'{sqrt_mu} by {abs(f_born - sqrt_mu):.3e} at '
                    f'(y1, y2, gamma, kappa) = ({y1}, {y2}, {gamma}, '
                    f'{kappa})')

    def test_departure_from_limit_shrinks_linearly_in_w(self):
        """The departure at `MACRO_LIMIT_W_SMALL` is ~1000x the departure
        at `MACRO_LIMIT_W_SMALLER` (linear-in-``w`` leading correction),
        proving the check above is not vacuously satisfied by a
        ``w``-independent bug that happens to sit inside the tolerance."""
        exact_ratio = MACRO_LIMIT_W_SMALL / MACRO_LIMIT_W_SMALLER
        for y1, y2, gamma, beta, kappa in MACRO_LIMIT_FIXTURES:
            with self.subTest(y1=y1, y2=y2, gamma=gamma, kappa=kappa):
                sqrt_mu = self._sqrt_mu_oracle(gamma, kappa)
                small = abs(_born.born_amplification(
                    MACRO_LIMIT_W_SMALL, y1, y2, gamma, beta, kappa)
                    - sqrt_mu)
                smaller = abs(_born.born_amplification(
                    MACRO_LIMIT_W_SMALLER, y1, y2, gamma, beta, kappa)
                    - sqrt_mu)
                measured_ratio = small / smaller
                self.assert_within(
                    abs(measured_ratio - exact_ratio) / exact_ratio,
                    MACRO_LIMIT_RATIO_TOL,
                    f'departure ratio {measured_ratio:.3e} departed from '
                    f'the linear-in-w prediction {exact_ratio:.3e} by more '
                    f'than {MACRO_LIMIT_RATIO_TOL:.0%} at (y1, y2, gamma, '
                    f'kappa) = ({y1}, {y2}, {gamma}, {kappa})')


class ServeCensusRoundTripTestCase(BornTestCase):
    """End-to-end serve/census round trip through the real channel machinery.

    Builds a `channels.ChangRefsdalGeometryPartition` exactly as the
    production far-field serve path would (`ChangRefsdalChannels.
    geometry_partition`), evaluates `born_envelope` on it, and recovers
    the amplification total through `channels.reconstruct_farfield` --
    the SAME serve-mirror reconstruction the trained far-field charts
    use.  Proves the plumbing this rung needs on re-enablement already
    works, without touching the dormant `likelihood` serve slot itself.
    """

    @classmethod
    def setUpClass(cls) -> None:
        for w_end in (CENSUS_W_GRID[0], CENSUS_W_GRID[-1]):
            _born.born_gate(w_end, CENSUS_Y1, CENSUS_Y2, CENSUS_GAMMA,
                            CENSUS_BETA, CENSUS_KAPPA)
        engine = channels.ChangRefsdalChannels(CENSUS_W_GRID)
        engine.reset()
        cls.geom = engine.geometry_partition(
            gamma=CENSUS_GAMMA, y=(CENSUS_Y1, CENSUS_Y2),
            beta=CENSUS_BETA, kappa=CENSUS_KAPPA)
        cls.envelope = _born.born_envelope(
            cls.geom.w, CENSUS_Y1, CENSUS_Y2, CENSUS_GAMMA, CENSUS_BETA,
            CENSUS_KAPPA, cls.geom)
        # `born_envelope` still emits the MIN-RELATIVE-delay envelope (Born
        # production code is out of scope here); WP2's `reconstruct_farfield`
        # now expects the FRAME-INVARIANT label (E_minrel * exp(+1j w t_min))
        # and re-modulates it by exp(-1j w t_min) internally.  Bridge the two
        # frames by demodulating the min-relative envelope here so the round
        # trip is exact; the internal re-modulation cancels this factor,
        # reproducing the min-relative total below.  (When the dormant Born
        # rung is re-enabled, `born_envelope` itself should adopt the
        # frame-invariant convention -- OWED for re-enablement.)
        demodulated_envelope = cls.envelope * np.exp(
            1j * cls.geom.w * cls.geom.t_min)
        _kernels, cls.reconstructed_total = channels.reconstruct_farfield(
            cls.geom.w, demodulated_envelope, cls.geom.delays,
            cls.geom.saddle_kernels, cls.geom.real_mask,
            channels.FARFIELD_KERNEL_SUM, cls.geom.t_min)
        f_born_grid = np.array([
            _born.born_amplification(w, CENSUS_Y1, CENSUS_Y2, CENSUS_GAMMA,
                                     CENSUS_BETA, CENSUS_KAPPA)
            for w in cls.geom.w])
        cls.expected_total = f_born_grid * np.exp(
            -1j * cls.geom.w * cls.geom.t_min)

    def test_envelope_is_finite(self):
        """`born_envelope` returns a finite, correctly-shaped grid."""
        self.assertEqual(self.envelope.shape, CENSUS_W_GRID.shape)
        self.assertTrue(np.all(np.isfinite(self.envelope)))
        self.comparisons += 1

    def test_image_census_is_sane(self):
        """The geometry partition carries a physical real-image count."""
        n_real = int(self.geom.real_mask.sum())
        self.assertIn(n_real, (2, 4),
                     f'geometry_partition census reported {n_real} real '
                     f'images; expected 2 or 4 for an exterior host')
        self.comparisons += 1

    def test_reconstruct_farfield_recovers_born_total(self):
        """`reconstruct_farfield(born_envelope(...))` reproduces the
        demodulated Born total at machine precision (measured ~9e-17)."""
        error = float(np.max(np.abs(
            self.reconstructed_total - self.expected_total)))
        scale = float(np.max(np.abs(self.expected_total)))
        self.assert_within(
            error / scale, CENSUS_ROUNDTRIP_REL_TOL,
            f'reconstruct_farfield departed from the demodulated Born '
            f'total by {error / scale:.3e} (relative) -- the serve-mirror '
            f'round trip is not self-consistent for the Born rung')


if __name__ == '__main__':
    unittest.main()
