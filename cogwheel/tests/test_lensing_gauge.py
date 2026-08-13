"""
Tests for the `lensing.chang_refsdal._gauge` partition primitives.

These primitives are pure algebra: they never approximate the total
amplification handed to them, they only redistribute it between
channels. That makes them testable with NO lens physics and no
wave-optics oracle -- a synthetic total is just as good as a real one,
because the identity

    sum_j exp(1j*w*tau_j) * K_j == F

is claimed to hold for *arbitrary* input. The tests exploit this
directly: they feed adversarial totals, delays, targets and switches,
including combinations no real lens would produce, because a gauge that
is only exact on physical input is not exact.

The headline property is that `exact_transition_channels` reconstructs
its input for ANY switch value, including a switch that is nonsense
(outside [0, 1]) and a physical target that is wildly wrong. That is
the whole point of the residual projection: the switch buys smoothness,
never accuracy, and cannot inject error into the total. A test suite
that only checked S=0 and S=1 would pass on an implementation that
silently dropped the projection in between.

TOLERANCES ARE SCALE-AWARE, DELIBERATELY. The identity is algebraic, so
the only error is roundoff -- and roundoff scales with the largest
INTERMEDIATE, sum_j abs(K_j), which diverges like sqrt(abs(mu_a)) at a
critical point, NOT with abs(F) ~ 1. A flat relative gate is therefore
unachievable near a fold and its failure would indicate nothing.
`NearFoldScalingTestCase` pins this empirically: it drives the targets
to abs(H) ~ 1e8 and asserts BOTH that the scale-aware bound still holds
AND that a flat 1e-12 gate is violated by orders of magnitude, so the
choice of gate is evidence rather than assertion.

`ExactnessTestCase.tearDown` guards against a sweep whose comparisons
are all skipped and which therefore asserts nothing.

SACR-C PHYSICS GATES -- THE END-TO-END RECONSTRUCTION LAYER
-----------------------------------------------------------
The second half of the suite pins the SACR-C decomposition the Build-3f
engine (`channels.ChangRefsdalChannels`) and its downstream likelihood
use:

    F(w) = sum_a exp(1j*w*tau_a) * S_a(w)*H_a(w) + exp(1j*w*tau_c) * E(w),

where ``H_a`` is the carrier-free analytic saddle kernel
(`geometry.image_kernel`), ``S_a`` the criticality-separation switch,
``tau_c`` the parked critical carrier delay and ``E`` the single
demodulated transition envelope.  Five gates fence it:

* GATE 1 -- the reconstruction is an ALGEBRAIC identity: the SACR-C sum
  reproduces `ChangRefsdalPartition.exact_total` to ``1e-13`` relative on
  every anchor (measured ``~2e-15``, i.e. the telescoping is bit-exact),
  because only ``E`` is ever approximated, never the carrier algebra.
* GATE 2 -- a hindsight GREEDY node oracle on ``E`` reaches
  ``max|dF|/max|F| < 1e-3`` with ``<= 26`` nodes on every anchor
  (measured 17-21), isolating envelope SMOOTHNESS from the production
  placement heuristic.  The gate lives here, at the gauge/reconstruction
  layer; the SHIPPED LOO placement and its timing (GATE 3, STRUCTURAL)
  live with the likelihood suite that owns `likelihood.py`.
* GATE 4 -- the headline boundedness claim: at fold and cusp caustic
  crossings, ``|S_a H_a| <= 2`` (measured ``<= 1.3``) even though ``H_a``
  diverges like ``sqrt|mu_a|`` at the caustic, because the switch shuts
  off (``S_a -> 0``) at exactly the rate that tames it.  Its fixture is
  built from `geometry`, `operator` and `_gauge` ONLY -- NEVER
  `channels` -- so it is an independent oracle for the switch, not a
  tautology (F002); `SacrcFixtureIndependenceTestCase` enforces that by
  AST inspection.
* GATE 5 -- the deep-band (w -> 0) macro-magnification limit: the
  reconstruction tends to the LITERAL Gaussian closed form
  ``1/sqrt((1-kappa)**2 - gamma**2)`` (computed independently of the
  pipeline) to ``1e-6`` relative and FLAT across three decades, so a
  spurious ``1/w`` prefactor leak would be caught.
* F001 -- carrier-phase correctness at thousands of radians: the
  range-reduced carriers in the reconstruction agree with an independent
  high-precision `mpmath` ``exp(1j*w*tau)`` reference to ``1e-10``
  relative up to ``w*tau ~ 1.5e4`` rad, with no float64 large-argument
  precision loss.  (`mpmath` is a TEST-ONLY oracle, never a runtime
  dependency of `_gauge`; `GaugeIndependenceTestCase` pins that.)

TOLERANCES.  GATE 1/F001 gates sit at the algebraic roundoff floor
(``1e-13``/``1e-10``), orders below the measured ``~2e-15``/``~5e-13``,
so they fail on a genuinely broken carrier and nothing else.  GATE 2's
``26`` node ceiling is ~25% above the measured worst (21).  GATE 4's
``2`` is provably more conservative than the ``1.3`` measured, and the
``S_a`` product is a tighter object than F008's kernel-sum ceiling.
`SacrcSelfFalsificationTestCase` proves the whole layer can go RED:
forcing ``S_a = 1`` at a crossing blows ``|S_a H_a|`` to ``~1e8``,
perturbing ``E`` breaks the GATE 1 identity, and an extreme-phase
carrier (``w*tau ~ 1e12``) breaks F001.
"""

import ast
import inspect
import itertools
import pathlib
import textwrap
from unittest import TestCase, main, mock

import numpy as np
import mpmath
from scipy.interpolate import CubicSpline

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_MPL = False

from cogwheel.lensing.chang_refsdal import (
    channels, geometry, operator, _gauge)

#: Diagnostic-plot directory, shared with the sibling lensing suites.
_OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'


#: Float64 machine epsilon; the roundoff unit of every bound here.
EPS = np.finfo(np.float64).eps

#: Slack over the roundoff model in the scale-aware reconstruction
#: bound, per the build plan. The measured margin is ~30x at the worst
#: configuration tested, so this is not a fitted constant.
RECONSTRUCTION_SLACK = 100.0

#: Documented default switch window, shared with the channel tracker.
#: DERIVED from the production window so these fixtures follow the engine:
#: a pinned ``0.5``/``4.0`` would keep passing while measuring a window the
#: engine no longer uses.  (The crossing-fixture helper further down this
#: file already reads ``operator.RHO_START``/``operator.RHO_END`` directly;
#: these module constants were the only remaining second copy.)
RHO_START = operator.RHO_START
RHO_END = operator.RHO_END

#: Seed for every random sweep, so that a failure is reproducible and a
#: frozen backstop is a fixed number rather than a lottery.
SEED = 20260716

#: Representative dimensionless frequency grid.
W_GRID = np.linspace(1.0, 40.0, 9)

#: Representative member delays: unequal spacing, so that a bug which
#: assumes a uniform cluster shows up.
MEMBER_DELAYS = np.array([0.0, 0.3, 0.55, 0.9])


def _random_complex(rng, shape):
    """Return complex noise of the given shape."""
    return rng.normal(size=shape) + 1j*rng.normal(size=shape)


def _imported_top_level_modules(module):
    """Return the set of top-level module names a module imports."""
    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split('.')[0]
                         for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names.add('.'*node.level + (node.module or ''))
            else:
                names.add((node.module or '').split('.')[0])
    return names


class ExactnessTestCase(TestCase):
    """
    Base class for the identities the gauge primitives must satisfy
    exactly.

    Subclasses call `assert_reconstructs`, which applies the scale-aware
    roundoff bound and counts itself, so that `tearDown` can prove the
    test actually compared something.
    """

    def setUp(self):
        self.rng = np.random.default_rng(SEED)
        self._comparisons = 0

    def tearDown(self):
        self.assertGreater(
            self._comparisons, 0,
            'Vacuous test: no exactness comparison ran, so this test '
            'asserted nothing. Check the sweep bounds.')

    def assert_reconstructs(self, w, delays, kernels, expected, msg):
        """
        Assert sum_j exp(i w tau_j) K_j == expected, to roundoff.

        The bound is ``100 * eps * (abs(expected) + sum_j abs(K_j))``.
        The second term is what makes it survive near a fold: the
        returned kernels carry the projected residual, so their sum of
        magnitudes bounds the largest intermediate the reconstruction
        passes through.
        """
        kernels = np.asarray(kernels)
        got = _gauge.reconstructed_total(w, delays, kernels)
        error = np.abs(got - expected)
        scale = np.abs(expected) + np.sum(np.abs(kernels), axis=-1)
        bound = RECONSTRUCTION_SLACK * EPS * scale
        self._comparisons += int(np.size(error))
        self.assertTrue(
            np.all(error <= bound),
            msg=f'{msg}: max reconstruction error '
                f'{np.max(error):.3e} exceeds scale-aware bound '
                f'{np.max(bound):.3e}')


class SmootherstepTestCase(TestCase):
    """The C2 switch S_j that hands a channel to its physical target."""

    def test_clamps_to_zero_below_and_one_above_the_window(self):
        below = _gauge.smootherstep([-10.0, 0.0, RHO_START],
                                    RHO_START, RHO_END)
        above = _gauge.smootherstep([RHO_END, 5.0, 1e6],
                                    RHO_START, RHO_END)
        np.testing.assert_array_equal(below, 0.0)
        np.testing.assert_array_equal(above, 1.0)

    def test_is_monotonic_and_bounded_across_the_window(self):
        x = np.linspace(RHO_START - 1.0, RHO_END + 1.0, 2001)
        switch = _gauge.smootherstep(x, RHO_START, RHO_END)
        self.assertTrue(np.all(np.diff(switch) >= 0.0),
                        'switch must be non-decreasing')
        self.assertTrue(np.all((switch >= 0.0) & (switch <= 1.0)),
                        'switch must stay within [0, 1]')

    def test_is_one_half_at_the_window_midpoint(self):
        middle = 0.5*(RHO_START + RHO_END)
        self.assertAlmostEqual(
            float(_gauge.smootherstep(middle, RHO_START, RHO_END)),
            0.5, places=15)

    def test_second_derivative_vanishes_at_both_joins(self):
        """
        The defining property, and the reason smootherstep is used
        instead of the C1 smoothstep.

        The second difference at a join must vanish LINEARLY in the step
        h: D2(h) ~ 10*h/width**3. The C1 smoothstep would instead tend
        to the constant 3/width**2 = 7.0e-2, which exceeds this bound by
        three orders of magnitude at h=1e-4 -- so this test genuinely
        discriminates between the two, rather than passing on any smooth
        function.
        """
        width = RHO_END - RHO_START
        for join in (RHO_START, RHO_END):
            for step in (1e-2, 1e-3, 1e-4):
                with self.subTest(join=join, step=step):
                    second_difference = abs(
                        float(_gauge.smootherstep(join + step,
                                                  RHO_START, RHO_END))
                        - 2.0*float(_gauge.smootherstep(
                            join, RHO_START, RHO_END))
                        + float(_gauge.smootherstep(join - step,
                                                    RHO_START, RHO_END))
                    ) / step**2
                    self.assertLessEqual(
                        second_difference, 20.0*step/width**3,
                        f'second difference {second_difference:.3e} at '
                        f'join {join} does not vanish linearly in the '
                        f'step; a C1 switch would give {3/width**2:.3e}')

    def test_preserves_shape_and_accepts_scalars(self):
        self.assertEqual(
            _gauge.smootherstep(np.zeros((3, 4)), RHO_START,
                                RHO_END).shape, (3, 4))
        self.assertEqual(
            np.ndim(_gauge.smootherstep(1.0, RHO_START, RHO_END)), 0)

    def test_raises_on_a_non_ordered_window(self):
        for x0, x1 in ((4.0, 0.5), (1.0, 1.0), (0.0, np.nan)):
            with self.subTest(x0=x0, x1=x1):
                with self.assertRaises(ValueError):
                    _gauge.smootherstep(1.0, x0, x1)


class ExactClusterKernelTestCase(ExactnessTestCase):
    """The demodulated residual carried by an unresolved cluster."""

    def test_reproduces_its_defining_split(self):
        """total == persistent + exp(i w tau_c) * K_cluster."""
        tau_cluster = 0.4
        total = _random_complex(self.rng, W_GRID.size)
        persistent = _random_complex(self.rng, W_GRID.size)

        cluster = _gauge.exact_cluster_kernel(
            W_GRID, total, persistent, tau_cluster)
        rebuilt = persistent + np.exp(1j*W_GRID*tau_cluster)*cluster

        error = np.abs(rebuilt - total)
        scale = np.abs(total) + np.abs(persistent)
        self._comparisons += int(np.size(error))
        self.assertTrue(
            np.all(error <= RECONSTRUCTION_SLACK*EPS*scale),
            f'cluster kernel does not invert its own definition: '
            f'max error {np.max(error):.3e}')

    def test_vanishes_when_the_persistent_images_are_the_whole_total(self):
        total = _random_complex(self.rng, W_GRID.size)
        cluster = _gauge.exact_cluster_kernel(W_GRID, total, total, 0.4)
        self._comparisons += int(np.size(cluster))
        np.testing.assert_array_equal(cluster, 0.0)

    def test_accepts_a_scalar_frequency(self):
        cluster = _gauge.exact_cluster_kernel(7.0, 1.0+2.0j, 0.5j, 0.4)
        self._comparisons += 1
        self.assertEqual(np.ndim(cluster), 0)


class UnresolvedMemberChannelsTestCase(ExactnessTestCase):
    """The artificial split of one exact cluster among its members."""

    def test_split_is_exact_on_a_frequency_grid(self):
        tau_cluster = float(np.mean(MEMBER_DELAYS))
        cluster = _random_complex(self.rng, W_GRID.size)
        expected = np.exp(1j*W_GRID*tau_cluster)*cluster

        kernels = _gauge.unresolved_member_channels(
            W_GRID, cluster, tau_cluster, MEMBER_DELAYS)

        self.assertEqual(kernels.shape, (W_GRID.size,
                                         MEMBER_DELAYS.size))
        self.assert_reconstructs(W_GRID, MEMBER_DELAYS, kernels,
                                 expected, 'equal-weight split')

    def test_split_is_exact_for_arbitrary_weights(self):
        tau_cluster = float(np.mean(MEMBER_DELAYS))
        cluster = _random_complex(self.rng, W_GRID.size)
        expected = np.exp(1j*W_GRID*tau_cluster)*cluster

        for weights in ([1.0, 1.0, 1.0, 1.0],
                        [0.7, 0.1, 0.15, 0.05],
                        [1.0, 0.0, 0.0, 0.0],
                        [3.0, 11.0, 0.25, 6.5]):
            with self.subTest(weights=weights):
                kernels = _gauge.unresolved_member_channels(
                    W_GRID, cluster, tau_cluster, MEMBER_DELAYS,
                    weights)
                self.assert_reconstructs(
                    W_GRID, MEMBER_DELAYS, kernels, expected,
                    f'split with weights {weights}')

    def test_weights_are_normalized_not_taken_literally(self):
        """Unit sum is enforced internally, so scale must not matter."""
        tau_cluster = float(np.mean(MEMBER_DELAYS))
        cluster = _random_complex(self.rng, W_GRID.size)
        base = np.array([0.7, 0.1, 0.15, 0.05])

        reference = _gauge.unresolved_member_channels(
            W_GRID, cluster, tau_cluster, MEMBER_DELAYS, base)
        scaled = _gauge.unresolved_member_channels(
            W_GRID, cluster, tau_cluster, MEMBER_DELAYS, 137.0*base)

        self._comparisons += int(np.size(reference))
        np.testing.assert_allclose(scaled, reference, rtol=0.0,
                                   atol=8*EPS*np.max(np.abs(reference)))

    def test_default_weights_are_equal(self):
        tau_cluster = float(np.mean(MEMBER_DELAYS))
        cluster = _random_complex(self.rng, W_GRID.size)
        default = _gauge.unresolved_member_channels(
            W_GRID, cluster, tau_cluster, MEMBER_DELAYS)
        explicit = _gauge.unresolved_member_channels(
            W_GRID, cluster, tau_cluster, MEMBER_DELAYS,
            np.full(MEMBER_DELAYS.size, 0.25))
        self._comparisons += int(np.size(default))
        np.testing.assert_array_equal(default, explicit)

    def test_split_is_exact_at_a_scalar_frequency(self):
        tau_cluster = float(np.mean(MEMBER_DELAYS))
        cluster = 0.3 - 1.4j
        expected = np.exp(1j*13.0*tau_cluster)*cluster
        kernels = _gauge.unresolved_member_channels(
            13.0, cluster, tau_cluster, MEMBER_DELAYS)
        self.assertEqual(kernels.shape, (MEMBER_DELAYS.size,))
        self.assert_reconstructs(13.0, MEMBER_DELAYS, kernels, expected,
                                 'scalar-frequency split')


class ExactTransitionChannelsTestCase(ExactnessTestCase):
    """The blended gauge plus residual projection: the headline claim."""

    def _total(self):
        return _random_complex(self.rng, W_GRID.size)

    def test_reconstruction_is_exact_for_any_switch_value(self):
        """
        The projection must hold the total for EVERY switch, not just at
        the endpoints. Values outside [0, 1] are included on purpose:
        the identity is algebraic in the switch, so a nonsense switch
        must still reconstruct. This is what catches an implementation
        that drops the projection mid-transition.
        """
        total = self._total()
        tau_cluster = float(np.mean(MEMBER_DELAYS))
        targets = _random_complex(self.rng,
                                  (W_GRID.size, MEMBER_DELAYS.size))

        for switch in (0.0, 0.25, 0.5, 0.75, 1.0, -3.0, 7.5):
            with self.subTest(switch=switch):
                kernels = _gauge.exact_transition_channels(
                    W_GRID, total, tau_cluster, MEMBER_DELAYS, targets,
                    switch)
                self.assert_reconstructs(
                    W_GRID, MEMBER_DELAYS, kernels, total,
                    f'transition at switch={switch}')

    def test_reconstruction_is_exact_under_the_documented_switch(self):
        """The real switch: smootherstep(w*delta, rho_start, rho_end)."""
        total = self._total()
        tau_cluster = float(np.mean(MEMBER_DELAYS))
        targets = _random_complex(self.rng,
                                  (W_GRID.size, MEMBER_DELAYS.size))

        for separation in (1e-6, 0.01, 0.2, 1.0, 50.0):
            with self.subTest(separation=separation):
                switch = _gauge.smootherstep(W_GRID*separation,
                                             RHO_START, RHO_END)
                kernels = _gauge.exact_transition_channels(
                    W_GRID, total, tau_cluster, MEMBER_DELAYS, targets,
                    switch)
                self.assert_reconstructs(
                    W_GRID, MEMBER_DELAYS, kernels, total,
                    f'smootherstep switch at separation={separation}')

    def test_accepts_a_per_channel_two_dimensional_switch(self):
        """
        The shape the channel tracker needs: each label switches on its
        own delay separation, so the switch is (n_w, n_members) rather
        than one value per frequency.
        """
        total = self._total()
        tau_cluster = float(np.mean(MEMBER_DELAYS))
        targets = _random_complex(self.rng,
                                  (W_GRID.size, MEMBER_DELAYS.size))
        separations = np.array([1e-6, 0.05, 0.4, 2.0])
        switch = _gauge.smootherstep(
            np.multiply.outer(W_GRID, separations), RHO_START, RHO_END)
        self.assertEqual(switch.shape,
                         (W_GRID.size, MEMBER_DELAYS.size))

        kernels = _gauge.exact_transition_channels(
            W_GRID, total, tau_cluster, MEMBER_DELAYS, targets, switch)
        self.assert_reconstructs(W_GRID, MEMBER_DELAYS, kernels, total,
                                 'per-channel switch')

    def test_a_one_dimensional_switch_is_per_frequency(self):
        """Pins the documented broadcasting convention."""
        total = self._total()
        tau_cluster = float(np.mean(MEMBER_DELAYS))
        targets = _random_complex(self.rng,
                                  (W_GRID.size, MEMBER_DELAYS.size))
        per_frequency = _gauge.smootherstep(W_GRID*0.1, RHO_START,
                                            RHO_END)

        from_one_d = _gauge.exact_transition_channels(
            W_GRID, total, tau_cluster, MEMBER_DELAYS, targets,
            per_frequency)
        from_two_d = _gauge.exact_transition_channels(
            W_GRID, total, tau_cluster, MEMBER_DELAYS, targets,
            np.repeat(per_frequency[:, None], MEMBER_DELAYS.size,
                      axis=1))

        self._comparisons += int(np.size(from_one_d))
        np.testing.assert_array_equal(from_one_d, from_two_d)

    def test_reduces_to_the_artificial_split_at_zero_switch(self):
        total = self._total()
        tau_cluster = float(np.mean(MEMBER_DELAYS))
        targets = _random_complex(self.rng,
                                  (W_GRID.size, MEMBER_DELAYS.size))

        blended = _gauge.exact_transition_channels(
            W_GRID, total, tau_cluster, MEMBER_DELAYS, targets, 0.0)
        cluster = np.exp(-1j*W_GRID*tau_cluster)*total
        split = _gauge.unresolved_member_channels(
            W_GRID, cluster, tau_cluster, MEMBER_DELAYS)

        self._comparisons += int(np.size(blended))
        np.testing.assert_allclose(
            blended, split, rtol=0.0,
            atol=RECONSTRUCTION_SLACK*EPS*np.max(np.abs(split)))

    def test_projection_vanishes_when_the_targets_are_already_exact(self):
        """
        The resolved limit must cost nothing: if the physical targets
        already reproduce the total, a fully switched-on gauge must
        return them untouched.
        """
        tau_cluster = float(np.mean(MEMBER_DELAYS))
        targets = _random_complex(self.rng,
                                  (W_GRID.size, MEMBER_DELAYS.size))
        consistent_total = _gauge.reconstructed_total(
            W_GRID, MEMBER_DELAYS, targets)

        kernels = _gauge.exact_transition_channels(
            W_GRID, consistent_total, tau_cluster, MEMBER_DELAYS,
            targets, 1.0)

        self._comparisons += int(np.size(kernels))
        np.testing.assert_allclose(
            kernels, targets, rtol=0.0,
            atol=RECONSTRUCTION_SLACK*EPS*np.max(np.abs(targets)))

    def test_reconstruction_is_exact_at_a_scalar_frequency(self):
        tau_cluster = float(np.mean(MEMBER_DELAYS))
        total = 0.8 - 0.2j
        targets = _random_complex(self.rng, MEMBER_DELAYS.size)
        kernels = _gauge.exact_transition_channels(
            13.0, total, tau_cluster, MEMBER_DELAYS, targets, 0.5)
        self.assertEqual(kernels.shape, (MEMBER_DELAYS.size,))
        self.assert_reconstructs(13.0, MEMBER_DELAYS, kernels, total,
                                 'scalar-frequency transition')

    def test_raises_on_mismatched_target_shape(self):
        with self.assertRaises(ValueError):
            _gauge.exact_transition_channels(
                W_GRID, self._total(), 0.4, MEMBER_DELAYS,
                np.zeros((W_GRID.size, 3), complex), 0.5)
        self._comparisons += 1

    def test_raises_on_a_switch_that_does_not_broadcast(self):
        with self.assertRaises(ValueError):
            _gauge.exact_transition_channels(
                W_GRID, self._total(), 0.4, MEMBER_DELAYS,
                np.zeros((W_GRID.size, MEMBER_DELAYS.size), complex),
                np.zeros(3))
        self._comparisons += 1


class NearFoldScalingTestCase(ExactnessTestCase):
    """
    Why the reconstruction gate is scale-aware and not a flat 1e-12.

    Near a critical point the stationary-phase targets diverge like
    sqrt(abs(mu_a)). The identity stays algebraically exact, but the
    roundoff floor rises with the largest intermediate, so a flat
    relative gate fails for reasons that have nothing to do with a bug.
    This test drives that regime deliberately and pins BOTH halves of
    the claim, so the choice of gate is evidence.
    """

    def test_scale_aware_bound_holds_with_diverging_targets(self):
        total = _random_complex(self.rng, W_GRID.size)
        tau_cluster = float(np.mean(MEMBER_DELAYS))
        for magnitude in (1e0, 1e2, 1e4, 1e6, 1e8):
            with self.subTest(magnitude=magnitude):
                targets = magnitude*_random_complex(
                    self.rng, (W_GRID.size, MEMBER_DELAYS.size))
                kernels = _gauge.exact_transition_channels(
                    W_GRID, total, tau_cluster, MEMBER_DELAYS, targets,
                    0.5)
                self.assert_reconstructs(
                    W_GRID, MEMBER_DELAYS, kernels, total,
                    f'diverging targets at magnitude {magnitude:.0e}')

    def test_flat_relative_gate_is_genuinely_unachievable_near_a_fold(self):
        """
        The falsifiable half: a flat 1e-12 gate -- the build brief's
        original requirement -- MUST fail here. If this test ever goes
        green, the scale-aware bound is unnecessary and this suite is
        over-engineered; that would be worth knowing.
        """
        total = _random_complex(self.rng, W_GRID.size)
        tau_cluster = float(np.mean(MEMBER_DELAYS))
        worst = 0.0
        for _ in range(64):
            targets = 1e8*_random_complex(
                self.rng, (W_GRID.size, MEMBER_DELAYS.size))
            kernels = _gauge.exact_transition_channels(
                W_GRID, total, tau_cluster, MEMBER_DELAYS, targets, 0.5)
            got = _gauge.reconstructed_total(W_GRID, MEMBER_DELAYS,
                                             kernels)
            worst = max(worst, float(np.max(np.abs(got - total))))
        self._comparisons += 1
        self.assertGreater(
            worst, 1e-12,
            'a flat 1e-12 reconstruction gate now passes with targets '
            'of magnitude 1e8; the scale-aware bound may no longer be '
            'needed')


class ReconstructedTotalTestCase(ExactnessTestCase):
    """The single authoritative coherent channel sum."""

    def test_total_is_invariant_under_relabelling(self):
        """
        The property the whole labelling scheme rests on: the total is
        symmetric in the channel index, so only the SMOOTHNESS of
        individual kernels needs labels continued consistently. If this
        failed, a label swap would change the likelihood and the
        posterior would depend on evaluation history.

        Note the measured deviation for O(1) kernels sits at ~9.9e-16,
        i.e. within 1% of the build plan's literal 1e-15 gate: that
        number is a property of this fixed configuration, not a robust
        bound. The scale-aware bound is asserted first, for that reason.
        """
        tau = self.rng.normal(size=4)
        kernels = _random_complex(self.rng, (W_GRID.size, 4))
        reference = _gauge.reconstructed_total(W_GRID, tau, kernels)
        scale = np.sum(np.abs(kernels), axis=-1)

        worst = 0.0
        for permutation in itertools.permutations(range(4)):
            order = list(permutation)
            shuffled = _gauge.reconstructed_total(
                W_GRID, tau[order], kernels[:, order])
            error = np.abs(shuffled - reference)
            self._comparisons += int(np.size(error))
            self.assertTrue(
                np.all(error <= RECONSTRUCTION_SLACK*EPS*scale),
                f'relabelling by {permutation} changed the total by '
                f'{np.max(error):.3e}')
            worst = max(worst, float(np.max(error)))

        self.assertLessEqual(
            worst, 1e-15,
            f'label-permutation invariance of the total degraded to '
            f'{worst:.3e}, above the build plan gate of 1e-15')

    def test_matches_an_explicit_sum(self):
        tau = MEMBER_DELAYS
        kernels = _random_complex(self.rng, (W_GRID.size, tau.size))
        expected = np.array([
            sum(np.exp(1j*w*tau[j])*kernels[i, j]
                for j in range(tau.size))
            for i, w in enumerate(W_GRID)])
        got = _gauge.reconstructed_total(W_GRID, tau, kernels)
        self._comparisons += int(np.size(got))
        np.testing.assert_allclose(
            got, expected, rtol=0.0,
            atol=8*EPS*np.max(np.sum(np.abs(kernels), axis=-1)))

    def test_accepts_a_scalar_frequency(self):
        kernels = _random_complex(self.rng, MEMBER_DELAYS.size)
        got = _gauge.reconstructed_total(13.0, MEMBER_DELAYS, kernels)
        self._comparisons += 1
        self.assertEqual(np.ndim(got), 0)


class InputValidationTestCase(TestCase):
    """
    Guard clauses at the boundary.

    Every one of these would otherwise be a silent wrong answer: a
    non-normalizable weight vector breaks the projection's unit-sum
    requirement and so breaks exactness itself, while a mis-shaped delay
    array would broadcast into a plausible-looking result.
    """

    def test_rejects_weights_that_cannot_be_normalized(self):
        cluster = np.ones(W_GRID.size, complex)
        for weights in ([0.0, 0.0, 0.0, 0.0],
                        [1.0, -1.0, 0.5, 0.5],
                        [np.nan, 1.0, 1.0, 1.0],
                        [np.inf, 1.0, 1.0, 1.0]):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    _gauge.unresolved_member_channels(
                        W_GRID, cluster, 0.4, MEMBER_DELAYS, weights)

    def test_rejects_a_wrong_number_of_weights(self):
        cluster = np.ones(W_GRID.size, complex)
        for weights in ([0.5, 0.5], [0.2]*5, [[0.25]*4]):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    _gauge.unresolved_member_channels(
                        W_GRID, cluster, 0.4, MEMBER_DELAYS, weights)

    def test_rejects_malformed_member_delays(self):
        cluster = np.ones(W_GRID.size, complex)
        for delays in (np.zeros((2, 2)), np.array([]),
                       np.array([0.0, np.nan])):
            with self.subTest(delays=delays.shape):
                with self.assertRaises(ValueError):
                    _gauge.unresolved_member_channels(
                        W_GRID, cluster, 0.4, delays)

    def test_rejects_malformed_frequencies(self):
        for w in (np.zeros((2, 2)), np.array([1.0, np.nan]),
                  np.array([np.inf])):
            with self.subTest(w=np.shape(w)):
                with self.assertRaises(ValueError):
                    _gauge.reconstructed_total(
                        w, MEMBER_DELAYS,
                        np.zeros((np.size(w), MEMBER_DELAYS.size),
                                 complex))


class GaugeIndependenceTestCase(TestCase):
    """
    The gauge primitives must stay dependency-free.

    They are imported by both the production channel tracker and the
    crossing-scenario builders. If they ever reach back into geometry,
    the operator, or the tracker, that shared-leaf status is gone and
    the two consumers become coupled through this module.
    """

    def test_imports_nothing_but_numpy(self):
        self.assertEqual(_imported_top_level_modules(_gauge),
                         {'__future__', 'numpy'})

    def test_does_not_import_the_rest_of_the_subpackage(self):
        imported = _imported_top_level_modules(_gauge)
        for forbidden in ('geometry', 'operator', 'channels',
                          'crossings', '_dd', '_hyp1f1', 'scipy',
                          'mpmath'):
            with self.subTest(module=forbidden):
                self.assertNotIn(forbidden, imported)


# --------------------------------------------------------------------------
# SACR-C end-to-end reconstruction gates (GATE 1, 2, 4, 5, F001)
# --------------------------------------------------------------------------

#: Positive-parity anchor configs shared with the lens-engine suite:
#: ``(label, gamma, y, beta, kappa)``.  Declared here rather than imported
#: from `test_lensing_likelihood`, so this suite never reaches into
#: another suite's internals for its ground truth.
ANCHOR_CONFIGS = (
    ('two-image',     0.20,  (0.50, 0.00),  0.0,  0.0),
    ('four-image',    0.20,  (0.08, 0.06),  0.0,  0.0),
    ('near-cusp',     0.20,  (-0.38, 0.00), 0.0,  0.0),
    ('kappa',         0.112, (0.30, 0.10),  0.0,  0.30),
    ('rotated-shear', 0.20,  (0.25, 0.10),  0.70, 0.0),
)

#: Two-decade production frequency window on which every anchor's
#: operator branch converges; ``w >= ~50`` trips a named engine refusal
#: for the two-image / near-cusp / rotated-shear anchors, so the window
#: stops short of it.
W_DECADE_LO = 0.3
W_DECADE_HI = 30.0

#: Dense reconstruction grid for the GATE 1 identity.
DENSE_W_POINTS = 400

#: Dense TRUTH grid for the GATE 2 greedy node-count oracle, per brief.
GREEDY_TRUTH_POINTS = 506

#: GATE 1 relative reconstruction-identity ceiling.  The telescoping
#: carrier algebra is exact, so the measured worst is ~2e-15; this gate
#: sits ~two orders above it and fails only on a genuinely broken carrier.
RECONSTRUCTION_REL_GATE = 1e-13

#: GATE 2 greedy-oracle target and node ceiling.  The greedy achieves the
#: ``1e-3`` target with 17-21 nodes across the anchors; ``26`` leaves
#: ~25% headroom while still excluding a non-smooth envelope.
GREEDY_EPS_TARGET = 1e-3
GREEDY_NODE_CEILING = 26

#: GATE 4 ceiling on ``|S_a H_a|`` at a caustic crossing.  ``H_a``
#: diverges like ``sqrt|mu_a|`` at the caustic; the switch tames it to a
#: measured worst ~1.21 (the report cites <= 1.3), so ``2`` is a
#: deliberately conservative separator from the ~1e8 an unswitched kernel
#: reaches.
SADDLE_SWITCH_CEILING = 2.0

#: Source displacement off the caustic for the crossing scenarios, on
#: BOTH sides (the brief's ``eta = +/- 0.002``).
CROSSING_ETA = 0.002

#: The fold/cusp crossing scenarios: ``(label, gamma, theta_c, kind)``.
CROSSING_SCENARIOS = (
    ('fold gamma=0.20 theta=4.00', 0.20, 4.00, 'fold'),
    ('fold gamma=0.30 theta=2.50', 0.30, 2.50, 'fold'),
    ('cusp gamma=0.20 theta=pi',   0.20, float(np.pi), 'cusp'),
    ('cusp gamma=0.25 theta=0.00', 0.25, 0.00, 'cusp'),
    ('cusp gamma=0.20 theta=0.70', 0.20, 0.70, 'cusp'),
)

#: Frequency window for the crossing boundedness scan: wide enough that
#: the switch both shuts off (near the caustic) and ramps fully on.
W_CROSS = np.geomspace(0.3, 30.0, 120)

#: GATE 5 deep-band macro limit: a sheared positive-parity config and the
#: tiny-``w`` decades it is probed on (``kappa = 0`` so the closed form is
#: the pure-shear Gaussian magnification).
MACRO_GAMMA = 0.20
MACRO_KAPPA = 0.0
MACRO_SOURCE = (0.30, 0.10)
MACRO_TINY_W = np.array([1e-12, 1e-11, 1e-10, 1e-9, 1e-8])
MACRO_REL_GATE = 1e-6

#: F001 large-carrier-phase gate and the synthetic large delays that push
#: ``w*tau`` into the thousands of radians the physical anchors (whose
#: relative delays are ``O(0.1)``) never reach on this window.
CARRIER_REL_GATE = 1e-10
CARRIER_DELAYS = np.array([0.0, 137.24, 301.55, 499.91])
CARRIER_CRITICAL_DELAY = 71.4
CARRIER_SWITCH = np.array([1.0, 0.6, 0.3, 0.0])
CARRIER_W = np.geomspace(1.0, 30.0, 9)

#: Working precision of the mpmath carrier oracle (F001).  50 decimal
#: digits is ~166 bits, far beyond float64's 53, so the oracle's own
#: carrier is exact at the ~1e4 rad scale under test.
MPMATH_DPS = 50

#: Extreme-phase falsification point: ``w*tau ~ 1e12`` rad, where the
#: float64 ``w*tau`` product itself loses precision BEFORE any mod-2*pi
#: reduction can act.  The delays are irrational-valued on purpose: an
#: exact-power-of-ten product would be representable and lose nothing.
EXTREME_W = 1e12
EXTREME_DELAYS = np.array([0.0, 0.37, 0.61, 0.9])
EXTREME_CRITICAL_DELAY = 0.5
EXTREME_SWITCH = np.array([1.0, 0.6, 0.3, 0.0])

#: Names the GATE 4 crossing fixture is FORBIDDEN to reference, so it
#: cannot smuggle in the switch it is meant to be an independent oracle
#: for (F002).  ``geometry``, ``operator`` and ``_gauge`` are permitted.
CHANNELS_FORBIDDEN = frozenset(
    {'channels', 'ChangRefsdalChannels', 'ChangRefsdalPartition',
     'reconstruct_from_envelope', '_channel_switch', 'envelope_total'})

#: Names the F001 mpmath oracle is FORBIDDEN to reference, so it shares no
#: derivation with the range-reduced carrier under test: it must be PURE
#: mpmath, never numpy or the production reconstruction.
MPMATH_ORACLE_FORBIDDEN = frozenset(
    {'_gauge', 'channels', 'geometry', 'operator', 'np', 'numpy',
     'envelope_total', 'reconstructed_total', 'exp'})


def _referenced_names(func):
    """Return every name a function's own source references.

    Extends `_imported_top_level_modules` (the same `ast.Import` /
    `ast.ImportFrom` walk) with `ast.Name` ids and `ast.Attribute`
    attribute names, because a forbidden dependency would enter a helper
    as ``channels.ChangRefsdalChannels`` or a bare name rather than as an
    import statement inside the function.  Shared, verbatim, with
    `test_lensing_channels`.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split('.')[0])
                if alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or '').split('.')[0])
            for alias in node.names:
                names.add(alias.name)
                if alias.asname:
                    names.add(alias.asname)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    names.discard('')
    return names


def _evaluate_anchor(gamma, y, beta, kappa, w):
    """Evaluate a fresh, reset SACR-C partition at one anchor config.

    A new tracker per call with `reset` gives the deterministic initial
    labelling, so the returned partition is independent of evaluation
    history -- the tests must not depend on call order.
    """
    engine = channels.ChangRefsdalChannels(w)
    engine.reset()
    return engine.evaluate(gamma=gamma, y=list(y), beta=beta, kappa=kappa)


def _crossing_saddle_switch(gamma, theta_c, kind, sign, w,
                            *, beta=0.0, kappa=0.0):
    """SACR-C switch and saddle kernels at a caustic crossing.

    Built from `geometry`, `operator` and `_gauge` ONLY -- it never
    imports or calls `channels`, so it is an INDEPENDENT oracle for the
    switched-kernel boundedness the tracker claims (F002).
    `SacrcFixtureIndependenceTestCase` enforces that by AST inspection.

    The source is displaced ``sign * CROSSING_ETA`` off the caustic along
    the soft eigenvector (fold) or the outward hard eigenvector (cusp) of
    the critical point at ``theta_c``.  For each real image it returns the
    carrier-free analytic saddle kernel ``H_a`` (`geometry.image_kernel`)
    and the criticality-separation switch
    ``S_a = smootherstep(w*|tau_a - tau_c|, rho_start, rho_end)`` keyed on
    the delay of the near critical point ``tau_c`` -- the object whose
    shut-off tames the ``sqrt|mu_a|`` divergence of ``H_a``.

    Returns
    -------
    dict
        ``tau`` (relative image delays), ``tau_c`` (critical carrier
        delay), ``saddle`` and ``switch`` of shape ``(n_w, n_images)``,
        and ``n_images``.
    """
    critical = geometry.critical_point(gamma, theta_c, beta, kappa)
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    if kind == 'fold':
        axis = np.asarray(critical.soft_axis, dtype=float)
    else:
        axis = np.asarray(critical.hard_axis, dtype=float)
        if float(np.asarray(critical.image, dtype=float) @ axis) < 0.0:
            axis = -axis
    source = np.asarray(critical.source, dtype=float) + sign*CROSSING_ETA*axis
    images = geometry.find_images(source, matrix)
    absolute = np.array(
        [geometry.delay(image, source, matrix) for image in images],
        dtype=float)
    t_min = float(absolute.min())
    tau = absolute - t_min
    caustic = geometry.nearest_caustic_point(gamma, beta, source, kappa=kappa)
    tau_c = float(geometry.delay(caustic.image, source, matrix)) - t_min
    saddle = np.stack(
        [geometry.image_kernel(w, image, matrix) for image in images],
        axis=1)
    separation = np.abs(tau - tau_c)
    switch = _gauge.smootherstep(
        np.multiply.outer(w, separation),
        operator.RHO_START, operator.RHO_END)
    return {'tau': tau, 'tau_c': tau_c, 'saddle': saddle,
            'switch': switch, 'n_images': len(images)}


def _greedy_envelope_nodes(w, partition, target, cap=GREEDY_NODE_CEILING+16):
    """Hindsight greedy node oracle on the SACR-C envelope ``E(w)``.

    Repeatedly adds the node that maximally reduces the RECONSTRUCTION
    error of ``F`` -- interpolating ``E`` in ``ln w`` with a not-a-knot
    cubic spline and rebuilding ``F`` through the authoritative
    `_gauge.envelope_total` (saddle kernels and switch stay exact at every
    frequency; only ``E`` is approximated) -- until
    ``max|dF|/max|F| < target``.  Judged against the dense
    `ChangRefsdalPartition.exact_total`, an oracle independent of the
    interpolation, so the count reflects envelope SMOOTHNESS, not the
    production placement heuristic (GATE 2).

    Returns
    -------
    tuple
        ``(n_nodes, achieved_eps, history)`` where ``history`` is the list
        of ``(n_nodes, eps)`` along the greedy path.
    """
    lnw = np.log(w)
    truth = partition.envelope
    scale = float(np.max(np.abs(partition.exact_total)))
    n_points = w.size
    picked = [0, n_points - 1]
    history = []
    eps = np.inf
    for _ in range(cap):
        idx = sorted(set(picked))
        approx = (CubicSpline(lnw[idx], truth.real[idx])(lnw)
                  + 1j*CubicSpline(lnw[idx], truth.imag[idx])(lnw))
        recon = _gauge.envelope_total(
            w, partition.delays, partition.saddle_kernels,
            partition.switch, partition.critical_delay, approx)
        error = np.abs(recon - partition.exact_total)
        eps = float(np.max(error)) / scale
        history.append((len(idx), eps))
        if eps < target:
            return len(idx), eps, history
        picked.append(int(np.argmax(error)))
    return len(sorted(set(picked))), eps, history


def _mpmath_envelope_total(w, delays, saddle, switch, critical_delay,
                           envelope, dps):
    """High-precision SACR-C total, in PURE mpmath (F001 oracle).

    Evaluates ``sum_a exp(1j*w*tau_a)*S_a*H_a + exp(1j*w*tau_c)*E`` with
    the carrier arguments carried at ``dps`` decimal digits, so that no
    float64 large-argument phase loss can occur.  It references ONLY
    `mpmath` and builtins -- never numpy, `_gauge` or the production
    reconstruction -- so it shares no derivation with the range-reduced
    carrier under test (`MPMATH_ORACLE_FORBIDDEN`,
    `SacrcFixtureIndependenceTestCase`).
    """
    mpmath.mp.dps = dps
    imag_unit = mpmath.mpc(0, 1)
    total = mpmath.mpc(0)
    for index in range(len(delays)):
        phase = mpmath.mpf(float(w)) * mpmath.mpf(float(delays[index]))
        carrier = mpmath.e ** (imag_unit * phase)
        total += (carrier * mpmath.mpf(float(switch[index]))
                  * mpmath.mpc(complex(saddle[index])))
    phase_c = mpmath.mpf(float(w)) * mpmath.mpf(float(critical_delay))
    total += mpmath.e ** (imag_unit * phase_c) * mpmath.mpc(complex(envelope))
    return complex(total)


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


class SacrcTestCase(TestCase):
    """
    Base for the SACR-C reconstruction gates.

    Carries a seeded RNG and the anti-vacuity comparison counter shared
    with `ExactnessTestCase`: `tearDown` FAILS if no comparison ran, so a
    sweep that silently skips every case cannot read green.
    """

    def setUp(self):
        self.rng = np.random.default_rng(SEED)
        self._comparisons = 0

    def tearDown(self):
        self.assertGreater(
            self._comparisons, 0,
            'Vacuous SACR-C test: no comparison ran, so this test '
            'asserted nothing. Check the sweep bounds.')


class SacrcReconstructionIdentityTestCase(SacrcTestCase):
    """GATE 1: the SACR-C sum reproduces the exact total, algebraically."""

    def test_envelope_identity_matches_exact_total_on_every_anchor(self):
        """
        ``F = sum_a exp(i w tau_a) S_a H_a + exp(i w tau_c) E`` must equal
        `ChangRefsdalPartition.exact_total` (the untouched oracle) to
        ``1e-13`` relative on all five anchors.  Only ``E`` is ever
        approximated in production; the carrier algebra is exact, so the
        measured departure sits at ~2e-15 and this gate has ~two orders of
        headroom.
        """
        worst = 0.0
        records = []
        for label, gamma, y, beta, kappa in ANCHOR_CONFIGS:
            with self.subTest(anchor=label):
                w = np.geomspace(W_DECADE_LO, W_DECADE_HI, DENSE_W_POINTS)
                partition = _evaluate_anchor(gamma, y, beta, kappa, w)
                self.assertTrue(
                    np.all(partition.operator_converged),
                    f'{label}: operator did not converge on the window, '
                    'so exact_total is not a trustworthy oracle here')
                recon = _gauge.envelope_total(
                    w, partition.delays, partition.saddle_kernels,
                    partition.switch, partition.critical_delay,
                    partition.envelope)
                relative = (np.abs(recon - partition.exact_total)
                            / np.abs(partition.exact_total))
                self._comparisons += int(relative.size)
                peak = float(np.max(relative))
                worst = max(worst, peak)
                records.append((label, w, relative))
                self.assertLessEqual(
                    peak, RECONSTRUCTION_REL_GATE,
                    f'{label}: SACR-C reconstruction departs from '
                    f'exact_total by {peak:.2e}, above the identity gate '
                    f'{RECONSTRUCTION_REL_GATE:.0e}')
        self._plot(records, worst)

    def _plot(self, records, worst):
        if not _HAVE_MPL:
            return
        fig, ax = plt.subplots()
        for label, w, relative in records:
            ax.loglog(w, np.maximum(relative, 1e-18), label=label)
        ax.axhline(RECONSTRUCTION_REL_GATE, color='k', ls='--',
                   label=f'gate {RECONSTRUCTION_REL_GATE:.0e}')
        ax.set_xlabel('w'); ax.set_ylabel('|F_recon - F_exact| / |F_exact|')
        ax.set_title(f'GATE 1 reconstruction identity (worst {worst:.1e})')
        ax.legend(fontsize=6)
        _savefig(fig, 'gate1_reconstruction_identity.png')


class GreedyEnvelopeNodeCountTestCase(SacrcTestCase):
    """GATE 2: the envelope is smooth enough for a small node budget."""

    def test_greedy_oracle_reaches_target_within_node_ceiling(self):
        """
        A hindsight greedy on ``E`` must reach ``max|dF|/max|F| < 1e-3``
        with ``<= 26`` nodes on every anchor.  This isolates envelope
        SMOOTHNESS from the production placement heuristic: if the
        envelope itself needed a large budget, no placement rule could be
        cheap.  Measured 17-21.
        """
        curves = []
        for label, gamma, y, beta, kappa in ANCHOR_CONFIGS:
            with self.subTest(anchor=label):
                w = np.geomspace(W_DECADE_LO, W_DECADE_HI,
                                 GREEDY_TRUTH_POINTS)
                partition = _evaluate_anchor(gamma, y, beta, kappa, w)
                n_nodes, eps, history = _greedy_envelope_nodes(
                    w, partition, GREEDY_EPS_TARGET)
                self._comparisons += 1
                curves.append((label, history))
                self.assertLess(
                    eps, GREEDY_EPS_TARGET,
                    f'{label}: greedy oracle stalled at eps={eps:.2e}, '
                    f'never reaching the target {GREEDY_EPS_TARGET:.0e}')
                self.assertLessEqual(
                    n_nodes, GREEDY_NODE_CEILING,
                    f'{label}: greedy needed {n_nodes} nodes to reach '
                    f'{GREEDY_EPS_TARGET:.0e}, above the ceiling '
                    f'{GREEDY_NODE_CEILING}; the envelope is not as smooth '
                    'as the SACR-C demodulation claims')
        self._plot(curves)

    def _plot(self, curves):
        if not _HAVE_MPL:
            return
        fig, ax = plt.subplots()
        for label, history in curves:
            counts, epsilons = zip(*history)
            ax.semilogy(counts, epsilons, marker='.', label=label)
        ax.axhline(GREEDY_EPS_TARGET, color='k', ls='--',
                   label=f'target {GREEDY_EPS_TARGET:.0e}')
        ax.axvline(GREEDY_NODE_CEILING, color='r', ls=':',
                   label=f'ceiling {GREEDY_NODE_CEILING}')
        ax.set_xlabel('nodes N'); ax.set_ylabel('max|dF| / max|F|')
        ax.set_title('GATE 2 greedy-oracle envelope node count')
        ax.legend(fontsize=6)
        _savefig(fig, 'gate2_greedy_envelope_nodes.png')


class SaddleSwitchBoundednessTestCase(SacrcTestCase):
    """
    GATE 4: ``|S_a H_a|`` stays bounded at fold and cusp crossings.

    The headline claim.  ``H_a`` diverges like ``sqrt|mu_a|`` at the
    caustic; the criticality-separation switch shuts off at exactly the
    rate that keeps the product ``<= 2`` (measured ~1.3).  The fixture is
    an INDEPENDENT reconstruction of ``S_a`` and ``H_a`` from geometry --
    never `channels` -- so a passing gate certifies the tracker's switch
    rather than restating it.
    """

    def test_switched_saddle_kernels_are_bounded_at_crossings(self):
        worst = 0.0
        diagnostics = []
        for label, gamma, theta_c, kind in CROSSING_SCENARIOS:
            for sign in (+1, -1):
                with self.subTest(scenario=label, sign=sign):
                    fixture = _crossing_saddle_switch(
                        gamma, theta_c, kind, sign, W_CROSS)
                    product = np.abs(fixture['switch'] * fixture['saddle'])
                    self._comparisons += int(product.size)
                    peak = float(np.max(product))
                    worst = max(worst, peak)
                    diagnostics.append((label, sign, product))
                    self.assertLessEqual(
                        peak, SADDLE_SWITCH_CEILING,
                        f'{label} sign={sign:+d}: max|S_a H_a|={peak:.3f} '
                        f'exceeds the boundedness ceiling '
                        f'{SADDLE_SWITCH_CEILING}; the switch is not '
                        'shutting off fast enough near the caustic')
        self._plot(diagnostics, worst)

    def test_unswitched_saddle_kernels_are_genuinely_singular(self):
        """
        Non-vacuity control: the caustic really IS singular, so the
        boundedness above is the switch doing work, not ``H`` being small.
        The bare kernel must exceed ``1e6`` at ``eta = 0.002`` from the
        fold.
        """
        fixture = _crossing_saddle_switch(0.20, 4.00, 'fold', +1, W_CROSS)
        bare = float(np.max(np.abs(fixture['saddle'])))
        self._comparisons += 1
        self.assertGreater(
            bare, 1e6,
            f'the near-critical saddle kernel peaks at only {bare:.2e}; '
            'if H is not large, the GATE 4 boundedness gate is vacuous')

    def _plot(self, diagnostics, worst):
        if not _HAVE_MPL:
            return
        fig, ax = plt.subplots()
        for label, sign, product in diagnostics:
            ax.semilogx(W_CROSS, np.max(product, axis=1),
                        label=f'{label} {sign:+d}')
        ax.axhline(SADDLE_SWITCH_CEILING, color='k', ls='--',
                   label=f'ceiling {SADDLE_SWITCH_CEILING}')
        ax.set_xlabel('w'); ax.set_ylabel('max_a |S_a H_a|')
        ax.set_title(f'GATE 4 switched-kernel boundedness (worst {worst:.2f})')
        ax.legend(fontsize=5)
        _savefig(fig, 'gate4_saddle_switch_boundedness.png')


class DeepBandMacroLimitTestCase(SacrcTestCase):
    """GATE 5: the deep-band (w -> 0) macro-magnification limit (F009)."""

    def test_reconstruction_tends_to_the_gaussian_macro_limit(self):
        """
        As ``w -> 0`` the reconstruction must tend to the LITERAL Gaussian
        closed form ``1/sqrt((1-kappa)**2 - gamma**2)`` -- computed here
        from the shear and convergence alone, NOT from ``F_op`` / channels
        / geometry (the F002 oracle-tautology trap) -- to ``1e-6``
        relative, and FLAT across three decades.  A ``1/w`` prefactor leak
        would show as a slope rather than a plateau.
        """
        closed_form = 1.0 / np.sqrt((1.0 - MACRO_KAPPA)**2 - MACRO_GAMMA**2)
        partition = _evaluate_anchor(
            MACRO_GAMMA, MACRO_SOURCE, 0.0, MACRO_KAPPA, MACRO_TINY_W)
        recon = _gauge.envelope_total(
            MACRO_TINY_W, partition.delays, partition.saddle_kernels,
            partition.switch, partition.critical_delay, partition.envelope)
        magnitudes = np.abs(recon)
        relative = np.abs(magnitudes - closed_form) / closed_form
        self._comparisons += int(relative.size)
        for w_value, magnitude, rel in zip(MACRO_TINY_W, magnitudes,
                                            relative):
            with self.subTest(w=w_value):
                self.assertLess(
                    rel, MACRO_REL_GATE,
                    f'w={w_value:.0e}: |F_recon|={magnitude:.8f} departs '
                    f'from the macro limit {closed_form:.8f} by {rel:.2e}, '
                    f'above {MACRO_REL_GATE:.0e}')
        spread = float(np.ptp(magnitudes) / np.mean(magnitudes))
        self.assertLess(
            spread, MACRO_REL_GATE,
            f'the macro-limit plateau varies by {spread:.2e} across three '
            'decades of w; a flat plateau is required, so a 1/w prefactor '
            'may be leaking into the reconstruction')
        self._plot(magnitudes, closed_form)

    def _plot(self, magnitudes, closed_form):
        if not _HAVE_MPL:
            return
        fig, ax = plt.subplots()
        ax.semilogx(MACRO_TINY_W, magnitudes, marker='o', label='|F_recon|')
        ax.axhline(closed_form, color='k', ls='--',
                   label=f'1/sqrt((1-k)^2 - g^2) = {closed_form:.6f}')
        ax.set_xlabel('w'); ax.set_ylabel('|F_recon|')
        ax.set_title('GATE 5 deep-band macro-magnification plateau')
        ax.legend(fontsize=7)
        _savefig(fig, 'gate5_macro_limit_plateau.png')


class LargeCarrierPhaseTestCase(SacrcTestCase):
    """F001: carrier-phase correctness at thousands of radians."""

    def test_range_reduced_carriers_match_mpmath_at_large_phase(self):
        """
        With synthetic delays up to ``~500`` and ``w`` up to ``30``, the
        carrier argument ``w*tau`` reaches ``~1.5e4`` rad -- far beyond the
        physical anchors' ``O(0.1)`` delays.  The range-reduced float64
        reconstruction must still agree with the independent mpmath
        ``exp(1j w tau)`` reference to ``1e-10`` relative, i.e. no
        large-argument float64 phase loss.  Measured ~5e-13.
        """
        saddle = _random_complex(self.rng, CARRIER_DELAYS.size)
        envelope = _random_complex(self.rng, CARRIER_W.size)
        recon = _gauge.envelope_total(
            CARRIER_W, CARRIER_DELAYS,
            np.repeat(saddle[None, :], CARRIER_W.size, axis=0),
            np.repeat(CARRIER_SWITCH[None, :], CARRIER_W.size, axis=0),
            CARRIER_CRITICAL_DELAY, envelope)
        worst = 0.0
        errors = []
        for index, w_value in enumerate(CARRIER_W):
            reference = _mpmath_envelope_total(
                float(w_value), CARRIER_DELAYS, saddle, CARRIER_SWITCH,
                CARRIER_CRITICAL_DELAY, envelope[index], MPMATH_DPS)
            rel = abs(recon[index] - reference) / abs(reference)
            self._comparisons += 1
            worst = max(worst, rel)
            errors.append((float(w_value) * float(CARRIER_DELAYS.max()),
                           rel))
            with self.subTest(w=float(w_value)):
                self.assertLessEqual(
                    rel, CARRIER_REL_GATE,
                    f'w={w_value:.3f} (w*tau~{w_value*CARRIER_DELAYS.max():.0f}'
                    f' rad): carrier reconstruction differs from mpmath by '
                    f'{rel:.2e}, above {CARRIER_REL_GATE:.0e}')
        self._plot(errors, worst)

    def _plot(self, errors, worst):
        if not _HAVE_MPL:
            return
        fig, ax = plt.subplots()
        phases, rels = zip(*errors)
        ax.loglog(phases, np.maximum(rels, 1e-18), marker='.')
        ax.axhline(CARRIER_REL_GATE, color='k', ls='--',
                   label=f'gate {CARRIER_REL_GATE:.0e}')
        ax.set_xlabel('w * max|tau| (rad)')
        ax.set_ylabel('|F_recon - F_mpmath| / |F_mpmath|')
        ax.set_title(f'F001 carrier-phase accuracy (worst {worst:.1e})')
        ax.legend(fontsize=7)
        _savefig(fig, 'f001_carrier_phase_accuracy.png')


class SacrcFixtureIndependenceTestCase(TestCase):
    """
    The GATE 4 fixture and the F001 oracle must be independent of the
    code they judge (F002).

    A crossing fixture built from the tracker COULD NOT FAIL, and a
    carrier oracle sharing the range-reduced arithmetic would restate the
    bug rather than catch it.  Both are pinned by AST inspection of the
    helper's own source.
    """

    def test_crossing_fixture_never_references_channels(self):
        names = _referenced_names(_crossing_saddle_switch)
        for forbidden in CHANNELS_FORBIDDEN:
            with self.subTest(name=forbidden):
                self.assertNotIn(
                    forbidden, names,
                    f'the GATE 4 crossing fixture references {forbidden!r}; '
                    'a fixture built from the tracker it tests is a '
                    'tautology, not an oracle (F002)')

    def test_crossing_fixture_is_built_from_the_permitted_primitives(self):
        names = _referenced_names(_crossing_saddle_switch)
        self.assertIn(
            'geometry', names,
            'the GATE 4 fixture must build H_a and the crossing from '
            'geometry; if it does not, it is not exercising the physics')
        self.assertIn(
            '_gauge', names,
            'the GATE 4 fixture must build S_a from the shared _gauge '
            'smootherstep, not a private copy that could drift')

    def test_mpmath_oracle_shares_no_derivation_with_the_reconstruction(self):
        names = _referenced_names(_mpmath_envelope_total)
        self.assertIn(
            'mpmath', names,
            'the F001 oracle must actually use mpmath; a float64 oracle '
            'would inherit the same large-argument phase loss it checks')
        for forbidden in MPMATH_ORACLE_FORBIDDEN:
            with self.subTest(name=forbidden):
                self.assertNotIn(
                    forbidden, names,
                    f'the F001 mpmath oracle references {forbidden!r}; it '
                    'must be pure mpmath so it shares no derivation with '
                    'the carrier under test (F002)')


class SacrcSelfFalsificationTestCase(SacrcTestCase):
    """
    Proof the SACR-C layer can go RED.

    Each gate is broken deliberately: a green suite here means every gate
    above is load-bearing, not a rubber stamp.
    """

    def test_forced_switch_breaks_the_boundedness_gate(self):
        """GATE 4: ``S_a = 1`` re-exposes the ``sqrt|mu_a|`` divergence."""
        fixture = _crossing_saddle_switch(0.20, 4.00, 'fold', +1, W_CROSS)
        forced = np.abs(np.ones_like(fixture['switch']) * fixture['saddle'])
        self._comparisons += 1
        self.assertGreater(
            float(np.max(forced)), SADDLE_SWITCH_CEILING * 1e5,
            'forcing S=1 did not blow |S H| past the ceiling; the GATE 4 '
            'boundedness gate cannot detect a switch that fails to shut off')

    def test_perturbed_envelope_breaks_the_reconstruction_identity(self):
        """GATE 1: a ``1e-6`` envelope error must exceed the ``1e-13`` gate."""
        label, gamma, y, beta, kappa = ANCHOR_CONFIGS[1]
        w = np.geomspace(W_DECADE_LO, W_DECADE_HI, DENSE_W_POINTS)
        partition = _evaluate_anchor(gamma, y, beta, kappa, w)
        corrupted = partition.envelope * (1.0 + 1e-6)
        recon = _gauge.envelope_total(
            w, partition.delays, partition.saddle_kernels,
            partition.switch, partition.critical_delay, corrupted)
        relative = float(np.max(np.abs(recon - partition.exact_total)
                                / np.abs(partition.exact_total)))
        self._comparisons += 1
        self.assertGreater(
            relative, RECONSTRUCTION_REL_GATE,
            f'a 1e-6 envelope perturbation moved the reconstruction by only '
            f'{relative:.2e}; the GATE 1 identity is not sensitive to E')

    def test_extreme_phase_breaks_carrier_reconstruction(self):
        """
        F001: at ``w*tau ~ 1e12`` rad the float64 ``w*tau`` product itself
        loses precision before any mod-2*pi reduction, so the
        reconstruction must diverge from mpmath by MORE than the ``1e-10``
        gate -- proving the positive F001 result is a real accuracy claim,
        not an unfalsifiable one.
        """
        saddle = _random_complex(self.rng, EXTREME_DELAYS.size)
        envelope = 0.4 + 0.1j
        recon = _gauge.envelope_total(
            np.array([EXTREME_W]), EXTREME_DELAYS,
            saddle[None, :], EXTREME_SWITCH[None, :],
            EXTREME_CRITICAL_DELAY, np.array([envelope]))[0]
        reference = _mpmath_envelope_total(
            EXTREME_W, EXTREME_DELAYS, saddle, EXTREME_SWITCH,
            EXTREME_CRITICAL_DELAY, envelope, MPMATH_DPS)
        relative = abs(recon - reference) / abs(reference)
        self._comparisons += 1
        self.assertGreater(
            relative, CARRIER_REL_GATE,
            f'at w*tau~1e12 rad the float64 reconstruction still agrees '
            f'with mpmath to {relative:.2e}; the F001 gate would then be '
            'unable to detect large-argument phase loss')


if __name__ == '__main__':
    main()
