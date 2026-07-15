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
"""

import ast
import itertools
import pathlib
from unittest import TestCase, main

import numpy as np

from cogwheel.lensing.chang_refsdal import _gauge


#: Float64 machine epsilon; the roundoff unit of every bound here.
EPS = np.finfo(np.float64).eps

#: Slack over the roundoff model in the scale-aware reconstruction
#: bound, per the build plan. The measured margin is ~30x at the worst
#: configuration tested, so this is not a fitted constant.
RECONSTRUCTION_SLACK = 100.0

#: Documented default switch window, shared with the channel tracker.
RHO_START = 0.5
RHO_END = 4.0

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


if __name__ == '__main__':
    main()
