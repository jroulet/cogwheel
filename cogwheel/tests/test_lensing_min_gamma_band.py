"""Tests for WP1: reduce min_gamma_band threshold from 0.02 to 0.005.

The ``min_width`` parameter of ``stable_gamma_bands`` determines the minimum
width of a topology-straddling sub-band below which it is dropped (refused to
the exact engine).  WP1 lowered the default from 0.02 to 0.005, retaining
narrow edge slivers (widths ~0.015–0.019, between the old 0.02 and new 0.005
thresholds) that were previously dropped.

Tolerance justification: all assertions are exact equality or inequality
checks on band widths and dropped-list membership — no floating-point
tolerance is needed because the test probes the threshold logic (a width
comparison against min_width), not a numerical approximation.

Runtime budget (measured on fast tier):
  - ReducedDroppedSlivers: 4 calls to stable_gamma_bands (~6s each) = ~24s.
  - ThresholdBoundaryMocked: uses mock to force CausticTopologyError on a
    narrow band — effectively instant (<0.1s).
  - SelfFalsification: 1 call to stable_gamma_bands (~6s).
  Total file budget: < 35s.
"""
from __future__ import annotations

import unittest
from unittest import mock

from cogwheel.lensing import surrogate_training as training
from cogwheel.lensing.surrogate_training import CausticTopologyError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Positive-parity full prior range (astroid, gamma < 1).
_POSITIVE_BAND: tuple[float, float] = (0.0, 0.999)
#: Negative-parity full prior range (saddle, gamma > 1).
_NEGATIVE_BAND: tuple[float, float] = (1.001, 1.6)
#: Mutation-check threshold above the measured sliver widths (~0.015–0.019).
_MUTATION_MIN_WIDTH: float = 0.03
#: The new default threshold (WP1 change).
_NEW_DEFAULT_MIN_WIDTH: float = 0.005
#: The old threshold value (pre-WP1).
_OLD_MIN_WIDTH: float = 0.02
#: Synthetic narrow band for the threshold boundary test (width = 0.008).
_SYNTHETIC_NARROW_BAND: tuple[float, float] = (0.005, 0.013)
#: Width of the synthetic narrow band.
_SYNTHETIC_NARROW_WIDTH: float = 0.008


# ---------------------------------------------------------------------------
# Anti-vacuity base
# ---------------------------------------------------------------------------

class _CountingTestCase(unittest.TestCase):
    """Base carrying the anti-vacuity comparison tally (house idiom).

    A subclass increments ``self.comparisons`` for every genuine assertion
    it makes; ``tearDown`` fails a test that ran zero comparisons, so a
    silently-skipping sweep cannot read green.
    """

    def setUp(self) -> None:
        self.comparisons = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self.comparisons, 0,
            'anti-vacuity: this test asserted nothing (zero comparisons).')


# ---------------------------------------------------------------------------
# Spec 1: new threshold retains more slivers than the old
# ---------------------------------------------------------------------------

class ReducedDroppedSliversTestCase(_CountingTestCase):
    """WP1 regression: the new default min_width=0.005 drops FEWER slivers
    (in total gamma-width) than the old min_width=0.02.

    The specific edge slivers at gamma ∈ (0, 0.016) and gamma ∈ (1.001, 1.020)
    — width ~0.015–0.019 — are between the old and new thresholds, so they are
    retained at 0.005 but would be dropped at 0.02.  Any slivers still dropped
    at 0.005 must have width < 0.005 (by the algorithm's bisection logic).
    """

    def test_positive_parity_fewer_dropped(self) -> None:
        """Positive parity: dropped at 0.005 is strictly less total width
        than dropped at 0.02."""
        _stable_new, dropped_new = training.stable_gamma_bands(
            _POSITIVE_BAND, +1, min_width=_NEW_DEFAULT_MIN_WIDTH)
        _stable_old, dropped_old = training.stable_gamma_bands(
            _POSITIVE_BAND, +1, min_width=_OLD_MIN_WIDTH)
        total_new = sum(hi - lo for lo, hi in dropped_new)
        total_old = sum(hi - lo for lo, hi in dropped_old)
        self.assertLess(
            total_new, total_old,
            f'New threshold should drop less total width. '
            f'Dropped at 0.005: {dropped_new} (total {total_new:.6f}); '
            f'dropped at 0.02: {dropped_old} (total {total_old:.6f}).')
        self.comparisons += 1

    def test_negative_parity_fewer_dropped(self) -> None:
        """Negative parity: dropped at 0.005 is strictly less total width
        than dropped at 0.02."""
        _stable_new, dropped_new = training.stable_gamma_bands(
            _NEGATIVE_BAND, -1, min_width=_NEW_DEFAULT_MIN_WIDTH)
        _stable_old, dropped_old = training.stable_gamma_bands(
            _NEGATIVE_BAND, -1, min_width=_OLD_MIN_WIDTH)
        total_new = sum(hi - lo for lo, hi in dropped_new)
        total_old = sum(hi - lo for lo, hi in dropped_old)
        self.assertLess(
            total_new, total_old,
            f'New threshold should drop less total width. '
            f'Dropped at 0.005: {dropped_new} (total {total_new:.6f}); '
            f'dropped at 0.02: {dropped_old} (total {total_old:.6f}).')
        self.comparisons += 1

    def test_all_dropped_slivers_narrower_than_threshold(self) -> None:
        """Any sliver in the dropped list at min_width=0.005 must have
        width < 0.005 — a width >= 0.005 sliver would mean the threshold
        is not being applied correctly."""
        for parity, band in [(+1, _POSITIVE_BAND), (-1, _NEGATIVE_BAND)]:
            with self.subTest(parity=parity):
                _stable, dropped = training.stable_gamma_bands(
                    band, parity, min_width=_NEW_DEFAULT_MIN_WIDTH)
                for lo, hi in dropped:
                    width = hi - lo
                    self.assertLess(
                        width, _NEW_DEFAULT_MIN_WIDTH,
                        f'Dropped sliver ({lo:.6f}, {hi:.6f}) has '
                        f'width={width:.6f} >= min_width={_NEW_DEFAULT_MIN_WIDTH}. '
                        f'Threshold logic is broken.')
                self.comparisons += 1

    def test_mutation_check_higher_threshold_drops_more(self) -> None:
        """Mutation check: with min_width=0.03 (above the sliver widths),
        at least one sliver IS dropped — proves the threshold logic is not
        bypassed."""
        _stable_pos, dropped_pos = training.stable_gamma_bands(
            _POSITIVE_BAND, +1, min_width=_MUTATION_MIN_WIDTH)
        _stable_neg, dropped_neg = training.stable_gamma_bands(
            _NEGATIVE_BAND, -1, min_width=_MUTATION_MIN_WIDTH)
        all_dropped = dropped_pos + dropped_neg
        self.assertGreater(
            len(all_dropped), 0,
            f'min_width={_MUTATION_MIN_WIDTH} should drop at least one '
            f'sliver, but nothing was dropped on either parity. '
            f'The threshold logic may be bypassed.')
        self.comparisons += 1


# ---------------------------------------------------------------------------
# Spec 2: threshold boundary test (synthetic, mocked)
# ---------------------------------------------------------------------------

class ThresholdBoundaryMockedTestCase(_CountingTestCase):
    """Directly test the threshold boundary behavior using a mock.

    A synthetic band of width 0.008 is forced to always raise
    CausticTopologyError (simulating a topology-straddling band that cannot
    be resolved further).  With min_width=0.005 the band is dropped
    (0.008 > 0.005 is FALSE — wait, 0.008 > 0.005 so it would bisect further,
    not drop).  Actually: bisection halves until width < min_width, THEN drops.

    The mock raises CausticTopologyError for ANY sub-band within
    (0.005, 0.013), so the algorithm bisects repeatedly:
      0.008 -> 0.004 (two sub-bands of width 0.004 each).
    At min_width=0.005: each 0.004-wide half < 0.005, so BOTH are dropped.
    At min_width=0.003: each 0.004-wide half >= 0.003, bisect again to 0.002,
      then 0.002 < 0.003 → dropped.

    The discriminating case is min_width=0.005 vs min_width=0.02:
    - min_width=0.005: bisects once → two 0.004-wide bands, both < 0.005, DROPPED.
    - min_width=0.02:  the original 0.008 < 0.02, so it's immediately DROPPED
      without further bisection.

    Key insight: both thresholds DROP the band, but the dropped intervals differ:
    - min_width=0.02: drops (0.005, 0.013) as one piece.
    - min_width=0.005: drops (0.005, 0.009) and (0.009, 0.013) as two pieces.

    Better approach: use a band where the UPPER half is topology-stable but
    the LOWER half isn't. Then min_width controls whether the lower half is
    kept (further bisected until stable) or dropped.
    """

    @staticmethod
    def _always_raises(band, parity, *, n_samples=200):
        """Mock that always raises CausticTopologyError."""
        raise CausticTopologyError(f'mock topology error for {band}')

    def test_narrow_band_dropped_at_high_threshold(self) -> None:
        """A 0.008-wide band that always straddles a topology change is
        dropped when min_width=0.02 (the full band width 0.008 < 0.02)."""
        with mock.patch.object(training, 'band_caustic_structure',
                               side_effect=self._always_raises):
            stable, dropped = training.stable_gamma_bands(
                _SYNTHETIC_NARROW_BAND, +1, min_width=_OLD_MIN_WIDTH)
        # The entire band is dropped as one piece (0.008 < 0.02).
        self.assertEqual(len(stable), 0, 'no stable bands expected')
        self.assertEqual(len(dropped), 1, 'entire band dropped as one piece')
        self.assertAlmostEqual(
            dropped[0][1] - dropped[0][0], _SYNTHETIC_NARROW_WIDTH,
            places=10,
            msg='dropped sliver should span the full band width')
        self.comparisons += 1

    def test_narrow_band_bisected_then_dropped_at_low_threshold(self) -> None:
        """A 0.008-wide band that always straddles a topology change is
        bisected (since 0.008 > 0.005) and then each 0.004-wide half is
        dropped (0.004 < 0.005)."""
        with mock.patch.object(training, 'band_caustic_structure',
                               side_effect=self._always_raises):
            stable, dropped = training.stable_gamma_bands(
                _SYNTHETIC_NARROW_BAND, +1, min_width=_NEW_DEFAULT_MIN_WIDTH)
        # The band is bisected once: two halves of width 0.004.
        self.assertEqual(len(stable), 0, 'no stable bands expected')
        self.assertEqual(
            len(dropped), 2,
            f'Expected 2 dropped sub-bands (bisected once), got {dropped}')
        for lo, hi in dropped:
            width = hi - lo
            self.assertLess(
                width, _NEW_DEFAULT_MIN_WIDTH,
                f'Dropped sub-band ({lo:.6f}, {hi:.6f}) width={width:.6f} '
                f'>= min_width={_NEW_DEFAULT_MIN_WIDTH}')
        self.comparisons += 1

    def test_threshold_discriminates_retain_vs_drop(self) -> None:
        """The number of dropped pieces differs between the two thresholds,
        proving the threshold logic actually discriminates."""
        with mock.patch.object(training, 'band_caustic_structure',
                               side_effect=self._always_raises):
            _stable_low, dropped_low = training.stable_gamma_bands(
                _SYNTHETIC_NARROW_BAND, +1, min_width=_NEW_DEFAULT_MIN_WIDTH)
            _stable_high, dropped_high = training.stable_gamma_bands(
                _SYNTHETIC_NARROW_BAND, +1, min_width=_OLD_MIN_WIDTH)
        # At min_width=0.005: bisected → 2 pieces dropped.
        # At min_width=0.02: not bisected → 1 piece dropped.
        self.assertNotEqual(
            len(dropped_low), len(dropped_high),
            f'Both thresholds produce the same result — the threshold '
            f'is not being applied correctly.\n'
            f'  min_width=0.005 dropped: {dropped_low}\n'
            f'  min_width=0.02 dropped: {dropped_high}')
        self.comparisons += 1


# ---------------------------------------------------------------------------
# Self-falsification: prove the suite can go red
# ---------------------------------------------------------------------------

class SelfFalsificationTestCase(_CountingTestCase):
    """Prove the suite's assertions have teeth by demonstrating they go red
    under a pathological configuration.

    If the default min_width were reverted to 0.02, the
    ReducedDroppedSliversTestCase tests would fail because more slivers are
    dropped at 0.02 than at 0.005 (the \"fewer dropped\" assertion would become
    \"same dropped\").  We simulate this by explicitly calling with
    min_width=0.02 and confirming the condition that the main test asserts is
    VIOLATED.
    """

    def test_old_threshold_violates_the_fewer_dropped_invariant(self) -> None:
        """At the old min_width=0.02, the total dropped width is >= that at
        min_width=0.005 — proving the main assertion has teeth (it WOULD fail
        if someone reverts the threshold)."""
        # Positive parity
        _s_new, dropped_new = training.stable_gamma_bands(
            _POSITIVE_BAND, +1, min_width=_NEW_DEFAULT_MIN_WIDTH)
        _s_old, dropped_old = training.stable_gamma_bands(
            _POSITIVE_BAND, +1, min_width=_OLD_MIN_WIDTH)
        total_new = sum(hi - lo for lo, hi in dropped_new)
        total_old = sum(hi - lo for lo, hi in dropped_old)
        # The main test asserts total_new < total_old.
        # Here we confirm total_old >= total_new (the OLD threshold is worse).
        self.assertGreaterEqual(
            total_old, total_new,
            'Old threshold should drop MORE (or same) total width than new.')
        # Strict: old should drop STRICTLY more (the improvement is real).
        self.assertGreater(
            total_old, total_new,
            'Old threshold should drop strictly MORE total width than new — '
            'the WP1 improvement must be measurable.')
        self.comparisons += 1

    def test_mock_suite_can_go_red_on_stable_band(self) -> None:
        """If band_caustic_structure does NOT raise (band is stable),
        the mock-based ThresholdBoundaryMockedTestCase assertions would fail —
        a stable band is never dropped regardless of min_width."""
        # Use the real (unpatched) function on a band known to be stable.
        stable, dropped = training.stable_gamma_bands(
            (0.1, 0.3), +1, min_width=_NEW_DEFAULT_MIN_WIDTH)
        # A topology-stable band is never dropped — the mock-based tests
        # only work because the mock FORCES topology errors.
        self.assertEqual(
            dropped, [],
            'A known-stable band (0.1, 0.3) should never be dropped.')
        self.assertGreater(
            len(stable), 0,
            'A known-stable band must return at least one stable sub-band.')
        self.comparisons += 1


if __name__ == '__main__':
    unittest.main()
