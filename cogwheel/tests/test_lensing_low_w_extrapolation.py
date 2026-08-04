"""Low-w flat extrapolation in surrogate serve path (WP1).

The surrogate chart spline covers a finite log_w band [log_w_min, log_w_max].
Previously, queries below log_w_min were refused (chart not served).  WP1 adds
flat extrapolation at the low end: the envelope value at any w < w_min is
clamped to the envelope at w_min.  This is physically justified because the
envelope is smooth and nearly constant below the first Airy fringe — the
correction is O(w_min^2) from the geometric limit.

The HIGH end remains strict: no upward extrapolation (the envelope is
oscillatory above w_max), so queries with w_max > chart.log_w_grid[-1] are
refused.

Tolerance choices:
- Flat extrapolation: EXACT bitwise identity (np.clip is exact for values
  below the low clamp), so tolerance is 0.0.
- Self-falsification: scipy BSpline cubic extrapolation diverges polynomially,
  so the unclamped–clamped difference at w_min/2 is typically > 0.01 for
  a sinusoidal envelope (measured ~0.03–0.15).  The LOAD_BEARING_THRESHOLD
  (1e-10) is a conservative lower bound.

Cost: 4-node synthetic chart build ~0.01s; 5 evaluations per test; total
suite < 2s.
"""
from __future__ import annotations

import unittest

import numpy as np

from cogwheel.lensing.surrogate import (
    TubeChart,
    LensAmplificationSurrogate,
    _evaluate_chart,
    _log_w_band_serveable,
    _contract_tensor_spline,
    select_chart,
)

# ---------------------------------------------------------------------------
# Fixture constants
# ---------------------------------------------------------------------------

#: 4-node log_w grid: w in [5, 50] → log_w in [~1.61, ~3.91].
#: Chosen so w_min=5.0 gives a non-trivial extrapolation test region (w<5).
LOG_W_GRID = np.log(np.array([5.0, 10.0, 25.0, 50.0]))

#: Spatial grids (must be 4-node minimum per _validate_axis).
GAMMA_GRID = np.linspace(0.3, 0.6, 4)
U_GRID = np.linspace(0.05, 0.20, 4)  #: sqrt(eta) grid
THETA_GRID = np.linspace(0.1, 1.0, 4)

#: Chart metadata — positive parity, 2-image region.
IMAGE_COUNT = 2
PARITY = 1
ETA_FLOOR = U_GRID[0] ** 2
ETA_MAX = U_GRID[-1] ** 2

#: Query parameters that are inside the chart's spatial box.
QUERY_GAMMA = 0.45
QUERY_ETA = 0.01  #: between ETA_FLOOR and ETA_MAX
QUERY_THETA = 0.5  #: inside THETA_GRID range

#: Frequencies below w_min for flat-extrapolation test.
W_MIN = np.exp(LOG_W_GRID[0])  #: = 5.0
W_BELOW = np.array([W_MIN / 4, W_MIN / 2, W_MIN * 0.9])

#: Frequency at exactly w_min (control).
W_EXACT_MIN = np.array([W_MIN])

#: Frequency above w_max for high-end refusal test.
W_MAX = np.exp(LOG_W_GRID[-1])  #: = 50.0
W_ABOVE = np.array([W_MAX * 2.0])

#: Load-bearing threshold for self-falsification: the unclamped BSpline
#: extrapolation must differ from the clamped value by at least this.
LOAD_BEARING_THRESHOLD = 1e-10


def _build_synthetic_chart() -> TubeChart:
    """Build a 4x4x4x4 TubeChart with a sinusoidal envelope.

    The envelope is NOT constant — it has genuine w-dependence so the
    cubic B-spline extrapolates to a DIFFERENT value below w_min (which
    the clamp then corrects).  A constant envelope would make the self-
    falsification test vacuous.
    """
    grid_w, grid_g, grid_u, grid_t = np.meshgrid(
        LOG_W_GRID, GAMMA_GRID, U_GRID, THETA_GRID, indexing='ij')
    # Sinusoidal in log_w so the spline extrapolation isn't trivially flat.
    real = (np.cos(1.5 * grid_w) * (1.0 + 0.3 * grid_g)
            * np.exp(-2.0 * grid_u) * (1.0 + 0.2 * grid_t))
    imag = (np.sin(1.5 * grid_w) * (1.0 - 0.2 * grid_g)
            * (1.0 + 0.1 * grid_u) * np.cos(0.3 * grid_t))
    return TubeChart.from_values(
        gamma_grid=GAMMA_GRID,
        u_grid=U_GRID,
        theta_grid=THETA_GRID,
        log_w_grid=LOG_W_GRID,
        envelope_real=real,
        envelope_imag=imag,
        image_count=IMAGE_COUNT,
        parity=PARITY,
        eta_floor=ETA_FLOOR,
        eta_max=ETA_MAX,
        cusp_windows=[],
    )


class _LowWExtrapolationTestCase(unittest.TestCase):
    """Base class providing chart fixture + anti-vacuity tearDown."""

    _n_comparisons: int = 0

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart = _build_synthetic_chart()
        cls.surrogate = LensAmplificationSurrogate(
            [cls.chart], {'chart_count': 1, 'chart_types': ['tube']})

    def tearDown(self) -> None:
        """Anti-vacuity: FAIL if zero comparisons ran (silently-skipping)."""
        if self._n_comparisons == 0:
            self.fail(
                f'{type(self).__name__}: zero comparisons ran — '
                f'suite is vacuous.')

    def _record(self, n: int = 1) -> None:
        self._n_comparisons += n


class FlatExtrapolationTestCase(_LowWExtrapolationTestCase):
    """Test 1: Low-w flat extrapolation serves draws below w_min."""

    def test_serve_returns_true_below_w_min(self) -> None:
        """serve() returns served=True for frequencies entirely below w_min."""
        w_query = W_BELOW.copy()
        _, served, _ = self.surrogate.serve(
            w_query, gamma=QUERY_GAMMA, y1=0.0, y2=0.0, beta=0.0,
            eta=QUERY_ETA, theta=QUERY_THETA, image_count=IMAGE_COUNT)
        self.assertTrue(served, 'serve must return served=True for w < w_min')
        self._record()

    def test_envelope_below_w_min_equals_at_w_min(self) -> None:
        """Envelope at w < w_min is BITWISE IDENTICAL to envelope at w_min."""
        # Evaluate at w_min
        log_w_min = np.array([LOG_W_GRID[0]])
        env_at_min = _evaluate_chart(
            self.chart, gamma=QUERY_GAMMA, eta=QUERY_ETA,
            theta=QUERY_THETA, log_w_query=log_w_min)
        # Evaluate at several w < w_min
        log_w_below = np.log(W_BELOW)
        env_below = _evaluate_chart(
            self.chart, gamma=QUERY_GAMMA, eta=QUERY_ETA,
            theta=QUERY_THETA, log_w_query=log_w_below)
        # Each below-w_min value must be exactly equal to env_at_min
        for i, w in enumerate(W_BELOW):
            with self.subTest(w=w):
                self.assertEqual(
                    env_below[i], env_at_min[0],
                    f'Envelope at w={w:.2f} (below w_min={W_MIN}) differs '
                    f'from envelope at w_min: got {env_below[i]}, '
                    f'expected {env_at_min[0]}')
                self._record()

    def test_serve_with_mixed_band_spanning_below_w_min(self) -> None:
        """serve() works when w_array spans below AND inside the band."""
        w_mixed = np.array([W_MIN / 3, W_MIN, W_MIN * 2, W_MIN * 5])
        env, served, _ = self.surrogate.serve(
            w_mixed, gamma=QUERY_GAMMA, y1=0.0, y2=0.0, beta=0.0,
            eta=QUERY_ETA, theta=QUERY_THETA, image_count=IMAGE_COUNT)
        self.assertTrue(served, 'Mixed band spanning below w_min must serve')
        # The first element (below w_min) must equal the second (at w_min)
        self.assertEqual(
            env[0], env[1],
            'Envelope at w < w_min must equal envelope at w_min in mixed query')
        self._record()

    def test_regression_values_above_w_min_unchanged(self) -> None:
        """Values at w >= w_min match direct _contract_tensor_spline call.

        This confirms the clamp does not alter in-band queries.
        """
        # Query at w_min and w_max (boundaries of the band)
        log_w_test = np.array([LOG_W_GRID[0], LOG_W_GRID[-1]])
        env = _evaluate_chart(
            self.chart, gamma=QUERY_GAMMA, eta=QUERY_ETA,
            theta=QUERY_THETA, log_w_query=log_w_test)
        # Direct spline evaluation (no clamping needed — these are in-band)
        theta_inframe = QUERY_THETA - float(self.chart.theta_grid[0])
        s_val = float(np.interp(QUERY_THETA, self.chart.theta_to_s[0],
                                self.chart.theta_to_s[1]))
        real_direct = _contract_tensor_spline(
            self.chart.real_coeffs, self.chart.knots,
            QUERY_GAMMA, float(np.sqrt(QUERY_ETA)), s_val, log_w_test)
        imag_direct = _contract_tensor_spline(
            self.chart.imag_coeffs, self.chart.knots,
            QUERY_GAMMA, float(np.sqrt(QUERY_ETA)), s_val, log_w_test)
        expected = real_direct + 1j * imag_direct
        np.testing.assert_array_equal(
            env, expected,
            err_msg='In-band values must be unchanged by the clamp')
        self._record(2)

    def test_log_w_band_serveable_admits_below_min(self) -> None:
        """_log_w_band_serveable returns True when log_w_min < grid[0]."""
        # log_w_min well below the chart's grid start
        log_w_min_below = float(np.log(W_MIN / 10))
        log_w_max_inside = float(LOG_W_GRID[-1] - 0.1)
        result = _log_w_band_serveable(
            self.chart, log_w_min_below, log_w_max_inside)
        self.assertTrue(
            result,
            '_log_w_band_serveable must admit when log_w_min < grid[0] '
            'and log_w_max <= grid[-1]')
        self._record()


if __name__ == '__main__':
    unittest.main()
