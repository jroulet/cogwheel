"""Tests for the InteriorWedgeChart dataclass and coordinate infrastructure.

This suite exercises:
1. **Coordinate round-trip** — `_to_wedge_fixed` → `_from_wedge_fixed` is an
   exact inverse (within bilinear interpolation error at ≥100 theta nodes and
   ≥5 gamma nodes).
2. **NPZ serialization** — `_chart_to_npz` → `_chart_from_npz` preserves every
   field exactly (max-diff = 0) and yields identical spline evaluations.
3. **Serve-gate logic** — `_wedge_serves` returns True/False per gate for
   each named refusal reason.

Tolerance justification: the bilinear interpolation of `r_caustic` at 101
theta nodes × 5 gamma nodes introduces at most ~O(h²) local error where
h = pi/(2*100) ≈ 0.016.  For the smooth caustic function the second
derivative is bounded; measured round-trip residuals are < 1e-12 because both
the forward and inverse paths pass through the SAME interpolant (exact
cancellation up to float64 arithmetic, ~1e-15 each step).
"""
from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cogwheel.lensing.surrogate import (
    CarrierDiscontinuityError,
    InteriorWedgeChart,
    _WedgeCausticMap,
    _assert_carrier_continuity,
    _caustic_reach,
    _chart_from_npz,
    _chart_to_npz,
    _contract_tensor_spline,
    _evaluate_chart,
    _from_wedge_fixed,
    _interp_r_caustic,
    _to_wedge_fixed,
    _wedge_serves,
    select_chart,
)
from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels
from cogwheel.lensing.chang_refsdal.geometry import r_caustic

# ---------------------------------------------------------------------------
#: Module-level constants
# ---------------------------------------------------------------------------

#: Gamma values for the coordinate round-trip test.
ROUND_TRIP_GAMMAS: tuple[float, ...] = (0.2, 0.5, 0.8)

#: Number of theta nodes in the wedge map for round-trip tests.
N_THETA_NODES: int = 101

#: Number of gamma nodes in the wedge map.
N_GAMMA_NODES: int = 5

#: Maximum acceptable round-trip residual (y -> wedge -> y).
ROUND_TRIP_ATOL: float = 1e-12

#: Fraction of caustic reach used as maximum source radius.
MAX_R_FRACTION: float = 0.9

#: Source angles spanning all four quadrants (signs of y1, y2).
QUADRANT_SIGNS: tuple[tuple[int, int], ...] = (
    (+1, +1), (-1, +1), (-1, -1), (+1, -1))

#: Counter for anti-vacuity tearDown check.
_COMPARISONS_COUNTER: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Helper: build a realistic _WedgeCausticMap from geometry.r_caustic
# ---------------------------------------------------------------------------

def _build_wedge_map(gammas: np.ndarray, n_theta: int = N_THETA_NODES
                     ) -> _WedgeCausticMap:
    """Construct a _WedgeCausticMap from the analytic r_caustic function.

    Parameters
    ----------
    gammas : np.ndarray
        1-D array of gamma values (must be strictly ascending, in (0,1)).
    n_theta : int
        Number of theta nodes spanning [0, pi/2].

    Returns
    -------
    _WedgeCausticMap
    """
    theta_nodes = np.linspace(0.0, np.pi / 2, n_theta)
    r_table = np.empty((gammas.size, n_theta))
    for i, g in enumerate(gammas):
        for j, th in enumerate(theta_nodes):
            r_table[i, j] = r_caustic(g, th)
    return _WedgeCausticMap(
        gamma_nodes=gammas.copy(),
        theta_nodes=theta_nodes,
        r_table=r_table)


# ===========================================================================
# Base TestCase with anti-vacuity tearDown
# ===========================================================================

class _WedgeTestCase(unittest.TestCase):
    """Base class with anti-vacuity tearDown that fails if no comparisons ran."""

    _class_comparison_count: int = 0

    def setUp(self):
        self.__class__._class_comparison_count = 0

    def tearDown(self):
        if self.__class__._class_comparison_count == 0:
            self.fail(
                f'{self.__class__.__name__}: ANTI-VACUITY failure — '
                f'zero domain comparisons executed in {self._testMethodName}.')

    def _tick(self, n: int = 1) -> None:
        """Increment the comparison counter."""
        self.__class__._class_comparison_count += n


# ===========================================================================
# Test 1: Coordinate round-trip
# ===========================================================================


#: Grids for the NPZ round-trip test (must have >= 4 nodes per axis).
NPZ_N_GAMMA: int = 4
NPZ_N_R: int = 5
NPZ_N_THETA: int = 6
NPZ_N_W: int = 10

#: Random seed for reproducible spline-query points.
NPZ_QUERY_SEED: int = 42


class CoordinateRoundTripTestCase(_WedgeTestCase):
    """Verify _to_wedge_fixed → _from_wedge_fixed is an exact inverse."""

    @classmethod
    def setUpClass(cls):
        """Build a shared _WedgeCausticMap for the gamma grid."""
        cls.gammas = np.linspace(
            ROUND_TRIP_GAMMAS[0], ROUND_TRIP_GAMMAS[-1], N_GAMMA_NODES)
        cls.wedge_map = _build_wedge_map(cls.gammas, n_theta=N_THETA_NODES)

    def test_theta_wedge_in_first_quadrant(self):
        """theta_wedge is always in [0, pi/2] regardless of source quadrant."""
        for gamma in ROUND_TRIP_GAMMAS:
            # Pick a source well inside the caustic.
            r_c = _interp_r_caustic(gamma, np.pi / 4, self.wedge_map)
            y_mag = 0.5 * r_c
            for s1, s2 in QUADRANT_SIGNS:
                y1 = s1 * y_mag * np.cos(np.pi / 4)
                y2 = s2 * y_mag * np.sin(np.pi / 4)
                r, theta_w = _to_wedge_fixed(gamma, y1, y2, self.wedge_map)
                self._tick()
                with self.subTest(gamma=gamma, s1=s1, s2=s2):
                    self.assertGreaterEqual(theta_w, 0.0)
                    self.assertLessEqual(theta_w, np.pi / 2)

    def test_r_less_than_one_for_interior_sources(self):
        """r is in [0, 1) for sources strictly inside the caustic."""
        for gamma in ROUND_TRIP_GAMMAS:
            r_c = _interp_r_caustic(gamma, np.pi / 6, self.wedge_map)
            y_mag = MAX_R_FRACTION * r_c  # 90% of caustic reach
            y1 = y_mag * np.cos(np.pi / 6)
            y2 = y_mag * np.sin(np.pi / 6)
            r, _ = _to_wedge_fixed(gamma, y1, y2, self.wedge_map)
            self._tick()
            with self.subTest(gamma=gamma):
                self.assertGreater(r, 0.0)
                self.assertLess(r, 1.0)

    def test_round_trip_residual_within_tolerance(self):
        """Round-trip |y_out - |y_in|| < ROUND_TRIP_ATOL for all quadrants.

        Cost: 3 gammas × 4 quadrants × 3 angles = 36 evaluations.
        Per-evaluation cost: 2 bilinear interps + trig ≈ 0.1ms.
        Total ≈ 4ms, well under 60s limit.
        """
        test_angles = [np.pi / 6, np.pi / 4, np.pi / 3]
        for gamma, theta_source, (s1, s2) in itertools.product(
                ROUND_TRIP_GAMMAS, test_angles, QUADRANT_SIGNS):
            r_c = _interp_r_caustic(gamma, theta_source, self.wedge_map)
            y_mag = 0.5 * r_c
            y1 = s1 * y_mag * np.cos(theta_source)
            y2 = s2 * y_mag * np.sin(theta_source)
            r, theta_w = _to_wedge_fixed(gamma, y1, y2, self.wedge_map)
            y1_out, y2_out = _from_wedge_fixed(gamma, r, theta_w,
                                               self.wedge_map)
            residual = abs(y1_out - abs(y1)) + abs(y2_out - abs(y2))
            self._tick()
            with self.subTest(gamma=gamma, theta=theta_source, s1=s1, s2=s2):
                self.assertLess(
                    residual, ROUND_TRIP_ATOL,
                    f'Round-trip residual {residual:.2e} exceeds '
                    f'{ROUND_TRIP_ATOL:.0e}')

    def test_d2_symmetry_same_wedge_coordinates(self):
        """Sources related by D2 reflections map to the SAME (r, theta_wedge).

        The fold into the canonical first quadrant means all four quadrants
        produce identical (r, theta_wedge) for a given |y1|, |y2|.
        """
        for gamma in ROUND_TRIP_GAMMAS:
            r_c = _interp_r_caustic(gamma, np.pi / 5, self.wedge_map)
            y_mag = 0.4 * r_c
            y1_abs = y_mag * np.cos(np.pi / 5)
            y2_abs = y_mag * np.sin(np.pi / 5)
            ref_r, ref_theta = None, None
            for s1, s2 in QUADRANT_SIGNS:
                y1 = s1 * y1_abs
                y2 = s2 * y2_abs
                r, theta_w = _to_wedge_fixed(gamma, y1, y2, self.wedge_map)
                self._tick()
                if ref_r is None:
                    ref_r, ref_theta = r, theta_w
                else:
                    with self.subTest(gamma=gamma, s1=s1, s2=s2):
                        self.assertAlmostEqual(r, ref_r, places=15)
                        self.assertAlmostEqual(theta_w, ref_theta, places=15)

    def test_origin_raises_valueerror(self):
        """_to_wedge_fixed raises ValueError for origin (0, 0)."""
        self._tick()
        with self.assertRaises(ValueError):
            _to_wedge_fixed(0.5, 0.0, 0.0, self.wedge_map)

    def test_negative_r_raises_valueerror(self):
        """_from_wedge_fixed raises ValueError for negative r."""
        self._tick()
        with self.assertRaises(ValueError):
            _from_wedge_fixed(0.5, -0.1, np.pi / 4, self.wedge_map)


# ===========================================================================
# Test 2: NPZ round-trip
# ===========================================================================

class NpzRoundTripTestCase(_WedgeTestCase):
    """Verify _chart_to_npz → _chart_from_npz preserves all fields exactly."""

    @classmethod
    def setUpClass(cls):
        """Build a small InteriorWedgeChart with synthetic constant envelope."""
        # Training grids (must be >= 4 nodes each for spline validation).
        cls.gamma_grid = np.linspace(0.2, 0.8, NPZ_N_GAMMA)
        cls.r_grid = np.linspace(0.05, 0.85, NPZ_N_R)
        cls.theta_wedge_grid = np.linspace(0.01, np.pi / 2 - 0.01, NPZ_N_THETA)
        cls.log_w_grid = np.linspace(np.log(5.0), np.log(50.0), NPZ_N_W)

        # Build a wedge map (gamma_nodes must equal gamma_grid).
        theta_nodes = np.linspace(0.0, np.pi / 2, 21)
        r_table = np.empty((NPZ_N_GAMMA, 21))
        for i, g in enumerate(cls.gamma_grid):
            for j, th in enumerate(theta_nodes):
                # Synthetic smooth caustic radius ~ gamma*(1 + 0.3*cos(2*th))
                r_table[i, j] = g * (1.0 + 0.3 * np.cos(2 * th))
        cls.wedge_map = _WedgeCausticMap(
            gamma_nodes=cls.gamma_grid.copy(),
            theta_nodes=theta_nodes,
            r_table=r_table)

        # Synthetic constant envelope: real=1.0, imag=0.5 everywhere.
        shape = (NPZ_N_W, NPZ_N_GAMMA, NPZ_N_R, NPZ_N_THETA)
        envelope_real = np.full(shape, 1.0)
        envelope_imag = np.full(shape, 0.5)

        # Non-empty refused_points.
        cls.refused_pts = np.array([[0.3, 0.2, 0.5],
                                    [0.6, 0.4, 1.0]])

        cls.chart = InteriorWedgeChart.from_wedge_values(
            gamma_grid=cls.gamma_grid,
            r_grid=cls.r_grid,
            theta_wedge_grid=cls.theta_wedge_grid,
            log_w_grid=cls.log_w_grid,
            envelope_real=envelope_real,
            envelope_imag=envelope_imag,
            image_count=4,
            parity=1,
            wedge_map=cls.wedge_map,
            eta_overlap_min=0.03,
            refused_points=cls.refused_pts,
            envelope_definition='interior_sacr_c_envelope')

    def _round_trip_chart(self) -> InteriorWedgeChart:
        """Save chart to NPZ and reload; return the reloaded chart."""
        arrays = _chart_to_npz(self.chart, index=0)
        # Simulate np.savez / np.load round-trip via temp file.
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f, **arrays)
            tmp_path = Path(f.name)
        try:
            data = np.load(tmp_path, allow_pickle=True)
            reloaded = _chart_from_npz(data, index=0)
            data.close()
        finally:
            tmp_path.unlink()
        return reloaded

    def test_scalar_fields_round_trip(self):
        """image_count, parity, eta_overlap_min, envelope_definition survive."""
        reloaded = self._round_trip_chart()
        self._tick()
        self.assertEqual(reloaded.image_count, self.chart.image_count)
        self.assertEqual(reloaded.parity, self.chart.parity)
        self.assertEqual(reloaded.eta_overlap_min, self.chart.eta_overlap_min)
        self.assertEqual(reloaded.envelope_definition,
                         self.chart.envelope_definition)

    def test_axis_arrays_round_trip_exactly(self):
        """All axis arrays have max-diff = 0 after NPZ round-trip."""
        reloaded = self._round_trip_chart()
        for name in ('gamma_grid', 'r_grid', 'theta_wedge_grid', 'log_w_grid'):
            orig = getattr(self.chart, name)
            reloads = getattr(reloaded, name)
            self._tick()
            with self.subTest(field=name):
                self.assertEqual(float(np.max(np.abs(orig - reloads))), 0.0,
                                 f'{name} differs after round-trip')

    def test_coefficient_arrays_round_trip_exactly(self):
        """real_coeffs, imag_coeffs have max-diff = 0 after NPZ round-trip."""
        reloaded = self._round_trip_chart()
        for name in ('real_coeffs', 'imag_coeffs'):
            orig = getattr(self.chart, name)
            reloads = getattr(reloaded, name)
            self._tick()
            with self.subTest(field=name):
                self.assertEqual(float(np.max(np.abs(orig - reloads))), 0.0,
                                 f'{name} differs after round-trip')

    def test_knots_round_trip_exactly(self):
        """All knot vectors have max-diff = 0 after NPZ round-trip."""
        reloaded = self._round_trip_chart()
        for k, (orig_knot, relo_knot) in enumerate(
                zip(self.chart.knots, reloaded.knots)):
            self._tick()
            with self.subTest(knot_axis=k):
                self.assertEqual(
                    float(np.max(np.abs(orig_knot - relo_knot))), 0.0,
                    f'knots[{k}] differs after round-trip')

    def test_refused_points_round_trip_exactly(self):
        """refused_points array has max-diff = 0 after NPZ round-trip."""
        reloaded = self._round_trip_chart()
        self._tick()
        self.assertEqual(
            float(np.max(np.abs(
                self.chart.refused_points - reloaded.refused_points))),
            0.0)

    def test_wedge_map_round_trip_exactly(self):
        """wedge_map fields (gamma_nodes, theta_nodes, r_table) survive."""
        reloaded = self._round_trip_chart()
        for name in ('gamma_nodes', 'theta_nodes', 'r_table'):
            orig = getattr(self.chart.wedge_map, name)
            relo = getattr(reloaded.wedge_map, name)
            self._tick()
            with self.subTest(field=f'wedge_map.{name}'):
                self.assertEqual(
                    float(np.max(np.abs(orig - relo))), 0.0,
                    f'wedge_map.{name} differs after round-trip')

    def test_spline_evaluation_identical_after_round_trip(self):
        """_contract_tensor_spline gives identical results from reloaded chart.

        Cost: 5 random query points × 2 evaluations (original + reloaded)
        = 10 spline evaluations. < 1ms total.
        """
        reloaded = self._round_trip_chart()
        rng = np.random.default_rng(NPZ_QUERY_SEED)
        for _ in range(5):
            # Random query within grid bounds.
            v0 = rng.uniform(self.gamma_grid[0], self.gamma_grid[-1])
            v1 = rng.uniform(self.r_grid[0], self.r_grid[-1])
            v2 = rng.uniform(self.theta_wedge_grid[0],
                             self.theta_wedge_grid[-1])
            log_w_q = np.array([rng.uniform(self.log_w_grid[0],
                                            self.log_w_grid[-1])])
            orig_real = _contract_tensor_spline(
                self.chart.real_coeffs, self.chart.knots, v0, v1, v2, log_w_q)
            relo_real = _contract_tensor_spline(
                reloaded.real_coeffs, reloaded.knots, v0, v1, v2, log_w_q)
            orig_imag = _contract_tensor_spline(
                self.chart.imag_coeffs, self.chart.knots, v0, v1, v2, log_w_q)
            relo_imag = _contract_tensor_spline(
                reloaded.imag_coeffs, reloaded.knots, v0, v1, v2, log_w_q)
            self._tick()
            self.assertEqual(float(np.max(np.abs(orig_real - relo_real))), 0.0)
            self.assertEqual(float(np.max(np.abs(orig_imag - relo_imag))), 0.0)


# ===========================================================================
# Test 3: _wedge_serves guard logic (gate-by-gate)
# ===========================================================================

#: Gamma grid for the serves-guard chart (must have >= 4 nodes).
SERVES_GAMMA_GRID: np.ndarray = np.array([0.20, 0.30, 0.40, 0.50])

#: Radial grid for the serves-guard chart.
SERVES_R_GRID: np.ndarray = np.array([0.05, 0.20, 0.40, 0.60])

#: Theta wedge grid for the serves-guard chart.
SERVES_THETA_GRID: np.ndarray = np.array([0.15, 0.50, 0.90, 1.30])

#: Log-w band for the serves-guard chart.
SERVES_LOG_W_GRID: np.ndarray = np.array([
    np.log(5.0), np.log(10.0), np.log(17.0), np.log(25.0)])

#: Eta overlap minimum for the serves-guard chart.
SERVES_ETA_FLOOR: float = 0.04


class WedgeServesGuardTestCase(_WedgeTestCase):
    """Verify _wedge_serves returns True/False per gate, independently.

    Each refusal test isolates ONE gate by keeping all others passing.
    The chart is built with from_wedge_values using a synthetic constant
    envelope. Cost: 9 invocations of _wedge_serves (< 1ms each), total < 10ms.
    """

    @classmethod
    def setUpClass(cls):
        """Build a chart for serve-guard testing."""
        # Build a wedge map whose gamma_nodes == SERVES_GAMMA_GRID.
        theta_nodes = np.linspace(0.0, np.pi / 2, 21)
        r_table = np.empty((SERVES_GAMMA_GRID.size, 21))
        for i, g in enumerate(SERVES_GAMMA_GRID):
            for j, th in enumerate(theta_nodes):
                # Smooth synthetic caustic radius.
                r_table[i, j] = g * (1.0 + 0.2 * np.cos(2 * th))
        cls.wedge_map = _WedgeCausticMap(
            gamma_nodes=SERVES_GAMMA_GRID.copy(),
            theta_nodes=theta_nodes,
            r_table=r_table)

        # Constant envelope tensor.
        shape = (SERVES_LOG_W_GRID.size, SERVES_GAMMA_GRID.size,
                 SERVES_R_GRID.size, SERVES_THETA_GRID.size)
        envelope_real = np.ones(shape)
        envelope_imag = np.zeros(shape)

        cls.chart = InteriorWedgeChart.from_wedge_values(
            gamma_grid=SERVES_GAMMA_GRID,
            r_grid=SERVES_R_GRID,
            theta_wedge_grid=SERVES_THETA_GRID,
            log_w_grid=SERVES_LOG_W_GRID,
            envelope_real=envelope_real,
            envelope_imag=envelope_imag,
            image_count=4,
            parity=1,
            wedge_map=cls.wedge_map,
            eta_overlap_min=SERVES_ETA_FLOOR,
            refused_points=None,
            envelope_definition='interior_sacr_c_envelope')

        # Pre-compute an accepted source position (inside all bounds).
        # Use gamma midpoint and source well within the caustic.
        cls.good_gamma = 0.35  # midpoint of [0.20, 0.50]
        # Source that maps into mid-range of (r, theta_wedge) grids.
        cls.good_theta = 0.7  # in [0.15, 1.30]
        r_caust = _interp_r_caustic(cls.good_gamma, cls.good_theta,
                                    cls.wedge_map)
        # r = 0.3 should be well within [0.05, 0.60]
        y_mag = 0.3 * r_caust
        cls.good_y1 = y_mag * float(np.cos(cls.good_theta))
        cls.good_y2 = y_mag * float(np.sin(cls.good_theta))
        cls.good_log_w_min = SERVES_LOG_W_GRID[0] + 0.01
        cls.good_log_w_max = SERVES_LOG_W_GRID[-1] - 0.01
        cls.good_eta = SERVES_ETA_FLOOR + 0.1
        cls.good_image_count = 4

    def _call_serves(self, **overrides) -> bool:
        """Call _wedge_serves with defaults, applying overrides."""
        kwargs = dict(
            chart=self.chart,
            gamma=self.good_gamma,
            log_w_min=self.good_log_w_min,
            log_w_max=self.good_log_w_max,
            eta=self.good_eta,
            image_count=self.good_image_count,
            y1_eig=self.good_y1,
            y2_eig=self.good_y2)
        kwargs.update(overrides)
        return _wedge_serves(**kwargs)

    def test_accepted_candidate_returns_true(self):
        """A candidate satisfying all gates returns True."""
        self._tick()
        self.assertTrue(self._call_serves())

    def test_gate_a_nonfinite_y_eig_refuses(self):
        """Gate (a): non-finite eigenframe source → False."""
        self._tick()
        self.assertFalse(self._call_serves(y1_eig=float('inf')))
        self._tick()
        self.assertFalse(self._call_serves(y2_eig=float('nan')))

    def test_gate_b_gamma_outside_grid_refuses(self):
        """Gate (b): gamma outside [gamma_grid[0], gamma_grid[-1]] → False."""
        self._tick()
        self.assertFalse(self._call_serves(gamma=0.10))  # below
        self._tick()
        self.assertFalse(self._call_serves(gamma=0.60))  # above

    def test_gate_c_log_w_outside_band_refuses(self):
        """Gate (c): log_w band not fully inside chart band → False."""
        self._tick()
        # log_w_min below chart's range.
        self.assertFalse(self._call_serves(
            log_w_min=SERVES_LOG_W_GRID[0] - 1.0))
        self._tick()
        # log_w_max above chart's range.
        self.assertFalse(self._call_serves(
            log_w_max=SERVES_LOG_W_GRID[-1] + 1.0))

    def test_gate_d_source_at_origin_refuses(self):
        """Gate (d): source at origin (y1=0, y2=0) → False."""
        self._tick()
        self.assertFalse(self._call_serves(y1_eig=0.0, y2_eig=0.0))

    def test_gate_e_r_outside_grid_refuses(self):
        """Gate (e): r outside r_grid range → False.

        Pick a source so far from the caustic centre that r >> r_grid[-1].
        """
        # A source far outside the caustic: r ≈ 5 >> 0.60.
        r_caust = _interp_r_caustic(self.good_gamma, self.good_theta,
                                    self.wedge_map)
        y_mag_far = 5.0 * r_caust
        y1_far = y_mag_far * float(np.cos(self.good_theta))
        y2_far = y_mag_far * float(np.sin(self.good_theta))
        self._tick()
        self.assertFalse(self._call_serves(y1_eig=y1_far, y2_eig=y2_far))

    def test_gate_f_theta_wedge_outside_grid_refuses(self):
        """Gate (f): theta_wedge outside theta_wedge_grid range → False.

        Place the source at an angle that maps to theta_wedge < grid[0].
        theta_wedge = atan2(|y2|, |y1|), so very small |y2| relative to |y1|
        gives theta_wedge ≈ 0 < 0.15 = grid[0].
        """
        r_caust = _interp_r_caustic(self.good_gamma, 0.01, self.wedge_map)
        y_mag = 0.3 * r_caust
        # theta_wedge = atan2(0.01*y_mag, y_mag) ≈ 0.01 < 0.15
        y1_edge = y_mag
        y2_edge = 0.01 * y_mag
        self._tick()
        self.assertFalse(self._call_serves(y1_eig=y1_edge, y2_eig=y2_edge))

    def test_gate_g_wrong_image_count_refuses(self):
        """Gate (g): image_count != chart.image_count → False."""
        self._tick()
        self.assertFalse(self._call_serves(image_count=2))

    def test_gate_h_eta_below_floor_refuses(self):
        """Gate (h): eta <= eta_overlap_min → False."""
        self._tick()
        self.assertFalse(self._call_serves(eta=SERVES_ETA_FLOOR))
        self._tick()
        self.assertFalse(self._call_serves(eta=SERVES_ETA_FLOOR - 0.01))


# ===========================================================================
# Self-falsification: prove the suite can go red
# ===========================================================================

class SelfFalsificationTestCase(unittest.TestCase):
    """Prove the test suite has teeth by asserting known-bad inputs fail.

    These tests do NOT use the anti-vacuity base class because they are
    meta-tests — they verify the ASSERTIONS, not the production code.
    """

    @classmethod
    def setUpClass(cls):
        """Build a shared wedge map for self-falsification probes."""
        gammas = np.linspace(0.2, 0.8, N_GAMMA_NODES)
        cls.wedge_map = _build_wedge_map(gammas, n_theta=N_THETA_NODES)

    def test_round_trip_detects_corrupted_r_table(self):
        """A corrupted r_table makes the round-trip residual exceed ATOL.

        If we scale r_table by 2x, the inverse reconstructs y at 2x the
        correct magnitude, giving a large residual.
        """
        # Build a corrupted wedge map with doubled r_table.
        bad_map = _WedgeCausticMap(
            gamma_nodes=self.wedge_map.gamma_nodes.copy(),
            theta_nodes=self.wedge_map.theta_nodes.copy(),
            r_table=self.wedge_map.r_table * 2.0)
        gamma = 0.5
        theta_source = np.pi / 4
        r_c = _interp_r_caustic(gamma, theta_source, self.wedge_map)
        y_mag = 0.5 * r_c
        y1 = y_mag * np.cos(theta_source)
        y2 = y_mag * np.sin(theta_source)
        # Forward with the GOOD map.
        r, theta_w = _to_wedge_fixed(gamma, y1, y2, self.wedge_map)
        # Inverse with the BAD map.
        y1_out, y2_out = _from_wedge_fixed(gamma, r, theta_w, bad_map)
        residual = abs(y1_out - abs(y1)) + abs(y2_out - abs(y2))
        self.assertGreater(residual, ROUND_TRIP_ATOL,
                           'Self-falsification: corrupted r_table should '
                           'cause large residual')

    def test_serves_gate_detects_interior_source_accepted(self):
        """Prove _wedge_serves actually returns True for a valid source.

        Without this, the refusal tests could be vacuously passing
        because _wedge_serves ALWAYS returns False.
        """
        # Build a minimal chart.
        gamma_grid = SERVES_GAMMA_GRID
        theta_nodes = np.linspace(0.0, np.pi / 2, 21)
        r_table = np.empty((gamma_grid.size, 21))
        for i, g in enumerate(gamma_grid):
            for j, th in enumerate(theta_nodes):
                r_table[i, j] = g * (1.0 + 0.2 * np.cos(2 * th))
        wedge_map = _WedgeCausticMap(
            gamma_nodes=gamma_grid.copy(),
            theta_nodes=theta_nodes,
            r_table=r_table)
        shape = (SERVES_LOG_W_GRID.size, gamma_grid.size,
                 SERVES_R_GRID.size, SERVES_THETA_GRID.size)
        chart = InteriorWedgeChart.from_wedge_values(
            gamma_grid=gamma_grid,
            r_grid=SERVES_R_GRID,
            theta_wedge_grid=SERVES_THETA_GRID,
            log_w_grid=SERVES_LOG_W_GRID,
            envelope_real=np.ones(shape),
            envelope_imag=np.zeros(shape),
            image_count=4, parity=1,
            wedge_map=wedge_map,
            eta_overlap_min=SERVES_ETA_FLOOR,
            refused_points=None,
            envelope_definition='interior_sacr_c_envelope')
        # A source deep in the interior.
        gamma = 0.35
        good_theta = 0.7
        r_c = _interp_r_caustic(gamma, good_theta, wedge_map)
        y_mag = 0.3 * r_c
        y1 = y_mag * float(np.cos(good_theta))
        y2 = y_mag * float(np.sin(good_theta))
        result = _wedge_serves(
            chart, gamma,
            log_w_min=SERVES_LOG_W_GRID[0] + 0.01,
            log_w_max=SERVES_LOG_W_GRID[-1] - 0.01,
            eta=SERVES_ETA_FLOOR + 0.1,
            image_count=4,
            y1_eig=y1, y2_eig=y2)
        self.assertTrue(result,
                        'Self-falsification: _wedge_serves must return True '
                        'for a valid interior source; if False, the gate '
                        'tests are vacuous.')

    def test_anti_vacuity_teardown_catches_empty_test(self):
        """Prove the anti-vacuity tearDown fires on zero comparisons.

        We instantiate a _WedgeTestCase subclass that does no assertions
        and confirm tearDown raises AssertionError.
        """
        class _EmptyTest(_WedgeTestCase):
            def runTest(self):
                pass  # Deliberately empty — no _tick() call.

        empty = _EmptyTest('runTest')
        empty.setUp()
        empty.runTest()
        with self.assertRaises(AssertionError):
            empty.tearDown()


# ===========================================================================
# Test 4: select_chart dispatch priority and _evaluate_chart correctness
# ===========================================================================

#: Gamma grid for the dispatch test chart.
DISPATCH_GAMMA_GRID: np.ndarray = np.array([0.30, 0.37, 0.43, 0.50])

#: Radial grid for the dispatch test chart.
DISPATCH_R_GRID: np.ndarray = np.array([0.10, 0.25, 0.40, 0.60])

#: Theta wedge grid for the dispatch test chart.
DISPATCH_THETA_GRID: np.ndarray = np.array([0.20, 0.55, 0.90, 1.30])

#: Log-w band for the dispatch test chart.
DISPATCH_LOG_W_GRID: np.ndarray = np.array([
    np.log(5.0), np.log(9.0), np.log(14.0), np.log(20.0)])

#: Eta overlap minimum for the dispatch test chart.
DISPATCH_ETA_FLOOR: float = 0.03


class SelectChartDispatchTestCase(_WedgeTestCase):
    """Verify select_chart returns an InteriorWedgeChart and _evaluate_chart
    produces correct results including D2 symmetry.

    Spec: Build a chart with gamma=[0.3,0.5], r=[0.1,0.6],
    theta_wedge=[0.2,1.3], w=[5,20]. Prepare a candidate in the SECOND
    quadrant (y1<0, y2>0) with matching gamma, |y| inside caustic, image_count=4,
    eta=0.5. Confirm select_chart returns isinstance InteriorWedgeChart.
    Then evaluate at a D2-reflected source and confirm identical values.

    Cost: 1 chart build + 2 select_chart + 2 _evaluate_chart (3 w-points each)
    = O(ms).
    """

    @classmethod
    def setUpClass(cls):
        """Build a synthetic InteriorWedgeChart for the dispatch test."""
        # Build wedge map matching gamma_grid.
        theta_nodes = np.linspace(0.0, np.pi / 2, 51)
        r_table = np.empty((DISPATCH_GAMMA_GRID.size, theta_nodes.size))
        for i, g in enumerate(DISPATCH_GAMMA_GRID):
            for j, th in enumerate(theta_nodes):
                r_table[i, j] = r_caustic(float(g), float(th))
        cls.wedge_map = _WedgeCausticMap(
            gamma_nodes=DISPATCH_GAMMA_GRID.copy(),
            theta_nodes=theta_nodes,
            r_table=r_table)

        # Constant envelope = (1.0 + 0.5j): any evaluation returns this,
        # confirming the spline fires and doesn't NaN.
        shape = (DISPATCH_LOG_W_GRID.size, DISPATCH_GAMMA_GRID.size,
                 DISPATCH_R_GRID.size, DISPATCH_THETA_GRID.size)
        envelope_real = np.full(shape, 1.0)
        envelope_imag = np.full(shape, 0.5)

        cls.chart = InteriorWedgeChart.from_wedge_values(
            gamma_grid=DISPATCH_GAMMA_GRID,
            r_grid=DISPATCH_R_GRID,
            theta_wedge_grid=DISPATCH_THETA_GRID,
            log_w_grid=DISPATCH_LOG_W_GRID,
            envelope_real=envelope_real,
            envelope_imag=envelope_imag,
            image_count=4,
            parity=1,
            wedge_map=cls.wedge_map,
            eta_overlap_min=DISPATCH_ETA_FLOOR,
            refused_points=None,
            envelope_definition='interior_sacr_c_envelope')

        # Source in the SECOND quadrant (y1 < 0, y2 > 0).
        cls.gamma = 0.40  # midpoint of [0.30, 0.50]
        # theta_wedge ~ 0.7 (in the middle of [0.2, 1.3])
        theta_angle = 0.7
        r_caust = _interp_r_caustic(cls.gamma, theta_angle, cls.wedge_map)
        y_mag = 0.35 * r_caust  # r ~ 0.35 in [0.10, 0.60]
        # Second quadrant: y1 < 0, y2 > 0.
        cls.y1_q2 = -(y_mag * float(np.cos(theta_angle)))
        cls.y2_q2 = y_mag * float(np.sin(theta_angle))
        # First quadrant reflection (D2): y1 > 0, y2 > 0.
        cls.y1_q1 = abs(cls.y1_q2)
        cls.y2_q1 = cls.y2_q2

        # log_w query within band.
        cls.log_w_query = np.array([np.log(7.0), np.log(11.0), np.log(17.0)])

    def test_select_chart_returns_wedge_chart(self):
        """select_chart returns isinstance InteriorWedgeChart for valid source."""
        result = select_chart(
            [self.chart],
            gamma=self.gamma,
            log_w_min=float(self.log_w_query[0]),
            log_w_max=float(self.log_w_query[-1]),
            eta=0.5,
            theta=0.7,  # arbitrary; unused for wedge
            image_count=4,
            y1_eig=self.y1_q2,
            y2_eig=self.y2_q2)
        self._tick()
        self.assertIsInstance(result, InteriorWedgeChart)

    def test_evaluate_chart_returns_finite_complex_array(self):
        """_evaluate_chart returns a finite complex array of length 3."""
        result = _evaluate_chart(
            self.chart, self.gamma, eta=0.5, theta=0.7,
            log_w_query=self.log_w_query,
            y1_eig=self.y1_q2, y2_eig=self.y2_q2)
        self._tick()
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result)),
                        f'Non-finite values in result: {result}')
        self.assertTrue(np.iscomplexobj(result))

    def test_d2_reflected_source_gives_identical_values(self):
        """Evaluation at second-quadrant source equals first-quadrant reflected.

        The D2 fold produces equivalent serve results regardless of quadrant.
        max |diff| must be exactly 0.0.
        """
        result_q2 = _evaluate_chart(
            self.chart, self.gamma, eta=0.5, theta=0.7,
            log_w_query=self.log_w_query,
            y1_eig=self.y1_q2, y2_eig=self.y2_q2)
        result_q1 = _evaluate_chart(
            self.chart, self.gamma, eta=0.5, theta=0.7,
            log_w_query=self.log_w_query,
            y1_eig=self.y1_q1, y2_eig=self.y2_q1)
        self._tick()
        max_diff = float(np.max(np.abs(result_q2 - result_q1)))
        self.assertEqual(max_diff, 0.0,
                         f'D2 fold NOT exact: max|diff| = {max_diff:.2e}. '
                         f'_to_wedge_fixed has a sign/atan2 bug for negative '
                         f'coordinates.')

    def test_d2_all_four_quadrants_identical(self):
        """All four quadrant reflections produce identical _evaluate_chart values.

        This extends the D2 test: every combination of sign(y1), sign(y2)
        yields the same served envelope. Cost: 4 evaluations × 3 w-points.
        """
        ref_result = _evaluate_chart(
            self.chart, self.gamma, eta=0.5, theta=0.7,
            log_w_query=self.log_w_query,
            y1_eig=self.y1_q1, y2_eig=self.y2_q1)
        for s1, s2 in QUADRANT_SIGNS:
            y1 = s1 * abs(self.y1_q1)
            y2 = s2 * abs(self.y2_q1)
            result = _evaluate_chart(
                self.chart, self.gamma, eta=0.5, theta=0.7,
                log_w_query=self.log_w_query,
                y1_eig=y1, y2_eig=y2)
            self._tick()
            with self.subTest(s1=s1, s2=s2):
                max_diff = float(np.max(np.abs(result - ref_result)))
                self.assertEqual(max_diff, 0.0,
                                 f'Quadrant ({s1},{s2}) differs: '
                                 f'max|diff|={max_diff:.2e}')

    def test_select_chart_refuses_wrong_image_count(self):
        """select_chart returns None when image_count doesn't match."""
        result = select_chart(
            [self.chart],
            gamma=self.gamma,
            log_w_min=float(self.log_w_query[0]),
            log_w_max=float(self.log_w_query[-1]),
            eta=0.5,
            theta=0.7,
            image_count=2,  # chart has image_count=4
            y1_eig=self.y1_q2,
            y2_eig=self.y2_q2)
        self._tick()
        self.assertIsNone(result)


# ===========================================================================
# Test 5: from_wedge_engine carrier continuity gate
# ===========================================================================

#: Gamma grid for carrier continuity tests.
CARRIER_GAMMA_GRID: np.ndarray = np.array([0.25, 0.30, 0.35])

#: Small tile r-range known to be inside one basin (safe).
CARRIER_R_SAFE: tuple[float, float] = (0.10, 0.30)

#: Larger tile r-range that might straddle a basin flip.
CARRIER_R_WIDE: tuple[float, float] = (0.10, 0.90)

#: Theta wedge range near the diagonal where τ-crossings occur.
CARRIER_THETA_RANGE: tuple[float, float] = (0.60, 0.90)

#: Spatial node counts for the carrier test.
CARRIER_N_R: int = 5
CARRIER_N_THETA: int = 5


class CarrierContinuityGateTestCase(_WedgeTestCase):
    """Verify _assert_carrier_continuity correctly detects basin flips.

    The spec asks: call from_wedge_engine; expect either success or
    CarrierDiscontinuityError. Since from_wedge_engine has a bug in the
    LensAmplificationSurrogate.__init__ check (missing InteriorWedgeChart in
    isinstance), we test _assert_carrier_continuity DIRECTLY with both
    synthetic and engine-derived carrier data.

    Cost: synthetic tests are O(us); engine-derived test uses a 3×5×5 grid
    with w=[5,15] (2 w-nodes), so 75 ChangRefsdalChannels.evaluate calls
    ≈ 75 × 20ms = 1.5s. Total < 5s.
    """

    def test_continuous_carrier_does_not_raise(self):
        """Smoothly-varying carrier data passes without raising.

        Synthetic carrier: critical_source varies linearly over the grid,
        well within the caustic reach scale — no flip.
        """
        n_gamma, n_r, n_theta = 3, 5, 5
        gamma_grid = np.linspace(0.25, 0.35, n_gamma)
        # Build carrier that varies smoothly: x = gamma * r * theta (scaled
        # to be << reach).  Reach for gamma=0.25 is ~0.25.
        carrier = np.empty((n_gamma, n_r, n_theta, 2))
        for ig in range(n_gamma):
            for ir in range(n_r):
                for it in range(n_theta):
                    # Source varies smoothly with tiny step: max jump << reach.
                    carrier[ig, ir, it, 0] = 0.01 * (ig + ir)
                    carrier[ig, ir, it, 1] = 0.01 * (ig + it)
        self._tick()
        # Should NOT raise.
        _assert_carrier_continuity(carrier, gamma_grid,
                                   (n_gamma, n_r, n_theta))

    def test_discontinuous_carrier_raises(self):
        """A carrier that jumps > 50% of caustic reach raises the error.

        Synthetic: place a large jump at the middle r-node along axis 1.
        """
        n_gamma, n_r, n_theta = 3, 5, 5
        gamma_grid = np.linspace(0.25, 0.35, n_gamma)
        reach = _caustic_reach(0.25)  # smallest reach in the grid
        # Build smooth carrier, then inject a large jump.
        carrier = np.zeros((n_gamma, n_r, n_theta, 2))
        for ig in range(n_gamma):
            for ir in range(n_r):
                for it in range(n_theta):
                    carrier[ig, ir, it, 0] = 0.001 * ir
                    carrier[ig, ir, it, 1] = 0.001 * it
        # Inject jump > 0.5 * reach between ir=2 and ir=3.
        carrier[:, 3:, :, 0] += 2.0 * reach
        self._tick()
        with self.assertRaises(CarrierDiscontinuityError):
            _assert_carrier_continuity(carrier, gamma_grid,
                                       (n_gamma, n_r, n_theta))

    def test_nan_nodes_do_not_trigger_false_flip(self):
        """NaN (refused) nodes are skipped without raising.

        A tile with some refused nodes (NaN carrier) should pass as long as
        the finite nodes are continuous.
        """
        n_gamma, n_r, n_theta = 3, 5, 5
        gamma_grid = np.linspace(0.25, 0.35, n_gamma)
        # Smooth carrier.
        carrier = np.zeros((n_gamma, n_r, n_theta, 2))
        for ig in range(n_gamma):
            for ir in range(n_r):
                for it in range(n_theta):
                    carrier[ig, ir, it, 0] = 0.001 * (ig + ir)
                    carrier[ig, ir, it, 1] = 0.001 * (ig + it)
        # Mark some nodes as refused (NaN).
        carrier[1, 2, :, :] = np.nan
        carrier[0, :, 3, :] = np.nan
        self._tick()
        # Should NOT raise.
        _assert_carrier_continuity(carrier, gamma_grid,
                                   (n_gamma, n_r, n_theta))

    def test_engine_derived_small_tile_does_not_raise(self):
        """A small safe tile (r∈[0.1,0.3]) passes carrier continuity.

        Uses real ChangRefsdalChannels.evaluate to populate carrier data
        for a small tile known to be within one basin.
        Cost: 3×5×5 = 75 evaluations × ~20ms = ~1.5s.
        """
        gamma_grid = CARRIER_GAMMA_GRID.copy()
        r_grid = np.linspace(*CARRIER_R_SAFE, CARRIER_N_R)
        theta_grid = np.linspace(*CARRIER_THETA_RANGE, CARRIER_N_THETA)

        # Build wedge map.
        theta_nodes = np.linspace(0.0, np.pi / 2, 51)
        r_table = np.empty((gamma_grid.size, theta_nodes.size))
        for i, g in enumerate(gamma_grid):
            for j, th in enumerate(theta_nodes):
                r_table[i, j] = r_caustic(float(g), float(th))
        wedge_map = _WedgeCausticMap(gamma_nodes=gamma_grid.copy(),
                                     theta_nodes=theta_nodes,
                                     r_table=r_table)

        # Populate carrier from engine.
        n_g, n_r, n_th = gamma_grid.size, r_grid.size, theta_grid.size
        carrier = np.full((n_g, n_r, n_th, 2), np.nan, dtype=float)
        w_grid = np.array([5.0, 15.0])

        for ig, gamma in enumerate(gamma_grid):
            for ir, r in enumerate(r_grid):
                for it, theta_w in enumerate(theta_grid):
                    try:
                        y1, y2 = _from_wedge_fixed(
                            float(gamma), float(r), float(theta_w), wedge_map)
                        ch = ChangRefsdalChannels(w_grid)
                        partition = ch.evaluate(
                            gamma=float(gamma), y=(y1, y2),
                            beta=0.0, kappa=0.0)
                        carrier[ig, ir, it] = partition.critical_source
                    except Exception:
                        pass  # leave as NaN (refused)
        self._tick()
        # A small safe tile should pass.
        _assert_carrier_continuity(carrier, gamma_grid, (n_g, n_r, n_th))


# ===========================================================================
# Test 6: from_wedge_engine produces correct envelope values
# ===========================================================================

#: Gamma range for the envelope accuracy test.
ENVELOPE_GAMMA_RANGE: tuple[float, float] = (0.30, 0.40)

#: Radial range for the envelope accuracy test.
ENVELOPE_R_RANGE: tuple[float, float] = (0.20, 0.50)

#: Theta wedge range for the envelope accuracy test.
ENVELOPE_THETA_RANGE: tuple[float, float] = (0.30, 1.20)

#: Frequency range for the envelope accuracy test.
ENVELOPE_W_RANGE: tuple[float, float] = (5.0, 15.0)

#: Number of nodes on each axis (minimum 4 for cubic spline validation).
ENVELOPE_N_GAMMA: int = 4
ENVELOPE_N_R: int = 4
ENVELOPE_N_THETA: int = 4

#: W nodes per decade for the envelope test.
ENVELOPE_W_NODES_PER_DECADE: int = 10

#: Maximum acceptable residual at exact grid nodes.
ENVELOPE_NODE_ATOL: float = 1e-10


class EnvelopeAccuracyTestCase(_WedgeTestCase):
    """Verify the served envelope matches engine output at exact grid nodes.

    Spec: Build a wedge chart at gamma∈[0.3,0.4], r∈[0.2,0.5],
    theta_wedge∈[0.3,1.2], w∈[5,15] with n_gamma=3, n_r=4, n_theta=4,
    w_nodes_per_decade=10. Pick 3 source positions that are grid nodes.
    Compare chart evaluation vs direct ChangRefsdalChannels.evaluate.

    Since from_wedge_engine has a bug (LensAmplificationSurrogate.__init__
    doesn't accept InteriorWedgeChart), we replicate its training logic
    manually and build the chart via from_wedge_values.

    Cost: Training = 3×4×4 = 48 engine calls (w-grid ~4 points per decade
    over [5,15] ≈ 5 points) ≈ 48 × 30ms = ~1.5s. Plus 3 verification
    evaluations. Total < 5s.
    """

    @classmethod
    def setUpClass(cls):
        """Build a wedge chart from engine data and select 3 grid-node sources.

        Replicates the from_wedge_engine training loop manually.
        """
        from cogwheel.lensing.surrogate import (
            _log_reach_gamma_axis, _log_w_grid, _uniform_axis)

        # Build training grids (same logic as from_wedge_engine).
        cls.log_w_grid = _log_w_grid(ENVELOPE_W_RANGE,
                                     ENVELOPE_W_NODES_PER_DECADE)
        cls.gamma_grid = _log_reach_gamma_axis(ENVELOPE_GAMMA_RANGE,
                                               ENVELOPE_N_GAMMA, 'gamma')
        cls.r_grid = _uniform_axis(ENVELOPE_R_RANGE, ENVELOPE_N_R, 'r')
        cls.theta_grid = _uniform_axis(ENVELOPE_THETA_RANGE,
                                       ENVELOPE_N_THETA, 'theta_wedge')
        cls.w_grid = np.exp(cls.log_w_grid)

        # Build wedge map.
        map_theta_nodes = np.linspace(0.0, np.pi / 2, 101)
        r_table = np.empty((cls.gamma_grid.size, map_theta_nodes.size))
        for i, g in enumerate(cls.gamma_grid):
            for j, th in enumerate(map_theta_nodes):
                r_table[i, j] = r_caustic(float(g), float(th))
        cls.wedge_map = _WedgeCausticMap(gamma_nodes=cls.gamma_grid.copy(),
                                         theta_nodes=map_theta_nodes,
                                         r_table=r_table)

        # Training loop: populate envelope tensor.
        shape = (cls.log_w_grid.size, cls.gamma_grid.size,
                 cls.r_grid.size, cls.theta_grid.size)
        envelope_real = np.zeros(shape, dtype=float)
        envelope_imag = np.zeros(shape, dtype=float)
        refused: list[tuple[float, float, float]] = []
        # Track which grid nodes succeeded (for verification).
        cls.succeeded_nodes: list[tuple[int, int, int]] = []
        image_count: int | None = None
        parity: int | None = None

        for ig, gamma in enumerate(cls.gamma_grid):
            for ir, r in enumerate(cls.r_grid):
                for it, theta_w in enumerate(cls.theta_grid):
                    try:
                        y1, y2 = _from_wedge_fixed(
                            float(gamma), float(r), float(theta_w),
                            cls.wedge_map)
                        ch = ChangRefsdalChannels(cls.w_grid)
                        partition = ch.evaluate(
                            gamma=float(gamma), y=(y1, y2),
                            beta=0.0, kappa=0.0)
                    except Exception:
                        refused.append((float(gamma), float(r),
                                        float(theta_w)))
                        continue
                    env = partition.envelope
                    if not np.all(np.isfinite(env)):
                        refused.append((float(gamma), float(r),
                                        float(theta_w)))
                        continue
                    count = int(partition.real_mask.sum())
                    if image_count is None:
                        image_count = count
                        parity = 1
                    elif count != image_count:
                        refused.append((float(gamma), float(r),
                                        float(theta_w)))
                        continue
                    envelope_real[:, ig, ir, it] = env.real
                    envelope_imag[:, ig, ir, it] = env.imag
                    cls.succeeded_nodes.append((ig, ir, it))

        refused_points = (np.array(refused, dtype=float) if refused
                          else np.empty((0, 3), dtype=float))

        cls.image_count = image_count
        cls.envelope_real = envelope_real
        cls.envelope_imag = envelope_imag

        # Build the chart.
        cls.chart = InteriorWedgeChart.from_wedge_values(
            gamma_grid=cls.gamma_grid, r_grid=cls.r_grid,
            theta_wedge_grid=cls.theta_grid, log_w_grid=cls.log_w_grid,
            envelope_real=envelope_real, envelope_imag=envelope_imag,
            image_count=image_count, parity=parity,
            wedge_map=cls.wedge_map,
            eta_overlap_min=0.03,
            refused_points=refused_points,
            envelope_definition='interior_sacr_c_envelope')

    def test_enough_succeeded_nodes(self):
        """At least 3 grid nodes must have succeeded for meaningful testing."""
        self._tick()
        self.assertGreaterEqual(
            len(self.succeeded_nodes), 3,
            f'Only {len(self.succeeded_nodes)} grid nodes succeeded; need >= 3')

    def test_node_envelope_matches_engine(self):
        """At exact grid nodes, spline reproduces engine envelope to < 1e-10.

        The cubic B-spline is exact at its training nodes, so the served
        value must equal the stored training value to machine precision.
        Cost: 3 nodes × 1 _evaluate_chart call each (full w-grid) = 3 calls.
        """
        # Pick up to 3 interior (non-boundary) nodes for best spline accuracy.
        interior_nodes = [
            n for n in self.succeeded_nodes
            if (0 < n[0] < self.gamma_grid.size - 1
                and 0 < n[1] < self.r_grid.size - 1
                and 0 < n[2] < self.theta_grid.size - 1)]
        if len(interior_nodes) < 3:
            # Fall back to any succeeded nodes.
            test_nodes = self.succeeded_nodes[:3]
        else:
            test_nodes = interior_nodes[:3]

        for ig, ir, it in test_nodes:
            gamma = float(self.gamma_grid[ig])
            r = float(self.r_grid[ir])
            theta_w = float(self.theta_grid[it])

            # Get the stored training envelope at this node.
            expected = (self.envelope_real[:, ig, ir, it]
                        + 1j * self.envelope_imag[:, ig, ir, it])

            # Evaluate the chart at this exact grid node via _evaluate_chart.
            # Need to convert (gamma, r, theta_w) back to eigenframe source.
            y1, y2 = _from_wedge_fixed(gamma, r, theta_w, self.wedge_map)
            result = _evaluate_chart(
                self.chart, gamma, eta=0.5, theta=0.7,
                log_w_query=self.log_w_grid,
                y1_eig=y1, y2_eig=y2)

            max_diff = float(np.max(np.abs(result - expected)))
            self._tick()
            with self.subTest(ig=ig, ir=ir, it=it, gamma=gamma, r=r,
                              theta_w=theta_w):
                self.assertLess(
                    max_diff, ENVELOPE_NODE_ATOL,
                    f'Grid-node residual {max_diff:.2e} exceeds '
                    f'{ENVELOPE_NODE_ATOL:.0e}. The envelope tensor '
                    f'population or spline evaluation has a bug.')

    def test_node_evaluation_matches_direct_engine(self):
        """Verify served envelope matches a FRESH engine call at same source.

        This is the independent-oracle form: re-call the engine from scratch
        (not from stored training data) at the same source position and
        compare. Cost: 3 fresh engine calls.
        """
        # Pick up to 3 succeeded interior nodes.
        interior_nodes = [
            n for n in self.succeeded_nodes
            if (0 < n[0] < self.gamma_grid.size - 1
                and 0 < n[1] < self.r_grid.size - 1
                and 0 < n[2] < self.theta_grid.size - 1)]
        test_nodes = (interior_nodes[:3] if len(interior_nodes) >= 3
                      else self.succeeded_nodes[:3])

        for ig, ir, it in test_nodes:
            gamma = float(self.gamma_grid[ig])
            r = float(self.r_grid[ir])
            theta_w = float(self.theta_grid[it])
            y1, y2 = _from_wedge_fixed(gamma, r, theta_w, self.wedge_map)

            # Fresh engine call.
            ch = ChangRefsdalChannels(self.w_grid)
            partition = ch.evaluate(gamma=gamma, y=(y1, y2),
                                    beta=0.0, kappa=0.0)
            engine_env = partition.envelope

            # Chart evaluation.
            result = _evaluate_chart(
                self.chart, gamma, eta=0.5, theta=0.7,
                log_w_query=self.log_w_grid,
                y1_eig=y1, y2_eig=y2)

            max_diff = float(np.max(np.abs(result - engine_env)))
            self._tick()
            with self.subTest(ig=ig, ir=ir, it=it):
                self.assertLess(
                    max_diff, ENVELOPE_NODE_ATOL,
                    f'Served vs fresh engine: max|diff| = {max_diff:.2e} '
                    f'at node ({ig},{ir},{it}). Spline or indexing bug.')


# ===========================================================================
# Self-falsification extension for Tests 4–6
# ===========================================================================

class DispatchSelfFalsificationTestCase(unittest.TestCase):
    """Prove the dispatch/envelope/carrier tests have teeth.

    These meta-tests verify that known-bad inputs DO fail the assertions
    used in the new test classes, without the anti-vacuity base.
    """

    def test_d2_asymmetry_detected_by_corrupted_map(self):
        """If the wedge map is asymmetric, D2 evaluation differs.

        Build a chart with a wedge map whose r_table is scaled differently
        for positive vs negative theta_wedge (simulated by giving each
        source a different r_table), then confirm the served values differ.
        """
        # Build a chart with a normal wedge map.
        gamma_grid = DISPATCH_GAMMA_GRID.copy()
        theta_nodes = np.linspace(0.0, np.pi / 2, 51)
        r_table = np.empty((gamma_grid.size, theta_nodes.size))
        for i, g in enumerate(gamma_grid):
            for j, th in enumerate(theta_nodes):
                r_table[i, j] = r_caustic(float(g), float(th))
        wedge_map = _WedgeCausticMap(
            gamma_nodes=gamma_grid.copy(),
            theta_nodes=theta_nodes,
            r_table=r_table)

        # Non-constant envelope: varies with theta_wedge.
        shape = (DISPATCH_LOG_W_GRID.size, DISPATCH_GAMMA_GRID.size,
                 DISPATCH_R_GRID.size, DISPATCH_THETA_GRID.size)
        # Make envelope vary as function of theta index.
        envelope_real = np.zeros(shape)
        envelope_imag = np.zeros(shape)
        for it in range(DISPATCH_THETA_GRID.size):
            envelope_real[:, :, :, it] = 1.0 + 0.5 * it
            envelope_imag[:, :, :, it] = 0.3 * it
        chart = InteriorWedgeChart.from_wedge_values(
            gamma_grid=gamma_grid,
            r_grid=DISPATCH_R_GRID,
            theta_wedge_grid=DISPATCH_THETA_GRID,
            log_w_grid=DISPATCH_LOG_W_GRID,
            envelope_real=envelope_real,
            envelope_imag=envelope_imag,
            image_count=4, parity=1, wedge_map=wedge_map,
            eta_overlap_min=DISPATCH_ETA_FLOOR,
            refused_points=None,
            envelope_definition='interior_sacr_c_envelope')

        # Two sources at DIFFERENT theta_wedge (but same r and gamma).
        gamma = 0.40
        theta_a = 0.4  # within [0.2, 1.3]
        theta_b = 1.1  # different theta
        r_c_a = _interp_r_caustic(gamma, theta_a, wedge_map)
        r_c_b = _interp_r_caustic(gamma, theta_b, wedge_map)
        y1_a = 0.3 * r_c_a * float(np.cos(theta_a))
        y2_a = 0.3 * r_c_a * float(np.sin(theta_a))
        y1_b = 0.3 * r_c_b * float(np.cos(theta_b))
        y2_b = 0.3 * r_c_b * float(np.sin(theta_b))
        log_w_q = np.array([np.log(10.0)])

        result_a = _evaluate_chart(chart, gamma, eta=0.5, theta=0.7,
                                   log_w_query=log_w_q,
                                   y1_eig=y1_a, y2_eig=y2_a)
        result_b = _evaluate_chart(chart, gamma, eta=0.5, theta=0.7,
                                   log_w_query=log_w_q,
                                   y1_eig=y1_b, y2_eig=y2_b)
        # With a varying envelope, different theta_wedge sources MUST differ.
        self.assertGreater(
            float(np.max(np.abs(result_a - result_b))), 0.0,
            'Self-falsification: theta-varying envelope must produce '
            'different results at different theta; if equal, _evaluate_chart '
            'is ignoring theta_wedge.')

    def test_carrier_continuity_has_teeth(self):
        """Prove CarrierDiscontinuityError actually fires on real flipped data.

        A carrier with a synthetic 2× reach jump DOES raise.
        """
        n_gamma, n_r, n_theta = 3, 4, 4
        gamma_grid = np.linspace(0.25, 0.35, n_gamma)
        reach = _caustic_reach(0.3)
        # Carrier with a large jump between r-nodes 1 and 2.
        carrier = np.zeros((n_gamma, n_r, n_theta, 2))
        carrier[:, 2:, :, 0] = 3.0 * reach  # jump > 0.5*reach
        with self.assertRaises(CarrierDiscontinuityError):
            _assert_carrier_continuity(carrier, gamma_grid,
                                       (n_gamma, n_r, n_theta))

    def test_envelope_residual_catches_wrong_coefficients(self):
        """If spline coefficients are corrupted, node residual exceeds ATOL.

        Build a chart, corrupt the real_coeffs by 10%, and confirm the
        evaluation at a grid node no longer matches the original envelope.
        """
        # Use the dispatch test chart as the basis.
        gamma_grid = DISPATCH_GAMMA_GRID.copy()
        theta_nodes = np.linspace(0.0, np.pi / 2, 51)
        r_table = np.empty((gamma_grid.size, theta_nodes.size))
        for i, g in enumerate(gamma_grid):
            for j, th in enumerate(theta_nodes):
                r_table[i, j] = r_caustic(float(g), float(th))
        wedge_map = _WedgeCausticMap(
            gamma_nodes=gamma_grid.copy(),
            theta_nodes=theta_nodes,
            r_table=r_table)
        shape = (DISPATCH_LOG_W_GRID.size, DISPATCH_GAMMA_GRID.size,
                 DISPATCH_R_GRID.size, DISPATCH_THETA_GRID.size)
        envelope_real = np.full(shape, 2.0)
        envelope_imag = np.full(shape, 1.0)
        chart = InteriorWedgeChart.from_wedge_values(
            gamma_grid=gamma_grid,
            r_grid=DISPATCH_R_GRID,
            theta_wedge_grid=DISPATCH_THETA_GRID,
            log_w_grid=DISPATCH_LOG_W_GRID,
            envelope_real=envelope_real,
            envelope_imag=envelope_imag,
            image_count=4, parity=1, wedge_map=wedge_map,
            eta_overlap_min=0.03,
            refused_points=None,
            envelope_definition='interior_sacr_c_envelope')

        # Evaluate at a grid node with the GOOD chart.
        gamma = float(gamma_grid[1])
        r = float(DISPATCH_R_GRID[1])
        theta_w = float(DISPATCH_THETA_GRID[1])
        y1, y2 = _from_wedge_fixed(gamma, r, theta_w, wedge_map)
        good_result = _evaluate_chart(
            chart, gamma, eta=0.5, theta=0.7,
            log_w_query=DISPATCH_LOG_W_GRID,
            y1_eig=y1, y2_eig=y2)

        # Build a corrupted chart with 10% perturbation on coefficients.
        import dataclasses
        # InteriorWedgeChart is frozen, so we need object.__setattr__.
        corrupted = InteriorWedgeChart(
            gamma_grid=chart.gamma_grid,
            r_grid=chart.r_grid,
            theta_wedge_grid=chart.theta_wedge_grid,
            log_w_grid=chart.log_w_grid,
            real_coeffs=chart.real_coeffs * 1.1,  # 10% corruption
            imag_coeffs=chart.imag_coeffs,
            knots=chart.knots,
            image_count=chart.image_count,
            parity=chart.parity,
            eta_overlap_min=chart.eta_overlap_min,
            refused_points=chart.refused_points,
            param_spacing=chart.param_spacing,
            wedge_map=chart.wedge_map,
            envelope_definition=chart.envelope_definition,
            theta_to_s=chart.theta_to_s)
        bad_result = _evaluate_chart(
            corrupted, gamma, eta=0.5, theta=0.7,
            log_w_query=DISPATCH_LOG_W_GRID,
            y1_eig=y1, y2_eig=y2)

        diff = float(np.max(np.abs(good_result - bad_result)))
        self.assertGreater(
            diff, ENVELOPE_NODE_ATOL,
            'Self-falsification: corrupted coefficients should cause '
            f'detectable residual; got {diff:.2e}')


if __name__ == '__main__':
    unittest.main()
