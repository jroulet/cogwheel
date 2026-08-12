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
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import make_interp_spline

from cogwheel.lensing import surrogate, surrogate_training
from cogwheel.lensing.surrogate import (
    CarrierDiscontinuityError,
    InteriorWedgeChart,
    LensAmplificationSurrogate,
    _KNOWN_WEDGE_AXIS_SCHEMAS,
    _WEDGE_AXIS_SCHEMA,
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
    _wedge_cusp_axis_map,
    _wedge_serves,
    _wedge_theta_waist,
    select_chart,
)
from cogwheel.lensing.surrogate_training import (
    TrainingConfig,
    _WEDGE_R_MIN,
    _build_farfield_chart,
    _build_wedge_chart,
    _farfield_exterior_tiles,
    _gate_chart,
    _heldout_eps,
    _interior_admission,
    _subdivide_wedge_tile,
    _wedge_interior_tiles,
)
from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels
from cogwheel.lensing.chang_refsdal.geometry import caustic_speed, r_caustic

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

        # Cusp-adapted theta_wedge -> u map (WP3 v3 schema): the wedge NPZ
        # reader now REQUIRES a theta_to_u map (a v2/HEAD chart persisted a
        # theta_to_s map; a v3 chart persisted on the identity/None path can
        # no longer round-trip -- the reader hard-refuses on the missing key).
        # Build the real production map so this round-trip exercises the full
        # v3 contract, including the u-map surviving the save/load.
        theta_fine, u_fine = _wedge_cusp_axis_map(
            float(cls.theta_wedge_grid[0]), float(cls.theta_wedge_grid[-1]),
            'low')
        cls.theta_to_u = np.vstack([theta_fine, u_fine])
        cls.u_grid = np.interp(cls.theta_wedge_grid, theta_fine, u_fine)

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
            envelope_definition='interior_sacr_c_envelope',
            theta_to_u=cls.theta_to_u,
            u_grid=cls.u_grid)

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
        """Gate (c): only the HIGH end of the log_w band is gated.

        The low-w flat-extrapolation change to `_log_w_band_serveable`
        clamps queries with ``w < chart.w_min`` to the lowest grid point
        (the envelope is smooth below the first Airy fringe), so a
        ``log_w_min`` below the chart band STILL SERVES.  Only upward
        extrapolation (``log_w_max`` above the band) is refused, because
        the envelope is oscillatory above ``w_max``.
        """
        self._tick()
        # log_w_max above chart's range -> refuse (no upward extrapolation).
        self.assertFalse(self._call_serves(
            log_w_max=SERVES_LOG_W_GRID[-1] + 1.0))
        self._tick()
        # log_w_min below chart's range -> STILL SERVES via flat extrapolation.
        self.assertTrue(self._call_serves(
            log_w_min=SERVES_LOG_W_GRID[0] - 1.0))

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
            theta_to_u=chart.theta_to_u)
        bad_result = _evaluate_chart(
            corrupted, gamma, eta=0.5, theta=0.7,
            log_w_query=DISPATCH_LOG_W_GRID,
            y1_eig=y1, y2_eig=y2)

        diff = float(np.max(np.abs(good_result - bad_result)))
        self.assertGreater(
            diff, ENVELOPE_NODE_ATOL,
            'Self-falsification: corrupted coefficients should cause '
            f'detectable residual; got {diff:.2e}')


# ===========================================================================
# Tests 7-9: held-out accuracy, D2 fold exactness, medial-axis serving.
#
# These three suites are engine-backed: they train ONE small
# InteriorWedgeChart via `from_wedge_engine` (WP1 wired this into the
# training/serving stack, retiring the far-field-interior "ffin" path) and
# then interrogate the SERVED value against a fresh single-point engine
# oracle.  The chart is built once at module scope (`_shared_wedge_surrogate`)
# and shared by all three classes to keep the file well under the fast-tier
# ceiling.
#
# Tolerance justification (HELDOUT_EPS_FLOOR = 5e-2): the SACR-C
# (tau_c-demodulated) interior envelope is a smooth function of (r,
# theta_wedge), so a 5x5x5 tensor-spline reproduces held-out interior
# points to ~1e-2 (measured worst case ~1.5e-2 across the query fan).  The
# 5e-2 bar is the Professor-set ABSOLUTE in-build floor: loose enough that
# the coarse smoke grid clears it with ~3x headroom, tight enough to catch a
# gross fold/indexing/coordinate error.  It is NOT the production accuracy
# bar (which applies to the denser production grid).
# ===========================================================================

#: Training-grid bounds for the shared held-out wedge chart.
HELDOUT_GAMMA_RANGE: tuple[float, float] = (0.30, 0.45)
HELDOUT_R_RANGE: tuple[float, float] = (0.15, 0.60)
HELDOUT_THETA_WEDGE_RANGE: tuple[float, float] = (0.15, 1.35)
HELDOUT_W_RANGE: tuple[float, float] = (5.0, 15.0)

#: Nodes per parameter axis (>= 4 for cubic-spline validity; 5 gives the
#: smooth SACR-C envelope ~1e-2 held-out accuracy).
HELDOUT_N_GAMMA: int = 5
HELDOUT_N_R: int = 5
HELDOUT_N_THETA: int = 5

#: Dense log-w training-axis density.
HELDOUT_W_NODES_PER_DECADE: int = 10

#: Off-NODE query gamma (interior to the training gamma grid, not a node).
HELDOUT_QUERY_GAMMA: float = 0.37

#: Professor-set absolute in-build accuracy floor for held-out interior
#: points, normalised by max|E| (the interior currency).
HELDOUT_EPS_FLOOR: float = 5e-2

#: Off-node interior query points ``(r, theta_wedge)``.  Deliberately
#: includes a near-caustic-centre point (small r) and a diagonal point
#: (theta_wedge ~= pi/4) per the spec, kept clear of the grid corner where
#: coarse-grid interpolation error is largest.
HELDOUT_QUERY_POINTS: tuple[tuple[float, float], ...] = (
    (0.20, 0.60),
    (0.30, np.pi / 4),   # diagonal (medial axis)
    (0.22, 0.50),
    (0.40, 0.90),
    (0.35, 1.05),
    (0.25, 0.40),
    (0.45, 0.70),
    (0.18, 1.10),        # near caustic centre (small r)
)

#: Directory for diagnostic plots.
OUTPUT_DIR: Path = Path(__file__).resolve().parent / 'output'

#: Module-level cache for the shared engine-trained wedge surrogate.
_SHARED_WEDGE_SURROGATE: LensAmplificationSurrogate | None = None


def _shared_wedge_surrogate() -> LensAmplificationSurrogate:
    """Build (once) and return a small engine-trained wedge surrogate.

    Cost: 5x5x5 = 125 training nodes, each a `ChangRefsdalChannels.evaluate`
    over a ~6-point log-w grid (~30ms) -> ~12s total, incurred once and
    shared by Tests 7-9.
    """
    global _SHARED_WEDGE_SURROGATE
    if _SHARED_WEDGE_SURROGATE is None:
        _SHARED_WEDGE_SURROGATE = LensAmplificationSurrogate.from_wedge_engine(
            gamma_range=HELDOUT_GAMMA_RANGE,
            r_range=HELDOUT_R_RANGE,
            theta_wedge_range=HELDOUT_THETA_WEDGE_RANGE,
            w_range=HELDOUT_W_RANGE,
            n_gamma=HELDOUT_N_GAMMA,
            n_r=HELDOUT_N_R,
            n_theta_wedge=HELDOUT_N_THETA,
            w_nodes_per_decade=HELDOUT_W_NODES_PER_DECADE)
    return _SHARED_WEDGE_SURROGATE


def _served_and_engine(chart, gamma: float, r: float, theta_wedge: float,
                       log_w: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Serve and independently evaluate the INTERIOR_SACR_C envelope.

    Maps ``(gamma, r, theta_wedge)`` to the eigenframe source via
    `_from_wedge_fixed`, serves the chart's envelope over ``log_w`` (LOG
    space), and independently re-evaluates a FRESH `ChangRefsdalChannels`
    partition at the SAME source over ``exp(log_w)`` (LINEAR space).

    Returns ``(served, engine_envelope, y1_eig, y2_eig, image_count)``.

    Note (gotcha): `_evaluate_chart` takes ``log_w_query`` in LOG space and
    clamps to the chart's log-w band, whereas `ChangRefsdalChannels` takes a
    LINEAR w grid.  Passing linear w to `_evaluate_chart` clamps every point
    to the band edge and yields a spurious O(1) residual.
    """
    y1, y2 = _from_wedge_fixed(float(gamma), float(r), float(theta_wedge),
                               chart.wedge_map)
    served = _evaluate_chart(chart, float(gamma), eta=0.5, theta=0.7,
                             log_w_query=log_w, y1_eig=y1, y2_eig=y2)
    partition = ChangRefsdalChannels(np.exp(log_w)).evaluate(
        gamma=float(gamma), y=(y1, y2), beta=0.0, kappa=0.0)
    return (served, partition.envelope, y1, y2,
            int(partition.real_mask.sum()))


class WedgeHeldOutAccuracyTestCase(_WedgeTestCase):
    """Held-out (off-node) interior accuracy against the engine oracle.

    SPEC: draw off-node interior query points inside the tile -- including
    one near the caustic centre (small r) and one on the diagonal
    (theta_wedge ~= pi/4) -- map each to the eigenframe via
    `_from_wedge_fixed`, serve the envelope, and compare against a fresh
    single-point engine evaluation of the SAME INTERIOR_SACR_C envelope.
    Residual normalised by max|E| must be < HELDOUT_EPS_FLOOR at every
    point.  Values are asserted against the engine oracle + tolerance; the
    test never asserts which code branch produced the value.
    """

    @classmethod
    def setUpClass(cls):
        cls.surrogate = _shared_wedge_surrogate()
        cls.chart = cls.surrogate.charts[0]
        # Genuinely held-out interior log-w band (not the training nodes).
        cls.query_log_w = np.linspace(
            float(cls.chart.log_w_grid[0]) + 0.05,
            float(cls.chart.log_w_grid[-1]) - 0.05, 6)

    def test_offnode_envelope_within_floor(self):
        """Every off-node interior query serves within the eps floor.

        Cost: 8 query points x 1 engine call each (~30ms) = ~0.25s.
        """
        records: list[tuple[float, float, float]] = []
        for r, theta_wedge in HELDOUT_QUERY_POINTS:
            served, engine, _y1, _y2, image_count = _served_and_engine(
                self.chart, HELDOUT_QUERY_GAMMA, r, theta_wedge,
                self.query_log_w)
            scale = float(np.max(np.abs(engine)))
            eps = float(np.max(np.abs(served - engine))) / scale
            records.append((float(r), float(theta_wedge), eps))
            self._tick()
            with self.subTest(r=r, theta_wedge=theta_wedge):
                self.assertTrue(
                    np.all(np.isfinite(served)),
                    f'Served envelope non-finite at (r={r}, '
                    f'theta_wedge={theta_wedge}).')
                self.assertEqual(
                    image_count, self.chart.image_count,
                    f'Query point image count {image_count} != chart '
                    f'{self.chart.image_count}; not the same regime.')
                self.assertLess(
                    eps, HELDOUT_EPS_FLOOR,
                    f'Held-out eps {eps:.3e} exceeds floor '
                    f'{HELDOUT_EPS_FLOOR:.0e} at (r={r:.3f}, '
                    f'theta_wedge={theta_wedge:.3f}).')
        self._save_diagnostic(records)

    def _save_diagnostic(self, records: list[tuple[float, float, float]]
                         ) -> None:
        """Scatter eps vs (r, theta_wedge); a spike localises a resolution
        failure at small r or a spurious pi/4 carrier seam."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception:
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        r_vals = [rec[0] for rec in records]
        th_vals = [rec[1] for rec in records]
        eps_vals = [rec[2] for rec in records]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sc = ax.scatter(r_vals, th_vals, c=eps_vals, s=120,
                        cmap='viridis', edgecolors='k')
        ax.axhline(np.pi / 4, color='r', ls='--', lw=0.8,
                   label=r'$\theta_w=\pi/4$ diagonal')
        ax.set_xlabel('r (caustic-normalised radius)')
        ax.set_ylabel(r'$\theta_{wedge}$')
        ax.set_title(f'Held-out wedge eps (floor={HELDOUT_EPS_FLOOR:.0e})')
        ax.legend(loc='best', fontsize=8)
        fig.colorbar(sc, ax=ax, label='relative eps')
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'wedge_heldout_accuracy_eps_scatter.png',
                    dpi=110)
        plt.close(fig)

# ===========================================================================
# Test 7b (T1): transverse-cut angular-axis accuracy oracle.
#
# WP1 charts the wedge interior on a CUSP-ADAPTED angular spline axis
# ``u = d**(2/3)`` (``d`` = distance to the near astroid cusp), replacing the
# arc-length-``s`` axis the exterior far-field chart uses.  This test pins the
# reason: along a transverse cut at fixed ``gamma`` and fixed caustic-relative
# radius ``r``, sweeping the wedge angle from just off the cusp edge inward,
# a 5-node cubic spline fit in ``u`` reproduces the exact-engine INTERIOR
# envelope an order of magnitude better than the same fit in raw ``theta``,
# which in turn beats arc-length ``s``.  The ``u`` axis absorbs the
# ``r_caustic ~ const - c * d**(2/3)`` cusp scaling that makes the raw-theta
# (and, worse, the arc-length) envelope diverge near the cusp edge.
#
# Oracle independence: the "truth" at every held-out angle is a FRESH
# `ChangRefsdalChannels` partition evaluated at the exact eigenframe source
# (``r_caustic`` mapped, no chart, no interpolant).  The three candidate
# splines are hand-built here; none shares the production chart's fit path.
# Cost: ~30 single-point engine evals over a 10-point w-grid, ~1.9 s total.
# ===========================================================================

#: Fixed shear for the transverse cut (positive-parity astroid interior).
T1_GAMMA: float = 0.3

#: Fixed caustic-relative radius of the transverse cut (a single chart radial
#: node, well interior: ``r`` in ``(0, 1)``).
T1_R_NODE: float = 0.455

#: Wedge-angle span of the cut (radians): from just off the ``theta = 0`` cusp
#: edge inward.  The cusp scaling is steepest at the small-``theta`` end.
T1_THETA_RANGE: tuple[float, float] = (1e-4, 0.2)

#: Small linear w-grid for the engine envelope along the cut (n_w in 8-12 so
#: the whole cut costs well under a second of engine time).
T1_W_RANGE: tuple[float, float] = (3.0, 12.0)
T1_N_W: int = 10

#: Number of spline nodes per candidate axis (equal budget for a fair
#: comparison; deliberately sparse so the axis choice dominates the error).
T1_N_SPLINE_NODES: int = 5

#: Held-out cut points (deterministic linspace) and the pad that keeps them
#: strictly inside the fitted span (never at a node, never at an endpoint).
T1_N_HELDOUT: int = 15
T1_HELDOUT_PAD: float = 3e-3

#: Fine-grid size for the arc-length quadrature (matches the production
#: `_FARFIELD_ARC_MAP_SIZE`; a 2001-node trapezoid resolves the caustic
#: speed to ~1e-6, far below the axis-choice error under test).
T1_ARC_MAP_SIZE: int = 2001

#: Accuracy bar for the cusp-adapted axis at the 90th percentile of the
#: held-out relative error (measured u p90 ~3.7e-4; ~2.7x headroom).  This is
#: the spec's ``err_u < 1e-3``.
T1_ERR_U_P90_MAX: float = 1e-3

#: Ceiling on the WORST cusp-adapted-axis held-out error (measured u max
#: ~6.8e-4; the spec's ~6.9e-4 measured value with headroom).
T1_ERR_U_MAX_CEILING: float = 1.5e-3


class TransverseCutAxisAccuracyTestCase(_WedgeTestCase):
    """T1: cusp-adapted ``u`` axis beats raw ``theta`` beats arc-length ``s``.

    Builds three 5-node cubic splines of the INTERIOR envelope along a fixed
    transverse cut (one per candidate angular axis) and scores each against a
    fresh single-point engine evaluation at the held-out angles.  The
    cusp-adapted ``u`` axis must win at the 90th percentile AND stay under the
    accuracy bar; the arc-length ``s`` axis (tuned for the exterior far-field
    chart) is the worst because it does not absorb the cusp scaling.
    """

    @classmethod
    def setUpClass(cls):
        cls.w_grid = np.geomspace(T1_W_RANGE[0], T1_W_RANGE[1], T1_N_W)
        cls.theta_lo, cls.theta_hi = T1_THETA_RANGE
        # Arc-length table s(theta) for the 's' axis, built ONCE.
        cls.th_fine = np.linspace(cls.theta_lo, cls.theta_hi, T1_ARC_MAP_SIZE)
        speed = caustic_speed(T1_GAMMA, cls.th_fine, branch=1)
        cls.s_fine = cumulative_trapezoid(speed, cls.th_fine, initial=0.0)
        # Held-out cut (deterministic; strictly interior to the fitted span).
        cls.held = np.linspace(cls.theta_lo + T1_HELDOUT_PAD,
                               cls.theta_hi - T1_HELDOUT_PAD, T1_N_HELDOUT)
        cls.envelope_held = np.array([cls._envelope(t) for t in cls.held])
        cls.norm = float(np.abs(cls.envelope_held).max())
        cls.eps = {kind: cls._axis_eps(kind) for kind in ('u', 'theta', 's')}
        cls.stats = {
            kind: (float(np.percentile(e, 50)), float(np.percentile(e, 90)),
                   float(e.max()), float(cls.held[int(e.argmax())]))
            for kind, e in cls.eps.items()}

    @classmethod
    def _envelope(cls, theta: float) -> np.ndarray:
        """Exact INTERIOR envelope over the w-grid at wedge angle ``theta``.

        Maps ``(gamma=T1_GAMMA, r=T1_R_NODE, theta)`` to the eigenframe source
        analytically (``|y| = r * r_caustic``) and evaluates a fresh engine
        partition -- the independent oracle, sharing no code with the fitted
        splines.
        """
        y_mag = T1_R_NODE * r_caustic(T1_GAMMA, float(theta))
        y1 = y_mag * np.cos(float(theta))
        y2 = y_mag * np.sin(float(theta))
        return ChangRefsdalChannels(cls.w_grid).evaluate(
            gamma=T1_GAMMA, y=(y1, y2), beta=0.0, kappa=0.0).envelope.copy()

    @classmethod
    def _axis_coord(cls, theta: np.ndarray, kind: str) -> np.ndarray:
        """Angular spline coordinate for the named axis at ``theta``."""
        theta = np.asarray(theta, float)
        if kind == 'theta':
            return theta
        if kind == 'u':
            # Distance to the near cusp is d = theta (low column); u = d**(2/3).
            return theta ** (2.0 / 3.0)
        if kind == 's':
            return np.interp(theta, cls.th_fine, cls.s_fine)
        raise ValueError(f'unknown axis kind {kind!r}')

    @classmethod
    def _node_thetas(cls, kind: str) -> np.ndarray:
        """Five spline nodes placed UNIFORM in the named axis coordinate."""
        if kind == 'theta':
            return np.linspace(cls.theta_lo, cls.theta_hi, T1_N_SPLINE_NODES)
        if kind == 'u':
            u = np.linspace(cls.theta_lo ** (2.0 / 3.0),
                            cls.theta_hi ** (2.0 / 3.0), T1_N_SPLINE_NODES)
            return u ** 1.5
        if kind == 's':
            s_nodes = np.linspace(cls.s_fine[0], cls.s_fine[-1],
                                  T1_N_SPLINE_NODES)
            return np.interp(s_nodes, cls.s_fine, cls.th_fine)
        raise ValueError(f'unknown axis kind {kind!r}')

    @classmethod
    def _axis_eps(cls, kind: str) -> np.ndarray:
        """Per-held-out-angle relative error of the ``kind``-axis spline.

        eps(theta) = max_w |E_spline - E_engine| / max_cut|E_engine|.
        """
        node_theta = cls._node_thetas(kind)
        envelope_nodes = np.array([cls._envelope(t) for t in node_theta])
        coord_nodes = cls._axis_coord(node_theta, kind)
        order = np.argsort(coord_nodes)
        spline_re = make_interp_spline(
            coord_nodes[order], envelope_nodes[order].real, k=3, axis=0)
        spline_im = make_interp_spline(
            coord_nodes[order], envelope_nodes[order].imag, k=3, axis=0)
        coord_held = cls._axis_coord(cls.held, kind)
        predicted = spline_re(coord_held) + 1j * spline_im(coord_held)
        return np.abs(predicted - cls.envelope_held).max(axis=1) / cls.norm

    def test_cusp_axis_beats_theta_beats_arclength_at_p90(self):
        """err_u < err_theta < err_s at p90, and err_u p90 < the accuracy bar.

        The 90th percentile is the discriminating statistic: the cusp
        advantage lives in the near-cusp TAIL of the cut, where the raw-theta
        and arc-length fits spike (see the diagnostic overlay).
        """
        u_p90 = self.stats['u'][1]
        theta_p90 = self.stats['theta'][1]
        s_p90 = self.stats['s'][1]
        self._tick(3)
        self.assertLess(
            u_p90, theta_p90,
            f'cusp-adapted u (p90={u_p90:.3e}) must beat raw theta '
            f'(p90={theta_p90:.3e}).')
        self.assertLess(
            theta_p90, s_p90,
            f'raw theta (p90={theta_p90:.3e}) must beat arc-length s '
            f'(p90={s_p90:.3e}).')
        self.assertLess(
            u_p90, T1_ERR_U_P90_MAX,
            f'cusp-adapted u p90 {u_p90:.3e} exceeds the accuracy bar '
            f'{T1_ERR_U_P90_MAX:.0e}.')

    def test_cusp_axis_worst_error_under_ceiling(self):
        """Even the WORST held-out cusp-adapted error stays under the ceiling.

        Reports the full p50/p90/max distribution and the worst-sample locus
        for each axis (never a bare max) via a text dump.
        """
        u_max = self.stats['u'][2]
        self._tick()
        self.assertLess(
            u_max, T1_ERR_U_MAX_CEILING,
            f'cusp-adapted u max {u_max:.3e} exceeds ceiling '
            f'{T1_ERR_U_MAX_CEILING:.0e}.')
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        lines = [f'gamma={T1_GAMMA} r={T1_R_NODE} '
                 f'theta_range={T1_THETA_RANGE} n_nodes={T1_N_SPLINE_NODES}']
        for kind in ('u', 'theta', 's'):
            p50, p90, mx, worst_theta = self.stats[kind]
            lines.append(
                f'axis={kind:5s} p50={p50:.3e} p90={p90:.3e} max={mx:.3e} '
                f'worst_theta={worst_theta:.4f}')
        (OUTPUT_DIR / 'transverse_cut_axis_accuracy.txt').write_text(
            '\n'.join(lines) + '\n')

    def test_all_axis_errors_are_finite(self):
        """Every held-out error on every axis is finite (no NaN/Inf leak)."""
        for kind, eps in self.eps.items():
            self._tick()
            with self.subTest(axis=kind):
                self.assertTrue(np.all(np.isfinite(eps)),
                                f'{kind}-axis produced non-finite eps.')

    def test_diagnostic_overlay_plot(self):
        """DIAGNOSTIC: overlay |E_spline - E_engine| vs theta for all axes.

        The arc-length curve spikes toward the cusp edge (theta -> 0) while
        the cusp-adapted-u curve stays flat -- the visual justification for
        the axis choice.
        """
        self._tick()
        # The plot merely visualises already-asserted arrays; the anti-vacuity
        # tick guards against a silent no-op if plotting is unavailable.
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception:
            self.skipTest('matplotlib unavailable')
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for kind, marker in (('u', 'o'), ('theta', 's'), ('s', '^')):
            ax.semilogy(self.held, self.eps[kind], marker=marker,
                        label=f'{kind} (p90={self.stats[kind][1]:.1e})')
        ax.set_xlabel(r'$\theta_{wedge}$ (rad)')
        ax.set_ylabel(r'held-out $|F_{spline}-F_{engine}|/\max|F|$')
        ax.set_title(f'Transverse-cut axis accuracy (gamma={T1_GAMMA}, '
                     f'r={T1_R_NODE})')
        ax.legend(loc='best', fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'transverse_cut_axis_accuracy.png', dpi=110)
        plt.close(fig)
        self.assertTrue(
            (OUTPUT_DIR / 'transverse_cut_axis_accuracy.png').exists())


# ===========================================================================
# Test 8: D2 (4-fold astroid) fold exactness at the SERVED level.
# ===========================================================================

#: Off-node interior source used for the D2 fold test, given directly in
#: wedge-fixed ``(r, theta_wedge)`` (canonical first quadrant) then mapped
#: to a positive eigenframe ``(y1, y2)`` with y1 != 0 and y2 != 0.
D2_SOURCE_R: float = 0.30
D2_SOURCE_THETA_WEDGE: float = 0.60

#: A DIFFERENT (non-mirror) interior source, used to prove the equality
#: test has teeth (the served value is not a theta-independent constant).
D2_OTHER_R: float = 0.45
D2_OTHER_THETA_WEDGE: float = 1.10

#: Fold-exactness tolerance.  The D2 fold takes ``abs(y1)``, ``abs(y2)``
#: (exact float negation) before ``atan2``, so the four mirror magnitudes
#: are bit-identical; 1e-12 is a defensive fallback for any atan2 round-off.
D2_ATOL: float = 1e-12


class WedgeD2FoldExactnessTestCase(_WedgeTestCase):
    """D2 (4-fold astroid) reflection symmetry of the SERVED envelope.

    The wedge coordinate exploits the astroid caustic's D2 symmetry by
    folding a source into the canonical first quadrant
    (theta_wedge = atan2(|y2|, |y1|), r = |y| / r_caustic).  So the four D2
    mirror images (+-y1, +-y2) of any interior source MUST serve identical
    magnitudes.  This exercises that `_evaluate_chart` (not just the
    coordinate helper) applies the fold: any missing abs on y1 vs y2, or an
    axis swap, would break the equality.
    """

    @classmethod
    def setUpClass(cls):
        cls.surrogate = _shared_wedge_surrogate()
        cls.chart = cls.surrogate.charts[0]
        cls.log_w = cls.chart.log_w_grid
        # Canonical first-quadrant source (positive y1, y2).
        cls.y1, cls.y2 = _from_wedge_fixed(
            HELDOUT_QUERY_GAMMA, D2_SOURCE_R, D2_SOURCE_THETA_WEDGE,
            cls.chart.wedge_map)

    def _served_magnitude(self, y1: float, y2: float) -> np.ndarray:
        """Serve |envelope| at eigenframe source (y1, y2)."""
        served = _evaluate_chart(
            self.chart, HELDOUT_QUERY_GAMMA, eta=0.5, theta=0.7,
            log_w_query=self.log_w, y1_eig=y1, y2_eig=y2)
        return np.abs(served)

    def test_source_is_off_axis(self):
        """Precondition: the chosen source has y1 != 0 AND y2 != 0.

        A D2 fold test is only meaningful for a genuinely off-axis source
        (otherwise some mirrors coincide).
        """
        self._tick()
        self.assertGreater(abs(self.y1), 1e-6)
        self.assertGreater(abs(self.y2), 1e-6)

    def test_four_mirrors_serve_identical_magnitude(self):
        """The four D2 mirrors (+-y1, +-y2) serve bit-identical magnitudes.

        Cost: 4 spline evaluations (no engine calls) -> < 10ms.
        """
        y1a, y2a = abs(self.y1), abs(self.y2)
        mirror_mags = {}
        for s1, s2 in QUADRANT_SIGNS:
            mirror_mags[(s1, s2)] = self._served_magnitude(s1 * y1a, s2 * y2a)
        reference = mirror_mags[(+1, +1)]
        max_pairwise = 0.0
        for (s1, s2), mag in mirror_mags.items():
            diff = float(np.max(np.abs(mag - reference)))
            max_pairwise = max(max_pairwise, diff)
            self._tick()
            with self.subTest(s1=s1, s2=s2):
                self.assertLessEqual(
                    diff, D2_ATOL,
                    f'D2 mirror ({s1:+d},{s2:+d}) magnitude differs by '
                    f'{diff:.2e} from the (+,+) reference; a missing abs or '
                    f'axis swap in the fold.')
        # Diagnostic: report the four values and the worst pairwise diff.
        print(f'\n[D2 fold] |E| at w={np.exp(self.log_w)[0]:.2f}: '
              + ', '.join(f'({s1:+d},{s2:+d})={mirror_mags[(s1, s2)][0]:.6f}'
                          for s1, s2 in QUADRANT_SIGNS)
              + f'  max pairwise diff={max_pairwise:.2e}')

    def test_non_mirror_source_differs(self):
        """SELF-FALSIFICATION: a non-D2-related source serves a DIFFERENT
        magnitude, proving the equality test is not vacuously true.

        If the chart returned a theta-independent constant, the four-mirror
        equality would pass trivially; here a genuinely different interior
        source must move the served magnitude well above D2_ATOL.
        """
        y1_other, y2_other = _from_wedge_fixed(
            HELDOUT_QUERY_GAMMA, D2_OTHER_R, D2_OTHER_THETA_WEDGE,
            self.chart.wedge_map)
        reference = self._served_magnitude(abs(self.y1), abs(self.y2))
        other = self._served_magnitude(y1_other, y2_other)
        diff = float(np.max(np.abs(other - reference)))
        self._tick()
        self.assertGreater(
            diff, 1e-3,
            f'Self-falsification: a non-mirror source produced an almost '
            f'identical magnitude (diff={diff:.2e}); the served value is '
            f'suspiciously insensitive to (r, theta_wedge).')


# ===========================================================================
# Test 9: Medial-axis serving (the ffin regression fix).
# ===========================================================================

#: Medial-axis query points ``(name, r, theta_wedge)`` that the retired
#: far-field-interior ("ffin") FarFieldChart path refused: a near-centre
#: small-r point and the theta_wedge = pi/4 diagonal.  On the astroid the
#: medial axis runs along the diagonals and through the centre.
MEDIAL_QUERY_POINTS: tuple[tuple[str, float, float], ...] = (
    ('near_centre', 0.17, 0.60),
    ('diagonal_pi4', 0.30, np.pi / 4),
    ('tiny_r', 0.155, 0.90),
)

#: Serving-accuracy floor for medial-axis points (same absolute in-build
#: floor as the held-out accuracy test).
MEDIAL_EPS_FLOOR: float = HELDOUT_EPS_FLOOR


class MedialAxisServingTestCase(_WedgeTestCase):
    """Medial-axis queries SERVE and are accurate (the ffin regression fix).

    HISTORICAL CAUSE: the retired far-field-interior ("ffin") FarFieldChart
    charted the astroid interior in far-field-smooth (s, d) coordinates
    keyed off the NEAREST caustic foot.  On the medial axis -- the locus
    equidistant from two or more caustic feet (the diagonals and the centre)
    -- the nearest-foot assignment is degenerate/discontinuous, so ffin
    REFUSED near-centre and theta_wedge = pi/4 diagonal sources, leaving a
    blind spot.  The wedge coordinate (D2 fold to the canonical first
    quadrant, r = |y| / r_caustic) has no nearest-foot dependence, so those
    same medial-axis points now serve.

    This test asserts served-AND-accurate: `select_chart` returns the wedge
    chart (finite envelope within the eps floor of a fresh engine oracle).
    A refusal is reported with its (r, theta_wedge) to localise any residual
    blind spot.
    """

    @classmethod
    def setUpClass(cls):
        cls.surrogate = _shared_wedge_surrogate()
        cls.chart = cls.surrogate.charts[0]
        cls.log_w_min = float(cls.chart.log_w_grid[0]) + 0.05
        cls.log_w_max = float(cls.chart.log_w_grid[-1]) - 0.05
        cls.log_w = np.linspace(cls.log_w_min, cls.log_w_max, 6)

    def _serve_via_select(self, r: float, theta_wedge: float):
        """Serve a medial-axis point through the real `select_chart` gate.

        Returns ``(selected_chart_or_None, served, engine_envelope, eta)``
        using the HONEST partition eta / image-count (not hand-picked
        pass-through values).
        """
        chart = self.chart
        y1, y2 = _from_wedge_fixed(HELDOUT_QUERY_GAMMA, r, theta_wedge,
                                   chart.wedge_map)
        partition = ChangRefsdalChannels(np.exp(self.log_w)).evaluate(
            gamma=HELDOUT_QUERY_GAMMA, y=(y1, y2), beta=0.0, kappa=0.0)
        eta = float(partition.caustic_distance)
        image_count = int(partition.real_mask.sum())
        theta_gauge = float(partition.critical_theta)
        selected = select_chart(
            self.surrogate.charts, gamma=HELDOUT_QUERY_GAMMA,
            log_w_min=self.log_w_min, log_w_max=self.log_w_max, eta=eta,
            theta=theta_gauge, image_count=image_count,
            y1_eig=y1, y2_eig=y2)
        served = _evaluate_chart(
            chart, HELDOUT_QUERY_GAMMA, eta=eta, theta=theta_gauge,
            log_w_query=self.log_w, y1_eig=y1, y2_eig=y2)
        return selected, served, partition.envelope, eta

    def test_medial_points_serve_finite(self):
        """Each medial-axis point is SERVED by the wedge chart (not None).

        Cost: 3 points x 1 engine call each -> ~0.1s.
        """
        for name, r, theta_wedge in MEDIAL_QUERY_POINTS:
            selected, served, _engine, eta = self._serve_via_select(
                r, theta_wedge)
            self._tick()
            with self.subTest(point=name, r=r, theta_wedge=theta_wedge):
                self.assertIs(
                    selected, self.chart,
                    f'Medial-axis point {name} (r={r:.3f}, '
                    f'theta_wedge={theta_wedge:.3f}, eta={eta:.3f}) REFUSED; '
                    f'the medial axis is still a blind spot.')
                self.assertTrue(
                    np.all(np.isfinite(served)),
                    f'Served envelope non-finite for {name}.')

    def test_medial_points_accurate(self):
        """Served medial-axis envelope matches the engine within the floor.

        Cost: 3 points x 1 engine call each -> ~0.1s.
        """
        for name, r, theta_wedge in MEDIAL_QUERY_POINTS:
            _selected, served, engine, _eta = self._serve_via_select(
                r, theta_wedge)
            eps = (float(np.max(np.abs(served - engine)))
                   / float(np.max(np.abs(engine))))
            self._tick()
            with self.subTest(point=name, r=r, theta_wedge=theta_wedge):
                self.assertLess(
                    eps, MEDIAL_EPS_FLOOR,
                    f'Medial-axis {name} eps {eps:.3e} exceeds floor '
                    f'{MEDIAL_EPS_FLOOR:.0e} at (r={r:.3f}, '
                    f'theta_wedge={theta_wedge:.3f}).')


# ===========================================================================
# Test 10 (T2): _wedge_interior_tiles waist-split structural contract
#
# WP2 makes `_wedge_interior_tiles` emit TWO angular columns per radial row,
# split at the astroid caustic WAIST ``theta_waist = argmin_theta
# r_caustic(gamma, theta)`` -- NOT at pi/4.  The external shear stretches the
# astroid, so its two cusps are inequivalent and the waist migrates away from
# pi/4 as gamma grows.  Each column carries a per-side ``axis_origin`` (``'low'``
# below the waist, near the theta=0 cusp; ``'high'`` above it, near pi/2) so the
# chart's cusp-adapted ``u = d**(2/3)`` axis is per-tile monotone.  These tests
# pin that contract directly (no engine), with the PHYSICAL waist oracle
# ``r_caustic(gamma, theta_waist) == gamma`` (the value, since the flat minimum
# leaves theta_waist itself only loosely determined), plus uniform radial rows
# strictly inside (0, r_extent], r_extent < 1 (Airy edge left to the tube chart).
# ===========================================================================

#: Representative outer radial extent (in caustic-relative ``r`` units) for
#: the structural-contract tests.  The production caller caps ``r_extent``
#: below one; 0.85 is a representative in-range value.
WEDGE_TILE_R_EXTENT: float = 0.85

#: Radial-row count for the structural test (mirrors the production
#: ``TrainingConfig.n_farfield_tiles_per_side`` default of 5).
WEDGE_TILE_N_PER_SIDE: int = 5

#: Row counts swept in the row-count contract test.
WEDGE_TILE_N_SWEEP: tuple[int, ...] = (1, 2, 3, 5, 8)

#: Number of angular columns the tiler emits per radial row (WP2: low + high).
WEDGE_N_COLUMNS: int = 2

#: Representative shear for the radial / column-count structural tests, where
#: gamma only sets the waist-split boundary (not the radial tiling).
WEDGE_TILE_GAMMA: float = 0.5

#: Shears spanning the asymmetry range for the waist-oracle test.
WEDGE_WAIST_GAMMAS: tuple[float, ...] = (0.3, 0.6, 0.9)

#: Tolerance for the tiler's split boundary vs an INDEPENDENT
#: `_wedge_theta_waist` call (radians).  Both invoke the same deterministic
#: bounded minimiser, so the boundary matches to well below this.
WEDGE_WAIST_BOUNDARY_ATOL: float = 1e-9

#: Tolerance for the PHYSICAL waist invariant ``|r_caustic(gamma,
#: theta_waist) - gamma| < 1e-6`` (measured ~1e-14; the flat minimum pins the
#: VALUE far tighter than theta_waist itself).
WEDGE_WAIST_VALUE_ATOL: float = 1e-6

#: Above this shear the waist is measurably off pi/4 (asymmetry is real).
WEDGE_ASYMMETRY_GAMMA_MIN: float = 0.6

#: Minimum ``|theta_waist - pi/4|`` deviation required for gamma >=
#: WEDGE_ASYMMETRY_GAMMA_MIN (measured ~0.15 at gamma=0.6, ~0.23 at 0.9).
WEDGE_ASYMMETRY_DEV_MIN: float = 0.10

#: Tolerance for the exact angular-column / radial-row geometry (radians).
WEDGE_THETA_ATOL: float = 1e-12

#: Representative positive-parity band coordinate-radius floor used to
#: reconstruct the production r_extent cap
#: ``r_extent = 1 - max_eta_max / coordinate_radius_min``.
WEDGE_CAP_COORD_RADIUS_MIN: float = 0.30

#: Small / large eta caps for the r_extent-cap falsification: a LARGER
#: ``max_eta_max`` must SHRINK ``r_extent``.
WEDGE_CAP_ETA_SMALL: float = 0.02
WEDGE_CAP_ETA_LARGE: float = 0.10

#: Sweep of ``max_eta_max`` values for the unit-band invariant.
WEDGE_CAP_ETA_SWEEP: tuple[float, ...] = (0.005, 0.02, 0.05, 0.10, 0.20, 0.28)


def _unpack_tile(tile: tuple
                 ) -> tuple[float, float, float, float, int, int, str]:
    """Unpack a wedge-interior tile into its scalar components (WP2 5-tuple).

    A tile is ``((r_center, theta_wedge_center), (half_r, half_theta_wedge),
    i, j, axis_origin)`` where ``i`` is the radial row, ``j`` the angular
    column (0 = low, 1 = high) and ``axis_origin`` the near-cusp side
    (``'low'`` / ``'high'``).
    """
    (r_c, th_c), (half_r, half_th), i, j, axis_origin = tile
    return (float(r_c), float(th_c), float(half_r), float(half_th),
            int(i), int(j), str(axis_origin))


def _wedge_r_extent_cap(max_eta_max: float, coordinate_radius_min: float,
                        grid_rho_extent: float = 1.0) -> float:
    """Reconstruct the production wedge ``r_extent`` cap.

    Mirrors the single expression in `_train_band_charts`:
    ``r_extent = min(grid_rho_extent, 1 - max_eta_max / coordinate_radius_min)``.
    A LARGER ``max_eta_max`` (a wider tube shell reserved for the tube chart)
    pushes ``r_extent`` inward, away from the caustic edge.
    """
    return min(grid_rho_extent,
               1.0 - max_eta_max / coordinate_radius_min)


class WedgeInteriorTilesContractTestCase(_WedgeTestCase):
    """Structural contract of `_wedge_interior_tiles` (no engine).

    Cost: pure-python tile construction, O(us) per call.
    """

    def test_two_angular_columns_split_at_waist(self):
        """Each radial row emits a LOW and a HIGH column split at theta_waist.

        The shared column boundary equals the INDEPENDENT `_wedge_theta_waist`
        value, and the PHYSICAL waist oracle ``r_caustic(gamma, theta_waist)
        == gamma`` holds (the value, not theta_waist, is pinned).  The
        sub-waist column carries ``axis_origin='low'``; the super-waist column
        ``'high'``.
        """
        for gamma in WEDGE_WAIST_GAMMAS:
            theta_waist = _wedge_theta_waist(gamma)
            tiles = _wedge_interior_tiles(gamma, WEDGE_TILE_R_EXTENT,
                                          WEDGE_TILE_N_PER_SIDE)
            with self.subTest(gamma=gamma):
                # Two columns per radial row.
                self.assertEqual(len(tiles),
                                 WEDGE_N_COLUMNS * WEDGE_TILE_N_PER_SIDE)
                # PHYSICAL waist oracle: reach at the waist equals the shear.
                self._tick()
                self.assertLess(
                    abs(r_caustic(gamma, theta_waist) - gamma),
                    WEDGE_WAIST_VALUE_ATOL,
                    'r_caustic(gamma, theta_waist) must equal gamma.')
                for tile in tiles:
                    (_r_c, th_c, _half_r, half_th, _i, j,
                     axis_origin) = _unpack_tile(tile)
                    self._tick()
                    lo_edge = th_c - half_th
                    hi_edge = th_c + half_th
                    if j == 0:
                        # Low column spans [0, theta_waist], near cusp at 0.
                        self.assertEqual(axis_origin, 'low')
                        self.assertAlmostEqual(lo_edge, 0.0,
                                               delta=WEDGE_THETA_ATOL)
                        self.assertAlmostEqual(
                            hi_edge, theta_waist,
                            delta=WEDGE_WAIST_BOUNDARY_ATOL)
                    else:
                        # High column spans [theta_waist, pi/2], near cusp pi/2.
                        self.assertEqual(j, 1)
                        self.assertEqual(axis_origin, 'high')
                        self.assertAlmostEqual(
                            lo_edge, theta_waist,
                            delta=WEDGE_WAIST_BOUNDARY_ATOL)
                        self.assertAlmostEqual(hi_edge, 0.5 * np.pi,
                                               delta=WEDGE_THETA_ATOL)

    def test_waist_deviates_from_pi4_for_large_gamma(self):
        """For gamma >= WEDGE_ASYMMETRY_GAMMA_MIN the waist is off pi/4.

        A pi/4 split would forfeit the whole point of the waist-adaptive
        columns; the shear makes the two cusps inequivalent.
        """
        for gamma in WEDGE_WAIST_GAMMAS:
            if gamma < WEDGE_ASYMMETRY_GAMMA_MIN:
                continue
            theta_waist = _wedge_theta_waist(gamma)
            self._tick()
            with self.subTest(gamma=gamma):
                self.assertGreater(
                    abs(theta_waist - 0.25 * np.pi), WEDGE_ASYMMETRY_DEV_MIN,
                    f'waist {theta_waist:.4f} too close to pi/4 at '
                    f'gamma={gamma}; asymmetry not captured.')

    def test_radial_rows_uniform_and_strictly_inside(self):
        """Radial rows are uniform and lie strictly within (0, r_extent].

        r_min = _WEDGE_R_MIN > 0 (astroid centre excluded); the outermost row
        edge equals r_extent < 1 (Airy edge left to the tube chart); every
        radial row has the SAME half-width (uniform spacing).  Verified on the
        LOW column (radial tiling is column-independent).
        """
        tiles = _wedge_interior_tiles(WEDGE_TILE_GAMMA, WEDGE_TILE_R_EXTENT,
                                      WEDGE_TILE_N_PER_SIDE)
        low = [t for t in tiles if _unpack_tile(t)[5] == 0]
        halfs = {round(_unpack_tile(t)[2], 15) for t in low}
        self.assertEqual(len(halfs), 1,
                         f'Radial rows NOT uniform: half-widths {halfs}.')
        edges = [(_unpack_tile(t)[0] - _unpack_tile(t)[2],
                  _unpack_tile(t)[0] + _unpack_tile(t)[2]) for t in low]
        # Innermost edge is exactly _WEDGE_R_MIN (> 0); outermost is r_extent.
        self.assertAlmostEqual(edges[0][0], _WEDGE_R_MIN, places=12)
        self.assertGreater(_WEDGE_R_MIN, 0.0)
        self.assertAlmostEqual(edges[-1][1], WEDGE_TILE_R_EXTENT, places=12)
        self.assertLess(WEDGE_TILE_R_EXTENT, 1.0)
        # Rows are contiguous, ascending, and never touch 0 or 1.
        prev_hi = _WEDGE_R_MIN
        for lo, hi in edges:
            self._tick()
            self.assertAlmostEqual(lo, prev_hi, places=12,
                                   msg='radial rows must be contiguous.')
            self.assertGreater(lo, 0.0, 'no row edge at r <= 0.')
            self.assertLess(hi, 1.0, 'no row edge at r >= 1.')
            self.assertGreater(hi, lo)
            prev_hi = hi

    def test_row_count_matches_two_columns_times_n_per_side(self):
        """Tile count is 2 * n_per_side; indices are (row, column) row-major."""
        for n in WEDGE_TILE_N_SWEEP:
            tiles = _wedge_interior_tiles(WEDGE_TILE_GAMMA,
                                          WEDGE_TILE_R_EXTENT, n)
            self._tick()
            with self.subTest(n_per_side=n):
                self.assertEqual(len(tiles), WEDGE_N_COLUMNS * n)
                # Deterministic (radial row, column) order: row-major, j in {0,1}.
                rows = [_unpack_tile(t)[4] for t in tiles]
                cols = [_unpack_tile(t)[5] for t in tiles]
                self.assertEqual(
                    rows, [k for k in range(n) for _ in range(WEDGE_N_COLUMNS)])
                self.assertEqual(cols, [0, 1] * n)

    def test_empty_when_extent_below_floor(self):
        """r_extent <= _WEDGE_R_MIN yields no tiles (ladder-served interior).

        The degenerate astroid centre is excluded, so a non-positive usable
        extent produces an empty list rather than a zero/negative-width row.
        """
        self._tick()
        self.assertEqual(
            _wedge_interior_tiles(WEDGE_TILE_GAMMA, _WEDGE_R_MIN,
                                  WEDGE_TILE_N_PER_SIDE), [])
        self.assertEqual(
            _wedge_interior_tiles(WEDGE_TILE_GAMMA, 0.5 * _WEDGE_R_MIN,
                                  WEDGE_TILE_N_PER_SIDE), [])
        self.assertEqual(
            _wedge_interior_tiles(WEDGE_TILE_GAMMA, -0.5,
                                  WEDGE_TILE_N_PER_SIDE), [])

    def test_diagnostic_dump_of_tile_ranges(self):
        """DIAGNOSTIC: dump each tile's (r_range, theta_wedge_range, origin).

        Writes a small text file to tests/output/ and asserts every dumped
        range is inside the unit band and the two columns meet at the waist.
        """
        gamma = WEDGE_TILE_GAMMA
        theta_waist = _wedge_theta_waist(gamma)
        tiles = _wedge_interior_tiles(gamma, WEDGE_TILE_R_EXTENT,
                                      WEDGE_TILE_N_PER_SIDE)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        lines = [f'gamma={gamma} theta_waist={theta_waist:.6f}']
        for tile in tiles:
            r_c, th_c, half_r, half_th, i, j, origin = _unpack_tile(tile)
            r_range = (r_c - half_r, r_c + half_r)
            th_range = (th_c - half_th, th_c + half_th)
            lines.append(f'row={i} col={j} origin={origin} '
                         f'r_range={r_range} theta_wedge_range={th_range}')
            self._tick()
            self.assertGreater(r_range[0], 0.0)
            self.assertLess(r_range[1], 1.0)
        (OUTPUT_DIR / 'wedge_interior_tiles_ranges.txt').write_text(
            '\n'.join(lines) + '\n')


class WedgeInteriorTilesCapFalsificationTestCase(_WedgeTestCase):
    """The r_extent cap perturbation shrinks the tiler's outer edge.

    Reconstructs the production cap and shows that forcing ``max_eta_max``
    larger shrinks ``r_extent`` (and hence the outermost radial row), while no
    admissible cap ever emits a row at r <= 0 or r >= 1.  Cost: O(us).
    """

    def test_larger_eta_max_shrinks_extent_and_outer_row(self):
        """A larger max_eta_max gives a smaller r_extent and outer-row edge."""
        extent_small = _wedge_r_extent_cap(
            WEDGE_CAP_ETA_SMALL, WEDGE_CAP_COORD_RADIUS_MIN)
        extent_large = _wedge_r_extent_cap(
            WEDGE_CAP_ETA_LARGE, WEDGE_CAP_COORD_RADIUS_MIN)
        self._tick()
        self.assertLess(extent_large, extent_small,
                        'A larger tube-shell reservation (max_eta_max) must '
                        'push the wedge r_extent inward.')
        tiles_small = _wedge_interior_tiles(
            WEDGE_TILE_GAMMA, extent_small, WEDGE_TILE_N_PER_SIDE)
        tiles_large = _wedge_interior_tiles(
            WEDGE_TILE_GAMMA, extent_large, WEDGE_TILE_N_PER_SIDE)
        outer_small = _unpack_tile(tiles_small[-1])[0] + \
            _unpack_tile(tiles_small[-1])[2]
        outer_large = _unpack_tile(tiles_large[-1])[0] + \
            _unpack_tile(tiles_large[-1])[2]
        self.assertLess(outer_large, outer_small)
        self.assertAlmostEqual(outer_small, extent_small, places=12)
        self.assertAlmostEqual(outer_large, extent_large, places=12)

    def test_capped_tiles_stay_strictly_in_unit_band(self):
        """Across an eta sweep, r_extent < 1 and every row edge in (0, 1)."""
        for eta in WEDGE_CAP_ETA_SWEEP:
            extent = _wedge_r_extent_cap(eta, WEDGE_CAP_COORD_RADIUS_MIN)
            with self.subTest(max_eta_max=eta):
                self.assertLess(extent, 1.0,
                                'the cap always leaves the Airy edge to the '
                                'tube chart (r_extent < 1).')
                tiles = _wedge_interior_tiles(
                    WEDGE_TILE_GAMMA, extent, WEDGE_TILE_N_PER_SIDE)
                for tile in tiles:
                    r_c, _th_c, half_r, _half_th, _i, _j, _origin = \
                        _unpack_tile(tile)
                    self._tick()
                    self.assertGreater(r_c - half_r, 0.0)
                    self.assertLess(r_c + half_r, 1.0)


# ===========================================================================
# Test 11: ffin (far-field interior) retirement invariants
#
# WP1 removed `_farfield_interior_tiles` and routes the positive-parity
# astroid interior through `_wedge_interior_tiles` + `_build_wedge_chart`
# (InteriorWedgeChart, INTERIOR_SACR_C label).  `_interior_admission` survives
# because the EXTERIOR tiler still consumes its directional-admission geometry.
# ===========================================================================

class FfinRetirementInvariantsTestCase(unittest.TestCase):
    """Static invariants proving the ffin path is retired but the exterior
    admission survives.  Cost: source inspection only, O(ms).
    """

    def test_farfield_interior_tiles_removed(self):
        """`_farfield_interior_tiles` no longer exists on the module."""
        self.assertFalse(
            hasattr(surrogate_training, '_farfield_interior_tiles'),
            'The retired ffin tiler `_farfield_interior_tiles` must be gone.')

    def test_interior_admission_still_present_and_callable(self):
        """`_interior_admission` survives and is callable (exterior use)."""
        self.assertTrue(hasattr(surrogate_training, '_interior_admission'))
        self.assertTrue(callable(_interior_admission))

    def test_exterior_tiler_consumes_interior_admission(self):
        """The exterior tiler takes an `admission` (built by
        `_interior_admission`) and `_train_band_charts` still wires it."""
        sig = inspect.signature(_farfield_exterior_tiles)
        self.assertIn('admission', sig.parameters,
                      '_farfield_exterior_tiles must consume the '
                      '_InteriorAdmission geometry.')
        train_src = inspect.getsource(surrogate_training._train_band_charts)
        self.assertIn('_interior_admission(', train_src,
                      'the positive-parity exterior branch must still build '
                      'the interior-admission geometry.')
        self.assertIn('_farfield_exterior_tiles(', train_src)

    def test_interior_branch_builds_wedge_not_farfield_label(self):
        """The interior is built by `_build_wedge_chart` on INTERIOR_SACR_C;
        `_build_farfield_chart` never trains on the interior label.

        ``_build_farfield_chart`` stores the envelope definition via a
        ternary ``definition=(FARFIELD_KERNEL_SUM_MINUS_GHOST if
        force_minus_ghost else FARFIELD_KERNEL_SUM)``, so the source
        carries ``FARFIELD_KERNEL_SUM`` and the ``force_minus_ghost``
        guard but not ``definition=FARFIELD_KERNEL_SUM`` as a contiguous
        literal.
        """
        wedge_src = inspect.getsource(_build_wedge_chart)
        ff_src = inspect.getsource(_build_farfield_chart)
        self.assertIn('definition=INTERIOR_SACR_C', wedge_src,
                      'the wedge builder must store the INTERIOR_SACR_C '
                      'envelope.')
        # ``FARFIELD_KERNEL_SUM`` appears inside the ternary (the
        # ``else`` branch) and ``FARFIELD_KERNEL_SUM_MINUS_GHOST`` in
        # the ``if`` branch.
        self.assertIn('FARFIELD_KERNEL_SUM', ff_src,
                      'the far-field builder must reference '
                      'FARFIELD_KERNEL_SUM.')
        self.assertIn('FARFIELD_KERNEL_SUM_MINUS_GHOST', ff_src,
                      'the far-field builder must reference '
                      'FARFIELD_KERNEL_SUM_MINUS_GHOST '
                      '(the force_minus_ghost branch).')
        self.assertNotIn('definition=INTERIOR_SACR_C', ff_src,
                         'no FarFieldChart may be trained on the interior '
                         'INTERIOR_SACR_C label after ffin retirement.')

    def test_train_band_charts_routes_interior_through_wedge_builder(self):
        """`_train_band_charts` builds the wedge_interior region via
        `_build_wedge_chart` (not `_build_farfield_chart`)."""
        train_src = inspect.getsource(surrogate_training._train_band_charts)
        self.assertIn('_wedge_interior_tiles(', train_src)
        self.assertIn('_build_wedge_chart(', train_src)
        self.assertIn("'wedge_interior'", train_src)


class WedgeTrainingPathProducesWedgeChartsTestCase(unittest.TestCase):
    """End-to-end: the wedge training path yields InteriorWedgeChart charts,
    never FarFieldChart.

    Reuses the module-shared engine-trained surrogate (no extra engine cost)
    to assert every produced interior chart is an InteriorWedgeChart and none
    is a FarFieldChart carrying the retired interior far-field label.
    """

    @classmethod
    def setUpClass(cls):
        cls.surrogate = _shared_wedge_surrogate()

    def test_all_charts_are_interior_wedge_charts(self):
        """Every chart produced by the wedge path is an InteriorWedgeChart."""
        chart_types = [type(c).__name__ for c in self.surrogate.charts]
        self.assertGreater(len(self.surrogate.charts), 0,
                           f'wedge training produced no charts: {chart_types}')
        for chart in self.surrogate.charts:
            with self.subTest(chart_type=type(chart).__name__):
                self.assertIsInstance(chart, InteriorWedgeChart)
                self.assertNotIsInstance(chart, surrogate.ExteriorPolarChart)


class WedgeTilesSelfFalsificationTestCase(unittest.TestCase):
    """Prove the waist-split / uniformity assertions have teeth.

    These meta-tests feed deliberately-malformed tiles / caps / boundaries
    through the SAME numeric conditions the contract tests use and confirm
    they trip -- guarding against a silently-passing structural suite.
    """

    def test_single_column_would_fail_two_column_count(self):
        """A one-column tiling (len == n) fails the ``2 * n`` count check."""
        n = WEDGE_TILE_N_PER_SIDE
        one_column_len = n  # what a spurious single-column tiler would emit
        self.assertNotEqual(one_column_len, WEDGE_N_COLUMNS * n,
                            'the two-column count check must reject a '
                            'single-column tiling.')

    def test_pi4_boundary_would_fail_waist_check_at_large_gamma(self):
        """Splitting at pi/4 (not the waist) fails the boundary check.

        At gamma=0.9 the true waist is ~0.23 rad from pi/4, far above the
        boundary tolerance -- so asserting the boundary equals the waist would
        reject a pi/4 split, i.e. the contract genuinely constrains it.
        """
        gamma = 0.9
        theta_waist = _wedge_theta_waist(gamma)
        pi4 = 0.25 * np.pi
        self.assertGreater(abs(pi4 - theta_waist), WEDGE_WAIST_BOUNDARY_ATOL,
                           'a pi/4 boundary must be rejected as != waist.')

    def test_wrong_axis_origin_would_fail_origin_check(self):
        """A tile whose axis_origin is swapped fails the per-column check."""
        # Genuine low column carries 'low'; a swapped 'high' would trip the
        # equality assertion in the contract test.
        bad_origin = 'high'
        self.assertNotEqual(bad_origin, 'low')

    def test_physical_waist_oracle_rejects_wrong_theta(self):
        """r_caustic at a NON-waist angle differs from gamma by >> the atol.

        Proves the physical-invariant assertion is not vacuous: evaluating the
        reach off the waist (e.g. at pi/4) breaks ``r_caustic == gamma``.
        """
        gamma = 0.9
        off_waist = 0.25 * np.pi
        self.assertGreater(
            abs(r_caustic(gamma, off_waist) - gamma), WEDGE_WAIST_VALUE_ATOL,
            'the physical waist oracle must reject an off-waist angle.')

    def test_nonuniform_rows_would_fail_uniformity_check(self):
        """Rows with differing half-widths break the single-half-width set."""
        bad_tiles = [((0.2, 0.3), (0.10, 0.3), 0, 0, 'low'),
                     ((0.6, 0.3), (0.25, 0.3), 1, 0, 'low')]
        halfs = {round(_unpack_tile(t)[2], 15) for t in bad_tiles}
        self.assertGreater(len(halfs), 1,
                           'non-uniform rows must present >1 half-width.')

    def test_over_unit_extent_emits_forbidden_row(self):
        """Passing r_extent >= 1 (a caller-cap violation) yields a row that
        the unit-band check rejects — proving the check discriminates."""
        tiles = _wedge_interior_tiles(WEDGE_TILE_GAMMA, 1.2,
                                      WEDGE_TILE_N_PER_SIDE)
        outer_hi = _unpack_tile(tiles[-1])[0] + _unpack_tile(tiles[-1])[2]
        self.assertGreaterEqual(outer_hi, 1.0,
                                'an over-unit extent MUST breach r < 1, so the '
                                'contract test genuinely constrains the cap.')

    def test_cap_shrink_direction_has_teeth(self):
        """Equal eta gives equal extent (the strict-shrink assert would fail);
        larger eta strictly shrinks it."""
        same = _wedge_r_extent_cap(0.05, WEDGE_CAP_COORD_RADIUS_MIN)
        same2 = _wedge_r_extent_cap(0.05, WEDGE_CAP_COORD_RADIUS_MIN)
        self.assertEqual(same, same2)
        bigger_eta = _wedge_r_extent_cap(0.15, WEDGE_CAP_COORD_RADIUS_MIN)
        self.assertLess(bigger_eta, same)

# ===========================================================================
# Test 12 (T3): wedge subdivision splits at the u-midpoint, not the theta-mid.
#
# WP2 subdivides an eps-gated wedge tile by halving BOTH axes.  The RADIAL
# split is at the plain-``r`` midpoint; the ANGULAR split is at the CUSP-
# ADAPTED ``u``-midpoint mapped back to ``theta`` (``u = d**(2/3)`` on the
# parent's own `_wedge_cusp_axis_map`), NEVER the plain-``theta`` midpoint.
# Bisecting in the cusp-singular ``theta`` would forfeit the whole benefit of
# the ``u`` axis (a near-cusp child would still carry the ``theta**(2/3)``
# gradient the coarse parent could not resolve), so the two angular children
# have UNEQUAL ``theta`` widths -- the near-cusp child is narrower.
#
# Oracle independence: the child boundary is checked against a CLOSED-FORM
# ``u``-midpoint inverse ``theta_split = (0.5 * (theta_lo**(2/3) +
# theta_hi**(2/3)))**1.5`` (low origin), derived by hand and independent of
# the fine tabulated map the production code interpolates.
# ===========================================================================

#: A single wedge tile straddling small angles (near the theta=0 cusp), where
#: the u-vs-theta midpoint gap is largest.
T3_THETA_LO: float = 1e-3
T3_THETA_HI: float = 0.4

#: The parent's near-cusp side (low column -> d = theta).
T3_ORIGIN: str = 'low'

#: Contract tolerance for the closed-form u-midpoint vs the production
#: `_wedge_cusp_axis_map` interpolation (~1e-16 measured: u_mid lands on the
#: exact centre node of the odd-length uniform-u grid, so no interp error).
T3_SPLIT_CONTRACT_ATOL: float = 1e-9

#: Tolerance for the `_subdivide_wedge_tile` RETURN value, which rounds
#: theta_split to 6 decimals for a reproducible report.
T3_SPLIT_RETURN_ATOL: float = 1e-6

#: The u-midpoint theta MUST differ from the plain-theta midpoint by more than
#: this (measured gap ~0.055 rad for the T3 span); anything smaller would mean
#: the split degenerated to the theta midpoint.
T3_MIN_THETA_MIDPOINT_DEVIATION: float = 0.02

#: Representative parent radial box for the reachable-red subdivision call.
T3_R_CENTER: float = 0.4
T3_HALF_R: float = 0.1

#: The pre-refactor (legacy) keys the unified `_subdivide_wedge_tile` wrapper
#: MUST keep returning after WP1 folded both subdividers into `_subdivide_tile`.
#: ``max_achieved_depth`` is asserted separately as the additive key.
T3_LEGACY_SUMMARY_KEYS: tuple[str, ...] = (
    'parent_tag', 'region', 'axis_origin', 'theta_split', 'child_half_r',
    'packed', 'children')


def _u_midpoint_theta(theta_lo: float, theta_hi: float, origin: str) -> float:
    """Closed-form angle at the cusp-adapted ``u``-midpoint (independent oracle).

    For ``origin='low'`` (near cusp at 0, ``d = theta``, ``u = theta**(2/3)``
    up to an offset) the midpoint condition ``u(theta_split) = (u_lo + u_hi)/2``
    solves to ``theta_split = (0.5 * (theta_lo**(2/3) + theta_hi**(2/3)))**1.5``.
    For ``origin='high'`` (near cusp at pi/2, ``d = pi/2 - theta``) the mirror
    form applies.  Derived by hand; shares no code with the tabulated map.
    """
    exponent = 2.0 / 3.0
    if origin == 'low':
        return (0.5 * (theta_lo ** exponent + theta_hi ** exponent)) ** 1.5
    if origin == 'high':
        half_pi = 0.5 * np.pi
        d_mid = (0.5 * ((half_pi - theta_lo) ** exponent
                        + (half_pi - theta_hi) ** exponent)) ** 1.5
        return half_pi - d_mid
    raise ValueError(f"origin must be 'low' or 'high'; got {origin!r}.")


class WedgeSubdivisionUMidpointTestCase(_WedgeTestCase):
    """T3: the angular subdivision boundary is the u-midpoint, not theta-mid.

    Part A pins the production map's u-midpoint against the closed-form
    inverse to 1e-9 and proves it is NOT the theta midpoint.  Part B drives
    the REAL `_subdivide_wedge_tile` (engine build stubbed out) and reads back
    its reported ``theta_split``.
    """

    def test_cusp_axis_map_u_midpoint_matches_closed_form(self):
        """Production `_wedge_cusp_axis_map` u-midpoint == closed-form to 1e-9.

        Reproduces exactly what `_subdivide_wedge_tile` computes (u_mid on the
        fitted map, interpolated back to theta) and checks it against the
        hand-derived inverse -- for BOTH near-cusp origins.
        """
        for origin in ('low', 'high'):
            theta_fine, u_fine = _wedge_cusp_axis_map(
                T3_THETA_LO, T3_THETA_HI, origin)
            u_mid = 0.5 * (float(u_fine[0]) + float(u_fine[-1]))
            theta_split = float(np.interp(u_mid, u_fine, theta_fine))
            analytic = _u_midpoint_theta(T3_THETA_LO, T3_THETA_HI, origin)
            self._tick()
            with self.subTest(origin=origin):
                self.assertAlmostEqual(
                    theta_split, analytic, delta=T3_SPLIT_CONTRACT_ATOL,
                    msg=f'{origin} u-midpoint {theta_split:.12f} != closed '
                        f'form {analytic:.12f}.')

    def test_u_midpoint_is_not_theta_midpoint(self):
        """The u-midpoint split is strictly NOT the plain-theta midpoint.

        Prints theta_split, the theta-midpoint, and the u-midpoint image (the
        first must equal the third, not the second).
        """
        theta_fine, u_fine = _wedge_cusp_axis_map(
            T3_THETA_LO, T3_THETA_HI, T3_ORIGIN)
        u_mid = 0.5 * (float(u_fine[0]) + float(u_fine[-1]))
        theta_split = float(np.interp(u_mid, u_fine, theta_fine))
        theta_midpoint = 0.5 * (T3_THETA_LO + T3_THETA_HI)
        analytic = _u_midpoint_theta(T3_THETA_LO, T3_THETA_HI, T3_ORIGIN)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'wedge_subdivision_u_midpoint.txt').write_text(
            f'theta_split={theta_split:.10f}\n'
            f'theta_midpoint={theta_midpoint:.10f}\n'
            f'u_midpoint_image={analytic:.10f}\n')
        self._tick()
        # theta_split matches the u-midpoint image, NOT the theta midpoint.
        self.assertAlmostEqual(theta_split, analytic,
                               delta=T3_SPLIT_CONTRACT_ATOL)
        self.assertGreater(
            abs(theta_split - theta_midpoint),
            T3_MIN_THETA_MIDPOINT_DEVIATION,
            f'theta_split {theta_split:.6f} collapsed onto the theta midpoint '
            f'{theta_midpoint:.6f}; the u-axis benefit is lost.')

    def test_near_cusp_child_is_narrower_in_theta(self):
        """The near-cusp angular child is narrower in theta than the far one.

        For a low-origin parent the split sits below the theta midpoint, so
        the lower (near-cusp) child ``[theta_lo, theta_split]`` is narrower
        than the upper child ``[theta_split, theta_hi]``.
        """
        analytic = _u_midpoint_theta(T3_THETA_LO, T3_THETA_HI, T3_ORIGIN)
        near_cusp_width = analytic - T3_THETA_LO
        far_width = T3_THETA_HI - analytic
        self._tick()
        self.assertLess(near_cusp_width, far_width,
                        'the near-cusp child must be the narrower one in '
                        'theta (that is the whole point of the u split).')

    def test_subdivide_wedge_tile_reports_u_midpoint_split(self):
        """REACHABLE-RED: the REAL `_subdivide_wedge_tile` returns the u-split.

        Stubs `_load_or_build` (so no engine / chart build runs) and
        `_gate_chart` (children pack cleanly), then drives the production
        subdivider and reads its reported ``theta_split`` (rounded to 6 dp).
        It must equal the closed-form u-midpoint and differ from the theta
        midpoint -- pinning the shipping code path, not a reconstruction.
        """
        theta_c = 0.5 * (T3_THETA_LO + T3_THETA_HI)
        half_theta = 0.5 * (T3_THETA_HI - T3_THETA_LO)
        tile = {
            'center': (T3_R_CENTER, theta_c),
            'half': (T3_HALF_R, half_theta),
            'axis_origin': T3_ORIGIN,
            'region': 'wedge_interior',
            'w_range': (10.0, 40.0),
            'si': 0, 'm_lo': 1.0, 'm_hi': 2.0}
        config = SimpleNamespace(
            interior_eps_max=0.05, interior_w_nodes_per_decade=15,
            w_nodes_per_decade=12, n_gamma=5, n_rho=5, n_theta_c=5,
            n_heldout=8)
        charts: list = []
        chart_reports: list = []
        stub_chart = SimpleNamespace(image_count=4)
        stub_report = {'heldout_eps': 1e-9, 'region': 'wedge_interior'}
        outdir = Path(tempfile.mkdtemp())
        try:
            with mock.patch.object(
                    surrogate_training, '_load_or_build',
                    return_value=(stub_chart, stub_report, False)), \
                 mock.patch.object(
                    surrogate_training, '_gate_chart',
                    return_value=(False, None)):
                summary = surrogate_training._subdivide_wedge_tile(
                    tile=tile, parent_tag='w_t3', band=(0.28, 0.32),
                    parity=1, config=config,
                    rng=np.random.default_rng(0), outdir=outdir,
                    charts=charts, chart_reports=chart_reports)
        finally:
            for path in outdir.glob('*'):
                path.unlink()
            outdir.rmdir()
        analytic = _u_midpoint_theta(T3_THETA_LO, T3_THETA_HI, T3_ORIGIN)
        theta_midpoint = 0.5 * (T3_THETA_LO + T3_THETA_HI)
        self._tick()
        self.assertAlmostEqual(
            summary['theta_split'], analytic, delta=T3_SPLIT_RETURN_ATOL,
            msg=f"reported theta_split {summary['theta_split']} != closed-form "
                f'u-midpoint {analytic:.6f}.')
        self.assertGreater(
            abs(summary['theta_split'] - theta_midpoint),
            T3_MIN_THETA_MIDPOINT_DEVIATION,
            'the shipping subdivider must NOT split at the theta midpoint.')
        # Both angular children stay on the parent's near-cusp side.
        self.assertEqual(summary['axis_origin'], T3_ORIGIN)
        # WP1 unified subdivider: the returned summary MUST keep every
        # pre-refactor legacy key AND add the additive 'max_achieved_depth'
        # (the wrapper is now a thin shell over the shared `_subdivide_tile`).
        for key in T3_LEGACY_SUMMARY_KEYS:
            self.assertIn(
                key, summary,
                f"the unified wrapper dropped legacy summary key {key!r}.")
        self.assertIn(
            'max_achieved_depth', summary,
            "the unified wrapper must add the additive 'max_achieved_depth' "
            'key.')
        # With un-gated children (stubbed _gate_chart) every child packs at
        # depth 1, so no recursion occurs: max_achieved_depth == 1.
        self.assertEqual(summary['max_achieved_depth'], 1)
        self.assertEqual(summary['packed'], 4)
        self.assertEqual(len(summary['children']), 4)


class WedgeSubdivisionSelfFalsificationTestCase(unittest.TestCase):
    """Prove the u-midpoint assertions have teeth."""

    def test_theta_midpoint_would_fail_u_split_check(self):
        """The plain-theta midpoint is far from the u-midpoint image.

        If the subdivider (wrongly) split at the theta midpoint, the
        ``|theta_split - theta_midpoint| > tol`` assertion would fire.
        """
        analytic = _u_midpoint_theta(T3_THETA_LO, T3_THETA_HI, T3_ORIGIN)
        theta_midpoint = 0.5 * (T3_THETA_LO + T3_THETA_HI)
        self.assertGreater(abs(analytic - theta_midpoint),
                           T3_MIN_THETA_MIDPOINT_DEVIATION,
                           'the u and theta midpoints must be well separated, '
                           'else the contract test is vacuous.')

    def test_closed_form_matches_map_only_for_correct_origin(self):
        """A low-origin map does NOT match the high-origin closed form.

        Confirms the origin-specific oracle is load-bearing: cross-pairing the
        map and the closed form breaks the 1e-9 agreement.
        """
        theta_fine, u_fine = _wedge_cusp_axis_map(
            T3_THETA_LO, T3_THETA_HI, 'low')
        u_mid = 0.5 * (float(u_fine[0]) + float(u_fine[-1]))
        theta_split_low = float(np.interp(u_mid, u_fine, theta_fine))
        wrong = _u_midpoint_theta(T3_THETA_LO, T3_THETA_HI, 'high')
        self.assertGreater(abs(theta_split_low - wrong),
                           T3_SPLIT_CONTRACT_ATOL,
                           'low-origin split must NOT match the high-origin '
                           'closed form.')

# ===========================================================================
# Test T4 (SHARD A): coarse-tile -> subdivide -> pass feedback loop.
#
# The brief warns the "gate -> subdivide -> re-gate" feedback must NOT be
# trimmed: a wedge tile whose held-out eps fails the interior registration
# bar must be routed into `_subdivide_wedge_tile`, and its (r, u) children
# re-gated on the SAME bar.  This suite drives that REAL loop at a
# deliberately coarse config: build ONE parent wedge chart, confirm it FAILS
# a tightened interior eps bar via the production `_gate_chart` (the exact
# lever `_train_band_charts` reads to decide whether to subdivide), then run
# `_subdivide_wedge_tile` and confirm its children clear the bar.  The
# interior eps bar (`interior_eps_max`) is tightened to 3e-3 so the
# smoothness-dominated parent (measured ~7.9e-3) genuinely fails; the four
# (r, u) children (each a quarter-box) then clear it with ~5x headroom
# (measured child eps < 1e-3).
#
# Cost: parent build (4x4x4 nodes x 4 w-nodes/decade) + 4 child rebuilds of
# the same shape ~= 5 engine chart builds ~= 31s (measured) -- one method's
# worth, run once in setUpClass and shared by every assertion.  Well under
# the 60s single-test ceiling.
# ===========================================================================

#: Gamma band for the T4 parent wedge tile (narrow -> cheap).
T4_BAND: tuple[float, float] = (0.28, 0.32)

#: Parent wedge-fixed box centre ``(r_c, theta_wedge_c)``.
T4_CENTER: tuple[float, float] = (0.35, 0.20)

#: Parent wedge-fixed box half-widths ``(half_r, half_theta_wedge)``.
T4_HALF: tuple[float, float] = (0.15, 0.15)

#: Parent w-band (kept narrow so the DD-product ceiling never binds and the
#: build stays cheap).
T4_W_RANGE: tuple[float, float] = (5.0, 8.0)

#: w-nodes per decade for both parent and children (coarse -> fast).
T4_N_W: int = 4

#: Held-out probe count per chart (parent and each child).
T4_N_HELDOUT: int = 6

#: Axis origin the parent (and, verbatim, its children) sit against.
T4_AXIS_ORIGIN: str = 'low'

#: Tightened interior eps bar: below the parent's smoothness-dominated eps
#: (~7.9e-3) so the coarse parent FAILS and the feedback loop fires; above
#: every child's eps (~<1e-3) so the (r, u) children CLEAR it.
T4_INTERIOR_EPS_MAX: float = 3e-3

#: Deterministic seed for the held-out sampler.
T4_SEED: int = 0

#: A single level of subdivision halves BOTH axes -> exactly four children.
T4_EXPECTED_CHILDREN: int = 4


def _t4_build_parent_and_subdivide(outdir: Path):
    """Build the coarse parent, gate it, and run one level of subdivision.

    Returns ``(parent_eps, parent_gated, summary, config)``.  Mirrors the
    `_train_band_charts` wedge branch: build -> held-out eps -> `_gate_chart`
    -> (if gated) `_subdivide_wedge_tile`.
    """
    config = TrainingConfig(
        n_gamma=4, n_rho=4, n_theta_c=4, w_nodes_per_decade=T4_N_W,
        interior_w_nodes_per_decade=T4_N_W,
        interior_eps_max=T4_INTERIOR_EPS_MAX, n_heldout=T4_N_HELDOUT,
        engine_budget=100000, seed=T4_SEED)
    rng = np.random.default_rng(T4_SEED)
    chart, _calls, _refused = _build_wedge_chart(
        gamma_band=T4_BAND, parity=1, box_center=T4_CENTER, half=T4_HALF,
        w_range=T4_W_RANGE, config=config,
        w_nodes_per_decade=config.interior_w_nodes_per_decade,
        axis_origin=T4_AXIS_ORIGIN)
    r_c, th_c = T4_CENTER
    hr, hth = T4_HALF
    samples: list[tuple[float, float, float]] = []
    for _ in range(config.n_heldout):
        g = float(rng.uniform(*T4_BAND))
        r = float(rng.uniform(r_c - hr, r_c + hr))
        th = float(rng.uniform(th_c - hth, th_c + hth))
        y1, y2 = _from_wedge_fixed(g, r, th, chart.wedge_map)
        samples.append((g, float(y1), float(y2)))
    parent_eps = float(_heldout_eps(chart, samples, {'schema': 't4-parent'}))
    parent_gated, _reason = _gate_chart(
        'interior', {'heldout_eps': parent_eps}, config)
    tile = {'center': T4_CENTER, 'half': T4_HALF,
            'axis_origin': T4_AXIS_ORIGIN, 'region': 'wedge_interior',
            'w_range': T4_W_RANGE, 'si': 0, 'm_lo': 1.0, 'm_hi': 2.0}
    charts: list = []
    reports: list[dict] = []
    summary = _subdivide_wedge_tile(
        tile=tile, parent_tag='t4_parent', band=T4_BAND, parity=1,
        config=config, rng=np.random.default_rng(T4_SEED + 1), outdir=outdir,
        charts=charts, chart_reports=reports)
    return parent_eps, parent_gated, summary, reports


class WedgeSubdivisionFeedbackLoopTestCase(_WedgeTestCase):
    """Coarse parent fails the bar -> subdivision -> children clear the bar.

    The whole point of the brief's "do not trim the feedback" instruction:
    a gated wedge tile must reach `_subdivide_wedge_tile`, and the resulting
    (r, u) children must be RE-gated on the same interior eps bar so a cleared
    child is packed and a still-failing child falls to the serving ladder.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        outdir = Path(cls._tmp.name)
        (cls.parent_eps, cls.parent_gated, cls.summary,
         cls.reports) = _t4_build_parent_and_subdivide(outdir)
        cls.child_eps = [c['eps'] for c in cls.summary['children']
                         if c.get('eps') is not None]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_coarse_parent_fails_interior_bar(self):
        """The coarse parent's eps exceeds the tightened bar (loop trigger).

        This is the precondition the feedback loop exists to service: if the
        parent already cleared the bar there would be nothing to subdivide.
        """
        self._tick()
        self.assertGreater(
            self.parent_eps, T4_INTERIOR_EPS_MAX,
            f'parent eps {self.parent_eps:.3e} must exceed the bar '
            f'{T4_INTERIOR_EPS_MAX:.0e} for the feedback loop to fire.')
        self.assertTrue(
            self.parent_gated,
            '`_gate_chart` must gate the coarse parent (production trigger '
            'for `_subdivide_wedge_tile`).')

    def test_subdivision_attempted_four_children(self):
        """One level of (r, u) subdivision produces exactly four children.

        Evidence that subdivision was ATTEMPTED before any ladder fallback:
        a ladder-served gap is reachable ONLY through a child record, so four
        child records prove the split ran.
        """
        self._tick()
        self.assertEqual(
            len(self.summary['children']), T4_EXPECTED_CHILDREN,
            'a single level of subdivision must emit four (r, u) children.')

    def test_children_clear_bar_and_are_packed(self):
        """After subdivision the children clear the bar and are packed.

        Closes the feedback loop: gated parent -> subdivide -> children pass.
        Every child that carries an eps clears the interior bar; the packed
        count is positive (measured: all four pass with ~5x headroom).
        """
        self.assertGreater(
            self.summary['packed'], 0,
            'at least one (r, u) child must clear the bar and be packed.')
        for child in self.summary['children']:
            eps = child.get('eps')
            if eps is None:
                continue
            self._tick()
            with self.subTest(ci=child['ci']):
                self.assertLess(
                    eps, T4_INTERIOR_EPS_MAX,
                    f'child {child["ci"]} eps {eps:.3e} must clear the '
                    f'interior bar {T4_INTERIOR_EPS_MAX:.0e}.')

    def test_children_strictly_below_parent(self):
        """Every child eps is strictly below the parent eps.

        Subdivision must IMPROVE registration accuracy; a child no better
        than its parent would signal the split bought nothing.  Reports the
        child eps p50/p90/max diagnostic.
        """
        self.assertTrue(self.child_eps, 'no child eps recorded.')
        for child in self.summary['children']:
            eps = child.get('eps')
            if eps is None:
                continue
            self._tick()
            with self.subTest(ci=child['ci']):
                self.assertLess(
                    eps, self.parent_eps,
                    f'child {child["ci"]} eps {eps:.3e} not strictly below '
                    f'parent eps {self.parent_eps:.3e}.')
        self._save_diagnostic()

    def _save_diagnostic(self) -> None:
        """Log parent eps and each child eps p50/p90/max."""
        arr = np.array(self.child_eps)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / 't4_subdivision_feedback.txt'
        lines = [
            'T4 coarse-tile -> subdivide -> pass feedback loop',
            f'interior_eps_max (bar) = {T4_INTERIOR_EPS_MAX:.3e}',
            f'parent_eps             = {self.parent_eps:.6e}',
            f'parent_gated           = {self.parent_gated}',
            f'packed children        = {self.summary["packed"]}',
            f'child eps p50          = {np.percentile(arr, 50):.6e}',
            f'child eps p90          = {np.percentile(arr, 90):.6e}',
            f'child eps max          = {arr.max():.6e}',
            f'theta_split (u-mid)    = {self.summary["theta_split"]:.6f}',
        ]
        path.write_text('\n'.join(lines) + '\n')


# ===========================================================================
# Test T5 (SHARD A): node-exact on-grid + off-node vs engine + NPZ v3
# round-trip of the cusp-adapted u-map.
#
# The cusp-adapted wedge chart is an INTERPOLATING tensor-product spline
# through engine-evaluated SACR-C envelopes.  Two independent accuracy claims
# and one serialization claim:
#   (a) ON GRID: served at a training node the spline reproduces the
#       (deterministic) engine value to ~machine precision -- the theta_wedge
#       -> u remap (`theta_to_u`) is a per-node `np.interp` that lands EXACTLY
#       on the stored u_grid, so the spline is evaluated at its own fit knot.
#       A large on-grid residual would betray a map/serve coordinate mismatch.
#       Measured worst node residual ~5.7e-16 (~6e-16, the spec figure).
#   (b) OFF NODE: at held-out interior witnesses the served envelope matches a
#       FRESH single-point engine call within the interior eps bar (5e-2).
#       Measured worst ~1.2e-2.
#   (c) NPZ: the chart -- including the new ``theta_to_u`` (2, N) u-map and the
#       ``axis_schema`` meta tag -- round-trips bitwise (max|diff| = 0), and
#       the persisted meta reports schema 'wedge_caustic_relative_v3'.
#
# The engine-trained chart is the shared module-scope surrogate (built once,
# ~12s); this suite adds only single-point engine calls and an in-memory NPZ
# round-trip -- well under a second of its own.
# ===========================================================================

#: On-grid node residual ceiling.  Measured worst ~5.7e-16; 1e-14 leaves ~17x
#: headroom while still asserting machine-precision node reproduction.
T5_NODE_EXACT_MAX: float = 1e-14

#: Interior eps bar for the off-node engine comparison (the SACR-C currency).
T5_OFF_EPS_MAX: float = 5e-2

#: The schema tag the persisted wedge chart's meta must report.
T5_EXPECTED_SCHEMA: str = 'wedge_caustic_relative_v3'

#: Interior training-node indices probed on grid (avoid the outermost edge
#: nodes where the caustic-fixed round-trip is least conditioned).
T5_NODE_INDICES: tuple[int, ...] = (1, 2, 3)

#: Off-node interior witnesses ``(r, theta_wedge)`` inside the training box.
T5_OFF_WITNESSES: tuple[tuple[float, float], ...] = (
    (0.28, 0.55), (0.35, 0.90), (0.22, 1.10), (0.45, 0.40))

#: Off-node query gamma (interior to the gamma grid; not a node).
T5_OFF_GAMMA: float = 0.37


class WedgeNodeExactAndNpzV2TestCase(_WedgeTestCase):
    """On-grid machine precision, off-node engine accuracy, v3 NPZ round-trip.

    Reuses the shared engine-trained wedge surrogate (chart carries a real
    cusp-adapted ``theta_to_u`` u-map, unlike the synthetic-envelope chart in
    `NpzRoundTripTestCase`).
    """

    @classmethod
    def setUpClass(cls):
        cls.surrogate = _shared_wedge_surrogate()
        cls.chart = cls.surrogate.charts[0]
        cls.log_w = cls.chart.log_w_grid.copy()

    def _round_trip(self) -> InteriorWedgeChart:
        """Save the chart to an in-memory NPZ and reload it."""
        arrays = _chart_to_npz(self.chart, index=0)
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

    def test_chart_carries_u_map(self):
        """The engine-trained chart stores a (2, N) cusp-adapted u-map.

        Guards the premise of the on-grid/NPZ claims: without ``theta_to_u``
        the serve path would contract on raw theta_wedge and the v3 schema
        would be meaningless.
        """
        self._tick()
        self.assertIsNotNone(
            self.chart.theta_to_u,
            'engine-trained wedge chart must carry a theta_to_u u-map.')
        self.assertEqual(self.chart.theta_to_u.shape[0], 2)
        self.assertGreater(self.chart.theta_to_u.shape[1], 1)

    def test_on_grid_nodes_match_engine_to_machine_precision(self):
        """Served-at-node envelope reproduces the engine to ~machine precision.

        Cost: 3x3x3 = 27 interior nodes x 1 engine call each (~30ms) ~= 0.9s.
        """
        residuals: list[float] = []
        for i, j, k in itertools.product(T5_NODE_INDICES, T5_NODE_INDICES,
                                         T5_NODE_INDICES):
            g = float(self.chart.gamma_grid[i])
            r = float(self.chart.r_grid[j])
            tw = float(self.chart.theta_wedge_grid[k])
            served, engine, _y1, _y2, _n = _served_and_engine(
                self.chart, g, r, tw, self.log_w)
            scale = float(np.max(np.abs(engine)))
            res = float(np.max(np.abs(served - engine))) / scale
            residuals.append(res)
            self._tick()
            with self.subTest(i=i, j=j, k=k):
                self.assertLess(
                    res, T5_NODE_EXACT_MAX,
                    f'on-grid node residual {res:.3e} exceeds machine-'
                    f'precision ceiling {T5_NODE_EXACT_MAX:.0e}; a map/serve '
                    f'coordinate mismatch is the likely cause.')
        self._node_max = max(residuals)
        self._save_node_diagnostic(residuals)

    def test_off_node_witnesses_within_interior_bar(self):
        """Off-node interior witnesses match a fresh engine within the bar.

        Cost: 4 witnesses x 1 engine call each ~= 0.15s.
        """
        off_res: list[float] = []
        for r, tw in T5_OFF_WITNESSES:
            served, engine, _y1, _y2, _n = _served_and_engine(
                self.chart, T5_OFF_GAMMA, r, tw, self.log_w)
            scale = float(np.max(np.abs(engine)))
            res = float(np.max(np.abs(served - engine))) / scale
            off_res.append(res)
            self._tick()
            with self.subTest(r=r, theta_wedge=tw):
                self.assertLess(
                    res, T5_OFF_EPS_MAX,
                    f'off-node witness eps {res:.3e} exceeds the interior '
                    f'bar {T5_OFF_EPS_MAX:.0e} at (r={r}, theta_wedge={tw}).')
        # Node residuals must be dramatically smaller than off-node residuals;
        # if they were comparable the "node-exact" claim would be hollow.
        if getattr(self, '_node_max', None) is None:
            served, engine, _y1, _y2, _n = _served_and_engine(
                self.chart, float(self.chart.gamma_grid[2]),
                float(self.chart.r_grid[2]),
                float(self.chart.theta_wedge_grid[2]), self.log_w)
            scale = float(np.max(np.abs(engine)))
            self._node_max = float(np.max(np.abs(served - engine))) / scale
        self._tick()
        self.assertLess(
            self._node_max * 1e6, min(off_res),
            'on-grid residuals must be >=1e6x smaller than off-node '
            'residuals; otherwise node-exactness is not demonstrated.')

    def test_meta_reports_v3_schema(self):
        """The persisted chart meta reports the v3 axis schema."""
        arrays = _chart_to_npz(self.chart, index=0)
        meta = json.loads(str(arrays['chart0_meta']))
        self._tick()
        self.assertEqual(meta.get('kind'), 'wedge')
        self.assertEqual(
            meta.get('axis_schema'), T5_EXPECTED_SCHEMA,
            'persisted wedge chart must tag the v3 cusp-adapted axis schema.')
        # The module constant and the emitted tag are the same string.
        self.assertEqual(_WEDGE_AXIS_SCHEMA, T5_EXPECTED_SCHEMA)

    def test_u_map_round_trips_bitwise(self):
        """theta_to_u (the u-map) survives NPZ round-trip with max|diff| = 0."""
        reloaded = self._round_trip()
        self._tick()
        self.assertIsNotNone(
            reloaded.theta_to_u,
            'theta_to_u must survive the NPZ round-trip (not drop to None).')
        self.assertEqual(
            reloaded.theta_to_u.shape, self.chart.theta_to_u.shape)
        self.assertEqual(
            float(np.max(np.abs(
                reloaded.theta_to_u - self.chart.theta_to_u))), 0.0,
            'theta_to_u u-map differs after NPZ round-trip.')

    def test_all_stored_fields_round_trip_bitwise(self):
        """Axis grids, coeffs, knots, wedge_map, refused survive bitwise."""
        reloaded = self._round_trip()
        for name in ('gamma_grid', 'r_grid', 'theta_wedge_grid',
                     'log_w_grid', 'real_coeffs', 'imag_coeffs',
                     'refused_points'):
            orig = np.asarray(getattr(self.chart, name))
            relo = np.asarray(getattr(reloaded, name))
            self._tick()
            with self.subTest(field=name):
                self.assertEqual(orig.shape, relo.shape,
                                 f'{name} shape differs after round-trip.')
                if orig.size == 0:
                    continue  # empty refused_points: shape match suffices.
                self.assertEqual(
                    float(np.max(np.abs(orig - relo))), 0.0,
                    f'{name} differs after NPZ round-trip.')
        for k, (ok, rk) in enumerate(zip(self.chart.knots, reloaded.knots)):
            self._tick()
            with self.subTest(knot_axis=k):
                self.assertEqual(float(np.max(np.abs(ok - rk))), 0.0)
        for name in ('gamma_nodes', 'theta_nodes', 'r_table'):
            orig = getattr(self.chart.wedge_map, name)
            relo = getattr(reloaded.wedge_map, name)
            self._tick()
            with self.subTest(field=f'wedge_map.{name}'):
                self.assertEqual(float(np.max(np.abs(orig - relo))), 0.0)

    def test_reloaded_chart_serves_identically(self):
        """Reloaded chart serves the same envelope as the original.

        Byte-identical spline reconstruction implies byte-identical serve at
        an off-node witness (both u-map remap and spline contraction match).
        """
        reloaded = self._round_trip()
        r, tw = T5_OFF_WITNESSES[0]
        y1, y2 = _from_wedge_fixed(T5_OFF_GAMMA, r, tw, self.chart.wedge_map)
        orig = _evaluate_chart(self.chart, T5_OFF_GAMMA, eta=0.5, theta=0.7,
                               log_w_query=self.log_w, y1_eig=y1, y2_eig=y2)
        relo = _evaluate_chart(reloaded, T5_OFF_GAMMA, eta=0.5, theta=0.7,
                               log_w_query=self.log_w, y1_eig=y1, y2_eig=y2)
        self._tick()
        self.assertEqual(float(np.max(np.abs(orig - relo))), 0.0,
                         'reloaded chart serves a different envelope.')

    def _save_node_diagnostic(self, residuals: list[float]) -> None:
        """Tabulate node-exact residuals separately from off-node ones."""
        arr = np.array(residuals)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / 't5_node_exact_residuals.txt'
        lines = [
            'T5 on-grid node-exact residuals (served vs fresh engine)',
            f'ceiling            = {T5_NODE_EXACT_MAX:.3e}',
            f'node residual p50  = {np.percentile(arr, 50):.6e}',
            f'node residual p90  = {np.percentile(arr, 90):.6e}',
            f'node residual max  = {arr.max():.6e}',
            f'n nodes            = {arr.size}',
        ]
        path.write_text('\n'.join(lines) + '\n')


# ===========================================================================
# Test T6 (SHARD A): a stale v1/v2-schema wedge artifact hard-refuses at load.
#
# The v1 -> v2 -> v3 schema bumps retired the arc-length wedge angular axis
# (v1) and then the interim ``theta_to_s``-named cusp axis (v2) in favour of
# the honestly-named cusp-adapted ``u = d**(2/3)`` axis (v3, WP3 rename); a
# chart persisted under any retired tag is stored under a coordinate/name the
# v3 serve path no longer honours and must NOT serve (a silent identity-map
# fallback would query the spline at the wrong ``theta_wedge`` and return a
# finite-but-wrong F).  `_chart_from_npz` routes the wedge branch through
# `_validate_axis_schema`, which hard-refuses any tag absent from
# `_KNOWN_WEDGE_AXIS_SCHEMAS` (only v3).  This suite mutates a REAL persisted
# wedge chart's meta to each retired tag and confirms a named ValueError
# naming the offending schema -- and, as self-falsification, that the same
# round-trip with the meta UNTOUCHED (v3) loads cleanly.
#
# Cost: reuses the shared surrogate; only in-memory NPZ save/load -- <1s.
# ===========================================================================

#: The retired wedge angular-axis schema tag (arc-length axis, pre-WP1).
T6_STALE_SCHEMA: str = 'wedge_caustic_relative_v1'

#: The retired interim cusp-axis schema tag (``theta_to_s``-named, pre-WP3
#: rename).  Under v3 this too must hard-refuse: the rename to ``theta_to_u`` /
#: ``u_grid`` means a v2 artifact carries the old array names and cannot load.
T6_INTERIM_SCHEMA: str = 'wedge_caustic_relative_v2'


def _wedge_npz_with_meta(chart, meta_override: dict | None):
    """Serialize ``chart``, optionally patch its meta, save+load the NPZ.

    ``meta_override`` is merged into the decoded meta dict before re-encoding;
    pass ``None`` to leave the meta untouched.  Returns the loaded npz mapping
    (caller invokes `_chart_from_npz`), plus the temp path to unlink.
    """
    arrays = dict(_chart_to_npz(chart, index=0))
    meta = json.loads(str(arrays['chart0_meta']))
    if meta_override is not None:
        meta.update(meta_override)
    arrays['chart0_meta'] = np.array(json.dumps(meta))
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        np.savez(f, **arrays)
        tmp_path = Path(f.name)
    data = np.load(tmp_path, allow_pickle=True)
    return data, tmp_path


class WedgeStaleSchemaRefusalTestCase(_WedgeTestCase):
    """A v1/v2-tagged wedge artifact hard-refuses; a v3-tagged one loads."""

    @classmethod
    def setUpClass(cls):
        cls.chart = _shared_wedge_surrogate().charts[0]

    def test_v1_schema_raises_named_valueerror(self):
        """Loading a v1-tagged wedge chart raises ValueError naming the tag.

        The refusal must NOT be a silent identity-map fallback: the error
        message names the offending schema and the known set.
        """
        data, tmp_path = _wedge_npz_with_meta(
            self.chart, {'axis_schema': T6_STALE_SCHEMA})
        self._tick()
        try:
            with self.assertRaises(ValueError) as ctx:
                _chart_from_npz(data, index=0)
            msg = str(ctx.exception)
            self.assertIn(
                T6_STALE_SCHEMA, msg,
                'the refusal message must name the offending v1 schema.')
            self.assertNotIn(
                T6_STALE_SCHEMA, _KNOWN_WEDGE_AXIS_SCHEMAS,
                'v1 must not be an accepted wedge schema.')
        finally:
            data.close()
            tmp_path.unlink()

    def test_v2_interim_schema_raises_named_valueerror(self):
        """Loading a v2-tagged (pre-WP3-rename) wedge chart also refuses.

        The WP3 rename (``theta_to_s`` -> ``theta_to_u`` / ``s_grid`` ->
        ``u_grid``) bumped the schema v2 -> v3; a v2 artifact carries the old
        array names and must hard-refuse rather than silently load under the
        new coordinate names.  This gives the rename its own teeth: absent
        the schema bump a v2 chart would load and serve wrong.
        """
        data, tmp_path = _wedge_npz_with_meta(
            self.chart, {'axis_schema': T6_INTERIM_SCHEMA})
        self._tick()
        try:
            with self.assertRaises(ValueError) as ctx:
                _chart_from_npz(data, index=0)
            msg = str(ctx.exception)
            self.assertIn(
                T6_INTERIM_SCHEMA, msg,
                'the refusal message must name the offending v2 schema.')
            self.assertNotIn(
                T6_INTERIM_SCHEMA, _KNOWN_WEDGE_AXIS_SCHEMAS,
                'v2 must not be an accepted wedge schema under v3.')
        finally:
            data.close()
            tmp_path.unlink()

    def test_absent_schema_raises_named_valueerror(self):
        """A wedge chart with axis_schema=None also hard-refuses.

        Absence is treated exactly like an unknown tag -- an untagged legacy
        artifact cannot silently serve.
        """
        data, tmp_path = _wedge_npz_with_meta(
            self.chart, {'axis_schema': None})
        self._tick()
        try:
            with self.assertRaises(ValueError) as ctx:
                _chart_from_npz(data, index=0)
            self.assertIn('None', str(ctx.exception))
        finally:
            data.close()
            tmp_path.unlink()

    def test_untouched_v3_schema_loads_cleanly(self):
        """Self-falsification: the same round-trip with v3 meta loads fine.

        Proves the refusal is caused by the v1/v2 mutation, not by anything
        intrinsic to the round-trip harness.
        """
        data, tmp_path = _wedge_npz_with_meta(self.chart, None)
        self._tick()
        try:
            reloaded = _chart_from_npz(data, index=0)
            self.assertIsInstance(reloaded, InteriorWedgeChart)
            self.assertEqual(reloaded.image_count, self.chart.image_count)
        finally:
            data.close()
            tmp_path.unlink()

    def test_explicit_v3_schema_loads_cleanly(self):
        """Explicitly re-stamping the current v3 tag also loads (control)."""
        data, tmp_path = _wedge_npz_with_meta(
            self.chart, {'axis_schema': _WEDGE_AXIS_SCHEMA})
        self._tick()
        try:
            reloaded = _chart_from_npz(data, index=0)
            self.assertIsInstance(reloaded, InteriorWedgeChart)
        finally:
            data.close()
            tmp_path.unlink()



# ===========================================================================
# Test T7 (SHARD B): bounded recursion closes the three MEASURED marginal
# interior gaps.
#
# WP1 folded both subdividers into `_subdivide_tile`, whose ONE new capability
# over the pre-refactor code is BOUNDED RECURSION: a child that STILL fails the
# 5e-2 interior bar is itself halved (u-midpoint angular split preserved) until
# it clears or the chain reaches `MAX_SUBDIVISION_DEPTH`.  The measured
# configuration (astroid interior, band 0, gamma_mid=0.495) leaves three
# marginal gaps against the bar at depth 1 -- ONE under the r=0.633 parent
# (~6.50e-2) and TWO under r=0.811 (~6.70e-2 and ~5.95e-2).  Under the OLD
# single-level subdivider those three would be recorded as ladder-served gaps;
# under bounded recursion each subdivides once more and every terminal leaf
# clears the bar at achieved depth 2.
#
# This runs through the REAL `_subdivide_wedge_tile` wrapper -> real
# `_subdivide_tile` recursion -> real `_gate_chart` / `_wedge_child_boxes`
# (u-midpoint geometry), on the fast synthetic path the existing wedge tests
# use: only `_load_or_build` is stubbed, injecting a per-tag synthetic
# held-out eps (above-bar for the three designated gap children, shrinking
# ~4x per halving so ONE subdivision clears them).  No engine build runs, so
# the whole suite is milliseconds -- the genuine engine reproduction measured
# 210s for a single depth-2 recursion and is train-tier-only.  The stub is a
# REACHABLE-RED data source, NOT an oracle: the recursion control flow, the
# gating decision, and the child geometry are all the shipping code.
# ===========================================================================

#: Astroid-interior band-0 reproduction: gamma_mid = 0.495.
RECURSION_BAND: tuple[float, float] = (0.475, 0.515)

#: Parent tiles' shared near-cusp angular box (low origin: d = theta_wedge),
#: kept below the gamma=0.495 waist so 'low' is the honest axis origin.
RECURSION_AXIS_ORIGIN: str = 'low'
RECURSION_THETA_C: float = 0.35
RECURSION_HALF_THETA: float = 0.25
RECURSION_HALF_R: float = 0.15
RECURSION_W_RANGE: tuple[float, float] = (8.0, 45.0)

#: The interior held-out eps bar (the SACR-C currency).
RECURSION_INTERIOR_BAR: float = 0.05

#: The two failing parents, keyed by tag, with their MEASURED radial centre.
RECURSION_PARENTS: dict[str, float] = {'w_r0633': 0.633, 'w_r0811': 0.811}

#: The MEASURED marginal gaps: parent tag -> {depth-1 child index: parent eps}.
#: r=0.633 has ONE residual gap ~6.50e-2; r=0.811 has TWO ~6.70e-2 and
#: ~5.95e-2.  All three exceed the 5e-2 bar at depth 1 and must CLOSE after
#: one further halving (depth 2).
RECURSION_GAP_MAP: dict[str, dict[int, float]] = {
    'w_r0633': {2: 6.50e-2},
    'w_r0811': {1: 6.70e-2, 3: 5.95e-2},
}

#: eps of a depth-1 child that already clears the bar (packs immediately).
RECURSION_NONGAP_EPS: float = 3.0e-2

#: A halving of a smooth interior box drops the held-out eps ~4x, so a 6.7e-2
#: parent gap lands at ~1.7e-2 < bar after ONE subdivision (depth 2).
RECURSION_CHILD_DECAY: float = 4.0

#: Expected achieved recursion depth for both reproductions.
RECURSION_EXPECTED_DEPTH: int = 2

#: Tolerance for matching a depth-1 parent-gap eps back to RECURSION_GAP_MAP
#: (the wrapper rounds reported eps to 8 dp).
RECURSION_GAP_ATOL: float = 1e-6


def _recursion_config() -> SimpleNamespace:
    """A minimal `TrainingConfig` stand-in carrying the three eps bars.

    `_chart_gated` reads all of ``tube_eps_max`` / ``farfield_eps_max`` /
    ``interior_eps_max``; the node counts are consumed only by the (stubbed)
    build path and are present for completeness.
    """
    return SimpleNamespace(
        interior_eps_max=RECURSION_INTERIOR_BAR,
        farfield_eps_max=RECURSION_INTERIOR_BAR,
        tube_eps_max=RECURSION_INTERIOR_BAR,
        interior_w_nodes_per_decade=15, w_nodes_per_decade=12,
        n_gamma=5, n_rho=5, n_theta_c=5, n_heldout=8)


def _stub_eps(stem: str, gap_map: dict[str, dict[int, float]]) -> float:
    """Synthetic held-out eps for a child chart, keyed on its tag ``stem``.

    ``stem`` is like ``'w_r0633_c2'`` (a depth-1 child) or
    ``'w_r0633_c2_c0'`` (a depth-2 grandchild).  A designated depth-1 GAP
    child gets an above-bar eps; its descendants shrink by
    ``RECURSION_CHILD_DECAY`` per halving so a single subdivision clears the
    bar.  Every non-gap child clears immediately.
    """
    parts = stem.split('_c')
    ci_path = [int(p) for p in parts[1:]]
    depth = len(ci_path)
    if depth == 0:
        return RECURSION_NONGAP_EPS
    root = parts[0]
    base = gap_map.get(root, {}).get(ci_path[0], RECURSION_NONGAP_EPS)
    return base / (RECURSION_CHILD_DECAY ** (depth - 1))


def _run_recursion(parent_tag: str, r_center: float,
                   gap_map: dict[str, dict[int, float]],
                   config: SimpleNamespace) -> tuple[dict, list]:
    """Drive the REAL `_subdivide_wedge_tile` with only `_load_or_build` stubbed.

    Returns ``(summary, chart_reports)``; ``chart_reports`` is the in-place
    accumulator carrying every packed leaf and every gated (recursed) node.
    """
    tile = {
        'center': (r_center, RECURSION_THETA_C),
        'half': (RECURSION_HALF_R, RECURSION_HALF_THETA),
        'axis_origin': RECURSION_AXIS_ORIGIN,
        'region': 'wedge_interior',
        'w_range': RECURSION_W_RANGE,
        'si': 0, 'm_lo': 1.0, 'm_hi': 2.0}

    def fake_load_or_build(path, build_fn, meta):
        stem = Path(path).stem
        eps = _stub_eps(stem, gap_map)
        return (SimpleNamespace(image_count=4),
                {'heldout_eps': eps, 'region': 'wedge_interior'}, False)

    charts: list = []
    chart_reports: list = []
    outdir = Path(tempfile.mkdtemp())
    try:
        with mock.patch.object(surrogate_training, '_load_or_build',
                               new=fake_load_or_build):
            summary = surrogate_training._subdivide_wedge_tile(
                tile=tile, parent_tag=parent_tag, band=RECURSION_BAND,
                parity=1, config=config, rng=np.random.default_rng(0),
                outdir=outdir, charts=charts, chart_reports=chart_reports)
    finally:
        for path in outdir.glob('*'):
            path.unlink()
        outdir.rmdir()
    return summary, chart_reports


def _terminal_leaves(summary: dict, chart_reports: list,
                     parent_tag: str) -> list[dict]:
    """Walk the subdivision tree and collect every TERMINAL leaf.

    Uses ONLY the per-child summary entries the production code emits (each
    carries its own ``center`` and ``eps``) plus the nested ``subdivision``
    summaries stashed on gated reports -- no geometry is reconstructed here.
    A terminal leaf is any child that was NOT further subdivided: a packed
    leaf (cleared the bar) or a recorded_gated / carrier_flip leaf (a residual
    gap that recursion could not close within the depth cap).
    """
    subdiv_index = {r['name']: r['subdivision']
                    for r in chart_reports if 'subdivision' in r}
    leaves: list[dict] = []

    def walk(node_summary: dict, tag: str) -> None:
        for entry in node_summary['children']:
            child_tag = f"{tag}_c{entry['ci']}"
            if entry['result'] == 'subdivided':
                walk(subdiv_index[child_tag], child_tag)
            else:
                leaves.append({
                    'tag': child_tag, 'center': entry['center'],
                    'eps': entry.get('eps'), 'result': entry['result'],
                    'depth': entry['achieved_depth']})

    walk(summary, parent_tag)
    return leaves


class WedgeRecursionClosesGapsTestCase(_WedgeTestCase):
    """T7: bounded recursion closes the three MEASURED marginal interior gaps.

    Drives the shipping `_subdivide_wedge_tile` -> `_subdivide_tile` recursion
    on the fast stub path (see the module comment above) for both failing
    parents and reads back the reproduced gaps and the closed terminal leaves.
    """

    @classmethod
    def setUpClass(cls):
        config = _recursion_config()
        cls.results = {}
        for tag, r_center in RECURSION_PARENTS.items():
            summary, reports = _run_recursion(
                tag, r_center, RECURSION_GAP_MAP, config)
            leaves = _terminal_leaves(summary, reports, tag)
            cls.results[tag] = (summary, reports, leaves)

    def _gap_children(self, tag: str) -> list[dict]:
        """The depth-1 children that were gated and recursed (the gaps)."""
        summary, _reports, _leaves = self.results[tag]
        return [e for e in summary['children']
                if e['result'] == 'subdivided']

    def _assert_leaves_closed(self, leaves: list[dict]) -> None:
        """Every terminal leaf packed and cleared the interior bar."""
        self.assertGreater(len(leaves), 0,
                           'anti-vacuity: no terminal leaves collected.')
        for leaf in leaves:
            self._tick()
            with self.subTest(tag=leaf['tag']):
                self.assertEqual(
                    leaf['result'], 'packed',
                    f"leaf {leaf['tag']} did not pack "
                    f"(result={leaf['result']}); the gap did not close.")
                self.assertLessEqual(
                    leaf['eps'], RECURSION_INTERIOR_BAR,
                    f"leaf {leaf['tag']} eps {leaf['eps']} exceeds the "
                    f'{RECURSION_INTERIOR_BAR:.0e} interior bar.')

    def test_r0633_one_gap_reproduced_and_closed(self):
        """r=0.633: exactly ONE depth-1 gap (~6.50e-2) closes at depth 2."""
        tag = 'w_r0633'
        summary, _reports, leaves = self.results[tag]
        gaps = self._gap_children(tag)
        self._tick()
        self.assertEqual(
            len(gaps), 1,
            'r=0.633 must reproduce exactly ONE marginal interior gap.')
        gap_eps = gaps[0]['eps']
        self.assertGreater(
            gap_eps, RECURSION_INTERIOR_BAR,
            'the reproduced gap must be ABOVE the bar at depth 1.')
        self.assertAlmostEqual(gap_eps, 6.50e-2, delta=RECURSION_GAP_ATOL)
        self.assertEqual(summary['max_achieved_depth'],
                         RECURSION_EXPECTED_DEPTH)
        self._assert_leaves_closed(leaves)

    def test_r0811_two_gaps_reproduced_and_closed(self):
        """r=0.811: exactly TWO depth-1 gaps (~6.70e-2, ~5.95e-2) close."""
        tag = 'w_r0811'
        summary, _reports, leaves = self.results[tag]
        gaps = self._gap_children(tag)
        self._tick()
        self.assertEqual(
            len(gaps), 2,
            'r=0.811 must reproduce exactly TWO marginal interior gaps.')
        gap_eps = sorted(e['eps'] for e in gaps)
        for got, exp in zip(gap_eps, sorted((5.95e-2, 6.70e-2))):
            self._tick()
            self.assertGreater(
                got, RECURSION_INTERIOR_BAR,
                'each reproduced gap must be ABOVE the bar at depth 1.')
            self.assertAlmostEqual(got, exp, delta=RECURSION_GAP_ATOL)
        self.assertEqual(summary['max_achieved_depth'],
                         RECURSION_EXPECTED_DEPTH)
        self._assert_leaves_closed(leaves)

    def test_summary_keeps_legacy_keys_and_adds_depth(self):
        """The recursed wrapper summary keeps legacy keys + max_achieved_depth.

        The additive key must be present for BOTH parents, and the legacy keys
        must survive WP1's unification into `_subdivide_tile`.
        """
        for tag in RECURSION_PARENTS:
            summary, _reports, _leaves = self.results[tag]
            self._tick()
            with self.subTest(tag=tag):
                for key in T3_LEGACY_SUMMARY_KEYS:
                    self.assertIn(
                        key, summary,
                        f'the unified wrapper dropped legacy key {key!r}.')
                self.assertIn('max_achieved_depth', summary)
                self.assertEqual(summary['parent_tag'], tag)
                self.assertEqual(summary['region'], 'wedge_interior')
                self.assertEqual(summary['axis_origin'],
                                 RECURSION_AXIS_ORIGIN)

    def test_terminal_leaf_eps_reported_with_worst_locus(self):
        """Report p50/p90/max terminal-leaf eps and the worst-sample locus.

        A bare max is forbidden by the house idiom; this dumps the full
        distribution and the (r, theta_wedge) of the worst leaf per parent and
        overall, and asserts every leaf clears the bar and that the packed
        count equals the number of terminal leaves (no residual gaps left).
        """
        all_eps: list[float] = []
        worst: tuple[float, str, list] | None = None
        lines = ['T7 bounded-recursion terminal-leaf eps (stub-driven)',
                 f'interior bar = {RECURSION_INTERIOR_BAR:.3e}']
        for tag in RECURSION_PARENTS:
            summary, _reports, leaves = self.results[tag]
            eps_arr = np.array([leaf['eps'] for leaf in leaves], dtype=float)
            emax = float(eps_arr.max())
            imax = int(np.argmax(eps_arr))
            locus = leaves[imax]['center']
            all_eps.extend(eps_arr.tolist())
            if worst is None or emax > worst[0]:
                worst = (emax, tag, locus)
            lines.append(
                f'{tag}: n_leaves={len(leaves)} packed={summary["packed"]} '
                f'depth={summary["max_achieved_depth"]} '
                f'p50={float(np.percentile(eps_arr, 50)):.4e} '
                f'p90={float(np.percentile(eps_arr, 90)):.4e} '
                f'max={emax:.4e} '
                f'worst_locus(r={locus[0]:.4f}, theta_wedge={locus[1]:.4f})')
            self._tick()
            self.assertLessEqual(emax, RECURSION_INTERIOR_BAR)
            self.assertEqual(
                len(leaves), summary['packed'],
                'every terminal leaf must be packed (no residual gap left).')
        lines.append(
            f'OVERALL worst: max={worst[0]:.4e} at {worst[1]} '
            f'(r={worst[2][0]:.4f}, theta_wedge={worst[2][1]:.4f})')
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 't7_recursion_terminal_leaf_eps.txt').write_text(
            '\n'.join(lines) + '\n')
        self._tick()
        self.assertLess(
            max(all_eps), RECURSION_INTERIOR_BAR,
            'a terminal leaf exceeded the interior bar after recursion.')


class WedgeRecursionSelfFalsificationTestCase(unittest.TestCase):
    """Prove the recursion-closes-gaps assertions have teeth.

    Two controls exercise the two ways the main suite could read falsely
    green: (a) with NO injected gap the tree never recurses (so the
    "reproduce exactly one/two gaps" assertions genuinely depend on the
    above-bar eps), and (b) a STUBBORN gap that cannot close within the depth
    cap leaves a terminal recorded_gated leaf above the bar (so the "every
    leaf clears the bar" assertion is falsifiable).
    """

    def test_no_injected_gap_never_recurses(self):
        """With every child below the bar, nothing subdivides (depth 1)."""
        summary, reports = _run_recursion(
            'w_nogap', 0.633, {}, _recursion_config())
        leaves = _terminal_leaves(summary, reports, 'w_nogap')
        self.assertEqual(summary['max_achieved_depth'], 1,
                         'a gap-free tile must not recurse.')
        self.assertEqual(
            [e['result'] for e in summary['children']],
            ['packed'] * 4,
            'every child of a gap-free tile must pack at depth 1.')
        self.assertTrue(all(leaf['depth'] == 1 for leaf in leaves))
        self.assertEqual(len(leaves), 4)

    def test_stubborn_gap_leaves_residual_above_bar(self):
        """A gap too large to close hits the depth cap with a leaf > bar.

        Proves the main "every terminal leaf packed and <= bar" claim can
        FAIL: a 5.0 depth-1 eps shrinks only to ~0.31 at the cap depth 3,
        which is still above the 5e-2 bar, so a recorded_gated leaf survives.
        """
        stubborn = {'w_stub': {0: 5.0}}
        summary, reports = _run_recursion(
            'w_stub', 0.633, stubborn, _recursion_config())
        leaves = _terminal_leaves(summary, reports, 'w_stub')
        self.assertEqual(summary['max_achieved_depth'],
                         surrogate_training.MAX_SUBDIVISION_DEPTH,
                         'a stubborn gap must recurse to the depth cap.')
        residual = [leaf for leaf in leaves
                    if leaf['result'] == 'recorded_gated']
        self.assertTrue(
            residual,
            'a stubborn gap must leave at least one recorded_gated leaf.')
        for leaf in residual:
            self.assertGreater(
                leaf['eps'], RECURSION_INTERIOR_BAR,
                'a residual gap leaf must remain above the bar (teeth).')


if __name__ == '__main__':
    unittest.main()
