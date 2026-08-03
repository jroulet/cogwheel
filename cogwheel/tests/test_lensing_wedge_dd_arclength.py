"""Tests for InteriorWedgeChart DD w-ceiling and arc-length axis features.

This suite verifies WP1's two new capabilities added to ``from_wedge_engine``:

1. **DD-product w-ceiling** — the dimensionless-frequency upper limit is
   capped at ``_DD_PRODUCT_MARGIN / (r_max * reach_max)`` so no training
   node violates the engine's diffraction-delay ceiling.

2. **Caustic arc-length axis** — the chart's spline fourth axis uses the
   arc-length ``s`` parametrisation (via ``theta_to_s``) rather than raw
   ``theta_wedge``, improving interpolation fidelity near cusps.

Tolerance justification
-----------------------
Node-exact accuracy (< 1e-10): the cubic B-spline exactly reproduces its
training values at grid nodes (up to O(eps_mach) from the fit residual).
The theta_to_s remap does not degrade this because both training and
serving use the same monotone map (theta → s) — the s-coordinate at a
grid node is looked up through the SAME table on both sides.

Cost budget
-----------
4×4×4 = 64 nodes per build.  The DD cap reduces w_max from 500 to ~30,
yielding ~6 w-nodes per decade over [5, 30] ≈ 5 points.  Total:
64 × 5 evaluations × ~30ms = ~10s per build.  Three builds in this suite
(DD-cap, arc-length, no-DD-cap) share one via setUpClass = ~10s + ~5s + ~5s
≈ 20s total.  Well within the 5-minute ceiling.
"""
from __future__ import annotations

import copy
import unittest

import numpy as np
from scipy.integrate import cumulative_trapezoid

from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels
from cogwheel.lensing.chang_refsdal.geometry import caustic_speed, r_caustic
from cogwheel.lensing.surrogate import (
    InteriorWedgeChart,
    LensAmplificationSurrogate,
    _DD_PRODUCT_MARGIN,
    _evaluate_chart,
    _from_wedge_fixed,
)

# ---------------------------------------------------------------------------
#: Module-level test constants — DD-cap-triggering fixture
# ---------------------------------------------------------------------------

#: Gamma range (positive parity interior).
DD_GAMMA_RANGE: tuple[float, float] = (0.30, 0.50)

#: Radial axis bounds — wide enough that r_max * reach_max triggers the cap.
DD_R_RANGE: tuple[float, float] = (0.15, 0.70)

#: Wedge-angle range inside the first quadrant.
DD_THETA_RANGE: tuple[float, float] = (0.20, 1.30)

#: Frequency range — w_max = 500 is far above the DD cap at this geometry.
DD_W_RANGE: tuple[float, float] = (5.0, 500.0)

#: Nodes per spatial axis (minimum 4 for cubic spline validation).
DD_N_GAMMA: int = 4
DD_N_R: int = 4
DD_N_THETA: int = 4

#: W-nodes per decade (sparse for speed).
DD_W_NODES_PER_DECADE: int = 8

#: The DD margin constant (duplicated here for the test's oracle).
DD_MARGIN: float = 58.0

#: Node-exact accuracy tolerance.
NODE_ATOL: float = 1e-9


class _WedgeDDTestCase(unittest.TestCase):
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


class DDWCeilingTestCase(_WedgeDDTestCase):
    """Verify the DD-product w-ceiling is applied by from_wedge_engine.

    Spec: Build a wedge chart with parameters that trigger the DD
    constraint (w_range upper end >> DD_MARGIN/(r_max*reach_max)).
    The returned surrogate's chart w_max must satisfy the DD formula.

    The DD cap prevents the engine from receiving requests where
    w * |y| > 58 (those would be refused as DD-product violations).
    For this geometry, the DD cap gives w_max≈121.6, which is below
    the requested 500 but above the Schwinger ceiling (~60).  Most
    refusals at the capped w_max are Schwinger-related (not DD),
    so we verify the FORMULA not the success rate.

    Cost: 4×4×4 = 64 nodes × ~13 w-points × 30ms ≈ 25s.
    """

    _surrogate: LensAmplificationSurrogate | None = None

    @classmethod
    def setUpClass(cls):
        """Build the surrogate via from_wedge_engine (end-to-end)."""
        cls._surrogate = LensAmplificationSurrogate.from_wedge_engine(
            gamma_range=DD_GAMMA_RANGE,
            r_range=DD_R_RANGE,
            theta_wedge_range=DD_THETA_RANGE,
            w_range=DD_W_RANGE,
            n_gamma=DD_N_GAMMA,
            n_r=DD_N_R,
            n_theta_wedge=DD_N_THETA,
            w_nodes_per_decade=DD_W_NODES_PER_DECADE)

    def test_single_chart_returned(self):
        """The surrogate must contain exactly one chart."""
        self._tick()
        self.assertEqual(len(self._surrogate.charts), 1)

    def test_chart_is_interior_wedge(self):
        """The chart must be an InteriorWedgeChart instance."""
        self._tick()
        self.assertIsInstance(self._surrogate.charts[0], InteriorWedgeChart)

    def test_w_max_respects_dd_cap(self):
        """exp(log_w_grid[-1]) <= DD_MARGIN / (r_max * reach_max).

        The DD ceiling formula: w * |y| <= 58, with |y| = r * reach.
        At the largest r in the grid, the cap binds first.
        """
        chart = self._surrogate.charts[0]
        w_max_chart = float(np.exp(chart.log_w_grid[-1]))
        r_max = float(chart.r_grid[-1])

        # Compute reach_max from the wedge map over the theta/gamma range.
        theta_mask = (
            (chart.wedge_map.theta_nodes >= DD_THETA_RANGE[0])
            & (chart.wedge_map.theta_nodes <= DD_THETA_RANGE[1]))
        reach_max = float(chart.wedge_map.r_table[:, theta_mask].max())

        dd_cap = DD_MARGIN / (r_max * reach_max)
        self._tick()
        self.assertLessEqual(
            w_max_chart, dd_cap + 1e-10,  # float tolerance
            f'Chart w_max={w_max_chart:.2f} exceeds DD cap={dd_cap:.2f}. '
            f'The DD ceiling was not applied.')

    def test_w_max_below_requested(self):
        """The capped w_max must be strictly below the requested 500."""
        chart = self._surrogate.charts[0]
        w_max_chart = float(np.exp(chart.log_w_grid[-1]))
        self._tick()
        self.assertLess(
            w_max_chart, DD_W_RANGE[1],
            f'Chart w_max={w_max_chart:.2f} was not capped below the '
            f'requested {DD_W_RANGE[1]}. DD constraint not binding.')

    def test_dd_product_never_exceeds_margin(self):
        """The max DD product (w_max * r_max * reach_max) <= 58.

        This is the invariant the DD cap is designed to guarantee:
        no training node can have w * |y| > DD_MARGIN.
        """
        chart = self._surrogate.charts[0]
        w_max_chart = float(np.exp(chart.log_w_grid[-1]))
        r_max = float(chart.r_grid[-1])
        theta_mask = (
            (chart.wedge_map.theta_nodes >= DD_THETA_RANGE[0])
            & (chart.wedge_map.theta_nodes <= DD_THETA_RANGE[1]))
        reach_max = float(chart.wedge_map.r_table[:, theta_mask].max())
        product = w_max_chart * r_max * reach_max
        self._tick()
        self.assertLessEqual(
            product, DD_MARGIN + 1e-6,
            f'DD product w*r*reach = {product:.2f} exceeds margin {DD_MARGIN}.')

    def test_refused_fewer_than_total(self):
        """Some nodes must succeed — not all refused."""
        chart = self._surrogate.charts[0]
        total_nodes = DD_N_GAMMA * DD_N_R * DD_N_THETA
        self._tick()
        self.assertLess(
            chart.refused_points.shape[0], total_nodes,
            'All nodes refused — the surrogate would be empty.')


# ---------------------------------------------------------------------------
#: Module-level constants — arc-length axis fixture (low-w, all nodes pass)
# ---------------------------------------------------------------------------

#: Gamma range for the arc-length test (same as DD).
ARC_GAMMA_RANGE: tuple[float, float] = (0.30, 0.50)

#: R range — moderate, away from caustic boundary.
ARC_R_RANGE: tuple[float, float] = (0.20, 0.50)

#: Theta wedge range.
ARC_THETA_RANGE: tuple[float, float] = (0.30, 1.20)

#: Low w_range that does NOT trigger Schwinger or DD issues.
ARC_W_RANGE: tuple[float, float] = (5.0, 30.0)

#: Nodes per spatial axis.
ARC_N_GAMMA: int = 4
ARC_N_R: int = 4
ARC_N_THETA: int = 4

#: W nodes per decade.
ARC_W_NODES_PER_DECADE: int = 10

#: Minimum number of arc-length map nodes (spec: >= 100).
ARC_MIN_MAP_NODES: int = 100


class ArcLengthAxisTestCase(_WedgeDDTestCase):
    """Verify the arc-length axis is active and correctly wired.

    Spec: Build a wedge chart via from_wedge_engine at low w where all
    nodes succeed.  The chart's theta_to_s attribute must be populated,
    have the correct shape, span the theta range, and yield node-exact
    accuracy through the s-remap.

    Cost: 4×4×4 = 64 nodes × ~4 w-points × 30ms ≈ 8s build +
    3 evaluation calls.
    """

    _surrogate: LensAmplificationSurrogate | None = None

    @classmethod
    def setUpClass(cls):
        """Build a low-w surrogate where all/most nodes succeed."""
        cls._surrogate = LensAmplificationSurrogate.from_wedge_engine(
            gamma_range=ARC_GAMMA_RANGE,
            r_range=ARC_R_RANGE,
            theta_wedge_range=ARC_THETA_RANGE,
            w_range=ARC_W_RANGE,
            n_gamma=ARC_N_GAMMA,
            n_r=ARC_N_R,
            n_theta_wedge=ARC_N_THETA,
            w_nodes_per_decade=ARC_W_NODES_PER_DECADE)

    def test_theta_to_s_not_none(self):
        """The chart must have a theta_to_s attribute (arc-length active)."""
        chart = self._surrogate.charts[0]
        self._tick()
        self.assertIsNotNone(
            chart.theta_to_s,
            'theta_to_s is None — arc-length axis was not built.')

    def test_theta_to_s_shape(self):
        """theta_to_s must have shape (2, N) with N >= 100."""
        chart = self._surrogate.charts[0]
        theta_to_s = chart.theta_to_s
        self._tick()
        self.assertEqual(
            theta_to_s.ndim, 2,
            f'theta_to_s.ndim = {theta_to_s.ndim}, expected 2.')
        self.assertEqual(
            theta_to_s.shape[0], 2,
            f'theta_to_s.shape[0] = {theta_to_s.shape[0]}, expected 2.')
        self._tick()
        self.assertGreaterEqual(
            theta_to_s.shape[1], ARC_MIN_MAP_NODES,
            f'theta_to_s has only {theta_to_s.shape[1]} columns; '
            f'expected >= {ARC_MIN_MAP_NODES}.')

    def test_theta_row_spans_grid(self):
        """Row 0 must start at theta_wedge_grid[0], end at grid[-1]."""
        chart = self._surrogate.charts[0]
        theta_row = chart.theta_to_s[0]
        self._tick()
        self.assertAlmostEqual(
            float(theta_row[0]), float(chart.theta_wedge_grid[0]),
            places=12,
            msg='theta_to_s row 0 start != theta_wedge_grid[0].')
        self._tick()
        self.assertAlmostEqual(
            float(theta_row[-1]), float(chart.theta_wedge_grid[-1]),
            places=12,
            msg='theta_to_s row 0 end != theta_wedge_grid[-1].')

    def test_s_row_starts_near_zero(self):
        """Row 1 (arc-length) must start at ~0.0."""
        chart = self._surrogate.charts[0]
        s_row = chart.theta_to_s[1]
        self._tick()
        self.assertAlmostEqual(
            float(s_row[0]), 0.0, places=10,
            msg=f's_row[0] = {s_row[0]:.2e}, expected ~0.0.')

    def test_s_row_strictly_increasing(self):
        """Row 1 must be strictly increasing (valid arc-length)."""
        chart = self._surrogate.charts[0]
        s_row = chart.theta_to_s[1]
        diffs = np.diff(s_row)
        self._tick()
        self.assertTrue(
            np.all(diffs > 0),
            f'Arc-length row is not strictly increasing. '
            f'Min diff = {float(diffs.min()):.2e}.')

    def test_s_row_not_linear(self):
        """Arc-length must NOT be linear in theta (proves curvature present).

        A linear s(theta) would mean raw theta was used (identity map).
        The astroid caustic speed varies with theta, so the arc-length
        parametrisation must be nonlinear.
        """
        chart = self._surrogate.charts[0]
        theta_row = chart.theta_to_s[0]
        s_row = chart.theta_to_s[1]
        # Linear fit residual: if s = a*theta + b exactly, the map is trivial.
        coeffs = np.polyfit(theta_row, s_row, 1)
        linear_pred = np.polyval(coeffs, theta_row)
        residual = float(np.max(np.abs(s_row - linear_pred)))
        self._tick()
        self.assertGreater(
            residual, 1e-4,
            f'Arc-length is nearly linear (max residual={residual:.2e}). '
            f'This suggests the curvature-based remap is not active.')

    def test_grid_node_accuracy(self):
        """Served envelope at grid nodes matches training to < 1e-9.

        The cubic B-spline reproduces training values at grid nodes.
        The theta→s remap does not degrade this because both training
        and serving use the same map.  Cost: 3 evaluations.
        """
        chart = self._surrogate.charts[0]
        total = DD_N_GAMMA * DD_N_R * DD_N_THETA
        refused_set = set()
        if chart.refused_points.shape[0] > 0:
            for row in chart.refused_points:
                refused_set.add(tuple(np.round(row, 10)))

        # Find 3 succeeded interior nodes.
        succeeded_nodes: list[tuple[int, int, int]] = []
        for ig in range(1, chart.gamma_grid.size - 1):
            for ir in range(1, chart.r_grid.size - 1):
                for it in range(1, chart.theta_wedge_grid.size - 1):
                    pt = (float(chart.gamma_grid[ig]),
                          float(chart.r_grid[ir]),
                          float(chart.theta_wedge_grid[it]))
                    # Check if refused.
                    is_refused = False
                    for row in chart.refused_points:
                        if (abs(row[0] - pt[0]) < 1e-8
                                and abs(row[1] - pt[1]) < 1e-8
                                and abs(row[2] - pt[2]) < 1e-8):
                            is_refused = True
                            break
                    if not is_refused:
                        succeeded_nodes.append((ig, ir, it))
                    if len(succeeded_nodes) >= 3:
                        break
                if len(succeeded_nodes) >= 3:
                    break
            if len(succeeded_nodes) >= 3:
                break

        self.assertGreaterEqual(
            len(succeeded_nodes), 1,
            'No interior succeeded nodes found for accuracy test.')

        for ig, ir, it in succeeded_nodes[:3]:
            gamma = float(chart.gamma_grid[ig])
            r = float(chart.r_grid[ir])
            theta_w = float(chart.theta_wedge_grid[it])

            # Evaluate chart at this node.
            y1, y2 = _from_wedge_fixed(gamma, r, theta_w, chart.wedge_map)
            result = _evaluate_chart(
                chart, gamma, eta=0.5, theta=0.7,
                log_w_query=chart.log_w_grid,
                y1_eig=y1, y2_eig=y2)

            # The result must be finite.
            self.assertTrue(
                np.all(np.isfinite(result)),
                f'Non-finite result at node ({ig},{ir},{it}).')

            # Compare against a fresh engine call at the same source.
            w_grid = np.exp(chart.log_w_grid)
            ch = ChangRefsdalChannels(w_grid)
            partition = ch.evaluate(gamma=gamma, y=(y1, y2),
                                    beta=0.0, kappa=0.0)
            engine_env = partition.envelope
            max_diff = float(np.max(np.abs(result - engine_env)))
            self._tick()
            with self.subTest(ig=ig, ir=ir, it=it):
                self.assertLess(
                    max_diff, NODE_ATOL,
                    f'Grid-node residual {max_diff:.2e} exceeds '
                    f'{NODE_ATOL:.0e} at node ({ig},{ir},{it}).')


# ---------------------------------------------------------------------------
#: Module-level constants — non-triggering DD cap fixture
# ---------------------------------------------------------------------------

#: Low w_range that does NOT trigger the DD cap.
NODD_W_RANGE: tuple[float, float] = (5.0, 15.0)


class NoDDCapLowWTestCase(_WedgeDDTestCase):
    """Verify behaviour when the DD cap is NOT binding.

    Spec: Build a wedge chart with a low w_range (5-15) that does not
    trigger the DD cap.  The chart's w_max should equal the requested 15
    (not capped).  The arc-length axis is still active, and grid-node
    accuracy remains < 1e-10.

    Cost: same as ArcLengthAxisTestCase (~8s build + 3 evals).
    """

    _surrogate: LensAmplificationSurrogate | None = None

    @classmethod
    def setUpClass(cls):
        """Build surrogate with low w_range (DD cap not binding)."""
        cls._surrogate = LensAmplificationSurrogate.from_wedge_engine(
            gamma_range=ARC_GAMMA_RANGE,
            r_range=ARC_R_RANGE,
            theta_wedge_range=ARC_THETA_RANGE,
            w_range=NODD_W_RANGE,
            n_gamma=ARC_N_GAMMA,
            n_r=ARC_N_R,
            n_theta_wedge=ARC_N_THETA,
            w_nodes_per_decade=ARC_W_NODES_PER_DECADE)

    def test_dd_cap_not_binding(self):
        """Chart w_max equals the requested 15.0 (not capped)."""
        chart = self._surrogate.charts[0]
        w_max_chart = float(np.exp(chart.log_w_grid[-1]))
        self._tick()
        self.assertAlmostEqual(
            w_max_chart, NODD_W_RANGE[1], places=5,
            msg=f'Chart w_max={w_max_chart:.4f} != requested '
                f'{NODD_W_RANGE[1]}. DD cap should not be binding.')

    def test_arc_length_still_active(self):
        """theta_to_s must still be populated even without DD cap."""
        chart = self._surrogate.charts[0]
        self._tick()
        self.assertIsNotNone(
            chart.theta_to_s,
            'theta_to_s is None at low w_range — arc-length should always '
            'be active regardless of DD cap.')

    def test_node_accuracy_preserved(self):
        """Grid-node accuracy < 1e-10 even with arc-length remap.

        The s-remap does not degrade node-exact reproduction because
        training and serving use the same theta_to_s table.
        """
        chart = self._surrogate.charts[0]

        # Find 3 succeeded interior nodes.
        succeeded_nodes: list[tuple[int, int, int]] = []
        for ig in range(1, chart.gamma_grid.size - 1):
            for ir in range(1, chart.r_grid.size - 1):
                for it in range(1, chart.theta_wedge_grid.size - 1):
                    pt = (float(chart.gamma_grid[ig]),
                          float(chart.r_grid[ir]),
                          float(chart.theta_wedge_grid[it]))
                    is_refused = False
                    for row in chart.refused_points:
                        if (abs(row[0] - pt[0]) < 1e-8
                                and abs(row[1] - pt[1]) < 1e-8
                                and abs(row[2] - pt[2]) < 1e-8):
                            is_refused = True
                            break
                    if not is_refused:
                        succeeded_nodes.append((ig, ir, it))
                    if len(succeeded_nodes) >= 3:
                        break
                if len(succeeded_nodes) >= 3:
                    break
            if len(succeeded_nodes) >= 3:
                break

        self.assertGreaterEqual(
            len(succeeded_nodes), 1,
            'No interior nodes succeeded for accuracy test.')

        for ig, ir, it in succeeded_nodes[:3]:
            gamma = float(chart.gamma_grid[ig])
            r = float(chart.r_grid[ir])
            theta_w = float(chart.theta_wedge_grid[it])
            y1, y2 = _from_wedge_fixed(gamma, r, theta_w, chart.wedge_map)
            result = _evaluate_chart(
                chart, gamma, eta=0.5, theta=0.7,
                log_w_query=chart.log_w_grid,
                y1_eig=y1, y2_eig=y2)

            # Fresh engine oracle.
            w_grid = np.exp(chart.log_w_grid)
            ch = ChangRefsdalChannels(w_grid)
            partition = ch.evaluate(gamma=gamma, y=(y1, y2),
                                    beta=0.0, kappa=0.0)
            engine_env = partition.envelope
            max_diff = float(np.max(np.abs(result - engine_env)))
            self._tick()
            with self.subTest(ig=ig, ir=ir, it=it):
                self.assertLess(
                    max_diff, 1e-10,
                    f'Grid-node residual {max_diff:.2e} exceeds 1e-10 '
                    f'at node ({ig},{ir},{it}) — arc-length remap '
                    f'degraded node-exact accuracy.')

    def test_most_nodes_succeed_at_low_w(self):
        """At low w (no Schwinger/DD issues), most nodes should succeed."""
        chart = self._surrogate.charts[0]
        total = ARC_N_GAMMA * ARC_N_R * ARC_N_THETA
        refused = chart.refused_points.shape[0]
        succeeded = total - refused
        fraction = succeeded / total
        self._tick()
        self.assertGreaterEqual(
            fraction, 0.80,
            f'Only {succeeded}/{total} = {fraction:.0%} nodes succeeded '
            f'at low w. Expected >= 80% when Schwinger is not binding.')


class SelfFalsificationTestCase(_WedgeDDTestCase):
    """Prove the suite can go red: reachable-red tests for each claim.

    Each test verifies that the corresponding assertion in the main
    classes WOULD fail if the feature were broken.  Without these, the
    suite could pass vacuously (e.g., if from_wedge_engine silently
    skipped the DD cap or the arc-length map).
    """

    def test_dd_cap_teeth_uncapped_would_exceed(self):
        """If we DON'T apply the DD cap, w_max would exceed the formula.

        Teeth: at r_max=0.7, reach≈0.68, the DD cap is 58/(0.7*0.68)≈121.6.
        The requested 500 >> 121.6, so without the cap the product would be
        500 * 0.7 * 0.68 ≈ 238 >> 58.
        """
        r_max = DD_R_RANGE[1]
        # Approximate reach_max (known from our measurements to be ~0.68).
        # reach at gamma=0.5, theta=0 (axis) is the max for this range.
        reach_approx = max(
            r_caustic(0.5, th)
            for th in np.linspace(DD_THETA_RANGE[0], DD_THETA_RANGE[1], 20))
        uncapped_product = DD_W_RANGE[1] * r_max * reach_approx
        self._tick()
        self.assertGreater(
            uncapped_product, DD_MARGIN,
            f'Uncapped product {uncapped_product:.1f} <= {DD_MARGIN}. '
            f'The DD cap would be non-binding (test has no teeth).')

    def test_arc_length_teeth_linear_would_fail(self):
        """If theta_to_s were an identity (linear), the nonlinearity test fails.

        Teeth: a perfectly linear s = a*theta + b has zero residual from
        a degree-1 polyfit.  The real arc-length has curvature > 1e-4.
        """
        # Synthesize a linear map.
        theta = np.linspace(0.3, 1.2, 2001)
        s_linear = theta - theta[0]  # identity shift
        coeffs = np.polyfit(theta, s_linear, 1)
        pred = np.polyval(coeffs, theta)
        residual = float(np.max(np.abs(s_linear - pred)))
        self._tick()
        self.assertLess(
            residual, 1e-10,
            'Linear map residual is not ~0 — polyfit control broken.')
        # Now check that a REAL arc-length map has curvature.
        gamma = 0.4  # representative
        speed = np.asarray(caustic_speed(gamma, theta, branch=1), dtype=float)
        s_real = cumulative_trapezoid(speed, theta, initial=0.0)
        coeffs_real = np.polyfit(theta, s_real, 1)
        pred_real = np.polyval(coeffs_real, theta)
        residual_real = float(np.max(np.abs(s_real - pred_real)))
        self._tick()
        self.assertGreater(
            residual_real, 1e-4,
            f'Real arc-length residual {residual_real:.2e} too small — '
            f'curvature test would not distinguish arc-length from linear.')

    def test_node_accuracy_teeth_wrong_map_degrades(self):
        """If we evaluate a grid node with a WRONG theta_to_s, accuracy degrades.

        Teeth: perturbing the theta→s map (scaling s by 1.1) causes the
        spline to be queried at the wrong coordinate, yielding a non-zero
        residual at what was a grid node.
        """
        # Use the ArcLengthAxisTestCase surrogate if available, else build.
        surr = LensAmplificationSurrogate.from_wedge_engine(
            gamma_range=ARC_GAMMA_RANGE,
            r_range=ARC_R_RANGE,
            theta_wedge_range=ARC_THETA_RANGE,
            w_range=NODD_W_RANGE,
            n_gamma=ARC_N_GAMMA,
            n_r=ARC_N_R,
            n_theta_wedge=ARC_N_THETA,
            w_nodes_per_decade=ARC_W_NODES_PER_DECADE)
        chart = surr.charts[0]

        # Find a succeeded interior node.
        for ig in range(1, chart.gamma_grid.size - 1):
            for ir in range(1, chart.r_grid.size - 1):
                for it in range(1, chart.theta_wedge_grid.size - 1):
                    is_refused = False
                    for row in chart.refused_points:
                        if (abs(row[0] - chart.gamma_grid[ig]) < 1e-8
                                and abs(row[1] - chart.r_grid[ir]) < 1e-8
                                and abs(row[2] - chart.theta_wedge_grid[it]) < 1e-8):
                            is_refused = True
                            break
                    if not is_refused:
                        gamma = float(chart.gamma_grid[ig])
                        r = float(chart.r_grid[ir])
                        theta_w = float(chart.theta_wedge_grid[it])
                        y1, y2 = _from_wedge_fixed(gamma, r, theta_w,
                                                   chart.wedge_map)

                        # Normal evaluation (should be accurate).
                        result_ok = _evaluate_chart(
                            chart, gamma, eta=0.5, theta=0.7,
                            log_w_query=chart.log_w_grid,
                            y1_eig=y1, y2_eig=y2)

                        # Perturb the chart's theta_to_s map.
                        bad_chart = copy.copy(chart)
                        perturbed_map = chart.theta_to_s.copy()
                        perturbed_map[1] *= 1.1  # stretch s by 10%
                        object.__setattr__(bad_chart, 'theta_to_s',
                                           perturbed_map)

                        result_bad = _evaluate_chart(
                            bad_chart, gamma, eta=0.5, theta=0.7,
                            log_w_query=chart.log_w_grid,
                            y1_eig=y1, y2_eig=y2)

                        diff = float(np.max(np.abs(result_ok - result_bad)))
                        self._tick()
                        self.assertGreater(
                            diff, 1e-6,
                            f'Perturbed theta_to_s gives same result — '
                            f'the map is not load-bearing (diff={diff:.2e}).')
                        return  # one node is enough

        self.fail('No succeeded interior node found for falsification test.')


if __name__ == '__main__':
    unittest.main()
