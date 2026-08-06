"""Tests for InteriorWedgeChart DD w-ceiling and cusp-adapted angular axis.

This suite verifies two capabilities of ``from_wedge_engine``:

1. **DD-product w-ceiling** — the dimensionless-frequency upper limit is
   capped at ``_DD_PRODUCT_MARGIN / (r_max * reach_max)`` so no training
   node violates the engine's diffraction-delay ceiling.  This block is
   UNTOUCHED by WP1/WP2; ``DDWCeilingTestCase`` re-confirms the cap still
   binds and the wedge build still succeeds under it.

2. **Cusp-adapted angular axis (WP1/WP2)** — the chart's spline angular
   axis is reparametrised by ``u = d**(2/3)`` where ``d`` is the angular
   distance to the NEAR astroid cusp (``theta_wedge = 0`` on the LOW side
   of the caustic waist, ``pi/2`` on the HIGH side).  The ``2/3`` exponent
   is the exact caustic-reach cusp scaling (``r_caustic ~ const - c *
   d**(2/3)``), so the spline coordinate stays smooth instead of diverging
   as ``d**(-1/3)`` on the raw ``theta`` axis.  The dense ``theta -> u`` map
   is stored as the chart's ``theta_to_s`` ``(2, N)`` table (row 0 =
   ``theta_fine``, row 1 = ``u_fine``) and consumed at serve time via
   ``np.interp`` (see ``_evaluate_chart``).  This replaces the retired
   arc-length axis (``ArcLengthAxisTestCase`` is gone).

Tolerance justification
-----------------------
Closed-form axis match (< 1e-9): ``theta_to_s`` row 1 is the algebraic
image of row 0 under the per-side closed form
``u = theta**(2/3) - theta_lo**(2/3)`` (LOW) /
``u = (pi/2 - theta_lo)**(2/3) - (pi/2 - theta)**(2/3)`` (HIGH); FP
round-off in the ``(2/3)->(3/2)`` round trip is ~1e-16 relative, so the
1e-9 bar carries ~7 orders of margin.  Uniform-in-u (< 1e-9): row 1 is
``np.linspace(0, u_max, N)`` so successive differences are constant to
machine precision, which forces the ``theta`` nodes to cluster near the
cusp (``theta ~ (u + base)**(3/2)``).

Serve degradation (> 5e-2): the interior off-node accuracy bar.  A clean
u-map serves within it; corrupting a fraction of the stored row moves the
served envelope past it, proving the map is load-bearing on the serve path
(``SelfFalsificationTestCase``).

Cost budget
-----------
The fast-tier classes build 4x4x4 = 64-node charts at ``w <= 20`` (below
the Schwinger double-double ceiling, ~0.2 s/eval): ``CuspAdaptedAxisTestCase``
builds two (LOW + HIGH sides, ~16 s), ``NoDDCapLowWTestCase`` one (~8 s),
and ``SelfFalsificationTestCase`` one (~8 s) plus ~27 fast engine oracle
calls.  Whole file stays under the 5-minute fast-tier ceiling.

``DDWCeilingTestCase`` is the exception and is SLOW-TIERED
(``COGWHEEL_TRAIN_TIER``): its DD cap lands at ``w_max ~ 121.6`` — above
the Schwinger ceiling (60) — so its capped-w nodes take the mpmath path at
~85-120 s EACH (F061).  No assertion in it was weakened; only its gate is
train-tier.
"""
from __future__ import annotations

import copy
import os
import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels
from cogwheel.lensing.chang_refsdal.geometry import r_caustic
from cogwheel.lensing.surrogate import (
    InteriorWedgeChart,
    LensAmplificationSurrogate,
    _DD_PRODUCT_MARGIN,
    _FARFIELD_ARC_MAP_SIZE,
    _evaluate_chart,
    _from_wedge_fixed,
    _log_reach_gamma_axis,
    _wedge_cusp_axis_map,
    _wedge_theta_waist,
)

#: Diagnostic-plot / report output directory (created on demand).
OUTPUT_DIR = Path(__file__).resolve().parent / 'output'

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


#: `DDWCeilingTestCase` is the one class here whose geometry forces training
#: nodes above the Schwinger double-double ceiling (``w > 60``), where each
#: evaluation costs ~85-120 s on the mpmath path instead of ~0.2 s (F061).
#: The DD cap cannot be pushed under 60 for the astroid, so the cost is
#: intrinsic to what the class tests rather than a fixture mistake.
_TRAIN_TIER_SKIP = unittest.skipUnless(
    os.environ.get('COGWHEEL_TRAIN_TIER'),
    'engine-backed training tier: set COGWHEEL_TRAIN_TIER=1 (builds real '
    'surrogate charts above the Schwinger ceiling at ~85-120 s per node; '
    'the driver runs these post-build)')


@_TRAIN_TIER_SKIP
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

    This block is UNCHANGED by WP1/WP2 (the cusp-axis build); it is
    re-run to confirm the w*r*reach_max<=58 cap still binds and the
    wedge build still succeeds under the cap.

    Cost: 4×4×4 = 64 nodes × ~13 w-points.  Nodes at the capped w_max sit
    ABOVE the Schwinger ceiling (60) and cost ~85-120 s each on the mpmath
    path, not ~30 ms (F061) — hence the training-tier gate on this class.
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
        """Some nodes must succeed — not all refused (build succeeds)."""
        chart = self._surrogate.charts[0]
        total_nodes = DD_N_GAMMA * DD_N_R * DD_N_THETA
        self._tick()
        self.assertLess(
            chart.refused_points.shape[0], total_nodes,
            'All nodes refused — the surrogate would be empty.')


# ---------------------------------------------------------------------------
#: Module-level constants — cusp-adapted angular axis fixture (WP1/WP2)
# ---------------------------------------------------------------------------

#: Gamma range for the cusp-axis test (positive-parity interior).
CUSP_GAMMA_RANGE: tuple[float, float] = (0.30, 0.50)

#: R range — moderate interior, away from the caustic boundary.
CUSP_R_RANGE: tuple[float, float] = (0.20, 0.50)

#: Low w so every node sits below the Schwinger double-double ceiling.
CUSP_W_RANGE: tuple[float, float] = (5.0, 20.0)

#: Nodes per spatial axis (>= 4 for cubic-spline validation).
CUSP_N_GAMMA: int = 4
CUSP_N_R: int = 4
CUSP_N_THETA: int = 4

#: W nodes per decade (sparse for speed).
CUSP_W_NODES_PER_DECADE: int = 8

#: Closed-form reconstruction tolerance (spec: within 1e-9).
CUSP_FORM_ATOL: float = 1e-9

#: Uniform-in-u tolerance: successive u-differences constant to ~1e-9.
CUSP_UNIFORM_ATOL: float = 1e-9

#: The wrong-side closed form must differ from row 1 by at least this much
#: (teeth: proves the per-side form check discriminates LOW from HIGH).
CUSP_WRONG_SIDE_MIN: float = 1e-3

#: Expected dense-map node count (serve-time interpolation table).
CUSP_MAP_NODES: int = _FARFIELD_ARC_MAP_SIZE  # 2001


def _reconstruct_u(theta_fine: np.ndarray, origin: str) -> np.ndarray:
    """Closed-form cusp coordinate ``u(theta)`` for one wedge side.

    LOW  (near cusp ``theta = 0``):
        ``u = theta**(2/3) - theta_lo**(2/3)``
    HIGH (near cusp ``pi/2``):
        ``u = (pi/2 - theta_lo)**(2/3) - (pi/2 - theta)**(2/3)``

    with ``theta_lo = theta_fine[0]``.  Both are monotone increasing with
    ``u(theta_lo) = 0``.  This is an INDEPENDENT re-derivation of the
    production ``_wedge_cusp_axis_map`` form (its oracle), written from the
    cusp-scaling physics rather than transcribed from the function body.

    Parameters
    ----------
    theta_fine : np.ndarray
        Strictly increasing wedge angles spanning ``[theta_lo, theta_hi]``.
    origin : str
        ``'low'`` or ``'high'``.

    Returns
    -------
    np.ndarray
        The cusp-adapted coordinate ``u`` at each ``theta_fine`` node.
    """
    theta_fine = np.asarray(theta_fine, dtype=float)
    theta_lo = float(theta_fine[0])
    exponent = 2.0 / 3.0
    if origin == 'low':
        return theta_fine ** exponent - theta_lo ** exponent
    if origin == 'high':
        half_pi = np.pi / 2.0
        return ((half_pi - theta_lo) ** exponent
                - (half_pi - theta_fine) ** exponent)
    raise ValueError(f"origin must be 'low' or 'high'; got {origin!r}.")


class CuspAdaptedAxisTestCase(_WedgeDDTestCase):
    """Verify the cusp-adapted angular axis (``u = d**(2/3)``) on both sides.

    Spec (SHARD B): build the wedge u-map for a tile on each side of the
    caustic waist (``axis_origin`` LOW and HIGH), then inspect the stored
    ``theta_to_s`` map rows.  Row 1 (the ``u`` values) must be strictly
    increasing and offset so it starts at ~0; the per-side closed form must
    reconstruct it within 1e-9; and the fine grid must be uniform-in-u so
    the ``theta`` nodes cluster near the near cusp.

    The two tiles are placed relative to the waist for the SAME
    ``rep_gamma`` production uses (``median`` of the log-reach gamma axis),
    so ``from_wedge_engine``'s midpoint-vs-waist classification is
    deterministic and matches the side each chart is checked against.

    Cost: two 4×4×4 = 64-node charts at w<=20 (~16 s), shared across all
    tests via ``setUpClass``; every test only inspects the stored map (no
    per-test engine calls).
    """

    _sides: dict[str, InteriorWedgeChart] = {}
    _waist: float = float('nan')
    _rep_gamma: float = float('nan')

    @classmethod
    def setUpClass(cls):
        """Build one LOW-side and one HIGH-side wedge chart across the waist."""
        gamma_grid = _log_reach_gamma_axis(
            CUSP_GAMMA_RANGE, CUSP_N_GAMMA, 'gamma')
        rep_gamma = float(np.median(gamma_grid))
        waist = _wedge_theta_waist(rep_gamma)
        cls._rep_gamma = rep_gamma
        cls._waist = waist
        # LOW tile: midpoint = waist - 0.20 (< waist -> near cusp theta=0).
        low_range = (max(1e-2, waist - 0.35), waist - 0.05)
        # HIGH tile: midpoint = waist + 0.20 (> waist -> near cusp pi/2).
        high_range = (waist + 0.05, min(np.pi / 2.0 - 1e-2, waist + 0.35))
        cls._sides = {}
        for name, rng in (('low', low_range), ('high', high_range)):
            surr = LensAmplificationSurrogate.from_wedge_engine(
                gamma_range=CUSP_GAMMA_RANGE,
                r_range=CUSP_R_RANGE,
                theta_wedge_range=rng,
                w_range=CUSP_W_RANGE,
                n_gamma=CUSP_N_GAMMA,
                n_r=CUSP_N_R,
                n_theta_wedge=CUSP_N_THETA,
                w_nodes_per_decade=CUSP_W_NODES_PER_DECADE)
            cls._sides[name] = surr.charts[0]

    def _chart(self, side: str) -> InteriorWedgeChart:
        return self._sides[side]

    def test_stored_theta_to_s_shape_and_endpoints(self):
        """theta_to_s is a (2, 2001) table with exact tile-bound endpoints."""
        for side in ('low', 'high'):
            with self.subTest(side=side):
                chart = self._chart(side)
                t2s = chart.theta_to_s
                self.assertEqual(
                    t2s.shape, (2, CUSP_MAP_NODES),
                    f'theta_to_s shape {t2s.shape} != (2, {CUSP_MAP_NODES}).')
                self.assertAlmostEqual(
                    float(t2s[0, 0]), float(chart.theta_wedge_grid[0]),
                    places=12,
                    msg='theta_to_s row-0 start != theta_wedge_grid[0].')
                self.assertAlmostEqual(
                    float(t2s[0, -1]), float(chart.theta_wedge_grid[-1]),
                    places=12,
                    msg='theta_to_s row-0 end != theta_wedge_grid[-1].')
                self._tick()

    def test_storage_matches_direct_cusp_map(self):
        """Stored theta_to_s == vstack(_wedge_cusp_axis_map(...)) bit-for-bit.

        Proves the training path wires the cusp map (for the origin the
        waist classification selects) straight into the chart with no
        re-derivation drift.  Bit-identity also confirms the chart got the
        SIDE we placed it on (a wrong-origin build would mismatch loudly).
        """
        for side in ('low', 'high'):
            with self.subTest(side=side):
                chart = self._chart(side)
                theta_lo = float(chart.theta_wedge_grid[0])
                theta_hi = float(chart.theta_wedge_grid[-1])
                theta_fine, u_fine = _wedge_cusp_axis_map(
                    theta_lo, theta_hi, side)
                expected = np.vstack([theta_fine, u_fine])
                max_diff = float(np.max(np.abs(chart.theta_to_s - expected)))
                self._tick()
                self.assertEqual(
                    max_diff, 0.0,
                    f'Stored theta_to_s differs from _wedge_cusp_axis_map '
                    f'by {max_diff:.2e} on the {side} side.')

    def test_u_row_increasing_from_zero(self):
        """Row 1 (u) is strictly increasing and starts at ~0."""
        for side in ('low', 'high'):
            with self.subTest(side=side):
                u_row = self._chart(side).theta_to_s[1]
                self.assertAlmostEqual(
                    float(u_row[0]), 0.0, places=12,
                    msg=f'u_row[0] = {u_row[0]:.2e}, expected ~0.0.')
                diffs = np.diff(u_row)
                self.assertTrue(
                    np.all(diffs > 0),
                    f'u row not strictly increasing (min diff '
                    f'{float(diffs.min()):.2e}).')
                self._tick()

    def test_u_row_matches_per_side_closed_form(self):
        """Row 1 == per-side closed form to <1e-9; wrong side mismatches.

        LOW: u = theta**(2/3) - theta_lo**(2/3).
        HIGH: u = (pi/2 - theta_lo)**(2/3) - (pi/2 - theta)**(2/3).
        """
        for side, other in (('low', 'high'), ('high', 'low')):
            with self.subTest(side=side):
                chart = self._chart(side)
                theta_fine = chart.theta_to_s[0]
                u_row = chart.theta_to_s[1]

                u_expected = _reconstruct_u(theta_fine, side)
                max_err = float(np.max(np.abs(u_row - u_expected)))
                self.assertLess(
                    max_err, CUSP_FORM_ATOL,
                    f'{side}-side u row deviates from the closed form by '
                    f'{max_err:.2e} (> {CUSP_FORM_ATOL:.0e}).')

                # Teeth: the WRONG-side form must not spuriously match.
                u_wrong = _reconstruct_u(theta_fine, other)
                wrong_err = float(np.max(np.abs(u_row - u_wrong)))
                self.assertGreater(
                    wrong_err, CUSP_WRONG_SIDE_MIN,
                    f'{other}-side form matches the {side} row too closely '
                    f'({wrong_err:.2e}); the form check has no teeth.')
                self._tick()

    def test_grid_uniform_in_u(self):
        """Successive u-differences are constant to ~1e-9 (uniform-in-u)."""
        for side in ('low', 'high'):
            with self.subTest(side=side):
                u_row = self._chart(side).theta_to_s[1]
                du = np.diff(u_row)
                spread = float(np.max(np.abs(du - du.mean())))
                self.assertLess(
                    spread, CUSP_UNIFORM_ATOL,
                    f'{side}-side u grid not uniform: max|du - mean| = '
                    f'{spread:.2e} (> {CUSP_UNIFORM_ATOL:.0e}).')
                self._tick()

    def test_theta_nodes_cluster_toward_cusp(self):
        """theta node spacing tightens toward the near cusp, monotonically.

        LOW (cusp at theta=0): spacing smallest at theta_lo, growing away
        from the cusp -> np.diff(theta_fine) is monotone increasing.
        HIGH (cusp at pi/2): spacing smallest at theta_hi -> np.diff is
        monotone decreasing.
        """
        for side in ('low', 'high'):
            with self.subTest(side=side):
                theta_fine = self._chart(side).theta_to_s[0]
                dtheta = np.diff(theta_fine)
                if side == 'low':
                    self.assertLess(
                        dtheta[0], dtheta[-1],
                        'LOW spacing does not tighten toward theta=0.')
                    self.assertTrue(
                        np.all(np.diff(dtheta) > -1e-12),
                        'LOW spacing not monotone increasing away from cusp.')
                else:
                    self.assertLess(
                        dtheta[-1], dtheta[0],
                        'HIGH spacing does not tighten toward pi/2.')
                    self.assertTrue(
                        np.all(np.diff(dtheta) < 1e-12),
                        'HIGH spacing not monotone decreasing toward cusp.')
                self._tick()

    def test_diagnostic_plot_written(self):
        """Save u-vs-theta overlays for both sides (visual clustering check)."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for side in ('low', 'high'):
            with self.subTest(side=side):
                chart = self._chart(side)
                theta_fine = chart.theta_to_s[0]
                u_row = chart.theta_to_s[1]
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(theta_fine, u_row, '-', lw=1.0, color='navy',
                        label='u(theta)')
                # Sparse node markers reveal theta clustering toward the cusp.
                ax.plot(theta_fine[::100], u_row[::100], 'o', ms=3,
                        color='crimson', label='every 100th node')
                ax.set_xlabel('theta_wedge [rad]')
                ax.set_ylabel('u = d**(2/3)')
                ax.set_title(
                    f'Cusp-adapted wedge axis ({side} side, '
                    f'waist={self._waist:.3f})')
                ax.legend(loc='best', fontsize=8)
                fig.tight_layout()
                out = OUTPUT_DIR / f'cusp_adapted_axis_{side}.png'
                fig.savefig(out, dpi=90)
                plt.close(fig)
                self.assertTrue(
                    out.exists(), f'Diagnostic plot not written: {out}.')
                self._tick()


# ---------------------------------------------------------------------------
#: Module-level constants — non-DD-cap fixture shared by NoDDCap + falsify
# ---------------------------------------------------------------------------

#: Gamma range (positive-parity interior).
ARC_GAMMA_RANGE: tuple[float, float] = (0.30, 0.50)

#: R range — moderate, away from the caustic boundary.
ARC_R_RANGE: tuple[float, float] = (0.20, 0.50)

#: Theta wedge range.
ARC_THETA_RANGE: tuple[float, float] = (0.30, 1.20)

#: Nodes per spatial axis.
ARC_N_GAMMA: int = 4
ARC_N_R: int = 4
ARC_N_THETA: int = 4

#: W nodes per decade.
ARC_W_NODES_PER_DECADE: int = 10

#: Low w_range that does NOT trigger the DD cap.
NODD_W_RANGE: tuple[float, float] = (5.0, 15.0)


class NoDDCapLowWTestCase(_WedgeDDTestCase):
    """Verify behaviour when the DD cap is NOT binding.

    Spec: Build a wedge chart with a low w_range (5-15) that does not
    trigger the DD cap.  The chart's w_max should equal the requested 15
    (not capped).  The cusp-adapted axis is still active, and grid-node
    accuracy remains < 1e-10.

    Cost: one 4×4×4 = 64-node chart at w<=15 (~8s build + 3 evals).
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

    def test_cusp_axis_still_active(self):
        """theta_to_s must still be populated even without the DD cap."""
        chart = self._surrogate.charts[0]
        self._tick()
        self.assertIsNotNone(
            chart.theta_to_s,
            'theta_to_s is None at low w_range — the cusp-adapted axis '
            'should always be active regardless of the DD cap.')

    def test_node_accuracy_preserved(self):
        """Grid-node accuracy < 1e-10 even with the cusp-axis remap.

        The theta->u remap does not degrade node-exact reproduction because
        training and serving use the same theta_to_s table, and a grid
        node's theta lands on a knot of the u-axis.
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
                    f'at node ({ig},{ir},{it}) — the cusp-axis remap '
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


# ---------------------------------------------------------------------------
#: Module-level constants — u-map load-bearing falsification
# ---------------------------------------------------------------------------

#: Interior off-node accuracy bar (absolute max|F| residual over the w-grid).
#: A clean u-map serves within it; a corrupted one must exceed it.
INTERIOR_EPS_BAR: float = 5e-2

#: Central fraction [start, stop] of the dense map whose row-1 (u) values are
#: corrupted.  Wide enough that every interior cell midpoint (theta at
#: ~1/6..5/6 of the tile span) falls inside the corrupted theta window.
CORRUPT_BAND: tuple[float, float] = (0.05, 0.95)

#: Corruption offset added to the corrupted u values, as a fraction of the
#: u-axis span u_max.  Large enough to shift serve-time v2 well off the
#: clean interpolant.
CORRUPT_OFFSET_FRACTION: float = 0.30


class SelfFalsificationTestCase(_WedgeDDTestCase):
    """Prove the suite can go red: DD-cap teeth + load-bearing u-map serve.

    Two independent falsifications:

    1. ``test_dd_cap_teeth_uncapped_would_exceed`` — without the DD cap the
       requested w_max would blow past the DD product margin, so the
       ``DDWCeilingTestCase`` assertions are not vacuous.

    2. ``test_perturbed_umap_degrades_serve`` — corrupting a fraction of the
       stored ``theta_to_s`` row 1 (the ``u`` values) moves the served
       envelope past the interior accuracy bar relative to a fresh engine,
       proving the map is load-bearing on the SERVE path.  The clean-vs-
       degraded eps is reported as a distribution (p50/p90/max), not a bare
       max.
    """

    _chart: InteriorWedgeChart | None = None

    @classmethod
    def setUpClass(cls):
        """Build one low-w chart shared by the serve-degradation test."""
        surr = LensAmplificationSurrogate.from_wedge_engine(
            gamma_range=ARC_GAMMA_RANGE,
            r_range=ARC_R_RANGE,
            theta_wedge_range=ARC_THETA_RANGE,
            w_range=NODD_W_RANGE,
            n_gamma=ARC_N_GAMMA,
            n_r=ARC_N_R,
            n_theta_wedge=ARC_N_THETA,
            w_nodes_per_decade=ARC_W_NODES_PER_DECADE)
        cls._chart = surr.charts[0]

    def test_dd_cap_teeth_uncapped_would_exceed(self):
        """If we DON'T apply the DD cap, w_max would exceed the formula.

        Teeth: at r_max=0.7, reach≈0.68, the DD cap is 58/(0.7*0.68)≈121.6.
        The requested 500 >> 121.6, so without the cap the product would be
        500 * 0.7 * 0.68 ≈ 238 >> 58.
        """
        r_max = DD_R_RANGE[1]
        # Approximate reach_max (max directional caustic reach over the band).
        reach_approx = max(
            r_caustic(0.5, th)
            for th in np.linspace(DD_THETA_RANGE[0], DD_THETA_RANGE[1], 20))
        uncapped_product = DD_W_RANGE[1] * r_max * reach_approx
        self._tick()
        self.assertGreater(
            uncapped_product, DD_MARGIN,
            f'Uncapped product {uncapped_product:.1f} <= {DD_MARGIN}. '
            f'The DD cap would be non-binding (test has no teeth).')

    @staticmethod
    def _write_eps_report(eps_clean: np.ndarray, eps_bad: np.ndarray,
                          move: np.ndarray) -> None:
        """Dump the clean/degraded/move eps distributions to a diagnostic."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / 'self_falsification_umap_eps.txt'
        lines = [
            'Self-falsification: stored u-map is load-bearing on serve.',
            f'interior eps bar = {INTERIOR_EPS_BAR:.2e}',
            f'n comparisons    = {eps_clean.size}',
            '',
            'distribution        p50         p90         max',
        ]
        for label, arr in (('clean eps', eps_clean),
                           ('degraded eps', eps_bad),
                           ('|degraded-clean|', move)):
            lines.append(
                f'{label:<18} {np.percentile(arr, 50):.3e}  '
                f'{np.percentile(arr, 90):.3e}  {arr.max():.3e}')
        out.write_text('\n'.join(lines) + '\n')

    def test_perturbed_umap_degrades_serve(self):
        """Corrupting stored row-1 (u) degrades the serve past the eps bar.

        Off-node interior queries (cell midpoints in gamma, r, theta_wedge)
        are served with the intact chart and with a chart whose row 1 is
        corrupted over its central band.  The clean serve stays within the
        interior accuracy bar vs a fresh engine; the corrupted serve moves
        past it, and the served value shifts by more than the bar.
        """
        chart = self._chart

        # Off-node interior queries: cell midpoints on each spatial axis.
        gmids = 0.5 * (chart.gamma_grid[:-1] + chart.gamma_grid[1:])
        rmids = 0.5 * (chart.r_grid[:-1] + chart.r_grid[1:])
        tmids = 0.5 * (chart.theta_wedge_grid[:-1] + chart.theta_wedge_grid[1:])
        w_lin = np.exp(chart.log_w_grid)

        # Corrupt a central fraction of row 1 (the u values).
        t2s = chart.theta_to_s
        n_map = t2s.shape[1]
        u_max = float(t2s[1, -1])
        i0 = int(CORRUPT_BAND[0] * n_map)
        i1 = int(CORRUPT_BAND[1] * n_map)
        bad_map = t2s.copy()
        bad_map[1, i0:i1] += CORRUPT_OFFSET_FRACTION * u_max
        bad_chart = copy.copy(chart)
        object.__setattr__(bad_chart, 'theta_to_s', bad_map)

        eps_clean: list[float] = []
        eps_bad: list[float] = []
        move: list[float] = []
        for gamma in gmids:
            for r in rmids:
                for th in tmids:
                    gamma_f = float(gamma)
                    r_f = float(r)
                    th_f = float(th)
                    y1, y2 = _from_wedge_fixed(
                        gamma_f, r_f, th_f, chart.wedge_map)

                    ch = ChangRefsdalChannels(w_lin)
                    partition = ch.evaluate(
                        gamma=gamma_f, y=(y1, y2), beta=0.0, kappa=0.0)
                    engine = partition.envelope

                    clean = _evaluate_chart(
                        chart, gamma_f, eta=0.5, theta=0.7,
                        log_w_query=chart.log_w_grid, y1_eig=y1, y2_eig=y2)
                    bad = _evaluate_chart(
                        bad_chart, gamma_f, eta=0.5, theta=0.7,
                        log_w_query=chart.log_w_grid, y1_eig=y1, y2_eig=y2)

                    if not (np.all(np.isfinite(engine))
                            and np.all(np.isfinite(clean))
                            and np.all(np.isfinite(bad))):
                        continue
                    eps_clean.append(float(np.max(np.abs(clean - engine))))
                    eps_bad.append(float(np.max(np.abs(bad - engine))))
                    move.append(float(np.max(np.abs(bad - clean))))
                    self._tick()

        eps_clean_arr = np.asarray(eps_clean)
        eps_bad_arr = np.asarray(eps_bad)
        move_arr = np.asarray(move)
        self.assertGreater(
            eps_clean_arr.size, 0,
            'No finite comparisons collected — cannot assess degradation.')
        self._write_eps_report(eps_clean_arr, eps_bad_arr, move_arr)

        # Clean u-map serves within the interior bar (median over queries).
        self.assertLess(
            float(np.median(eps_clean_arr)), INTERIOR_EPS_BAR,
            f'Clean serve median eps {np.median(eps_clean_arr):.2e} already '
            f'exceeds the bar {INTERIOR_EPS_BAR:.0e}; the fixture is too '
            f'coarse to attribute degradation to the corruption.')
        # Corrupted u-map degrades past the bar (median over queries).
        self.assertGreater(
            float(np.median(eps_bad_arr)), INTERIOR_EPS_BAR,
            f'Corrupted serve median eps {np.median(eps_bad_arr):.2e} did '
            f'not exceed the bar {INTERIOR_EPS_BAR:.0e}; the stored u-map '
            f'is not load-bearing on serve.')
        # The served value moves by more than the bar (map is consumed).
        self.assertGreater(
            float(np.median(move_arr)), INTERIOR_EPS_BAR,
            f'Median |degraded - clean| {np.median(move_arr):.2e} <= bar '
            f'{INTERIOR_EPS_BAR:.0e}; corrupting row-1 did not move serve.')
        # Corruption strictly worsens accuracy (not merely a lateral shift).
        self.assertGreater(
            float(np.median(eps_bad_arr)), float(np.median(eps_clean_arr)),
            'Corrupted serve is not worse than clean at the median.')


if __name__ == '__main__':
    unittest.main()
