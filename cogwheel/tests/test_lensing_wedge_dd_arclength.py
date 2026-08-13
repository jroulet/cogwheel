"""Tests for InteriorWedgeChart DD w-ceiling and the cusp-adapted ``u`` axis.

Historical note on the file name
--------------------------------
This suite is named ``..._arclength`` for historical reasons: the wedge
chart's angular spline axis was ORIGINALLY a caustic arc length ``s``.  That
axis is retired.  The wedge chart is now charted in the cusp-adapted
coordinate ``u = d**(2/3)`` (``d`` = angular distance to the near astroid
cusp), and WP3 renamed the serialized fields accordingly
(``theta_to_s`` -> ``theta_to_u``, ``s_grid`` -> ``u_grid``) and bumped the
wedge axis schema ``wedge_caustic_relative_v2`` -> ``...v3``.  The file name
is left unchanged to preserve git history; every assertion below is on ``u``,
not arc length.

What this suite verifies
------------------------
1. **DD-product w-ceiling** — ``from_wedge_engine`` caps the frequency upper
   limit at ``_DD_PRODUCT_MARGIN / (r_max * reach_max)`` so no training node
   violates the engine's diffraction-delay ceiling.  ``DDWCeilingTestCase``
   re-confirms the cap binds (train-tier gated: its capped nodes sit above
   the Schwinger double-double ceiling).

2. **Cusp-adapted angular axis** — the chart stores a dense ``theta -> u``
   map in ``theta_to_u`` (row 0 = ``theta_fine``, row 1 = ``u_fine``) and
   consumes it at serve time via ``np.interp`` (see ``_evaluate_chart``).
   ``CuspAdaptedAxisTestCase`` checks the stored map bit-for-bit against
   ``_wedge_cusp_axis_map`` and against the per-side closed form.

3. **Field naming (SHARD C)** — the wedge chart exposes ``theta_to_u``
   (NOT ``theta_to_s``/``s_grid``); the arc-length charts (Tube, Lobe,
   FarField) are UNTOUCHED — Tube still exposes ``theta_to_s`` and
   FarField still exposes ``s_grid`` + ``arc_map``.  ``FieldExposureTestCase``
   and ``ValidatorContractTestCase`` pin the rename's blast radius.

4. **Serve is coordinate-agnostic + stale artifacts hard-refuse (SHARD C)** —
   the rename is nominal (``np.interp`` through the stored map depends only on
   the tabulated numbers, not on the field name), so an NPZ round-trip serves
   bit-identically; a stale ``v2``/``theta_to_s`` wedge artifact hard-refuses
   at load because ``theta_to_u`` is REQUIRED under ``v3``
   (``ServeCoordinateAgnosticTestCase``, ``StaleArtifactRefusalTestCase``).

5. **Domain guard (SHARD C)** — ``_wedge_cusp_axis_map`` raises ``ValueError``
   for any tile bound outside the D2-folded fundamental domain ``[0, pi/2]``
   instead of returning a silently NaN/complex array
   (``DomainGuardTestCase``).

Tolerance justification
-----------------------
Closed-form axis match (< 1e-9): ``theta_to_u`` row 1 is the algebraic image
of row 0 under the per-side closed form ``u = theta**(2/3) - theta_lo**(2/3)``
(LOW) / ``u = (pi/2 - theta_lo)**(2/3) - (pi/2 - theta)**(2/3)`` (HIGH); FP
round-off in the ``(2/3)->(3/2)`` round trip is ~1e-16 relative, so the 1e-9
bar carries ~7 orders of margin.  Bit-for-bit (== 0.0): the training path
wires ``_wedge_cusp_axis_map`` straight in with no re-derivation.  Serve
round-trip (< 1e-12 rel, measured 0.0): the persisted spline coefficients and
map are byte-identical, so the reloaded chart serves the same numbers.  Serve
degradation (> 5e-2): the interior off-node accuracy bar; corrupting a
fraction of the stored ``u`` row moves the served envelope past it
(``SelfFalsificationTestCase``).

Cost budget
-----------
Fast-tier engine builds are 4x4x4 = 64-node charts at ``w <= 20`` (below the
Schwinger double-double ceiling, ~0.2 s/eval).  ONE low-w chart is built once
and SHARED (``_shared_loww_surrogate``) across ``NoDDCapLowWTestCase``,
``SelfFalsificationTestCase``, ``ServeCoordinateAgnosticTestCase`` and
``StaleArtifactRefusalTestCase`` (~8 s total).  ``CuspAdaptedAxisTestCase``
builds two side-specific charts (~16 s).  ``DomainGuardTestCase``,
``ValidatorContractTestCase`` and ``FieldExposureTestCase`` are engine-free.
Whole file stays well under the 5-minute fast-tier ceiling.

``DDWCeilingTestCase`` is the exception and is SLOW-TIERED
(``COGWHEEL_TRAIN_TIER``): its DD cap lands at ``w_max ~ 121.6`` — above the
Schwinger ceiling (60) — so its capped-w nodes take the mpmath path at
~85-120 s EACH (F061).  No assertion in it was weakened; only its gate is
train-tier.
"""
from __future__ import annotations

import dataclasses
import json
import os
import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cogwheel.lensing.surrogate import (
    ExteriorPolarChart,
    InteriorWedgeChart,
    LensAmplificationSurrogate,
    LobeInteriorChart,
    TubeChart,
    _DD_PRODUCT_MARGIN,
    _FARFIELD_ARC_MAP_SIZE,
    _KNOWN_WEDGE_AXIS_SCHEMAS,
    _WEDGE_AXIS_SCHEMA,
    _chart_from_npz,
    _chart_to_npz,
    _evaluate_chart,
    _from_wedge_fixed,
    _log_reach_gamma_axis,
    _validate_theta_to_s,
    _validate_theta_to_u,
    _wedge_cusp_axis_map,
    _wedge_theta_waist,
)

#: Diagnostic-plot / report output directory (created on demand).
OUTPUT_DIR = Path(__file__).resolve().parent / 'output'

#: The current (WP3) wedge axis schema tag.
V3_SCHEMA: str = 'wedge_caustic_relative_v3'

#: The retired (pre-WP3) wedge axis schema tag — a stale artifact carries it.
V2_SCHEMA: str = 'wedge_caustic_relative_v2'

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

#: The DD margin constant, BOUND from production (already imported above)
#: rather than duplicated as a literal: the oracle below is built from it,
#: and a re-typed 58.0 would silently certify the wrong band the day
#: production moves the margin.
DD_MARGIN: float = _DD_PRODUCT_MARGIN

# ---------------------------------------------------------------------------
#: Module-level constants — non-DD-cap low-w fixture (SHARED build)
# ---------------------------------------------------------------------------

#: Gamma range (positive-parity interior).
ARC_GAMMA_RANGE: tuple[float, float] = (0.30, 0.50)

#: R range — moderate, away from the caustic boundary.
ARC_R_RANGE: tuple[float, float] = (0.20, 0.50)

#: Theta wedge range (a single tile spanning past the caustic waist).
ARC_THETA_RANGE: tuple[float, float] = (0.30, 1.20)

#: Nodes per spatial axis.
ARC_N_GAMMA: int = 4
ARC_N_R: int = 4
ARC_N_THETA: int = 4

#: W nodes per decade.
ARC_W_NODES_PER_DECADE: int = 10

#: Low w_range that does NOT trigger the DD cap.
NODD_W_RANGE: tuple[float, float] = (5.0, 15.0)

#: Module-level cache for the single shared low-w surrogate (built once).
_SHARED_LOWW: LensAmplificationSurrogate | None = None


def _shared_loww_surrogate() -> LensAmplificationSurrogate:
    """Build (once) and return the shared low-w wedge surrogate.

    Four fast-tier classes need a real, DD-uncapped wedge chart; building it
    once here keeps the file's engine cost to a single 64-node build for all
    of them (rather than one per class).

    Returns
    -------
    LensAmplificationSurrogate
        A one-chart surrogate over ``ARC_*`` with ``w_range = NODD_W_RANGE``.
    """
    global _SHARED_LOWW
    if _SHARED_LOWW is None:
        _SHARED_LOWW = LensAmplificationSurrogate.from_wedge_engine(
            gamma_range=ARC_GAMMA_RANGE,
            r_range=ARC_R_RANGE,
            theta_wedge_range=ARC_THETA_RANGE,
            w_range=NODD_W_RANGE,
            n_gamma=ARC_N_GAMMA,
            n_r=ARC_N_R,
            n_theta_wedge=ARC_N_THETA,
            w_nodes_per_decade=ARC_W_NODES_PER_DECADE)
    return _SHARED_LOWW


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

    This block is UNCHANGED by the cusp-axis / rename work; it is re-run to
    confirm the ``w*r*reach_max<=58`` cap still binds and the wedge build
    still succeeds under the cap.

    Cost: 4x4x4 = 64 nodes.  Nodes at the capped w_max sit ABOVE the
    Schwinger ceiling (60) and cost ~85-120 s each on the mpmath path,
    not ~30 ms (F061) — hence the training-tier gate on this class.
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
        """exp(log_w_grid[-1]) <= DD_MARGIN / (r_max * reach_max)."""
        chart = self._surrogate.charts[0]
        w_max_chart = float(np.exp(chart.log_w_grid[-1]))
        r_max = float(chart.r_grid[-1])
        theta_mask = (
            (chart.wedge_map.theta_nodes >= DD_THETA_RANGE[0])
            & (chart.wedge_map.theta_nodes <= DD_THETA_RANGE[1]))
        reach_max = float(chart.wedge_map.r_table[:, theta_mask].max())
        dd_cap = DD_MARGIN / (r_max * reach_max)
        self._tick()
        self.assertLessEqual(
            w_max_chart, dd_cap + 1e-10,
            f'Chart w_max={w_max_chart:.2f} exceeds DD cap={dd_cap:.2f}.')

    def test_w_max_below_requested(self):
        """The capped w_max must be strictly below the requested 500."""
        chart = self._surrogate.charts[0]
        w_max_chart = float(np.exp(chart.log_w_grid[-1]))
        self._tick()
        self.assertLess(
            w_max_chart, DD_W_RANGE[1],
            f'Chart w_max={w_max_chart:.2f} was not capped below '
            f'{DD_W_RANGE[1]}.')

    def test_dd_product_never_exceeds_margin(self):
        """The max DD product (w_max * r_max * reach_max) <= 58."""
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
            f'DD product w*r*reach = {product:.2f} exceeds {DD_MARGIN}.')

    def test_refused_fewer_than_total(self):
        """Some nodes must succeed — not all refused (build succeeds)."""
        chart = self._surrogate.charts[0]
        total_nodes = DD_N_GAMMA * DD_N_R * DD_N_THETA
        self._tick()
        self.assertLess(
            chart.refused_points.shape[0], total_nodes,
            'All nodes refused — the surrogate would be empty.')


# ===========================================================================
# Part 2 — engine-free structural / contract classes (SHARD C specs 1 & 3)
# ===========================================================================


class DomainGuardTestCase(_WedgeDDTestCase):
    """`_wedge_cusp_axis_map` hard-raises outside the D2 domain [0, pi/2].

    SHARD C spec 3.  A wedge tile bound below 0 or above ``pi/2`` can only
    come from a caller that failed to fold the source into the first
    quadrant.  The map must raise ``ValueError`` at the boundary rather than
    return a silently NaN/complex array that only surfaces later inside the
    serve-time ``np.interp``.  A valid in-domain call still returns strictly
    increasing ``theta_fine`` / ``u_fine`` with ``u_fine[0] == 0``.
    """

    #: A bound just above pi/2 (origin='high' near cusp) — must refuse.
    _THETA_HI_OVER: float = np.pi / 2.0 + 0.1

    #: A bound below 0 (origin='low' near cusp) — must refuse.
    _THETA_LO_UNDER: float = -0.1

    def test_theta_lo_below_zero_raises(self):
        """theta_lo < 0 raises ValueError naming the domain."""
        with self.assertRaises(ValueError) as ctx:
            _wedge_cusp_axis_map(self._THETA_LO_UNDER, 0.5, 'low')
        self._tick()
        self.assertIn('0, pi/2', str(ctx.exception))

    def test_theta_hi_above_half_pi_raises(self):
        """theta_hi > pi/2 raises ValueError naming the domain."""
        with self.assertRaises(ValueError) as ctx:
            _wedge_cusp_axis_map(1.0, self._THETA_HI_OVER, 'high')
        self._tick()
        self.assertIn('0, pi/2', str(ctx.exception))

    def test_both_bounds_out_of_domain_raise(self):
        """Sweep several out-of-domain bound pairs; each must raise."""
        cases = (
            (-1e-6, 0.5, 'low'),
            (0.1, np.pi / 2.0 + 1e-6, 'high'),
            (-0.5, np.pi, 'low'),
            (np.pi / 2.0 + 0.3, np.pi / 2.0 + 0.4, 'high'),
        )
        for theta_lo, theta_hi, origin in cases:
            with self.subTest(theta_lo=theta_lo, theta_hi=theta_hi):
                with self.assertRaises(ValueError):
                    _wedge_cusp_axis_map(theta_lo, theta_hi, origin)
                self._tick()

    def test_inverted_bounds_raise(self):
        """theta_lo >= theta_hi (in-domain) still raises ValueError."""
        with self.assertRaises(ValueError):
            _wedge_cusp_axis_map(0.6, 0.3, 'low')
        self._tick()

    def test_unknown_origin_raises(self):
        """An origin other than 'low'/'high' raises ValueError."""
        with self.assertRaises(ValueError):
            _wedge_cusp_axis_map(0.1, 0.5, 'middle')
        self._tick()

    def test_valid_in_domain_call_is_strictly_increasing(self):
        """A valid call returns strictly increasing theta/u with u[0]==0."""
        for origin, (theta_lo, theta_hi) in (
                ('low', (0.05, 0.60)), ('high', (0.90, np.pi / 2.0))):
            with self.subTest(origin=origin):
                theta_fine, u_fine = _wedge_cusp_axis_map(
                    theta_lo, theta_hi, origin)
                self.assertEqual(theta_fine.shape, (_FARFIELD_ARC_MAP_SIZE,))
                self.assertEqual(u_fine.shape, (_FARFIELD_ARC_MAP_SIZE,))
                self.assertTrue(np.all(np.diff(theta_fine) > 0.0))
                self.assertTrue(np.all(np.diff(u_fine) > 0.0))
                self.assertEqual(float(u_fine[0]), 0.0)
                # Exact endpoints (no extrapolation at serve time).
                self.assertEqual(float(theta_fine[0]), theta_lo)
                self.assertEqual(float(theta_fine[-1]), theta_hi)
                self.assertTrue(np.isfinite(u_fine).all())
                self._tick()


class ValidatorContractTestCase(_WedgeDDTestCase):
    """`_validate_theta_to_u` enforces monotone + u[0]==0 with NO magnitude bound.

    SHARD C spec 1.  The wedge validator delegates to the shared axis-map
    core with ``ordinate_name='u'``.  It must accept ANY strictly increasing
    ordinate starting at ~0 regardless of its magnitude (``u`` is
    ``rad**(2/3)``, not a length), and reject a non-monotone ordinate, a
    non-zero start, or a row-0 that does not begin at ``theta_grid[0]``.  The
    arc-length validator ``_validate_theta_to_s`` shares the same core and is
    therefore UNCHANGED by the rename.
    """

    #: A modest theta grid whose lower bound the map must start at.
    _THETA_GRID: np.ndarray = np.linspace(0.20, 1.20, 4)

    def _valid_u_map(self, scale: float = 1.0) -> np.ndarray:
        """A well-formed (2, N) theta->u map scaled by ``scale`` on row 1."""
        theta_fine = np.linspace(self._THETA_GRID[0], self._THETA_GRID[-1], 64)
        u_fine = scale * (theta_fine - theta_fine[0]) ** (2.0 / 3.0)
        return np.vstack([theta_fine, u_fine])

    def test_valid_map_accepted(self):
        """A monotone map starting at 0 validates and round-trips shape."""
        out = _validate_theta_to_u(self._valid_u_map(), self._THETA_GRID)
        self._tick()
        self.assertEqual(out.shape[0], 2)
        self.assertTrue(np.all(np.diff(out[1]) > 0.0))

    def test_huge_magnitude_u_row_accepted(self):
        """A u-row scaled by 1e6 still validates — there is NO magnitude bound."""
        # If a length-scale bound had leaked in from the arc-length axis this
        # would spuriously refuse; the point of spec 1 is that it does not.
        out = _validate_theta_to_u(self._valid_u_map(scale=1e6),
                                   self._THETA_GRID)
        self._tick()
        self.assertGreater(float(out[1].max()), 1e5)

    def test_tiny_magnitude_u_row_accepted(self):
        """A u-row scaled by 1e-9 still validates — no lower magnitude floor."""
        out = _validate_theta_to_u(self._valid_u_map(scale=1e-9),
                                   self._THETA_GRID)
        self._tick()
        self.assertLess(float(out[1].max()), 1e-6)

    def test_nonmonotone_u_row_rejected(self):
        """A non-monotone ordinate raises ValueError."""
        bad = self._valid_u_map()
        bad[1, 30] = bad[1, 10]  # break strict increase
        with self.assertRaises(ValueError):
            _validate_theta_to_u(bad, self._THETA_GRID)
        self._tick()

    def test_nonzero_start_rejected(self):
        """A u-row that does not start at ~0 raises ValueError."""
        bad = self._valid_u_map()
        bad[1] = bad[1] + 0.5  # shift so u[0] != 0
        with self.assertRaises(ValueError):
            _validate_theta_to_u(bad, self._THETA_GRID)
        self._tick()

    def test_wrong_theta_start_rejected(self):
        """A row-0 not starting at theta_grid[0] raises ValueError."""
        bad = self._valid_u_map()
        bad[0] = bad[0] + 0.05  # theta_fine[0] no longer == theta_grid[0]
        with self.assertRaises(ValueError):
            _validate_theta_to_u(bad, self._THETA_GRID)
        self._tick()

    def test_shared_core_accepts_same_map_as_arclength(self):
        """The arc-length validator accepts the identical map (shared core)."""
        # Proves the rename did not fork the numeric core: _validate_theta_to_s
        # (tube/lobe/far-field) and _validate_theta_to_u agree on a valid map.
        the_map = self._valid_u_map()
        out_u = _validate_theta_to_u(the_map, self._THETA_GRID)
        out_s = _validate_theta_to_s(the_map.copy(), self._THETA_GRID)
        self._tick()
        self.assertTrue(np.array_equal(out_u, out_s))

    def test_production_axis_map_passes_validator(self):
        """`_wedge_cusp_axis_map` output validates against `_validate_theta_to_u`."""
        theta_fine, u_fine = _wedge_cusp_axis_map(0.20, 1.20, 'low')
        the_map = np.vstack([theta_fine, u_fine])
        out = _validate_theta_to_u(the_map, np.array([0.20, 1.20]))
        self._tick()
        self.assertEqual(out.shape, the_map.shape)


class FieldExposureTestCase(_WedgeDDTestCase):
    """The rename's blast radius: only the wedge chart moved to ``theta_to_u``.

    SHARD C spec 1.  Inspect the dataclass field names directly (engine-free)
    so the invariant is pinned without a build:

    * ``InteriorWedgeChart`` exposes ``theta_to_u`` and NOT ``theta_to_s`` /
      ``s_grid`` / ``u_grid`` (``u_grid`` is a construction kwarg, not a
      stored field).
    * ``TubeChart`` still exposes ``theta_to_s`` and NOT ``theta_to_u``;
      ``LobeInteriorChart`` exposes ``theta_to_u`` (NOT ``theta_to_s``).
    * ``ExteriorPolarChart`` charts its spatial axes via ``rho_grid`` +
      ``theta_c_grid`` (it has neither ``s_grid`` nor ``arc_map``).
    """

    @staticmethod
    def _fields(cls) -> set[str]:
        """The set of dataclass field names on ``cls``."""
        return {f.name for f in dataclasses.fields(cls)}

    def test_wedge_exposes_theta_to_u_only(self):
        """Wedge chart has theta_to_u, not theta_to_s/s_grid/u_grid."""
        names = self._fields(InteriorWedgeChart)
        self._tick()
        self.assertIn('theta_to_u', names)
        self.assertNotIn('theta_to_s', names)
        self.assertNotIn('s_grid', names)
        self.assertNotIn('u_grid', names)

    def test_tube_still_exposes_theta_to_s(self):
        """Tube chart is untouched: theta_to_s present, theta_to_u absent."""
        names = self._fields(TubeChart)
        self._tick()
        self.assertIn('theta_to_s', names)
        self.assertNotIn('theta_to_u', names)

    def test_lobe_exposes_theta_to_u(self):
        """Lobe chart exposes theta_to_u, not theta_to_s."""
        names = self._fields(LobeInteriorChart)
        self._tick()
        self.assertIn('theta_to_u', names)
        self.assertNotIn('theta_to_s', names)

    def test_exterior_polar_uses_caustic_fixed_axes(self):
        """Exterior-polar charts expose rho_grid + theta_c_grid + theta_to_u (optional field)."""
        names = self._fields(ExteriorPolarChart)
        self._tick()
        self.assertIn('rho_grid', names)
        self.assertIn('theta_c_grid', names)
        self.assertNotIn('s_grid', names)
        self.assertNotIn('arc_map', names)
        self.assertIn('theta_to_u', names)


# ===========================================================================
# Part 3 — cusp-adapted axis wiring (SHARD C spec 1: ported theta_to_s port)
# ===========================================================================

#: Low-side theta tile: midpoint 0.25 sits well below any caustic waist
#: (~0.55-0.74 for gamma in [0.3, 0.5]) so the derived origin is 'low'.
CUSP_LOW_THETA_RANGE: tuple[float, float] = (0.10, 0.40)

#: High-side theta tile: midpoint 1.20 sits well above the waist so the
#: derived origin is 'high'.
CUSP_HIGH_THETA_RANGE: tuple[float, float] = (1.00, 1.40)


def _reconstruct_u(theta_fine: np.ndarray, theta_lo: float,
                   origin: str) -> np.ndarray:
    """INDEPENDENT closed form for the cusp-adapted ordinate ``u``.

    Hand-derived from the physics (``u = d**(2/3)`` offset so ``u(theta_lo) =
    0``), NOT a call into `_wedge_cusp_axis_map` — so it is a genuine oracle
    for the stored ``u`` values rather than a re-run of the code under test.

    Parameters
    ----------
    theta_fine : np.ndarray
        The wedge angles at which to evaluate ``u`` (radians).
    theta_lo : float
        The tile's lower wedge-angle bound (the offset anchor).
    origin : str
        ``'low'`` (near cusp at ``theta = 0``) or ``'high'`` (near cusp at
        ``pi/2``).

    Returns
    -------
    np.ndarray
        ``u`` evaluated at ``theta_fine``.
    """
    exponent = 2.0 / 3.0
    if origin == 'low':
        return theta_fine ** exponent - theta_lo ** exponent
    half_pi = np.pi / 2.0
    return (half_pi - theta_lo) ** exponent - (half_pi - theta_fine) ** exponent


def _derived_origin(gamma_range: tuple[float, float], n_gamma: int,
                    theta_range: tuple[float, float]) -> str:
    """Reconstruct `from_wedge_engine`'s midpoint-vs-waist origin choice.

    Mirrors the production single-sourcing exactly: ``rep_gamma`` is the
    median of the log-reach gamma axis, the split is at the caustic waist
    ``_wedge_theta_waist(rep_gamma)`` (NOT pi/4), and a tile whose midpoint is
    at or below the waist is 'low', otherwise 'high'.
    """
    gamma_grid = _log_reach_gamma_axis(gamma_range, n_gamma, 'gamma')
    rep_gamma = float(np.median(gamma_grid))
    theta_mid = 0.5 * (theta_range[0] + theta_range[1])
    return 'low' if theta_mid <= _wedge_theta_waist(rep_gamma) else 'high'


#: Module cache: two side-specific wedge charts (built once each).
_CUSP_CHARTS: dict[str, LensAmplificationSurrogate] = {}


def _cusp_surrogate(theta_range: tuple[float, float]
                    ) -> LensAmplificationSurrogate:
    """Build (once) a low-w wedge surrogate over ``theta_range``."""
    key = f'{theta_range[0]:.3f}_{theta_range[1]:.3f}'
    if key not in _CUSP_CHARTS:
        _CUSP_CHARTS[key] = LensAmplificationSurrogate.from_wedge_engine(
            gamma_range=ARC_GAMMA_RANGE,
            r_range=ARC_R_RANGE,
            theta_wedge_range=theta_range,
            w_range=NODD_W_RANGE,
            n_gamma=ARC_N_GAMMA,
            n_r=ARC_N_R,
            n_theta_wedge=ARC_N_THETA,
            w_nodes_per_decade=ARC_W_NODES_PER_DECADE)
    return _CUSP_CHARTS[key]


class CuspAdaptedAxisTestCase(_WedgeDDTestCase):
    """The stored ``theta_to_u`` equals the cusp-axis map bit-for-bit.

    SHARD C spec 1 — the port of the retired ``Stored theta_to_s ==
    vstack(...)`` assertion onto the new ``theta_to_u`` field.  Two tiles are
    trained, one nearest each astroid cusp (origin 'low' and 'high'), so both
    per-side closed forms are exercised.

    Checks per tile:
    1. The chart exposes ``theta_to_u`` (shape ``(2, 2001)``) and no
       ``theta_to_s`` attribute.
    2. ``theta_to_u`` is BIT-FOR-BIT ``np.vstack(_wedge_cusp_axis_map(
       theta_lo, theta_hi, derived_origin))`` — the training path wires the
       map straight in with no re-derivation.
    3. The stored ``u`` row matches the INDEPENDENT hand-derived closed form
       ``_reconstruct_u`` to < 1e-9 (genuine oracle, not a re-run of the
       code under test).
    """

    _low: LensAmplificationSurrogate | None = None
    _high: LensAmplificationSurrogate | None = None

    @classmethod
    def setUpClass(cls):
        cls._low = _cusp_surrogate(CUSP_LOW_THETA_RANGE)
        cls._high = _cusp_surrogate(CUSP_HIGH_THETA_RANGE)

    def _cases(self):
        """Yield (label, chart, theta_range, expected_origin) per tile."""
        yield ('low', self._low.charts[0], CUSP_LOW_THETA_RANGE, 'low')
        yield ('high', self._high.charts[0], CUSP_HIGH_THETA_RANGE, 'high')

    def test_derived_origin_matches_expected_side(self):
        """The reconstructed origin is 'low'/'high' as the tile intends."""
        for label, _chart, theta_range, expected in self._cases():
            with self.subTest(tile=label):
                origin = _derived_origin(
                    ARC_GAMMA_RANGE, ARC_N_GAMMA, theta_range)
                self._tick()
                self.assertEqual(origin, expected)

    def test_chart_exposes_theta_to_u_shape(self):
        """Each chart stores a (2, 2001) theta_to_u and no theta_to_s attr."""
        for label, chart, _theta_range, _origin in self._cases():
            with self.subTest(tile=label):
                self._tick()
                self.assertTrue(hasattr(chart, 'theta_to_u'))
                self.assertFalse(hasattr(chart, 'theta_to_s'))
                self.assertEqual(
                    chart.theta_to_u.shape, (2, _FARFIELD_ARC_MAP_SIZE))

    def test_stored_theta_to_u_equals_cusp_axis_map_bitwise(self):
        """Ported assertion: theta_to_u == vstack(_wedge_cusp_axis_map(...))."""
        for label, chart, theta_range, _expected in self._cases():
            with self.subTest(tile=label):
                origin = _derived_origin(
                    ARC_GAMMA_RANGE, ARC_N_GAMMA, theta_range)
                theta_fine, u_fine = _wedge_cusp_axis_map(
                    theta_range[0], theta_range[1], origin)
                oracle = np.vstack([theta_fine, u_fine])
                self._tick()
                self.assertTrue(
                    np.array_equal(chart.theta_to_u, oracle),
                    f'{label}: stored theta_to_u differs from '
                    f'vstack(_wedge_cusp_axis_map(...)).')

    def test_stored_u_row_matches_independent_closed_form(self):
        """The stored u row matches the hand-derived closed form to <1e-9."""
        worst = 0.0
        for label, chart, theta_range, _expected in self._cases():
            with self.subTest(tile=label):
                origin = _derived_origin(
                    ARC_GAMMA_RANGE, ARC_N_GAMMA, theta_range)
                theta_row = chart.theta_to_u[0]
                u_row = chart.theta_to_u[1]
                u_expected = _reconstruct_u(theta_row, theta_range[0], origin)
                err = float(np.max(np.abs(u_row - u_expected)))
                worst = max(worst, err)
                self._tick()
                self.assertLess(
                    err, 1e-9,
                    f'{label}: stored u row deviates from the independent '
                    f'closed form by {err:.2e}.')
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'cusp_axis_closed_form_residual.txt').write_text(
            f'worst |u_stored - u_closedform| over both tiles: {worst:.3e}\n')

    def test_u_row_starts_at_zero_and_increases(self):
        """The stored u row starts at 0 and is strictly increasing."""
        for label, chart, _theta_range, _origin in self._cases():
            with self.subTest(tile=label):
                u_row = chart.theta_to_u[1]
                self._tick()
                self.assertEqual(float(u_row[0]), 0.0)
                self.assertTrue(np.all(np.diff(u_row) > 0.0))


# ===========================================================================
# Part 4 — serve is coordinate-agnostic + stale artifacts hard-refuse
#          (SHARD C spec 2)
# ===========================================================================


def _serve_envelope(chart, gamma: float, r: float, theta_wedge: float,
                    log_w_query: np.ndarray) -> np.ndarray:
    """Serve the wedge chart's complex envelope at a wedge-fixed query.

    The wedge-fixed ``(gamma, r, theta_wedge)`` node is mapped to a physical
    eigenframe source via `_from_wedge_fixed` (the inverse of the transform
    `_evaluate_chart` applies internally), so the query lands inside the
    chart's trained basin.
    """
    y1_eig, y2_eig = _from_wedge_fixed(gamma, r, theta_wedge, chart.wedge_map)
    return _evaluate_chart(chart, gamma, float('nan'), float('nan'),
                           log_w_query, y1_eig, y2_eig)


def _interior_query_points(chart) -> list[tuple[float, float, float]]:
    """A handful of interior (gamma, r, theta_wedge) query points.

    Grid-cell midpoints plus the geometric centre — all strictly inside the
    trained axes so the served envelope is meaningful (not extrapolated).
    """
    gamma_grid = chart.gamma_grid
    r_grid = chart.r_grid
    theta_grid = chart.theta_wedge_grid

    def _mid(arr, frac):
        return float(arr[0] + frac * (arr[-1] - arr[0]))

    return [
        (_mid(gamma_grid, 0.5), _mid(r_grid, 0.5), _mid(theta_grid, 0.5)),
        (_mid(gamma_grid, 0.3), _mid(r_grid, 0.6), _mid(theta_grid, 0.4)),
        (_mid(gamma_grid, 0.7), _mid(r_grid, 0.35), _mid(theta_grid, 0.65)),
    ]


class ServeCoordinateAgnosticTestCase(_WedgeDDTestCase):
    """The rename is nominal: an NPZ round-trip serves bit-identically.

    SHARD C spec 2(a).  ``np.interp`` through the stored map depends only on
    the tabulated numbers, not on whether the field is called ``theta_to_s``
    or ``theta_to_u``, and the persisted spline coefficients are byte-
    identical.  So a chart reconstructed from its own NPZ arrays must serve
    the SAME ``F`` (< 1e-12 relative) at identical inputs.
    """

    _chart = None
    _reloaded = None
    _log_w_query: np.ndarray | None = None

    @classmethod
    def setUpClass(cls):
        surrogate = _shared_loww_surrogate()
        cls._chart = surrogate.charts[0]
        cls._reloaded = _chart_from_npz(_chart_to_npz(cls._chart, 0), 0)
        lo = float(cls._chart.log_w_grid[0])
        hi = float(cls._chart.log_w_grid[-1])
        cls._log_w_query = np.linspace(lo, hi, 12)

    def test_reloaded_is_interior_wedge(self):
        """The NPZ round-trip reconstructs an InteriorWedgeChart."""
        self._tick()
        self.assertIsInstance(self._reloaded, InteriorWedgeChart)

    def test_theta_to_u_bitwise_equal_after_roundtrip(self):
        """theta_to_u survives serialization byte-for-byte."""
        self._tick()
        self.assertTrue(
            np.array_equal(self._chart.theta_to_u, self._reloaded.theta_to_u))

    def test_served_F_identical_after_roundtrip(self):
        """Served |F| matches to < 1e-12 relative at several interior points."""
        worst = 0.0
        clean_curve = None
        reload_curve = None
        for gamma, r, theta_wedge in _interior_query_points(self._chart):
            with self.subTest(gamma=gamma, r=r, theta_wedge=theta_wedge):
                f_clean = _serve_envelope(
                    self._chart, gamma, r, theta_wedge, self._log_w_query)
                f_reload = _serve_envelope(
                    self._reloaded, gamma, r, theta_wedge, self._log_w_query)
                self.assertTrue(np.all(np.isfinite(f_clean)))
                denom = np.maximum(np.abs(f_clean), 1e-30)
                rel = float(np.max(np.abs(f_clean - f_reload) / denom))
                worst = max(worst, rel)
                self._tick()
                self.assertLess(
                    rel, 1e-12,
                    f'Served F drifted by {rel:.2e} after NPZ round-trip.')
                if clean_curve is None:
                    clean_curve, reload_curve = f_clean, f_reload
        # Diagnostic overlay of |F| for the first query point.
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        w = np.exp(self._log_w_query)
        ax.plot(w, np.abs(clean_curve), 'o-', label='original chart')
        ax.plot(w, np.abs(reload_curve), 'x--', label='NPZ round-trip')
        ax.set_xlabel('w')
        ax.set_ylabel('|F envelope|')
        ax.set_title(f'Serve coordinate-agnostic (max rel diff {worst:.1e})')
        ax.legend()
        fig.savefig(OUTPUT_DIR / 'serve_coordinate_agnostic_overlay.png',
                    dpi=80)
        plt.close(fig)


class StaleArtifactRefusalTestCase(_WedgeDDTestCase):
    """A stale v2 / theta_to_s wedge artifact hard-refuses at load.

    SHARD C spec 2(b).  Under the v3 schema ``theta_to_u`` is REQUIRED.  A
    stale artifact carrying the old ``wedge_caustic_relative_v2`` schema (and
    the old ``theta_to_s`` key) must hard-refuse:

    * a ``v2`` (or ``None``) axis schema raises ``ValueError`` at
      ``_validate_axis_schema`` BEFORE the map is read;
    * a well-labelled ``v3`` artifact that is nonetheless missing the
      ``theta_to_u`` key raises ``KeyError`` (it must not silently serve on a
      wrong angular coordinate).

    The unmutated arrays load cleanly — proving the refusals are not trivially
    always-raising.
    """

    _arrays: dict | None = None

    @classmethod
    def setUpClass(cls):
        chart = _shared_loww_surrogate().charts[0]
        cls._arrays = _chart_to_npz(chart, 0)

    def _with_meta(self, **overrides) -> dict:
        """Shallow copy of the arrays with meta ``axis_schema`` overridden."""
        stale = dict(self._arrays)
        meta = json.loads(str(self._arrays['chart0_meta']))
        meta.update(overrides)
        stale['chart0_meta'] = np.array(json.dumps(meta))
        return stale

    def test_unmutated_roundtrip_loads_clean(self):
        """Control: the untouched arrays reconstruct an InteriorWedgeChart."""
        chart = _chart_from_npz(dict(self._arrays), 0)
        self._tick()
        self.assertIsInstance(chart, InteriorWedgeChart)

    def test_v2_schema_hard_refuses(self):
        """A v2 axis schema raises ValueError naming the offending tag."""
        stale = self._with_meta(axis_schema=V2_SCHEMA)
        with self.assertRaises(ValueError) as ctx:
            _chart_from_npz(stale, 0)
        self._tick()
        self.assertIn(V2_SCHEMA, str(ctx.exception))

    def test_v2_with_theta_to_s_key_hard_refuses(self):
        """A fully-stale v2 artifact (theta_to_s key) refuses on schema first."""
        stale = self._with_meta(axis_schema=V2_SCHEMA)
        # Move the map under the retired key, exactly as a v2 artifact stored.
        stale['chart0_theta_to_s'] = stale.pop('chart0_theta_to_u')
        with self.assertRaises(ValueError) as ctx:
            _chart_from_npz(stale, 0)
        self._tick()
        self.assertIn(V2_SCHEMA, str(ctx.exception))

    def test_none_schema_hard_refuses(self):
        """A missing (None) axis schema raises ValueError mentioning None."""
        stale = self._with_meta(axis_schema=None)
        with self.assertRaises(ValueError) as ctx:
            _chart_from_npz(stale, 0)
        self._tick()
        self.assertIn('None', str(ctx.exception))

    def test_v3_missing_theta_to_u_raises_keyerror(self):
        """A v3 artifact missing theta_to_u raises KeyError (map is required)."""
        stale = dict(self._arrays)  # schema stays v3
        stale['chart0_theta_to_s'] = stale.pop('chart0_theta_to_u')
        with self.assertRaises(KeyError) as ctx:
            _chart_from_npz(stale, 0)
        self._tick()
        self.assertIn('theta_to_u', str(ctx.exception))


# ===========================================================================
# Part 5 — DD cap does not over-trigger + self-falsification + entry point
# ===========================================================================


def _dd_cap_for(chart, theta_range: tuple[float, float]) -> float:
    """Reconstruct the DD w-ceiling ``58 / (r_max * reach_max)`` for a chart."""
    r_max = float(chart.r_grid[-1])
    theta_mask = ((chart.wedge_map.theta_nodes >= theta_range[0])
                  & (chart.wedge_map.theta_nodes <= theta_range[1]))
    reach_max = float(chart.wedge_map.r_table[:, theta_mask].max())
    return _DD_PRODUCT_MARGIN / (r_max * reach_max)


class NoDDCapLowWTestCase(_WedgeDDTestCase):
    """At low ``w`` the DD cap is inactive and does NOT shrink ``w_max``.

    The complement of the (train-tier) ``DDWCeilingTestCase``: with
    ``w_range = NODD_W_RANGE`` the requested ``w_max`` sits well below the DD
    ceiling, so ``from_wedge_engine`` must leave it unchanged.  Guards against
    a cap that spuriously fires and silently narrows every chart's band.
    """

    _chart = None

    @classmethod
    def setUpClass(cls):
        cls._chart = _shared_loww_surrogate().charts[0]

    def test_dd_cap_exceeds_requested_w_max(self):
        """The reconstructed DD ceiling is above the requested w_max."""
        dd_cap = _dd_cap_for(self._chart, ARC_THETA_RANGE)
        self._tick()
        self.assertGreater(
            dd_cap, NODD_W_RANGE[1],
            f'DD cap {dd_cap:.2f} did not exceed requested '
            f'{NODD_W_RANGE[1]}; the fixture no longer tests an INACTIVE cap.')

    def test_w_max_not_capped_below_request(self):
        """exp(log_w_grid[-1]) equals the requested w_max (cap inactive)."""
        w_max_chart = float(np.exp(self._chart.log_w_grid[-1]))
        self._tick()
        self.assertLessEqual(w_max_chart, NODD_W_RANGE[1] + 1e-9)
        self.assertGreaterEqual(
            w_max_chart, NODD_W_RANGE[1] - 1e-6,
            f'Chart w_max={w_max_chart:.4f} was narrowed below the requested '
            f'{NODD_W_RANGE[1]} even though the DD cap is inactive.')

    def test_dd_product_stays_below_margin(self):
        """The realised DD product w_max * r_max * reach_max stays <= 58."""
        w_max_chart = float(np.exp(self._chart.log_w_grid[-1]))
        r_max = float(self._chart.r_grid[-1])
        theta_mask = (
            (self._chart.wedge_map.theta_nodes >= ARC_THETA_RANGE[0])
            & (self._chart.wedge_map.theta_nodes <= ARC_THETA_RANGE[1]))
        reach_max = float(self._chart.wedge_map.r_table[:, theta_mask].max())
        product = w_max_chart * r_max * reach_max
        self._tick()
        self.assertLessEqual(product, _DD_PRODUCT_MARGIN + 1e-6)


class SelfFalsificationTestCase(_WedgeDDTestCase):
    """Prove this suite can go RED — its green is not vacuous.

    Three independent teeth:
    1. Corrupting the stored ``theta_to_u`` map changes the served ``F`` by
       far more than the 1e-12 round-trip tolerance — so the
       coordinate-agnostic equality in ``ServeCoordinateAgnosticTestCase`` is
       a real constraint, not a tautology.
    2. The known-schema set genuinely EXCLUDES the retired ``v2`` tag and
       INCLUDES ``v3`` — the premise of the stale-artifact refusal.
    3. The wedge validator rejects a hand-broken (non-monotone) map.
    """

    _chart = None

    @classmethod
    def setUpClass(cls):
        cls._chart = _shared_loww_surrogate().charts[0]

    def test_corrupted_map_changes_served_F(self):
        """A corrupted theta_to_u yields a materially different served F."""
        chart = self._chart
        gamma, r, theta_wedge = _interior_query_points(chart)[0]
        log_w_query = np.linspace(
            float(chart.log_w_grid[0]), float(chart.log_w_grid[-1]), 8)
        f_clean = _serve_envelope(chart, gamma, r, theta_wedge, log_w_query)
        # Scale the u-row by 4x: np.interp then lands the query far outside
        # the spline's u-knot range, so the served envelope changes sharply.
        corrupt_map = chart.theta_to_u.copy()
        corrupt_map[1] = corrupt_map[1] * 4.0
        corrupt_chart = dataclasses.replace(chart, theta_to_u=corrupt_map)
        f_corrupt = _serve_envelope(
            corrupt_chart, gamma, r, theta_wedge, log_w_query)
        delta = float(np.max(np.abs(f_clean - f_corrupt)))
        self._tick()
        self.assertGreater(
            delta, 1e-3,
            'Corrupting theta_to_u did NOT change the served F — serve does '
            'not depend on the stored map, so the round-trip equality test '
            'has no teeth.')

    def test_known_schema_set_excludes_v2_includes_v3(self):
        """The wedge known-schema set excludes v2 and includes v3."""
        self._tick()
        self.assertIn(V3_SCHEMA, _KNOWN_WEDGE_AXIS_SCHEMAS)
        self.assertNotIn(V2_SCHEMA, _KNOWN_WEDGE_AXIS_SCHEMAS)
        self.assertEqual(_WEDGE_AXIS_SCHEMA, V3_SCHEMA)

    def test_validator_rejects_broken_map(self):
        """A non-monotone u-row is rejected — the validator has teeth."""
        theta_fine = np.linspace(0.20, 1.20, 32)
        u_fine = (theta_fine - theta_fine[0]) ** (2.0 / 3.0)
        broken = np.vstack([theta_fine, u_fine])
        broken[1, 16] = broken[1, 4]  # destroy strict monotonicity
        with self.assertRaises(ValueError):
            _validate_theta_to_u(broken, np.array([0.20, 1.20]))
        self._tick()


if __name__ == '__main__':
    unittest.main()
