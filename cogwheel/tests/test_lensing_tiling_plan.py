"""Engine-free unit tests for ``cogwheel.lensing.tiling_plan``.

The module under test is a *campaign cost predictor*: given a serve-route
census and a tiler config it sizes every ``(region x gamma-band x w-band)``
chart in node counts and gates each on measured ``engine_residual`` demand.
It performs NO wave evaluation.  These tests therefore probe the three axis
laws and the demand gate directly at the helper level -- never through the
expensive :func:`tiling_plan.run` (which would trigger a 10k-sample census
and a full tiling census) -- so the whole file runs in a few seconds.

Three Architect specifications are pinned, each written to kill a specific
regression mode of the old fixed-grid tiler:

* ``AxisSpanScalingLawTestCase`` (Spec 1) -- doubling an axis SPAN while
  holding its RESOLUTION fixed must (approximately) double that axis' node
  count.  A count that stays put under span-doubling is a bare hardcoded
  count and fails here.
* ``GammaResolutionTowardWallTestCase`` (Spec 2) -- the gamma resolution
  ``C * r_caustic / |d r_caustic / d gamma|`` tightens monotonically toward
  the ``gamma = 1`` parity wall (from both sides), so bands nearest the wall
  carry MORE gamma nodes, and no built band straddles ``gamma = 1``.  The
  independent oracle is a polar sweep of ``geometry.r_caustic`` -- a code
  path fully disjoint from production's closed-form
  ``ppgo_map.caustic_geometry`` reach.
* ``DemandGatedTilingTestCase`` (Spec 3) -- an astroid-exterior census cell
  that Born / saddle-c3 / certified-ppGO already serve (``engine_residual ==
  0``) contributes exactly zero chart tiles/nodes, while a cell with
  positive residual contributes a positive tile count.
* ``MeasuredWAxisEdgeTestCase`` (Spec 4) -- each region's w-axis upper edge
  is its OWN measured ``engine_residual`` demand edge (lobe-exterior 38, not
  the blanket 60 a fixed grid would stamp everywhere); no residual falls back
  to the prior box range, flagged as such.
* ``AnnulusGaugeRoundTripTestCase`` (Spec 5) -- the far-field annulus record
  declares a single explicit ``gauge``; the astroid ``caustic_rho`` outer
  edge round-trips through the independent authoritative converter
  ``ppgo_map.caustic_rho`` to ``1e-6``, and the saddle ``rho_lobe`` prior
  demand edge is the real ``~20`` scale, not a retired ``1.25-2.40`` cap.
* ``EscalationTripwireTestCase`` (Spec 6) -- ``_escalation_verdict`` records
  (never raises) the tripwire: benign ledgers pass, a ``>5e5``-call ledger
  and a ``>40%``-region-share ledger each escalate with the matching reason,
  both cap boundaries are strict ``>``, and the cost currency is pinned
  (``wall_clock_s == total_calls * 0.0903``, ``_LABELS_PER_NODE == 8``).
* ``EngineFreePlanRunTestCase`` (Spec 7) -- the WHOLE-TOOL invariant.  The
  real :func:`tiling_plan.run` main entry is executed once on a tiny
  synthetic census under ``mock.patch`` booby-traps on the four
  wave-amplitude doors ``tiling_census`` guards (``ChangRefsdalChannels
  .evaluate``, ``_schwinger.f_schwinger``, ``_schwinger._f_schwinger_mpmath``
  and ``mpmath.gauss_quadrature``); each door raises a unique sentinel, so a
  completed run with every door ``call_count == 0`` proves the plan tool
  makes ZERO engine evaluations -- matching the ``tiling_census`` /
  ``serve_route_census`` engine-free guarantee.  The literal spec phrase
  "mpmath never in ``sys.modules``" is UNACHIEVABLE (``mpmath`` is imported
  at package-import time by ``_schwinger``, long before ``run`` is called),
  so the load-bearing substitute is the ``mpmath.gauss_quadrature`` door
  ``call_count == 0`` -- no mpmath special-function evaluation is entered.

Tolerances
----------
* ``_ORACLE_REACH_RTOL = 1e-9`` -- the astroid caustic reach is attained
  ON-axis (``theta = 0``, a grid point of the polar sweep), so the two
  independent reach evaluations agree to machine precision (measured
  ``<= 2e-16`` across the identity gammas); the ``1e-9`` bar is a generous
  machine-noise envelope that still has real teeth.  The saddle reach is
  attained OFF-axis at a deltoid extremum the coarse polar grid misses by a
  few percent, so the ``r_caustic`` identity/oracle checks are ASTROID-only;
  the saddle-side monotonicity claim is carried by production
  ``_gamma_resolution`` (closed-form, exact on both sides).
* ``_SPAN_DOUBLING_TOL_NODES = 1`` -- the axis laws are ``ceil(span/res)``,
  so an exact doubling can land one node either side of ``2 * n1`` from the
  ceiling; the invariant asserted is ``n2 > n1`` (kills the hardcoded count)
  AND ``|n2 - 2*n1| <= 1``.
"""
from __future__ import annotations

import functools
import math
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest import mock

import mpmath
import numpy as np

from cogwheel.lensing import ppgo_map
from cogwheel.lensing import tiling_plan as tp
from cogwheel.lensing import tiling_census as tc
from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels, _schwinger
from cogwheel.lensing.chang_refsdal import geometry

#: Astroid gammas approaching the wall from below (all strictly < 1); the
#: near-wall window where production ``_gamma_resolution`` is strictly
#: monotone (a small non-monotone bump lives at gamma ~ 0.5-0.6, below this).
_ASTROID_WALL_GAMMAS: tuple[float, ...] = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

#: Saddle gammas approaching the wall from above (all strictly > 1), ordered
#: AWAY from the wall so a *rising* resolution == tightening toward the wall.
_SADDLE_WALL_GAMMAS: tuple[float, ...] = (1.05, 1.10, 1.15, 1.20, 1.25, 1.30)

#: Astroid gammas for the independent reach-oracle identity (on-axis reach,
#: so the polar sweep is machine-exact).
_ORACLE_IDENTITY_GAMMAS: tuple[float, ...] = (0.30, 0.60, 0.85, 0.95)

#: Polar samples for the ``geometry.r_caustic`` max-over-angle oracle; the
#: grid includes ``theta = 0`` (where the astroid reach is attained).
_R_CAUSTIC_THETA_SAMPLES: int = 91

#: Astroid reach agrees with the closed form to machine precision on-axis.
_ORACLE_REACH_RTOL: float = 1.0e-9

#: Ceiling slack for the span-doubling law (see module docstring).
_SPAN_DOUBLING_TOL_NODES: int = 1

#: Relative tolerance for the astroid annulus gauge round-trip
#: (caustic_rho -> physical |y| -> caustic_rho); measured ``0.0`` on-axis, the
#: ``1e-6`` bar is the Architect spec's envelope and still has real teeth.
_ANNULUS_ROUNDTRIP_RTOL: float = 1.0e-6

#: The retired far-field annulus outer-edge cap (``1.25-2.40`` caustic_rho).
#: A saddle lobe prior demand of ``rho_lobe ~ 20`` must sit FAR above this;
#: an outer edge <= this value is the old hardcoded cap resurfacing.
_RETIRED_ANNULUS_CAP: float = 2.40


@functools.lru_cache(maxsize=1)
def _prod() -> tuple[Any, Any]:
    """Load the production sizing modules once for the whole file.

    ``_load_production_modules`` only IMPORTS the engine-bearing modules; it
    makes zero wave evaluations, and every helper exercised below is pure
    geometry / arithmetic.
    """
    return tc._load_production_modules()


def _oracle_scalar_reach(gamma: float,
                         n_theta: int = _R_CAUSTIC_THETA_SAMPLES) -> float:
    """Independent scalar caustic reach: max over theta of ``r_caustic``.

    A polar sweep of :func:`geometry.r_caustic` -- a code path disjoint from
    production's closed-form ``ppgo_map.caustic_geometry`` -- so agreement is
    a genuine cross-check, not a function graded against itself.
    """
    best = 0.0
    for theta in np.linspace(0.0, math.pi, n_theta):
        try:
            radius = float(geometry.r_caustic(gamma, float(theta)))
        except geometry.LensDomainError:
            continue
        best = max(best, radius)
    return best


def _oracle_gamma_resolution(gamma: float, step: float = 1.0e-3) -> float:
    """Law-1 resolution from the independent ``r_caustic`` reach oracle."""
    reach = _oracle_scalar_reach(gamma)
    reach_plus = _oracle_scalar_reach(gamma + step)
    reach_minus = _oracle_scalar_reach(gamma - step)
    dreach = abs(reach_plus - reach_minus) / (2.0 * step)
    return tp._C_GAMMA * reach / dreach


class _TilingPlanTestCase(unittest.TestCase):
    """Base carrying the anti-vacuity guard shared by every suite here.

    Every comparison calls :meth:`_observe`; ``tearDown`` fails if a test
    body somehow made zero comparisons (an all-skipped or silently-empty
    sweep would otherwise read green).
    """

    def setUp(self) -> None:
        self._observations = 0

    def _observe(self) -> None:
        self._observations += 1

    def tearDown(self) -> None:
        self.assertGreater(
            self._observations, 0,
            'anti-vacuity: the test made zero comparisons.')


class AxisSpanScalingLawTestCase(_TilingPlanTestCase):
    """Spec 1: node count scales with span at fixed resolution.

    Each axis-sizing helper is driven directly with a synthetic band/config
    so the ONLY thing that changes between the two calls is the axis span.
    A hardcoded count (``n2 == n1``) fails ``assertGreater``.
    """

    class _ExpReachST:
        """Stub with an exponential caustic reach.

        ``reach = exp(k*gamma)`` makes ``r_caustic / |d r_caustic/d gamma|``
        (hence the gamma resolution) INDEPENDENT of gamma -- resolution is
        held exactly fixed while only the band span changes.
        """

        def __init__(self, rate: float) -> None:
            self._rate = rate

        def _scalar_caustic_reach(self, gamma: float) -> float:
            return math.exp(self._rate * gamma)

    def test_gamma_count_doubles_when_span_doubles(self) -> None:
        # Exponential reach => constant resolution, so the wall-nearest-edge
        # resolution is identical for both bands (verified below).
        stub = self._ExpReachST(rate=4.0)
        anchor = 0.20
        span1, span2 = 0.35, 0.70  # span2 == 2 * span1
        band1 = (anchor, anchor + span1)
        band2 = (anchor, anchor + span2)

        res1 = tp._gamma_resolution(stub, tp._wall_nearest_edge(band1, 1), 1)
        res2 = tp._gamma_resolution(stub, tp._wall_nearest_edge(band2, 1), 1)
        self._observe()
        self.assertAlmostEqual(
            res1, res2, places=12,
            msg='resolution must be held fixed across the two bands.')

        n1 = tp._n_gamma_in_band(stub, band1, 1)
        n2 = tp._n_gamma_in_band(stub, band2, 1)
        self._observe()
        self.assertGreater(
            n2, n1, 'doubling gamma span left n_gamma fixed => hardcoded count.')
        self.assertLessEqual(abs(n2 - 2 * n1), _SPAN_DOUBLING_TOL_NODES)

    def test_theta_count_doubles_when_span_doubles(self) -> None:
        span1, span2 = 0.50, 1.00  # rad; span2 == 2 * span1
        n1 = tp._n_theta_for_span(span1, 1)
        n2 = tp._n_theta_for_span(span2, 1)
        self._observe()
        self.assertGreater(
            n2, n1, 'doubling arc span left n_theta fixed => hardcoded count.')
        self.assertLessEqual(abs(n2 - 2 * n1), _SPAN_DOUBLING_TOL_NODES)

    def test_w_count_doubles_when_log_range_doubles(self) -> None:
        # Config with the per-decade density held fixed; only the w log-range
        # (number of decades) changes between the two calls.
        config = SimpleNamespace(w_nodes_per_decade=4.0,
                                 interior_w_nodes_per_decade=15.0)
        n1 = tp._n_w_nodes(10.0, 100.0, 'exterior', config)     # 1 decade
        n2 = tp._n_w_nodes(10.0, 1000.0, 'exterior', config)    # 2 decades
        self._observe()
        self.assertGreater(
            n2, n1, 'doubling the w log-range left n_w fixed => hardcoded.')
        self.assertLessEqual(abs(n2 - 2 * n1), _SPAN_DOUBLING_TOL_NODES)

    def test_all_three_axes_flat_line_would_fail(self) -> None:
        # Negative control: a constant (hardcoded) sizer returns the same
        # count for both spans, which is exactly what the span law forbids.
        flat = lambda _span: 7
        self._observe()
        self.assertFalse(
            flat(0.35) < flat(0.70),
            'a flat/hardcoded count must NOT satisfy the span-scaling law.')


class GammaResolutionTowardWallTestCase(_TilingPlanTestCase):
    """Spec 2: gamma resolution tightens monotonically toward gamma = 1.

    ``d r_caustic / d gamma -> inf`` at the parity wall, so the resolution
    ``C * r_caustic / |d r_caustic/d gamma|`` collapses there and bands
    nearest the wall carry MORE gamma nodes.  The independent oracle is a
    polar sweep of ``geometry.r_caustic``; the shipping behaviour is
    ``_gamma_resolution`` (closed-form reach), checked on both sides.
    """

    def test_independent_reach_oracle_matches_production(self) -> None:
        # Cross-check: the closed-form production reach == an independent
        # polar-sweep maximum of r_caustic (astroid: on-axis, machine-exact).
        st, _sg = _prod()
        for gamma in _ORACLE_IDENTITY_GAMMAS:
            with self.subTest(gamma=gamma):
                oracle = _oracle_scalar_reach(gamma)
                production = float(st._scalar_caustic_reach(gamma))
                self._observe()
                self.assertGreaterEqual(oracle, 0.0)
                self.assertLessEqual(
                    abs(oracle - production) / production, _ORACLE_REACH_RTOL,
                    f'independent r_caustic reach disagrees at gamma={gamma}.')

    def test_oracle_resolution_decreases_toward_wall_astroid(self) -> None:
        # The independent r_caustic-based resolution must fall monotonically
        # as gamma -> 1 from below.
        values = [_oracle_gamma_resolution(g) for g in _ASTROID_WALL_GAMMAS]
        for lower, upper in zip(values, values[1:]):
            self._observe()
            self.assertLess(
                upper, lower,
                'oracle gamma resolution must tighten toward the wall.')

    def test_production_resolution_decreases_toward_wall_both_sides(self) -> None:
        st, _sg = _prod()
        # Astroid: ascending gamma -> resolution strictly falls (toward wall).
        astro = [tp._gamma_resolution(st, g, 1) for g in _ASTROID_WALL_GAMMAS]
        for lower, upper in zip(astro, astro[1:]):
            self._observe()
            self.assertLess(upper, lower,
                            'astroid resolution must fall toward gamma=1.')
        # Saddle: ascending gamma moves AWAY from the wall, so resolution
        # strictly RISES -- equivalently it falls toward the wall.
        saddle = [tp._gamma_resolution(st, g, -1) for g in _SADDLE_WALL_GAMMAS]
        for lower, upper in zip(saddle, saddle[1:]):
            self._observe()
            self.assertGreater(upper, lower,
                               'saddle resolution must fall toward gamma=1.')

    def test_band_nearest_wall_carries_more_gamma_nodes(self) -> None:
        # Two astroid bands of IDENTICAL span; the wall-nearest one must be
        # sized with strictly more gamma nodes.
        st, _sg = _prod()
        far_band = (0.50, 0.60)
        near_band = (0.85, 0.95)
        n_far = tp._n_gamma_in_band(st, far_band, 1)
        n_near = tp._n_gamma_in_band(st, near_band, 1)
        self._observe()
        self.assertGreater(
            n_near, n_far,
            'equal-span band nearer the wall must carry more gamma nodes.')

    def test_no_built_band_straddles_the_parity_wall(self) -> None:
        # Structural: every topology-stable band context lives wholly on one
        # side of gamma = 1 (bands butt the wall, never cross it).
        st, _sg = _prod()
        box = st.PriorBox.from_prior_classes()
        config = st.TrainingConfig()
        seen = 0
        for parity in (1, -1):
            contexts, _dropped = tc._collect_band_contexts(
                st, box, parity, config)
            for ctx in contexts:
                lo, hi = ctx.band
                seen += 1
                self._observe()
                self.assertFalse(
                    lo < 1.0 < hi,
                    f'band {ctx.band} (parity {parity}) straddles gamma=1.')
                if parity == 1:
                    self.assertLessEqual(hi, 1.0)
                else:
                    self.assertGreaterEqual(lo, 1.0)
        self.assertGreater(seen, 0, 'no band contexts were built to check.')


class _StubExpReachST:
    """Minimal ``st`` stub exposing only the scalar caustic reach."""

    def __init__(self, rate: float = 1.0) -> None:
        self._rate = rate

    def _scalar_caustic_reach(self, gamma: float) -> float:
        return math.exp(self._rate * gamma)


def _served_cell(region: str, band: str) -> dict[str, Any]:
    """A census cell fully served by analytics (zero engine residual)."""
    return {'region': region, 'gamma_band': band,
            'routes': {'born_analytic': 10, 'saddle_c3': 5, 'engine_residual': 0}}


def _demand_cell(region: str, band: str, residual: int) -> dict[str, Any]:
    """A census cell carrying positive engine-residual demand."""
    return {'region': region, 'gamma_band': band,
            'routes': {'engine_residual': residual}}


class DemandGatedTilingTestCase(_TilingPlanTestCase):
    """Spec 3: analytic-served cells contribute zero chart tiles/nodes."""

    def test_residual_lookup_zeroes_served_cell(self) -> None:
        # An astroid-exterior served cell alongside a demand cell (and a
        # second demand cell in the SAME key to exercise aggregation).
        cells = {
            'c0': _served_cell('exterior', '0.4-0.6'),
            'c1': _demand_cell('exterior', '0.6-0.8', 5),
            'c2': _demand_cell('exterior', '0.6-0.8', 3),
        }
        lookup = tp._residual_by_region_band(cells)
        self._observe()
        self.assertEqual(
            lookup.get(('exterior', '0.4-0.6'), 0), 0,
            'analytic-served astroid-exterior cell must carry zero demand.')
        self.assertEqual(
            lookup[('exterior', '0.6-0.8')], 8,
            'demand cells in one key must sum (5 + 3).')

    def test_plan_band_gates_served_cell_to_none(self) -> None:
        # Positive tile geometry but zero census demand => no plan, no nodes.
        st = _StubExpReachST()
        box = SimpleNamespace(w_range=lambda parity: (10.0, 100.0))
        ctx = SimpleNamespace(band=(0.4, 0.6), gamma_mid=0.5,
                              exclusion_rho=1.2, rho_outer_region=5.0)
        config = SimpleNamespace(w_nodes_per_decade=4.0,
                                 interior_w_nodes_per_decade=15.0)
        gamma_edges = np.array([0.4, 0.6])
        with mock.patch.object(tp, '_band_tile_geometry',
                               return_value=(4, 8, [])):
            entry, status = tp._plan_band(
                st, box, ctx, 'exterior', 1, config, [], {}, gamma_edges)
        self._observe()
        self.assertIsNone(entry, 'served cell must produce no plan entry.')
        self.assertEqual(status, 'gated_no_demand')

    def test_plan_band_sizes_demand_cell_positively(self) -> None:
        st = _StubExpReachST()
        box = SimpleNamespace(w_range=lambda parity: (10.0, 100.0))
        ctx = SimpleNamespace(band=(0.4, 0.6), gamma_mid=0.5,
                              exclusion_rho=1.2, rho_outer_region=5.0)
        config = SimpleNamespace(w_nodes_per_decade=4.0,
                                 interior_w_nodes_per_decade=15.0)
        gamma_edges = np.array([0.4, 0.6])
        residual_lookup = {('exterior', '0.4-0.6'): 5}
        with mock.patch.object(tp, '_band_tile_geometry',
                               return_value=(4, 8, [])):
            entry, status = tp._plan_band(
                st, box, ctx, 'exterior', 1, config, [], residual_lookup,
                gamma_edges)
        self._observe()
        self.assertEqual(status, 'planned')
        self.assertIsNotNone(entry)
        self.assertGreater(entry['band_nodes'], 0)
        # band_nodes must factor as spatial_total x n_gamma x n_w.
        self.assertEqual(
            entry['band_nodes'],
            entry['spatial_nodes_total'] * entry['n_gamma_in_band']
            * entry['n_w'])
        self.assertEqual(entry['n_tiles'], 4)

    def test_plan_region_counts_only_the_demand_band(self) -> None:
        # One served band + one demand band in the same region: the served
        # band must add exactly zero nodes and zero tiles.
        st = _StubExpReachST()
        box = SimpleNamespace(w_range=lambda parity: (10.0, 100.0))
        served_ctx = SimpleNamespace(band=(0.4, 0.6), gamma_mid=0.5,
                                     exclusion_rho=1.2, rho_outer_region=5.0)
        demand_ctx = SimpleNamespace(band=(0.6, 0.8), gamma_mid=0.7,
                                     exclusion_rho=1.2, rho_outer_region=5.0)
        config = SimpleNamespace(w_nodes_per_decade=4.0,
                                 interior_w_nodes_per_decade=15.0)
        gamma_edges = np.array([0.4, 0.6, 0.8])
        residual_lookup = {('exterior', '0.6-0.8'): 5}
        with mock.patch.object(tp, '_band_tile_geometry',
                               return_value=(4, 8, [])):
            rec = tp._plan_region(
                st, box, [served_ctx, demand_ctx], 'exterior', 1, config,
                [], residual_lookup, gamma_edges)
        # The demand band alone: 8 spatial x n_gamma x 4 w nodes.
        expected_nodes = 8 * tp._n_gamma_in_band(st, (0.6, 0.8), 1) * 4
        self._observe()
        self.assertEqual(rec['n_bands_planned'], 1)
        self.assertEqual(rec['n_bands_gated_no_demand'], 1)
        self.assertEqual(rec['region_tiles'], 4,
                         'only the demand band contributes tiles.')
        self.assertEqual(rec['region_nodes'], expected_nodes,
                         'served band must contribute exactly zero nodes.')


class TilingPlanSelfFalsificationTestCase(_TilingPlanTestCase):
    """Prove the pins above can actually go red.

    Each method feeds a deliberately-wrong construction into the SAME
    assertion shape used by a real test and confirms it fails.
    """

    def test_flat_sizer_fails_span_law(self) -> None:
        # A hardcoded count violates the span-doubling assertion.
        flat = lambda _span: 11
        self._observe()
        with self.assertRaises(AssertionError):
            self.assertGreater(flat(0.70), flat(0.35),
                               'hardcoded count must not scale with span.')

    def test_constant_sequence_fails_monotonicity(self) -> None:
        # A flat (0.04-band-style) resolution fails the toward-wall check.
        flat_resolution = [0.04, 0.04, 0.04, 0.04]
        self._observe()
        with self.assertRaises(AssertionError):
            for lower, upper in zip(flat_resolution, flat_resolution[1:]):
                self.assertLess(upper, lower)

    def test_positive_demand_is_not_gated(self) -> None:
        # Teeth for the gate: a positive-residual cell is NOT gated, so the
        # gate is not vacuously always-declining.
        st = _StubExpReachST()
        box = SimpleNamespace(w_range=lambda parity: (10.0, 100.0))
        ctx = SimpleNamespace(band=(0.4, 0.6), gamma_mid=0.5,
                              exclusion_rho=1.2, rho_outer_region=5.0)
        config = SimpleNamespace(w_nodes_per_decade=4.0,
                                 interior_w_nodes_per_decade=15.0)
        gamma_edges = np.array([0.4, 0.6])
        with mock.patch.object(tp, '_band_tile_geometry',
                               return_value=(4, 8, [])):
            _entry, status = tp._plan_band(
                st, box, ctx, 'exterior', 1, config, [],
                {('exterior', '0.4-0.6'): 1}, gamma_edges)
        self._observe()
        self.assertEqual(status, 'planned',
                         'positive demand must not be gated away.')


def _residual_record(region: str, gamma_band: str, w_lo: float, w_hi: float
                     ) -> dict[str, Any]:
    """A single ``engine_residual`` census record (natural-log w edges)."""
    return {'route': 'engine_residual', 'region': region,
            'gamma_band': gamma_band,
            'log_w_min': math.log(w_lo), 'log_w_max': math.log(w_hi)}


class MeasuredWAxisEdgeTestCase(_TilingPlanTestCase):
    """Spec: each region's w-axis upper edge is its OWN measured demand edge.

    ``_measured_w_range`` reads the ``engine_residual`` records for a
    ``(region, gamma_band)`` and returns ``exp(max log_w_max)``.  A region
    with a measured ``w_hi`` of 38 must plan a 38 ceiling, NEVER the blanket
    60 that a fixed-grid tiler would stamp on every region.
    """

    #: Distinct measured w_hi per region: lobe-exterior 38, saddle_c3-served
    #: exterior ~51.6, interior 60.  ``box.w_range`` returns the blanket 60 so
    #: any region inheriting 60 while its own demand is lower is the bug.
    _CELLS: tuple[tuple[str, str, int, float, float], ...] = (
        ('lobe_exterior', '1.10-1.30', -1, 2.0, 38.0),
        ('exterior', '0.60-0.80', 1, 2.0, 51.6),
        ('wedge_interior', '0.20-0.40', 1, 2.0, 60.0),
    )

    def _records(self) -> list[dict[str, Any]]:
        return [_residual_record(reg, band, lo, hi)
                for reg, band, _par, lo, hi in self._CELLS]

    def test_each_region_upper_edge_equals_its_own_measured_w_hi(self) -> None:
        # Truth table region -> (w_lo, w_hi); each upper edge must match the
        # region's OWN measured w_hi, not a shared 60 ceiling.
        records = self._records()
        box = SimpleNamespace(w_range=lambda parity: (2.0, 60.0))
        for region, band, parity, _lo, w_hi in self._CELLS:
            with self.subTest(region=region):
                got_lo, got_hi, status = tp._measured_w_range(
                    records, region, band, box, parity)
                self._observe()
                self.assertEqual(status, 'measured')
                self.assertAlmostEqual(
                    got_hi, w_hi, places=9,
                    msg=f'{region} upper edge must equal its measured w_hi.')
                self.assertAlmostEqual(got_lo, 2.0, places=9)

    def test_lobe_exterior_edge_is_not_the_blanket_ceiling(self) -> None:
        # The regression this spec kills: lobe-exterior silently inheriting
        # the global 60 ceiling instead of its measured 38.
        records = self._records()
        box = SimpleNamespace(w_range=lambda parity: (2.0, 60.0))
        _lo, w_hi, _status = tp._measured_w_range(
            records, 'lobe_exterior', '1.10-1.30', box, -1)
        self._observe()
        self.assertAlmostEqual(w_hi, 38.0, places=9)
        self.assertNotAlmostEqual(
            w_hi, 60.0, places=6,
            msg='lobe-exterior must NOT inherit the blanket 60 ceiling.')

    def test_measured_edges_aggregate_min_and_max_across_records(self) -> None:
        # Two residual records in one cell: w_lo is the min, w_hi the max.
        records = [
            _residual_record('lobe_exterior', '1.10-1.30', 4.0, 38.0),
            _residual_record('lobe_exterior', '1.10-1.30', 1.5, 30.0),
        ]
        box = SimpleNamespace(w_range=lambda parity: (2.0, 60.0))
        got_lo, got_hi, status = tp._measured_w_range(
            records, 'lobe_exterior', '1.10-1.30', box, -1)
        self._observe()
        self.assertEqual(status, 'measured')
        self.assertAlmostEqual(got_lo, 1.5, places=9,
                               msg='w_lo must be the min over records.')
        self.assertAlmostEqual(got_hi, 38.0, places=9,
                               msg='w_hi must be the max over records.')

    def test_no_demand_falls_back_to_prior_box_range(self) -> None:
        # A tile admitted by geometry but empty of measured demand takes the
        # prior box range and is flagged as a fallback, not a measurement.
        box = SimpleNamespace(w_range=lambda parity: (3.0, 60.0))
        got_lo, got_hi, status = tp._measured_w_range(
            [], 'exterior', '0.60-0.80', box, 1)
        self._observe()
        self.assertEqual(status, 'prior_box_fallback')
        self.assertEqual((got_lo, got_hi), (3.0, 60.0))

    def test_records_from_other_regions_do_not_leak(self) -> None:
        # A demand record in a different region/band must not raise the
        # queried region's edge -- edges are strictly per (region, band).
        records = [
            _residual_record('lobe_exterior', '1.10-1.30', 2.0, 38.0),
            _residual_record('exterior', '0.60-0.80', 2.0, 60.0),
        ]
        box = SimpleNamespace(w_range=lambda parity: (2.0, 60.0))
        _lo, w_hi, _status = tp._measured_w_range(
            records, 'lobe_exterior', '1.10-1.30', box, -1)
        self._observe()
        self.assertAlmostEqual(
            w_hi, 38.0, places=9,
            msg="exterior's 60 must not leak into lobe_exterior's edge.")

    def test_above_ceiling_demand_is_clipped_to_the_dd_ceiling(self) -> None:
        # Regression (INS-1-001): the ``engine_residual`` route fires for any
        # draw whose node kinds include ``exact_wave``, and that route tallies
        # draws straddling the (60, 150] QD/mpmath band with ``log_w_max``
        # left unclipped.  A DD-band chart tile (serves w <= 60) must clip its
        # upper edge to the ceiling and NEVER plan a node that would need
        # QD/mpmath.  Without the clip, this w_hi=150 record would size the
        # tile to 150.
        records = [_residual_record('exterior', '0.60-0.80', 2.0, 150.0)]
        box = SimpleNamespace(w_range=lambda parity: (2.0, 60.0))
        got_lo, got_hi, status = tp._measured_w_range(
            records, 'exterior', '0.60-0.80', box, 1)
        self._observe()
        self.assertEqual(status, 'measured_clipped_dd')
        self.assertAlmostEqual(
            got_hi, 60.0, places=9,
            msg='above-ceiling engine_residual demand must clip to 60.')
        self.assertAlmostEqual(
            got_lo, 2.0, places=9,
            msg='the lower edge stays measured; only the upper edge clips.')

    def test_prior_box_fallback_is_clipped_to_the_dd_ceiling(self) -> None:
        # Companion fallback path: a tile admitted by geometry but empty of
        # measured demand takes the prior box's ``w_range`` (up to 480 for the
        # astroid), whose upper edge must ALSO clip at the DD ceiling so an
        # empty DD-band tile cannot inherit an above-ceiling span.
        box = SimpleNamespace(w_range=lambda parity: (2.0, 480.0))
        got_lo, got_hi, status = tp._measured_w_range(
            [], 'exterior', '0.60-0.80', box, 1)
        self._observe()
        self.assertEqual(status, 'prior_box_fallback_clipped_dd')
        self.assertAlmostEqual(got_hi, 60.0, places=9,
                               msg='prior-box upper edge must clip to 60.')
        self.assertAlmostEqual(got_lo, 2.0, places=9)

    def test_explicit_ceiling_argument_drives_the_clip(self) -> None:
        # The ceiling is single-sourced from the census header, not hardcoded:
        # an explicit non-default ``w_ceiling_dd`` (45, distinct from the
        # module's canonical 60) must be the value the clip enforces, proving
        # the ceiling is read from config rather than a baked-in constant.
        records = [_residual_record('exterior', '0.60-0.80', 2.0, 150.0)]
        box = SimpleNamespace(w_range=lambda parity: (2.0, 60.0))
        got_lo, got_hi, status = tp._measured_w_range(
            records, 'exterior', '0.60-0.80', box, 1, w_ceiling_dd=45.0)
        self._observe()
        self.assertEqual(status, 'measured_clipped_dd')
        self.assertAlmostEqual(
            got_hi, 45.0, places=9,
            msg='the supplied ceiling (45), not the module constant (60), '
                'must drive the clip.')
        self.assertAlmostEqual(got_lo, 2.0, places=9)


class AnnulusGaugeRoundTripTestCase(_TilingPlanTestCase):
    """Spec: the far-field annulus declares one gauge and tracks the real rho.

    ``_annulus_record`` labels each far-field remainder with an explicit
    ``gauge`` and records its outer edge in that gauge.  For the astroid
    exterior (``caustic_rho``) the round-trip through the AUTHORITATIVE,
    independently-implemented converter ``ppgo_map.caustic_rho`` recovers the
    prior edge to ``1e-6``.  For the saddle lobe exterior (``rho_lobe``) the
    prior demand edge is the real ``rho_lobe ~ 20`` scale, NOT a retired
    ``1.25-2.40`` cap; the physical rho_lobe conversion is served downstream
    (analytic ladder / deltoid redesign) and is out of scope for this module.
    """

    def test_astroid_declares_caustic_rho_gauge(self) -> None:
        st, _sg = _prod()
        ctx = SimpleNamespace(gamma_mid=0.70, exclusion_rho=1.2,
                              rho_outer_region=1.85)
        rec = tp._annulus_record(st, ctx, 'exterior')
        self._observe()
        self.assertIsNotNone(rec)
        self.assertEqual(rec['gauge'], 'caustic_rho')
        self.assertEqual(rec['rho_inner'], ctx.exclusion_rho)
        self.assertEqual(rec['rho_outer'], ctx.rho_outer_region)

    def test_astroid_outer_edge_round_trips_through_caustic_rho(self) -> None:
        # The declared-gauge (caustic_rho) outer edge -> physical |y| via the
        # record's caustic_reach -> back to caustic_rho via the independent
        # authoritative converter, recovering the prior edge to 1e-6.
        st, _sg = _prod()
        for gamma_mid in _ORACLE_IDENTITY_GAMMAS:
            with self.subTest(gamma_mid=gamma_mid):
                ctx = SimpleNamespace(gamma_mid=gamma_mid, exclusion_rho=1.2,
                                      rho_outer_region=1.85)
                rec = tp._annulus_record(st, ctx, 'exterior')
                physical = rec['rho_outer'] * rec['caustic_reach']
                round_trip = ppgo_map.caustic_rho(gamma_mid, physical)
                self._observe()
                self.assertLessEqual(
                    abs(round_trip - ctx.rho_outer_region)
                    / ctx.rho_outer_region, _ANNULUS_ROUNDTRIP_RTOL,
                    'caustic_rho gauge round-trip must recover the prior edge.')

    def test_saddle_declares_rho_lobe_gauge_at_real_scale(self) -> None:
        # Saddle lobe exterior: gauge is rho_lobe and the prior demand outer
        # edge is the real ~20 scale, far above the retired 2.40 cap.
        st, _sg = _prod()
        ctx = SimpleNamespace(gamma_mid=1.20, exclusion_rho=1.0,
                              rho_outer_region=5.0)
        rec = tp._annulus_record(st, ctx, 'lobe_exterior')
        self._observe()
        self.assertIsNotNone(rec)
        self.assertEqual(rec['gauge'], 'rho_lobe')
        self.assertEqual(rec['rho_inner'], 1.0)
        prior_rho = rec['prior_demand_rho_outer_lobe']
        self.assertAlmostEqual(prior_rho, tp._SADDLE_LOBE_DEMAND_RHO_OUTER,
                               places=9)
        self.assertGreater(
            prior_rho, _RETIRED_ANNULUS_CAP,
            'saddle prior demand must be the ~20 scale, not a 2.40 cap.')

    def test_interior_regions_have_no_annulus(self) -> None:
        # Only far-field regions carry an annulus; interior regions return
        # None (they are charted, not annular).
        st, _sg = _prod()
        ctx = SimpleNamespace(gamma_mid=0.70, exclusion_rho=1.2,
                              rho_outer_region=1.85)
        for region in ('wedge_interior', 'lobe_interior'):
            with self.subTest(region=region):
                self._observe()
                self.assertIsNone(tp._annulus_record(st, ctx, region))

    def test_round_trip_has_teeth_against_a_wrong_reach(self) -> None:
        # Self-falsification: converting the outer edge with a WRONG reach
        # (double the record's) must break the round-trip -- proving the
        # 1e-6 gate is not vacuously satisfied.
        st, _sg = _prod()
        ctx = SimpleNamespace(gamma_mid=0.70, exclusion_rho=1.2,
                              rho_outer_region=1.85)
        rec = tp._annulus_record(st, ctx, 'exterior')
        wrong_physical = rec['rho_outer'] * (2.0 * rec['caustic_reach'])
        round_trip = ppgo_map.caustic_rho(ctx.gamma_mid, wrong_physical)
        self._observe()
        self.assertGreater(
            abs(round_trip - ctx.rho_outer_region) / ctx.rho_outer_region,
            _ANNULUS_ROUNDTRIP_RTOL,
            'a wrong reach must fail the round-trip (gate has teeth).')


def _per_region_ledger(nodes: tuple[int, ...]) -> dict[str, dict[str, Any]]:
    """Synthetic per-region node ledger keyed ``region:parity`` -> rec."""
    return {f'r{i}:+1': {'region_nodes': int(n)} for i, n in enumerate(nodes)}


class EscalationTripwireTestCase(_TilingPlanTestCase):
    """Spec: ``_escalation_verdict`` records (never raises) the tripwire call.

    Truth table (total_calls, max_region_share) -> should_escalate:
      * benign  (1.2e5 calls, shares 1/3)      -> False
      * call    (6.4e5 calls, shares 1/4)      -> True, cites the >5e5 cap
      * share   (1.12e5 calls, one share 0.71) -> True, cites the >40% cap
    """

    def test_benign_ledger_does_not_escalate(self) -> None:
        ledger = _per_region_ledger((5000, 5000, 5000))
        total_nodes = 15000
        total_calls = total_nodes * tp._LABELS_PER_NODE
        verdict = tp._escalation_verdict(total_calls, ledger, total_nodes)
        self._observe()
        self.assertIsInstance(verdict, dict)  # records, never raises
        self.assertFalse(verdict['should_escalate'])
        self.assertEqual(verdict['reasons'], [])
        self.assertEqual(total_calls, 120000)  # pins _LABELS_PER_NODE == 8

    def test_call_cap_exceeded_escalates_with_call_reason(self) -> None:
        ledger = _per_region_ledger((20000, 20000, 20000, 20000))
        total_nodes = 80000
        total_calls = total_nodes * tp._LABELS_PER_NODE  # 640000 > 5e5
        verdict = tp._escalation_verdict(total_calls, ledger, total_nodes)
        self._observe()
        self.assertTrue(verdict['should_escalate'])
        self.assertLessEqual(verdict['max_region_share'],
                             tp._ESCALATION_REGION_SHARE)
        self.assertTrue(
            any('total_calls' in r and 'exceeds limit' in r
                for r in verdict['reasons']),
            'the call-cap reason must cite total_calls exceeding the limit.')

    def test_region_share_exceeded_escalates_with_share_reason(self) -> None:
        ledger = _per_region_ledger((10000, 2000, 2000))
        total_nodes = 14000
        total_calls = total_nodes * tp._LABELS_PER_NODE  # 112000 < 5e5
        verdict = tp._escalation_verdict(total_calls, ledger, total_nodes)
        self._observe()
        self.assertTrue(verdict['should_escalate'])
        self.assertLess(total_calls, tp._ESCALATION_CALL_LIMIT)
        self.assertGreater(verdict['max_region_share'],
                           tp._ESCALATION_REGION_SHARE)
        self.assertTrue(
            any('share' in r for r in verdict['reasons']),
            'the share-cap reason must cite the region node share.')

    def test_call_cap_boundary_is_strict_greater(self) -> None:
        # Exactly at the cap does NOT escalate; one node over does.  Nodes are
        # spread over four equal regions (shares 0.25 < 0.40) so the SHARE cap
        # never fires and the call cap is isolated.
        at_cap_nodes = int(tp._ESCALATION_CALL_LIMIT // tp._LABELS_PER_NODE)
        self.assertEqual(at_cap_nodes * tp._LABELS_PER_NODE,
                         tp._ESCALATION_CALL_LIMIT)  # lands exactly on the cap
        self.assertEqual(at_cap_nodes % 4, 0)  # splits cleanly into 4 regions
        quarter = at_cap_nodes // 4
        at_cap = tp._escalation_verdict(
            at_cap_nodes * tp._LABELS_PER_NODE,
            _per_region_ledger((quarter, quarter, quarter, quarter)),
            at_cap_nodes)
        over_nodes = at_cap_nodes + 4  # keep the 4-way split exact
        over_quarter = over_nodes // 4
        over = tp._escalation_verdict(
            over_nodes * tp._LABELS_PER_NODE,
            _per_region_ledger(
                (over_quarter, over_quarter, over_quarter, over_quarter)),
            over_nodes)
        self._observe()
        self.assertFalse(at_cap['should_escalate'],
                         'exactly at the call cap must not escalate.')
        self.assertEqual(at_cap['reasons'], [])
        self.assertTrue(over['should_escalate'])
        self.assertTrue(any('total_calls' in r for r in over['reasons']))

    def test_region_share_boundary_is_strict_greater(self) -> None:
        # max share exactly 0.40 does NOT escalate (strict >), calls well
        # under the cap so the share cap is isolated.
        ledger = _per_region_ledger((4, 3, 3))  # shares 0.4, 0.3, 0.3
        verdict = tp._escalation_verdict(
            10 * tp._LABELS_PER_NODE, ledger, 10)
        self._observe()
        self.assertAlmostEqual(verdict['max_region_share'],
                               tp._ESCALATION_REGION_SHARE, places=12)
        self.assertFalse(verdict['should_escalate'],
                         'a share exactly at the cap must not escalate.')

    def test_wall_clock_currency_is_calls_times_seconds_per_call(self) -> None:
        # Pin the cost currency: SECONDS_PER_CALL is exactly 0.0903 and
        # wall_clock_s == total_calls * 0.0903 for a known ledger.
        self._observe()
        self.assertEqual(tp.SECONDS_PER_CALL, 0.0903)
        total_calls = 15000 * tp._LABELS_PER_NODE
        wall_clock_s = total_calls * tp.SECONDS_PER_CALL
        self.assertEqual(wall_clock_s, total_calls * 0.0903)


# ---------------------------------------------------------------------------
# Spec 7: whole-tool engine-free invariant (the real ``run`` main entry)
# ---------------------------------------------------------------------------

#: A tiny synthetic census size: big enough that ``run`` populates a real,
#: non-empty plan (measured ``total_nodes`` in the thousands at 40 samples),
#: small enough that the single ``setUpClass`` invocation stays well under
#: the fast-tier per-test ceiling (measured ~13 s at 40 samples; ~15-18 s
#: here).  ``run`` is executed exactly ONCE and shared across the class.
_ENGINE_FREE_N_SAMPLES: int = 60


class _EvaluateDoor(Exception):
    """Sentinel: ``ChangRefsdalChannels.evaluate`` was called."""


class _FSchwingerDoor(Exception):
    """Sentinel: ``_schwinger.f_schwinger`` was called."""


class _FSchwingerMpmathDoor(Exception):
    """Sentinel: ``_schwinger._f_schwinger_mpmath`` was called."""


class _MpmathDoor(Exception):
    """Sentinel: ``mpmath.gauss_quadrature`` (special-function path) entered."""


#: The four wave-amplitude "doors" the plan tool must never open, each keyed
#: to a UNIQUE plain-``Exception`` sentinel.  Every sentinel is deliberately
#: NOT a member of the census caught-refusal tuple (asserted in
#: :meth:`EngineFreePlanRunTestCase.test_sentinels_disjoint_from_caught_tuple`)
#: so a door hit would PROPAGATE out of ``run`` -- never be swallowed into an
#: ``engine_residual`` and read as a benign refusal.
_PLAN_DOORS: tuple[tuple[str, type[Exception]], ...] = (
    ('ChangRefsdalChannels.evaluate', _EvaluateDoor),
    ('_schwinger.f_schwinger', _FSchwingerDoor),
    ('_schwinger._f_schwinger_mpmath', _FSchwingerMpmathDoor),
    ('mpmath.gauss_quadrature', _MpmathDoor),
)


def _arm_plan_doors(stack: ExitStack) -> dict[str, Any]:
    """Booby-trap the four wave-amplitude doors on ``stack``; return the mocks.

    Each patch installs a ``side_effect`` raising that door's unique sentinel,
    so ANY call to a door aborts loudly rather than silently evaluating a
    waveform.  Returns ``{door_name: mock}`` keyed as in :data:`_PLAN_DOORS`.
    """
    evaluate_mock = stack.enter_context(mock.patch.object(
        ChangRefsdalChannels, 'evaluate', side_effect=_EvaluateDoor()))
    f_mock = stack.enter_context(mock.patch.object(
        _schwinger, 'f_schwinger', side_effect=_FSchwingerDoor()))
    f_mp_mock = stack.enter_context(mock.patch.object(
        _schwinger, '_f_schwinger_mpmath', side_effect=_FSchwingerMpmathDoor()))
    mpmath_mock = stack.enter_context(mock.patch.object(
        mpmath, 'gauss_quadrature', side_effect=_MpmathDoor()))
    return {
        'ChangRefsdalChannels.evaluate': evaluate_mock,
        '_schwinger.f_schwinger': f_mock,
        '_schwinger._f_schwinger_mpmath': f_mp_mock,
        'mpmath.gauss_quadrature': mpmath_mock,
    }


class EngineFreePlanRunTestCase(_TilingPlanTestCase):
    """The real ``tiling_plan.run`` main entry makes zero engine evaluations.

    This is the WHOLE-TOOL counterpart to the six helper-level suites above:
    rather than probing a sizing helper in isolation, it runs the actual
    ``run`` entry once (on a ``_ENGINE_FREE_N_SAMPLES``-sample synthetic
    census) with every wave-amplitude door booby-trapped, and asserts the run
    completes with a well-formed ``tiling_plan_v1`` report while every door's
    ``call_count`` is ``0``.  A single nonzero count means an engine path
    leaked into the cost predictor, breaking its by-construction guarantee.

    On the mpmath substitution: the Architect's literal "mpmath never present
    in ``sys.modules``" cannot hold -- importing ``cogwheel.lensing`` pulls in
    ``_schwinger`` which imports ``mpmath`` at module load, long before ``run``
    is ever called.  The precise, load-bearing property is that no mpmath
    *special-function evaluation* is entered, which the
    ``mpmath.gauss_quadrature`` door (``call_count == 0``) pins directly.
    """

    @classmethod
    def setUpClass(cls) -> None:
        with ExitStack() as stack:
            cls.doors = _arm_plan_doors(stack)
            cls.report = tp.run(n_samples=_ENGINE_FREE_N_SAMPLES, seed=0)

    def test_run_completes_with_wellformed_report(self) -> None:
        """Reaching a completed report under the four patches is itself proof.

        A door hit would have raised its sentinel inside ``setUpClass`` and
        aborted the class; here we confirm the returned report is complete,
        carries the expected schema, and sized a genuinely non-empty plan.
        """
        self.assertEqual(self.report['schema'], 'tiling_plan_v1')
        self._observe()
        total_nodes = self.report['totals']['total_nodes']
        self.assertIsInstance(total_nodes, int)
        self.assertGreater(
            total_nodes, 0,
            'the engine-free run produced an empty plan -- the invariant '
            'would then be vacuously satisfied')
        self._observe()

    def test_zero_calls_on_every_wave_door(self) -> None:
        """Every wave-amplitude door mock has ``call_count == 0``.

        The direct, load-bearing check: each door's sentinel would have fired
        on any touch, so zero calls == zero engine evaluations across the
        entire plan+cost computation.
        """
        for name, door in self.doors.items():
            self.assertEqual(
                door.call_count, 0,
                f'wave-amplitude door {name} was called during '
                'tiling_plan.run -- an engine path leaked into the plan tool')
            self._observe()

    def test_doors_are_live_positive_control(self) -> None:
        """The booby-traps are armed: each patched door raises its sentinel.

        Guards against a vacuous invariant -- if a patch target were wrong
        (a stale attribute, a same-name import elsewhere), the door would be a
        silent no-op and ``call_count == 0`` would prove nothing.  Here we
        re-arm the doors and confirm each one, when called, raises EXACTLY its
        sentinel, so the zero-count result above genuinely has teeth.
        """
        with ExitStack() as stack:
            doors = _arm_plan_doors(stack)
            with self.assertRaises(_EvaluateDoor):
                ChangRefsdalChannels.evaluate()
            self._observe()
            with self.assertRaises(_FSchwingerDoor):
                _schwinger.f_schwinger()
            self._observe()
            with self.assertRaises(_FSchwingerMpmathDoor):
                _schwinger._f_schwinger_mpmath()
            self._observe()
            with self.assertRaises(_MpmathDoor):
                mpmath.gauss_quadrature()
            self._observe()
            # And after firing, each mock recorded the touch -- proving a
            # real engine hit during ``run`` WOULD have been counted.
            for door in doors.values():
                self.assertEqual(door.call_count, 1)
                self._observe()

    def test_sentinels_disjoint_from_caught_tuple(self) -> None:
        """No door sentinel is a member of the census caught-refusal tuple.

        If a sentinel were a subclass of a caught refusal type, a genuine
        engine hit could be swallowed into an ``engine_residual`` and the run
        would still 'succeed' -- silently defeating the whole invariant.  The
        caught set is the census refusal tuple plus the ``(ValueError,
        ZeroDivisionError)`` the per-draw handlers add.
        """
        _st, sg = _prod()
        caught = tuple(sg._REFUSAL_ERRORS) + (ValueError, ZeroDivisionError)
        for _name, sentinel in _PLAN_DOORS:
            self.assertFalse(
                issubclass(sentinel, caught),
                f'{sentinel.__name__} is a subclass of a caught refusal type '
                '-- a leaked engine hit could be swallowed as a benign '
                'refusal, making the zero-call check vacuous')
            self._observe()


if __name__ == '__main__':
    unittest.main()
