"""Tests for `lensing.serve_route_census` -- the engine-free demand census.

These are STRUCTURAL / CONTRACT invariants, not physics-accuracy oracles:
the census records a serve ROUTE per draw, never an amplitude, so there is
no numerical tolerance to certify.  Three properties are load-bearing and
are pinned here:

1. MECE routing (`MeceSmallRunTestCase`).  Every draw lands in EXACTLY one
   of the `SERVE_ROUTES` labels: the reported ``route_counts`` has
   exactly those keys, sums to ``n_samples`` (exhaustiveness), and -- with
   no artifact attached -- emits zero ``surrogate`` (that route is
   artifact-mode-only).  The independent oracle is a from-scratch re-tally
   of the per-draw ``records`` with a private ``collections.Counter``,
   compared against the production ``route_counts``; a re-tally that
   disagrees, or sums below ``n_samples``, is the exhaustiveness break the
   spec's bar-chart diagnostic describes.

2. Residual-partition disjointness + gauge (`ResidualPartitionTestCase`).
   The three-way caustic-rho split of the ``engine_residual`` population
   (``>2`` born-chart demand, ``(1,2]`` near-caustic tube, ``<=1`` interior,
   plus an ``undetermined`` bucket for an unresolved rho) is itself MECE
   within the residual: the buckets sum EXACTLY to
   ``route_counts['engine_residual']`` and the reported ``split_gauge`` is
   ``'caustic_rho'`` (a value of ``'rho_lobe'`` is the F073 gauge
   regression).  The oracle re-bins the residual ``records`` by their own
   ``caustic_rho`` field with an independently written threshold ladder.

3. Engine-free classification (`EngineFreeTestCase`).  The forbidden
   exact-wave DOORS (`ChangRefsdalChannels.evaluate`,
   ``_schwinger.f_schwinger`` / ``_f_schwinger_mpmath`` and the ``mpmath``
   special function the mpmath path calls) are monkeypatched to raise a
   UNIQUE sentinel that is NOT a member of the census's caught refusal
   tuple.  ``run`` must COMPLETE without the sentinel escaping and with the
   door mocks' ``call_count`` still zero: this is the load-bearing
   'demand map, not evaluator' guarantee -- a sentinel escaping ``run`` is
   the smoking gun that a draw reached the engine.

`_CensusTestCase.tearDown` is the anti-vacuity guard: a test that iterated
zero draws / comparisons asserts nothing and is failed.
`SelfFalsificationTestCase` proves each guard above can actually go red.

Beyond the three report-level properties, hand-placed single draws pin the
production BAND-SPLIT serving semantics (commit 6958f0c): the c3 saddle
band split (whole-band admit AND in-band ``w_split``, recorded per draw),
the above-ceiling ppGO intercept with its ceiling-keyed gate
(``150 * min_delta_tau >= RHO_END``) firing BEFORE the saddle rung
(production rung order), and ``wave_refused`` as the DERIVED deferred-2b
set (above-ceiling band AND failed ceiling gate) -- plus the per-node band
ladder witnesses on draws that genuinely reach the node pass.

Cost note (fast tier): the shared demand run is 150 draws x 32 freq nodes
(~14 s wall, memoized once for the report-level classes); the engine-free
run is 120 draws x 32 nodes under four active patches (~12 s).  Two `run`
calls total, well under the 5-minute file ceiling.
"""

from __future__ import annotations

import functools
import math
from collections import Counter
from contextlib import ExitStack
from unittest import TestCase, main, mock

import mpmath
import numpy as np

from cogwheel.lensing import serve_route_census as src
from cogwheel.lensing import likelihood as lk
from cogwheel.lensing import ppgo_map
from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels, _schwinger
from cogwheel.lensing.chang_refsdal import geometry


# ---------------------------------------------------------------------------
# Fixture-scale constants (single-sourced; grids derive from these)
# ---------------------------------------------------------------------------

#: Draw count for the shared demand run (small but enough to populate
#: ``engine_residual``, the ``wave_refused`` production-refusal route and at
#: least one analytic-intercept route).
N_SAMPLES = 150

#: RNG seed -- fixed so the shared report is deterministic across classes.
SEED = 0

#: Frequency-node count per draw; kept small so the per-node arm pass is
#: cheap while still spanning the production Hz band.
N_FREQ = 32

#: Draw count for the engine-free run (spec: ~120); smaller than the shared
#: run because it runs under four active mock patches.
ENGINE_FREE_N = 120

#: The census's own schema tag; a bump here is a serialized-contract change.
EXPECTED_SCHEMA = 'serve_route_census_v1'

#: The gauge the residual split MUST report (F073 guard: never 'rho_lobe').
EXPECTED_SPLIT_GAUGE = 'caustic_rho'

#: The residual sub-buckets, in caustic-rho-descending order plus the
#: unresolved-rho bucket -- the four keys `residual_demand` must partition
#: the ``engine_residual`` population into.
_RESIDUAL_BUCKETS = (
    'born_chart_demand', 'near_caustic_tube', 'interior', 'undetermined')

#: Routes with a legitimate per-draw certificate split (``w_split``): the
#: exact engine serves the band at or below ``w_split``, an analytic /
#: engine-hosted closure serves above it.  Single-sourced here so a future
#: band-split route extends this ONE set, not every w_split-detail test.
_W_SPLIT_ROUTES = frozenset({'saddle_c3', 'diffractive_engine_hosted'})


@functools.lru_cache(maxsize=1)
def _shared_report() -> dict:
    """Build the demand-mode census report ONCE and memoize it.

    Classes 1 and 2 both read this single report (the spec's "same small
    run() result"), so the 150-draw run happens exactly once per process.
    """
    config = src.ServeRouteCensusConfig(
        n_samples=N_SAMPLES, seed=SEED, n_freq=N_FREQ)
    return src.run(config)


# ---------------------------------------------------------------------------
# Single-draw classification harness (specs 4-5).  Shares the production
# module binding and grids with `run`, but calls `classify_draw` directly so
# a handful of hand-placed draws can be probed without a full census sweep.
# ---------------------------------------------------------------------------

#: Mass (Msun) for the hand-placed single draws; large enough that the band
#: floor ``w_lo`` sits in the analytic-serve regime for the far-field draws.
_PROBE_MASS_MSUN = 1.0e3

#: Off-axis source angle (rad) for the D2 quadruple fixtures.  Chosen away
#: from 0 and pi/4 so that ``cos != sin`` and both coordinates are nonzero:
#: the four IEEE sign-flips (+-,-+,...) are then four genuinely DISTINCT
#: source points, not a degenerate on-axis pair.
_D2_ANGLE_RAD = 0.4

#: D2 representative draws: (label, gamma, target caustic_rho, route).  They
#: span BOTH parities (gamma<1 astroid, gamma>1 saddle) and three distinct
#: serve routes (a per-node ``engine_residual``, the ``born_analytic``
#: intercept, the ``saddle_c3`` intercept, and a near-caustic residual).  The
#: source magnitude for each is DERIVED at test time from the live caustic
#: reach, never pinned, so a reach change moves the fixture with the boundary.
_D2_FIXTURES = (
    ('astroid_interior', 0.5, 0.5, 'engine_residual'),
    ('born_exterior', 0.5, 3.0, 'born_analytic'),
    ('near_caustic', 0.5, 1.05, 'engine_residual'),
    ('saddle_farfield_c3', 3.0, 3.0, 'saddle_c3'),
)

#: Spec-5 near-critical saddle draw: a macro saddle (gamma>1) whose merging
#: image pair sits just outside its directional caustic (caustic_rho just
#: above 1, ON the cusp axis where the omitted-term estimate is maximal), so
#: ``ppgo_error_estimate`` is FINITE but astronomically large (~1e15) rather
#: than ``None``.  Derived from the live reach at test time.
_SADDLE_HUGE_GAMMA = 3.0
_SADDLE_HUGE_RHO = 1.001

#: Floor the near-critical estimate must exceed to count as "astronomically
#: large" -- 9+ orders above the ~1e-3 certificate bar and ~15 orders above
#: the modest finite estimates (~1) seen just inside the tube.  Measured
#: value at the fixture is ~4.8e15, so this floor carries wide margin.
_HUGE_EST_FLOOR = 1.0e9


@functools.lru_cache(maxsize=1)
def _classify_env() -> tuple:
    """Bind the production modules + grids ONCE for the single-draw probes.

    Returns ``(mods, f_grid, gamma_edges)`` -- the exact triple `run` feeds
    to `classify_draw`, so a hand-placed draw is classified through the real
    production waterfall, not a reimplementation.
    """
    config = src.ServeRouteCensusConfig(
        n_samples=1, seed=SEED, n_freq=N_FREQ)
    mods = src._load_production_modules()
    f_grid = src._frequency_grid(config)
    gamma_edges = ppgo_map._gamma_band_edges()
    return mods, f_grid, gamma_edges


def _caustic_reach(gamma: float) -> float:
    """Live scalar caustic reach for ``gamma`` (kappa=0).

    ``caustic_rho(gamma, |y|) = |y| / reach``, so ``reach = 1 /
    caustic_rho(gamma, 1)``.  Deriving the reach from the SAME production
    converter the census uses keeps every fixture magnitude on the live
    boundary.
    """
    return 1.0 / ppgo_map.caustic_rho(gamma, 1.0, 0.0)


def _source_at_rho(gamma: float, target_rho: float,
                   angle_rad: float) -> tuple[float, float]:
    """Source ``(y1, y2)`` at a target ``caustic_rho`` and polar angle."""
    abs_y = target_rho * _caustic_reach(gamma)
    return abs_y * math.cos(angle_rad), abs_y * math.sin(angle_rad)


def _classify_probe(gamma: float, y1: float, y2: float,
                    m_lens_msun: float = _PROBE_MASS_MSUN):
    """Classify one hand-placed draw through the real `classify_draw`."""
    mods, f_grid, gamma_edges = _classify_env()
    return src.classify_draw(
        mods, gamma=gamma, m_lens_msun=m_lens_msun, y1=y1, y2=y2,
        f_grid=f_grid, gamma_edges=gamma_edges)


class _CensusTestCase(TestCase):
    """Base carrying the anti-vacuity guard shared by the numeric suite.

    Every test increments ``self._comparisons`` for each draw / bucket it
    actually checks; ``tearDown`` fails if a test asserted over nothing (a
    silently-empty ``records`` list would otherwise pass vacuously).
    """

    def setUp(self) -> None:
        self._comparisons = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self._comparisons, 0,
            'anti-vacuity: the test iterated zero draws/buckets and so '
            'asserted nothing -- the census report was empty.')


# ---------------------------------------------------------------------------
# 1. MECE small-run partition
# ---------------------------------------------------------------------------

class MeceSmallRunTestCase(_CensusTestCase):
    """Every draw lands in exactly one of the `SERVE_ROUTES` labels."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = _shared_report()

    def test_schema_tag_is_v1(self) -> None:
        """The report advertises the expected serialized-contract schema."""
        self.assertEqual(self.report['schema'], EXPECTED_SCHEMA)
        self._comparisons += 1

    def test_every_record_route_is_a_serve_route(self) -> None:
        """No per-draw record carries a label outside the SERVE_ROUTES set."""
        routes = frozenset(SERVE_ROUTES := src.SERVE_ROUTES)
        self.assertEqual(
            len(routes), len(SERVE_ROUTES),
            'SERVE_ROUTES contains duplicate labels')
        self.assertIn('wave_refused', routes)
        for record in self.report['records']:
            self.assertIn(
                record['route'], routes,
                f"record route {record['route']!r} is not in SERVE_ROUTES")
            self._comparisons += 1

    def test_route_counts_keys_are_exactly_serve_routes(self) -> None:
        """``route_counts`` is keyed on exactly the `SERVE_ROUTES` set."""
        self.assertEqual(
            set(self.report['route_counts']), set(src.SERVE_ROUTES))
        self._comparisons += 1

    def test_route_counts_sum_equals_n_samples(self) -> None:
        """Exhaustiveness + mutual exclusivity: the tally sums to n_samples.

        Diagnostic: a sum below ``n_samples`` is a draw that fell through
        every gate (the spec's bar-chart-short-of-total break).
        """
        self.assertEqual(
            sum(self.report['route_counts'].values()),
            self.report['n_samples'])
        self.assertEqual(self.report['n_samples'], N_SAMPLES)
        self._comparisons += 1

    def test_independent_retally_matches_route_counts(self) -> None:
        """A from-scratch Counter of ``records`` reproduces ``route_counts``.

        Independent oracle: the production ``route_counts`` is built by the
        module; this re-derives it from the per-draw records with a private
        Counter, catching an aggregation drop/double-count.
        """
        tally = Counter(r['route'] for r in self.report['records'])
        for route in src.SERVE_ROUTES:
            self.assertEqual(
                tally.get(route, 0), self.report['route_counts'][route],
                f'route_counts[{route!r}] disagrees with a record re-tally')
            self._comparisons += 1

    def test_no_artifact_means_zero_surrogate(self) -> None:
        """Demand mode (no artifact) never emits the ``surrogate`` route."""
        self.assertEqual(self.report['route_counts']['surrogate'], 0)
        self.assertEqual(self.report['header']['mode'], 'demand')
        self._comparisons += 1

    def test_wave_refused_route_is_populated(self) -> None:
        """The production-refusal route is reachable at this scale.

        Reachability makes the residual-exclusion guard bite: with
        ``wave_refused`` draws present, `ResidualPartitionTestCase`'s
        ``residual_demand['total'] == route_counts['engine_residual']``
        equality would go red if refusals were ever folded back into the
        engine-demand residual.
        """
        self.assertGreater(self.report['route_counts']['wave_refused'], 0)
        self._comparisons += 1


# ---------------------------------------------------------------------------
# 2. Residual-partition disjointness and gauge
# ---------------------------------------------------------------------------

class ResidualPartitionTestCase(_CensusTestCase):
    """The residual caustic-rho split is MECE within ``engine_residual``."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = _shared_report()

    def test_split_gauge_is_caustic_rho_not_rho_lobe(self) -> None:
        """The reported gauge is ``caustic_rho`` (F073 gauge regression)."""
        self.assertEqual(
            self.report['residual_demand']['split_gauge'],
            EXPECTED_SPLIT_GAUGE,
            "split_gauge must be 'caustic_rho'; 'rho_lobe' is the F073 "
            'gauge regression')
        self.assertNotEqual(
            self.report['residual_demand']['split_gauge'], 'rho_lobe')
        self._comparisons += 1

    def test_buckets_partition_the_engine_residual_population(self) -> None:
        """The four sub-buckets sum EXACTLY to ``engine_residual`` count.

        Diagnostic: a stacked bar of the buckets that over/under-shoots the
        ``engine_residual`` total exposes a double-count or a drop.
        """
        residual = self.report['residual_demand']
        engine_residual = self.report['route_counts']['engine_residual']
        bucket_sum = sum(
            residual[name]['count'] for name in _RESIDUAL_BUCKETS)
        self.assertEqual(bucket_sum, engine_residual)
        self.assertEqual(residual['total'], engine_residual)
        self._comparisons += 1

    def test_independent_rebin_matches_reported_buckets(self) -> None:
        """A from-scratch rho-ladder re-bin reproduces every bucket count.

        Independent oracle: re-partition the ``engine_residual`` records by
        their own ``caustic_rho`` field with a locally written threshold
        ladder (``>2`` -> born, ``(1,2]`` -> tube, ``<=1`` -> interior,
        ``None`` -> undetermined) and compare against the reported counts.
        Catches a mis-routed bucket boundary or a swapped gauge.
        """
        expected = Counter()
        for record in self.report['records']:
            if record['route'] != 'engine_residual':
                continue
            rho = record['caustic_rho']
            if rho is None:
                expected['undetermined'] += 1
            elif rho > 2.0:
                expected['born_chart_demand'] += 1
            elif rho > 1.0:
                expected['near_caustic_tube'] += 1
            else:
                expected['interior'] += 1
        residual = self.report['residual_demand']
        for name in _RESIDUAL_BUCKETS:
            self.assertEqual(
                residual[name]['count'], expected.get(name, 0),
                f'residual bucket {name!r} disagrees with an independent '
                'rho re-bin of the engine_residual records')
            self._comparisons += 1

    def test_prior_mass_fraction_equals_count_over_total(self) -> None:
        """Equal-weight draws: each bucket fraction is count / total."""
        residual = self.report['residual_demand']
        total = residual['total']
        if total == 0:
            self.skipTest('no engine_residual draws to weigh at this scale')
        for name in _RESIDUAL_BUCKETS:
            bucket = residual[name]
            self.assertAlmostEqual(
                bucket['prior_mass_fraction'], bucket['count'] / total,
                places=12)
            self._comparisons += 1


# ---------------------------------------------------------------------------
# 3. Engine-free: mock-to-raise on every exact-wave door
# ---------------------------------------------------------------------------

class _EvaluateDoor(Exception):
    """Sentinel: `ChangRefsdalChannels.evaluate` was called."""


class _FSchwingerDoor(Exception):
    """Sentinel: ``_schwinger.f_schwinger`` was called."""


class _FSchwingerMpmathDoor(Exception):
    """Sentinel: ``_schwinger._f_schwinger_mpmath`` was called."""


class _MpmathDoor(Exception):
    """Sentinel: the mpmath special-function path was entered."""


#: The four unique door sentinels; each is a plain ``Exception`` subclass so
#: NONE is a member of the census's caught refusal tuple (asserted in
#: `SelfFalsificationTestCase.test_sentinels_are_outside_the_caught_tuple`),
#: guaranteeing a door hit propagates out of ``run`` rather than being
#: swallowed into ``engine_residual``.
_DOOR_SENTINELS = (
    _EvaluateDoor, _FSchwingerDoor, _FSchwingerMpmathDoor, _MpmathDoor)


def _patched_engine_free_run(n_samples: int = ENGINE_FREE_N):
    """Run the census with every exact-wave door booby-trapped.

    Returns ``(report, door_mocks)``.  Each door mock raises its own unique
    sentinel; if ``run`` returns at all, no draw reached a door.
    """
    config = src.ServeRouteCensusConfig(
        n_samples=n_samples, seed=SEED, n_freq=N_FREQ)
    with ExitStack() as stack:
        evaluate_mock = stack.enter_context(mock.patch.object(
            ChangRefsdalChannels, 'evaluate', side_effect=_EvaluateDoor()))
        f_mock = stack.enter_context(mock.patch.object(
            _schwinger, 'f_schwinger', side_effect=_FSchwingerDoor()))
        f_mp_mock = stack.enter_context(mock.patch.object(
            _schwinger, '_f_schwinger_mpmath',
            side_effect=_FSchwingerMpmathDoor()))
        mpmath_mock = stack.enter_context(mock.patch.object(
            mpmath, 'gauss_quadrature', side_effect=_MpmathDoor()))
        report = src.run(config)
    return report, (evaluate_mock, f_mock, f_mp_mock, mpmath_mock)


class EngineFreeTestCase(_CensusTestCase):
    """Classification completes without ever opening an exact-wave door."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report, cls.door_mocks = _patched_engine_free_run()

    def test_run_completes_without_the_sentinel_escaping(self) -> None:
        """``run`` returned a full report under the four active patches.

        Reaching ``setUpClass`` without a raised sentinel is itself the
        proof; here we confirm the report is complete and well-formed.
        """
        self.assertEqual(self.report['schema'], EXPECTED_SCHEMA)
        self.assertEqual(
            sum(self.report['route_counts'].values()), ENGINE_FREE_N)
        self._comparisons += 1

    def test_zero_exact_wave_door_calls(self) -> None:
        """Every exact-wave door mock has ``call_count == 0``.

        This is the direct, load-bearing check: the sentinel would have
        fired on any door touch, so zero calls == zero engine evaluations.
        """
        names = ('evaluate', 'f_schwinger', '_f_schwinger_mpmath',
                 'mpmath.gauss_quadrature')
        for name, door in zip(names, self.door_mocks):
            self.assertEqual(
                door.call_count, 0,
                f'exact-wave door {name} was called -- a draw reached the '
                'engine, breaking the demand-map guarantee')
            self._comparisons += 1

    def test_every_route_valid_under_patches(self) -> None:
        """Even engine-free, every draw carries a valid `SERVE_ROUTES` label."""
        routes = frozenset(src.SERVE_ROUTES)
        for record in self.report['records']:
            self.assertIn(record['route'], routes)
            self._comparisons += 1


# ---------------------------------------------------------------------------
# 4. D2 route + node-kind invariance under the sign-flip quadruple
# ---------------------------------------------------------------------------

class D2SignFlipInvarianceTestCase(_CensusTestCase):
    """The serve ROUTE is D2-invariant under IEEE source sign flips.

    ``caustic_rho`` depends on ``gamma`` and ``|y|`` only, and the geometry
    partition is reflection-symmetric under ``(y1, y2) -> (+-y1, +-y2)``, so
    every gate feeding `classify_draw` commutes with a coordinate sign flip.
    The route label and the per-node ``node_route_kinds`` vector are DISCRETE
    labels, hence robust to the sub-ULP arithmetic noise a reflection can
    introduce, and must be identical -- ELEMENTWISE, not merely up to a
    permutation -- across all four mirror images.  A route that differs
    between a draw and its negation reveals a gate keyed on a SIGNED
    coordinate that should key on a magnitude.  Lobe / arm INDICES do permute
    under the mirror and are deliberately NOT asserted here.
    """

    def _quadruple(self, gamma: float, y1: float, y2: float):
        """Classify the four IEEE sign-flips of ``(y1, y2)``."""
        out = {}
        for s1 in (+1.0, -1.0):
            for s2 in (+1.0, -1.0):
                out[(s1, s2)] = _classify_probe(gamma, s1 * y1, s2 * y2)
        return out

    def test_route_and_node_kinds_are_d2_invariant(self) -> None:
        """Route + node_route_kinds match elementwise across the quadruple."""
        for label, gamma, target_rho, expected_route in _D2_FIXTURES:
            with self.subTest(fixture=label):
                y1, y2 = _source_at_rho(gamma, target_rho, _D2_ANGLE_RAD)
                # Premise 1: the four sign-flips are genuinely distinct points
                # (both coords nonzero, |y1| != |y2|), so the invariance is
                # non-trivial rather than four evaluations of one point.
                self.assertGreater(abs(y1), 0.0)
                self.assertGreater(abs(y2), 0.0)
                self.assertNotAlmostEqual(abs(y1), abs(y2), places=6)

                quad = self._quadruple(gamma, y1, y2)
                base = quad[(+1.0, +1.0)]
                # Premise 2: the base draw actually lands in the intended
                # route, so the fixture exercises the route it claims to.
                self.assertEqual(
                    base.route, expected_route,
                    f'{label}: base route {base.route!r} != expected '
                    f'{expected_route!r} -- fixture no longer on its route')

                for signs, result in quad.items():
                    self.assertEqual(
                        result.route, base.route,
                        f'{label} {signs}: route {result.route!r} != base '
                        f'{base.route!r} -- a signed-coordinate gate broke D2')
                    self.assertEqual(
                        result.node_route_kinds, base.node_route_kinds,
                        f'{label} {signs}: node_route_kinds differ '
                        'elementwise from the base draw -- D2 symmetry break')
                    self._comparisons += 1

    def test_saddle_c3_fixture_spans_both_parities(self) -> None:
        """The fixture set covers gamma<1 AND gamma>1 across >=3 routes.

        Guards the spec's "spanning both parities and multiple routes"
        premise: a shrunk fixture set that lost a parity or collapsed to one
        route would silently weaken the invariance evidence above.
        """
        gammas = {gamma for _, gamma, _, _ in _D2_FIXTURES}
        routes = {route for *_, route in _D2_FIXTURES}
        self.assertTrue(any(g < 1.0 for g in gammas), 'no astroid parity')
        self.assertTrue(any(g > 1.0 for g in gammas), 'no saddle parity')
        self.assertGreaterEqual(
            len(routes), 3, 'fixtures collapsed to fewer than 3 routes')
        self.assertIn('saddle_c3', routes)
        self.assertIn('born_analytic', routes)
        self._comparisons += 1


# ---------------------------------------------------------------------------
# 5. Saddle finite-but-huge c3 estimate REFUSES (not serves)
# ---------------------------------------------------------------------------

class SaddleFiniteHugeEstimateRefusesTestCase(_CensusTestCase):
    """A finite-but-astronomical c3 estimate must NOT admit ``saddle_c3``.

    A near-critical saddle image pair (caustic_rho just above 1) makes
    ``ppgo_error_estimate`` return a FINITE but astronomically large value
    (~1e15) instead of ``None``.  The census's saddle intercept admits only
    when ``_SADDLE_FARFIELD_SAFETY * est <= _SADDLE_FARFIELD_CERT_BAR``, so
    it must REFUSE this draw: a route of ``saddle_c3`` here would mean the
    gate admitted on "``est`` is finite" instead of on the safety-scaled bar
    -- the F069/F074 "certificate that bounds the wrong object" failure.  The
    draw instead falls through (no artifact; caustic_rho <= 2 so not
    ``born_analytic``) to a per-node exact-wave residual.
    """

    @classmethod
    def setUpClass(cls) -> None:
        gamma = _SADDLE_HUGE_GAMMA
        # On the cusp axis (angle 0) the omitted-term estimate is maximal.
        y1, y2 = _source_at_rho(gamma, _SADDLE_HUGE_RHO, 0.0)
        mods, f_grid, _ = _classify_env()
        w_grid = mods.dimensionless_frequency(f_grid, _PROBE_MASS_MSUN, 0.0)
        w_lo = float(w_grid.min())
        source = np.array([y1, y2], dtype=float)
        geom = mods.channels_cls(w_grid).geometry_partition(
            gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
        images = np.asarray(geom.images)
        matrix = mods.macro_matrix(gamma, 0.0, 0.0)
        cls.gamma = gamma
        cls.y1, cls.y2 = y1, y2
        cls.est = geometry.ppgo_error_estimate(images, source, matrix, w_lo)
        cls.real_serves = mods.saddle_farfield_serves(
            images, source, matrix, w_lo)
        cls.n_images = len(images)
        cls.result = _classify_probe(gamma, y1, y2)

    def test_estimate_is_finite_but_astronomically_large(self) -> None:
        """``ppgo_error_estimate`` is a finite huge number, not ``None``.

        This is the regime the spec targets: the near-critical merging pair
        keeps ``mu`` finite (not yet on the critical curve, so not ``None``)
        but enormous.
        """
        self.assertIsNotNone(
            self.est, 'estimate collapsed to None -- fixture is AT the '
            'critical curve, not just outside it')
        self.assertTrue(math.isfinite(self.est))
        self.assertGreater(
            self.est, _HUGE_EST_FLOOR,
            'estimate is not astronomically large -- fixture drifted away '
            'from the near-critical regime the spec probes')
        self._comparisons += 1

    def test_safety_scaled_estimate_exceeds_the_certificate_bar(self) -> None:
        """The gate's actual admission quantity is far ABOVE the bar.

        Reads the live likelihood constants so a bar/safety retune moves the
        premise with production.  ``safety * est`` dwarfs the bar, which is
        exactly why the correct gate refuses.
        """
        self.assertGreater(
            lk._SADDLE_FARFIELD_SAFETY * self.est,
            lk._SADDLE_FARFIELD_CERT_BAR)
        self._comparisons += 1

    def test_production_gate_refuses_despite_finite_estimate(self) -> None:
        """``_saddle_farfield_analytic_serves`` returns False here.

        The load-bearing contrast: the naive "``est`` is not None" predicate
        WOULD admit (est is finite), yet the real safety-scaled gate refuses.
        """
        naive_admits = self.est is not None
        self.assertTrue(naive_admits, 'fixture no longer exercises the '
                        'finite-estimate branch the naive gate would admit')
        self.assertFalse(
            self.real_serves,
            'saddle far-field gate ADMITTED a finite-but-huge estimate -- '
            'it is certifying its own blow-up (F069/F074)')
        self._comparisons += 1

    def test_route_is_not_saddle_c3_and_lands_in_engine_residual(self) -> None:
        """The census route is engine_residual, never saddle_c3."""
        self.assertEqual(self.gamma, _SADDLE_HUGE_GAMMA)  # saddle parity
        self.assertGreater(self.n_images, 0)
        self.assertNotEqual(
            self.result.route, 'saddle_c3',
            'census admitted saddle_c3 on a finite-but-huge estimate')
        self.assertEqual(
            self.result.route, 'engine_residual',
            f'near-critical saddle draw routed to {self.result.route!r}, '
            'expected engine_residual (falls through every analytic gate)')
        self.assertIn('exact_wave', self.result.node_route_kinds)
        self._comparisons += 1


# ---------------------------------------------------------------------------
# 6. Production band-ladder fixtures (audit witnesses)
# ---------------------------------------------------------------------------

#: Mass (Msun) at the top of the lens prior for the above-ceiling ladder
#: fixture: the band ceiling ``w_hi ~ 444`` clears the QD ceiling (150) so
#: the above-ceiling rung is live, while the band floor ``w_lo ~ 8.7`` stays
#: in DD-engine territory -- one draw spans all three ladder rungs.
_LADDER_MASS_MSUN = 3500.0

#: Mass (Msun) for the mixed mpmath-band fixture: the band ceiling
#: ``w_hi ~ 127`` stays AT OR BELOW the QD ceiling, so the draw can never be
#: absorbed by the ``ppgo_above_ceiling`` intercept (whose entry guard is
#: ``w_hi > 150``) and is guaranteed to reach the per-node pass.
_LADDER_TUBE_MASS_MSUN = 1000.0

#: Source polar angle (rad) for the mixed tube fixture (the audit's 30 deg).
_LADDER_ANGLE_RAD = math.radians(30.0)

#: Source polar angle for the unresolved interior fixture: ON the cusp axis
#: (angle 0), where the interior fold pair is maximally degenerate and the
#: minimum real-image delay separation collapses (measured ``min_delta_tau
#: ~ 9.1e-3`` at ``rho = 0.5``, ``gamma = 0.5``), so the production ceiling
#: gate ``W_CEILING_SCHWINGER_QD * min_delta_tau >= RHO_END`` FAILS and the
#: draw reaches the per-node pass -- the deferred-2b unresolved near-caustic
#: corner.  Off-axis (the audit's 30 deg) the same draw now gate-passes and
#: is served by the above-ceiling intercept, so it can no longer witness the
#: refusal ladder.
_LADDER_CUSP_ANGLE_RAD = 0.0


class ProductionBandLadderTestCase(_CensusTestCase):
    """Per-node kinds follow the production band ladder (audit witnesses).

    Two hand-placed ``gamma = 0.5`` draws that REACH the per-node pass under
    the band-split serving semantics pin the audited ladder invariants:

    * DEFERRED-2B REFUSAL: the on-cusp-axis interior draw (``caustic_rho =
      0.5``, angle 0, ``M = 3500``, w band ~ [8.7, 444]) FAILS the
      production ceiling gate (its fold-degenerate ``min_delta_tau``
      collapses, ``150 * min_delta_tau < RHO_END`` -- premise-guarded), so
      the above-ceiling intercept declines and the draw falls to the node
      pass, where its arm-declined above-QD-ceiling wave nodes are
      DETERMINISTIC production refusers (``SchwingerCertificationError``,
      ``lnL = -inf``): the census labels those nodes ``refused`` and the
      draw ``wave_refused`` -- even though it also carries exact_wave nodes
      (the precedence pin: refusal beats ``engine_residual``, keeping
      refusals out of the sizing residual).
    * ARM THRESHOLD: the mixed tube-shell draw (``caustic_rho = 1.5``,
      ``M = 1000``, w band ~ [2.5, 127] -- ceiling ``<= 150`` so the
      above-ceiling intercept can never absorb it) shows arm-served nodes in
      the mpmath band ``(60, 150]``, mirroring production's arms-first offer
      there; every ``w <= 60`` node stays exact_wave (production has no arms
      below the DD ceiling).
    * SELECT_BRANCH BELOW CEILING: no node at or below the QD ceiling is
      labelled ``geometric`` (or ``refused``) -- production consults
      ``select_branch`` only above 150.
    """

    @classmethod
    def setUpClass(cls) -> None:
        mods, f_grid, _ = _classify_env()
        cls.mods = mods
        cls.masses = {'interior_unresolved': _LADDER_MASS_MSUN,
                      'tube_mixed': _LADDER_TUBE_MASS_MSUN}
        cls.w_grids = {
            label: np.asarray(mods.dimensionless_frequency(
                f_grid, mass, 0.0), dtype=float)
            for label, mass in cls.masses.items()}
        cls.results = {}
        for label, rho, angle in (
                ('interior_unresolved', 0.5, _LADDER_CUSP_ANGLE_RAD),
                ('tube_mixed', 1.5, _LADDER_ANGLE_RAD)):
            y1, y2 = _source_at_rho(0.5, rho, angle)
            cls.results[label] = _classify_probe(
                0.5, y1, y2, m_lens_msun=cls.masses[label])

    def _kinds(self, label: str) -> np.ndarray:
        """Node-kind vector as an array, length-checked against the grid."""
        kinds = np.array(self.results[label].node_route_kinds)
        self.assertEqual(kinds.size, self.w_grids[label].size)
        return kinds

    def test_band_spans_the_ladder_rungs(self) -> None:
        """Premise guard: the fixture bands cover the intended rungs.

        The unresolved interior fixture crosses BOTH ceilings (all three
        rungs live); the mixed fixture crosses the DD ceiling but stays at
        or below the QD ceiling, so it reaches the per-node pass by
        construction (the above-ceiling intercept's ``w_hi > 150`` entry
        guard cannot fire).
        """
        interior = self.w_grids['interior_unresolved']
        self.assertLessEqual(interior.min(), self.mods.w_ceiling_dd)
        self.assertGreater(interior.max(), self.mods.w_ceiling_qd)
        tube = self.w_grids['tube_mixed']
        self.assertLessEqual(tube.min(), self.mods.w_ceiling_dd)
        self.assertGreater(tube.max(), self.mods.w_ceiling_dd)
        self.assertLessEqual(tube.max(), self.mods.w_ceiling_qd)
        self._comparisons += 1

    def test_unresolved_above_ceiling_draw_is_wave_refused(self) -> None:
        """The gate-failing w_hi~444 interior draw classifies wave_refused.

        Premise guard: the production ceiling gate genuinely FAILS here
        (``W_CEILING_SCHWINGER_QD * min_delta_tau < RHO_END``, recomputed
        from the geometry-only partition with production-bound constants),
        so the draw is the deferred-2b unresolved corner the above-ceiling
        intercept must NOT absorb.  Its ``refused`` nodes all sit above the
        QD ceiling, and exact_wave demand coexists in the same draw -- yet
        the deterministic refusal takes precedence, so the draw is excluded
        from the campaign-sizing residual.
        """
        result = self.results['interior_unresolved']
        w_grid = self.w_grids['interior_unresolved']
        geom = self.mods.channels_cls(w_grid).geometry_partition(
            gamma=result.gamma, y=(result.y1, result.y2),
            beta=0.0, kappa=0.0)
        real_delays = np.asarray(geom.delays)[
            np.asarray(geom.real_mask, dtype=bool)]
        deltas = np.diff(np.sort(real_delays))
        min_delta_tau = float(deltas[deltas > 0].min())
        self.assertLess(
            self.mods.w_ceiling_qd * min_delta_tau, self.mods.rho_end,
            'premise: the ceiling gate PASSES for this fixture, so the '
            'above-ceiling intercept serves it and the refusal ladder is '
            'no longer witnessed -- move the fixture deeper into the '
            'unresolved corner')
        kinds = self._kinds('interior_unresolved')
        self.assertEqual(
            result.route, 'wave_refused',
            'gate-failing above-ceiling draw was not routed to '
            'wave_refused -- a production refusal is being conflated with '
            'engine demand')
        refused = kinds == 'refused'
        self.assertTrue(refused.any())
        self.assertTrue(
            (w_grid[refused] > self.mods.w_ceiling_qd).all(),
            "'refused' at or below the QD ceiling: no deterministic "
            'refusal exists there (an engine does)')
        self.assertIn('exact_wave', kinds)  # precedence, not vacuity
        self._comparisons += 1

    def test_mixed_draw_arms_serve_the_mpmath_band(self) -> None:
        """Witness v-a: the corrected node map arm-serves (60, 150] nodes."""
        kinds = self._kinds('tube_mixed')
        w_grid = self.w_grids['tube_mixed']
        mid = ((w_grid > self.mods.w_ceiling_dd)
               & (w_grid <= self.mods.w_ceiling_qd))
        self.assertTrue(mid.any())
        self.assertTrue(
            np.isin(kinds[mid], tuple(src._UNIFORM_ARM_KINDS)).any(),
            'no arm-served node in (60, 150]: the census is not mirroring '
            "production's arms-first mpmath band")
        self.assertEqual(self.results['tube_mixed'].route, 'engine_residual')
        self._comparisons += 1

    def test_no_select_branch_at_or_below_the_qd_ceiling(self) -> None:
        """No 'geometric'/'refused' label at w <= 150 in either fixture."""
        for label in self.results:
            kinds = self._kinds(label)
            below = self.w_grids[label] <= self.mods.w_ceiling_qd
            self.assertFalse(
                np.isin(kinds[below], ('geometric', 'refused')).any(),
                f'{label}: select_branch consulted at or below the QD '
                'ceiling -- production band-ladder violation')
            self._comparisons += 1

    def test_dd_band_nodes_are_unconditional_engine_demand(self) -> None:
        """Every w <= 60 node is exact_wave in both fixtures (no arms there)."""
        for label in self.results:
            kinds = self._kinds(label)
            low = self.w_grids[label] <= self.mods.w_ceiling_dd
            self.assertTrue(low.any())
            self.assertTrue(
                (kinds[low] == 'exact_wave').all(),
                f'{label}: a w<=60 node escaped the DD engine -- production '
                'offers no arm or asymptote below the DD ceiling')
            self._comparisons += 1


# ---------------------------------------------------------------------------
# 7. Band-split serving intercepts (c3 split + above-ceiling split)
# ---------------------------------------------------------------------------

#: Band-split saddle fixture: ``gamma = 3`` at ``caustic_rho = 1.2`` fails
#: the whole-band gate at the band floor (premise-guarded) but its c3
#: certificate split point lies strictly inside the band and below the
#: engine ceiling, so the revived band-split rung serves it end-to-end.
_C3_SPLIT_RHO = 1.2

#: Whole-band saddle fixture (same as the D2 ``saddle_farfield_c3`` draw):
#: the gate passes at the band floor, i.e. ``w_split <= w_lo``.
_C3_WHOLE_BAND_RHO = 3.0


class BandSplitInterceptTestCase(_CensusTestCase):
    """The census mirrors the band-split serving rungs of commit 6958f0c.

    Three hand-placed saddle draws pin the revived taxonomy:

    * c3 BAND SPLIT: a draw the whole-band gate REFUSES at ``w_lo`` but the
      per-draw certificate split ``w_lo < w_split < w_hi`` (and ``<= 150``)
      admits classifies ``saddle_c3``, with ``w_split`` recorded in the
      per-draw detail and equal to the production
      ``_saddle_c3_split_point`` value (the census binds the helper, never
      re-derives it).
    * WHOLE-BAND ADMIT: the historical whole-band serve stays ``saddle_c3``
      and records its (``<= w_lo``) split point.
    * PRODUCTION PRECEDENCE: a draw BOTH rungs would serve (ceiling gate
      passes AND the c3 gate passes, ``w_hi > 150``) classifies
      ``ppgo_above_ceiling`` -- the above-ceiling rung fires before the
      saddle rung in ``likelihood._amplification_coefficients``.
    """

    @classmethod
    def setUpClass(cls) -> None:
        mods, f_grid, _ = _classify_env()
        cls.mods = mods
        cls.f_grid = f_grid

    def _saddle_env(self, rho: float, mass: float):
        """Classify a gamma=3 saddle draw and return production quantities."""
        gamma = 3.0
        y1, y2 = _source_at_rho(gamma, rho, _D2_ANGLE_RAD)
        w_grid = np.asarray(self.mods.dimensionless_frequency(
            self.f_grid, mass, 0.0), dtype=float)
        geom = self.mods.channels_cls(w_grid).geometry_partition(
            gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
        images = np.asarray(geom.images)
        source = np.array([y1, y2], dtype=float)
        matrix = self.mods.macro_matrix(gamma, 0.0, 0.0)
        w_lo, w_hi = float(w_grid.min()), float(w_grid.max())
        serves_lo = self.mods.saddle_farfield_serves(
            images, source, matrix, w_lo)
        w_split = lk._saddle_c3_split_point(images, source, matrix)
        result = _classify_probe(gamma, y1, y2, m_lens_msun=mass)
        return result, serves_lo, w_split, w_lo, w_hi, geom

    def test_band_split_saddle_serves_and_records_w_split(self) -> None:
        """The revived c3 band split admits and records the split point.

        Premise guards: the whole-band gate refuses at ``w_lo`` (this draw
        was ``wave_refused``/residual under the old whole-band-only mirror)
        and the production split point lies strictly inside the band, at or
        below the engine ceiling.  The pin: route ``saddle_c3``, with the
        recorded ``w_split`` numerically identical to the production
        helper's value (same helper, no re-typed certificate).
        """
        (result, serves_lo, w_split, w_lo, w_hi,
         _geom) = self._saddle_env(_C3_SPLIT_RHO, _PROBE_MASS_MSUN)
        self.assertFalse(
            serves_lo, 'premise: whole-band gate now admits at w_lo -- '
            'fixture no longer exercises the band-split rung')
        self.assertIsNotNone(w_split)
        self.assertGreater(w_split, w_lo)
        self.assertLess(w_split, w_hi)
        self.assertLessEqual(w_split, self.mods.w_ceiling_qd)
        self.assertEqual(
            result.route, 'saddle_c3',
            'band-splittable saddle draw not served by the c3 rung -- the '
            'census still mirrors the retired whole-band-only gate')
        self.assertEqual(result.node_route_kinds, ())
        self.assertEqual(result.w_split, w_split)
        self._comparisons += 1

    def test_whole_band_saddle_records_split_at_or_below_floor(self) -> None:
        """The historical whole-band admit stays saddle_c3, w_split <= w_lo."""
        (result, serves_lo, w_split, w_lo, _w_hi,
         _geom) = self._saddle_env(_C3_WHOLE_BAND_RHO, _PROBE_MASS_MSUN)
        self.assertTrue(
            serves_lo, 'premise: whole-band gate refuses at w_lo -- '
            'fixture drifted off the whole-band-admit branch')
        self.assertEqual(result.route, 'saddle_c3')
        self.assertIsNotNone(result.w_split)
        self.assertEqual(result.w_split, w_split)
        self.assertLessEqual(result.w_split, w_lo)
        self._comparisons += 1

    def test_ceiling_intercept_precedes_the_saddle_rung(self) -> None:
        """A draw BOTH rungs would serve classifies ppgo_above_ceiling.

        Premise guards: ``w_hi > 150``, the production ceiling gate passes
        (``150 * min_delta_tau >= RHO_END``, recomputed from the
        geometry-only partition), AND the c3 whole-band gate passes.  The
        pin is the production rung ORDER: the above-ceiling split fires
        first, so the draw counts as ``ppgo_above_ceiling`` (whole-band
        intercept: empty node kinds, no ``w_split`` detail).
        """
        (result, serves_lo, _w_split, _w_lo, w_hi,
         geom) = self._saddle_env(_C3_WHOLE_BAND_RHO, _LADDER_MASS_MSUN)
        self.assertGreater(w_hi, self.mods.w_ceiling_qd)
        self.assertTrue(
            serves_lo, 'premise: the saddle rung would not serve this draw, '
            'so it cannot witness the precedence')
        real_delays = np.asarray(geom.delays)[
            np.asarray(geom.real_mask, dtype=bool)]
        deltas = np.diff(np.sort(real_delays))
        min_delta_tau = float(deltas[deltas > 0].min())
        self.assertGreaterEqual(
            self.mods.w_ceiling_qd * min_delta_tau, self.mods.rho_end,
            'premise: the ceiling gate would not admit this draw, so it '
            'cannot witness the precedence')
        self.assertEqual(
            result.route, 'ppgo_above_ceiling',
            'production precedence broken: the above-ceiling rung fires '
            'before the saddle rung in _amplification_coefficients')
        self.assertEqual(result.node_route_kinds, ())
        self.assertIsNone(result.w_split)
        self._comparisons += 1


# ---------------------------------------------------------------------------
# 8. Report-level derived-set and detail pins (shared small run)
# ---------------------------------------------------------------------------

class ReportBandSplitDetailTestCase(_CensusTestCase):
    """Report-level pins for the band-split taxonomy on the shared run."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = _shared_report()

    def test_w_split_detail_is_recorded_iff_saddle_c3(self) -> None:
        """``w_split`` is present only on the ``_W_SPLIT_ROUTES`` records.

        ``saddle_c3`` and ``diffractive_engine_hosted`` are the routes with
        a legitimate per-draw certificate split (module-level
        ``_W_SPLIT_ROUTES``, single-sourced so a future band-split route
        extends one set, not this test).  Both ``saddle_c3`` admit branches
        record it (the whole-band admit's inversion is ``<= w_lo``); every
        route outside the set carries ``None`` -- the c3/ceiling
        engine-below bands are per-draw detail, never new residual keys.
        """
        saddle_seen = 0
        for record in self.report['records']:
            if record['route'] == 'saddle_c3':
                saddle_seen += 1
                self.assertIsNotNone(
                    record['w_split'],
                    'saddle_c3 record without its w_split detail')
                self.assertGreater(record['w_split'], 0.0)
            elif record['route'] in _W_SPLIT_ROUTES:
                if record['w_split'] is not None:
                    self.assertGreater(record['w_split'], 0.0)
            else:
                self.assertIsNone(
                    record['w_split'],
                    f"non-split route {record['route']!r} carries a "
                    'w_split detail')
            self._comparisons += 1
        self.assertGreater(
            saddle_seen, 0,
            'premise: no saddle_c3 draw at this scale -- the iff pin is '
            'half-vacuous')

    def test_wave_refused_is_the_failed_ceiling_gate_set(self) -> None:
        """Every wave_refused record sits above the ceiling with a failed gate.

        The derived-set pin: ``wave_refused`` is ONLY the deferred-2b
        corner -- draws whose band reaches above ``W_CEILING_SCHWINGER_QD``
        AND whose production ceiling gate fails (``150 * min_delta_tau <
        RHO_END``, recomputed here from the geometry-only partition with
        production-bound constants).  A wave_refused record that gate-passes
        would be a draw the production above-ceiling rung serves -- the
        pre-band-split blindness this census update removes.
        """
        mods, f_grid, _ = _classify_env()
        refused_records = [r for r in self.report['records']
                           if r['route'] == 'wave_refused']
        self.assertGreater(
            len(refused_records), 0,
            'premise: no wave_refused draw at this scale')
        for record in refused_records:
            self.assertGreater(
                math.exp(record['log_w_max']), mods.w_ceiling_qd,
                'wave_refused record whose band never exceeds the QD '
                'ceiling -- refusal is impossible there')
            w_grid = mods.dimensionless_frequency(
                f_grid, record['m_lens_msun'], 0.0)
            geom = mods.channels_cls(w_grid).geometry_partition(
                gamma=record['gamma'], y=(record['y1'], record['y2']),
                beta=0.0, kappa=0.0)
            self.assertFalse(
                src._ppgo_ceiling_gate_passes(mods, geom),
                'wave_refused record whose ceiling gate PASSES -- '
                'production serves this draw via _ppgo_above_ceiling, so '
                'the census is refusing a served draw')
            self._comparisons += 1


# ---------------------------------------------------------------------------
# 9. Self-falsification -- prove each guard above can go red
# ---------------------------------------------------------------------------

class SelfFalsificationTestCase(TestCase):
    """Each guard the suite relies on is shown to have teeth."""

    def test_route_membership_guard_has_teeth(self) -> None:
        """A record with a bogus route fails the membership predicate."""
        routes = frozenset(src.SERVE_ROUTES)
        forged = [{'route': 'born_analytic'}, {'route': '__not_a_route__'}]
        self.assertFalse(all(r['route'] in routes for r in forged))

    def test_exhaustiveness_guard_has_teeth(self) -> None:
        """A dropped draw makes the re-tally fall short of n_samples."""
        records = [{'route': 'engine_residual'}] * (N_SAMPLES - 1)
        tally = Counter(r['route'] for r in records)
        self.assertNotEqual(sum(tally.values()), N_SAMPLES)

    def test_residual_sum_guard_has_teeth(self) -> None:
        """A dropped residual bucket makes the sub-counts miss the total."""
        residual = {'born_chart_demand': {'count': 10},
                    'near_caustic_tube': {'count': 5},
                    'interior': {'count': 3},
                    'undetermined': {'count': 0}}
        engine_residual = 20  # deliberately != 18 = the true bucket sum
        bucket_sum = sum(residual[name]['count'] for name in _RESIDUAL_BUCKETS)
        self.assertNotEqual(bucket_sum, engine_residual)

    def test_gauge_regression_guard_has_teeth(self) -> None:
        """A ``rho_lobe`` gauge value trips the split-gauge assertion."""
        with self.assertRaises(AssertionError):
            self.assertEqual('rho_lobe', EXPECTED_SPLIT_GAUGE)

    def test_engine_door_patch_is_wired(self) -> None:
        """Touching any patched door raises its unique sentinel.

        Positive control for `EngineFreeTestCase`: proves the booby-trap is
        live (not a no-op patch), so ``run`` completing means zero touches.
        """
        with mock.patch.object(
                _schwinger, 'f_schwinger', side_effect=_FSchwingerDoor()):
            with self.assertRaises(_FSchwingerDoor):
                _schwinger.f_schwinger(80.0, None, 0.5)
        with mock.patch.object(
                ChangRefsdalChannels, 'evaluate', side_effect=_EvaluateDoor()):
            with self.assertRaises(_EvaluateDoor):
                ChangRefsdalChannels(__import__('numpy').array([1.0, 2.0])
                                     ).evaluate(gamma=0.5, y=(0.1, 0.1),
                                                beta=0.0, kappa=0.0)

    def test_sentinels_are_outside_the_caught_tuple(self) -> None:
        """No door sentinel is a member of the census's caught refusal tuple.

        The classifier catches ``_REFUSAL_ERRORS + (ValueError,
        ZeroDivisionError)``; a sentinel matching any of those would be
        silently swallowed into ``engine_residual`` (the masking bug the
        spec warns against).  Each sentinel must be disjoint from that tuple.
        """
        caught = src._load_production_modules().refusal_errors + (
            ValueError, ZeroDivisionError)
        for sentinel in _DOOR_SENTINELS:
            self.assertFalse(
                issubclass(sentinel, caught),
                f'{sentinel.__name__} is caught by the classifier and would '
                'be swallowed into engine_residual')

    def test_d2_invariance_guard_has_teeth(self) -> None:
        """A sign-keyed route makes the quadruple-equality predicate fail.

        Positive control for `D2SignFlipInvarianceTestCase`: a classifier
        that keys on ``sign(y1)`` yields two distinct route labels across the
        four sign-flips, so the "all four equal" check must reject it.
        """
        def _sign_keyed_route(y1: float) -> str:
            return 'pos' if y1 >= 0.0 else 'neg'

        y1 = 0.7
        routes = {(s1, s2): _sign_keyed_route(s1 * y1)
                  for s1 in (+1.0, -1.0) for s2 in (+1.0, -1.0)}
        base = routes[(+1.0, +1.0)]
        self.assertFalse(all(r == base for r in routes.values()))

    def test_elementwise_kinds_guard_rejects_a_permutation(self) -> None:
        """Elementwise tuple equality rejects a permuted node-kind vector.

        The spec requires ELEMENTWISE equality of ``node_route_kinds``, not
        multiset equality: a vector that matches only up to a permutation is
        a symmetry break.  A tuple compare (unlike a ``Counter`` compare)
        catches it.
        """
        base = ('geometric', 'exact_wave', 'exact_wave')
        permuted = ('exact_wave', 'geometric', 'exact_wave')
        self.assertNotEqual(base, permuted)          # elementwise: differ
        self.assertEqual(Counter(base), Counter(permuted))  # multiset: same

    def test_w_split_iff_guard_has_teeth(self) -> None:
        """A forged w_split on a non-saddle route fails the iff predicate.

        Positive control for `ReportBandSplitDetailTestCase`: the detail
        pin requires ``(route == 'saddle_c3') == (w_split is not None)``;
        a record that leaks the detail onto another route (or a saddle_c3
        record that drops it) violates the predicate.
        """
        legit = ({'route': 'saddle_c3', 'w_split': 30.7},
                 {'route': 'engine_residual', 'w_split': None})
        forged = ({'route': 'engine_residual', 'w_split': 30.7},
                  {'route': 'saddle_c3', 'w_split': None})
        predicate = (lambda r:
                     (r['route'] == 'saddle_c3') == (r['w_split'] is not None))
        self.assertTrue(all(predicate(r) for r in legit))
        for record in forged:
            self.assertFalse(predicate(record))

    def test_saddle_finite_huge_refusal_guard_has_teeth(self) -> None:
        """The naive 'est is not None' gate admits where the safe gate refuses.

        Positive control for `SaddleFiniteHugeEstimateRefusesTestCase`: on a
        finite-but-huge estimate the naive predicate returns True (admit)
        while ``safety * est <= bar`` returns False (refuse), so a regression
        to the naive gate would flip the suite's refusal assertions red.
        """
        huge_est = 4.8e15
        naive_admits = huge_est is not None
        safe_admits = (
            lk._SADDLE_FARFIELD_SAFETY * huge_est
            <= lk._SADDLE_FARFIELD_CERT_BAR)
        self.assertTrue(naive_admits)
        self.assertFalse(safe_admits)


if __name__ == '__main__':
    main()
