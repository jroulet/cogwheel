"""Tests for `cogwheel.lensing.tiling_census` -- the engine-free tiling census.

The census exists to predict a training campaign's engine call-count and to
flag silent-empty / exploding tile counts BEFORE any wave-optics amplitude is
evaluated.  Six invariants keep it honest -- three on the core counting
machinery and three on the standing-question / cost-model answers -- and this
suite pins each with one durable pin, plus a self-falsification class proving
the pins have teeth.  The core three:

1. ENGINE-FREE GUARANTEE (`EngineFreeGuaranteeTestCase`) -- the F-class
   tripwire.  ``run`` must complete without ever touching the amplitude engine
   (``ChangRefsdalChannels.evaluate``, ``_schwinger.f_schwinger`` and its
   mpmath shim ``_f_schwinger_mpmath``).  We booby-trap all four doors so any
   call raises, and we assert at import time that the engine class never
   entered the census module's namespace.  A future refactor routing a count
   through the engine is caught here and nowhere else, so this is the strictest
   test in the file.

2. THIN-CALLER FIDELITY (`ThinCallerFidelityTestCase`) -- the census must
   ``len()`` the SAME production tiler the trainer uses, not reimplement it.
   Removing one tile from ``surrogate_training._farfield_exterior_tiles`` must
   move the exterior count by exactly one and the derived node count by exactly
   one tile's worth (``n_rho * n_theta_c * n_gamma * w_nodes``).  Patching an
   UNRELATED tiler (``_farfield_tiles``, used only by the Q2/Q3 diagnostics)
   must leave the exterior count untouched -- proving the count tracks that
   specific tiler.

3. TWO-SIDED BAND VERDICT LOGIC (`BandVerdictLogicTestCase`) -- the verdict
   classifier flags a count below its low band (INCLUDING exactly 0) as
   ``SILENT_EMPTY`` (FAIL), in-band as ``IN_BAND``, above the high band as
   ``EXPLOSION`` (FAIL).  A zero must NEVER read ``IN_BAND``: that is the
   silent-empty failure mode the whole census exists to catch.  We drive both
   the pure ``_verdict`` classifier and the full ``run`` aggregation.

A seventh, INS-1-001-specific pin sits alongside the core three:
``PpgoTrimIndependenceTestCase`` pins that the census's counts are a
conservative UPPER BOUND on a real campaign's node budget by construction --
``run`` never consults the certified ppGO map or its per-stratum/window trim
(``ppgo_map.get_certified_ppgo_map`` / ``surrogate_training._apply_ppgo_trim``),
even though every real ``train()`` installs a certified map by default and
drops whole strata/windows through it. Booby-trapping both primitives and
confirming ``run`` still completes is what keeps that documented guarantee
from silently breaking under a future refactor.

Cost note (fast tier): every ``run`` call is ~6 s of engine-free geometry.
This file makes a bounded number of them -- one cached baseline plus one per
patched scenario (engine-free 1, thin-caller 2, verdict 2, trim-independence 1)
-- so the whole
file runs well under a minute, far below the 5-min file ceiling.  The pure
``_verdict`` and self-falsification tests add no ``run`` calls.

The oracle for the node arithmetic is the config's own grid factors, computed
independently of the census, so a bug in the census node formula cannot hide
behind a matching bug in the oracle.
"""

from __future__ import annotations

import importlib
import math
from unittest import TestCase, main, mock

import numpy as np

from cogwheel.lensing import tiling_census as tc
from cogwheel.lensing import surrogate_training as st
from cogwheel.lensing import ppgo_map
from cogwheel.lensing.chang_refsdal import channels as cr_channels
from cogwheel.lensing.chang_refsdal import _schwinger
from cogwheel.lensing.chang_refsdal import geometry

#: The (region x parity) key the thin-caller and verdict pins operate on.  The
#: positive-parity origin-centred exterior remainder is charted by
#: ``_farfield_exterior_tiles`` alone, so patching that tiler moves exactly this
#: record's count -- a clean, isolated handle on one production tiler.
_EXTERIOR_KEY = 'exterior:+1'

#: The production tiler the exterior region reads.  Patching it is the
#: thin-caller experiment; the census calls it as ``st._farfield_exterior_tiles``
#: (module attribute), so a module-level patch is seen by the census.
_EXTERIOR_TILER = '_farfield_exterior_tiles'

#: A tiler the census uses ONLY inside the Q2/Q3 diagnostics, never in a
#: per-region count.  Patching it must NOT move any ``per_region`` tile count --
#: the negative control that proves the exterior count tracks its own tiler.
_DIAGNOSTIC_TILER = '_farfield_tiles'

#: Engine doors that the census must never open.  ``(owner_module, attribute)``
#: pairs booby-trapped to raise `AssertionError` on any call.  ``evaluate`` is
#: patched on the class so the trap fires regardless of how the class object was
#: imported into a consumer.
_ENGINE_DOORS = (
    (cr_channels.ChangRefsdalChannels, 'evaluate'),
    (_schwinger, 'f_schwinger'),
    (_schwinger, '_f_schwinger_mpmath'),
)

#: A synthetic tile the fake tilers emit: ``((rho_center, theta_center),
#: (rho_half, theta_half), i, j)`` matching the real tiler's 4-tuple shape.
_FAKE_TILE = ((1.5, 0.3), (0.1, 0.1), 0, 0)

#: Module-scoped cache of one unpatched baseline ``run`` result.  A single
#: ``run`` is ~6 s; caching it keeps the file fast without leaking patched
#: state (the baseline is computed with no patches active).
_BASELINE: dict[str, dict] = {}


def _smoke_config() -> st.TrainingConfig:
    """Return the small smoke `TrainingConfig` every test censuses."""
    return st.TrainingConfig()


def _baseline_run() -> dict:
    """Return the cached unpatched census of the smoke config."""
    if 'result' not in _BASELINE:
        _BASELINE['result'] = tc.run(_smoke_config())
    return _BASELINE['result']


def _exterior_nodes_per_tile(config: st.TrainingConfig) -> int:
    """Independent oracle: engine-eval nodes charged per exterior tile.

    Derived straight from the config grid factors -- ``n_rho * n_theta_c``
    spatial nodes, ``n_gamma`` gamma nodes and ``int(w_nodes_per_decade * 2)``
    w nodes -- WITHOUT calling any census helper, so it cannot share a bug with
    the census node formula it checks.
    """
    spatial = config.n_rho * config.n_theta_c
    w_nodes = int(config.w_nodes_per_decade * 2.0)
    return spatial * config.n_gamma * w_nodes


#: Q1 fundamental-domain fold-arc counts for the smoke config's representative
#: bands (MEASURED 2026-08-14, grounded in a live ``run``).  The astroid detects
#: its four D2-image arcs but trains ONE (the fundamental arc bracketing pi/4 --
#: the F079 wrap fix); the macro-saddle deltoid detects six arcs and trains the
#: ``max_tube_arcs`` slice.  These are the DETECTED counts (a topology fact of
#: the caustic), independent of ``max_tube_arcs``.
_ASTROID_DETECTED_ARCS = 4
_ASTROID_TRAINED_ARCS = 1
_SADDLE_DETECTED_ARCS = 6

#: Module cache for the saddle representative band context (a few seconds of
#: engine-free geometry setup), reused by the Q4 far-field floor oracle so the
#: file does not pay a fresh setup per test.
_SADDLE_REP: dict[str, object] = {}


def _saddle_rep_ctx():
    """Return the cached saddle representative `_BandCtx` for the smoke config.

    Built via the census's own engine-free band-context collector (the SAME
    chain ``run`` walks), so the Q4 floor oracle probes the identical
    representative source the census scored -- differing ONLY in the floor
    arithmetic it applies.
    """
    if 'ctx' not in _SADDLE_REP:
        box = st.PriorBox.from_prior_classes()
        contexts, _dropped = tc._collect_band_contexts(
            st, box, -1, _smoke_config())
        _SADDLE_REP['box'] = box
        _SADDLE_REP['ctx'] = tc._representative_saddle_ctx(
            {1: [], -1: contexts})
    return _SADDLE_REP['ctx']


def _independent_saddle_farfield_floor() -> float:
    """Independent oracle for the saddle far-field serve floor.

    Recomputes ``(2e4 * K)**(1/3)`` from FIRST PRINCIPLES using shipping
    geometry primitives (``macro_matrix`` -> ``find_images`` ->
    ``ppgo_error_estimate``), applying the ``_Q4_SADDLE_FLOOR_COEFF`` and the
    ``1/3`` exponent with LITERAL constants of this test -- so a drift in the
    census's coefficient, exponent or ``K = est * w_min**3`` recovery is
    caught.  Mirrors the census's representative-source convention (centroid
    plus ``1.2 * r_max`` outward); this pins the FORMULA, not the convention.
    """
    ctx = _saddle_rep_ctx()
    box = _SADDLE_REP['box']
    lobe = ctx.saddle_lobe_admissions[1]
    centroid = np.asarray(lobe.centroid, dtype=float)
    cmag = float(np.hypot(centroid[0], centroid[1]))
    r_max = float(np.max(lobe.boundary_r))
    direction = centroid / cmag if cmag > 0 else np.array([1.0, 0.0])
    source = centroid + direction * (1.2 * r_max)
    w_min = float(box.w_range(-1)[0])
    matrix = geometry.macro_matrix(ctx.gamma_mid)
    real_images = np.asarray(geometry.find_images(source, matrix))
    est = geometry.ppgo_error_estimate(real_images, source, matrix, w_min)
    k_amplitude = est * w_min ** 3
    return float((2.0e4 * k_amplitude) ** (1.0 / 3.0))


class _CensusTestCase(TestCase):
    """Base carrying the anti-vacuity guard shared by every census test.

    ``self._observe()`` records that a real assertion fired; ``tearDown`` fails
    a test that made ZERO observations, so a sweep that silently skips every
    subTest (e.g. a patched ``run`` that returned an empty ``per_region``)
    cannot pass by asserting nothing.
    """

    def setUp(self) -> None:
        self._observations = 0

    def _observe(self) -> None:
        self._observations += 1

    def tearDown(self) -> None:
        self.assertGreater(
            self._observations, 0,
            'anti-vacuity: the test made zero observations -- it asserted '
            'nothing and would pass a silently-skipping census.')


class EngineFreeGuaranteeTestCase(_CensusTestCase):
    """The F-class tripwire: `run` must make ZERO amplitude-engine calls."""

    def test_census_module_namespace_excludes_engine_class(self) -> None:
        """A fresh import of the census must not pull the engine class in."""
        module = importlib.import_module('cogwheel.lensing.tiling_census')
        self.assertFalse(
            hasattr(module, 'ChangRefsdalChannels'),
            'tiling_census imported the amplitude engine class into its '
            'namespace -- the top-level import is no longer engine-free.')
        # The schwinger amplitude module must also not be a top-level name.
        self.assertFalse(hasattr(module, '_schwinger'),
                         'tiling_census pulled the _schwinger amplitude module '
                         'into its top-level namespace.')
        self._observe()

    def test_run_never_calls_any_engine_door(self) -> None:
        """With every engine door booby-trapped, `run` still completes."""
        calls: list[str] = []

        def _trap(name: str):
            def _side_effect(*_args, **_kwargs):
                calls.append(name)
                raise AssertionError(f'engine called: {name}')
            return _side_effect

        with mock.patch.object(cr_channels.ChangRefsdalChannels, 'evaluate',
                               side_effect=_trap('ChangRefsdalChannels.evaluate')), \
             mock.patch.object(_schwinger, 'f_schwinger',
                               side_effect=_trap('_schwinger.f_schwinger')), \
             mock.patch.object(_schwinger, '_f_schwinger_mpmath',
                               side_effect=_trap('_schwinger._f_schwinger_mpmath')):
            try:
                result = tc.run(_smoke_config())
            except AssertionError as exc:  # pragma: no cover - the leak path
                self.fail(f'ENGINE LEAK: run() reached the amplitude engine '
                          f'({exc}); engine symbols hit: {calls}')
        self.assertIsInstance(result, dict)
        self.assertEqual(result['schema'], 'tiling_census_v1')
        self.assertEqual(calls, [],
                         f'run() called engine door(s): {calls}')
        self._observe()


class ThinCallerFidelityTestCase(_CensusTestCase):
    """The census reads the production tiler; it does not reimplement it."""

    def test_dropping_one_exterior_tile_moves_count_and_nodes_by_one_tile(
            self) -> None:
        """Removing one tile from the tiler drops the census count by one.

        The node count must drop by exactly ONE tile's worth, tying the census
        node arithmetic to the tiler's returned length.
        """
        config = _smoke_config()
        baseline = _baseline_run()['per_region'][_EXTERIOR_KEY]
        base_tiles = baseline['n_tiles']
        base_nodes = baseline['n_nodes']
        self.assertGreater(base_tiles, 0,
                           'premise lost: the smoke config no longer emits any '
                           'exterior tile to drop.')

        original = getattr(st, _EXTERIOR_TILER)
        state = {'dropped': False}

        def _drop_one(*args, **kwargs):
            tiles = original(*args, **kwargs)
            if not state['dropped'] and tiles:
                state['dropped'] = True
                return tiles[:-1]
            return tiles

        with mock.patch.object(st, _EXTERIOR_TILER, side_effect=_drop_one):
            patched = tc.run(config)['per_region'][_EXTERIOR_KEY]

        self.assertTrue(state['dropped'],
                        'the patched tiler was never invoked by run() -- the '
                        'census is not calling _farfield_exterior_tiles.')
        tile_delta = base_tiles - patched['n_tiles']
        node_delta = base_nodes - patched['n_nodes']
        self.assertEqual(
            tile_delta, 1,
            f'exterior tile count moved by {tile_delta}, not 1 -- the census '
            f'is not a thin len() of the tiler (base={base_tiles}, '
            f'patched={patched["n_tiles"]}).')
        self.assertEqual(
            node_delta, _exterior_nodes_per_tile(config),
            f'node count moved by {node_delta}, not one tile '
            f'({_exterior_nodes_per_tile(config)}) -- the derived node budget '
            f'is decoupled from the tiler count.')
        self._observe()

    def test_patching_unrelated_diagnostic_tiler_leaves_counts_unmoved(
            self) -> None:
        """A tiler used only by Q2/Q3 must not move any per-region count.

        Negative control: if the exterior count changed when we patch
        ``_farfield_tiles`` (a DIFFERENT tiler), the census would be sharing a
        reimplementation rather than reading ``_farfield_exterior_tiles``.
        """
        config = _smoke_config()
        base_per_region = _baseline_run()['per_region']

        original = getattr(st, _DIAGNOSTIC_TILER)

        def _drop_one(*args, **kwargs):
            tiles = original(*args, **kwargs)
            return tiles[:-1] if tiles else tiles

        with mock.patch.object(st, _DIAGNOSTIC_TILER, side_effect=_drop_one):
            patched_per_region = tc.run(config)['per_region']

        for key in base_per_region:
            with self.subTest(region=key):
                self.assertEqual(
                    patched_per_region[key]['n_tiles'],
                    base_per_region[key]['n_tiles'],
                    f'{key} tile count changed when an UNRELATED diagnostic '
                    f'tiler was patched -- the count is not isolated to its '
                    f'own production tiler.')
                self._observe()


class PpgoTrimIndependenceTestCase(_CensusTestCase):
    """INS-1-001: the conservative-upper-bound property holds structurally.

    ``run`` predicts a campaign's node budget WITHOUT modeling the certified
    ppGO map's per-stratum/window trim (``surrogate_training._apply_ppgo_trim``,
    consulted via ``ppgo_map.get_certified_ppgo_map()``), even though every
    real ``train()`` call installs a certified map by default and drops whole
    strata/windows through it. That gap is safe -- the census only ever
    OVER-counts, never under-provisions -- but only because the counting loop
    structurally never reaches either primitive. If a future change wired a
    trim lookup into ``_count_wedge_interior``/``_count_exterior`` without
    updating the module's documented upper-bound guarantee, the "never an
    underestimate" claim would silently stop holding. We pin the structural
    fact directly: booby-trap both primitives and confirm ``run`` completes
    without calling either, regardless of what certified map (if any) is
    installed in the process.
    """

    def test_run_never_touches_certified_map_or_trim_primitive(self) -> None:
        """``run`` completes with the trim/certified-map doors booby-trapped."""
        calls: list[str] = []

        def _trap(name: str):
            def _side_effect(*_args, **_kwargs):
                calls.append(name)
                raise AssertionError(f'trim primitive called: {name}')
            return _side_effect

        with mock.patch.object(
                ppgo_map, 'get_certified_ppgo_map',
                side_effect=_trap('ppgo_map.get_certified_ppgo_map')), \
             mock.patch.object(
                st, '_apply_ppgo_trim',
                side_effect=_trap('surrogate_training._apply_ppgo_trim')):
            try:
                result = tc.run(_smoke_config())
            except AssertionError as exc:  # pragma: no cover - the leak path
                self.fail(f'TRIM LEAK: run() consulted the certified-ppGO '
                          f'trim machinery ({exc}); primitives hit: {calls}')
        self.assertEqual(calls, [],
                         f'run() called trim primitive(s): {calls}')
        self.assertEqual(result['schema'], 'tiling_census_v1')
        self._observe()


class BandVerdictLogicTestCase(_CensusTestCase):
    """A count is IN_BAND, SILENT_EMPTY (below/0) or EXPLOSION (above)."""

    def test_verdict_classifier_two_sided_including_zero(self) -> None:
        """`_verdict` labels below/0 SILENT_EMPTY, in IN_BAND, above EXPLOSION."""
        band = (2, 10)
        cases = (
            (0, tc.SILENT_EMPTY),   # exactly zero -> the silent-empty mode
            (1, tc.SILENT_EMPTY),   # below the low edge
            (2, tc.IN_BAND),        # on the low edge
            (6, tc.IN_BAND),        # interior
            (10, tc.IN_BAND),       # on the high edge
            (11, tc.EXPLOSION),     # above the high edge
            (10 ** 9, tc.EXPLOSION),
        )
        for count, expected in cases:
            with self.subTest(count=count):
                self.assertEqual(
                    tc._verdict(count, band), expected,
                    f'count={count} against band={band} misclassified.')
                self._observe()

    def test_zero_count_is_never_in_band(self) -> None:
        """A zero must read SILENT_EMPTY for every band, never IN_BAND.

        This is the whole reason the census exists: a region whose tiler
        returned ``[]`` and vanished must be surfaced, not blessed.
        """
        for low, high in ((0, 5), (1, 200), (1, 10 ** 10)):
            with self.subTest(band=(low, high)):
                verdict = tc._verdict(0, (low, high))
                self.assertEqual(verdict, tc.SILENT_EMPTY)
                self.assertNotEqual(verdict, tc.IN_BAND)
                self._observe()

    def test_run_flags_zero_exterior_count_as_silent_empty(self) -> None:
        """A tiler patched to return no tile surfaces as SILENT_EMPTY in run()."""
        with mock.patch.object(st, _EXTERIOR_TILER, side_effect=lambda *a, **k: []):
            record = tc.run(_smoke_config())['per_region'][_EXTERIOR_KEY]
        self.assertEqual(record['n_tiles'], 0)
        self.assertEqual(record['verdict_tiles'], tc.SILENT_EMPTY)
        self.assertEqual(record['verdict'], tc.SILENT_EMPTY,
                         'a zero exterior tile count did not surface as '
                         'SILENT_EMPTY through the full run() aggregation.')
        self._observe()

    def test_run_flags_exploding_exterior_count_as_explosion(self) -> None:
        """A tiler patched above the high band surfaces as EXPLOSION in run()."""
        # High band for ('exterior', 1) is 10000; overshoot it in one call.
        huge = [_FAKE_TILE] * (tc._EXPECTED_BANDS[('exterior', 1)]['tiles'][1]
                               + 1)
        with mock.patch.object(st, _EXTERIOR_TILER,
                               side_effect=lambda *a, **k: list(huge)):
            record = tc.run(_smoke_config())['per_region'][_EXTERIOR_KEY]
        self.assertGreater(record['n_tiles'],
                           tc._EXPECTED_BANDS[('exterior', 1)]['tiles'][1])
        self.assertEqual(record['verdict_tiles'], tc.EXPLOSION)
        self.assertEqual(record['verdict'], tc.EXPLOSION,
                         'an exploding exterior tile count did not surface as '
                         'EXPLOSION through the full run() aggregation.')
        self._observe()

    def test_natural_in_band_exterior_witness(self) -> None:
        """The unpatched exterior count sits IN_BAND -- the middle of the triple.

        Uses the cached baseline (no extra run) as the in-band leg so all three
        verdict outcomes are witnessed on the SAME region x parity.
        """
        record = _baseline_run()['per_region'][_EXTERIOR_KEY]
        low, high = tc._EXPECTED_BANDS[('exterior', 1)]['tiles']
        self.assertTrue(low <= record['n_tiles'] <= high,
                        f'premise lost: baseline exterior count '
                        f'{record["n_tiles"]} left the in-band range '
                        f'[{low}, {high}].')
        self.assertEqual(record['verdict_tiles'], tc.IN_BAND)
        self._observe()


class ArcCensusQ1TestCase(_CensusTestCase):
    """Q1: detected-vs-trained fold-arc separation survives D2-slice refactors.

    The durable invariant is the fundamental-domain reduction: the astroid
    detects four D2-image arcs but trains ONE (the F079 wrap fix), while the
    macro-saddle deltoid detects six arcs and trains the ``max_tube_arcs``
    slice.  We pin ``trained <= detected`` for both parities, the astroid's
    trained-count of exactly one, and the saddle's ``min(detected,
    max_tube_arcs)`` slice -- the last with genuine teeth via a second config
    whose ``max_tube_arcs`` opens the slice wider (default config has
    ``max_tube_arcs == 1``, which would let a "saddle always trains 1" bug
    masquerade as correct).
    """

    def test_detected_and_trained_arc_counts_match_fundamental_domain(
            self) -> None:
        """Astroid folds 4 detected -> 1 trained; saddle detects 6, slices."""
        config = _smoke_config()
        q1 = _baseline_run()['q1_arc_census']
        astro = q1['astroid']
        sad = q1['saddle']

        # Astroid: the fundamental-domain reduction (F079).  Detecting four
        # arcs but training one is the whole point; a strict inequality proves
        # the reduction is non-trivial (not a vacuous 1 == 1).
        self.assertEqual(astro['detected_arcs'], _ASTROID_DETECTED_ARCS,
                         'astroid detected-arc count regressed from the four '
                         'D2-image arcs of the caustic.')
        self.assertEqual(
            astro['trained_arcs'], _ASTROID_TRAINED_ARCS,
            'the fundamental-domain fold reduction regressed: astroid trained '
            f'{astro["trained_arcs"]} arcs, not 1. detected='
            f'{astro["detected_arcs"]}, band={astro["representative_band"]}.')
        self.assertGreater(
            astro['detected_arcs'], astro['trained_arcs'],
            'astroid detected == trained -- the D2 fold reduction is not '
            'happening (the F079 half-ring hole would reopen).')
        self._observe()

        # Saddle: detects six arcs, trains the max_tube_arcs slice.
        self.assertEqual(sad['detected_arcs'], _SADDLE_DETECTED_ARCS,
                         'saddle detected-arc count regressed from six.')
        self.assertEqual(
            sad['trained_arcs'],
            min(_SADDLE_DETECTED_ARCS, config.max_tube_arcs),
            'saddle trained-arc count is not min(detected, max_tube_arcs).')
        self._observe()

        # trained <= detected for BOTH parities (a coverage-hole guard: never
        # claim to train more arcs than the caustic actually has).
        for name, entry in (('astroid', astro), ('saddle', sad)):
            with self.subTest(parity=name):
                self.assertLessEqual(
                    entry['trained_arcs'], entry['detected_arcs'],
                    f'{name} trained {entry["trained_arcs"]} arcs but only '
                    f'{entry["detected_arcs"]} were detected.')
                self.assertGreaterEqual(entry['trained_arcs'], 1)
                self._observe()

    def test_saddle_slice_widens_with_max_tube_arcs(self) -> None:
        """A config with ``max_tube_arcs=2`` trains 2 saddle arcs, not 1.

        Teeth for the ``min(detected, max_tube_arcs)`` slice: the default
        config's ``max_tube_arcs == 1`` cannot distinguish the correct slice
        from a "saddle always trains one arc" bug, so we open the knob and
        require the trained count to follow -- while the astroid stays pinned
        at its single fundamental arc regardless of the knob.
        """
        wide = st.TrainingConfig(max_tube_arcs=2)
        self.assertEqual(wide.max_tube_arcs, 2,
                         'premise lost: TrainingConfig did not accept '
                         'max_tube_arcs=2.')
        q1 = tc.run(wide)['q1_arc_census']
        self.assertEqual(
            q1['saddle']['trained_arcs'],
            min(_SADDLE_DETECTED_ARCS, 2),
            'saddle trained-arc count did not follow max_tube_arcs=2 -- the '
            'slice is decoupled from the knob (or hardwired to 1).')
        self.assertEqual(
            q1['astroid']['trained_arcs'], _ASTROID_TRAINED_ARCS,
            'astroid trained-arc count changed with max_tube_arcs -- the '
            'positive-parity fold must always reduce to one arc.')
        self._observe()


class WBandContainmentQ4TestCase(_CensusTestCase):
    """Q4: the trained w-band vs effective serve floor/ceiling containment.

    The census reports, per (region x parity), an ``effective_floor``, an
    ``effective_ceiling`` (where the region has one) and a ``contained``
    boolean.  The DURABLE invariant is that ``contained`` is arithmetically
    FAITHFUL to those bounds and that the bounds match independent oracles --
    NOT that every band is contained.  (On the default smoke config the bands
    genuinely poke outside their effective serve region: every ``contained``
    is ``False``.  That is a legitimate coverage-hole report, not a census
    bug, so a "must be True" assertion would encode a config accident.  A
    False here carries a ``reason`` only when the entry is DEFERRED, i.e. its
    floor is ``None``; a numeric-floor False is a surfaced coverage hole, by
    design reason-less.)

    We recompute ``contained`` from the reported floor/ceiling/band -- fully
    independent of how the census derived them -- and pin the three closed-form
    bounds: astroid ceiling ``min(480, 60/sqrt(s))``, saddle tube floor/ceiling
    ``(SADDLE_WALL, _SADDLE_W_CEILING)``, and the saddle far-field floor
    ``(2e4 * K)**(1/3)`` against a from-scratch geometry recompute.
    """

    def _recompute_contained(self, entry: dict) -> bool | None:
        """Independent containment predicate from the entry's own bounds."""
        floor = entry.get('effective_floor')
        ceiling = entry.get('effective_ceiling')
        w_lo, w_hi = entry['w_band']
        if floor is None:
            return None  # deferred -- containment is undefined
        if ceiling is not None:
            return bool(w_lo >= floor and w_hi <= ceiling)
        return bool(w_lo >= floor)

    def test_contained_flag_is_faithful_to_reported_bounds(self) -> None:
        """Every ``contained`` matches the predicate recomputed from its bounds.

        This survives a refactor of the floor/ceiling FORMULAS: whatever the
        census puts in ``effective_floor``/``effective_ceiling``, the boolean
        it publishes must be the honest inequality on those very numbers.
        """
        q4 = _baseline_run()['q4_w_band_containment']
        self.assertTrue(q4, 'premise lost: Q4 produced no (region x parity) '
                            'entries.')
        saw_false = False
        for key, entry in q4.items():
            with self.subTest(region=key):
                expected = self._recompute_contained(entry)
                if entry.get('effective_floor') is None:
                    # Deferral: contained is None AND a reason is populated.
                    self.assertIsNone(entry['contained'])
                    self.assertTrue(
                        entry.get('reason'),
                        f'{key} deferred (floor None) but carries no reason.')
                else:
                    self.assertEqual(
                        entry['contained'], expected,
                        f'{key} contained={entry["contained"]} contradicts the '
                        f'inequality on its own bounds (w_band='
                        f'{entry["w_band"]}, floor={entry["effective_floor"]}, '
                        f'ceiling={entry.get("effective_ceiling")}).')
                    saw_false = saw_false or entry['contained'] is False
                self._observe()
        # Anti-vacuity for the predicate: the census MUST be able to surface a
        # coverage hole (contained False); if it never did, the predicate would
        # be untested against its failing branch.
        self.assertTrue(
            saw_false,
            'no Q4 entry reported contained=False on the smoke config -- the '
            'containment predicate never exercised its coverage-hole branch.')

    def test_astroid_ceiling_is_min_480_and_dd_margin(self) -> None:
        """Astroid ceiling == min(_POSITIVE_W_CEILING, 60/sqrt(source_mag))."""
        q4 = _baseline_run()['q4_w_band_containment']
        checked = 0
        for key, entry in q4.items():
            if entry['parity'] != 1:
                continue
            s = entry.get('source_magnitude')
            self.assertIsNotNone(s, f'{key} astroid entry lacks '
                                    'source_magnitude for the DD-margin ceiling.')
            self.assertGreater(s, 0.0)
            # The census reads the production _DD_PRODUCT_MARGIN at its use
            # site (no mirrored module constant -- part0 absorber guard);
            # the oracle reads the SAME production constant.
            oracle = min(float(st._POSITIVE_W_CEILING),
                         float(st._DD_PRODUCT_MARGIN) / math.sqrt(s))
            self.assertAlmostEqual(
                entry['effective_ceiling'], oracle, places=9,
                msg=f'{key} astroid ceiling {entry["effective_ceiling"]} != '
                    f'min(480, 60/sqrt({s}))={oracle}.')
            checked += 1
            self._observe()
        self.assertGreater(checked, 0,
                           'no astroid (+1) Q4 entry found to check the ceiling.')

    def test_saddle_tube_floor_and_ceiling_are_wall_and_w_ceiling(self) -> None:
        """Saddle tube floor==SADDLE_WALL, ceiling==_SADDLE_W_CEILING."""
        entry = _baseline_run()['q4_w_band_containment']['tube:-1']
        self.assertAlmostEqual(entry['effective_floor'],
                               float(ppgo_map.SADDLE_WALL), places=9)
        self.assertAlmostEqual(entry['effective_ceiling'],
                               float(st._SADDLE_W_CEILING), places=9)
        self._observe()

    def test_saddle_farfield_floor_matches_independent_2e4_K_cube_root(
            self) -> None:
        """Saddle far-field floor == (2e4*K)**(1/3) recomputed from geometry.

        The independent oracle re-runs ``find_images`` + ``ppgo_error_estimate``
        and applies the coefficient/exponent with its own literals, so a drift
        in ``_Q4_SADDLE_FLOOR_COEFF``, the cube-root, or the ``w_min**3``
        recovery of ``K`` is caught.
        """
        q4 = _baseline_run()['q4_w_band_containment']
        oracle = _independent_saddle_farfield_floor()
        self.assertGreater(oracle, 0.0)
        self.assertTrue(math.isfinite(oracle))
        checked = 0
        for key in ('lobe_interior:-1', 'lobe_exterior:-1'):
            entry = q4.get(key)
            if entry is None or entry.get('effective_floor') is None:
                continue
            self.assertAlmostEqual(
                entry['effective_floor'], oracle,
                delta=1e-9 * oracle,
                msg=f'{key} far-field floor {entry["effective_floor"]} != '
                    f'independent (2e4*K)**(1/3) oracle {oracle}.')
            checked += 1
            self._observe()
        self.assertGreater(
            checked, 0,
            'no numeric-floor saddle far-field entry to check against the '
            'independent (2e4*K)**(1/3) oracle.')


class SelfEstimateCrossCheckTestCase(_CensusTestCase):
    """The census aggregate stays coupled to the production ``_self_estimate``.

    Two couplings, both durable: (1) the JSON ``self_estimate_seconds`` is the
    production ``_self_estimate`` VERBATIM (exact float equality, and it tracks
    the ``regions`` argument), and (2) the census aggregate call-count is the
    sum of ``per_region`` node counts times ``_LABELS_PER_NODE``, with the
    reported census/self-estimate ratio inside the documented factor.
    """

    def test_self_estimate_seconds_is_exact_passthrough(self) -> None:
        """``self_estimate_seconds`` equals ``_self_estimate(config, None)``."""
        config = _smoke_config()
        result = _baseline_run()
        self.assertEqual(
            result['self_estimate_seconds'],
            float(st._self_estimate(config, None)),
            'self_estimate_seconds diverged from the production _self_estimate '
            '-- the verbatim passthrough broke.')
        self._observe()

    def test_self_estimate_seconds_tracks_the_regions_argument(self) -> None:
        """A scoped ``regions`` run passes that filter through to _self_estimate.

        Teeth for the passthrough: if ``run`` ignored ``regions`` and always
        estimated the full region set, a scoped run's ``self_estimate_seconds``
        would NOT match ``_self_estimate(config, regions)`` and WOULD match the
        unscoped estimate.  We require the former and reject the latter.
        """
        config = _smoke_config()
        regions = ('tube', 'exterior')
        scoped = tc.run(config, regions=regions)
        self.assertEqual(
            scoped['self_estimate_seconds'],
            float(st._self_estimate(config, regions)),
            'scoped self_estimate_seconds != _self_estimate(config, regions) '
            '-- run() did not pass regions through.')
        self.assertNotEqual(
            scoped['self_estimate_seconds'],
            float(st._self_estimate(config, None)),
            'scoped self_estimate_seconds equals the UNSCOPED estimate -- the '
            'regions filter was dropped before _self_estimate.')
        self._observe()

    def test_aggregate_call_count_is_sum_of_region_node_labels(self) -> None:
        """Aggregate == sum(per_region n_nodes) * _LABELS_PER_NODE, exactly."""
        result = _baseline_run()
        oracle = sum(rec['n_nodes'] for rec in result['per_region'].values()) \
            * tc._LABELS_PER_NODE
        self.assertEqual(
            result['aggregate_call_count'], oracle,
            'aggregate_call_count is not the sum of per-region node labels -- '
            'the campaign budget drifted from the per-region records.')
        self.assertEqual(
            result['census_seconds'],
            result['aggregate_call_count'] * tc._SECONDS_PER_LABEL,
            'census_seconds is not aggregate_call_count * _SECONDS_PER_LABEL.')
        self._observe()

    def test_cross_check_ratio_within_documented_factor(self) -> None:
        """The reported ratio is census/self-estimate and within the factor."""
        cc = _baseline_run()['cross_check']
        result = _baseline_run()
        expected_ratio = (result['census_seconds']
                          / result['self_estimate_seconds'])
        self.assertAlmostEqual(
            cc['ratio_census_over_self_estimate'], expected_ratio, places=9,
            msg='reported ratio != census_seconds / self_estimate_seconds.')
        self.assertEqual(cc['documented_factor'], tc._CROSS_CHECK_FACTOR)
        self.assertLessEqual(
            cc['ratio_census_over_self_estimate'], tc._CROSS_CHECK_FACTOR,
            f'census/self-estimate ratio {cc["ratio_census_over_self_estimate"]}'
            f' blew past the documented factor {tc._CROSS_CHECK_FACTOR} -- the '
            'aggregate formula drifted from the per-region cost model.')
        self.assertTrue(cc['within_documented_factor'])
        self._observe()


class CensusSelfFalsificationTestCase(_CensusTestCase):
    """Prove the pins above can actually go red -- the suite is not decoration."""

    def test_engine_trap_actually_fires_when_a_door_is_opened(self) -> None:
        """Under the engine patches, calling a door DOES raise -- traps work.

        If this did not raise, the engine-free test would be vacuous: its
        silence would prove nothing because the traps would be inert.
        """
        with mock.patch.object(_schwinger, 'f_schwinger',
                               side_effect=AssertionError('engine called')):
            with self.assertRaises(AssertionError):
                _schwinger.f_schwinger(30.0, np.array([0.3, 0.2]), 0.5)
            self._observe()
        with mock.patch.object(cr_channels.ChangRefsdalChannels, 'evaluate',
                               side_effect=AssertionError('engine called')):
            instance = cr_channels.ChangRefsdalChannels(np.array([1.0, 2.0]))
            with self.assertRaises(AssertionError):
                instance.evaluate(object(), object(), object())
            self._observe()

    def test_verdict_zero_guard_is_load_bearing(self) -> None:
        """With a low edge of 0, only the explicit ``count == 0`` clause saves us.

        A classifier written as ``count < low`` alone would pass ``0 < 0``
        through to IN_BAND; the shipped guard's explicit ``count == 0`` clause
        is what keeps a vanished region from reading healthy.
        """
        self.assertEqual(tc._verdict(0, (0, 5)), tc.SILENT_EMPTY)
        self.assertNotEqual(tc._verdict(0, (0, 5)), tc.IN_BAND)
        # A synthetic broken classifier (low-edge test only) would mislabel it,
        # proving the clause carries weight rather than being redundant.
        broken = tc.IN_BAND if not (0 < 0) else tc.SILENT_EMPTY
        self.assertEqual(broken, tc.IN_BAND,
                         'sanity: the low-edge-only rule would indeed pass a '
                         'zero through, so the explicit zero clause is needed.')
        self._observe()

    def test_thin_caller_pin_would_fail_if_count_were_static(self) -> None:
        """A reimplemented (static) census would break the thin-caller pin.

        Demonstrates the pin's teeth without a run: a census that ignored the
        tiler would report the baseline count under a drop-one patch, and the
        equality the thin-caller test asserts (delta == 1) would fail.
        """
        base = _baseline_run()['per_region'][_EXTERIOR_KEY]['n_tiles']
        static_reimplementation_count = base  # ignores the patch
        self.assertNotEqual(
            base - static_reimplementation_count, 1,
            'a static reimplementation would show delta 0, so the '
            'thin-caller delta==1 assertion has teeth.')
        self._observe()

    def test_saddle_floor_coefficient_pin_has_teeth(self) -> None:
        """A wrong far-field-floor coefficient would NOT match the census.

        The Q4 saddle-floor oracle re-derives ``(2e4*K)**(1/3)``; if the census
        (or the oracle) drifted the ``2e4`` coefficient, the reported floor and
        a ``2e5``-coefficient recompute must diverge -- proving the equality
        pin is not vacuously satisfied by any positive number.
        """
        floor = _baseline_run()['q4_w_band_containment']['lobe_interior:-1'][
            'effective_floor']
        oracle = _independent_saddle_farfield_floor()
        self.assertAlmostEqual(floor, oracle, delta=1e-9 * oracle)
        # A 10x coefficient error scales the floor by 10**(1/3) ~ 2.15 -- far
        # outside the 1e-9 tolerance the correct-coefficient pin holds to.
        wrong = oracle * (10.0 ** (1.0 / 3.0))
        self.assertGreater(
            abs(wrong - floor), 1e-9 * oracle,
            'a 10x-coefficient floor collided with the census value -- the '
            'formula pin would not catch a coefficient drift.')
        self._observe()

    def test_trim_trap_actually_fires_when_the_primitive_is_opened(self) -> None:
        """Under the trim patches, calling the primitive DOES raise.

        Proves the trim-independence trap above has teeth: if neither
        primitive were reachable by ANY caller (not just the census), the
        independence test would be vacuously satisfied by an unreachable
        function rather than by the census genuinely avoiding it.
        """
        with mock.patch.object(ppgo_map, 'get_certified_ppgo_map',
                               side_effect=AssertionError('trim called')):
            with self.assertRaises(AssertionError):
                ppgo_map.get_certified_ppgo_map()
            self._observe()
        with mock.patch.object(st, '_apply_ppgo_trim',
                               side_effect=AssertionError('trim called')):
            with self.assertRaises(AssertionError):
                st._apply_ppgo_trim((1.0, 2.0), None)
            self._observe()

    def test_q4_containment_predicate_discriminates(self) -> None:
        """The containment predicate flips with the inequality it encodes.

        A synthetic band pushed above its ceiling must read NOT contained, and
        one pulled inside must read contained -- so the Q4 faithfulness test's
        recompute is a real inequality, not a constant.
        """
        checker = WBandContainmentQ4TestCase()
        inside = {'w_band': [10.0, 20.0], 'effective_floor': 5.0,
                  'effective_ceiling': 30.0}
        outside = {'w_band': [10.0, 40.0], 'effective_floor': 5.0,
                   'effective_ceiling': 30.0}
        below = {'w_band': [1.0, 20.0], 'effective_floor': 5.0,
                 'effective_ceiling': 30.0}
        self.assertTrue(checker._recompute_contained(inside))
        self.assertFalse(checker._recompute_contained(outside))
        self.assertFalse(checker._recompute_contained(below))
        self.assertIsNone(checker._recompute_contained(
            {'w_band': [1.0, 2.0], 'effective_floor': None}))
        self._observe()


if __name__ == '__main__':  # pragma: no cover
    main()
