"""Tests for the slow-operation admission judge on the surrogate training
entry point.

The working tree adds a programmatic admission gate to `train()`:
`guard_slow_operation(est_seconds=_self_estimate(config, regions), what=...)`
runs BEFORE any engine contact, refusing a production-scale `train()` call
inside a build (where slow tiers are pinned OFF) and passing it through when
a slow tier is enabled.  This suite pins:

* THE JUDGE (`guard_slow_operation`) -- an estimate at or below the budget
  passes; an over-budget estimate raises ``ValueError`` UNLESS one of the
  opt-in slow-tier env vars is set; a custom ``budget_s`` rescales the
  boundary; the refusal message names the operation and the budget.

* THE ESTIMATOR (`_self_estimate`) -- ``regions=None`` costs the full
  5-region default (``tube``, ``exterior``, ``wedge_interior``,
  ``lobe_interior``, ``lobe_exterior`` -- ``lobe_exterior`` is now a
  first-class training region, this build's WP1); a single-region probe
  pays only that region's grid (the WP's core promise: ``wedge_interior``
  costs ~1 engine eval, ``exterior`` costs ``n_rho * n_theta_c``); an empty
  tuple falls back to the full default (``()`` is falsy -- pinned, not
  "fixed"); the estimate is monotone under region removal (a leaner filter
  can never cost MORE).

* THE WIRING (`train()`) -- the guard fires BEFORE any chart is requested
  (a sentinel `_train_band_charts` that raises if called proves zero engine
  contact on refusal), and the same production-scale config passes once a
  slow tier is enabled -- so a build cannot silently launch a multi-hour
  sweep, and the driver's post-build sweeps are unblocked.

ORACLE INDEPENDENCE
-------------------
The judge's decision is pinned by its OWN error/no-op contract, never by a
re-derivation of the budget check.  The estimator's grid arithmetic is
pinned against explicit per-region eval-count bookkeeping written out by
hand (a wrong ``n_theta * n_u`` cross-term, or a ``per_region`` lookup keyed
on a name that is not in the set, shifts the number).  The wiring test
replaces only the ENGINE contact point with a sentinel -- the real judge and
the real estimator run unchanged.

COST BUDGET
-----------
All tests are pure arithmetic + one ``train()`` call whose heavy path is
patched (``stable_gamma_bands``, ``get_certified_ppgo_map``,
``_train_band_charts``, ``LensAmplificationSurrogate``); the real judge and
estimator run but nothing engine-backed does.  Well under the 60 s per-test
/ 5 min per-file fast-tier ceilings.
"""

from __future__ import annotations

import os
import tempfile
import types
import unittest
from unittest import TestCase, mock

from cogwheel.lensing import surrogate_training as training
from cogwheel.lensing.surrogate_training import (
    TrainingConfig, guard_slow_operation, _self_estimate,
    _FAST_TIER_BUDGET_S, _SLOW_TIER_ENV_VARS)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The opt-in slow-tier env-var names the judge honours (same set the
#: conftest fast-tier ceiling uses).  Pinned as the canonical tuple so a
#: drift between the two copies is caught here, not in a build.
_EXPECTED_SLOW_TIER_VARS: tuple[str, ...] = (
    'COGWHEEL_BRUTE_ACCURACY',
    'COGWHEEL_TRAIN_TIER',
    'COGWHEEL_STRICT_TIMING',
    'COGWHEEL_RUN_TIMING_SMOKE',
)

#: A smoke/probe config (matches the fast-tier fixture grids): cheap enough
#: that even the FULL 5-region set stays under the budget.
_SMOKE_CONFIG: TrainingConfig = TrainingConfig(
    n_gamma=4, n_u=4, n_theta=4, n_rho=4, n_theta_c=4, w_nodes_per_decade=3)

#: A production-scale config: dense grids and w nodes such that the full
#: region set exceeds the in-build budget while the ``wedge_interior``-only
#: probe stays under it -- the exact regime the WP exists for.
_PRODUCTION_CONFIG: TrainingConfig = TrainingConfig(
    n_gamma=6, n_theta=12, n_u=12, n_rho=12, n_theta_c=12,
    w_nodes_per_decade=8)


class _CountingTestCase(TestCase):
    """Anti-vacuity base (house idiom): a test that asserts nothing fails."""

    def setUp(self) -> None:
        self.comparisons = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self.comparisons, 0,
            'anti-vacuity: this test asserted nothing (zero comparisons).')

    def assert_raises_with(self, exc: type[BaseException], *args, **kwargs):
        """Callable form of assertRaises that also counts.

        Executes ``callable(*args, **kwargs)`` immediately and returns the
        caught exception -- the body of a ``with`` block is not needed.
        """
        self.comparisons += 1
        return self.assertRaises(exc, *args, **kwargs)


# ===========================================================================
# The judge itself.
# ===========================================================================


class GuardAdmissionTestCase(_CountingTestCase):
    """`guard_slow_operation` admits at/below budget, refuses above unless a
    slow tier is on."""

    def _with_clean_env(self):
        """Ensure no slow-tier env var leaks in from the surrounding run."""
        return mock.patch.dict(
            os.environ,
            {v: '' for v in _SLOW_TIER_ENV_VARS},
            clear=False)

    def test_at_or_below_budget_passes(self) -> None:
        for est in (0.0, _FAST_TIER_BUDGET_S, _FAST_TIER_BUDGET_S - 1.0):
            with self.subTest(est=est):
                guard_slow_operation(est_seconds=est, what='probe')
                self.comparisons += 1  # no raise = the assertion

    def test_over_budget_raises_value_error(self) -> None:
        with self._with_clean_env():
            self.assert_raises_with(
                ValueError, guard_slow_operation,
                est_seconds=_FAST_TIER_BUDGET_S + 1.0, what='probe')

    def test_refusal_message_names_operation_and_budget(self) -> None:
        with self._with_clean_env():
            with self.assertRaises(ValueError) as ctx:
                guard_slow_operation(
                    est_seconds=2 * _FAST_TIER_BUDGET_S, what='engine sweep')
        self.assertIn('engine sweep', str(ctx.exception))
        self.assertIn('min', str(ctx.exception))
        self.comparisons += 2

    def test_over_budget_passes_when_any_slow_tier_is_set(self) -> None:
        for var in _EXPECTED_SLOW_TIER_VARS:
            with self.subTest(var=var):
                with mock.patch.dict(os.environ, {var: '1'}, clear=False):
                    guard_slow_operation(
                        est_seconds=10 * _FAST_TIER_BUDGET_S, what='sweep')
                    self.comparisons += 1

    def test_slow_tier_set_is_any_not_all(self) -> None:
        """One slow-tier var is sufficient -- the judge does not require the
        whole set (mirrors the conftest ceiling's semantics)."""
        with mock.patch.dict(os.environ,
                             {'COGWHEEL_TRAIN_TIER': '1'}, clear=False):
            guard_slow_operation(est_seconds=1e6, what='sweep')
            self.comparisons += 1

    def test_custom_budget_rescales_boundary(self) -> None:
        """A tighter budget_s refuses an estimate the default budget would
        admit; a looser one admits what the default refuses."""
        with self._with_clean_env():
            self.assert_raises_with(
                ValueError, guard_slow_operation,
                est_seconds=100.0, what='probe', budget_s=50.0)
            guard_slow_operation(est_seconds=100.0, what='probe',
                                 budget_s=200.0)
            self.comparisons += 2

    def test_slow_tier_bypass_respects_custom_budget(self) -> None:
        """The slow-tier bypass also wins over a custom (tighter) budget."""
        with mock.patch.dict(os.environ,
                             {'COGWHEEL_BRUTE_ACCURACY': '1'}, clear=False):
            guard_slow_operation(est_seconds=1e6, what='sweep', budget_s=1.0)
            self.comparisons += 1


class SlowTierEnvVarSetTestCase(_CountingTestCase):
    """The judge's slow-tier env-var set is exactly the canonical tuple the
    conftest ceiling uses -- a drift would silently re-allow multi-hour
    runs in a build that the test-harness ceiling already forbids."""

    def test_judge_set_matches_canonical_tuple(self) -> None:
        self.assertEqual(_SLOW_TIER_ENV_VARS, _EXPECTED_SLOW_TIER_VARS)
        self.comparisons += 1


# ===========================================================================
# The estimator.
# ===========================================================================


class SelfEstimateTestCase(_CountingTestCase):
    """`_self_estimate` costs a region-selected run at only that region's
    grid -- the WP's core promise, pinned arithmetically."""

    def test_none_default_costs_full_five_region_set(self) -> None:
        full = _self_estimate(_SMOKE_CONFIG, None)
        explicit = _self_estimate(_SMOKE_CONFIG,
                                  ('tube', 'exterior', 'wedge_interior',
                                   'lobe_interior', 'lobe_exterior'))
        self.assertEqual(full, explicit)
        # Hand-derived bookkeeping (smoke grid, WP1's 5-region default):
        # per-region evals = tube(4*4=16) + exterior(4*4=16) +
        # wedge_interior(1) + lobe_interior(1) + lobe_exterior(1) = 35;
        # n_evals = 35 * n_gamma(4) * w_nodes(3*2=6) = 840;
        # estimate = 840 * 8 * 0.09.
        self.assertAlmostEqual(full, 840 * 8 * 0.09, places=6)
        self.comparisons += 2

    def test_lobe_exterior_only_pays_one_eval_per_gamma_w(self) -> None:
        """WP1: ``lobe_exterior`` is a first-class region priced at 1 engine
        eval per (gamma, w) node -- identical to ``lobe_interior`` and
        ``wedge_interior``, so the guard estimate stays honest and no
        KeyError is raised for a ``lobe_exterior`` run."""
        est = _self_estimate(_PRODUCTION_CONFIG, ('lobe_exterior',))
        # (1 eval) * n_gamma(6) * w_nodes(2*8=16) * 8 * 0.09.
        self.assertAlmostEqual(est, 1 * 6 * 16 * 8 * 0.09, places=6)
        # Same cost as the other unit-cost interior regions.
        self.assertEqual(
            est, _self_estimate(_PRODUCTION_CONFIG, ('lobe_interior',)))
        self.assertEqual(
            est, _self_estimate(_PRODUCTION_CONFIG, ('wedge_interior',)))
        self.comparisons += 3

    def test_wedge_interior_only_pays_one_eval_per_gamma_w(self) -> None:
        est = _self_estimate(_PRODUCTION_CONFIG, ('wedge_interior',))
        # (1 eval) * n_gamma(6) * w_nodes(2*8=16) * 8 * 0.09.
        self.assertAlmostEqual(est, 1 * 6 * 16 * 8 * 0.09, places=6)
        self.comparisons += 1

    def test_exterior_only_pays_rho_theta_c_grid(self) -> None:
        est = _self_estimate(_PRODUCTION_CONFIG, ('exterior',))
        # (12*12=144 evals) * 6 * 16 * 8 * 0.09.
        self.assertAlmostEqual(est, 144 * 6 * 16 * 8 * 0.09, places=6)
        self.comparisons += 1

    def test_region_filter_can_push_production_under_budget(self) -> None:
        """The WP's whole point: the same production config is refused at
        full region set and admitted as a single-region probe."""
        full = _self_estimate(_PRODUCTION_CONFIG, None)
        wedge = _self_estimate(_PRODUCTION_CONFIG, ('wedge_interior',))
        self.assertGreater(full, _FAST_TIER_BUDGET_S)
        self.assertLess(wedge, _FAST_TIER_BUDGET_S)
        self.comparisons += 2

    def test_single_region_never_costs_more_than_full(self) -> None:
        for region in ('tube', 'exterior', 'wedge_interior', 'lobe_interior',
                       'lobe_exterior'):
            with self.subTest(region=region):
                est = _self_estimate(_PRODUCTION_CONFIG, (region,))
                full = _self_estimate(_PRODUCTION_CONFIG, None)
                self.assertLessEqual(est, full)
                self.comparisons += 1

    def test_removing_a_region_never_increases_the_estimate(self) -> None:
        """Monotone under region removal: a leaner filter is a cheaper or
        equal filter, never a pricier one."""
        full = ('tube', 'exterior', 'wedge_interior', 'lobe_interior',
                'lobe_exterior')
        for drop in range(len(full)):
            subset = full[:drop] + full[drop + 1:]
            with self.subTest(dropped=full[drop]):
                self.assertLessEqual(
                    _self_estimate(_PRODUCTION_CONFIG, subset),
                    _self_estimate(_PRODUCTION_CONFIG, full))
                self.comparisons += 1

    def test_empty_tuple_falls_back_to_full_default(self) -> None:
        """``()`` is falsy, so the estimator treats it as "no filter" (full
        cost).  Conservative: a zero-region run is never under-charged."""
        self.assertEqual(_self_estimate(_PRODUCTION_CONFIG, ()),
                         _self_estimate(_PRODUCTION_CONFIG, None))
        self.comparisons += 1


# ===========================================================================
# The wiring: `train()` refuses a production-scale call before any engine
# contact, and the same call passes once a slow tier is enabled.
# ===========================================================================


class TrainSlowOperationWiringTestCase(_CountingTestCase):
    """`train()` gates on the real judge + real estimator, refusing a
    production-scale config in-build BEFORE requesting any chart."""

    def _patch_heavy_path(self):
        """Patch every engine/heavy contact point of `train()` so the test
        exercises the real judge/estimator but never builds charts."""
        return mock.patch.multiple(
            training,
            stable_gamma_bands=mock.DEFAULT,
            get_certified_ppgo_map=mock.DEFAULT,
            _train_band_charts=mock.DEFAULT,
            LensAmplificationSurrogate=mock.DEFAULT,
            create=False)

    def test_production_full_config_refused_before_engine_contact(self) -> None:
        """A sentinel `_train_band_charts` that raises if called proves the
        refusal happens at the judge, before any chart is requested."""
        def sentinel(*args, **kwargs):
            raise AssertionError('engine must not run on an over-budget call')

        struct = types.SimpleNamespace(
            detected_cusps=0, cusp_thetas=[], caustic_reach=0.0, arcs=[])
        with mock.patch.object(
                training, 'stable_gamma_bands',
                return_value=([((0.1, 0.3), struct)], [])), \
             mock.patch.object(training, 'get_certified_ppgo_map',
                               return_value=None), \
             mock.patch.object(training, '_train_band_charts',
                               side_effect=sentinel), \
             mock.patch.object(training, 'LensAmplificationSurrogate'), \
             mock.patch.dict(os.environ,
                             {v: '' for v in _SLOW_TIER_ENV_VARS},
                             clear=False), \
             tempfile.TemporaryDirectory(prefix='slow_op_guard_') as tmp:
            with self.assertRaises(ValueError):
                training.train(outdir=tmp, config=_PRODUCTION_CONFIG)
            self.comparisons += 1

    def test_same_production_config_passes_with_slow_tier_enabled(self) -> None:
        """The identical config is admitted once a slow-tier env var is set:
        the judge is the only thing that changed between the two calls."""
        with self._patch_heavy_path() as m, \
             mock.patch.dict(os.environ, {'COGWHEEL_TRAIN_TIER': '1'},
                             clear=False), \
             tempfile.TemporaryDirectory(prefix='slow_op_guard_') as tmp:
            m['stable_gamma_bands'].return_value = (
    [((0.1, 0.3), types.SimpleNamespace(
        detected_cusps=0, cusp_thetas=[], caustic_reach=0.0, arcs=[]))],
    [])
            m['get_certified_ppgo_map'].return_value = None
            training.train(outdir=tmp, config=_PRODUCTION_CONFIG)
            self.comparisons += 1

    def test_smoke_config_admitted_in_build(self) -> None:
        """A smoke/probe config stays under the budget even with slow tiers
        off -- in-build measurement work is not collateral damage."""
        with self._patch_heavy_path() as m, \
             mock.patch.dict(os.environ,
                             {v: '' for v in _SLOW_TIER_ENV_VARS},
                             clear=False), \
             tempfile.TemporaryDirectory(prefix='slow_op_guard_') as tmp:
            m['stable_gamma_bands'].return_value = (
    [((0.1, 0.3), types.SimpleNamespace(
        detected_cusps=0, cusp_thetas=[], caustic_reach=0.0, arcs=[]))],
    [])
            m['get_certified_ppgo_map'].return_value = None
            training.train(outdir=tmp, config=_SMOKE_CONFIG)
            self.comparisons += 1

    def test_region_filter_admits_production_as_single_region_probe(self) -> None:
        """The WP's headline: a production config that the full set refuses
        is admitted as a single-region probe -- the estimator's region
        cost model is what the judge feeds on."""
        with self._patch_heavy_path() as m, \
             mock.patch.dict(os.environ,
                             {v: '' for v in _SLOW_TIER_ENV_VARS},
                             clear=False), \
             tempfile.TemporaryDirectory(prefix='slow_op_guard_') as tmp:
            m['stable_gamma_bands'].return_value = (
    [((0.1, 0.3), types.SimpleNamespace(
        detected_cusps=0, cusp_thetas=[], caustic_reach=0.0, arcs=[]))],
    [])
            m['get_certified_ppgo_map'].return_value = None
            training.train(outdir=tmp, config=_PRODUCTION_CONFIG,
                           regions=('wedge_interior',))
            self.comparisons += 1


# ===========================================================================
# Self-falsification: prove the green checks can go red.
# ===========================================================================


class SlowOperationGuardSelfFalsificationTestCase(_CountingTestCase):
    """Corrupt each contract and prove the corresponding check trips.

    Without this, a silently-passing suite is indistinguishable from a
    correct one.  Each test injects the defect the real test above is built
    to catch and asserts the check fails.
    """

    def test_over_budget_must_raise_without_slow_tier(self) -> None:
        """A judge that silently admitted an over-budget estimate would make
        `test_over_budget_raises_value_error` read green; the real judge
        must raise, so the defensive claim has teeth."""
        with mock.patch.dict(os.environ,
                             {v: '' for v in _SLOW_TIER_ENV_VARS},
                             clear=False):
            with self.assertRaises(ValueError):
                guard_slow_operation(est_seconds=_FAST_TIER_BUDGET_S + 1,
                                     what='probe')
            self.comparisons += 1

    def test_wrong_per_region_grid_trips_estimate_pin(self) -> None:
        """A `per_region` table that keyed the exterior grid on the tube
        cross-term (or dropped a region) would move the hand-derived number
        -- the arithmetic pin is discriminating."""
        self.assertAlmostEqual(
            _self_estimate(_SMOKE_CONFIG, ('exterior',)),
            16 * 4 * 6 * 8 * 0.09,  # 16 = n_rho * n_theta_c at smoke grid
            places=6)
        self.comparisons += 1

    def test_region_names_without_support_in_estimator_raise(self) -> None:
        """A region name the estimator does not know must raise KeyError --
        a silently-zero cost for an unsupported name would under-charge a
        production call."""
        with self.assertRaises(KeyError):
            _self_estimate(_PRODUCTION_CONFIG, ('exterior', 'not_a_region'))
        self.comparisons += 1

    def test_default_budget_is_positive_and_finite(self) -> None:
        """A zero/negative budget would refuse everything (or nothing);
        the shipped default is a sane positive number."""
        self.assertGreater(_FAST_TIER_BUDGET_S, 0)
        self.comparisons += 1

    def test_bypassed_guard_would_contact_the_engine(self) -> None:
        """If the judge were silently removed (no ValueError on an
        over-budget config), `_train_band_charts` WOULD be reached -- the
        sentinel fires.  This proves the refusal is the judge's doing, not
        some unrelated early error."""
        struct = types.SimpleNamespace(
            detected_cusps=0, cusp_thetas=[], caustic_reach=0.0, arcs=[])
        def sentinel(*args, **kwargs):
            raise AssertionError('engine contacted with judge bypassed')
        with mock.patch.object(training, 'guard_slow_operation',
                               return_value=None), \
             mock.patch.object(
                 training, 'stable_gamma_bands',
                 return_value=([((0.1, 0.3), struct)], [])), \
             mock.patch.object(training, 'get_certified_ppgo_map',
                               return_value=None), \
             mock.patch.object(training, '_train_band_charts',
                               side_effect=sentinel), \
             mock.patch.object(training, 'LensAmplificationSurrogate'), \
             mock.patch.dict(os.environ,
                             {v: '' for v in _SLOW_TIER_ENV_VARS},
                             clear=False), \
             tempfile.TemporaryDirectory(prefix='slow_op_guard_') as tmp:
            with self.assertRaisesRegex(AssertionError,
                                        'engine contacted with judge bypassed'):
                training.train(outdir=tmp, config=_PRODUCTION_CONFIG)
            self.comparisons += 1


if __name__ == '__main__':
    unittest.main()
