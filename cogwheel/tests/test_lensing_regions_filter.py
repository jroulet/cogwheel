"""Tests for the ``regions`` filter on the surrogate training entry points.

WP1 ("Add ``regions`` filter to training entry points") lets a caller train
ONLY a subset of the four chart regions -- ``tube``, ``exterior``,
``wedge_interior``, ``lobe_interior`` -- so a region measurement can invoke
the SHIPPING ``_train_band_charts`` for one region at its own cost instead of
reassembling the pipeline by hand.  This suite pins the filter's contract:

* DEFAULT = ALL (backward compat): ``regions=None`` builds exactly the same
  chart set and emits exactly the same report as the explicit
  ``regions=('tube', 'exterior', 'wedge_interior', 'lobe_interior')`` tuple,
  on both parities.  A future change to the inline default is caught by this
  equality, not silently shipped.

* PER-REGION EXCLUSIVITY: ``regions=(r,)`` invokes ``_load_or_build`` (the
  gate every chart build funnels through, and the ONLY engine-contact point
  for a region's build closures) for exactly the selected region's chart
  tags and for nothing else.  Because an excluded region's loop does not
  even iterate, its engine work is skipped structurally -- the counting
  mock is the oracle, never a re-derivation of the filter logic.

* FILTERED RUN == FULL RUN RESTRICTED: the report emitted by a single-region
  run equals the full (``regions=None``) run's report with the OTHER
  regions' records removed -- identical charts, identical per-region
  summaries, identical interior/strata bookkeeping.  This is the WP
  acceptance criterion "the same charts and chart_reports as the full path
  restricted to that region" pinned at the structural level.

* REPORT SEMANTICS: the interior summary records the filter's effect on the
  admission bookkeeping (wedge/lobe block ran vs. skipped) so a
  silently-wrong filter cannot read green.

* PLUMBING: ``train()`` accepts ``regions`` as keyword-only with default
  ``None`` and forwards it unchanged to every ``_train_band_charts`` call;
  the ``scripts/train_lens_surrogate.py`` CLI maps ``--regions`` onto the
  ``train(regions=...)`` tuple.

ORACLE INDEPENDENCE
-------------------
The exclusivity and equality claims are decided by STRING/structure
comparisons on the ``_load_or_build`` call targets and the report records,
with a counting stub as the recording oracle.  The stub never re-derives the
filter decision -- it just reports what ``_train_band_charts`` asked it to
build -- so a regression that routes an excluded region's tiles into the
build loop is caught as a leaked tag, and a regression that silently drops a
selected region is caught as a missing tag.  No production predicate is
reimplemented in this suite.

The engine itself is never exercised here: ``_load_or_build`` and
``_reprovision_w_nodes`` (the two engine-backed chokepoints in
``_train_band_charts``) are stubbed, so every run is pure geometry
(caustic sweeps, admission tiling, window bookkeeping) at smoke scale.

COST BUDGET
-----------
Each ``_run_band_charts`` call is engine-free and measured at 0.05-2.5 s
(the exterior-only astroid run is the heaviest at ~1.4 s; the lobe-config
saddle run ~2.5 s).  ~18 distinct runs are lru-cached per
``(parity, band, regions, config)`` key, so the whole suite pays each once:
~15-25 s of engine-free work plus the ~4 s import, well under the 60 s
per-test / 5 min per-file fast-tier ceilings.  The single engine-backed
acceptance class (`RegionsInteriorOnlyEngineTestCase`) is ``_TRAIN_TIER_SKIP``
gated (driver post-build tier), never run in a build.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import inspect
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import TestCase, mock

import numpy as np

from cogwheel.lensing import surrogate_training as training
from cogwheel.lensing.surrogate_training import (
    PriorBox, TrainingConfig, stable_gamma_bands, train, _train_band_charts)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The canonical region-name set, single-sourced here so both the
#: default-equality claim and the per-region exclusivity claims key on one
#: tuple (mirroring the production inline default).  WP1 promoted
#: ``lobe_exterior`` (the macro-saddle deltoid-exterior shell, charted in
#: lobe-local ``(rho_lobe, theta_local)`` coordinates) to a first-class
#: region, so the tuple grew from four names to five.
_ALL_REGIONS: tuple[str, ...] = ('tube', 'exterior', 'wedge_interior',
                                 'lobe_interior', 'lobe_exterior')

#: Positive-parity (astroid) band: interior origin-enclosing, wedge tiles
#: admitted, tube + exterior regions live.  Topology-stable at 60 samples.
_ASTROID_BAND: tuple[float, float] = (0.2, 0.5)
_ASTROID_N_SAMPLES: int = 60

#: Saddle-parity band (deltoid).  At the default ``f_max`` the tube shell
#: fills the lobes, so lobe tiles admit zero here -- which is exactly what the
#: report-semantics test distinguishes (lobe block ran vs. skipped).
_SADDLE_BAND: tuple[float, float] = (1.3, 1.55)
_SADDLE_N_SAMPLES: int = 60

#: Saddle-parity band with a NARROW tube shell (``f_max=0.05``) so the
#: deltoid lobes admit real interior tiles (measured 84 at
#: ``n_farfield_tiles_per_side=5``) and the lobe filter has genuine teeth.
_SADDLE_LOBE_BAND: tuple[float, float] = (1.1, 1.2)
_LOBE_N_SAMPLES: int = 200

#: Smoke grid/budget config shared by the structural runs.  The eps bars are
#: opened wide so every stub chart registers; the records this suite reads --
#: chart tags, admitted counts, report sets -- do not depend on interpolation
#: accuracy.  ``max_farfield_regions=None`` keeps the full-run comparison
#: free of cap truncation (which would drop the wedge tail the equality
#: claim needs).
_CONFIG: TrainingConfig = TrainingConfig(
    n_gamma=4, n_u=4, n_theta=4, n_rho=4, n_theta_c=4, w_nodes_per_decade=3,
    n_farfield_tiles_per_side=3, max_farfield_regions=None,
    n_caustic_samples=60, n_heldout=6,
    tube_eps_max=1e9, farfield_eps_max=1e9, interior_eps_max=1e9)

#: The lobe-teeth variant: ``f_max=0.05`` narrows the tube shell so the
#: saddle deltoid lobes admit interior tiles; denser caustic sampling for a
#: faithful deltoid.
_LOBE_CONFIG: TrainingConfig = TrainingConfig(
    n_gamma=4, n_u=4, n_theta=4, n_rho=4, n_theta_c=4, w_nodes_per_decade=3,
    n_farfield_tiles_per_side=5, max_farfield_regions=None,
    n_caustic_samples=200, n_heldout=6, f_max=0.05,
    tube_eps_max=1e9, farfield_eps_max=1e9, interior_eps_max=1e9)


class _CountingTestCase(TestCase):
    """Anti-vacuity base (house idiom): a test that asserts nothing fails."""

    def setUp(self) -> None:
        self.comparisons = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self.comparisons, 0,
            'anti-vacuity: this test asserted nothing (zero comparisons).')


#: ENGINE-BACKED TIER (opt-in).  `RegionsInteriorOnlyEngineTestCase` runs the
#: REAL `train()` entry point end-to-end (engine chart builds, minutes per
#: run); it belongs to the driver's post-build tier, never the in-build fast
#: gate.
_TRAIN_TIER_SKIP = unittest.skipUnless(
    os.environ.get('COGWHEEL_TRAIN_TIER'),
    'engine-backed training tier: set COGWHEEL_TRAIN_TIER=1 (builds real '
    'surrogate charts, minutes; the driver runs these post-build)')


# ---------------------------------------------------------------------------
# Fixture harness
# ---------------------------------------------------------------------------


def _tag_kind(tag: str) -> str:
    """Chart region a build tag belongs to, from the tag's own infix.

    The five production tag infixes are unambiguous: exterior far-field tags
    are ``_ff_{i}_{j}`` (no wedge/lobe infix), wedge tags ``_ffwedge_``,
    lobe-interior tags ``_fflobe_``, lobe-exterior tags ``_fflobeext_``
    (WP1's macro-saddle exterior shell), tube tags ``_tube_``.  ``_ff_`` is a
    substring of ``_ffwedge_``/``_fflobe_``, so the interior infixes are
    checked first.  ``_fflobeext_`` is NOT a superstring of ``_fflobe_`` (the
    char after ``_fflobe`` is ``e``, not ``_``) NOR of ``_ff_`` (``_ffl...``),
    so without its own branch a lobe-exterior tag would fall through to
    ``'other'`` -- its check is placed BEFORE the ``_ff_`` exterior check so
    it can never be misread as exterior.
    """
    if '_fflobeext_' in tag:
        return 'lobe_exterior'
    if '_fflobe_' in tag:
        return 'lobe_interior'
    if '_ffwedge_' in tag:
        return 'wedge_interior'
    if '_tube_' in tag:
        return 'tube'
    if '_ff_' in tag:
        return 'exterior'
    return 'other'


def _fake_load_or_build(tags: list[str]):
    """Stub `_load_or_build` that records each requested chart tag.

    Returns a non-gated chart + report so `_gate_chart` registers it; the
    build closure is NEVER invoked, so the engine is never touched.
    """
    def fake(path, build_fn, provenance):  # mirrors _load_or_build(path, fn, prov)
        stem = Path(path).stem
        tags.append(stem)
        chart = types.SimpleNamespace(name=stem)
        report = {'heldout_eps': 1e-12, 'image_count': 4, 'kind': 'tube'}
        return chart, report, False
    return fake


@functools.lru_cache(maxsize=32)
def _run_band_charts(parity: int, band: tuple[float, float],
                     regions: tuple[str, ...] | None, config: TrainingConfig,
                     n_samples: int) -> tuple[tuple[str, ...], tuple[dict, ...]]:
    """Drive the SHIPPING `_train_band_charts` for one band with the engine
    stubbed, returning ``(sorted_build_tags, chart_reports)``.

    The geometry pipeline (caustic sweeps, admissions, tiling, windows,
    containment) runs for real; only the two engine chokepoints
    (`_load_or_build`, `_reprovision_w_nodes`) are stubbed.
    """
    stable, _dropped = stable_gamma_bands(
        band, parity, n_samples=n_samples, min_width=1e-6)
    sub, structure = stable[0]
    tags: list[str] = []
    charts: list = []
    reports: list[dict] = []
    with tempfile.TemporaryDirectory(prefix='regions_filt_') as tmp:
        with mock.patch.object(training, '_load_or_build',
                               new=_fake_load_or_build(tags)), \
             mock.patch.object(training, '_reprovision_w_nodes',
                               return_value=(3, {'status': 'ok',
                                                 'n_rec': 3})):
            _train_band_charts(
                box=PriorBox.from_prior_classes(), config=config,
                rng=np.random.default_rng(0), outdir=Path(tmp), parity=parity,
                label='L', band=sub, structure=structure, charts=charts,
                chart_reports=reports, regions=regions)
    return tuple(sorted(tags)), tuple(reports)


def _norm_reports(reports: tuple[dict, ...]) -> list[str]:
    """Canonical, order-independent report strings with the per-run
    ``file`` path (a fresh tempdir every run) dropped."""
    return sorted(str({k: v for k, v in r.items() if k != 'file'})
                  for r in reports)


def _interior_summary(reports: tuple[dict, ...]) -> dict:
    """The single ``interior_summary`` record of a band run."""
    matches = [r for r in reports if r.get('interior_summary')]
    assert len(matches) == 1, f'expected exactly one interior summary: {matches}'
    return matches[0]

# ===========================================================================
# Contract 1: default (None) == explicit all-regions (backward compat).
# ===========================================================================


class RegionsDefaultEqualsAllTestCase(_CountingTestCase):
    """``regions=None`` must build and report exactly the all-regions set.

    The WP ships ``regions=None`` (the pre-filter behaviour) as "train every
    region".  A future refactor that re-tightens or re-orders the inline
    default, or drops a region from it, must not silently ship -- this
    equality is the tripwire.
    """

    def test_astroid_default_equals_explicit_all(self) -> None:
        """Same build tags and same reports (modulo tempdir paths)."""
        default = _run_band_charts(1, _ASTROID_BAND, None, _CONFIG,
                                   _ASTROID_N_SAMPLES)
        explicit = _run_band_charts(1, _ASTROID_BAND, _ALL_REGIONS, _CONFIG,
                                    _ASTROID_N_SAMPLES)
        self.assertEqual(default[0], explicit[0],
                         'default and explicit-all must request the same '
                         'chart tags')
        self.assertEqual(_norm_reports(default[1]),
                         _norm_reports(explicit[1]),
                         'default and explicit-all must emit the same reports')
        self.comparisons += 2

    def test_saddle_default_equals_explicit_all(self) -> None:
        """Same for the saddle parity (tube + exterior; lobes admit zero at
        the default ``f_max``, which both runs record identically)."""
        default = _run_band_charts(-1, _SADDLE_BAND, None, _CONFIG,
                                   _SADDLE_N_SAMPLES)
        explicit = _run_band_charts(-1, _SADDLE_BAND, _ALL_REGIONS, _CONFIG,
                                    _SADDLE_N_SAMPLES)
        self.assertEqual(default[0], explicit[0])
        self.assertEqual(_norm_reports(default[1]),
                         _norm_reports(explicit[1]))
        self.comparisons += 2

    def test_default_run_builds_every_region_astroid(self) -> None:
        """The default astroid run actually touches all three live regions
        (tube, exterior, wedge) -- so the equality claim is not vacuous."""
        tags, _reports = _run_band_charts(1, _ASTROID_BAND, None, _CONFIG,
                                          _ASTROID_N_SAMPLES)
        kinds = {_tag_kind(t) for t in tags}
        self.assertLessEqual(kinds, set(_ALL_REGIONS),
                             f'unexpected tag kinds {kinds}')
        self.assertIn('tube', kinds)
        self.assertIn('exterior', kinds)
        self.assertIn('wedge_interior', kinds)
        self.assertNotIn('lobe_interior', kinds)  # astroid parity has no lobes
        self.comparisons += 4


class TrainSignatureTestCase(_CountingTestCase):
    """The public `train()` surface for the filter: keyword-only ``regions``
    with a ``None`` default, so every pre-filter call site is unchanged."""

    def test_regions_is_keyword_only_with_none_default(self) -> None:
        sig = inspect.signature(train)
        param = sig.parameters['regions']
        self.assertEqual(param.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(param.default)
        self.comparisons += 2

# ===========================================================================
# Contract 2: per-region exclusivity -- a filtered run builds ONLY the
# selected region's charts (and its engine work is structurally skipped).
# ===========================================================================


class RegionExclusivityTestCase(_CountingTestCase):
    """For each region, ``regions=(r,)`` asks the build funnel for exactly
    ``r``'s chart tags and nothing else.

    The recorded tag list is the oracle: because ``_load_or_build`` is the
    only path by which a region's build closure (and hence its engine work)
    can run, an empty tag list for an excluded region proves its engine work
    was structurally skipped -- the loop never iterated.
    """

    def test_tube_only_astroid(self) -> None:
        tags, reports = _run_band_charts(1, _ASTROID_BAND, ('tube',), _CONFIG,
                                         _ASTROID_N_SAMPLES)
        self.assertGreater(len(tags), 0, 'tube filter must build >=1 tube')
        self.assertEqual([_tag_kind(t) for t in tags], ['tube'] * len(tags),
                         'tube-only run must build only tube charts')
        # The exterior region summary must be absent from the report.
        self.assertFalse(any(r.get('exterior_region_summary')
                             for r in reports),
                         'tube-only run must not emit an exterior summary')
        self.comparisons += 3

    def test_exterior_only_astroid(self) -> None:
        tags, _reports = _run_band_charts(1, _ASTROID_BAND, ('exterior',),
                                          _CONFIG, _ASTROID_N_SAMPLES)
        self.assertGreater(len(tags), 0, 'exterior filter must build >=1 tile')
        self.assertEqual({_tag_kind(t) for t in tags}, {'exterior'},
                         'exterior-only run must build only exterior charts')
        self.assertFalse(any('_ffwedge_' in t or '_fflobe_' in t for t in tags),
                         'exterior-only run must not build interior charts')
        self.comparisons += 3

    def test_wedge_interior_only_astroid(self) -> None:
        tags, _reports = _run_band_charts(1, _ASTROID_BAND,
                                          ('wedge_interior',), _CONFIG,
                                          _ASTROID_N_SAMPLES)
        self.assertGreater(len(tags), 0, 'wedge filter must build >=1 tile')
        self.assertEqual({_tag_kind(t) for t in tags}, {'wedge_interior'},
                         'wedge-only run must build only wedge charts')
        self.assertFalse(any('_ff_' in t and '_ffwedge_' not in t for t in tags),
                         'wedge-only run must not build exterior/tube charts')
        self.comparisons += 3

    def test_tube_only_saddle(self) -> None:
        tags, reports = _run_band_charts(-1, _SADDLE_BAND, ('tube',), _CONFIG,
                                         _SADDLE_N_SAMPLES)
        self.assertGreater(len(tags), 0)
        self.assertEqual({_tag_kind(t) for t in tags}, {'tube'})
        self.assertFalse(any(r.get('exterior_region_summary')
                             for r in reports))
        self.comparisons += 3

    def test_exterior_only_saddle(self) -> None:
        """After WP1's deltoid-exterior rewiring, the origin-polar
        saddle-exterior tiler is RETIRED: the exterior far-field block runs
        ONLY at positive parity (``'exterior' in regions and parity == 1``),
        so a saddle-parity run with ``regions=('exterior',)`` builds ZERO
        charts.  The macro-saddle exterior shell is now owned by the separate
        ``lobe_exterior`` region (see ``test_lobe_exterior_only_saddle``)."""
        tags, reports = _run_band_charts(-1, _SADDLE_BAND, ('exterior',),
                                         _CONFIG, _SADDLE_N_SAMPLES)
        self.assertEqual(tags, (),
                         'saddle exterior-only run must build no charts: the '
                         'exterior far-field block is positive-parity only')
        self.assertFalse(any('file' in r for r in reports),
                         'saddle exterior-only run must emit no chart records')
        self.assertFalse(any(r.get('exterior_region_summary')
                             for r in reports),
                         'the exterior region block never runs at saddle '
                         'parity, so no exterior summary is emitted')
        self.comparisons += 3

    def test_lobe_exterior_only_saddle(self) -> None:
        """The new ``lobe_exterior`` filter builds only lobe-exterior charts
        at the narrow-tube saddle band where the deltoid lobes' exterior
        shell genuinely admits tiles.

        Every produced tag must decode to ``'lobe_exterior'`` (only
        ``_fflobeext_`` tags), the set must be NON-EMPTY, and no tube /
        lobe-interior / wedge / origin-exterior tags may leak.  ``_fflobeext_``
        contains neither ``_fflobe_`` nor ``_ff_`` as a substring, so the
        no-leak assertions below are exact, not accidentally self-satisfied.
        """
        tags, _reports = _run_band_charts(-1, _SADDLE_LOBE_BAND,
                                          ('lobe_exterior',), _LOBE_CONFIG,
                                          _LOBE_N_SAMPLES)
        self.assertGreater(
            len(tags), 0,
            'lobe_exterior filter must build >=1 exterior-shell tile at the '
            'narrow-tube saddle band')
        self.assertEqual({_tag_kind(t) for t in tags}, {'lobe_exterior'},
                         'lobe-exterior-only run must build only '
                         'lobe_exterior charts')
        self.assertFalse(
            any('_tube_' in t or '_fflobe_' in t or '_ffwedge_' in t
                or '_ff_' in t for t in tags),
            'lobe-exterior-only run must not build tube/interior/exterior '
            'charts')
        self.comparisons += 3

    def test_lobe_interior_only_saddle(self) -> None:
        """Lobe filter builds only lobe charts at a band where the deltoid
        lobes genuinely admit interior tiles (narrow tube shell)."""
        tags, reports = _run_band_charts(-1, _SADDLE_LOBE_BAND,
                                         ('lobe_interior',), _LOBE_CONFIG,
                                         _LOBE_N_SAMPLES)
        self.assertGreater(len(tags), 0,
                           'lobe filter must build >=1 lobe tile at this band')
        self.assertEqual({_tag_kind(t) for t in tags}, {'lobe_interior'},
                         'lobe-only run must build only lobe charts')
        self.assertTrue(_interior_summary(reports)['served'],
                        'admitted lobe tiles must mark the interior served')
        self.comparisons += 3

    def test_excluded_lobe_engine_work_skipped(self) -> None:
        """Same band/config WITHOUT ``lobe_interior`` must not build any lobe
        chart -- the lobe loop did not iterate, so no lobe engine work ran."""
        tags, reports = _run_band_charts(-1, _SADDLE_LOBE_BAND, ('tube',),
                                         _LOBE_CONFIG, _LOBE_N_SAMPLES)
        self.assertFalse(any('_fflobe_' in t for t in tags),
                         'lobe-excluded run must not build lobe charts')
        self.assertTrue(_interior_summary(reports)['interior_zero_admission'],
                        'skipped lobe block must leave a zero-admission '
                        'interior record')
        self.comparisons += 2

    def test_empty_regions_builds_nothing(self) -> None:
        """``regions=()`` selects no region: zero chart builds, on both
        parities.  The summaries are still recorded (the ladder still needs
        the coverage bookkeeping)."""
        for parity, band, n_samples in ((1, _ASTROID_BAND, _ASTROID_N_SAMPLES),
                                        (-1, _SADDLE_BAND, _SADDLE_N_SAMPLES)):
            with self.subTest(parity=parity):
                tags, reports = _run_band_charts(
                    parity, band, (), _CONFIG, n_samples)
                self.assertEqual(tags, (),
                                 'empty regions must request no chart builds')
                # No CHART records (each carries a ``file``/``kind``) -- only
                # the coverage bookkeeping summaries remain.
                self.assertFalse(any('file' in r for r in reports),
                                 'empty regions must emit no chart records')
                self.assertFalse(any(r.get('exterior_region_summary')
                                     for r in reports))
                self.comparisons += 3

# ===========================================================================
# Contract 3: a filtered run == the full run restricted to that region
# (the WP acceptance criterion, pinned at the structural level).
# ===========================================================================


class RegionsFilterMatchesFullRunTestCase(_CountingTestCase):
    """The single-region run must emit exactly the full run's records for
    that region -- identical charts and identical per-region bookkeeping --
    and nothing of the excluded regions.

    The comparison set is the full report MINUS the other regions' chart
    records and summaries; the selected region's own chart records, its
    region summary, and (for the interior family, where it is the parity's
    only interior region) the interior/strata summaries must be byte-equal
    between the two runs.
    """

    def test_wedge_only_matches_full_restricted_astroid(self) -> None:
        full = _run_band_charts(1, _ASTROID_BAND, None, _CONFIG,
                                _ASTROID_N_SAMPLES)
        wedge = _run_band_charts(1, _ASTROID_BAND, ('wedge_interior',),
                                 _CONFIG, _ASTROID_N_SAMPLES)
        full_minus_others = [
            r for r in full[1]
            if not (r.get('exterior_region_summary')
                    or ('_tube_' in str(r.get('name')))
                    or ('_ff_' in str(r.get('name'))
                        and '_ffwedge_' not in str(r.get('name'))))]
        self.assertEqual(_norm_reports(tuple(full_minus_others)),
                         _norm_reports(wedge[1]),
                         'wedge-only run must equal the full run restricted '
                         'to the wedge region')
        self.assertEqual(
            [t for t in full[0] if _tag_kind(t) == 'wedge_interior'],
            list(wedge[0]),
            'wedge build tags must be identical to the full run\'s')
        self.comparisons += 2

    def test_exterior_only_matches_full_restricted_astroid(self) -> None:
        full = _run_band_charts(1, _ASTROID_BAND, None, _CONFIG,
                                _ASTROID_N_SAMPLES)
        ext = _run_band_charts(1, _ASTROID_BAND, ('exterior',), _CONFIG,
                               _ASTROID_N_SAMPLES)
        # Compare the EXTERIOR-region-specific records only: the interior and
        # strata summaries legitimately differ between the runs (the full run
        # admits the wedge interior, the exterior-only run excludes it).
        def exterior_records(reports):
            return [r for r in reports
                    if r.get('exterior_region_summary')
                    or ('_ff_' in str(r.get('name'))
                        and '_ffwedge_' not in str(r.get('name')))]
        self.assertEqual(_norm_reports(tuple(exterior_records(full[1]))),
                         _norm_reports(tuple(exterior_records(ext[1]))),
                         'exterior-only run must equal the full run '
                         'restricted to the exterior region')
        self.assertEqual(
            [t for t in full[0] if _tag_kind(t) == 'exterior'],
            list(ext[0]),
            'exterior build tags must be identical to the full run\'s')
        self.comparisons += 2

    def test_lobe_only_matches_full_restricted_saddle(self) -> None:
        """The lobe-INTERIOR-family comparison (narrow-tube band where the
        lobes admit tiles).

        WP1's full (``regions=None``) saddle run now ALSO builds the
        ``lobe_exterior`` family (``_fflobeext_`` chart records plus a
        ``lobe_exterior_summary`` report).  The ``lobe_interior``-only
        restriction excludes that family, so the baseline must strip it too --
        otherwise the leaked exterior-shell records make the equality go RED
        for the wrong reason.
        """
        full = _run_band_charts(-1, _SADDLE_LOBE_BAND, None, _LOBE_CONFIG,
                                _LOBE_N_SAMPLES)
        lobe = _run_band_charts(-1, _SADDLE_LOBE_BAND, ('lobe_interior',),
                                _LOBE_CONFIG, _LOBE_N_SAMPLES)
        full_minus_others = [
            r for r in full[1]
            if not (r.get('exterior_region_summary')
                    or r.get('lobe_exterior_summary')
                    or ('_tube_' in str(r.get('name')))
                    or ('_fflobeext_' in str(r.get('name')))
                    or ('_ff_' in str(r.get('name'))
                        and '_fflobe_' not in str(r.get('name'))))]
        self.assertEqual(_norm_reports(tuple(full_minus_others)),
                         _norm_reports(lobe[1]),
                         'lobe-only run must equal the full run restricted '
                         'to the lobe region')
        self.assertEqual(
            [t for t in full[0] if _tag_kind(t) == 'lobe_interior'],
            list(lobe[0]),
            'lobe build tags must be identical to the full run\'s')
        self.comparisons += 2


# ===========================================================================
# Contract 4: the interior summary records the filter's effect on the
# admission bookkeeping (so a silently-wrong filter cannot read green).
# ===========================================================================


class RegionsReportSemanticsTestCase(_CountingTestCase):
    """The interior summary distinguishes "region selected (block ran)" from
    "region excluded (block skipped)" on BOTH parities.

    For the saddle parity the distinguishing pair is ``interior_skipped``
    (set ONLY inside the lobe block, at zero admission) vs
    ``interior_zero_admission`` (set only when the lobe block did not run).
    For the astroid parity the wedge block's tiles drive
    ``interior_admitted_tiles`` / ``interior_zero_admission``.
    """

    def test_saddle_lobe_selected_records_lobe_skip(self) -> None:
        """At the default tube shell the lobes admit zero tiles, and the
        lobe block's own run records that as ``saddle_lobes_zero_admission``."""
        _tags, reports = _run_band_charts(-1, _SADDLE_BAND,
                                          ('lobe_interior',), _CONFIG,
                                          _SADDLE_N_SAMPLES)
        interior = _interior_summary(reports)
        self.assertEqual(interior.get('interior_skipped'),
                         'saddle_lobes_zero_admission')
        self.assertNotIn('interior_zero_admission', interior)
        self.assertFalse(interior['served'])
        self.comparisons += 3

    def test_saddle_lobe_excluded_records_zero_admission(self) -> None:
        """Skipping the lobe block yields ``interior_zero_admission`` instead
        of the block's own skip note -- the record proves the block did not
        run."""
        _tags, reports = _run_band_charts(-1, _SADDLE_BAND, ('tube',), _CONFIG,
                                          _SADDLE_N_SAMPLES)
        interior = _interior_summary(reports)
        self.assertNotIn('interior_skipped', interior)
        self.assertTrue(interior['interior_zero_admission'])
        self.assertEqual(interior['interior_admitted_tiles'], 0)
        self.comparisons += 3

    def test_astroid_wedge_excluded_records_zero_admission(self) -> None:
        """Excluding ``wedge_interior`` leaves the astroid interior unbuilt
        (``interior_zero_admission``), while the full run admits the wedge
        tiles -- the two interiors are not silently merged."""
        _tags, full_reports = _run_band_charts(
            1, _ASTROID_BAND, None, _CONFIG, _ASTROID_N_SAMPLES)
        _tags, ext_reports = _run_band_charts(
            1, _ASTROID_BAND, ('exterior',), _CONFIG, _ASTROID_N_SAMPLES)
        self.assertGreater(_interior_summary(full_reports)[
            'interior_admitted_tiles'], 0)
        self.assertTrue(_interior_summary(ext_reports)[
            'interior_zero_admission'])
        self.comparisons += 2

# ===========================================================================
# Contract 5: plumbing -- `train()` forwards the filter, and the CLI maps
# `--regions` onto the `train` tuple argument.
# ===========================================================================


class TrainRegionsPlumbingTestCase(_CountingTestCase):
    """`train(regions=...)` reaches every `_train_band_charts` call verbatim.

    The heavy pieces of `train()` are stubbed (topology detection, chart
    registration, artifact serialization) so the pass-through is pinned
    without engine work; only the region argument is under test.
    """

    def _train_with(self, regions) -> tuple:
        recorded: dict = {}
        fake_structure = types.SimpleNamespace(
            detected_cusps=0, cusp_thetas=[], caustic_reach=0.0, arcs=[])

        def fake_stable_gamma_bands(band, parity, **kwargs):
            return [((band[0], band[1]), fake_structure)], []

        def fake_train_band_charts(**kwargs):
            recorded.update(kwargs)

        with mock.patch.object(training, 'stable_gamma_bands',
                               side_effect=fake_stable_gamma_bands), \
             mock.patch.object(training, 'get_certified_ppgo_map',
                               return_value=None), \
             mock.patch.object(training, '_train_band_charts',
                               side_effect=fake_train_band_charts), \
             mock.patch.object(training, 'LensAmplificationSurrogate'), \
             tempfile.TemporaryDirectory(prefix='regions_train_') as tmp:
            train(outdir=tmp, config=_CONFIG, regions=regions)
        return recorded

    def test_regions_forwarded_to_every_band_charts_call(self) -> None:
        recorded = self._train_with(('wedge_interior',))
        self.assertIn('regions', recorded)
        self.assertEqual(recorded['regions'], ('wedge_interior',))
        self.comparisons += 2

    def test_none_default_forwarded_verbatim(self) -> None:
        """The pre-filter call shape (``regions=None``) still reaches the
        band trainer unchanged."""
        recorded = self._train_with(None)
        self.assertIn('regions', recorded)
        self.assertIsNone(recorded['regions'])
        self.comparisons += 2

    def test_empty_tuple_forwarded_verbatim(self) -> None:
        recorded = self._train_with(())
        self.assertIn('regions', recorded)
        self.assertEqual(recorded['regions'], ())
        self.comparisons += 2


class TrainLensSurrogateCliTestCase(_CountingTestCase):
    """`scripts/train_lens_surrogate.py --regions` maps onto the
    `train(regions=...)` tuple argument."""

    @staticmethod
    def _load_script() -> types.ModuleType:
        path = (Path(__file__).parents[2] / 'scripts'
                / 'train_lens_surrogate.py')
        spec = importlib.util.spec_from_file_location(
            'train_lens_surrogate_cli', path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_regions_flag_forwards_tuple(self) -> None:
        script = self._load_script()
        with mock.patch.object(script, 'train') as fake_train, \
             mock.patch.object(sys, 'argv', [
                 'train_lens_surrogate.py', 'outdir',
                 '--regions', 'wedge_interior', 'exterior']):
            fake_train.return_value = (
                {}, {'artifact': {'n_charts': 0, 'size_bytes': 0,
                                  'path': 'x'}})
            script.main()
        self.assertEqual(fake_train.call_args.kwargs['regions'],
                         ('wedge_interior', 'exterior'))
        self.comparisons += 1

    def test_no_regions_flag_forwards_none(self) -> None:
        script = self._load_script()
        with mock.patch.object(script, 'train') as fake_train, \
             mock.patch.object(sys, 'argv', [
                 'train_lens_surrogate.py', 'outdir']):
            fake_train.return_value = (
                {}, {'artifact': {'n_charts': 0, 'size_bytes': 0,
                                  'path': 'x'}})
            script.main()
        self.assertIsNone(fake_train.call_args.kwargs['regions'])
        self.comparisons += 1

    def test_unknown_region_choice_is_rejected_by_argparse(self) -> None:
        """The CLI's ``choices`` guard refuses an out-of-set region name at
        parse time -- the filter surface is closed by construction."""
        script = self._load_script()
        with mock.patch.object(sys, 'argv', [
                'train_lens_surrogate.py', 'outdir',
                '--regions', 'interior']):  # retired/unknown name
            with self.assertRaises(SystemExit):
                script.main()
        self.comparisons += 1

# ===========================================================================
# Self-falsification: prove the green checks can go red.
# ===========================================================================


class RegionsFilterSelfFalsificationTestCase(_CountingTestCase):
    """Corrupt each contract and prove the corresponding check fails.

    Without this, a silently-passing suite is indistinguishable from a
    correct one.  Each test injects the defect the real test above is built
    to catch and asserts the check trips.
    """

    def test_leaked_excluded_tag_trips_exclusivity(self) -> None:
        """A production regression that routed an excluded region's tile into
        the build loop would show up as a leaked tag; the exclusivity check
        must reject exactly that."""
        tags = ['chart_L_s0_ffwedge_0_0', 'chart_L_tube_0']
        violations = [t for t in tags if _tag_kind(t) != 'wedge_interior']
        self.assertEqual(violations, ['chart_L_tube_0'])
        with self.assertRaises(AssertionError):
            self.assertEqual(
                violations, [],
                'a leaked non-wedge tag must trip the wedge exclusivity check')
        self.comparisons += 2

    def test_missing_selected_tag_trips_nonvacuity_guard(self) -> None:
        """If the wedge filter built nothing at all, the `assertGreater(len(tags),
        0)` guard must fire -- a silently-empty run cannot read green."""
        with self.assertRaises(AssertionError):
            self.assertGreater(0, 0,
                               'wedge filter must build >=1 tile')
        self.comparisons += 1

    def test_wrong_default_breaks_default_equality(self) -> None:
        """The default==all claim is discriminating: a ``regions`` tuple that
        drops a region (simulating a future default regression) yields a
        DIFFERENT report set from the true all-regions default."""
        full = _run_band_charts(1, _ASTROID_BAND, None, _CONFIG,
                                _ASTROID_N_SAMPLES)
        tube_only = _run_band_charts(1, _ASTROID_BAND, ('tube',), _CONFIG,
                                     _ASTROID_N_SAMPLES)
        self.assertNotEqual(_norm_reports(full[1]),
                            _norm_reports(tube_only[1]),
                            'a default missing the exterior/wedge regions '
                            'must NOT equal the true default')
        self.assertNotEqual(list(full[0]), list(tube_only[0]))
        self.comparisons += 2

    def test_report_semantics_pair_is_mutually_exclusive(self) -> None:
        """The ``interior_skipped`` vs ``interior_zero_admission`` signals are
        genuinely alternative bookkeeping: the saddle lobe-selected and
        lobe-excluded runs must differ in exactly that pair."""
        _t1, lobe_on = _run_band_charts(-1, _SADDLE_BAND,
                                        ('lobe_interior',), _CONFIG,
                                        _SADDLE_N_SAMPLES)
        _t2, lobe_off = _run_band_charts(-1, _SADDLE_BAND, ('tube',), _CONFIG,
                                         _SADDLE_N_SAMPLES)
        on = _interior_summary(lobe_on)
        off = _interior_summary(lobe_off)
        self.assertNotEqual(on.get('interior_skipped'),
                            off.get('interior_skipped'))
        self.assertNotEqual(on.get('interior_zero_admission'),
                            off.get('interior_zero_admission'))
        self.comparisons += 2


# ===========================================================================
# Engine-backed acceptance (driver post-build tier; never in a build).
# ===========================================================================


@_TRAIN_TIER_SKIP
class RegionsInteriorOnlyEngineTestCase(_CountingTestCase):
    """WP acceptance, end-to-end: a REAL interior-only ``train()`` run
    completes (interior-scale time, not the ~40-min full exterior) and the
    packed artifact contains interior charts and nothing else.

    Structural comparison to the full run is already pinned engine-free in
    `RegionsFilterMatchesFullRunTestCase`; this class certifies the real
    engine path obeys the same filter.
    """

    def test_interior_only_real_run_builds_interior_charts_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix='regions_engine_') as tmp:
            _surrogate, report = train(
                outdir=tmp, config=_CONFIG, regions=('wedge_interior',))
        names = [str(r.get('name')) for r in report['charts']]
        self.assertTrue(
            any('_ffwedge_' in n for n in names),
            'interior-only run must build >=1 wedge chart')
        self.assertFalse(
            any('_ff_' in n and '_ffwedge_' not in n for n in names),
            'interior-only run must not build exterior far-field charts')
        self.assertFalse(
            any('_tube_' in n for n in names),
            'interior-only run must not build tube charts')
        self.assertFalse(
            any(r.get('exterior_region_summary') for r in report['charts']),
            'interior-only run must not emit an exterior region summary')
        interior = [r for r in report['charts'] if r.get('interior_summary')]
        self.assertTrue(interior and interior[0]['interior_admitted_tiles'] > 0,
                        'astroid wedge interior must be admitted and served')
        self.comparisons += 5
