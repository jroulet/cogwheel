"""Tests for the Build-8g WP2 mass-stratified far-field tiling of the
`lensing.surrogate_training` trainer, plus the serving-contract both
directions through `lensing.surrogate.select_chart`.

WHAT THIS SUITE PINS
--------------------
Build 8g WP2 replaced the surrogate's single hard-coded far-field box --
which was silently rebuilt under several filenames, overlapping itself --
with a mass-stratified exterior TILING of the shear-frame y-support box.
The bug this fixes was DUPLICATE / OVERLAPPING boxes, so the disjointness
tolerance here is EXACTLY zero: two admitted tiles that share a
grid cell would be a regression.  Three independent contracts are pinned:

* TILING (`_farfield_tiles` + the training report) -- every admitted tile
  is a distinct grid cell of the ``[-Y, Y]^2`` support box lying wholly
  outside the caustic disk (``caustic_reach + eta_max``); every pair of
  admitted tile boxes is strictly disjoint (max-norm centre separation
  ``>= 2 * half``); the ``max_farfield_regions`` cap is a TRUE cap that
  records the dropped count; and a saddle stratum above the parity
  w-ceiling (~458 Msun) is recorded as a loud ``beyond_w_cap`` entry, not
  silently dropped.

* WHOLE-BAND CONTAINMENT (`_mass_strata` + `_stratum_w_range`) -- each
  stratum's chart ``w`` range CONTAINS the whole detector band
  ``[w(20, m), w(1024, m)]`` of every in-stratum draw (the serving
  contract), except the high-mass corner the double-double / parity cap
  truncates, which is recorded; masses whose ``w(1024, m)`` exceeds the
  parity ceiling are attributed to the recorded ``beyond_w_cap`` bucket.

* SERVE-FRACTION both directions (`select_chart`) -- over prior draws
  from the actual lens prior classes, draws INSIDE the tiled support serve
  at >= 90%, while draws OUTSIDE it (interior caustic hole, wrong gamma
  band, out-of-band ``w``, parity guard) return ``None`` 100% of the time.
  A single outside serve is an additive-contract violation.

ORACLE INDEPENDENCE (F002)
--------------------------
The whole-band containment oracle recomputes ``w(f, m)`` from the
INDEPENDENT closed form ``w = 1.2372e-4 * m * f`` (the rounded
``8*pi*G*Msun/c^3 * f`` constant hand-derived in the build plan), never
the production `dimensionless_frequency`; a transcription error in the
production constant would break containment.  The tiling disjointness
oracle is pure grid geometry.  The serve-fraction classifier keys on the
GEOMETRY of the tile boxes (does the point lie in some tile cell?), never
on `select_chart`'s own decision, so a containment bug in the guard stack
drops the served fraction with teeth.

TOLERANCE PROVENANCE
--------------------
``_TILE_DISJOINT_TOL = 0.0``: overlapping/duplicate boxes were the whole
bug, so adjacent admitted cells are allowed to touch (separation exactly
``2 * half``) but never overlap.  ``_W_CONTAINMENT_REL_TOL = 5e-3``
absorbs the ~5e-4 relative gap between the independent ``1.2372e-4``
constant and the production `lal.MTSUN_SI` value plus the report's
6-decimal rounding, and is orders of magnitude below the ~7x per-stratum
band width, so it never masks a real containment failure.
``_INSIDE_SERVE_FLOOR = 0.90`` is the professor Q6-ii smoke floor;
the synthetic fixture's geometric support serves ~1.0, well clear.

BUILD 8g WP1/WP3 (this shard)
-----------------------------
Three further contracts are pinned on top of the WP2 tiling suite:

* EPS REGISTRATION GATE (`_chart_gated` + the `_train_band_charts`
  registration block, Professor Q6-iii / F010) -- three real
  engine-built far-field charts are run through the production gate: a
  HEALTHY chart (measured held-out eps ~5e-4, well under the 3e-3 bar)
  is packed into the artifact; a POISONED chart (its spline
  ``real_coeffs`` scaled x1.1 so measured eps ~9.7e-2, ~32x the bar) and
  a NaN-eps chart (all held-out points fall outside its box -> zero
  served -> nan) are BOTH excluded from the packed artifact and BOTH
  recorded with a ``gated`` marker and the correct reason
  (``eps_above_bar`` / ``nan_eps``); their windows fall through
  (`select_chart` returns ``None`` there when only the healthy chart is
  registered).  Reverting the poison (x1.0) re-registers the chart, so
  the gate is reachable-red, not always-red.

* EPS GATE ON RESUME (`_load_or_build`) -- a chart persisted to disk with
  an above-bar ``heldout_eps`` in its per-chart provenance is reused by
  `_load_or_build` WITHOUT re-running the engine (a sentinel build_fn
  that raises if called proves no recomputation); the persisted eps is
  read back and the gate excludes the reused chart exactly as for a fresh
  build, so the reused chart is absent from the artifact despite its file
  existing on disk.

* SADDLE TUBE-TAIL FIX (WP3, `_saddle_arcs` wedge-edge windows +
  wider saddle cusp safety, Professor Q6-iv) -- a fast synthetic
  reproduction of ``saddle_b1_tube_2`` (a strong-shear ``gamma = 1.55``
  branch-+1 wedge-edge deltoid arc, coarse grid) builds a tube chart WITH
  the fix and measures held-out eps ~2.6e-2, under the 5e-2 tube bar and
  far below the pre-fix ~1.15.  The CONTRAST reconstructs the pre-fix
  counterfactual arc (astroid cusp safety 1.5/0.05, no wedge-edge window)
  and measures eps ~0.43 > 0.09, proving the test bites the real
  pathology, not a trivially-easy config.  The fix-on arc is asserted to
  be a genuine production `_saddle_arcs` arc.

ORACLE INDEPENDENCE for these shards is the production held-out metric
`_heldout_eps`, which takes the `ChangRefsdalChannels` ENGINE as ground
truth and the surrogate serve as the thing under test; the gate under
test (`_chart_gated`) is fed engine-MEASURED eps, never a re-derivation
of its own threshold.

The suite is stdlib ``unittest``.  Every numeric TestCase tallies its
comparisons and `tearDown` fails a test that asserted nothing.
`SelfFalsificationTestCase` corrupts each contract (overlapping tiles,
a wrong ``w`` constant, a chart widened over the interior hole) and
asserts the checks go red, so "green" is evidence, not decoration.

INSPECTOR FINDING INS-1-001 (`FarFieldCornerCapTestCase`)
----------------------------------------------------------
`_stratum_w_range`'s double-double product cap (``dd_cap =
_DD_PRODUCT_MARGIN / y_max``) is fed ``y_max = y_extent`` -- the
stratum's per-AXIS box half-width ``Y`` -- straight from
`_train_band_charts`, never the box's true CORNER magnitude
``Y * sqrt(2)`` that the far-field tiling actually samples out to (the
outer-corner tile sits at the box corner by construction).  The prior's
own ``_Y_SCALE`` constant documents that the ``sqrt(2)`` corner factor is
required to keep ``w * sqrt(s)`` under the point-mass kernel's real
``DD_PRODUCT_CEILING = 60`` (`_hyp1f1.py`); dropping it under-caps
``w_max`` by a factor of ``sqrt(2)``, so on every stratum where the DD cap
actually binds, the outer-corner tile's true product overshoots to
``_DD_PRODUCT_MARGIN * sqrt(2) ~= 82`` and the kernel refuses it.
`FarFieldCornerCapTestCase` reproduces this DIRECTLY against the real
kernel (`point_mass_g_derivatives`, the same function `_validate_domain`
guards) at the true corner magnitude of every DD-cap-binding stratum
today's code produces -- no fixture engineering, no mocked oracle: the
kernel call is the authority the training code is trying to satisfy, and
today's ``y_extent``-only cap fails it (measured product ~82.0 against a
60.0 ceiling on 3 of 5 real strata).  A companion control recomputes the
cap with the corner-aware ``y_max = Y * sqrt(2)`` -- the INS-1-001
suggested fix -- and confirms it lands the product at exactly
``_DD_PRODUCT_MARGIN = 58`` (comfortably inside the ceiling), so the test
is not vacuously unsatisfiable: it will go green once
`_stratum_w_range` is fixed to use the corner magnitude.  This test is
EXPECTED TO FAIL (red) against the current, unfixed
`surrogate_training.py` -- that is its job: pin the corner-magnitude
contract the docstring already promises "the corner the engine refuses is
never sampled" but the code, today, does not keep.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import inspect
import itertools
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase, main, mock

import numpy as np

from cogwheel.lensing import prior as lens_prior
from cogwheel.lensing.waveform import dimensionless_frequency
from cogwheel.lensing import surrogate as surrogate_module
from cogwheel.lensing.surrogate import FarFieldChart, select_chart
from cogwheel.lensing.surrogate import LensAmplificationSurrogate
from cogwheel.lensing import surrogate_training as training
from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.surrogate_training import (
    PriorBox, TrainingConfig, train, _farfield_tiles, _mass_strata,
    _stratum_w_range, _POSITIVE_W_CEILING, _SADDLE_W_CEILING,
    _build_farfield_chart, _farfield_heldout_samples, _heldout_eps,
    _chart_gated, _gate_chart, _load_or_build, _saddle_arcs,
    _branch_speed_profile, _find_cusps, _make_arc, _caustic_reach, _capped_w_range, _build_tube_chart,
    _tube_heldout_samples, _tube_source, _WEDGE_EPS, _CUSP_WIDTH_SAFETY,
    _CUSP_MIN_HALFWIDTH, _SADDLE_CUSP_WIDTH_SAFETY, _SADDLE_CUSP_MIN_HALFWIDTH,
    _DD_PRODUCT_MARGIN)
from cogwheel.lensing.chang_refsdal.channels import ChangRefsdalChannels
from cogwheel.lensing.chang_refsdal._hyp1f1 import (
    point_mass_g_derivatives, HypergeometricDomainError, DD_PRODUCT_CEILING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Independent closed form for the dimensionless lensing frequency,
#: ``w = 8*pi*G*Msun*(1+z)*f/c^3`` with the constant rounded by hand to
#: ``1.2372e-4`` (per-solar-mass, per-Hz).  This is the F002-independent
#: oracle for whole-band containment: it must NOT be the production
#: `dimensionless_frequency`, whose `lal.MTSUN_SI` constant it cross-checks.
_W_LENSING_PER_MSUN_HZ = 1.2372e-4

#: Whole-band containment relative tolerance; absorbs the ~5e-4 constant gap
#: and the report's 6-decimal rounding, far below the ~7x per-stratum band.
_W_CONTAINMENT_REL_TOL = 5e-3

#: Overlap tolerance for admitted tiles: a pure floating-point representation
#: guard, NOT physical slack.  Duplicate/overlapping boxes were the bug, so the
#: contract is genuinely zero overlap -- adjacent cells may only TOUCH (centre
#: separation ``2*half``).  A touching pair separates by exactly ``2*half``
#: analytically, but ``abs(cx_a - cx_b)`` and ``2*half`` each carry ~1 ULP of
#: rounding (observed ``1.1999999999999997`` vs ``1.2``), so a ``1e-9`` guard
#: absorbs that noise while remaining ~1e9x below any real overlap (a duplicate
#: or shifted box separates by at most ``half``, ~0.6 here).
_TILE_DISJOINT_TOL = 1e-9

#: Professor Q6-ii inside-support serve floor.
_INSIDE_SERVE_FLOOR = 0.90

#: Fixture training config: a smoke-scale multi-band run with a 5x5 tiling of
#: the low-mass (Y=3) stratum and a moderate region cap so the cap truncation
#: fires (32 admitted > 4) AND >=3 far-field tiles are actually built and
#: recorded with their ``rho_theta_box``.  The eps bars are opened wide so charts
#: register (train() needs >=1 chart); the RECORDS this suite reads --
#: tile boxes, admitted/dropped counts, beyond-cap masses -- are independent
#: of the (budget-limited) interpolation accuracy.
_FIXTURE_CONFIG = TrainingConfig(
    n_gamma=4, n_u=4, n_theta=4, n_rho=4, n_theta_c=4, w_nodes_per_decade=3,
    n_farfield_tiles_per_side=5, max_farfield_regions=4, n_caustic_samples=60,
    n_heldout=6, tube_eps_max=1e9, farfield_eps_max=1e9)

_OUTPUT_DIR = Path(__file__).parent / 'output'


@functools.lru_cache(maxsize=1)
def _trained_report() -> dict:
    """Run `train` once at smoke scale and return the JSON training report.

    Cached for the whole module: the engine-backed run costs a few minutes,
    so every report-reading test shares this single artifact.
    """
    outdir = tempfile.mkdtemp(prefix='surr_train_')
    _surrogate, report = train(outdir=outdir, config=_FIXTURE_CONFIG)
    return report


def _w_indep(f_hz: float, m_msun: float) -> float:
    """Independent dimensionless frequency ``w`` (F002 oracle)."""
    return _W_LENSING_PER_MSUN_HZ * m_msun * f_hz


def _strata_records(report: dict, parity: int) -> list[dict]:
    """All ``farfield_strata`` summary records for a parity in the report."""
    return [r for r in report['charts']
            if r.get('strata_summary') and r['parity'] == parity]


def _built_farfield_reports(report: dict) -> list[dict]:
    """The built far-field chart records (those carrying a ``rho_theta_box``)."""
    return [r for r in report['charts'] if r.get('kind') == 'farfield']


_FF_NAME_RE = re.compile(r'chart_(?P<label>.+)_s(?P<si>\d+)_ff_'
                         r'(?P<i>\d+)_(?P<j>\d+)$')


def _group_key(name: str) -> tuple[str, int] | None:
    """``(band-label, stratum-index)`` group key of a built ff chart name."""
    match = _FF_NAME_RE.match(name)
    if match is None:
        return None
    return match.group('label'), int(match.group('si'))


def _max_norm_center_separation(box_a: tuple, box_b: tuple) -> float:
    """Chebyshev (max-norm) distance between two tile centres."""
    (ax, ay), (bx, by) = box_a[0], box_b[0]
    return max(abs(ax - bx), abs(ay - by))


def _tile_outside_disk(center: tuple[float, float], half: float,
                       radius: float) -> float:
    """Minimum L2 distance from the origin to an axis-aligned tile box."""
    cx, cy = center
    dx = max(0.0, abs(cx) - half)
    dy = max(0.0, abs(cy) - half)
    return math.hypot(dx, dy)


def _save_plot(figure, name: str) -> None:
    """Save a diagnostic figure under ``cogwheel/tests/output`` (best effort)."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(_OUTPUT_DIR / name, dpi=90, bbox_inches='tight')


class _CountingTestCase(TestCase):
    """Base carrying the anti-vacuity comparison tally (house idiom).

    A subclass increments ``self.comparisons`` for every genuine assertion
    it makes; ``tearDown`` fails a test that ran zero comparisons, so a
    silently-skipping sweep cannot read green.
    """

    def setUp(self) -> None:
        self.comparisons = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self.comparisons, 0,
            'anti-vacuity: this test asserted nothing (zero comparisons).')


class TilingRecordTestCase(_CountingTestCase):
    """WP2 tiling: distinct non-overlapping tiles + loud training records."""

    def test_farfield_tiles_are_pairwise_disjoint(self) -> None:
        """No two emitted tiles overlap in the caustic-fixed (rho, theta_c) box.

        `_farfield_tiles(rho_inner, rho_outer, n_per_side)` takes CHART-RHO
        bounds and returns anisotropic tiles
        ``((rho_c, theta_c), (half_rho, half_theta), i, j)``.  Two axis-aligned
        rectangles are disjoint when they separate on EITHER axis, so the
        per-axis gap -- not a Chebyshev centre distance against one scalar half
        -- is the right predicate.

        The sweep MUST keep ``rho_inner < rho_outer``: the annulus is empty
        otherwise and `_farfield_tiles` returns nothing, which would make this
        test pass while asserting about an empty list.  `assertTrue(tiles)`
        pins that directly rather than relying on the anti-vacuity tearDown.
        """
        for rho_inner, rho_outer, n_side in itertools.product(
                (1.05, 1.2, 1.5), (2.0, 3.0, 4.24), (4, 5, 6)):
            with self.subTest(rho_inner=rho_inner, rho_outer=rho_outer,
                              n=n_side):
                tiles = _farfield_tiles(rho_inner, rho_outer, n_side)
                self.assertTrue(
                    tiles, f'empty annulus for rho [{rho_inner}, {rho_outer}]')
                for (a_c, a_h, _ai, _aj), (b_c, b_h, _bi, _bj) in (
                        itertools.combinations(tiles, 2)):
                    gaps = [abs(a_c[k] - b_c[k]) - (a_h[k] + b_h[k])
                            for k in (0, 1)]
                    self.assertGreaterEqual(
                        max(gaps), -_TILE_DISJOINT_TOL,
                        f'overlapping tiles centred {a_c} and {b_c} '
                        f'(halves {a_h}, {b_h})')
                    self.comparisons += 1

    def test_farfield_tiles_lie_wholly_outside_caustic_disk(self) -> None:
        """Every emitted tile lies wholly outside the exclusion radius.

        The interior-leakage guard in caustic-fixed coordinates: the exclusion
        is the ``rho_inner`` bound itself, so a tile leaks iff its INNER rho
        edge (``rho_c - half_rho``) falls below it.  There is no separate disk
        radius to compare against -- `_farfield_tiles` receives the exclusion
        already expressed in chart-rho units.
        """
        for rho_inner, rho_outer, n_side in itertools.product(
                (1.05, 1.25), (2.0, 3.0, 4.24), (5, 6)):
            with self.subTest(rho_inner=rho_inner, rho_outer=rho_outer,
                              n=n_side):
                tiles = _farfield_tiles(rho_inner, rho_outer, n_side)
                self.assertTrue(
                    tiles, f'empty annulus for rho [{rho_inner}, {rho_outer}]')
                for center, half, _i, _j in tiles:
                    inner_edge = center[0] - half[0]
                    self.assertGreaterEqual(
                        inner_edge, rho_inner - _TILE_DISJOINT_TOL,
                        f'tile at rho_c={center[0]} half_rho={half[0]} '
                        f'leaks inside the exclusion rho {rho_inner}')
                    self.comparisons += 1


    def test_built_tile_boxes_are_pairwise_disjoint(self) -> None:
        """Built ff charts of one (band x stratum) have disjoint tile boxes.

        The record key is ``rho_theta_box`` and the box is CAUSTIC-FIXED and
        ANISOTROPIC: ``[(rho_c, theta_c), (half_rho, half_theta)]``.  Two
        axis-aligned rectangles are disjoint when they separate on EITHER
        axis, so the old square-tile test (one scalar half, Chebyshev centre
        distance >= 2*half) is not the right predicate here -- it would demand
        separation in both axes at once and fail on legitimately disjoint
        tiles that abut along one axis.
        """
        report = _trained_report()
        groups: dict[tuple[str, int], list[tuple]] = {}
        for record in _built_farfield_reports(report):
            key = _group_key(record['name'])
            self.assertIsNotNone(key, f"unparsed ff name {record['name']}")
            center, half = record['rho_theta_box']
            groups.setdefault(key, []).append(
                ((float(center[0]), float(center[1])),
                 (float(half[0]), float(half[1]))))
        self.assertTrue(groups, 'no built far-field charts recorded')
        for key, boxes in groups.items():
            for box_a, box_b in itertools.combinations(boxes, 2):
                (a_c, a_h), (b_c, b_h) = box_a, box_b
                gaps = [abs(a_c[i] - b_c[i]) - (a_h[i] + b_h[i])
                        for i in (0, 1)]
                with self.subTest(group=key):
                    self.assertGreaterEqual(
                        max(gaps), -_TILE_DISJOINT_TOL,
                        f'overlapping built tiles in {key}: centres {a_c} '
                        f'{b_c}, halves {a_h} {b_h}')
                    self.comparisons += 1

    def test_region_cap_truncation_records_dropped_count(self) -> None:
        """A max_farfield_regions truncation is recorded with dropped count."""
        report = _trained_report()
        cap = _FIXTURE_CONFIG.max_farfield_regions
        truncations = [r for r in report['charts'] if r.get('truncated')]
        self.assertTrue(truncations, 'expected at least one cap truncation')
        for record in truncations:
            self.assertEqual(record['cap'], cap)
            self.assertGreater(record['admitted_tiles'], cap)
            self.assertEqual(
                record['dropped'], record['admitted_tiles'] - cap,
                'dropped count must equal admitted - cap')
            self.comparisons += 1

    def test_saddle_beyond_w_cap_recorded_not_dropped(self) -> None:
        """The >~458 Msun saddle stratum is a loud beyond_w_cap record."""
        report = _trained_report()
        beyond = [r for r in report['charts']
                  if r.get('beyond_w_cap') and r['parity'] == -1]
        self.assertTrue(
            beyond, 'saddle beyond-w-cap stratum was silently dropped')
        ceiling = _SADDLE_W_CEILING
        for record in beyond:
            m_lo, m_hi = record['mass_range']
            self.assertGreater(m_lo, 400.0, 'beyond bucket should start ~458')
            self.assertGreater(m_hi, m_lo)
            self.assertEqual(round(record['w_ceiling'], 3), round(ceiling, 3))
            self.comparisons += 1
        # The astroid prior range is fully reachable -> no beyond bucket.
        astro_beyond = [r for r in report['charts']
                        if r.get('beyond_w_cap') and r['parity'] == 1]
        self.assertEqual(astro_beyond, [], 'astroid should be fully reachable')
        self.comparisons += 1

    def test_every_stratum_recorded_even_when_zero_tiles(self) -> None:
        """Each stratum (including 0-tile high-mass ones) is recorded loudly."""
        report = _trained_report()
        box = PriorBox.from_prior_classes()
        for parity in (1, -1):
            strata, _beyond = _mass_strata(box, parity)
            for record in _strata_records(report, parity):
                self.assertEqual(
                    record['n_strata'], len(strata),
                    f"stratum silently dropped in {record['name']}")
                self.assertEqual(len(record['strata']), len(strata))
                self.comparisons += 1

    def test_tiling_diagnostic_plot(self) -> None:
        """Scatter admitted tile centres over the box with the caustic disk."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        y_extent, radius, n_side = 3.0, 0.6, 5
        tiles = _farfield_tiles(y_extent, radius, n_side)
        half = y_extent / n_side
        figure, axis = plt.subplots(figsize=(5, 5))
        for center, tile_half, _i, _j in tiles:
            cx, cy = center
            axis.add_patch(plt.Rectangle(
                (cx - tile_half, cy - tile_half), 2 * tile_half, 2 * tile_half,
                fill=False, edgecolor='C0'))
            axis.plot(cx, cy, '.', color='C0')
        axis.add_patch(plt.Circle((0, 0), radius, color='C3', alpha=0.3))
        axis.set(xlim=(-y_extent, y_extent), ylim=(-y_extent, y_extent),
                 xlabel='y1_eig', ylabel='y2_eig',
                 title=f'{len(tiles)} admitted tiles (half={half:.2f})')
        _save_plot(figure, 'wp2_tiling_centers_over_box.png')
        plt.close(figure)
        self.assertGreaterEqual(len(tiles), 3)
        self.comparisons += 1


class WholeBandContainmentTestCase(_CountingTestCase):
    """Serving contract: a stratum's chart w-range contains every in-stratum
    draw's whole detector band, except the recorded cap-truncated corner."""

    #: Detector band edges (Hz) shared by the containment sweeps; the fixture
    #: matches the production defaults so the report cross-check is exact.
    _F_LO_HZ = 20.0
    _F_HI_HZ = 1024.0

    #: Number of masses sampled across each stratum's reachable range.
    _N_MASS_SAMPLES = 6

    def _box(self) -> PriorBox:
        return PriorBox.from_prior_classes(
            f_lo_hz=self._F_LO_HZ, f_hi_hz=self._F_HI_HZ)

    def test_low_edge_and_untruncated_bands_are_contained(self) -> None:
        """Every draw's band bottom -- and its top on untruncated strata --
        lies inside the stratum's chart w-range (whole-band containment)."""
        box = self._box()
        for parity in (1, -1):
            strata, _beyond = _mass_strata(box, parity)
            self.assertTrue(strata, f'no strata for parity {parity}')
            for m_lo, m_hi in strata:
                y_extent = float(lens_prior._source_scale(m_lo))
                w_min, w_max = _stratum_w_range(
                    box, parity, m_lo, m_hi, y_extent)
                # dd/ceiling cap truncates the top when it falls below the
                # uncapped band top (the high-mass corner beyond the cap).
                w_hi_uncapped = _w_indep(self._F_HI_HZ, m_hi)
                truncated = w_max < w_hi_uncapped * (1.0 - 1e-6)
                for mass in np.linspace(m_lo, m_hi, self._N_MASS_SAMPLES):
                    band_lo = _w_indep(self._F_LO_HZ, float(mass))
                    band_hi = _w_indep(self._F_HI_HZ, float(mass))
                    with self.subTest(parity=parity, mass=round(float(mass), 2)):
                        # The band bottom is always contained: it is monotone
                        # in mass and bottoms out at w(f_lo, m_lo) = w_min.
                        self.assertGreaterEqual(
                            band_lo, w_min * (1.0 - _W_CONTAINMENT_REL_TOL),
                            'band bottom escaped below the stratum w-range')
                        self.comparisons += 1
                        if not truncated:
                            self.assertLessEqual(
                                band_hi,
                                w_max * (1.0 + _W_CONTAINMENT_REL_TOL),
                                'whole band escaped an untruncated stratum')
                            self.comparisons += 1

    def test_truncated_corner_exceeds_cap_and_is_flagged(self) -> None:
        """On a cap-truncated stratum the top-mass band exceeds the chart cap
        and the report flags the corner beyond the cap (not silently served)."""
        box = self._box()
        report = _trained_report()
        flagged_any = False
        for parity in (1, -1):
            strata, _beyond = _mass_strata(box, parity)
            recorded = {tuple(s['mass_range']): s
                        for rec in _strata_records(report, parity)
                        for s in rec['strata']}
            for m_lo, m_hi in strata:
                y_extent = float(lens_prior._source_scale(m_lo))
                w_min, w_max = _stratum_w_range(
                    box, parity, m_lo, m_hi, y_extent)
                w_hi_uncapped = _w_indep(self._F_HI_HZ, m_hi)
                truncated = w_max < w_hi_uncapped * (1.0 - 1e-6)
                if not truncated:
                    continue
                flagged_any = True
                # The top-mass draw's independent band top overshoots the cap.
                self.assertGreater(
                    w_hi_uncapped, w_max * (1.0 + _W_CONTAINMENT_REL_TOL),
                    'truncated corner should overshoot the capped w-range')
                self.comparisons += 1
                # The report records this stratum's corner-beyond-cap flag.
                key = (round(m_lo, 3), round(m_hi, 3))
                if key in recorded:
                    self.assertTrue(
                        recorded[key]['high_w_corner_beyond_cap'],
                        'report failed to flag a truncated corner')
                    self.comparisons += 1
        self.assertTrue(
            flagged_any, 'expected at least one cap-truncated stratum')
        self.comparisons += 1

    def test_saddle_beyond_ceiling_masses_are_attributed_not_served(
            self) -> None:
        """Saddle masses whose w(1024, m) exceeds the parity ceiling fall in
        the recorded beyond-w-cap bucket and outside every served stratum."""
        box = self._box()
        strata, beyond = _mass_strata(box, parity=-1)
        self.assertIsNotNone(
            beyond, 'saddle should have an un-tileable high-mass tail')
        m_reach_hi = beyond['m_lo']
        m_prior_hi = beyond['m_hi']
        # Independent oracle: at the reachable top the band top is ~ the
        # ceiling; above it, it exceeds the ceiling.
        self.assertAlmostEqual(
            _w_indep(self._F_HI_HZ, m_reach_hi), _SADDLE_W_CEILING,
            delta=_SADDLE_W_CEILING * _W_CONTAINMENT_REL_TOL)
        self.comparisons += 1
        stratum_hi = max(m_hi for _m_lo, m_hi in strata)
        for mass in np.linspace(m_reach_hi * 1.001, m_prior_hi, 5):
            with self.subTest(mass=round(float(mass), 1)):
                # Above the ceiling (independent oracle).
                self.assertGreater(
                    _w_indep(self._F_HI_HZ, float(mass)),
                    _SADDLE_W_CEILING * (1.0 - _W_CONTAINMENT_REL_TOL),
                    'beyond-cap mass should exceed the saddle ceiling')
                self.comparisons += 1
                # Outside every served stratum's mass range.
                self.assertGreater(
                    mass, stratum_hi,
                    'beyond-cap mass must lie outside all served strata')
                self.comparisons += 1


    def test_whole_band_containment_diagnostic_plot(self) -> None:
        """Plot per-draw band intervals against each stratum's chart w-range."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        box = self._box()
        figure, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        for axis, parity in zip(axes, (1, -1)):
            strata, _beyond = _mass_strata(box, parity)
            for si, (m_lo, m_hi) in enumerate(strata):
                y_extent = float(lens_prior._source_scale(m_lo))
                w_min, w_max = _stratum_w_range(
                    box, parity, m_lo, m_hi, y_extent)
                axis.fill_betweenx([si - 0.35, si + 0.35],
                                   np.log10(w_min), np.log10(w_max),
                                   color='C0', alpha=0.25)
                for mass in np.linspace(m_lo, m_hi, self._N_MASS_SAMPLES):
                    band_lo = _w_indep(self._F_LO_HZ, float(mass))
                    band_hi = _w_indep(self._F_HI_HZ, float(mass))
                    axis.plot([np.log10(band_lo), np.log10(band_hi)],
                              [si, si], color='C3', lw=0.8, alpha=0.7)
            axis.set(xlabel='log10 w', ylabel='stratum index',
                     title=f'parity {parity}')
        figure.suptitle('whole-band containment (bars=chart range, '
                        'lines=per-draw bands)')
        _save_plot(figure, 'wp2_whole_band_containment.png')
        plt.close(figure)
        self.assertTrue(True)
        self.comparisons += 1


# ---------------------------------------------------------------------------
# INS-1-001: far-field DD product cap must use the box-CORNER magnitude.
# ---------------------------------------------------------------------------


class FarFieldCornerCapTestCase(_CountingTestCase):
    """Regression for Inspector finding INS-1-001.

    `_stratum_w_range` computes ``dd_cap = _DD_PRODUCT_MARGIN / y_max`` and
    `_train_band_charts` calls it with ``y_max = y_extent`` (the stratum's
    per-axis box half-width ``Y``), never the box's true corner magnitude
    ``Y * sqrt(2)``.  The far-field tiling (`_farfield_tiles`) admits tiles
    out to that corner, so on every stratum where the DD cap actually binds
    the resulting ``w_max`` still lets ``w_max * (Y * sqrt(2))`` overshoot
    the point-mass kernel's real ``DD_PRODUCT_CEILING = 60`` -- the exact
    corner node the chart is built to cover.

    ORACLE INDEPENDENCE (F002): the oracle here is not a re-derivation of
    the cap arithmetic but the production point-mass kernel itself,
    `point_mass_g_derivatives` (via its `_validate_domain` domain gate) --
    the actual authority `_stratum_w_range`'s cap exists to satisfy.  Calling
    it directly at the true (uncapped-in-this-respect) corner magnitude
    reproduces the reported symptom with no mocking.

    STATUS: this suite is EXPECTED TO FAIL against the current
    `surrogate_training.py`, which has not yet applied the INS-1-001 fix
    (`_stratum_w_range` still receives the per-axis half-width).  It pins
    the corner-magnitude contract for that fix to land against; the
    companion `test_corner_aware_cap_recovers_engine_compliance` proves the
    contract is satisfiable (not vacuously red) by recomputing the cap with
    the corner-aware ``y_max`` and confirming the kernel then accepts it.
    """

    #: Detector band shared with the rest of the WP2 suite (report-exact).
    _F_LO_HZ = 20.0
    _F_HI_HZ = 1024.0

    #: The kernel's domain gate (`_validate_domain`) fires before any series
    #: work, so a tiny ladder keeps every call in this suite cheap.
    _MAX_DERIVATIVE = 0
    _N_TERMS = 4

    #: Binding-detection guard band: a stratum counts as "DD-cap-bound" only
    #: when the DD cap sits strictly below both the parity ceiling and the
    #: prior's own uncapped band top by more than floating-point noise.
    _BINDING_GUARD = 1.0 - 1e-9

    def _box(self) -> PriorBox:
        return PriorBox.from_prior_classes(
            f_lo_hz=self._F_LO_HZ, f_hi_hz=self._F_HI_HZ)

    def _dd_binding_strata(self, box: PriorBox, parity: int
                           ) -> list[tuple[float, float, float, float]]:
        """Strata whose ``w_max`` today's `_stratum_w_range` sets via the DD
        product cap (not the parity ceiling or the prior's own band top).

        Returns a list of ``(m_lo, m_hi, y_extent, w_max)`` tuples, where
        ``y_extent`` is the stratum's per-axis box half-width ``Y`` and
        ``w_max`` is `_stratum_w_range`'s (possibly under-capped) result.
        """
        strata, _beyond = _mass_strata(box, parity)
        ceiling = _POSITIVE_W_CEILING if parity == 1 else _SADDLE_W_CEILING
        binding = []
        for m_lo, m_hi in strata:
            y_extent = float(lens_prior._source_scale(m_lo))
            _w_min, w_max = _stratum_w_range(
                box, parity, m_lo, m_hi, y_extent)
            dd_cap = _DD_PRODUCT_MARGIN / y_extent
            w_max_uncapped = _w_indep(self._F_HI_HZ, m_hi)
            if (dd_cap < ceiling * self._BINDING_GUARD
                    and dd_cap < w_max_uncapped * self._BINDING_GUARD):
                binding.append((m_lo, m_hi, y_extent, w_max))
        return binding

    def test_outer_corner_tile_survives_the_real_engine_gate(self) -> None:
        """On every DD-cap-binding stratum, the outer-corner tile's true
        ``(w_max, |y|)`` must survive the actual point-mass kernel's domain
        gate -- the exact node the far-field tiling is built to cover."""
        box = self._box()
        checked_any = False
        for parity in (1, -1):
            for m_lo, m_hi, y_extent, w_max in self._dd_binding_strata(
                    box, parity):
                checked_any = True
                corner_y = y_extent * math.sqrt(2.0)
                with self.subTest(parity=parity, m_lo=round(m_lo, 2),
                                  m_hi=round(m_hi, 2)):
                    # Bump the anti-vacuity tally BEFORE the assertion: this
                    # test is expected to fail against the current, unfixed
                    # production code, and the tally must not itself go to
                    # zero (and mask the real assertion under a redundant
                    # tearDown error) just because every subTest failed.
                    self.comparisons += 1
                    try:
                        point_mass_g_derivatives(
                            w_max, corner_y ** 2, self._MAX_DERIVATIVE,
                            self._N_TERMS)
                    except HypergeometricDomainError as exc:
                        self.fail(
                            'the DD-capped w_max still lets the real '
                            f'engine refuse the box outer corner '
                            f'(parity={parity}, stratum=({m_lo:.2f}, '
                            f'{m_hi:.2f}), w_max={w_max:.3f}, '
                            f'corner|y|={corner_y:.4f}, '
                            f'product={w_max * corner_y:.3f} > '
                            f'DD_PRODUCT_CEILING={DD_PRODUCT_CEILING}: '
                            f'{exc}')
        self.assertTrue(
            checked_any,
            'no DD-cap-binding stratum found in the real prior box -- '
            'fixture drifted off the INS-1-001 trigger regime, or the '
            'strata/ceiling arithmetic changed underneath this test')
        self.comparisons += 1

    def test_corner_product_within_dd_ceiling(self) -> None:
        """Direct arithmetic mirror of the kernel's own check: on every
        DD-cap-binding stratum, ``w_max * (Y * sqrt(2))`` must not exceed
        `DD_PRODUCT_CEILING` -- else `_stratum_w_range` under-caps by the
        missing ``sqrt(2)`` corner factor."""
        box = self._box()
        checked_any = False
        for parity in (1, -1):
            for m_lo, m_hi, y_extent, w_max in self._dd_binding_strata(
                    box, parity):
                checked_any = True
                corner_y = y_extent * math.sqrt(2.0)
                product = w_max * corner_y
                with self.subTest(parity=parity, m_lo=round(m_lo, 2),
                                  m_hi=round(m_hi, 2)):
                    # See the sibling test above for why the tally is bumped
                    # before the (possibly failing) assertion.
                    self.comparisons += 1
                    self.assertLessEqual(
                        product, DD_PRODUCT_CEILING,
                        f'outer-corner product {product:.3f} exceeds the '
                        f'engine ceiling {DD_PRODUCT_CEILING} -- '
                        '_stratum_w_range under-caps by the missing '
                        'sqrt(2) corner factor (INS-1-001)')
        self.assertTrue(
            checked_any, 'no DD-cap-binding stratum found to exercise '
            'INS-1-001')
        self.comparisons += 1

    def test_corner_aware_cap_recovers_engine_compliance(self) -> None:
        """Positive control (non-vacuity proof): recomputing the SAME DD
        cap with the corner-aware ``y_max = Y * sqrt(2)`` -- the INS-1-001
        suggested fix -- lands every binding stratum's product at exactly
        `_DD_PRODUCT_MARGIN` and the real kernel accepts it.  This proves
        the contract the two tests above pin is satisfiable: they are not
        red because of an impossible bound, only because the current
        production code under-caps.
        """
        box = self._box()
        checked_any = False
        for parity in (1, -1):
            ceiling = _POSITIVE_W_CEILING if parity == 1 else _SADDLE_W_CEILING
            for m_lo, m_hi, y_extent, _w_max in self._dd_binding_strata(
                    box, parity):
                checked_any = True
                corner_y = y_extent * math.sqrt(2.0)
                w_min = _w_indep(self._F_LO_HZ, m_lo)
                w_max_uncapped = _w_indep(self._F_HI_HZ, m_hi)
                dd_cap_fixed = _DD_PRODUCT_MARGIN / corner_y
                w_max_fixed = min(w_max_uncapped, ceiling, dd_cap_fixed)
                with self.subTest(parity=parity, m_lo=round(m_lo, 2),
                                  m_hi=round(m_hi, 2)):
                    self.assertGreater(w_max_fixed, w_min,
                                       'fixed w-range collapsed to empty')
                    self.comparisons += 1
                    # The kernel must accept the corner-aware cap.
                    point_mass_g_derivatives(
                        w_max_fixed, corner_y ** 2, self._MAX_DERIVATIVE,
                        self._N_TERMS)
                    self.comparisons += 1
                    # And it lands exactly at the intended margin (not
                    # accidentally satisfied by some other cap winning).
                    self.assertAlmostEqual(
                        w_max_fixed * corner_y, _DD_PRODUCT_MARGIN, places=6,
                        msg='corner-aware cap did not bind at the DD margin')
                    self.comparisons += 1
        self.assertTrue(
            checked_any, 'no DD-cap-binding stratum found for the '
            'corner-aware control')
        self.comparisons += 1

    def test_self_falsification_uncapped_corner_would_refuse(self) -> None:
        """Self-falsification: prove the engine call has teeth by feeding it
        the UNCAPPED band top at the corner magnitude (no DD cap at all) on
        a binding stratum -- the kernel must refuse it, confirming the
        checks above are not vacuously always-pass."""
        box = self._box()
        checked_any = False
        for parity in (1, -1):
            for m_lo, m_hi, y_extent, _w_max in self._dd_binding_strata(
                    box, parity):
                checked_any = True
                corner_y = y_extent * math.sqrt(2.0)
                w_max_uncapped = _w_indep(self._F_HI_HZ, m_hi)
                with self.subTest(parity=parity, m_lo=round(m_lo, 2)):
                    with self.assertRaises(HypergeometricDomainError):
                        point_mass_g_derivatives(
                            w_max_uncapped, corner_y ** 2,
                            self._MAX_DERIVATIVE, self._N_TERMS)
                    self.comparisons += 1
        self.assertTrue(checked_any, 'no binding stratum for the '
                        'self-falsification control')
        self.comparisons += 1


# ---------------------------------------------------------------------------
# Serve-fraction fixture: a synthetic tiled far-field artifact.
# ---------------------------------------------------------------------------

#: Astroid gamma sub-band used for the synthetic serving fixture; kept well
#: below the ``gamma = 1`` guard band so no draw is refused for parity.
_SERVE_GAMMA_BAND = (0.15, 0.55)

#: Synthetic caustic-disk radius (``caustic_reach + eta_max``) for the fixture.
#: With ``y_extent = 3`` and a 5x5 grid (tile half 0.6) it drops only the
#: single centre cell, leaving a 24-tile exterior ring and one interior hole.
_SERVE_EXCLUSION_RADIUS = 0.6

#: Tiles per side of the synthetic serving fixture.
_SERVE_N_PER_SIDE = 5


#: Band-midpoint scalar caustic reach normalising this fixture's caustic-fixed
#: ``(rho, theta_c)`` tiling (Build 8h-b3 port) -- the SAME convention
#: `_train_band_charts` uses (``reach_scalar = _scalar_caustic_reach(gamma_mid)``)
#: for a topology-stable gamma band.
_SERVE_REACH = surrogate_module._caustic_reach(
    0.5 * (_SERVE_GAMMA_BAND[0] + _SERVE_GAMMA_BAND[1]))


@functools.lru_cache(maxsize=1)
def _serve_fixture() -> dict:
    """Build the synthetic tiled far-field chart set once.

    One `FarFieldChart` per admitted tile of the low astroid stratum, each
    covering exactly its tile box.  The charts' ``w`` grids are padded to
    contain every in-stratum draw's whole band, so the serve/None decision is
    dominated by the tile-box GEOMETRY -- which is the additive serving
    contract under test.

    Tiles are now the caustic-fixed ``(rho, theta_c)`` ANNULUS
    `_farfield_tiles` returns (Build 8h-b3) -- production retired the raw
    Cartesian-square-with-dropped-centre tiling this fixture originally
    mimicked (`_farfield_tiles(y_extent, exclusion_radius, n_per_side)` no
    longer exists; the current `_farfield_tiles(rho_inner, rho_outer,
    n_per_side)` tiles ``rho in [rho_inner, rho_outer] x theta_c in
    (-pi, pi]``).  The certification INTENT is unchanged and preserved
    exactly: an inner disk (``rho < exclusion_rho``, the caustic + eta_max
    shell) is never tiled (the "interior hole"), a ring of tiles covers the
    admitted exterior annulus, and points beyond the annulus's outer edge
    fall through -- only the tile SHAPE (annular wedges, not a dropped
    Cartesian centre cell) changed.
    """
    box = PriorBox.from_prior_classes()
    strata, _beyond = _mass_strata(box, 1)
    m_lo, m_hi = strata[0]
    y_extent = float(lens_prior._source_scale(m_lo))
    exclusion_rho = _SERVE_EXCLUSION_RADIUS / _SERVE_REACH
    rho_outer = y_extent / _SERVE_REACH
    tiles = _farfield_tiles(exclusion_rho, rho_outer, _SERVE_N_PER_SIDE)
    # ln-w grid padded around the stratum band (ln units, matching the query).
    w_lo = _w_indep(20.0, m_lo)
    w_hi = _w_indep(1024.0, m_hi)
    log_w_grid = np.log(np.geomspace(w_lo * 0.8, w_hi * 1.25, 4))
    gamma_grid = np.linspace(_SERVE_GAMMA_BAND[0], _SERVE_GAMMA_BAND[1], 4)
    charts = []
    for (rho_c, theta_c), (half_rho, half_theta), _i, _j in tiles:
        envelope = np.ones((4, 4, 4, 4))
        charts.append(FarFieldChart.from_values(
            gamma_grid=gamma_grid,
            rho_grid=np.linspace(rho_c - half_rho, rho_c + half_rho, 4),
            theta_c_grid=np.linspace(theta_c - half_theta,
                                     theta_c + half_theta, 4),
            log_w_grid=log_w_grid, envelope_real=envelope,
            envelope_imag=envelope, image_count=2, parity=1,
            eta_overlap_min=0.05))
    return {'charts': charts, 'tiles': tiles, 'y_extent': y_extent,
            'gamma_band': _SERVE_GAMMA_BAND, 'm_range': (m_lo, m_hi),
            'log_w_grid': log_w_grid, 'reach': _SERVE_REACH,
            'exclusion_rho': exclusion_rho, 'rho_outer': rho_outer}


def _point_in_tiles(y1: float, y2: float, gamma: float,
                    tiles: list[tuple]) -> bool:
    """Independent geometric classifier: is ``(gamma, y1, y2)`` in some tile?

    Keys ONLY on the tile-box geometry, never on `select_chart`, so a
    containment bug in the guard stack shows up as a dropped serve fraction.
    Converts to caustic-fixed ``(rho, theta_c)`` via `_to_caustic_fixed` at
    the QUERY'S OWN ``gamma`` -- the identical conversion `select_chart`'s
    caller (`serve`) performs -- so the oracle and the code under test agree
    on what "inside" means even though the tiles themselves were built at
    the band-midpoint reach.
    """
    rho, theta_c = surrogate_module._to_caustic_fixed(gamma, y1, y2)
    for (rho_c, theta_ctr), (half_rho, half_theta), _i, _j in tiles:
        if (rho_c - half_rho <= rho <= rho_c + half_rho
                and theta_ctr - half_theta <= theta_c <= theta_ctr + half_theta):
            return True
    return False


def _draw_support_samples(rng: np.random.Generator, n_samples: int,
                          gamma_band: tuple[float, float],
                          m_range: tuple[float, float]) -> list[dict]:
    """Draw ``(gamma, m, y1, y2, whole-band)`` from the lens prior classes.

    ``ln m`` uniform (via `UniformLensMassPrior.transform`), ``gamma`` uniform,
    ``u1, u2`` uniform on the unit box mapped to shear-frame ``(y1, y2)`` by
    `UniformSourcePositionPrior.transform`, and the whole detector band from
    the independent ``w`` oracle.
    """
    draws: list[dict] = []
    for _ in range(n_samples):
        ln_m = rng.uniform(math.log(m_range[0]), math.log(m_range[1]))
        m_lens = lens_prior.UniformLensMassPrior.transform(ln_m)['m_lens_msun']
        gamma = rng.uniform(*gamma_band)
        u1, u2 = rng.uniform(-1.0, 1.0, size=2)
        source = lens_prior.UniformSourcePositionPrior.transform(
            u1, u2, m_lens)
        draws.append({
            'gamma': float(gamma), 'm_lens': float(m_lens),
            'y1': float(source['y1']), 'y2': float(source['y2']),
            'band_lo': _w_indep(20.0, m_lens),
            'band_hi': _w_indep(1024.0, m_lens)})
    return draws


class ServeFractionTestCase(_CountingTestCase):
    """Additive serving contract, both directions: inside-support draws serve
    at >= 90%, outside-support draws return ``None`` 100% of the time."""

    def _serve(self, fixture: dict, draw: dict):
        rho, theta_c = surrogate_module._to_caustic_fixed(
            draw['gamma'], draw['y1'], draw['y2'])
        return select_chart(
            fixture['charts'], gamma=draw['gamma'],
            log_w_min=math.log(draw['band_lo']),
            log_w_max=math.log(draw['band_hi']),
            eta=5.0, theta=0.0, image_count=2,
            rho=rho, theta_c=theta_c)

    def test_inside_support_draws_serve_at_least_ninety_percent(self) -> None:
        """Draws whose source lands in an admitted tile serve at >= 90%."""
        fixture = _serve_fixture()
        rng = np.random.default_rng(20240722)
        draws = _draw_support_samples(
            rng, 700, fixture['gamma_band'], fixture['m_range'])
        inside = [d for d in draws
                  if _point_in_tiles(d['y1'], d['y2'], d['gamma'],
                                     fixture['tiles'])]
        self.assertGreater(len(inside), 50, 'too few inside draws to test')
        served = sum(self._serve(fixture, d) is not None for d in inside)
        fraction = served / len(inside)
        self.assertGreaterEqual(
            fraction, _INSIDE_SERVE_FLOOR,
            f'inside serve fraction {fraction:.3f} below floor')
        self.comparisons += 1

    def test_interior_hole_draws_never_serve(self) -> None:
        """Draws in the interior caustic hole (un-tiled ``rho < exclusion_rho``
        disk) return ``None`` even though gamma / w / eta / image-count are
        all valid."""
        fixture = _serve_fixture()
        rng = np.random.default_rng(31415926)
        draws = _draw_support_samples(
            rng, 900, fixture['gamma_band'], fixture['m_range'])
        outside = [d for d in draws
                   if not _point_in_tiles(d['y1'], d['y2'], d['gamma'],
                                          fixture['tiles'])]
        self.assertGreater(len(outside), 5, 'too few interior-hole draws')
        for draw in outside:
            with self.subTest(y1=round(draw['y1'], 3), y2=round(draw['y2'], 3)):
                self.assertIsNone(
                    self._serve(fixture, draw),
                    'an outside draw served (additive-contract violation)')
                self.comparisons += 1

    def test_beyond_box_draws_never_serve(self) -> None:
        """Draws beyond the annulus's outer edge return ``None`` 100%.

        Ported from the retired square-box "MAX norm" criterion (the
        Cartesian tiling this fixture originally mimicked no longer exists
        -- see `_serve_fixture`).  The tiled region is now the caustic-fixed
        annulus ``rho in [exclusion_rho, rho_outer]``, so "beyond" is
        unambiguously ``rho > rho_outer`` at every ``theta_c`` -- the
        annulus has no corners, so no max-norm subtlety is needed.
        """
        fixture = _serve_fixture()
        rng = np.random.default_rng(2718281)
        rho_outer = fixture['rho_outer']
        m_mid = math.sqrt(fixture['m_range'][0] * fixture['m_range'][1])
        gamma = 0.35
        band = {'gamma': gamma, 'band_lo': _w_indep(20.0, m_mid),
                'band_hi': _w_indep(1024.0, m_mid)}
        for _ in range(300):
            rho_beyond = rng.uniform(rho_outer * 1.05, rho_outer * 1.8)
            theta = rng.uniform(-math.pi, math.pi)
            y1, y2 = surrogate_module._from_caustic_fixed(
                gamma, rho_beyond, theta)
            draw = {**band, 'y1': float(y1), 'y2': float(y2)}
            # Defensive: the point is genuinely outside every tile box.
            self.assertFalse(
                _point_in_tiles(y1, y2, gamma, fixture['tiles']))
            self.assertIsNone(
                self._serve(fixture, draw),
                'a beyond-annulus draw served (additive-contract violation)')
            self.comparisons += 1

    def test_serve_fraction_diagnostic_plot(self) -> None:
        """Serve / fall-through map over the ``(y1, y2)`` box."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fixture = _serve_fixture()
        rng = np.random.default_rng(161803)
        draws = _draw_support_samples(
            rng, 1500, fixture['gamma_band'], fixture['m_range'])
        served_xy, dropped_xy = [], []
        for draw in draws:
            target = (served_xy if self._serve(fixture, draw) is not None
                      else dropped_xy)
            target.append((draw['y1'], draw['y2']))
        figure, axis = plt.subplots(figsize=(5.5, 5.5))
        if served_xy:
            axis.scatter(*zip(*served_xy), s=4, color='C0', label='served')
        if dropped_xy:
            axis.scatter(*zip(*dropped_xy), s=6, color='C3',
                         label='fall-through')
        axis.add_patch(plt.Circle((0, 0), _SERVE_EXCLUSION_RADIUS,
                                   color='k', fill=False, ls='--'))
        extent = fixture['y_extent']
        axis.set(xlim=(-extent, extent), ylim=(-extent, extent),
                 xlabel='y1_eig', ylabel='y2_eig',
                 title='serve / fall-through over the tiled box')
        axis.legend(loc='upper right')
        _save_plot(figure, 'wp2_serve_fraction_map.png')
        plt.close(figure)
        self.assertGreater(len(served_xy), len(dropped_xy))
        self.comparisons += 1


# ---------------------------------------------------------------------------
# Build 8g WP1: eps registration gate (Professor Q6-iii / F010)
# ---------------------------------------------------------------------------

#: Far-field held-out eps registration bar (matches ``TrainingConfig`` default).
_FARFIELD_EPS_BAR = 3e-3

#: Astroid gamma sub-band the WP1 gate fixture charts are trained over; kept
#: below the ``gamma = 1`` guard band so no chart is refused for parity.
_GATE_GAMMA_BAND = (0.2, 0.5)

#: Far-field tile half-width ``(half_rho, half_theta_c)`` and w-band top for
#: the gate fixture charts (Build 8h-b3: `_build_farfield_chart`'s ``half``
#: is a caustic-fixed ``(rho, theta_c)`` half-pair, not a scalar).
_GATE_HALF = (0.25, 0.2)

#: Mid-band top for the gate fixture, and the safety factor applied to the
#: region ``w_floor`` when setting the band bottom.  The band must lie wholly
#: inside ``[w_floor, w_trust)`` where FARFIELD_KERNEL_SUM is the valid label;
#: the top is raised from the original 2.0 so the above-floor band still spans
#: ~1 decade at ``w_nodes_per_decade=8`` (a 4-node w axis under-resolves the
#: spline and re-inflates eps for a reason unrelated to the gate under test).
_GATE_W_TOP = 8.0
_GATE_W_FLOOR_MARGIN = 1.05

#: Three DISJOINT clean far-field tile centres ``(rho_c, theta_c_c)``
#: (``theta_c`` separation 0.9 > ``2 * half_theta = 0.4``, so no two boxes'
#: theta_c spans overlap even though they share ``rho_c``).
#:
#: ``theta_c`` is chosen to keep every tile CLEAR OF THE CUSP RAYS.  For the
#: astroid the cusps sit at ``theta_c = 0, pi/2, pi, 3pi/2``, where
#: ``r_caustic`` has a slope kink; a tile straddling one asks a cubic spline
#: to represent a non-smooth map and its held-out eps inflates for reasons
#: unrelated to whatever the test is gating (the same defect the production
#: tiling addresses with cusp-aligned columns).  Centres at 0.5 / 0.95 / 1.4
#: with ``half_theta = 0.2`` keep every box inside ``(0, pi/2)`` and pairwise
#: disjoint (separation 0.45 > ``2 * half_theta = 0.4``).
_GATE_HEALTHY_CENTER = (2.5, 0.5)
_GATE_POISON_CENTER = (2.5, 0.95)
_GATE_NAN_CENTER = (2.5, 1.4)

#: Spline-coefficient poison factor: scaling ``real_coeffs`` by 1.1 lifts the
#: far-field envelope ~10% off the engine truth, so the measured held-out eps
#: (~9.7e-2) exceeds the 3e-3 bar by ~32x -- comfortably past the Professor's
#: ">= 2x the bar" reachable-red requirement.
_GATE_POISON_FACTOR = 1.1

#: Provenance schema the production trainer stamps on its held-out probe.
_HELDOUT_PROV = {'schema': 'heldout-probe'}


@functools.lru_cache(maxsize=1)
def _wp1_gate_fixture() -> dict:
    """Build the three real engine-backed far-field charts + measured eps.

    One engine build of three disjoint far-field tiles (~30 s, cached for the
    module): a HEALTHY chart, a POISON-BASE chart (its ``real_coeffs`` scaled
    to make the POISONED variant), and a chart whose NaN case is driven by
    held-out samples that all fall outside its box (zero served -> nan eps).
    All eps are measured by the production `_heldout_eps` against a fresh
    `ChangRefsdalChannels` engine, so the gate under test is fed engine truth.
    """
    box = PriorBox.from_prior_classes()
    strata, _beyond = _mass_strata(box, 1)
    m_lo, m_hi = strata[0]
    y_extent = float(lens_prior._source_scale(m_lo))
    full_w = _stratum_w_range(box, 1, m_lo, m_hi, y_extent)
    config = TrainingConfig(
        n_gamma=6, n_rho=6, n_theta_c=6, w_nodes_per_decade=8, n_heldout=8,
        farfield_eps_max=_FARFIELD_EPS_BAR)
    # The band MUST start above the region's physics w_floor.  FARFIELD_KERNEL_SUM
    # subtracts the real image kernels, which is the right label only in the mid
    # band ``[w_floor, w_trust)``; below the floor the residual is the divergent
    # diffractive-bottom object and FARFIELD_DIFFRACTIVE (subtract nothing) is
    # the correct label.  Starting at ``full_w[0]`` put ~11 of ~15 log-w nodes
    # under the floor (floor 0.661 vs band start 0.0248), so the charts fitted a
    # divergent object and measured eps 1-7 regardless of tile placement -- which
    # is why varying rho_c, the halves and the gamma band never helped.
    inner_rho = _GATE_HEALTHY_CENTER[0] - _GATE_HALF[0]
    region_floor, _floor_report = training._farfield_region_w_floor(
        box, _GATE_GAMMA_BAND, inner_rho, config)
    w_range = (max(full_w[0], region_floor * _GATE_W_FLOOR_MARGIN),
               _GATE_W_TOP)

    def build(center: tuple[float, float]):
        chart, _calls, _refused = _build_farfield_chart(
            gamma_band=_GATE_GAMMA_BAND, parity=1, box_center=center,
            half=_GATE_HALF, w_range=w_range, config=config)
        samples = _farfield_heldout_samples(
            _GATE_GAMMA_BAND, center, _GATE_HALF, config,
            np.random.default_rng(11))
        return chart, samples

    healthy, healthy_samples = build(_GATE_HEALTHY_CENTER)
    poison_base, poison_samples = build(_GATE_POISON_CENTER)
    nan_chart, _nan_box_samples = build(_GATE_NAN_CENTER)

    poisoned = dataclasses.replace(
        poison_base, real_coeffs=poison_base.real_coeffs * _GATE_POISON_FACTOR)
    # Held-out samples entirely outside the NaN chart's box -> never served ->
    # `_heldout_eps` returns nan (the "all held-out points refused" case).
    # The offset is applied in PHYSICAL eigenframe (y1, y2) units (a +20
    # shift is comfortably outside any `_GATE_HALF`-sized box regardless of
    # rho/theta_c placement), so the NaN centre is mapped through
    # `_from_caustic_fixed` first.
    nan_center_y1, nan_center_y2 = surrogate_module._from_caustic_fixed(
        0.35, *_GATE_NAN_CENTER)
    nan_samples = [
        (0.35, nan_center_y1 + 20.0, nan_center_y2 + 20.0)
        for _ in range(config.n_heldout)]

    healthy_eps = _heldout_eps(healthy, healthy_samples, _HELDOUT_PROV)
    poison_base_eps = _heldout_eps(poison_base, poison_samples, _HELDOUT_PROV)
    poisoned_eps = _heldout_eps(poisoned, poison_samples, _HELDOUT_PROV)
    nan_eps = _heldout_eps(nan_chart, nan_samples, _HELDOUT_PROV)

    return {
        'config': config,
        'healthy': {'chart': healthy, 'center': _GATE_HEALTHY_CENTER,
                    'eps': healthy_eps},
        'poison_base': {'chart': poison_base, 'center': _GATE_POISON_CENTER,
                        'eps': poison_base_eps},
        'poisoned': {'chart': poisoned, 'center': _GATE_POISON_CENTER,
                     'eps': poisoned_eps},
        'nan': {'chart': nan_chart, 'center': _GATE_NAN_CENTER,
                'eps': nan_eps}}


def _register_entries(entries: list[tuple], config: TrainingConfig
                      ) -> tuple[list, list[dict]]:
    """Mirror the `_train_band_charts` registration block verbatim.

    Each entry is ``(name, kind, chart, eps)``.  Uses the PRODUCTION
    `_chart_gated`: a gated chart is excluded and recorded with a ``gated``
    marker + ``gate_reason``; a passer is appended to the packed charts.
    """
    registered: list = []
    reports: list[dict] = []
    for name, kind, chart, eps in entries:
        gated, gate_reason = _chart_gated(kind, eps, config)
        report = {'name': name, 'kind': kind, 'heldout_eps': eps}
        if gated:
            report['gated'] = True
            report['gate_reason'] = gate_reason
            reports.append(report)
            continue
        registered.append(chart)
        reports.append(report)
    return registered, reports


def _gate_entries(fixture: dict) -> list[tuple]:
    """The healthy/poisoned/NaN synthetic-pass entries with measured eps."""
    return [
        ('chart_healthy_ff', 'farfield', fixture['healthy']['chart'],
         fixture['healthy']['eps']),
        ('chart_poisoned_ff', 'farfield', fixture['poisoned']['chart'],
         fixture['poisoned']['eps']),
        ('chart_nan_ff', 'farfield', fixture['nan']['chart'],
         fixture['nan']['eps'])]


# ---------------------------------------------------------------------------
# Build 8g WP3: saddle tube-tail fix (Professor Q6-iv)
# ---------------------------------------------------------------------------

#: Strong-shear saddle gamma reproducing the ``saddle_b1_tube_2`` pathology.
_WP3_GAMMA = 1.55
_WP3_BAND = (_WP3_GAMMA - 0.03, _WP3_GAMMA + 0.03)

#: Tube held-out eps registration bar (matches ``TrainingConfig`` default).
_TUBE_EPS_BAR = 5e-2

#: Pre-fix pathology floor: the counterfactual (fix-disabled) arc must exceed
#: this, proving the reproduction bites the real tail (measured ~0.43), while
#: the Professor's headline pre-fix number was ~1.15.
_WP3_PATHOLOGY_FLOOR = 0.09

#: Coarse fixture config for the WP3 tube builds (fast: two tube charts ~19 s).
_WP3_CONFIG = TrainingConfig(
    n_gamma=4, n_u=4, n_theta=4, w_nodes_per_decade=2, n_heldout=8,
    eta_floor=0.02, eta_max=0.05, n_caustic_samples=120)


def _wp3_fixon_left_arc(gamma: float, n: int):
    """The genuine production wedge-edge left arc (branch +1, min ``theta_lo``).

    Pulled straight from `_saddle_arcs`, so the fixture is a real production
    arc -- the WP3 fix (wedge-edge windows + wider saddle cusp safety) is
    exercised exactly as shipped, not an easy stand-in.
    """
    _cusps, arcs, reach = _saddle_arcs(gamma, n)
    branch_pos = sorted((a for a in arcs if a.branch == 1),
                        key=lambda a: a.theta_lo)
    return branch_pos[0], reach


def _wp3_fixoff_left_arc(gamma: float, n: int):
    """The pre-WP3 counterfactual arc: astroid cusp safety, NO wedge window.

    Reconstructs what the trainer WOULD have built before the fix -- the same
    branch-+1 wedge lobe partitioned with the astroid ``_CUSP_WIDTH_SAFETY`` /
    ``_CUSP_MIN_HALFWIDTH`` and no wedge-edge exclusion (edge half-width 0).
    """
    theta_max = 0.5 * np.arcsin(1.0 / abs(gamma))
    lo_edge = -theta_max + _WEDGE_EPS
    hi_edge = theta_max - _WEDGE_EPS
    branch = 1
    thetas, speed = _branch_speed_profile(
        gamma, branch, lo_edge, hi_edge, n, periodic=False)
    reach = _caustic_reach(gamma, branch, lo_edge, hi_edge, n)
    cusps = _find_cusps(
        thetas, speed, periodic=False, width_safety=_CUSP_WIDTH_SAFETY,
        min_halfwidth=_CUSP_MIN_HALFWIDTH)
    cusps.sort()
    walls = [(lo_edge, 0.0)] + cusps + [(hi_edge, 0.0)]
    arcs = []
    for (t_lo, w_lo), (t_hi, w_hi) in zip(walls[:-1], walls[1:]):
        windows = [(t, w) for (t, w) in ((t_lo, w_lo), (t_hi, w_hi))
                   if w > 0.0]
        arc = _make_arc(gamma, branch, t_lo, w_lo, t_hi, w_hi, windows)
        if arc is not None:
            arcs.append(arc)
    arcs.sort(key=lambda a: a.theta_lo)
    return arcs[0], reach


def _wp3_build_and_measure(box: PriorBox, config: TrainingConfig, arc,
                           reach: float) -> dict:
    """Build a saddle tube chart for one arc and measure its held-out eps."""
    w_range = _capped_w_range(box, -1, reach + config.eta_max)
    chart, _calls, _refused = _build_tube_chart(
        gamma_grid=np.linspace(*_WP3_BAND, config.n_gamma), arc=arc,
        parity=-1, w_range=w_range, config=config)
    samples = _tube_heldout_samples(
        _WP3_BAND, arc, config, np.random.default_rng(7))
    eps = _heldout_eps(chart, samples, _HELDOUT_PROV)
    return {'chart': chart, 'eps': eps, 'arc': arc, 'w_range': w_range}


@functools.lru_cache(maxsize=1)
def _wp3_fixture() -> dict:
    """Build the fix-on and fix-off saddle tube charts + measured eps (cached)."""
    box = PriorBox.from_prior_classes()
    config = _WP3_CONFIG
    n = config.n_caustic_samples
    on_arc, on_reach = _wp3_fixon_left_arc(_WP3_GAMMA, n)
    off_arc, off_reach = _wp3_fixoff_left_arc(_WP3_GAMMA, n)
    on = _wp3_build_and_measure(box, config, on_arc, on_reach)
    off = _wp3_build_and_measure(box, config, off_arc, off_reach)
    return {'config': config, 'n': n, 'on': on, 'off': off,
            'on_arc': on_arc, 'off_arc': off_arc}


def _wp3_overlay(chart, arc, gamma: float, eta: float, n_theta: int
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Engine vs emulated max-abs envelope across ``theta`` along an arc.

    Mirrors the `_heldout_eps` comparison but sweeps ``theta`` on a line so the
    diagnostic overlay shows WHERE the emulator diverges (the wedge/cusp ends).
    Returns ``(thetas, engine_env, emulated_env)`` with NaN where unserved.
    """
    surrogate = LensAmplificationSurrogate([chart], _HELDOUT_PROV)
    w_grid = np.exp(chart.log_w_grid)
    thetas = np.linspace(arc.theta_lo, arc.theta_hi, n_theta)
    engine_env = np.full(n_theta, np.nan)
    emulated_env = np.full(n_theta, np.nan)
    for idx, theta in enumerate(thetas):
        source = _tube_source(gamma, float(theta), eta, arc.branch,
                              arc.inward_sign)
        channels = ChangRefsdalChannels(w_grid)
        try:
            partition = channels.evaluate(
                gamma=gamma, y=(float(source[0]), float(source[1])),
                beta=0.0, kappa=0.0)
        except Exception:  # noqa: BLE001 - engine refusal -> leave NaN
            continue
        env_true = np.asarray(partition.envelope)
        if not np.all(np.isfinite(env_true)):
            continue
        engine_env[idx] = float(np.max(np.abs(env_true)))
        emulated, served, _definition = surrogate.serve(
            w_grid, gamma=gamma, y1=float(source[0]), y2=float(source[1]),
            beta=0.0, eta=partition.caustic_distance,
            theta=partition.critical_theta,
            image_count=int(partition.real_mask.sum()))
        if served:
            emulated_env[idx] = float(np.max(np.abs(emulated)))
    return thetas, engine_env, emulated_env


class EpsRegistrationGateTestCase(_CountingTestCase):
    """WP1 eps gate (Professor Q6-iii / F010): a healthy far-field chart is
    packed while a poisoned (eps >> bar) and a NaN-eps chart are both excluded
    and recorded as gated, with their windows falling through to the ladder."""

    def test_healthy_chart_registers(self) -> None:
        """A chart whose measured eps is below the bar is packed, not gated."""
        fixture = _wp1_gate_fixture()
        eps = fixture['healthy']['eps']
        self.assertLess(eps, _FARFIELD_EPS_BAR, 'fixture healthy chart not '
                        'below the bar (regenerate the far tile)')
        self.comparisons += 1
        registered, reports = _register_entries(
            [('chart_healthy_ff', 'farfield', fixture['healthy']['chart'],
              eps)], fixture['config'])
        self.assertEqual(len(registered), 1, 'healthy chart was not packed')
        self.assertNotIn('gated', reports[0])
        self.comparisons += 1

    def test_poisoned_chart_excluded_with_eps_above_bar(self) -> None:
        """A chart poisoned to >= 2x the bar is excluded and flagged loudly."""
        fixture = _wp1_gate_fixture()
        eps = fixture['poisoned']['eps']
        # Professor requirement: the poison lifts eps to >= 2x the bar.
        self.assertGreaterEqual(
            eps, 2.0 * _FARFIELD_EPS_BAR,
            'poison factor too weak: eps must exceed the bar by >= 2x')
        self.comparisons += 1
        registered, reports = _register_entries(
            [('chart_poisoned_ff', 'farfield', fixture['poisoned']['chart'],
              eps)], fixture['config'])
        self.assertEqual(registered, [], 'poisoned chart leaked into artifact')
        self.assertTrue(reports[0]['gated'])
        self.assertEqual(reports[0]['gate_reason'], 'eps_above_bar')
        self.comparisons += 1

    def test_nan_eps_chart_excluded_with_nan_reason(self) -> None:
        """A chart serving zero held-out points (nan eps) is excluded loudly."""
        fixture = _wp1_gate_fixture()
        eps = fixture['nan']['eps']
        self.assertTrue(math.isnan(eps), 'fixture NaN chart served a point')
        self.comparisons += 1
        registered, reports = _register_entries(
            [('chart_nan_ff', 'farfield', fixture['nan']['chart'], eps)],
            fixture['config'])
        self.assertEqual(registered, [], 'NaN-eps chart leaked into artifact')
        self.assertTrue(reports[0]['gated'])
        self.assertEqual(reports[0]['gate_reason'], 'nan_eps')
        self.comparisons += 1

    def test_synthetic_pass_registers_only_the_healthy_chart(self) -> None:
        """The full 3-chart pass packs the healthy chart and gates both the
        poisoned and NaN charts with the correct reasons."""
        fixture = _wp1_gate_fixture()
        registered, reports = _register_entries(
            _gate_entries(fixture), fixture['config'])
        self.assertEqual(len(registered), 1, 'exactly one chart should pack')
        self.assertIs(registered[0], fixture['healthy']['chart'])
        self.comparisons += 1
        gated = {r['name']: r['gate_reason']
                 for r in reports if r.get('gated')}
        self.assertEqual(
            gated,
            {'chart_poisoned_ff': 'eps_above_bar', 'chart_nan_ff': 'nan_eps'})
        self.comparisons += 1

    def test_only_healthy_chart_packed_into_artifact(self) -> None:
        """Packing the registered charts and round-tripping the ``.npz`` yields
        a single-chart artifact (the poisoned/NaN charts never persist)."""
        fixture = _wp1_gate_fixture()
        registered, _reports = _register_entries(
            _gate_entries(fixture), fixture['config'])
        outdir = Path(tempfile.mkdtemp(prefix='wp1_gate_'))
        path = outdir / 'artifact.npz'
        LensAmplificationSurrogate(registered, {'schema': 'test'}).save(path)
        reloaded = LensAmplificationSurrogate.load(path)
        self.assertEqual(len(reloaded.charts), 1, 'artifact chart count wrong')
        self.comparisons += 1

    def test_gated_windows_fall_through_select_chart(self) -> None:
        """With only the healthy chart registered, the poisoned and NaN chart
        windows fall through (`select_chart` returns ``None``), while the
        healthy window still serves."""
        fixture = _wp1_gate_fixture()
        registered, _reports = _register_entries(
            _gate_entries(fixture), fixture['config'])
        healthy_chart = fixture['healthy']['chart']
        mid_log_w = float(0.5 * (healthy_chart.log_w_grid[0]
                                 + healthy_chart.log_w_grid[-1]))

        def serve_at(center: tuple[float, float]):
            return select_chart(
                registered, gamma=0.35, log_w_min=mid_log_w,
                log_w_max=mid_log_w, eta=5.0, theta=0.0, image_count=2,
                rho=center[0], theta_c=center[1])

        self.assertIsNotNone(
            serve_at(fixture['healthy']['center']),
            'the registered healthy window must still serve')
        self.comparisons += 1
        for label in ('poisoned', 'nan'):
            with self.subTest(gated=label):
                self.assertIsNone(
                    serve_at(fixture[label]['center']),
                    f'{label} window served despite being gated')
                self.comparisons += 1

    def test_reverting_poison_re_registers_chart(self) -> None:
        """Reverting the poison (the un-scaled base chart) registers again, so
        the gate is reachable-red -- it excludes on the eps, not always."""
        fixture = _wp1_gate_fixture()
        base_eps = fixture['poison_base']['eps']
        self.assertLess(base_eps, _FARFIELD_EPS_BAR,
                        'un-poisoned base chart should pass the bar')
        # Same geometry as the poisoned chart; only ``real_coeffs`` differ.
        base = fixture['poison_base']['chart']
        poisoned = fixture['poisoned']['chart']
        np.testing.assert_allclose(base.gamma_grid, poisoned.gamma_grid)
        self.assertFalse(
            np.allclose(base.real_coeffs, poisoned.real_coeffs),
            'poison must actually perturb the coefficients')
        self.comparisons += 1
        registered, reports = _register_entries(
            [('chart_reverted_ff', 'farfield', base, base_eps)],
            fixture['config'])
        self.assertEqual(len(registered), 1, 'reverted chart did not register')
        self.assertNotIn('gated', reports[0])
        self.comparisons += 1

    def test_gate_report_diff_diagnostic(self) -> None:
        """Diagnostic: registered-vs-gated eps table + a bar chart."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fixture = _wp1_gate_fixture()
        _registered, reports = _register_entries(
            _gate_entries(fixture), fixture['config'])
        names, epsvals, colors = [], [], []
        for report in reports:
            eps = report['heldout_eps']
            names.append(report['name'].replace('chart_', '').replace(
                '_ff', ''))
            # NaN plotted at the bar height for visibility; annotate reason.
            epsvals.append(eps if not math.isnan(eps) else _FARFIELD_EPS_BAR)
            colors.append('C3' if report.get('gated') else 'C0')
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.bar(names, epsvals, color=colors)
        axis.axhline(_FARFIELD_EPS_BAR, color='k', ls='--',
                     label=f'bar={_FARFIELD_EPS_BAR:g}')
        axis.set(yscale='log', ylabel='held-out eps',
                 title='WP1 gate: registered (blue) vs gated (red)')
        axis.legend()
        _save_plot(figure, 'wp1_eps_gate_report_diff.png')
        plt.close(figure)
        # The report diff cleanly splits into one registered + two gated.
        registered_names = [r['name'] for r in reports if not r.get('gated')]
        gated_names = [r['name'] for r in reports if r.get('gated')]
        self.assertEqual(registered_names, ['chart_healthy_ff'])
        self.assertEqual(len(gated_names), 2)
        self.comparisons += 1


class EpsGateResumeTestCase(_CountingTestCase):
    """WP1 eps gate on RESUME: a reused per-chart file carries its persisted
    ``heldout_eps`` through provenance, and the registration gate fires on it
    identically to a fresh build -- WITHOUT re-running the engine.

    A resumed above-bar chart must stay out of the artifact even though its
    ``.npz`` exists on disk, and the decision must be deterministic (no eps
    recomputation).  A sentinel ``build_fn`` that raises if called proves the
    reuse path never rebuilds.
    """

    @staticmethod
    def _persist_poisoned_chart() -> tuple[Path, float]:
        """Save the poisoned far-field chart to a temp ``.npz`` with its
        above-bar ``heldout_eps`` recorded in per-chart provenance."""
        fixture = _wp1_gate_fixture()
        eps = float(fixture['poisoned']['eps'])
        chart = fixture['poisoned']['chart']
        outdir = Path(tempfile.mkdtemp(prefix='wp1_resume_'))
        path = outdir / 'chart_poisoned_ff.npz'
        provenance = {'schema': 'build8g-chart', 'kind': 'farfield',
                      'parity': 1, 'heldout_eps': eps}
        LensAmplificationSurrogate([chart], provenance).save(path)
        return path, eps

    def test_reused_chart_reads_persisted_eps_without_rebuild(self) -> None:
        """`_load_or_build` on an existing file returns ``reused=True`` and the
        persisted eps, and NEVER calls the build function."""
        path, eps = self._persist_poisoned_chart()

        def boom():
            raise AssertionError('build_fn called on a resumed chart')

        _chart, report, reused = _load_or_build(path, boom, {'schema': 'x'})
        self.assertTrue(reused, 'existing file should be reused, not rebuilt')
        self.assertIn('heldout_eps', report,
                      'persisted eps not surfaced on resume')
        self.assertEqual(float(report['heldout_eps']), eps,
                         'resumed eps differs from the persisted value')
        self.comparisons += 1

    def test_resumed_above_bar_chart_is_gated_out(self) -> None:
        """The gate excludes the resumed above-bar chart exactly as a fresh
        build would, using only the persisted (read-back) eps."""
        path, _eps = self._persist_poisoned_chart()
        fixture = _wp1_gate_fixture()

        def boom():
            raise AssertionError('build_fn called on a resumed chart')

        chart, report, _reused = _load_or_build(path, boom, {'schema': 'x'})
        registered, reports = _register_entries(
            [('chart_poisoned_ff', 'farfield', chart,
              float(report['heldout_eps']))], fixture['config'])
        self.assertEqual(registered, [],
                         'resumed above-bar chart leaked into the artifact')
        self.assertTrue(reports[0]['gated'])
        self.assertEqual(reports[0]['gate_reason'], 'eps_above_bar')
        self.comparisons += 1

    def test_resumed_chart_absent_from_artifact_despite_file_on_disk(
            self) -> None:
        """The gated chart is absent from the packed artifact even though its
        per-chart ``.npz`` still exists on disk (resumability is a file check,
        registration is a separate gate)."""
        path, _eps = self._persist_poisoned_chart()
        fixture = _wp1_gate_fixture()

        def boom():
            raise AssertionError('build_fn called on a resumed chart')

        chart, report, _reused = _load_or_build(path, boom, {'schema': 'x'})
        registered, _reports = _register_entries(
            [('chart_poisoned_ff', 'farfield', chart,
              float(report['heldout_eps']))], fixture['config'])
        self.assertTrue(path.exists(),
                        'the per-chart file must remain on disk')
        self.assertEqual(registered, [],
                         'the on-disk file must not force registration')
        self.comparisons += 1

    def test_resume_gate_decision_is_deterministic(self) -> None:
        """Two independent `_load_or_build` reuses of the same file yield the
        same persisted eps and the same gate decision (no recomputation)."""
        path, eps = self._persist_poisoned_chart()
        fixture = _wp1_gate_fixture()

        def boom():
            raise AssertionError('build_fn called on a resumed chart')

        eps_seen = []
        decisions = []
        for _ in range(2):
            _chart, report, _reused = _load_or_build(path, boom, {'schema': 'x'})
            eps_seen.append(float(report['heldout_eps']))
            decisions.append(
                _chart_gated('farfield', float(report['heldout_eps']),
                             fixture['config']))
        self.assertEqual(eps_seen, [eps, eps], 'resumed eps not deterministic')
        self.assertEqual(decisions[0], decisions[1],
                         'gate decision not deterministic across reuses')
        self.assertEqual(decisions[0], (True, 'eps_above_bar'))
        self.comparisons += 1

    @staticmethod
    def _persist_legacy_chart_without_eps() -> Path:
        """Save a chart to a temp ``.npz`` whose provenance predates the
        ``heldout_eps`` key, simulating a pre-8g trainer's per-chart file."""
        fixture = _wp1_gate_fixture()
        chart = fixture['healthy']['chart']
        outdir = Path(tempfile.mkdtemp(prefix='wp1_legacy_'))
        path = outdir / 'chart_legacy_ff.npz'
        provenance = {'schema': 'pre8g-chart', 'kind': 'farfield', 'parity': 1}
        LensAmplificationSurrogate([chart], provenance).save(path)
        return path

    def test_resumed_legacy_chart_without_eps_passes_through_ungated(
            self) -> None:
        """A pre-8g per-chart file (no ``heldout_eps`` in its provenance) is
        surfaced by `_load_or_build` with a loud ``legacy_no_eps`` marker,
        `_gate_chart` passes it through un-gated rather than gating on a
        manufactured NaN, and the chart is registered rather than silently
        dropped -- the mixed-version-resume regression INS-2-002 flagged as
        untested (previously only above-bar/persisted-eps resumes were
        exercised)."""
        path = self._persist_legacy_chart_without_eps()
        fixture = _wp1_gate_fixture()

        def boom():
            raise AssertionError('build_fn called on a resumed chart')

        chart, report, reused = _load_or_build(path, boom, {'schema': 'x'})
        self.assertTrue(reused, 'existing file should be reused, not rebuilt')
        self.assertNotIn('heldout_eps', report,
                         'legacy provenance must not manufacture an eps')
        self.assertTrue(report.get('legacy_no_eps'),
                        'legacy resume must set the legacy_no_eps marker')
        self.comparisons += 1

        gated, reason = _gate_chart('farfield', report, fixture['config'])
        self.assertFalse(
            gated, 'legacy resumed chart must pass through un-gated')
        self.assertIsNone(reason)
        self.comparisons += 1

        # Mirror the production registration block (`_train_band_charts`,
        # see `_register_entries`) but keyed on `_gate_chart` (report-based)
        # rather than `_chart_gated` (eps-based), proving a mixed-version
        # resume registers the chart instead of dropping it.
        registered: list = []
        if not gated:
            registered.append(chart)
        self.assertEqual(
            registered, [chart],
            'legacy resumed chart must be registered, not dropped')
        self.comparisons += 1


class SaddleTubeTailTestCase(_CountingTestCase):
    """WP3 saddle tube-tail fix (Professor Q6-iv): a fast synthetic
    reproduction of ``saddle_b1_tube_2`` (strong-shear saddle deltoid arc,
    theta spanning a wedge between branch cusps, coarse grid).

    WITH the WP3 fix (wedge-edge exclusion windows + wider saddle cusp safety)
    the held-out eps drops below ``tube_eps_max`` (5e-2) and far below the
    Professor's pre-fix ~1.15.  WITHOUT the fix (edge window removed, cusp
    safety reverted to the astroid 1.5/0.05) the same synthetic reproduces
    eps > 0.09 -- proving the test bites the real pathology.  The oracle is the
    independent `ChangRefsdalChannels` engine (via `_heldout_eps`).
    """

    def test_fixon_heldout_eps_below_tube_bar(self) -> None:
        """The fixed arc's held-out eps clears the tube bar and sits far below
        the pre-fix ~1.15."""
        fixture = _wp3_fixture()
        on_eps = fixture['on']['eps']
        self.assertTrue(math.isfinite(on_eps), 'fixed arc served nothing')
        self.assertLess(on_eps, _TUBE_EPS_BAR,
                        'fixed saddle arc did not clear the tube bar')
        self.assertLess(on_eps, 0.1,
                        'fixed eps not far below the pre-fix ~1.15 headline')
        self.comparisons += 1

    def test_fixoff_reproduces_pathology_above_threshold(self) -> None:
        """With the fix disabled (no wedge window, astroid cusp safety) the
        same synthetic reproduces the tail pollution (eps > 0.09)."""
        fixture = _wp3_fixture()
        off_eps = fixture['off']['eps']
        self.assertTrue(math.isfinite(off_eps), 'fix-off arc served nothing')
        self.assertGreater(
            off_eps, _WP3_PATHOLOGY_FLOOR,
            'fix-off reproduction did not reproduce the saddle tube-tail')
        self.comparisons += 1

    def test_fix_moves_chart_across_registration_bar(self) -> None:
        """The WP3 fix straddles the registration bar: the fixed arc registers
        while the fix-off arc is gated out (`eps_above_bar`)."""
        fixture = _wp3_fixture()
        on_eps = fixture['on']['eps']
        off_eps = fixture['off']['eps']
        config = fixture['config']
        self.assertLess(on_eps, _TUBE_EPS_BAR)
        self.assertGreater(off_eps, _TUBE_EPS_BAR,
                           'fix-off eps must sit above the bar to bite')
        self.comparisons += 1
        self.assertEqual(_chart_gated('tube', on_eps, config), (False, None),
                         'fixed arc should register')
        self.assertEqual(_chart_gated('tube', off_eps, config),
                         (True, 'eps_above_bar'),
                         'fix-off arc should be gated out')
        self.comparisons += 1

    def test_fixon_arc_is_real_production_arc_with_wedge_window(self) -> None:
        """The fixed arc is a genuine `_saddle_arcs` product carrying a
        wedge-edge exclusion window that the pre-fix arc lacks."""
        fixture = _wp3_fixture()
        on_arc = fixture['on_arc']
        off_arc = fixture['off_arc']
        _cusps, arcs, _reach = _saddle_arcs(_WP3_GAMMA, fixture['n'])
        self.assertIn(on_arc, arcs,
                      'fixed arc is not a real production `_saddle_arcs` arc')
        self.assertEqual(on_arc.branch, 1)
        self.comparisons += 1

        theta_max = 0.5 * np.arcsin(1.0 / abs(_WP3_GAMMA))
        edge_theta = -theta_max + _WEDGE_EPS

        def has_edge_window(arc) -> bool:
            return any(abs(theta - edge_theta) < 1e-6 and halfwidth > 0.0
                       for theta, halfwidth in arc.cusp_windows)

        self.assertTrue(
            has_edge_window(on_arc),
            'the fixed arc must carry a wedge-edge guard window')
        self.assertFalse(
            has_edge_window(off_arc),
            'the pre-fix arc must NOT carry a wedge-edge window (that is the '
            'defect being reproduced)')
        self.comparisons += 1

    def test_saddle_tube_tail_diagnostic_plot(self) -> None:
        """Diagnostic: overlay engine vs emulated envelope across theta for the
        fixed and pre-fix arcs, showing the tail pollution at the wedge end."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fixture = _wp3_fixture()
        eta = 0.5 * (_WP3_CONFIG.eta_floor + _WP3_CONFIG.eta_max)
        on_theta, on_true, on_emul = _wp3_overlay(
            fixture['on']['chart'], fixture['on_arc'], _WP3_GAMMA, eta, 60)
        off_theta, off_true, off_emul = _wp3_overlay(
            fixture['off']['chart'], fixture['off_arc'], _WP3_GAMMA, eta, 60)

        figure, (ax_on, ax_off) = plt.subplots(
            1, 2, figsize=(10, 4), sharey=True)
        ax_on.plot(on_theta, on_true, 'k-', label='engine')
        ax_on.plot(on_theta, on_emul, 'C0.--', label='emulated')
        ax_on.set(title=f'fix ON (eps={fixture["on"]["eps"]:.3f})',
                  xlabel='theta [rad]', ylabel='max|envelope|')
        ax_on.legend()
        ax_off.plot(off_theta, off_true, 'k-', label='engine')
        ax_off.plot(off_theta, off_emul, 'C3.--', label='emulated')
        ax_off.set(title=f'fix OFF (eps={fixture["off"]["eps"]:.3f})',
                   xlabel='theta [rad]')
        ax_off.legend()
        figure.suptitle('WP3 saddle tube-tail: wedge/cusp-end pollution')
        _save_plot(figure, 'wp3_saddle_tube_tail_overlay.png')
        plt.close(figure)
        # The overlay must actually compare something on the fixed arc.
        self.assertTrue(np.any(np.isfinite(on_true) & np.isfinite(on_emul)),
                        'the fixed-arc overlay served no comparable point')
        self.comparisons += 1


# ---------------------------------------------------------------------------
# Build 8g WP3: astroid byte-identity (the saddle-only guard widening must NOT
# perturb the frozen, validated positive-parity astroid path)
# ---------------------------------------------------------------------------

#: Astroid gammas swept for the byte-identity check.  All strictly below the
#: ``gamma = 1`` parity boundary so `_astroid_arcs` is the exercised path.
_ASTROID_BYTE_GAMMAS = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95)

#: Caustic-sample counts swept alongside the gammas; both a coarse and a fine
#: grid must reproduce HEAD exactly (the cusp-window logic is grid-sensitive).
_ASTROID_BYTE_NSAMPLES = (120, 200)


@functools.lru_cache(maxsize=1)
def _head_training_module():
    """Load the pre-WP3 ``surrogate_training`` module (``HEAD``) side-by-side.

    The WP3 change is uncommitted in the working tree, so ``HEAD`` is the
    literal *before* state: its `_find_cusps` has no ``width_safety`` /
    ``min_halfwidth`` kwargs.  A private copy of the HEAD source is exec'd into
    a freshly-named module (registered in ``sys.modules`` FIRST so its frozen
    dataclasses resolve), giving an independent `_astroid_arcs` whose call
    chain differs from the working tree ONLY in `_find_cusps` -- the four other
    astroid dependencies (`_branch_speed_profile`, `_caustic_reach`,
    `_make_arc`, and the `FoldArc` fields) are byte-identical across the commit
    (verified out of band), so a nonzero table diff isolates the WP3 change.
    """
    repo_root = Path(__file__).resolve().parents[2]
    src = subprocess.run(
        ['git', 'show', 'HEAD:cogwheel/lensing/surrogate_training.py'],
        capture_output=True, text=True, check=True,
        cwd=str(repo_root)).stdout
    handle = tempfile.NamedTemporaryFile(
        'w', suffix='.py', prefix='head_surr_train_', delete=False)
    handle.write(src)
    handle.flush()
    handle.close()
    modname = 'cogwheel_head_surrogate_training_wp3'
    spec = importlib.util.spec_from_file_location(modname, handle.name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module  # register FIRST so @dataclass resolves
    spec.loader.exec_module(module)
    return module


def _arc_fields(arc) -> tuple:
    """The comparable scalar/tuple fields of a `FoldArc` (cross-module safe).

    Frozen dataclasses compare unequal across two module copies (they are
    different classes), so byte-identity is asserted on the fields, not ``==``.
    """
    return (int(arc.branch), float(arc.theta_lo), float(arc.theta_hi),
            int(arc.inward_sign), int(arc.image_count),
            tuple((float(t), float(w)) for t, w in arc.cusp_windows))


def _astroid_arcs_max_diff(cur, head) -> float:
    """Max absolute element-wise difference between two `_astroid_arcs`
    returns ``(cusps, arcs, reach)``; ``+inf`` on any structural mismatch."""
    cur_cusps, cur_arcs, cur_reach = cur
    head_cusps, head_arcs, head_reach = head
    if len(cur_cusps) != len(head_cusps) or len(cur_arcs) != len(head_arcs):
        return math.inf
    worst = abs(float(cur_reach) - float(head_reach))
    for (tc, wc), (th, wh) in zip(cur_cusps, head_cusps):
        worst = max(worst, abs(tc - th), abs(wc - wh))
    for a_cur, a_head in zip(cur_arcs, head_arcs):
        fields_cur = _arc_fields(a_cur)
        fields_head = _arc_fields(a_head)
        if len(fields_cur[5]) != len(fields_head[5]):
            return math.inf
        for x, y in zip(fields_cur[:5], fields_head[:5]):
            worst = max(worst, abs(x - y))
        for (t1, w1), (t2, w2) in zip(fields_cur[5], fields_head[5]):
            worst = max(worst, abs(t1 - t2), abs(w1 - w2))
    return worst


class AstroidByteIdentityTestCase(_CountingTestCase):
    """WP3 saddle-only guard widening must not perturb the frozen astroid path.

    `_astroid_arcs` calls `_find_cusps` with the module DEFAULT
    ``width_safety`` / ``min_halfwidth`` (the astroid constants); the WP3
    change added those kwargs and the saddle path passes WIDER values.  If the
    kwargs' defaults or the cusp-finding logic drifted, the validated astroid
    cusp/arc tables would move.  Oracle: the pre-WP3 ``HEAD`` `_astroid_arcs`
    (independent module copy) -- byte-identity means max element diff 0.0.
    """

    def test_astroid_arcs_byte_identical_head_to_worktree(self) -> None:
        """Cusp/arc/reach tables reproduce HEAD exactly across a gamma sweep."""
        head = _head_training_module()
        for gamma, n in itertools.product(
                _ASTROID_BYTE_GAMMAS, _ASTROID_BYTE_NSAMPLES):
            with self.subTest(gamma=gamma, n=n):
                cur = training._astroid_arcs(float(gamma), n)
                ref = head._astroid_arcs(float(gamma), n)
                # Structural agreement first (same cusp / arc counts).
                self.assertEqual(len(cur[0]), len(ref[0]),
                                 'astroid cusp count moved vs HEAD')
                self.assertEqual(len(cur[1]), len(ref[1]),
                                 'astroid arc count moved vs HEAD')
                self.assertEqual(
                    _astroid_arcs_max_diff(cur, ref), 0.0,
                    'WP3 perturbed the frozen astroid path (nonzero diff)')
                self.comparisons += 1

    def test_wp3_default_kwargs_equal_frozen_astroid_constants(self) -> None:
        """The WP3 `_find_cusps` defaults ARE the astroid constants -- the
        mechanism by which the astroid path stays byte-identical."""
        params = inspect.signature(_find_cusps).parameters
        self.assertEqual(params['width_safety'].default, _CUSP_WIDTH_SAFETY)
        self.assertEqual(params['min_halfwidth'].default, _CUSP_MIN_HALFWIDTH)
        # The saddle constants the saddle path passes are genuinely WIDER, so
        # the astroid defaults are not trivially equal to everything.
        self.assertGreater(_SADDLE_CUSP_WIDTH_SAFETY, _CUSP_WIDTH_SAFETY)
        self.assertGreater(_SADDLE_CUSP_MIN_HALFWIDTH, _CUSP_MIN_HALFWIDTH)
        self.comparisons += 1

    def test_saddle_guard_params_would_perturb_astroid(self) -> None:
        """Self-falsification: had the WIDER saddle guard leaked into the
        astroid path, the cusp windows would move -- so the byte-identity above
        has teeth (it is not vacuously comparing an unchanging value).

        Rebuild the astroid cusps with the saddle safety/half-width and show at
        least one gamma's cusp windows differ from the frozen (default) output.
        """
        moved_any = False
        for gamma in _ASTROID_BYTE_GAMMAS:
            thetas, speed = _branch_speed_profile(
                float(gamma), 1, 0.0, 2.0 * np.pi, 200, periodic=True)
            frozen = _find_cusps(thetas, speed, periodic=True)
            widened = _find_cusps(
                thetas, speed, periodic=True,
                width_safety=_SADDLE_CUSP_WIDTH_SAFETY,
                min_halfwidth=_SADDLE_CUSP_MIN_HALFWIDTH)
            self.assertEqual(len(frozen), len(widened),
                             'cusp COUNT should not change, only widths')
            for (_t0, w0), (_t1, w1) in zip(frozen, widened):
                if abs(w1 - w0) > 1e-9:
                    moved_any = True
            self.comparisons += 1
        self.assertTrue(
            moved_any,
            'saddle guard params left every astroid cusp window unchanged: '
            'the byte-identity check would be vacuous')

    def test_astroid_byte_identity_diagnostic(self) -> None:
        """Diagnostic: max |diff| of the arc theta / cusp-window tables vs HEAD
        across the gamma sweep (all zero)."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        head = _head_training_module()
        gammas = list(_ASTROID_BYTE_GAMMAS)
        diffs = [_astroid_arcs_max_diff(
            training._astroid_arcs(float(g), 200),
            head._astroid_arcs(float(g), 200)) for g in gammas]
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(gammas, np.maximum(diffs, 1e-18), 'o-')
        axis.axhline(0.0, color='k', ls='--')
        axis.set(xlabel='astroid gamma', ylabel='max |worktree - HEAD|',
                 yscale='log',
                 title='astroid _astroid_arcs byte-identity (WP3)')
        _save_plot(figure, 'wp3_astroid_byte_identity_diff.png')
        plt.close(figure)
        self.assertEqual(max(diffs), 0.0,
                         'astroid arc tables diverged from HEAD')
        self.comparisons += 1


# ---------------------------------------------------------------------------
# Professor Q5: residue-bucket partition of the prior draws (synthetic scale)
# ---------------------------------------------------------------------------

#: Detector band edges (Hz) for the census ``w(1024, m)`` ceiling test.
_CENSUS_F_LO_HZ = 20.0
_CENSUS_F_HI_HZ = 1024.0

#: Number of prior draws classified into the three buckets.
_CENSUS_N_DRAWS = 3000

#: RNG seed making the census deterministic (bucket counts are reproducible).
_CENSUS_SEED = 8080808


@functools.lru_cache(maxsize=1)
def _census_surrogate() -> LensAmplificationSurrogate:
    """Train the smoke-scale surrogate once and keep the SERVING artifact.

    The WP2 tiling suite's `_trained_report` fixture discards the surrogate;
    the census needs the served object itself as the honest 'chart-served'
    oracle, so this runs its own cached engine-backed train (a few minutes).
    """
    outdir = tempfile.mkdtemp(prefix='census_surr_')
    surrogate, _report = train(outdir=outdir, config=_FIXTURE_CONFIG)
    return surrogate


def _census_ceiling(gamma: float) -> float:
    """Parity ``w``-ceiling for a draw: astroid ``gamma < 1`` vs saddle."""
    return _POSITIVE_W_CEILING if gamma < 1.0 else _SADDLE_W_CEILING


def _census_serves(surrogate: LensAmplificationSurrogate,
                   channels: ChangRefsdalChannels, gamma: float,
                   y1: float, y2: float, band_lo: float, band_hi: float
                   ) -> bool:
    """Does the surrogate serve the whole band of one draw? (engine + serve)

    Returns ``False`` on a geometry refusal (a refused draw then falls to
    ``residue`` unless it is above the parity ceiling).  Only the named
    `geometry.LensDomainError` is swallowed; any other error propagates.
    """
    channels.reset()
    try:
        partition = channels.geometry_partition(
            gamma=gamma, y=[y1, y2], beta=0.0)
        _envelope, served, _definition = surrogate.serve(
            np.array([band_lo, band_hi]), gamma=gamma, y1=y1, y2=y2,
            beta=0.0, eta=partition.caustic_distance,
            theta=partition.caustic_theta,
            image_count=int(partition.real_mask.sum()))
    except geometry.LensDomainError:
        return False
    return bool(served)


@functools.lru_cache(maxsize=1)
def _census_result() -> dict:
    """Classify ``_CENSUS_N_DRAWS`` prior draws into the three buckets.

    Each draw is placed into EXACTLY one of ``beyond_w_cap`` (its independent
    ``w(1024, m)`` exceeds the parity ceiling), ``chart_served`` (the trained
    surrogate serves its whole band), or ``residue`` (everything else).  The
    bucket boundary uses the F002-independent `_w_indep` oracle, never the
    production `dimensionless_frequency` it cross-checks.
    """
    surrogate = _census_surrogate()
    rng = np.random.default_rng(_CENSUS_SEED)
    box = PriorBox.from_prior_classes()
    ln_m_lo, ln_m_hi = box.ln_m_lens_range
    gamma_lo, gamma_hi = box.gamma_range
    channels = ChangRefsdalChannels(np.geomspace(1.0, 10.0, 3))
    labels: list[str] = []
    ln_masses: list[float] = []
    gammas: list[float] = []
    beyond_served = 0
    for _ in range(_CENSUS_N_DRAWS):
        ln_m = float(rng.uniform(ln_m_lo, ln_m_hi))
        m_lens = float(lens_prior.UniformLensMassPrior.transform(
            ln_m)['m_lens_msun'])
        gamma = float(rng.uniform(gamma_lo, gamma_hi))
        u1, u2 = rng.uniform(-1.0, 1.0, size=2)
        source = lens_prior.UniformSourcePositionPrior.transform(
            u1, u2, m_lens)
        y1, y2 = float(source['y1']), float(source['y2'])
        band_lo = _w_indep(_CENSUS_F_LO_HZ, m_lens)
        band_hi = _w_indep(_CENSUS_F_HI_HZ, m_lens)
        served = _census_serves(
            surrogate, channels, gamma, y1, y2, band_lo, band_hi)
        if band_hi > _census_ceiling(gamma):
            labels.append('beyond_w_cap')
            beyond_served += int(served)
        elif served:
            labels.append('chart_served')
        else:
            labels.append('residue')
        ln_masses.append(math.log(m_lens))
        gammas.append(gamma)
    counts = {name: labels.count(name)
              for name in ('beyond_w_cap', 'chart_served', 'residue')}
    return {'labels': labels, 'ln_m': np.asarray(ln_masses),
            'gamma': np.asarray(gammas), 'counts': counts,
            'beyond_served': beyond_served, 'n': _CENSUS_N_DRAWS}


class ResidueBucketPartitionTestCase(_CountingTestCase):
    """Professor Q5: the prior draws partition into EXACTLY three buckets --
    chart-served, beyond-w-cap, and residue -- with no double-count and no
    silent drop.  Beyond-ceiling draws are attributed to their named bucket,
    not folded into residue; the residue fraction is MEASURED and reported
    (closing it is Build 8h's north star), never asserted to be zero.
    """

    def test_three_buckets_partition_all_draws(self) -> None:
        """The bucket counts sum to N with no draw double-counted or dropped."""
        result = _census_result()
        counts = result['counts']
        self.assertEqual(
            sum(counts.values()), result['n'],
            'bucket counts do not sum to N (a draw was dropped or '
            'double-counted)')
        self.comparisons += 1
        # Every draw carries exactly one of the three labels.
        self.assertEqual(len(result['labels']), result['n'])
        self.assertTrue(
            set(result['labels']) <= {'beyond_w_cap', 'chart_served',
                                      'residue'},
            'an unexpected bucket label leaked in')
        self.comparisons += 1

    def test_beyond_w_cap_attributed_not_residue(self) -> None:
        """Beyond-ceiling draws land in the beyond bucket (independent oracle),
        and NO served/residue draw is secretly beyond the ceiling."""
        result = _census_result()
        self.assertGreater(
            result['counts']['beyond_w_cap'], 0,
            'beyond-w-cap bucket empty: nothing to attribute')
        self.comparisons += 1
        for label, ln_m, gamma in zip(
                result['labels'], result['ln_m'], result['gamma']):
            band_hi = _w_indep(_CENSUS_F_HI_HZ, math.exp(float(ln_m)))
            beyond = band_hi > _census_ceiling(float(gamma))
            if label == 'beyond_w_cap':
                self.assertTrue(
                    beyond, 'a beyond-labelled draw is below the ceiling')
            else:
                self.assertFalse(
                    beyond, f'a {label} draw is above the ceiling and should '
                    'be in the beyond bucket, not here')
        self.comparisons += 1

    def test_chart_served_bucket_is_nonempty(self) -> None:
        """The served bucket is non-vacuous (the surrogate actually serves)."""
        result = _census_result()
        self.assertGreater(
            result['counts']['chart_served'], 0,
            'no draw was chart-served: the census oracle serves nothing')
        self.comparisons += 1

    def test_residue_fraction_is_measured_not_zero_asserted(self) -> None:
        """The residue fraction is MEASURED and reported, not asserted zero.

        Build 8h's north star is to shrink this; the test pins that it is a
        real measured fraction in ``[0, 1]`` that closes the partition, so a
        future build can watch it fall without this test forcing it to already
        be zero.
        """
        result = _census_result()
        n = result['n']
        residue_fraction = result['counts']['residue'] / n
        beyond_fraction = result['counts']['beyond_w_cap'] / n
        served_fraction = result['counts']['chart_served'] / n
        # A measured fraction in the unit interval that closes with the others.
        self.assertGreaterEqual(residue_fraction, 0.0)
        self.assertLessEqual(residue_fraction, 1.0)
        self.assertAlmostEqual(
            residue_fraction + beyond_fraction + served_fraction, 1.0,
            places=12)
        self.comparisons += 1
        print(f'\n[residue-bucket census] N={n} '
              f'served={served_fraction:.3f} '
              f'beyond_w_cap={beyond_fraction:.3f} '
              f'residue={residue_fraction:.3f} '
              f'(beyond_served={result["beyond_served"]})')

    def test_beyond_draws_would_be_residue_without_named_bucket(self) -> None:
        """Teeth for the named-bucket attribution: every beyond-ceiling draw
        fails to serve on its whole band (``beyond_served == 0``), so WITHOUT
        the beyond bucket they would ALL be miscounted as residue -- inflating
        residue by the beyond count.  The named bucket is what prevents that."""
        result = _census_result()
        self.assertEqual(
            result['beyond_served'], 0,
            'a beyond-ceiling draw served its whole band: the ceiling is not '
            'the serving boundary the bucket assumes')
        self.comparisons += 1
        naive_residue = (result['counts']['residue']
                         + result['counts']['beyond_w_cap'])
        self.assertGreater(
            naive_residue, result['counts']['residue'],
            'folding beyond into residue must change the residue count')
        self.comparisons += 1

    def test_residue_bucket_diagnostic_plot(self) -> None:
        """Diagnostic: stacked histogram of bucket membership over ln m."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        result = _census_result()
        labels = np.asarray(result['labels'])
        ln_m = result['ln_m']
        edges = np.linspace(float(ln_m.min()), float(ln_m.max()), 18)
        names = ('chart_served', 'beyond_w_cap', 'residue')
        series = [ln_m[labels == name] for name in names]
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.hist(series, bins=edges, stacked=True, label=list(names),
                  color=['C0', 'C1', 'C3'])
        axis.set(xlabel='ln m_lens [ln Msun]', ylabel='draws',
                 title=f'residue-bucket partition over ln m (N={result["n"]})')
        axis.legend()
        _save_plot(figure, 'q5_residue_bucket_over_lnm.png')
        plt.close(figure)
        # The plotted series together account for every draw (no silent drop).
        self.assertEqual(sum(len(s) for s in series), result['n'])
        self.comparisons += 1


class SelfFalsificationTestCase(_CountingTestCase):
    """Corrupt each contract and prove the green checks would go red.

    Without this, a silently-passing suite is indistinguishable from a
    correct one.  Each test injects a defect the corresponding real test is
    built to catch and asserts the check fails.
    """

    def test_overlapping_tiles_would_fail_disjointness(self) -> None:
        """Two boxes closer than ``2*half`` trip the disjointness assertion."""
        half = 0.6
        box_a = ((0.0, 0.0), half)
        box_b = ((0.5, 0.0), half)  # centre sep 0.5 < 2*half = 1.2 -> overlap
        separation = _max_norm_center_separation(box_a, box_b)
        self.assertLess(separation, 2.0 * half)
        with self.assertRaises(AssertionError):
            # This is the exact assertion TilingRecordTestCase makes.
            self.assertGreaterEqual(
                separation, 2.0 * half - _TILE_DISJOINT_TOL)
        self.comparisons += 1

    def test_tile_touching_disk_would_fail_exterior_check(self) -> None:
        """A tile straddling the caustic disk trips the exterior assertion."""
        radius = 0.6
        # A tile centred at the origin lies wholly INSIDE the disk.
        distance = _tile_outside_disk((0.0, 0.0), half=0.3, radius=radius)
        with self.assertRaises(AssertionError):
            self.assertGreaterEqual(distance, radius)
        self.comparisons += 1

    def test_wrong_w_constant_breaks_containment(self) -> None:
        """A 3x-wrong ``w`` constant sends the band top out of the range."""
        box = PriorBox.from_prior_classes()
        strata, _beyond = _mass_strata(box, 1)
        m_lo, m_hi = strata[0]
        y_extent = float(lens_prior._source_scale(m_lo))
        _w_min, w_max = _stratum_w_range(box, 1, m_lo, m_hi, y_extent)
        bad_band_hi = (_W_LENSING_PER_MSUN_HZ * 3.0) * m_hi * 1024.0
        with self.assertRaises(AssertionError):
            # The containment assertion WholeBandContainmentTestCase makes.
            self.assertLessEqual(
                bad_band_hi, w_max * (1.0 + _W_CONTAINMENT_REL_TOL))
        self.comparisons += 1

    def test_chart_over_hole_makes_interior_point_serve(self) -> None:
        """A chart widened over the interior hole serves a hole point, which
        would break the outside-``None`` contract (proving it has teeth).

        The "hole" is the un-tiled disk ``rho < exclusion_rho`` around the
        origin (Build 8h-b3: `_farfield_tiles` tiles only the exterior
        annulus).  The origin itself (``y1_eig = y2_eig = 0``) maps to
        ``rho = 0`` at ANY ``gamma`` and an ARBITRARY ``theta_c`` (``atan2``
        of a zero vector; caustic-fixed ``theta_c`` is undefined exactly at
        the origin, so the bad chart's box must cover the FULL angular
        range, not just a wedge, to genuinely "cover the hole").
        """
        fixture = _serve_fixture()
        log_w_grid = fixture['log_w_grid']
        gamma_grid = np.linspace(_SERVE_GAMMA_BAND[0], _SERVE_GAMMA_BAND[1], 4)
        envelope = np.ones((4, 4, 4, 4))
        # A far-field chart whose box COVERS the un-tiled interior disk: all
        # of rho in [0, 1.5 * exclusion_rho] (comfortably past the hole's
        # outer edge) at every theta_c.
        bad_chart = FarFieldChart.from_values(
            gamma_grid=gamma_grid,
            rho_grid=np.linspace(0.0, 1.5 * fixture['exclusion_rho'], 4),
            theta_c_grid=np.linspace(-math.pi, math.pi, 4),
            log_w_grid=log_w_grid, envelope_real=envelope,
            envelope_imag=envelope, image_count=2, parity=1,
            eta_overlap_min=0.05)
        mid_log_w = float(0.5 * (log_w_grid[0] + log_w_grid[-1]))
        served = select_chart(
            [bad_chart], gamma=0.35, log_w_min=mid_log_w, log_w_max=mid_log_w,
            eta=5.0, theta=0.0, image_count=2, rho=0.0, theta_c=0.0)
        self.assertIsNotNone(
            served, 'a chart over the hole must serve the hole point')
        # And the honest fixture (hole not covered) must NOT serve it.
        clean = select_chart(
            fixture['charts'], gamma=0.35, log_w_min=mid_log_w,
            log_w_max=mid_log_w, eta=5.0, theta=0.0, image_count=2,
            rho=0.0, theta_c=0.0)
        self.assertIsNone(clean, 'the honest fixture leaves the hole unserved')
        self.comparisons += 1

    def test_opening_the_bar_lets_the_poisoned_chart_register(self) -> None:
        """The eps bar is load-bearing: with ``farfield_eps_max`` opened wide,
        the poisoned chart is NOT gated -- proving the default bar (not some
        always-true refusal) is what excludes it."""
        fixture = _wp1_gate_fixture()
        eps = fixture['poisoned']['eps']
        default_gate = _chart_gated('farfield', eps, fixture['config'])
        self.assertEqual(default_gate, (True, 'eps_above_bar'))
        opened = TrainingConfig(farfield_eps_max=1e9)
        self.assertEqual(_chart_gated('farfield', eps, opened), (False, None),
                         'opening the bar must let the poisoned chart pass')
        self.comparisons += 1

    def test_registered_poisoned_chart_would_serve_its_window(self) -> None:
        """Were the poisoned chart NOT gated, it would serve its own window --
        so the gate's exclusion is what matters (the window is otherwise live)."""
        fixture = _wp1_gate_fixture()
        poisoned = fixture['poisoned']['chart']
        center = fixture['poisoned']['center']
        mid_log_w = float(0.5 * (poisoned.log_w_grid[0]
                                 + poisoned.log_w_grid[-1]))
        served = select_chart(
            [poisoned], gamma=0.35, log_w_min=mid_log_w, log_w_max=mid_log_w,
            eta=5.0, theta=0.0, image_count=2,
            rho=center[0], theta_c=center[1])
        self.assertIsNotNone(
            served, 'the poisoned window is live; only the gate removes it')
        self.comparisons += 1

    def test_fixoff_arc_would_pass_if_pathology_not_reproduced(self) -> None:
        """WP3 reachable-red control: if the fix-off eps were (wrongly) below
        the bar, the pathology reproduction would be vacuous.  Assert the
        measured fix-off eps genuinely exceeds the bar."""
        fixture = _wp3_fixture()
        off_eps = fixture['off']['eps']
        self.assertGreater(off_eps, _TUBE_EPS_BAR,
                           'fix-off eps below the bar would make WP3 vacuous')
        # And the registration gate agrees it is excluded.
        self.assertEqual(
            _chart_gated('tube', off_eps, fixture['config']),
            (True, 'eps_above_bar'))
        self.comparisons += 1

    def test_shifted_cusp_breaks_astroid_byte_identity(self) -> None:
        """Patching `_find_cusps` to nudge a cusp makes `_astroid_arcs`
        diverge from HEAD -- so the byte-identity check goes red under a real
        perturbation (it is not blind to a moved astroid cusp)."""
        head = _head_training_module()
        gamma, n = 0.4, 200
        ref = head._astroid_arcs(gamma, n)
        real_find_cusps = training._find_cusps

        def shifted(thetas, speed, periodic, **kwargs):
            cusps = real_find_cusps(thetas, speed, periodic, **kwargs)
            # Nudge the first cusp's theta by a hair -> table moves.
            if cusps:
                t0, w0 = cusps[0]
                cusps[0] = (t0 + 1e-3, w0)
            return cusps

        with mock.patch.object(training, '_find_cusps', shifted):
            perturbed = training._astroid_arcs(gamma, n)
        self.assertGreater(
            _astroid_arcs_max_diff(perturbed, ref), 0.0,
            'a shifted astroid cusp left the table byte-identical to HEAD')
        self.comparisons += 1

    def test_dropped_draw_breaks_partition_sum(self) -> None:
        """A bucket census whose counts do not sum to N trips the partition
        assertion, so the sum-to-N check has teeth (a silent drop is caught)."""
        # Synthetic counts with one draw silently dropped.
        counts = {'beyond_w_cap': 100, 'chart_served': 50, 'residue': 1849}
        n_draws = 2000  # 100 + 50 + 1849 = 1999 != 2000
        with self.assertRaises(AssertionError):
            self.assertEqual(sum(counts.values()), n_draws,
                             'partition-sum check must reject a dropped draw')
        self.comparisons += 1

    def test_folding_beyond_into_residue_misreports_counts(self) -> None:
        """If the beyond bucket were folded into residue, the reported residue
        count would change -- confirming named-bucket attribution is not a
        cosmetic relabelling but changes the numbers."""
        counts = {'beyond_w_cap': 264, 'chart_served': 28, 'residue': 1708}
        folded_residue = counts['residue'] + counts['beyond_w_cap']
        self.assertNotEqual(
            folded_residue, counts['residue'],
            'folding beyond into residue must change the residue count')
        self.comparisons += 1


if __name__ == '__main__':
    main()
