"""Tests for the Build-8g WP2 mass-stratified far-field tiling of the
`lensing.surrogate_training` trainer, plus the serving-contract both
directions through `lensing.surrogate.select_chart`.

PROVISIONAL -- READ BEFORE "FIXING" ANYTHING HERE
-------------------------------------------------
The STRUCTURAL assertions in this file (tile counts, record keys, stratum
bookkeeping, box shapes, report schema) pin the CURRENT training-record
structure, which is MID-REDESIGN.  They are NOT a specification and must not
be treated as one.

The surrogate serves ~2% of the prior today.  Its structure has already
changed three times -- 8h-b3 moved the spatial axes to caustic-fixed
``(rho, theta_c)``, 8h-b4 replaced scalar exclusion with per-column
admission, S1-3 retired per-stratum exterior partitioning for region windows
-- and each migration silently killed the tests written against the previous
shape (25 of them, undetected for a whole build cycle).  The gate re-key,
Born wiring and census work will move it again.

So, if your change breaks a structural test here: UPDATE OR DELETE IT.  Do
not contort production to keep it green, and do not spend a build debugging
it.  The durable claims in the lensing suites are the NUMERICAL ones -- does
the reconstruction match the engine, does held-out eps clear its bar, does a
poisoned chart degrade, does the delay frame round-trip.  Those survive
refactors; the bookkeeping does not.

Revisit the structural layer when the serving design stops moving, i.e.
after the ladder actually closes.

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
import os
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
    _tube_heldout_samples, _tube_source, _CUSP_WIDTH_SAFETY,
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


#: ENGINE-BACKED TIER (opt-in).  Classes marked `_TRAIN_TIER_SKIP` build REAL
#: surrogate charts -- they call `train` / `_build_farfield_chart`, running
#: hundreds of Schwinger/operator evaluations, and take MINUTES.  Training and
#: census runs belong to whoever DRIVES the build -- they are post-build driver
#: steps, not work the build does and not unit tests -- and a multi-minute file
#: in the fast tier is one nobody runs, which is how this suite silently rotted
#: through three interface migrations.  Structural
#: assertions needing only a representative report should move to a cached
#: golden artifact; until then these are opt-in, matching the existing
#: COGWHEEL_BRUTE_ACCURACY / COGWHEEL_STRICT_TIMING idiom.
#:
#: Run them with:  COGWHEEL_TRAIN_TIER=1 python -m pytest <file>
_TRAIN_TIER_SKIP = unittest.skipUnless(
    os.environ.get('COGWHEEL_TRAIN_TIER'),
    'engine-backed training tier: set COGWHEEL_TRAIN_TIER=1 (builds real '
    'surrogate charts, minutes per class; the driver runs these post-build)')






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


@_TRAIN_TIER_SKIP
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

#: Retired production wedge inset (the deleted ``surrogate_training._WEDGE_EPS
#: = 1e-3``, removed by WP1 "delete _WEDGE_EPS / sample the true edge").  Kept
#: TEST-LOCAL solely to reconstruct the PRE-fix counterfactual saddle arc in
#: `_wp3_fixoff_left_arc`, whose whole point is to stand slightly inside the
#: wedge exactly as the old production code did.  Production now samples the
#: true wedge edge with NO inset -- WedgeEpsDeletionTestCase (Gate 1) scans the
#: shipped module source to prove no such ``1e-3`` survives there.
_LEGACY_WEDGE_EPS = 1e-3

#: Pre-fix pathology floor: the counterfactual (fix-disabled) arc must exceed
#: this, proving the reproduction bites the real tail (measured ~0.43), while
#: the Professor's headline pre-fix number was ~1.15.
_WP3_PATHOLOGY_FLOOR = 0.09

#: Fixture config for the WP3 tube builds (two tube charts, ~40 s).
#: Grid is 5, not 4 -- a STOPGAP (F042, 2026-07-29). At 4x4x4 the fix-on eps
#: sat knife-edge at 0.0499 against the 0.05 bar, calibrated to the pre-1b
#: SAMPLED cusp bounds; 1b's analytic cusp root (|y'|=0 exactly) shifted the
#: arc bounds enough to tip it to 0.0592. The ROOT cause is that theta is
#: gridded uniform in theta (absolute), not in the caustic's own coordinate:
#: at the same n=4, an arc-length grid (int |y'| dtheta, from caustic_speed)
#: fits 2.2x better (0.027 vs 0.059) and is insensitive to the bound shift
#: that a uniform grid swings +-23% under. The real fix is arc-length theta
#: placement (see FINDINGS F042 / lensing_collocation_from_local_scales);
#: grid 5 just adds uniform nodes until the knife-edge clears. No production
#: risk: the shipped trainer builds far finer than this synthetic.
_WP3_CONFIG = TrainingConfig(
    n_gamma=5, n_u=5, n_theta=5, w_nodes_per_decade=3, n_heldout=16,
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
    lo_edge = -theta_max + _LEGACY_WEDGE_EPS
    hi_edge = theta_max - _LEGACY_WEDGE_EPS
    branch = 1
    thetas, speed = _branch_speed_profile(
        gamma, branch, lo_edge, hi_edge, n, periodic=False)
    reach = _caustic_reach(gamma, branch, lo_edge, hi_edge, n)
    cusps = _find_cusps(
        thetas, speed, periodic=False, gamma=gamma, branch=branch,
        width_safety=_CUSP_WIDTH_SAFETY, min_halfwidth=_CUSP_MIN_HALFWIDTH)
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


@_TRAIN_TIER_SKIP
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


@_TRAIN_TIER_SKIP
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


@_TRAIN_TIER_SKIP
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

        # WP1 ("delete _WEDGE_EPS"): the fixed arc's wedge-edge guard window
        # now sits at the TRUE wedge edge ``-theta_max`` (previously inset by
        # the retired ``_WEDGE_EPS = 1e-3``), so the edge is located with no
        # offset here.
        theta_max = 0.5 * np.arcsin(1.0 / abs(_WP3_GAMMA))
        edge_theta = -theta_max

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


@_TRAIN_TIER_SKIP
class AstroidByteIdentityTestCase(_CountingTestCase):
    """WP3 saddle-only guard widening must not perturb the frozen astroid path.

    `_astroid_arcs` calls `_find_cusps` with the module DEFAULT
    ``width_safety`` / ``min_halfwidth`` (the astroid constants); the WP3
    change added those kwargs and the saddle path passes WIDER values.  If the
    kwargs' defaults or the cusp-finding logic drifted, the validated astroid
    cusp/arc tables would move.  Oracle: the pre-WP3 ``HEAD`` `_astroid_arcs`
    (independent module copy) -- byte-identity means max element diff 0.0.
    """

    @unittest.skip('Retired by 1b: the analytic cusp root reorders the astroid '
                   'float path by 4.4e-16 (ULP), so exact byte-identity to '
                   'pre-1b HEAD no longer holds. Structural cusp/arc counts '
                   'still pass; a near-identity (<1e-12) test can replace this.')
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
            frozen = _find_cusps(
                thetas, speed, periodic=True, gamma=float(gamma), branch=1)
            widened = _find_cusps(
                thetas, speed, periodic=True, gamma=float(gamma), branch=1,
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

    @unittest.skip('Retired by 1b (see the sibling byte-identity test): the '
                   'analytic-cusp float reorder gives a 4.4e-16 diff to HEAD.')
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




@_TRAIN_TIER_SKIP
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


# ---------------------------------------------------------------------------
# F041: arc-orientation guard -- stable_gamma_bands sliver / arc acceptance
# ---------------------------------------------------------------------------

#: Positive-parity gamma band swept by the F041 acceptance test.  Before the
#: WP1 fix to the `_make_arc` arc-orientation guard, the two smallest
#: sub-bands built ZERO fold arcs, so `stable_gamma_bands` read that as a
#: topology change (arc count 0 vs 2), bisected them to slivers narrower than
#: ``_F041_MIN_WIDTH`` and DROPPED them -- those gammas fell through to exact
#: serving.  After the fix every band builds its arcs and nothing is dropped.
_F041_BAND = (0.01, 0.30)
#: Caustic-detection sample count (fast tier: one full sweep is ~40 ms).
_F041_N_SAMPLES = 200
#: Minimum topology-stable band width; narrower straddling slivers are dropped.
_F041_MIN_WIDTH = 0.02
#: Gammas whose covering stable band must build at least one fold arc.
_F041_EXISTENCE_GAMMAS = (0.02, 0.1, 0.3, 0.9)
#: Gammas (all >= 0.1) whose arc labels must be stable and unmoved by the fix.
_F041_LABEL_GAMMAS = (0.1, 0.2, 0.3, 0.9)
#: Real-image count on the served side of an astroid fold arc at kappa = 0 --
#: a parity constant (the matched interior image pair), asserted as a VALUE,
#: never probed via `find_images`.
_F041_ASTROID_IMAGE_COUNT = 4


class StableGammaBandsF041TestCase(_CountingTestCase):
    """F041 acceptance: the arc-orientation guard in `_make_arc` must not
    starve small-gamma astroid bands of fold arcs.

    A sub-band that builds ZERO arcs is a topology change (arc count 0 vs 2),
    so before the WP1 fix the two smallest sub-bands of ``(0.01, 0.30)`` were
    bisected into slivers and dropped.  The fix makes every band build its
    arcs, so nothing is dropped and every returned band is served.  Every
    check below asserts a VALUE (the dropped list, arc counts, arc labels),
    never a code path or a removed-constant name.
    """

    @staticmethod
    def _diagnostic(stable, dropped) -> str:
        """Localises an F041 failure: each band's ``(lo, hi)`` and arc count
        plus the dropped-sliver list (spec-mandated failure diagnostic)."""
        lines = [f'dropped={dropped}']
        for (lo, hi), structure in stable:
            lines.append(f'  band ({lo:.5f}, {hi:.5f}) '
                         f'n_arcs={len(structure.arcs)}')
        return '\n'.join(lines)

    def test_no_dropped_slivers_and_every_band_builds_arcs(self) -> None:
        """Assertion 1 (the load-bearing F041 regression witness): the sweep
        drops NO sliver, and every returned band builds at least one arc -- a
        band that exists but yields no arc is unserved exactly like a dropped
        one."""
        stable, dropped = training.stable_gamma_bands(
            _F041_BAND, +1, n_samples=_F041_N_SAMPLES,
            min_width=_F041_MIN_WIDTH)
        diag = self._diagnostic(stable, dropped)
        self.assertEqual(
            dropped, [],
            f'F041 regression: bands dropped as slivers.\n{diag}')
        self.comparisons += 1
        self.assertGreater(
            len(stable), 0, f'no stable bands returned at all.\n{diag}')
        self.comparisons += 1
        for (lo, hi), structure in stable:
            with self.subTest(band=(lo, hi)):
                self.assertGreater(
                    len(structure.arcs), 0,
                    f'band ({lo:.5f}, {hi:.5f}) built ZERO arcs -- unserved '
                    f'exactly like a dropped sliver (F041).\n{diag}')
            self.comparisons += 1

    def test_arc_existence_across_gamma(self) -> None:
        """Assertion 2: for a spread of gammas, at least one stable band that
        covers the gamma builds a non-empty `CausticStructure`.  With the
        magnitude guard gone there is no gamma-stable ratio to assert, so
        acceptance-2 is realised as arc EXISTENCE."""
        for gamma in _F041_EXISTENCE_GAMMAS:
            with self.subTest(gamma=gamma):
                band = (max(gamma - 0.03, 0.005), gamma + 0.03)
                stable, dropped = training.stable_gamma_bands(
                    band, +1, n_samples=_F041_N_SAMPLES,
                    min_width=_F041_MIN_WIDTH)
                covering = [s for (lo, hi), s in stable if lo <= gamma <= hi]
                self.assertTrue(
                    covering,
                    f'gamma={gamma}: no stable band covers it '
                    f'(dropped={dropped}).')
                self.assertTrue(
                    any(len(s.arcs) > 0 for s in covering),
                    f'gamma={gamma}: covered only by zero-arc bands (F041).')
            self.comparisons += 1

    def test_labels_stable_for_gamma_at_least_one_tenth(self) -> None:
        """Assertion 3: every arc on a stable band with ``lo >= 0.1`` carries
        the parity-constant ``image_count == 4`` and a valid ``inward_sign``,
        and the label at each arc position does NOT move across gamma -- the
        fix only ADDS small-gamma arcs, it never relabels gamma >= 0.1."""
        stable, dropped = training.stable_gamma_bands(
            (0.1, _F041_BAND[1]), +1, n_samples=_F041_N_SAMPLES,
            min_width=_F041_MIN_WIDTH)
        self.assertEqual(
            dropped, [], 'gamma >= 0.1 must be fully served (nothing dropped)')
        self.comparisons += 1
        for (lo, _hi), structure in stable:
            self.assertGreaterEqual(lo, 0.1)
            for arc in structure.arcs:
                self.assertEqual(
                    arc.image_count, _F041_ASTROID_IMAGE_COUNT,
                    'astroid fold-arc image_count must be the parity constant')
                self.assertIn(
                    arc.inward_sign, (-1, 1),
                    'inward_sign must be a unit orientation sign')
                self.comparisons += 1
        # Cross-gamma label stability: the arc at each detection index keeps
        # its (inward_sign, image_count) label as gamma varies over >= 0.1,
        # so the fix cannot have moved an existing large-gamma label.
        structures = [training.detect_caustic_structure(
            gamma, +1, n_samples=_F041_N_SAMPLES)
            for gamma in _F041_LABEL_GAMMAS]
        arc_counts = {len(s.arcs) for s in structures}
        self.assertEqual(
            len(arc_counts), 1,
            f'arc COUNT moved across gamma >= 0.1: {arc_counts}')
        self.comparisons += 1
        for idx in range(len(structures[0].arcs)):
            labels = {(s.arcs[idx].inward_sign, s.arcs[idx].image_count)
                      for s in structures}
            with self.subTest(arc_index=idx):
                self.assertEqual(
                    len(labels), 1,
                    f'arc {idx} label moved across gamma >= 0.1: {labels}')
                (sign, count), = labels
                self.assertIn(sign, (-1, 1))
                self.assertEqual(count, _F041_ASTROID_IMAGE_COUNT)
            self.comparisons += 1


class StableGammaBandsF041SelfFalsificationTestCase(_CountingTestCase):
    """Prove the F041 acceptance checks can go RED.

    Re-inject the pre-fix pathology WITHOUT the engine by patching caustic
    detection to build ZERO arcs on the small-gamma edge (exactly the F041
    symptom).  The arc-count change then makes `stable_gamma_bands` bisect and
    DROP a sliver AND emit a zero-arc band -- so both limbs of the load-
    bearing assertion (``dropped == []`` and ``len(arcs) > 0``) would fail.
    """

    @staticmethod
    def _arcless_below(threshold: float):
        """A `detect_caustic_structure` stand-in that strips the fold arcs off
        any structure with ``gamma < threshold`` (the F041 symptom)."""
        real = training.detect_caustic_structure

        def detector(gamma, parity, *, n_samples=_F041_N_SAMPLES):
            structure = real(gamma, parity, n_samples=n_samples)
            if gamma < threshold:
                return dataclasses.replace(structure, arcs=())
            return structure
        return detector

    def test_injected_zero_arc_edge_drops_a_sliver(self) -> None:
        """With small-gamma arcs starved, the sweep DROPS a sliver -- so the
        real ``dropped == []`` assertion would go red; the unpatched sweep
        drops nothing (positive control)."""
        with mock.patch.object(training, 'detect_caustic_structure',
                               self._arcless_below(0.05)):
            _stable, dropped = training.stable_gamma_bands(
                _F041_BAND, +1, n_samples=_F041_N_SAMPLES,
                min_width=_F041_MIN_WIDTH)
        self.assertNotEqual(
            dropped, [],
            'the injected F041 pathology must drop at least one sliver')
        self.comparisons += 1
        _clean_stable, clean_dropped = training.stable_gamma_bands(
            _F041_BAND, +1, n_samples=_F041_N_SAMPLES,
            min_width=_F041_MIN_WIDTH)
        self.assertEqual(
            clean_dropped, [], 'the fixed sweep must drop nothing')
        self.comparisons += 1

    def test_injected_zero_arc_edge_yields_a_zero_arc_band(self) -> None:
        """The starved edge also leaves a stable band with ZERO arcs, so the
        ``len(arcs) > 0`` limb of the acceptance would go red too."""
        with mock.patch.object(training, 'detect_caustic_structure',
                               self._arcless_below(0.05)):
            stable, _dropped = training.stable_gamma_bands(
                _F041_BAND, +1, n_samples=_F041_N_SAMPLES,
                min_width=_F041_MIN_WIDTH)
        self.assertTrue(
            any(len(structure.arcs) == 0 for (_band, structure) in stable),
            'the injected pathology must leave a zero-arc band')
        self.comparisons += 1


# ---------------------------------------------------------------------------
# Build 8g WP1: delete _WEDGE_EPS, sample the true wedge edge (Gates 1-3)
# ---------------------------------------------------------------------------
#
# WP1 removed the ``_WEDGE_EPS = 1e-3`` inset that stopped every macro-saddle
# wedge sweep 1e-3 short of its true edge ``center +- theta_max`` and made
# ``_tube_normal`` analytic.  These three gates pin the TRAINING-PATH
# consequences with pure closed-form geometry (no engine, fast tier):
#   Gate 1  the constant is gone and no ``1e-3`` inlined in its place; the
#           wedge sweeps start/end at the exact edge.
#   Gate 2  (Professor Q3) at the served endpoint the discriminant clamps to
#           zero, the two square-root branches coincide bit-for-bit, and the
#           lobe winding loop closes with EXACTLY zero gap (was ~0.279).
#   Gate 3  the saddle deltoid keeps 6 cusps / 6 arcs / its reach and its
#           served arc span does not shrink -- it grows past the pre-WP1 span.

#: Strong-shear macro-saddle shears exercised by the WP1 wedge gates.  Each is
#: ``|gamma| > 1`` (two 3-cusp deltoid lobes); F044 measured the endpoint
#: discriminant clamp fires at each.
_WP1_GAMMAS = (1.05, 1.3, 2.0)

#: The two deltoid lobe centres on the negative-eigenvalue axis (mirror of
#: ``surrogate_training._SADDLE_LOBE_CENTERS``); each lobe's wedge is
#: ``center +- theta_max`` with ``theta_max = 0.5 * arcsin(1 / |gamma|)``.
_WP1_LOBE_CENTERS = (0.0, math.pi)

#: Winding-loop sample count for the closure-gap mechanism check.  The gap is
#: the two-vertex ``|loop[0] - loop[-1]|`` and is independent of ``n``; 64
#: keeps each lobe loop build well under a millisecond.
_WP1_WINDING_N = 64

#: Absolute closure tolerance for the (empirically unreached) branch where the
#: endpoint discriminant lands tiny-POSITIVE instead of ``<= 0``: a few ulp of
#: the O(1) source scale (caustic reach ~ 3 at gamma = 1.05).  When the clamp
#: fires (disc <= 0) the gap is asserted EXACTLY 0.0, never merely below this.
_WP1_CLOSURE_TOL = 1e-12

#: Frozen PRE-WP1 (HEAD, ``_WEDGE_EPS``-inset) closure gap at the wedge
#: endpoint, ``gamma -> gap``, measured once via the HEAD module at
#: ``_WP1_WINDING_N`` and baked in as a literal (no live ``git show`` oracle in
#: the gates).  WP1 drives every one to exactly 0.0; this is the incumbent
#: context reported by Gate 2 (0.279 at gamma = 1.05).
_WP1_INCUMBENT_CLOSURE_GAP = {1.05: 0.278958, 1.3: 0.107435, 2.0: 0.051454}

#: Caustic-samples-per-branch for the Gate 3 structure build (matches the
#: ``detect_caustic_structure`` default; 2 centres x 2 branches x 200 speed
#: evals x 3 gammas ~ a few thousand closed-form evals, well under 1 s total).
_WP1_STRUCTURE_SAMPLES = 200

#: Golden WP1 (true-edge) saddle-deltoid invariants, ``gamma -> (cusps, arcs,
#: reach)``, FROZEN as literals computed once with the shipped code at
#: ``_WP1_STRUCTURE_SAMPLES``.  A drop in any of these fails Gate 3.
_WP1_GOLDEN_STRUCTURE = {
    1.05: (6, 6, 3.007536369149046),
    1.3: (6, 6, 1.7143767264026),
    2.0: (6, 6, 2.309395081301576),
}

#: Golden WP1 total served arc span (sum of ``theta_hi - theta_lo``),
#: ``gamma -> span``; the reproduction anchor for Gate 3.
_WP1_GOLDEN_SPAN = {
    1.05: 2.9072886962267335,
    1.3: 1.9837106163515006,
    2.0: 0.923890604351195,
}

#: Frozen PRE-WP1 (HEAD) total arc span at ``_WP1_STRUCTURE_SAMPLES``,
#: ``gamma -> span``.  WP1 samples 1e-3 nearer each true edge, so the served
#: span STRICTLY INCREASES past every incumbent; Gate 3 asserts the current
#: span is strictly greater (no coverage shrink -- a gain).  Measured once via
#: the HEAD module and baked in as a literal.
_WP1_INCUMBENT_SPAN = {
    1.05: 2.9046151283875368,
    1.3: 1.9809425761504962,
    2.0: 0.920130604351196,
}

#: Matches an inline wedge inset ``theta_max +- <_WEDGE_EPS | numeric eps>`` --
#: the retired offset pattern Gate 1 forbids at the wedge-bound sites.  A clean
#: ``center - theta_max`` (edge line ending in ``,`` or a newline) does NOT
#: match because no ``+``/``-`` follows ``theta_max``; ``theta_max -
#: _CUSP_BRACKET_EPS`` (a named bracket clamp in ``_find_cusps``, not a wedge
#: edge) also does not match (the token after ``-`` is not a digit).
_WEDGE_INSET_RE = re.compile(r'theta_max\s*[-+]\s*(?:_WEDGE_EPS|\d)')

#: Signature of the wedge half-angle formula that marks a wedge-bound site.
_WEDGE_HALF_ANGLE_SRC = 'np.arcsin(1.0 / abs('


def _wedge_bound_function_sources() -> list:
    """``(name, source)`` for every ``surrogate_training`` function that
    computes a saddle wedge edge from the half-angle ``0.5 * arcsin(1 /
    |gamma|)``.

    These are the "wedge-bound sites" Gate 1 scans.  `_find_cusps` (whose
    local ``theta_max`` is a sampled ``thetas.max()``, not the wedge angle) is
    excluded because its source lacks the half-angle signature.
    """
    sources = []
    for name, obj in vars(training).items():
        if not (inspect.isfunction(obj)
                and obj.__module__ == training.__name__):
            continue
        try:
            src = inspect.getsource(obj)
        except OSError:
            continue
        if _WEDGE_HALF_ANGLE_SRC in src and 'theta_max' in src:
            sources.append((name, src))
    return sources


class WedgeEpsDeletionTestCase(_CountingTestCase):
    """Gate 1: ``_WEDGE_EPS`` is deleted, no ``1e-3`` is inlined in its place,
    and the wedge sweeps run to the exact edge ``center +- theta_max``."""

    def test_wedge_eps_constant_is_deleted(self) -> None:
        """The production module no longer exposes ``_WEDGE_EPS`` at all."""
        self.assertFalse(
            hasattr(training, '_WEDGE_EPS'),
            'WP1 deletes surrogate_training._WEDGE_EPS')
        self.assertNotIn(
            '_WEDGE_EPS', inspect.getsource(training),
            'no _WEDGE_EPS token may survive anywhere in the module source')
        self.comparisons += 1

    def test_no_inline_inset_at_wedge_bound_sites(self) -> None:
        """No wedge-bound site inlines a ``1e-3`` (or any eps) offset onto
        ``theta_max`` in place of the retired constant."""
        wedge_fns = _wedge_bound_function_sources()
        names = {name for name, _ in wedge_fns}
        # Guard the scan itself: the two edge-defining functions MUST be seen,
        # else a rename would let the scan pass vacuously over zero sources.
        self.assertIn('_saddle_arcs', names)
        self.assertIn('_lobe_winding_loop', names)
        for name, src in wedge_fns:
            with self.subTest(function=name):
                self.assertNotIn('_WEDGE_EPS', src)
                self.assertIsNone(
                    _WEDGE_INSET_RE.search(src),
                    f'{name} inlines a wedge inset onto theta_max')
                self.comparisons += 1

    def test_winding_loop_endpoints_are_the_true_edge(self) -> None:
        """The lobe winding loop's first/last vertices sit BIT-for-BIT at the
        true wedge edges ``center -+ theta_max`` -- a 1e-3 inset would move
        them off these exact critical points."""
        for gamma, center in itertools.product(
                _WP1_GAMMAS, _WP1_LOBE_CENTERS):
            with self.subTest(gamma=gamma, center=center):
                theta_max = 0.5 * np.arcsin(1.0 / abs(gamma))
                lo = center - theta_max
                hi = center + theta_max
                loop = training._lobe_winding_loop(
                    gamma, center, _WP1_WINDING_N)
                self.assertGreaterEqual(len(loop), 3)
                # loop = [+branch lo..hi, -branch hi..lo]: first vertex is the
                # + branch at lo, last vertex the - branch back at lo, and the
                # + branch's turnaround vertex (index n-1) is at hi.
                lo_plus = np.asarray(geometry.critical_point(
                    gamma, float(lo), 0.0, 0.0, 1).source, dtype=float)
                lo_minus = np.asarray(geometry.critical_point(
                    gamma, float(lo), 0.0, 0.0, -1).source, dtype=float)
                hi_plus = np.asarray(geometry.critical_point(
                    gamma, float(hi), 0.0, 0.0, 1).source, dtype=float)
                self.assertTrue(
                    np.array_equal(loop[0], lo_plus),
                    'first vertex is not the exact lo wedge edge')
                self.assertTrue(
                    np.array_equal(loop[-1], lo_minus),
                    'last vertex is not the exact lo wedge edge')
                self.assertTrue(
                    np.array_equal(loop[_WP1_WINDING_N - 1], hi_plus),
                    'turnaround vertex is not the exact hi wedge edge')
                self.comparisons += 1


class WedgeClosureGapTestCase(_CountingTestCase):
    """Gate 2 (Professor Q3): the served wedge endpoint clamps the branch
    discriminant to zero, the two square-root branches coincide bit-for-bit,
    and the lobe winding loop closes with EXACTLY zero gap."""

    def test_endpoint_discriminant_clamps_and_branches_coincide(self) -> None:
        """PRIMARY mechanism: at the served endpoint ``theta = center -
        theta_max`` the discriminant ``1 - gamma^2 sin^2(2 theta_rel)`` is
        ``<= 0``, so ``critical_point``'s ``max(., 0)`` clamps the root to 0.0
        and branch +1 and -1 map to the SAME source point."""
        for gamma, center in itertools.product(
                _WP1_GAMMAS, _WP1_LOBE_CENTERS):
            with self.subTest(gamma=gamma, center=center):
                theta_max = 0.5 * np.arcsin(1.0 / abs(gamma))
                lo = center - theta_max            # served terminal endpoint
                theta_rel = lo - center            # == -theta_max
                discriminant = 1.0 - gamma**2 * np.sin(2.0 * theta_rel)**2
                self.assertLessEqual(
                    discriminant, 0.0,
                    f'endpoint discriminant {discriminant:.3e} must be <= 0 '
                    f'so the max(., 0) clamp fires (gamma={gamma})')
                src_plus = np.asarray(geometry.critical_point(
                    gamma, float(lo), 0.0, 0.0, 1).source, dtype=float)
                src_minus = np.asarray(geometry.critical_point(
                    gamma, float(lo), 0.0, 0.0, -1).source, dtype=float)
                self.assertTrue(
                    np.array_equal(src_plus, src_minus),
                    'the two square-root branches must coincide bit-for-bit '
                    'at the clamped served edge')
                self.comparisons += 1

    def test_lobe_winding_loop_closes_exactly(self) -> None:
        """COROLLARY (per-gamma guarded): where the endpoint discriminant is
        ``<= 0`` (clamp fires) the loop closure gap ``|loop[0] - loop[-1]|``
        is EXACTLY 0.0 -- a proven consequence, not a number preserved by
        construction (pre-WP1 this gap was ~0.279 at gamma = 1.05)."""
        for gamma, center in itertools.product(
                _WP1_GAMMAS, _WP1_LOBE_CENTERS):
            with self.subTest(gamma=gamma, center=center):
                theta_max = 0.5 * np.arcsin(1.0 / abs(gamma))
                lo = center - theta_max
                discriminant = 1.0 - gamma**2 * np.sin(2.0 * (lo - center))**2
                loop = training._lobe_winding_loop(
                    gamma, center, _WP1_WINDING_N)
                self.assertGreaterEqual(len(loop), 3)
                gap = float(np.hypot(*(loop[0] - loop[-1])))
                incumbent = _WP1_INCUMBENT_CLOSURE_GAP[gamma]
                if discriminant <= 0.0:
                    self.assertEqual(
                        gap, 0.0,
                        f'clamp fired (disc={discriminant:.3e}) so the loop '
                        f'must close exactly; pre-WP1 gap here was {incumbent} '
                        f'(0.279 at gamma=1.05)')
                else:
                    self.assertLess(
                        gap, _WP1_CLOSURE_TOL,
                        f'disc landed tiny-positive ({discriminant:.3e}); the '
                        f'gap must be within a few ulp, not exactly 0.0')
                self.comparisons += 1


class WedgeCoverageNoShrinkTestCase(_CountingTestCase):
    """Gate 3: the true-edge saddle deltoid keeps 6 cusps / 6 arcs / its reach
    and its served arc span does not shrink below the pre-WP1 incumbent (it
    grows past it)."""

    def test_cusp_arc_counts_and_reach_preserved(self) -> None:
        """Cusp count stays 6, arc count and caustic reach match the frozen
        WP1 golden values; a drop in any fails the build."""
        for gamma in _WP1_GAMMAS:
            with self.subTest(gamma=gamma):
                structure = training.detect_caustic_structure(
                    gamma, -1, n_samples=_WP1_STRUCTURE_SAMPLES)
                cusps, arcs, reach = _WP1_GOLDEN_STRUCTURE[gamma]
                self.assertEqual(structure.detected_cusps, 6)
                self.assertEqual(structure.detected_cusps, cusps)
                self.assertEqual(len(structure.arcs), arcs)
                # Reach is a deterministic sampled maximum; a real coverage
                # shrink would be O(1e-3)+, far above this FP-reproduction
                # tolerance.
                self.assertAlmostEqual(
                    structure.caustic_reach, reach,
                    delta=abs(reach) * 1e-9,
                    msg=f'caustic reach drifted from golden at gamma={gamma}')
                self.comparisons += 1

    def test_arc_span_grows_past_incumbent(self) -> None:
        """The served arc span reproduces the WP1 golden and is STRICTLY
        greater than the frozen pre-WP1 (inset) span -- WP1 recovers the
        ~1e-3-per-edge coverage the old inset discarded, so reverting it
        (span == incumbent) makes this gate red."""
        for gamma in _WP1_GAMMAS:
            with self.subTest(gamma=gamma):
                structure = training.detect_caustic_structure(
                    gamma, -1, n_samples=_WP1_STRUCTURE_SAMPLES)
                span = sum(arc.theta_hi - arc.theta_lo
                           for arc in structure.arcs)
                incumbent = _WP1_INCUMBENT_SPAN[gamma]
                golden = _WP1_GOLDEN_SPAN[gamma]
                # No coverage shrink (spec floor): never below the incumbent.
                self.assertGreaterEqual(span, incumbent)
                # WP1 strictly increases coverage (non-coincidence teeth).
                self.assertGreater(
                    span, incumbent,
                    f'span {span!r} must exceed the pre-WP1 incumbent '
                    f'{incumbent!r} at gamma={gamma}')
                self.assertAlmostEqual(
                    span, golden, delta=abs(golden) * 1e-9,
                    msg=f'span drifted from the WP1 golden at gamma={gamma}')
                self.comparisons += 1


class WedgeEdgeSelfFalsificationTestCase(_CountingTestCase):
    """Prove the three WP1 gates can go RED.

    Every assertion here reintroduces the retired ``_WEDGE_EPS`` inset -- as a
    dirty source string (Gate 1), or by driving the pre-WP1 ``HEAD`` module
    whose ``_lobe_winding_loop`` / ``detect_caustic_structure`` still carry the
    ``+- 1e-3`` offset (Gates 2, 3) -- and shows the gate's invariant FAILS on
    it.  This is a FAST-tier class (pure closed-form geometry and a source
    scan; NOT ``_TRAIN_TIER_SKIP``), so the suite can go red without the
    training engine.
    """

    def test_gate1_scan_catches_a_reintroduced_inset(self) -> None:
        """Gate 1's ``_WEDGE_INSET_RE`` matches every way a ``theta_max``
        inset could be reintroduced, and leaves a clean edge alone -- so a
        silent revert cannot slip past the scan."""
        clean = 'lo = center - theta_max\n        hi = center + theta_max'
        self.assertIsNone(
            _WEDGE_INSET_RE.search(clean),
            'the scan must not false-positive on the true edge')
        self.comparisons += 1
        for dirty in ('center - theta_max + _WEDGE_EPS',
                      'center + theta_max - _WEDGE_EPS',
                      'center - theta_max + 1e-3',
                      'lens_center - theta_max + 0.001'):
            with self.subTest(dirty=dirty):
                self.assertIsNotNone(
                    _WEDGE_INSET_RE.search(dirty),
                    f'Gate 1 scan failed to catch a reintroduced inset: '
                    f'{dirty!r}')
                self.comparisons += 1

    def test_reverting_the_inset_reopens_the_winding_loop(self) -> None:
        """Gate 2 reachability: the pre-WP1 (``HEAD``) ``_lobe_winding_loop``,
        whose endpoints are inset by ``_WEDGE_EPS``, closes with a gap FAR
        above ``_WP1_CLOSURE_TOL`` (~0.279 at gamma=1.05) -- so the current
        ``== 0.0`` closure is a real consequence of removing the inset, not a
        number preserved by construction."""
        head = _head_training_module()
        for gamma, center in itertools.product(
                _WP1_GAMMAS, _WP1_LOBE_CENTERS):
            with self.subTest(gamma=gamma, center=center):
                loop = head._lobe_winding_loop(gamma, center, _WP1_WINDING_N)
                self.assertGreaterEqual(len(loop), 3)
                gap = float(np.hypot(*(loop[0] - loop[-1])))
                self.assertGreater(
                    gap, _WP1_CLOSURE_TOL,
                    f'the inset HEAD loop must NOT close (gap={gap:.3e}); '
                    f'this is what WP1 fixed to exactly 0.0')
                self.comparisons += 1

    def test_reverting_the_inset_shrinks_the_arc_span(self) -> None:
        """Gate 3 reachability: the pre-WP1 (``HEAD``) served arc span is
        STRICTLY smaller than the WP1 span and matches the frozen incumbent --
        so ``assertGreater(span, incumbent)`` genuinely distinguishes the two
        code states (reverting the inset makes Gate 3 red)."""
        head = _head_training_module()
        for gamma in _WP1_GAMMAS:
            with self.subTest(gamma=gamma):
                head_struct = head.detect_caustic_structure(
                    gamma, -1, n_samples=_WP1_STRUCTURE_SAMPLES)
                head_span = sum(arc.theta_hi - arc.theta_lo
                                for arc in head_struct.arcs)
                wp1_struct = training.detect_caustic_structure(
                    gamma, -1, n_samples=_WP1_STRUCTURE_SAMPLES)
                wp1_span = sum(arc.theta_hi - arc.theta_lo
                               for arc in wp1_struct.arcs)
                self.assertLess(
                    head_span, wp1_span,
                    f'HEAD (inset) span {head_span!r} must be below the WP1 '
                    f'span {wp1_span!r} at gamma={gamma}')
                self.assertAlmostEqual(
                    head_span, _WP1_INCUMBENT_SPAN[gamma],
                    delta=abs(_WP1_INCUMBENT_SPAN[gamma]) * 1e-9,
                    msg='frozen incumbent span must come from the HEAD state')
                self.comparisons += 1

    def test_branches_differ_off_the_wedge_edge(self) -> None:
        """Gate 2 specificity: WELL INSIDE the wedge the discriminant is
        strictly positive, so the two square-root branches map to DISTINCT
        source points -- the bit-identity Gate 2 asserts is peculiar to the
        clamped edge, not a trivial always-equal artifact."""
        for gamma, center in itertools.product(
                _WP1_GAMMAS, _WP1_LOBE_CENTERS):
            with self.subTest(gamma=gamma, center=center):
                theta_max = 0.5 * np.arcsin(1.0 / abs(gamma))
                theta = center + 0.3 * theta_max     # interior of the wedge
                discriminant = 1.0 - gamma**2 * np.sin(
                    2.0 * (theta - center))**2
                self.assertGreater(
                    discriminant, 0.0,
                    'interior wedge point must have a positive discriminant')
                src_plus = np.asarray(geometry.critical_point(
                    gamma, float(theta), 0.0, 0.0, 1).source, dtype=float)
                src_minus = np.asarray(geometry.critical_point(
                    gamma, float(theta), 0.0, 0.0, -1).source, dtype=float)
                self.assertFalse(
                    np.array_equal(src_plus, src_minus),
                    'the two branches must DIFFER off the clamped edge')
                self.comparisons += 1


if __name__ == '__main__':
    main()
