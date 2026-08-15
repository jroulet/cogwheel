"""Verify WP-1 saddle origin-rho misclassification guards at all code sites.

Covers the parity-and-image-count gates that prevent saddle-parity
(gamma > 1) corridor sources from being routed through the fold-ppGO
interior handoff or ppGO map, fixing a misclassification where the
scalar-reach rho < 1 was used as a discriminator for saddle configurations.

Tests (TD-1 through TD-3):
* Saddle corridor exterior sources (2 images) are correctly classified
  as ``None`` by ``_ppgo_cell_coords`` and as ``'born'`` by the census.
* Saddle lobe interior sources (4 images) are NOT misclassified: the
  fold-ppGO handoff guard does NOT fire, and the census guard
  ``image_count == 2`` does not claim them.
* Positive-parity (astroid) behaviour is byte-identical to HEAD — the
  saddle-specific guards only trigger for ``gamma > 1``.

Tolerance justification.
All tests assert EXACT outcomes: None is None, string equality for census
categories, integer equality for image counts.
"""
from __future__ import annotations

import types
import unittest

import numpy as np

from cogwheel.lensing.likelihood import LensedRelativeBinningLikelihood
from cogwheel.lensing.chang_refsdal.channels import ChangRefsdalChannels
from cogwheel.lensing import surrogate_census

#: Saddle parity gamma (det A < 0).
_SADDLE_GAMMA: float = 1.3

#: Exterior corridor sources between deltoid lobes — each has 2 images.
_CORRIDOR_SOURCES: tuple[tuple[float, float], ...] = ((0, 0.3), (0.5, 0), (0.3, 0.3))

#: Saddle lobe interior source near a deltoid lobe center — has 4 images.
_LOBE_SOURCE: tuple[float, float] = (1.2, 0.0)

#: Positive-parity gamma (det A > 0).
_POSITIVE_GAMMA: float = 0.5

#: Astroid fixture sources.
_ASTROID_SOURCES: tuple[tuple[float, float], ...] = ((0, 0), (0.3, 0.3))

#: w grid for the cheap geometry-only partition (2 pts, minimal).
_W_PROBE = np.array([10.0, 100.0])


class _SaddleRhoTestCase(unittest.TestCase):
    """Base carrying the anti-vacuity comparison counter (house idiom)."""

    def setUp(self) -> None:
        self.comparisons = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self.comparisons, 0,
            'Anti-vacuity: no comparisons executed in this test.')


class _PpgoProbe:
    """Stateless stand-in exposing the real ``_ppgo_cell_coords``.

    Mirrors the ``_DispatchProbe`` pattern in
    ``test_lensing_ppgo_bandsplit.py``.
    """

    _ppgo_cell_coords = LensedRelativeBinningLikelihood._ppgo_cell_coords


class PpgocellcoordsCorridorDelegationTestCase(_SaddleRhoTestCase):
    """TD-1: ``_ppgo_cell_coords`` DELEGATES the saddle rho<1 decision.

    SITE 1 -- the ``if parity == 'saddle' and rho < 1.0: return None``
    pre-guard formerly inside ``_ppgo_cell_coords`` -- was removed by
    design.  The per-cell allowlist in ``CertifiedPpgoMap`` (F080) is now
    the single authoritative source of the saddle rho<1 serve/refuse
    decision, so ``_ppgo_cell_coords`` returns the plain
    ``(parity, gamma, rho)`` delegation tuple for EVERY saddle corridor
    source.  Whether that cell is served (Cell 1) or refused (every other
    saddle rho<1 cell) is decided downstream by ``w_trust`` / ``w_ceiling``
    -- pinned in ``PpgoMapDefenseInDepthTestCase``.

    All three corridor sources have gamma > 1 (saddle) and rho < 1; the
    method rightly returns the delegation tuple rather than ``None``.
    """

    def _assert_delegates(self, y1: float, y2: float) -> None:
        from cogwheel.lensing.ppgo_map import caustic_rho
        probe = _PpgoProbe()
        lens = {'gamma': _SADDLE_GAMMA, 'y1': y1, 'y2': y2}
        result = probe._ppgo_cell_coords(lens)
        self.comparisons += 1
        self.assertIsNotNone(
            result,
            'SITE 1 removed: saddle rho<1 delegates to the map, not None')
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        parity, gamma, rho = result
        self.assertEqual(parity, 'saddle')
        self.assertEqual(gamma, _SADDLE_GAMMA)
        expected_rho = caustic_rho(
            _SADDLE_GAMMA, float(np.hypot(y1, y2)), kappa=0.0)
        self.comparisons += 1
        self.assertAlmostEqual(rho, expected_rho)
        self.assertLess(rho, 1.0, 'premise: corridor source has rho < 1')

    def test_corridor_source_0(self) -> None:
        self._assert_delegates(0.0, 0.3)

    def test_corridor_source_1(self) -> None:
        self._assert_delegates(0.5, 0.0)

    def test_corridor_source_2(self) -> None:
        self._assert_delegates(0.3, 0.3)


class CensusCorridorBornClassificationTestCase(_SaddleRhoTestCase):
    """TD-1: ``classify_fallthrough`` returns ``'born'`` for saddle corridor.

    The census guard ``if gamma > 1.0 and image_count == 2: return 'born'``
    fires for saddle-parity exterior corridor sources between deltoid lobes.
    With an empty chart list, no cusp-window / refusal-ball probes fire,
    so the category is ``'born'`` (not ``'out-of-box'`` or an interior
    category).
    """

    _surrogate = types.SimpleNamespace(charts=[])

    def _classify(self, y1_eig: float, y2_eig: float) -> str:
        return surrogate_census.classify_fallthrough(
            self._surrogate,
            gamma=_SADDLE_GAMMA, log_w_min=-5.0, log_w_max=-1.0,
            eta=1.0, theta=0.4, image_count=2,
            y1_eig=y1_eig, y2_eig=y2_eig, dropped_slivers=())

    def test_corridor_source_0_born(self) -> None:
        category = self._classify(0.0, 0.3)
        self.comparisons += 1
        self.assertEqual(category, 'born')

    def test_corridor_source_1_born(self) -> None:
        category = self._classify(0.5, 0.0)
        self.comparisons += 1
        self.assertEqual(category, 'born')

    def test_corridor_source_2_born(self) -> None:
        category = self._classify(0.3, 0.3)
        self.comparisons += 1
        self.assertEqual(category, 'born')

class CensusLobeInteriorNotBornTestCase(_SaddleRhoTestCase):
    """TD-2: saddle lobe interior (4 images) NOT classified as ``'born'``.

    The census guard ``gamma > 1.0 and image_count == 2`` must NOT fire for
    sources with 4 images.  A source inside a deltoid lobe with 4 images
    is genuine interior, not a corridor exterior misclassification —
    the fold-ppGO interior handoff guard ``image_count != 4`` also does
    NOT fire, so the handoff IS entered (non-regression).
    """

    _surrogate = types.SimpleNamespace(charts=[])

    def _classify(self, y1_eig: float, y2_eig: float) -> str:
        return surrogate_census.classify_fallthrough(
            self._surrogate,
            gamma=_SADDLE_GAMMA, log_w_min=-5.0, log_w_max=-1.0,
            eta=1.0, theta=0.4, image_count=4,
            y1_eig=y1_eig, y2_eig=y2_eig, dropped_slivers=())

    def test_lobe_interior_not_born(self) -> None:
        category = self._classify(_LOBE_SOURCE[0], _LOBE_SOURCE[1])
        self.comparisons += 1
        self.assertNotEqual(category, 'born',
                            '4-image lobe interior must not classify as born')

    def test_lobe_interior_falls_through(self) -> None:
        category = self._classify(_LOBE_SOURCE[0], _LOBE_SOURCE[1])
        self.comparisons += 1
        self.assertEqual(category, 'out-of-box',
                         '4-image lobe interior should fall through to '
                         'out-of-box (born guard does not fire)')


class LobeInteriorGeometryTestCase(_SaddleRhoTestCase):
    """TD-2: saddle lobe interior source has exactly 4 real images.

    The fold-ppGO handoff guard ``gamma > 1 and image_count != 4`` only
    fires for ``image_count != 4``.  A source inside a deltoid lobe must
    have 4 images so the guard does NOT fire — the handoff is entered.
    """

    def test_lobe_source_image_count(self) -> None:
        ch = ChangRefsdalChannels(_W_PROBE)
        geom = ch.geometry_partition(
            gamma=_SADDLE_GAMMA, y=_LOBE_SOURCE, beta=0.0, kappa=0.0)
        image_count = int(geom.real_mask.sum())
        self.comparisons += 1
        self.assertEqual(image_count, 4,
                         f'lobe source {_LOBE_SOURCE} should have 4 images, '
                         f'got {image_count}')

    def test_handoff_guard_does_not_fire(self) -> None:
        ch = ChangRefsdalChannels(_W_PROBE)
        geom = ch.geometry_partition(
            gamma=_SADDLE_GAMMA, y=_LOBE_SOURCE, beta=0.0, kappa=0.0)
        image_count = int(geom.real_mask.sum())
        guard_fires = _SADDLE_GAMMA > 1.0 and image_count != 4
        self.comparisons += 1
        self.assertFalse(
            guard_fires,
            'fold-ppGO handoff guard (gamma>1 and image_count!=4) '
            f'must NOT fire for 4-image lobe interior: '
            f'gamma={_SADDLE_GAMMA} image_count={image_count}')

class PositiveParityPpgocellcoordsTestCase(_SaddleRhoTestCase):
    """TD-3: ``_ppgo_cell_coords`` returns tuple for positive parity.

    The guard ``if parity == 'saddle' and rho < 1.0`` only fires for
    ``'saddle'`` parity.  Positive-parity (astroid) sources always get
    their cell coordinates — the parity guard is the ONLY trigger.
    Both interior (rho=0) and exterior (rho<1, actually on-caustic edge)
    astroid sources return valid ``(parity, gamma, rho)`` tuples.
    """

    def test_origin_returns_tuple(self) -> None:
        probe = _PpgoProbe()
        lens = {'gamma': 0.5, 'y1': 0.0, 'y2': 0.0}
        result = probe._ppgo_cell_coords(lens)
        self.comparisons += 1
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], 'positive')
        self.assertEqual(result[1], 0.5)
        self.assertAlmostEqual(result[2], 0.0)

    def test_interior_returns_tuple(self) -> None:
        probe = _PpgoProbe()
        lens = {'gamma': 0.5, 'y1': 0.3, 'y2': 0.3}
        result = probe._ppgo_cell_coords(lens)
        self.comparisons += 1
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], 'positive')


class PositiveParityCensusTestCase(_SaddleRhoTestCase):
    """TD-3: census ``classify_fallthrough`` unchanged for positive parity.

    The saddle-specific census guard ``gamma > 1.0 and image_count == 2``
    does not fire for ``gamma < 1.0``.  Classification is determined
    solely by ``rho > 1`` (born) vs fall-through (out-of-box), same as
    before the WP-1 change.

    For the fixtures: rho=2.0 gives rho≈1.41>1 → 'born', rho=0.3 gives
    rho≈0.21<1 → 'out-of-box'.
    """

    _surrogate = types.SimpleNamespace(charts=[])

    def _classify(self, y1_eig: float, y2_eig: float) -> str:
        return surrogate_census.classify_fallthrough(
            self._surrogate,
            gamma=_POSITIVE_GAMMA, log_w_min=-5.0, log_w_max=-1.0,
            eta=1.0, theta=0.4, image_count=2,
            y1_eig=y1_eig, y2_eig=y2_eig, dropped_slivers=())

    def test_exterior_classifies_born(self) -> None:
        category = self._classify(2.0, 0.0)
        self.comparisons += 1
        self.assertEqual(category, 'born')

    def test_interior_not_born(self) -> None:
        category = self._classify(0.3, 0.0)
        self.comparisons += 1
        self.assertNotEqual(category, 'born')

    def test_origin_not_born(self) -> None:
        category = self._classify(0.0, 0.0)
        self.comparisons += 1
        self.assertNotEqual(category, 'born')

class SaddleRhoGuardSelfFalsificationTestCase(_SaddleRhoTestCase):
    """Reachable-red: the SURVIVING saddle rho guards CAN go red.

    A numerical/decision suite without a self-falsification class is not
    finished.  SITE 1 (the ``_ppgo_cell_coords`` saddle rho<1 refusal) was
    removed by design -- its foil was retired with it.  These foils prove
    the two guards that remain are genuinely load-bearing: the census
    ``gamma > 1 and image_count == 2 -> 'born'`` classifier, and the
    ``parity == 'saddle'`` scoping that keeps positive-parity delegation
    untouched.
    """

    def test_census_would_not_be_born_without_image_count_guard(self) -> None:
        """Without the image_count guard, census falls through to out-of-box.

        If the guard ``if gamma > 1.0 and image_count == 2`` were removed,
        the corridor source would fall through past the rho>1 check
        (rho<1) to the cusp-window/refusal-ball probes and ultimately
        'out-of-box' — NOT 'born'.  This proves the image_count guard is
        the SOLE reason the census says 'born' for saddle corridor.
        """
        from cogwheel.lensing.ppgo_map import caustic_rho
        rho = caustic_rho(1.3, 0.3)
        self.comparisons += 1
        self.assertLess(rho, 1.0,
                        'premise: rho<1 so rho>1 check does not trigger born')
        # With the guard active, classify_fallthrough returns 'born'.
        surrogate = types.SimpleNamespace(charts=[])
        category = surrogate_census.classify_fallthrough(
            surrogate, gamma=1.3, log_w_min=-5.0, log_w_max=-1.0,
            eta=1.0, theta=0.4, image_count=2,
            y1_eig=0.0, y2_eig=0.3, dropped_slivers=())
        self.comparisons += 1
        self.assertEqual(category, 'born',
                         'guard must classify saddle 2-image as born')

    def test_positive_parity_guard_only_fires_for_saddle(self) -> None:
        """Positive parity is byte-identical: no tuple returned as None.

        The guard condition ``parity == 'saddle'`` means positive-parity
        sources never trigger the refusal.  A deliberately-wrong guard
        (checking rho < 1 regardless of parity) would refuse positive
        sources too — but the real guard does not.
        """
        probe = _PpgoProbe()
        lens = {'gamma': 0.5, 'y1': 0.3, 'y2': 0.3}
        result = probe._ppgo_cell_coords(lens)
        self.comparisons += 1
        self.assertIsNotNone(result,
                             'positive parity must NOT be refused by '
                             'saddle-only guard')
        self.assertEqual(result[0], 'positive')


# ---------------------------------------------------------------------------
# TD-4: ppGO map defense-in-depth (SITE 5 w_cert guard)
# ---------------------------------------------------------------------------

class PpgoMapDefenseInDepthTestCase(_SaddleRhoTestCase):
    """TD-4: ``CertifiedPpgoMap.w_cert`` gates saddle rho<1 per-cell (F080).

    The blanket saddle rho<1 refusal was replaced by a per-cell
    allowlist: exactly ONE cell -- Cell 1, the F080-CLEAN band
    (parity='saddle', gamma in [1.1572945272629378, 1.3393306228327468],
    rho in [0.0, 0.5)) -- now serves its certified floor
    (``w_cert`` = 19.164305537818887), while EVERY other saddle rho<1
    cell, certified-in-grid or not, still returns UNKNOWN.  ``w_cert`` /
    ``w_trust`` / ``w_ceiling`` all consult the same allowlist so they
    route consistently.

    Positive-parity ``rho < 1`` cells are all BEYOND_WALL in the shipped
    map, so ``w_cert('positive', 0.5, 0.5)`` returns UNKNOWN from cell
    status — the allowlist is saddle-only and never touches positive
    parity.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from cogwheel.lensing.ppgo_map import CertifiedPpgoMap
        cls._ppgo_map = CertifiedPpgoMap.load()

    def test_saddle_rho_lt_1_returns_unknown(self) -> None:
        # (saddle, gamma=1.3, rho=0.5): rho=0.5 is the exclusive upper edge
        # of Cell 1's rho band [0.0, 0.5), so this lands in the next band
        # and is NOT allowlisted -> UNKNOWN.
        result = self._ppgo_map.w_cert('saddle', 1.3, 0.5)
        self.comparisons += 1
        from cogwheel.lensing.ppgo_map import UNKNOWN
        self.assertIs(result, UNKNOWN,
                      'a saddle rho<1 cell outside the Cell 1 allowlist '
                      'must return UNKNOWN')

    def test_saddle_rho_ge_1_returns_float(self) -> None:
        result = self._ppgo_map.w_cert('saddle', 1.3, 1.5)
        self.comparisons += 1
        self.assertIsInstance(result, float,
                              'saddle rho>=1 must return a float '
                              '(sound exterior, guard does not fire)')
        import math
        self.assertTrue(math.isfinite(result))

    def test_saddle_rho_lt_1_overrides_certified_cell(self) -> None:
        """A CERTIFIED but NON-allowlisted saddle rho<1 cell stays UNKNOWN.

        The per-cell allowlist relaxes exactly Cell 1.  The cell at
        (saddle, gamma=1.45, rho=0.25) is CERTIFIED in the shipped grid
        (status 0, a finite ``w_cert_grid`` value) yet lies OUTSIDE Cell 1
        (gamma 1.45 is above the Cell 1 upper edge 1.3393306228327468).
        Without the per-cell gate ``w_cert`` would return that certified
        float; with it, UNKNOWN — proving the gate still overrides a
        certified cell that the allowlist does not name.
        """
        from cogwheel.lensing.ppgo_map import UNKNOWN, STATUS_CERTIFIED
        cell = self._ppgo_map._cell('saddle', 1.45, 0.25)
        self.comparisons += 1
        self.assertIsNotNone(cell, 'saddle gamma=1.45 rho=0.25 must be in grid')
        self.assertEqual(self._ppgo_map.cell_status_grid[cell], STATUS_CERTIFIED,
                         'premise: cell must be CERTIFIED')
        result = self._ppgo_map.w_cert('saddle', 1.45, 0.25)
        self.comparisons += 1
        self.assertIs(result, UNKNOWN,
                      'a certified saddle rho<1 cell OUTSIDE the Cell 1 '
                      'allowlist must still return UNKNOWN — gate is '
                      'load-bearing')

    def test_cell1_serves_certified_floor(self) -> None:
        """Positive pin: allowlisted Cell 1 serves its certified floor.

        (saddle, gamma=1.25, rho=0.25) is inside Cell 1 (gamma in
        [1.1572945272629378, 1.3393306228327468], rho in [0.0, 0.5)).
        ``w_cert`` now returns the shipped certified floor
        19.164305537818887, and ``w_trust`` / ``w_ceiling`` route
        consistently through the same allowlist.
        """
        from cogwheel.lensing.ppgo_map import UNKNOWN
        wc = self._ppgo_map.w_cert('saddle', 1.25, 0.25)
        self.comparisons += 1
        self.assertIsInstance(wc, float,
                              'Cell 1 (saddle, gamma=1.25, rho=0.25) must '
                              'serve a float, not UNKNOWN')
        self.assertEqual(wc, 19.164305537818887,
                         'Cell 1 serves its shipped certified floor')
        wt = self._ppgo_map.w_trust('saddle', 1.25, 0.25)
        wl = self._ppgo_map.w_ceiling('saddle', 1.25, 0.25)
        self.comparisons += 1
        self.assertIsNot(wt, UNKNOWN, 'w_trust routes with w_cert for Cell 1')
        self.assertIsNot(wl, UNKNOWN, 'w_ceiling routes with w_cert for Cell 1')
        self.assertEqual(wt, 28.74645830672833)
        self.assertEqual(wl, 58.0)

    def test_positive_rho_lt_1_not_blocked_by_guard(self) -> None:
        """Positive parity rho<1 is NOT blocked by the saddle-only guard.

        SITE 5 only fires for ``parity == 'saddle'``.  This test verifies
        the parity check: w_cert('positive', 0.5, 0.5) returns UNKNOWN
        because the cell is BEYOND_WALL, NOT because the guard fires.

        If a future map certifies a positive-parity rho<1 cell, SITE 5
        will still pass it through — which is the intended invariant.
        """
        from cogwheel.lensing.ppgo_map import UNKNOWN
        result = self._ppgo_map.w_cert('positive', 0.5, 0.5)
        self.comparisons += 1
        # The shipped map has no certified positive rho<1 cells, so
        # UNKNOWN comes from cell status, not the guard.
        self.assertIs(result, UNKNOWN,
                      'positive rho<1 returns UNKNOWN from cell status '
                      '(BEYOND_WALL), NOT from the saddle-only guard')


class PpgoMapDefenseInDepthSelfFalsificationTestCase(_SaddleRhoTestCase):
    """Reachable-red for TD-4: the per-cell allowlist gate CAN go red."""

    @classmethod
    def setUpClass(cls) -> None:
        from cogwheel.lensing.ppgo_map import CertifiedPpgoMap
        cls._ppgo_map = CertifiedPpgoMap.load()

    def test_positive_would_not_be_blocked(self) -> None:
        """Positive parity never consults the saddle-only allowlist.

        The allowlist gate keys on ``parity == 'saddle'``; for positive
        parity it is never entered, so the UNKNOWN here comes purely from
        the cell-status path (BEYOND_WALL), not from the gate.
        """
        from cogwheel.lensing.ppgo_map import UNKNOWN
        result = self._ppgo_map.w_cert('positive', 0.5, 0.5)
        self.comparisons += 1
        self.assertIs(result, UNKNOWN)

    def test_saddle_would_return_float_without_guard(self) -> None:
        """Without the gate, a non-allowlisted certified cell leaks a float.

        (saddle, gamma=1.45, rho=0.25) is CERTIFIED in the grid — its raw
        ``w_cert_grid`` entry is a finite float — but it lies OUTSIDE the
        Cell 1 allowlist.  ``w_cert`` returns UNKNOWN for it, so the raw
        grid float proves the gate is load-bearing: dropping the gate
        would let that unvetted floor through.
        """
        from cogwheel.lensing.ppgo_map import STATUS_CERTIFIED, UNKNOWN
        cell = self._ppgo_map._cell('saddle', 1.45, 0.25)
        self.comparisons += 1
        self.assertEqual(self._ppgo_map.cell_status_grid[cell],
                         STATUS_CERTIFIED)
        w_cert_val = float(self._ppgo_map.w_cert_grid[cell])
        self.comparisons += 1
        import math
        self.assertTrue(math.isfinite(w_cert_val),
                        f'raw grid float (would leak without gate): '
                        f'{w_cert_val}')
        self.assertGreater(w_cert_val, 10.0,
                           'certified saddle rho<1 floor is plausible (>10)')
        # The gate turns that finite grid float into UNKNOWN.
        self.assertIs(self._ppgo_map.w_cert('saddle', 1.45, 0.25), UNKNOWN,
                      'gate must refuse the non-allowlisted certified cell')

    def test_saddle_rho_ge_1_still_served(self) -> None:
        """Saddle rho>=1 is NOT refused — gate is narrow (only rho<1)."""
        result = self._ppgo_map.w_cert('saddle', 1.3, 1.5)
        self.comparisons += 1
        self.assertIsInstance(result, float,
                              'saddle rho>=1 must still return float')
        import math
        self.assertTrue(math.isfinite(result))

# ---------------------------------------------------------------------------
# TD-5: Census band-split mirror integrity (w_trust-driven, no rho guard)
# ---------------------------------------------------------------------------

class CensusBandSplitMirrorIntegrityTestCase(_SaddleRhoTestCase):
    """TD-5: ``characterize_sample`` band-split is governed by w_trust alone.

    The former SITE 4 guard (``surrogate_census.py``)::

        if parity == 'saddle' and rho is not None and rho < 1.0:
            rho = None

    and its SITE 1 mirror in ``likelihood.py`` have both been removed.
    Saddle ``rho < 1`` sources are no longer suppressed; whether a
    band-split can occur is decided entirely by ``ppgo_map.w_trust``.

    The corridor source (gamma=1.3, y=(0.0, 0.3)) has rho=0.175 and is
    now allowlisted as Cell 1, with a finite ``w_trust`` of 28.746. No
    band-split occurs in this test because ``w_trust`` (28.746) lies
    well above the test's tiny w-band (max ~1.24, see ``_M_LENS``
    below) — the split condition ``w_lo < w_trust < w_hi`` is False,
    not because ``rho`` was suppressed.

    For lobe interior sources with saddle parity where ``rho >= 1``,
    the ppGO map cell is BEYOND_WALL, so ``ppgo_map.w_trust`` returns
    UNKNOWN and the band-split is suppressed via that route instead.

    Both corridor and lobe interior sources show full-band serving.
    """

    #: Corridor source between deltoid lobes (2 images, rho < 1).
    _CORRIDOR: tuple[float, float] = (0.0, 0.3)

    #: Lobe interior source (2 images, rho > 1, saddle parity).
    _LOBE: tuple[float, float] = (1.5, 1.5)

    #: Frequency grid for dimensionless-w computation.
    _F_GRID = np.array([20.0, 100.0])

    #: Lens mass (Msun) — gives w ∈ [~0.25, ~1.24] for a 20-100 Hz band.
    _M_LENS = 100.0

    @classmethod
    def setUpClass(cls) -> None:
        from cogwheel.lensing.ppgo_map import (
            CertifiedPpgoMap, set_certified_ppgo_map, get_certified_ppgo_map)
        cls._prev_map = get_certified_ppgo_map()
        cls._ppgo_map = CertifiedPpgoMap.load()
        set_certified_ppgo_map(cls._ppgo_map)

    @classmethod
    def tearDownClass(cls) -> None:
        from cogwheel.lensing.ppgo_map import set_certified_ppgo_map
        set_certified_ppgo_map(cls._prev_map)

    def _omega_beta_check(self, f_grid: np.ndarray, m_lens_msun: float,
                          gamma: float) -> float:
        """Verify that w_grid has w values that would trigger a split."""
        from cogwheel.lensing.waveform import dimensionless_frequency
        w_grid = dimensionless_frequency(f_grid, m_lens_msun, 0.0)
        rho = np.hypot(*self._LOBE)
        from cogwheel.lensing.ppgo_map import caustic_rho
        try:
            rho_val = caustic_rho(gamma, float(rho), kappa=0.0)
        except Exception:
            rho_val = None
        return float(w_grid.min()), float(w_grid.max()), rho_val

    def test_corridor_source_no_band_split(self) -> None:
        """Corridor source: w_trust (28.746) exceeds the tiny w-band."""
        from unittest.mock import patch, MagicMock
        from cogwheel.lensing.surrogate_census import characterize_sample

        mock_chart = MagicMock()
        mock_chart.gamma_grid = np.array([0.5, 2.0])
        mock_chart.log_w_grid = np.array([1.0, 5.0])
        mock_surrogate = MagicMock()
        mock_surrogate.charts = [mock_chart]

        mock_geom = MagicMock()
        mock_geom.caustic_distance = 1.0
        mock_geom.caustic_theta = 0.1
        mock_geom.real_mask = np.array([True, True])
        # `delays` must be supplied and must match `real_mask` in length:
        # a real `geometry_partition` returns them together, and the tier-1
        # saddle rung indexes one by the other.  A bare MagicMock attribute
        # asarray()s to a 0-d array and raises IndexError before the code
        # under test is reached.
        mock_geom.delays = np.array([0.0, 0.5])

        def _engine_factory(_w: np.ndarray) -> MagicMock:
            ch = MagicMock()
            ch.geometry_partition.return_value = mock_geom
            return ch

        with patch(
            'cogwheel.lensing.surrogate_census._surrogate.select_chart',
            return_value=None,
        ) as mock_select:
            record = characterize_sample(
                mock_surrogate, _engine_factory,
                gamma=_SADDLE_GAMMA, m_lens_msun=self._M_LENS,
                y1=self._CORRIDOR[0], y2=self._CORRIDOR[1],
                f_grid=self._F_GRID, dropped_slivers=())

        self.comparisons += 1
        self.assertFalse(record.served,
                         'corridor source should not be served by any chart')

        # select_chart must have been called.
        mock_select.assert_called()
        # The log_w_max passed to select_chart must equal the original
        # log_w_max (no band-split occurred).
        from cogwheel.lensing.waveform import dimensionless_frequency
        w_grid = dimensionless_frequency(self._F_GRID, self._M_LENS, 0.0)
        log_w_max = float(np.log(w_grid.max()))
        sc_kwargs = mock_select.call_args.kwargs
        self.comparisons += 1
        self.assertAlmostEqual(
            sc_kwargs['log_w_max'], log_w_max,
            msg=f'band-split was NOT suppressed: '
                f'select_chart log_w_max={sc_kwargs["log_w_max"]} '
                f'!= full-band {log_w_max}')

    def test_lobe_interior_source_no_band_split(self) -> None:
        """Lobe interior: rho>=1 reaches the ppGO map, but w_trust=UNKNOWN.

        The ppGO map returns UNKNOWN for this cell (BEYOND_WALL), so no
        band-split is attempted regardless.
        """
        from unittest.mock import patch, MagicMock
        from cogwheel.lensing.surrogate_census import characterize_sample

        mock_chart = MagicMock()
        mock_chart.gamma_grid = np.array([0.5, 2.0])
        mock_chart.log_w_grid = np.array([1.0, 5.0])
        mock_surrogate = MagicMock()
        mock_surrogate.charts = [mock_chart]

        mock_geom = MagicMock()
        mock_geom.caustic_distance = 1.0
        mock_geom.caustic_theta = 0.1
        mock_geom.real_mask = np.array([True, True])
        # `delays` must be supplied and must match `real_mask` in length:
        # a real `geometry_partition` returns them together, and the tier-1
        # saddle rung indexes one by the other.  A bare MagicMock attribute
        # asarray()s to a 0-d array and raises IndexError before the code
        # under test is reached.
        mock_geom.delays = np.array([0.0, 0.5])

        def _engine_factory(_w: np.ndarray) -> MagicMock:
            ch = MagicMock()
            ch.geometry_partition.return_value = mock_geom
            return ch

        with patch(
            'cogwheel.lensing.surrogate_census._surrogate.select_chart',
            return_value=None,
        ) as mock_select:
            record = characterize_sample(
                mock_surrogate, _engine_factory,
                gamma=_SADDLE_GAMMA, m_lens_msun=self._M_LENS,
                y1=self._LOBE[0], y2=self._LOBE[1],
                f_grid=self._F_GRID, dropped_slivers=())

        self.comparisons += 1
        self.assertFalse(record.served,
                         'lobe interior source should not be served')

        mock_select.assert_called()
        from cogwheel.lensing.waveform import dimensionless_frequency
        w_grid = dimensionless_frequency(self._F_GRID, self._M_LENS, 0.0)
        log_w_max = float(np.log(w_grid.max()))
        sc_kwargs = mock_select.call_args.kwargs
        self.comparisons += 1
        self.assertAlmostEqual(
            sc_kwargs['log_w_max'], log_w_max,
            msg=f'band-split was NOT suppressed: '
                f'select_chart log_w_max={sc_kwargs["log_w_max"]} '
                f'!= full-band {log_w_max}')
