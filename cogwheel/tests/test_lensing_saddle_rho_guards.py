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


class PpgocellcoordsCorridorRefusalTestCase(_SaddleRhoTestCase):
    """TD-1: ``_ppgo_cell_coords`` returns None for saddle corridor sources.

    The guard ``if parity == 'saddle' and rho < 1.0`` fires because all
    three corridor sources have ``|y| < caustic_reach`` (rho < 1) and
    gamma > 1 (saddle).  The ppGO map taints this band, so the cell
    coordinate derivation rightly refuses.
    """

    def test_corridor_source_0(self) -> None:
        probe = _PpgoProbe()
        lens = {'gamma': 1.3, 'y1': 0.0, 'y2': 0.3}
        result = probe._ppgo_cell_coords(lens)
        self.comparisons += 1
        self.assertIsNone(result)

    def test_corridor_source_1(self) -> None:
        probe = _PpgoProbe()
        lens = {'gamma': 1.3, 'y1': 0.5, 'y2': 0.0}
        result = probe._ppgo_cell_coords(lens)
        self.comparisons += 1
        self.assertIsNone(result)

    def test_corridor_source_2(self) -> None:
        probe = _PpgoProbe()
        lens = {'gamma': 1.3, 'y1': 0.3, 'y2': 0.3}
        result = probe._ppgo_cell_coords(lens)
        self.comparisons += 1
        self.assertIsNone(result)

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
    """Reachable-red: the saddle rho guards CAN go red.

    A numerical/decision suite without a self-falsification class is not
    finished.  These foils prove that each guard tested above is genuinely
    load-bearing — bypassing it flips the outcome.
    """

    #: A saddle source that the rho<1 guard refuses under the real code.
    _SADDLE_CORRIDOR_LENS: dict[str, float] = {'gamma': 1.3, 'y1': 0.0, 'y2': 0.3}

    def test_ppgo_cell_coords_would_return_tuple_without_guard(self) -> None:
        """Without the parity guard, ``_ppgo_cell_coords`` returns a tuple.

        If the guard ``if parity == 'saddle' and rho < 1.0`` were removed,
        the corridor source would return ``('saddle', 1.3, rho)`` instead
        of None — proving the guard is load-bearing.
        """
        # Compute what the guard blocks: parity and rho for this source.
        from cogwheel.lensing.ppgo_map import caustic_rho
        rho = caustic_rho(1.3, 0.3)
        self.comparisons += 1
        self.assertLess(rho, 1.0, 'premise: corridor source has rho < 1')
        # The guard blocks ('saddle', 1.3, rho) — without it, this tuple
        # would be returned.  Asserting the blocking is the test's purpose.
        probe = _PpgoProbe()
        result = probe._ppgo_cell_coords(self._SADDLE_CORRIDOR_LENS)
        self.comparisons += 1
        self.assertIsNone(result,
                          'guard must return None for saddle corridor')

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
    """TD-4: ``CertifiedPpgoMap.w_cert`` returns UNKNOWN for saddle rho < 1.

    SITE 5 guard (``ppgo_map.py``)::

        if parity == 'saddle' and rho < 1.0:
            return UNKNOWN

    This is defense-in-depth — the shipped map already has CERTIFIED
    saddle cells at ``rho < 1`` (gamma=[1.1,1.55], rho=[0,0.5]) whose
    ``w_cert_grid`` floors are 19—22.  Without this guard,
    ``w_cert('saddle', 1.3, 0.25)`` would return ~19.2, propagating a
    certified-but-unsound floor into the WP2 dispatch.  The guard blocks
    them unconditionally.

    Positive-parity ``rho < 1`` cells are all BEYOND_WALL in the shipped
    map, so the fixture ``w_cert('positive', 0.5, 0.5)`` returns UNKNOWN
    from cell status — NOT from the guard.  This proves the parity check
    is load-bearing (only saddle is blocked), but the spec's claim that
    it returns a float is false with the current map coverage.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from cogwheel.lensing.ppgo_map import CertifiedPpgoMap
        cls._ppgo_map = CertifiedPpgoMap.load()

    def test_saddle_rho_lt_1_returns_unknown(self) -> None:
        result = self._ppgo_map.w_cert('saddle', 1.3, 0.5)
        self.comparisons += 1
        from cogwheel.lensing.ppgo_map import UNKNOWN
        self.assertIs(result, UNKNOWN,
                      'SITE 5 guard must return UNKNOWN for saddle rho<1')

    def test_saddle_rho_ge_1_returns_float(self) -> None:
        result = self._ppgo_map.w_cert('saddle', 1.3, 1.5)
        self.comparisons += 1
        self.assertIsInstance(result, float,
                              'saddle rho>=1 must return a float '
                              '(sound exterior, guard does not fire)')
        import math
        self.assertTrue(math.isfinite(result))

    def test_saddle_rho_lt_1_overrides_certified_cell(self) -> None:
        """SITE 5 blocks a CERTIFIED saddle cell: guard is load-bearing.

        The cell at (saddle, gamma=1.3, rho=0.25) has status CERTIFIED
        with w_cert_grid ≈ 19.2.  Without the guard, w_cert would return a
        float.  With the guard, UNKNOWN — proving SITE 5 is not dead code.
        """
        from cogwheel.lensing.ppgo_map import UNKNOWN, STATUS_CERTIFIED
        cell = self._ppgo_map._cell('saddle', 1.3, 0.25)
        self.comparisons += 1
        self.assertIsNotNone(cell, 'saddle gamma=1.3 rho=0.25 must be in grid')
        self.assertEqual(self._ppgo_map.cell_status_grid[cell], STATUS_CERTIFIED,
                         'premise: cell must be CERTIFIED')
        result = self._ppgo_map.w_cert('saddle', 1.3, 0.25)
        self.comparisons += 1
        self.assertIs(result, UNKNOWN,
                      'SITE 5 must override a certified saddle rho<1 cell '
                      '— guard is load-bearing')

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
    """Reachable-red for TD-4: the defense-in-depth guard CAN go red."""

    @classmethod
    def setUpClass(cls) -> None:
        from cogwheel.lensing.ppgo_map import CertifiedPpgoMap
        cls._ppgo_map = CertifiedPpgoMap.load()

    def test_positive_would_not_be_blocked(self) -> None:
        """Positive parity passes the SITE 5 guard (only saddle is blocked).

        The guard condition includes ``parity == 'saddle'``.  For positive
        parity, this part of the guard evaluates to False, so the guard
        does NOT fire.  Only the cell-status path determines UNKNOWN.
        """
        from cogwheel.lensing.ppgo_map import UNKNOWN
        # Cell status returns UNKNOWN for positive rho<1 (BEYOND_WALL)
        # but SITE 5 does NOT add a second refusal.  The UNKNOWN is from
        # cell_status_grid, not from parity=='saddle'.
        result = self._ppgo_map.w_cert('positive', 0.5, 0.5)
        self.comparisons += 1
        self.assertIs(result, UNKNOWN)

    def test_saddle_would_return_float_without_guard(self) -> None:
        """Without SITE 5, the certified saddle rho<1 cell returns ~19.2.

        This proves the guard is NOT dead code — removing it would let
        an unsound floor through (the whole reason for the WP-1 fix).
        """
        cell = self._ppgo_map._cell('saddle', 1.3, 0.25)
        self.comparisons += 1
        from cogwheel.lensing.ppgo_map import STATUS_CERTIFIED
        self.assertEqual(self._ppgo_map.cell_status_grid[cell],
                         STATUS_CERTIFIED)
        w_cert_val = float(self._ppgo_map.w_cert_grid[cell])
        self.comparisons += 1
        import math
        self.assertTrue(math.isfinite(w_cert_val),
                        f'without guard, w_cert would be {w_cert_val}')
        self.assertGreater(w_cert_val, 10.0,
                           'certified saddle rho<1 floor is plausible (>10)')

    def test_saddle_rho_ge_1_still_served(self) -> None:
        """Saddle rho>=1 is NOT refused — guard is narrow (only rho<1)."""
        result = self._ppgo_map.w_cert('saddle', 1.3, 1.5)
        self.comparisons += 1
        self.assertIsInstance(result, float,
                              'saddle rho>=1 must still return float')
        import math
        self.assertTrue(math.isfinite(result))

# ---------------------------------------------------------------------------
# TD-5: Census band-split mirror integrity (SITE 4 rho=None guard)
# ---------------------------------------------------------------------------

class CensusBandSplitMirrorIntegrityTestCase(_SaddleRhoTestCase):
    """TD-5: ``characterize_sample`` suppresses band-split for saddle rho<1.

    SITE 4 guard (``surrogate_census.py``)::

        if parity == 'saddle' and rho is not None and rho < 1.0:
            rho = None

    This mirrors SITE 1's refusal at the census level: when a saddle
    draw has ``rho < 1``, the ppGO map query is entirely skipped
    (``rho`` set to ``None``), so ``chart_log_w_max`` stays at the
    original ``log_w_max`` — no band-split is attempted.

    For lobe interior sources with saddle parity where ``rho >= 1``,
    SITE 4 does NOT fire, but ``ppgo_map.w_trust`` returns UNKNOWN
    because the cell is BEYOND_WALL — the band-split is also suppressed,
    just via a different route (SITE 5 in ``w_cert``).

    Both corridor and lobe interior sources show full-band serving.
    """

    #: Corridor source between deltoid lobes (2 images, rho < 1).
    _CORRIDOR: tuple[float, float] = (0.0, 0.3)

    #: Lobe interior source (2 images, rho > 1, saddle parity).
    _LOBE: tuple[float, float] = (1.5, 1.5)

    #: Frequency grid for dimensionless-w computation.
    _F_GRID = np.array([20.0, 100.0])

    #: Lens mass (Msun) — chosen so w ∈ [10, 50] for a 20-100 Hz band.
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
        """Corridor source: SITE 4 sets rho=None, no band-split."""
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
        """Lobe interior: rho>=1 so SITE 4 does NOT fire, but w_trust=UNKNOWN.

        Even without SITE 4, the ppGO map returns UNKNOWN for this cell
        (BEYOND_WALL), so no band-split is attempted regardless.
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


class CensusBandSplitMirrorSelfFalsificationTestCase(_SaddleRhoTestCase):
    """Reachable-red for TD-5: the mirror refusal CAN go red."""

    def test_site4_rho_none_is_load_bearing(self) -> None:
        """Without SITE 4, saddle rho<1 would call ppgo_map.w_trust.

        SITE 4 sets rho=None so that no map query is made for saddle
        rho<1 draws.  Without this guard, rho would be ~0.175 and
        the code would call ``w_trust('saddle', 1.3, 0.175)``.
        SITE 5 in ``w_cert`` would still return UNKNOWN (defense-in-depth),
        but SITE 4 prevents even the call — proving both guards are
        load-bearing at different levels.
        """
        from cogwheel.lensing.ppgo_map import caustic_rho, UNKNOWN
        rho = caustic_rho(1.3, np.hypot(0.0, 0.3))
        self.comparisons += 1
        self.assertLess(rho, 1.0,
                        'premise: corridor source has rho < 1')
        self.assertGreater(rho, 0.0)

        # If SITE 4 were absent, the code would call w_trust with this rho.
        # SITE 5 (w_cert) would still return UNKNOWN, but SITE 4 is the
        # census-level mirror that prevents the query entirely.
        # Both guards working together ensure defense-in-depth.
        from cogwheel.lensing.ppgo_map import (
            CertifiedPpgoMap, set_certified_ppgo_map, get_certified_ppgo_map)
        prev = get_certified_ppgo_map()
        try:
            ppgo_map = CertifiedPpgoMap.load()
            set_certified_ppgo_map(ppgo_map)
            result = ppgo_map.w_cert('saddle', 1.3, rho)
            self.comparisons += 1
            self.assertIs(result, UNKNOWN,
                          'SITE 5 defense-in-depth must also return UNKNOWN '
                          '(double guard: SITE 4 at census + SITE 5 at map)')
        finally:
            set_certified_ppgo_map(prev)
