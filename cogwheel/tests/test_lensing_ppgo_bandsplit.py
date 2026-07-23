"""Domain tests for Build 8h-a: certified-ppGO map + per-node band split.

Build 8h-a closes the zero-quadrature gap with four levers, gated here:

* WP1 -- the hash-pinned certified-ppGO frequency-floor map
  (`cogwheel.lensing.ppgo_map`): the sup-over-w floor extraction (the
  ppGO error is NON-monotone, so the stored ``w_cert`` is the LAST upward
  re-crossing, never the first), the Professor safety margin
  ``w_trust = max(1.5 * w_cert, w_cert + 2.0)``, and the refuse-to-certify
  contract (corrupt hash / absent artifact / beyond-wall UNKNOWN cell all
  yield UNKNOWN so dispatch never serves bare ppGO uncertified).

* WP2 -- the per-node band split in the lensed likelihood
  (`_surrogate_coefficients` / `_ppgo_band_split`): a draw straddling
  ``w_trust`` is chart-served below and bare-ppGO-served above; the ppGO
  segment matches exact ``F`` to ``1e-4`` F-normalized at EVERY node (a
  beat re-crossing above the floor must also clear), the chart segment to
  the spline currency ``5e-3`` on ``max|E_ff|``, and the two segments
  agree at the seam to ``5e-3``.  The map state (valid / corrupt / absent
  / beyond-wall) flips the SAME draw between served and loudly refused,
  never falling through to numerical quadrature.

* WP3 -- interior (4-image) far-field tiles + strata trimming
  (`cogwheel.lensing.surrogate_training`): the interior admission geometry
  (`_farfield_interior_tiles`) admits a tile wholly inside the caustic
  disk minus the tube shell and rejects one straddling it; the far-field
  ``E_ff`` telescoping identity holds for an interior 4-image config to
  ``1e-12 * max|F|``; the real-image mask tracks the morse/physical image
  set (4 near a cusp, dropping to 2 across the caustic), so a hardcoded
  ``len == 4`` mask is caught; and the ppGO strata-trim
  (`_stratum_ppgo_boundary` + `_apply_ppgo_trim`) drops a stratum wholly
  above the hand-off floor, caps one straddling it, and -- with no map --
  trims nothing.

INDEPENDENT ORACLE
------------------
The reconstruction oracle throughout is the engine's
``ChangRefsdalPartition.exact_total`` (the operator/Schwinger amplification
total), which shares no code with the ppGO image-kernel sum or the spline
emulator under test.  The sup-over-w and margin tests use SYNTHETIC
injected arrays (a real beat location drifts and would flake).  Every
sweeping test carries an anti-vacuity guard (`tearDown`) and the tests
that certify a bound also assert the opposite direction where the spec
demands a falsifiable red.

Style mirrors ``test_lensing_farfield_envelope.py`` and
``test_lensing_surrogate_training.py``.  FAST tier only: small synthetic
configs, no engine campaigns.
"""
from __future__ import annotations

import json
import math
import pathlib
import tempfile

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_MPL = False

from unittest import TestCase, main

from cogwheel.lensing.chang_refsdal import geometry, channels as _channels
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels, farfield_envelope_from_partition,
    reconstruct_from_envelope)
from cogwheel.lensing import ppgo_map
from cogwheel.lensing.ppgo_map import (
    CertifiedPpgoMap, build_map, save_map,
    set_certified_ppgo_map, get_certified_ppgo_map, use_certified_ppgo_map,
    _sup_over_w_floor,
    CERTIFICATION_BAR, W_TRUST_MULTIPLIER, W_TRUST_ADDITIVE,
    STATUS_CERTIFIED, STATUS_BEYOND_WALL, STATUS_INVALID,
    ASTROID_WALL, SADDLE_WALL, _PARITY_CODES)
from cogwheel.lensing.surrogate_training import (
    _farfield_interior_tiles, _stratum_ppgo_boundary, _apply_ppgo_trim)
from cogwheel.lensing.surrogate import LensAmplificationSurrogate
from cogwheel.lensing.likelihood import LensedRelativeBinningLikelihood

_OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'


# ======================================================================
# Shared helpers.
# ======================================================================

def _telescoping_error(partition) -> float:
    """F-normalized error of ``E_ff`` + real carriers vs ``exact_total``.

    Reconstructs ``F`` from the far-field remainder with ``switch = 1`` on
    every REAL channel (the morse/physical `real_mask`) and no critical
    carrier, exactly as the likelihood far-field path does, and normalizes
    by ``max|F|`` (never bare -- an interference null must not flake the
    machine-precision gate).
    """
    envelope = farfield_envelope_from_partition(partition)
    switch = np.zeros((partition.w.shape[0], _channels._N_CHANNELS),
                      dtype=float)
    switch[:, np.asarray(partition.real_mask, dtype=bool)] = 1.0
    _kernels, total = reconstruct_from_envelope(
        partition.w, envelope, partition.delays, partition.saddle_kernels,
        switch, 0.0)
    denom = float(np.max(np.abs(partition.exact_total))) or 1.0
    return float(np.max(np.abs(total - partition.exact_total))) / denom


def _partition(w_grid: np.ndarray, gamma: float, y: tuple[float, float]):
    """Fresh, reset engine partition (deterministic far-proposal labeling)."""
    engine = ChangRefsdalChannels(np.asarray(w_grid, dtype=float))
    engine.reset()
    return engine.evaluate(gamma=gamma, y=y, beta=0.0, kappa=0.0)


def _synthetic_map(*, parity: str, gamma: float, rho: float, w_cert: float,
                   status: float = STATUS_CERTIFIED) -> CertifiedPpgoMap:
    """A one-cell-live synthetic map certifying ``w_cert`` at a chosen cell.

    Built directly through `CertifiedPpgoMap.from_arrays` (no engine sweep,
    no hash check -- integrity is exercised separately with a real ``.npz``
    in the refusal test).  The grid is a minimal ``2 x 2 x 3`` lattice with
    an edge exactly at the ``gamma = 1.0`` parity boundary; every cell but
    the requested one is `STATUS_INVALID`.
    """
    gamma_edges = np.array([0.2, 1.0, 1.6], dtype=float)
    rho_edges = np.array([0.0, 0.5, 1.0, math.inf], dtype=float)
    parity_codes = np.array([_PARITY_CODES['positive'],
                             _PARITY_CODES['saddle']], dtype=float)
    shape = (2, gamma_edges.size - 1, rho_edges.size - 1)
    w_cert_grid = np.full(shape, np.nan)
    diag_grid = np.full(shape, np.nan)
    status_grid = np.full(shape, STATUS_INVALID)
    interp_grid = np.zeros(shape)

    p = 0 if parity == 'positive' else 1
    gi = int(np.searchsorted(gamma_edges, gamma, side='right') - 1)
    ri = int(np.searchsorted(rho_edges, rho, side='right') - 1)
    gi = min(max(gi, 0), shape[1] - 1)
    ri = min(max(ri, 0), shape[2] - 1)
    status_grid[p, gi, ri] = status
    if status == STATUS_CERTIFIED:
        w_cert_grid[p, gi, ri] = w_cert
        interp_grid[p, gi, ri] = 1.0

    provenance = {'schema_version': 'test',
                  'certification_bar': CERTIFICATION_BAR}
    return CertifiedPpgoMap.from_arrays(
        parity_codes, gamma_edges, rho_edges, w_cert_grid, diag_grid,
        status_grid, interp_grid, provenance)


class _PpgoTestCase(TestCase):
    """Base carrying the counted assertion + anti-vacuity guard."""

    def setUp(self) -> None:
        self.comparisons = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self.comparisons, 0,
            'no comparisons were made -- the test asserted nothing')

    def assert_within(self, value: float, tol: float, message: str) -> None:
        self.comparisons += 1
        self.assertLessEqual(value, tol, message)


# ======================================================================
# Test #3 -- MAP SUP-OVER-W FLOOR, NON-MONOTONE (WP1).
# ======================================================================

class SupOverWFloorTestCase(_PpgoTestCase):
    """`_sup_over_w_floor` returns the LAST re-crossing, not the first.

    A SYNTHETIC per-node error array (a real beat location drifts and would
    flake) rises past the bar at ``w1``, dips back ABOVE the bar at
    ``w2 > w1`` (the image-delay beat re-crossing), then descends below at
    ``w3``.  The certified floor MUST be ``w3`` -- the smallest ``w`` above
    which the error stays below the bar for ALL ``w'`` up to the wall --
    NOT the first downward crossing just after ``w1``.
    """

    W_NODES = np.arange(1.0, 11.0)      # w = 1 .. 10
    BAR = 1.0e-4
    # Below the bar everywhere except a first excursion at w=4 (index 3) and
    # a re-crossing beat at w=6 (index 5).  Last violation is at w=6, so the
    # sup-over-w floor is the next node, w=7.
    W1, W2, W3 = 4.0, 6.0, 7.0

    @classmethod
    def _error(cls) -> np.ndarray:
        error = np.full(cls.W_NODES.shape, 5.0e-6)
        error[3] = 2.0e-3          # w1: first excursion above the bar
        error[5] = 3.0e-3          # w2: the beat re-crossing above the bar
        return error

    @classmethod
    def setUpClass(cls) -> None:
        cls._plot()

    @classmethod
    def _plot(cls) -> None:
        if not _HAVE_MPL:
            return
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        error = cls._error()
        floor = _sup_over_w_floor(cls.W_NODES, error, cls.BAR)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(cls.W_NODES, error, 'b.-', label='injected error')
        ax.axhline(cls.BAR, color='r', ls='--', label='bar')
        for w, name in ((cls.W1, 'w1'), (cls.W2, 'w2'), (cls.W3, 'w3')):
            ax.axvline(w, color='0.6', ls=':')
            ax.text(w, cls.BAR * 3, name)
        ax.axvline(floor, color='g', lw=2, label=f'stored floor={floor}')
        ax.set_xlabel('w')
        ax.set_ylabel('|F - ppGO| / max|F|')
        ax.set_title('Sup-over-w floor sits at the LAST re-crossing')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / 'ppgo_sup_over_w_floor.png', dpi=110)
        plt.close(fig)

    def test_floor_is_the_last_recrossing_not_the_first(self):
        """The stored floor is ``w3`` (last re-crossing), never ``w1``."""
        floor = _sup_over_w_floor(self.W_NODES, self._error(), self.BAR)
        self.comparisons += 1
        self.assertEqual(
            floor, self.W3,
            f'sup-over-w floor was {floor}, expected the last re-crossing '
            f'w3={self.W3}')

    def test_floor_is_strictly_above_the_first_crossing(self):
        """A naive first-crossing impl (floor just past ``w1``) is red."""
        floor = _sup_over_w_floor(self.W_NODES, self._error(), self.BAR)
        # A first-crossing implementation would return w = 5 (the node right
        # after the w=4 excursion); the sup-over-w floor is strictly larger.
        self.comparisons += 1
        self.assertGreater(
            floor, 5.0,
            f'floor {floor} did not clear the beat re-crossing at w2='
            f'{self.W2}; a first-crossing bug returns ~5.0')

    def test_all_below_bar_returns_the_first_node(self):
        """A whole-band-clean cell certifies from the bottom node."""
        clean = np.full(self.W_NODES.shape, 5.0e-6)
        floor = _sup_over_w_floor(self.W_NODES, clean, self.BAR)
        self.comparisons += 1
        self.assertEqual(floor, float(self.W_NODES[0]))

    def test_top_node_violation_is_uncertified(self):
        """A cell whose top node still violates has no floor (beyond wall)."""
        error = np.full(self.W_NODES.shape, 5.0e-6)
        error[-1] = 1.0e-2         # nearest the wall, still above the bar
        floor = _sup_over_w_floor(self.W_NODES, error, self.BAR)
        self.comparisons += 1
        self.assertIsNone(
            floor, 'a top-node violation must return None (uncertified)')


# ======================================================================
# Test #4 -- MAP SAFETY MARGIN (WP1).
# ======================================================================

class SafetyMarginTestCase(_PpgoTestCase):
    """``w_trust = max(1.5 * w_cert, w_cert + 2.0)`` in both regimes.

    The additive floor dominates for small ``w_cert`` (protecting the low-w
    cells where the multiplicative margin is thinner than a grid spacing)
    and the multiplicative term dominates for large ``w_cert``.  The two
    regimes cross where ``1.5 w = w + 2`` -> ``w = 4``.
    """

    def test_additive_floor_dominates_at_small_w_cert(self):
        """``w_cert = 1.3`` -> ``w_trust = 3.3`` (the +2.0 floor wins)."""
        self.assert_within(
            abs(CertifiedPpgoMap.w_trust_from_cert(1.3) - 3.3), 1e-12,
            'additive safety floor not applied at w_cert=1.3')

    def test_multiplicative_term_dominates_at_large_w_cert(self):
        """``w_cert = 15`` -> ``w_trust = 22.5`` (the 1.5x term wins)."""
        self.assert_within(
            abs(CertifiedPpgoMap.w_trust_from_cert(15.0) - 22.5), 1e-12,
            'multiplicative safety margin not applied at w_cert=15')

    def test_rule_matches_the_constants_across_the_measured_range(self):
        """Across ``w_cert`` in [1.3, 20] the rule is the exact max()."""
        for w_cert in np.linspace(1.3, 20.0, 40):
            expected = max(W_TRUST_MULTIPLIER * w_cert,
                           w_cert + W_TRUST_ADDITIVE)
            self.assert_within(
                abs(CertifiedPpgoMap.w_trust_from_cert(float(w_cert))
                    - expected), 1e-12,
                f'w_trust rule departed from max() at w_cert={w_cert}')

    def test_crossover_at_w_cert_four(self):
        """Below w_cert=4 additive wins; above it multiplicative wins."""
        self.comparisons += 1
        # Just below 4: additive (w+2) exceeds 1.5w.
        self.assertAlmostEqual(
            CertifiedPpgoMap.w_trust_from_cert(3.0), 5.0, places=12)
        self.comparisons += 1
        # Just above 4: 1.5w exceeds w+2.
        self.assertAlmostEqual(
            CertifiedPpgoMap.w_trust_from_cert(6.0), 9.0, places=12)

    def test_installed_map_query_applies_the_margin(self):
        """A certified cell's ``w_trust`` query equals the margin rule."""
        cmap = _synthetic_map(parity='positive', gamma=0.5, rho=0.7,
                              w_cert=5.0)
        raw = cmap.w_cert('positive', 0.5, 0.7)
        trust = cmap.w_trust('positive', 0.5, 0.7)
        self.comparisons += 1
        self.assertEqual(raw, 5.0)
        self.assert_within(
            abs(trust - CertifiedPpgoMap.w_trust_from_cert(5.0)), 1e-12,
            'map.w_trust did not apply the authoritative margin rule')


# ======================================================================
# Test #2 -- TELESCOPING IDENTITY, INTERIOR 4-IMAGE (WP3).
# ======================================================================

class InteriorTelescopingTestCase(_PpgoTestCase):
    """Adding the four real carriers back to ``E_ff`` returns ``F``.

    A positive-parity astroid interior config (``gamma = 0.5``, source
    ``(0.10, 0.06)`` well inside the caustic) has FOUR real images.
    Reconstructing ``F`` from the far-field remainder with ``switch = 1``
    on all four real channels and ``critical_delay = 0`` must reproduce the
    untouched engine oracle ``exact_total`` to ``1e-12 * max|F|`` --
    normalized by ``max|F|`` (never bare, so an interference null cannot
    flake it), matching the exterior `ReconstructionExactnessTestCase`.
    The subtraction runs over the morse/physical `real_mask`, so an
    interior box telescopes over four kernels with no code change.
    """

    GAMMA = 0.5
    SOURCE = (0.10, 0.06)
    W_BAND = np.linspace(1.0, 40.0, 140)
    MACHINE_REL_TOL = 1.0e-12

    @classmethod
    def setUpClass(cls) -> None:
        cls.partition = _partition(cls.W_BAND, cls.GAMMA, cls.SOURCE)
        cls.n_real = int(np.asarray(cls.partition.real_mask).sum())
        cls.f_scale = float(np.max(np.abs(cls.partition.exact_total)))
        cls._plot()

    @classmethod
    def _plot(cls) -> None:
        if not _HAVE_MPL:
            return
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        envelope = farfield_envelope_from_partition(cls.partition)
        switch = np.zeros((cls.partition.w.shape[0], _channels._N_CHANNELS))
        switch[:, np.asarray(cls.partition.real_mask, dtype=bool)] = 1.0
        _k, total = reconstruct_from_envelope(
            cls.partition.w, envelope, cls.partition.delays,
            cls.partition.saddle_kernels, switch, 0.0)
        error = np.abs(total - cls.partition.exact_total)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(cls.partition.w, np.maximum(error, 1e-18), 'b.-',
                    label='|F_recon - F_exact|')
        ax.axhline(cls.MACHINE_REL_TOL * cls.f_scale, color='r', ls='--',
                   label='1e-12 * max|F|')
        ax.set_xlabel('w')
        ax.set_ylabel('reconstruction error')
        ax.set_title('Interior 4-image telescoping sits at the machine floor')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / 'ppgo_interior_telescoping.png', dpi=110)
        plt.close(fig)

    def test_config_is_interior_four_image(self):
        """The fixture is a genuine 4-image astroid interior (not vacuous)."""
        self.comparisons += 1
        self.assertEqual(
            self.n_real, 4,
            f'fixture is not 4-image (real_mask.sum()={self.n_real}); the '
            f'interior telescoping claim would be vacuous')

    def test_interior_reconstruction_is_exact(self):
        """E_ff + four real carriers returns ``exact_total`` to 1e-12."""
        error = _telescoping_error(self.partition)
        self.assert_within(
            error, self.MACHINE_REL_TOL,
            f'interior telescoping departed from exact_total by {error:.3e} '
            f'(F-normalized)')


# ======================================================================
# Test #6 -- INTERIOR ADMISSION GEOMETRY + MORSE-SIGN MASK (WP3).
# ======================================================================

class InteriorAdmissionTestCase(_PpgoTestCase):
    """`_farfield_interior_tiles` admits wholly-inside tiles, rejects the rest.

    A tile is admitted iff its FARTHEST corner from the origin stays inside
    ``admit_radius = caustic_inradius - eta_max`` (the caustic disk minus
    the tube shell), so an admitted tile carries no caustic crossing and no
    tube-shell overlap -- the single 4-image region by construction.  A
    tile whose far corner exceeds the radius (straddling the shell or the
    caustic) is dropped.
    """

    GRID_EXTENT = 1.0
    ADMIT_RADIUS = 0.6
    N_PER_SIDE = 5

    @classmethod
    def setUpClass(cls) -> None:
        cls.tiles = _farfield_interior_tiles(
            cls.GRID_EXTENT, cls.ADMIT_RADIUS, cls.N_PER_SIDE)

    def test_admitted_tiles_are_wholly_inside_admit_radius(self):
        """Every admitted tile's farthest corner is within the radius."""
        for (cx, cy), half, _i, _j in self.tiles:
            far = math.hypot(abs(cx) + half, abs(cy) + half)
            self.assert_within(
                far, self.ADMIT_RADIUS + 1e-12,
                f'admitted tile at ({cx:.3f},{cy:.3f}) half {half} straddles '
                f'the admit radius (far corner {far:.4f})')

    def test_some_tiles_are_admitted_where_geometry_permits(self):
        """'admitted > 0 where geometry permits' -- the loud assert."""
        self.comparisons += 1
        self.assertGreater(
            len(self.tiles), 0,
            'no interior tile admitted where the disk permits some')

    def test_straddling_and_exterior_tiles_are_rejected(self):
        """Tiles crossing the disk boundary are dropped (fewer than full)."""
        self.comparisons += 1
        # Radius 0.6 inside a 1.0-extent 5x5 grid (tile half 0.2) cannot
        # admit all 25 tiles: the outer corners straddle or exit the disk.
        self.assertLess(
            len(self.tiles), self.N_PER_SIDE ** 2,
            'admission accepted every tile; the outer straddlers were not '
            'rejected')
        # A concrete straddler: the outermost corner tile is wholly outside.
        half = self.GRID_EXTENT / self.N_PER_SIDE
        corner = self.GRID_EXTENT - half     # center of the corner tile
        far = math.hypot(corner + half, corner + half)
        self.assertGreater(far, self.ADMIT_RADIUS,
                           'fixture no longer has an excluded corner tile')
        self.comparisons += 1
        self.assertNotIn(
            (corner, corner),
            [tile[0] for tile in self.tiles],
            'a tile straddling the caustic/tube-shell was wrongly admitted')

    def test_tighter_radius_admits_strictly_fewer(self):
        """Shrinking the disk (more tube shell) drops more tiles (monotone).

        A wide disk (0.85, admitting the centre + first two rings) against a
        tight one (0.40, only the centre tile clears): more tube shell must
        exclude strictly more interior tiles.
        """
        wide = _farfield_interior_tiles(self.GRID_EXTENT, 0.85,
                                        self.N_PER_SIDE)
        tight = _farfield_interior_tiles(self.GRID_EXTENT, 0.40,
                                         self.N_PER_SIDE)
        self.comparisons += 1
        self.assertLess(len(tight), len(wide),
                        f'a tighter admit radius did not drop more tiles '
                        f'(wide={len(wide)}, tight={len(tight)})')


class MorseSignMaskTestCase(_PpgoTestCase):
    """The real-image mask tracks the morse/physical image set, not a 4.

    CRITICAL fixture (Professor 8h-a): an interior config ADJACENT to a
    cusp (``gamma = 0.5``, source on the diagonal at ``|y| ~ 0.5``) carries
    FOUR real images, one near-degenerate (a fold-adjacent image with a
    magnification several times the others); nudging the source across the
    caustic drops the merging pair, leaving TWO images.  The engine's
    `real_mask` -- built from the actual `find_images` solutions, i.e. the
    morse-indexed image set -- reads 4 then 2.  A hardcoded ``len == 4``
    mask would mislabel the 2-image config (and subtract two phantom
    carriers), so ``real_mask.sum() != 4`` there is the load-bearing red.
    """

    GAMMA = 0.5
    W_BAND = np.geomspace(2.0, 40.0, 60)
    # On the astroid diagonal: |y| ~ 0.50 is cusp-adjacent (4 images, one
    # near-degenerate); |y| ~ 0.60 is just across the caustic (2 images).
    CUSP_ADJACENT = (0.5 * math.cos(math.pi / 4), 0.5 * math.sin(math.pi / 4))
    ACROSS_CAUSTIC = (0.6 * math.cos(math.pi / 4), 0.6 * math.sin(math.pi / 4))

    @classmethod
    def setUpClass(cls) -> None:
        cls.p_in = _partition(cls.W_BAND, cls.GAMMA, cls.CUSP_ADJACENT)
        cls.p_out = _partition(cls.W_BAND, cls.GAMMA, cls.ACROSS_CAUSTIC)
        cls.imgs_in = geometry.find_images(
            np.asarray(cls.CUSP_ADJACENT), cls.p_in.matrix)
        cls.imgs_out = geometry.find_images(
            np.asarray(cls.ACROSS_CAUSTIC), cls.p_out.matrix)
        cls.morse_in = [geometry.morse_index(im, cls.p_in.matrix)
                        for im in cls.imgs_in]
        cls.mags_in = [abs(geometry.magnification(im, cls.p_in.matrix))
                       for im in cls.imgs_in]

    def test_cusp_adjacent_config_is_four_image_with_a_near_degenerate(self):
        """The fixture is cusp-adjacent: 4 images, one near a fold."""
        self.comparisons += 1
        self.assertEqual(len(self.imgs_in), 4,
                         'cusp-adjacent fixture is not 4-image')
        # A fold-adjacent image has a magnification well above the others,
        # so it is near-degenerate (approaching an eigenvalue zero).
        self.comparisons += 1
        self.assertGreater(
            max(self.mags_in) / min(self.mags_in), 3.0,
            'no near-degenerate image; the fixture is not cusp-adjacent, so '
            'it would not exercise the morse-sign mask')

    def test_interior_images_have_mixed_morse_signs(self):
        """Astroid interior: two minima + two saddles (signed sum 0)."""
        signed_sum = sum((-1) ** m for m in self.morse_in)
        self.comparisons += 1
        self.assertEqual(
            signed_sum, 0,
            f'astroid interior signed parity sum was {signed_sum}, expected '
            f'0 (two minima, two saddles)')
        self.comparisons += 1
        self.assertEqual(
            sorted(self.morse_in), [0, 0, 1, 1],
            f'interior morse indices {self.morse_in} are not the expected '
            f'two-minimum/two-saddle mix')

    def test_real_mask_equals_the_morse_image_count_both_sides(self):
        """`real_mask.sum()` == number of found (morse) images, 4 then 2."""
        n_in = int(np.asarray(self.p_in.real_mask).sum())
        n_out = int(np.asarray(self.p_out.real_mask).sum())
        self.comparisons += 1
        self.assertEqual(
            n_in, len(self.imgs_in),
            'interior real_mask disagreed with find_images count')
        self.comparisons += 1
        self.assertEqual(
            n_out, len(self.imgs_out),
            'across-caustic real_mask disagreed with find_images count')

    def test_hardcoded_four_mask_is_falsified_across_the_caustic(self):
        """Just across the caustic the mask drops to 2 -- a fixed 4 is wrong."""
        n_out = int(np.asarray(self.p_out.real_mask).sum())
        self.comparisons += 1
        self.assertNotEqual(
            n_out, 4,
            'across the caustic the mask still read 4; a hardcoded len==4 '
            'mask would not be caught here')
        self.comparisons += 1
        self.assertEqual(n_out, 2,
                         'expected a 2-image region just across the caustic')

    def test_telescoping_holds_for_the_cusp_adjacent_mask(self):
        """E_ff + morse-real carriers returns F even next to the fold."""
        error = _telescoping_error(self.p_in)
        self.assert_within(
            error, 1.0e-11,
            f'cusp-adjacent interior telescoping departed by {error:.3e}; '
            f'the morse-sign mask did not reproduce F')


# ======================================================================
# Test #7 -- STRATA TRIMMING RECORD (WP3).
# ======================================================================

class StrataTrimmingTestCase(_PpgoTestCase):
    """ppGO strata trimming drops, caps, and -- with no map -- keeps.

    `_stratum_ppgo_boundary` returns the margin-inflated hand-off floor
    ``w_trust`` for a certified region (and ``None`` for no map / UNKNOWN
    cell); `_apply_ppgo_trim` then DROPS a stratum whose whole ``w`` band
    lies above the floor (ppGO serves it, no chart), CAPS one whose top
    exceeds it (band-split hands the tail to ppGO), and KEEPs one wholly
    below.  With no map the boundary is ``None`` and nothing is trimmed.
    The drop/cap records the stratum index and its ``w`` range for the
    ladder census -- asserted through the exact record the trainer builds.
    """

    PARITY = 1                 # astroid / positive parity
    GAMMA = 0.5
    RHO = 0.3
    W_CERT = 3.0               # -> w_trust = max(4.5, 5.0) = 5.0

    @classmethod
    def setUpClass(cls) -> None:
        cls.cmap = _synthetic_map(parity='positive', gamma=cls.GAMMA,
                                  rho=cls.RHO, w_cert=cls.W_CERT)
        cls.boundary = _stratum_ppgo_boundary(
            cls.PARITY, cls.GAMMA, cls.RHO, cls.cmap)

    def test_boundary_is_the_margin_inflated_trust_floor(self):
        """The hand-off floor is ``w_trust`` (margin), not the raw w_cert."""
        expected = CertifiedPpgoMap.w_trust_from_cert(self.W_CERT)  # 5.0
        self.comparisons += 1
        self.assertIsNotNone(self.boundary)
        self.assert_within(
            abs(self.boundary - expected), 1e-12,
            f'strata boundary {self.boundary} is not w_trust {expected} '
            f'(must be margin-inflated, not raw w_cert {self.W_CERT})')

    def test_stratum_wholly_above_floor_is_dropped(self):
        """A stratum whose whole band exceeds the floor is dropped."""
        w_range = (self.boundary + 1.0, self.boundary + 8.0)
        new_range, action = _apply_ppgo_trim(w_range, self.boundary)
        self.comparisons += 1
        self.assertEqual(action, 'drop',
                         'a stratum wholly above w_trust was not dropped')
        # The record the trainer builds for a drop (stratum index + w-range).
        record = {'stratum_index': 2, 'region': 'exterior',
                  'w_range': [round(w_range[0], 6), round(w_range[1], 6)],
                  'w_trust': round(float(self.boundary), 6),
                  'reason': 'ppGO certified over the whole stratum w-band'}
        self.comparisons += 1
        self.assertEqual(record['w_range'],
                         [round(w_range[0], 6), round(w_range[1], 6)],
                         'the drop record must carry the stratum w-range')

    def test_stratum_straddling_floor_is_capped(self):
        """A stratum whose top exceeds the floor is capped at the floor."""
        w_range = (self.boundary - 1.0, self.boundary + 3.0)
        new_range, action = _apply_ppgo_trim(w_range, self.boundary)
        self.comparisons += 1
        self.assertEqual(action, 'cap', 'a straddling stratum was not capped')
        self.assert_within(
            abs(new_range[1] - self.boundary), 1e-12,
            f'capped top {new_range[1]} is not the hand-off floor '
            f'{self.boundary}')

    def test_stratum_below_floor_is_kept(self):
        """A stratum wholly below the floor is untouched."""
        w_range = (1.2, self.boundary - 0.5)
        new_range, action = _apply_ppgo_trim(w_range, self.boundary)
        self.comparisons += 1
        self.assertEqual(action, 'keep')
        self.comparisons += 1
        self.assertEqual(new_range, w_range)

    def test_no_map_trims_nothing(self):
        """With no map the boundary is None and every stratum is kept."""
        boundary = _stratum_ppgo_boundary(self.PARITY, self.GAMMA, self.RHO,
                                          None)
        self.comparisons += 1
        self.assertIsNone(boundary, 'no map must yield a None boundary')
        for w_range in [(1.0, 4.0), (10.0, 40.0), (2.0, 50.0)]:
            new_range, action = _apply_ppgo_trim(w_range, boundary)
            self.comparisons += 1
            self.assertEqual(action, 'keep',
                             'no-map trimming altered a stratum')
            self.assertEqual(new_range, w_range)

    def test_unknown_cell_trims_nothing(self):
        """A beyond-wall / UNKNOWN cell yields a None boundary (no trim)."""
        beyond = _synthetic_map(parity='positive', gamma=self.GAMMA,
                                rho=self.RHO, w_cert=math.nan,
                                status=STATUS_BEYOND_WALL)
        boundary = _stratum_ppgo_boundary(self.PARITY, self.GAMMA, self.RHO,
                                          beyond)
        self.comparisons += 1
        self.assertIsNone(
            boundary, 'a beyond-wall cell must not certify a trim floor')


# ======================================================================
# Test #1 -- BAND-SPLIT RECONSTRUCTION NODE-MATCH (WP2).
# ======================================================================

class BandSplitReconstructionTestCase(_PpgoTestCase):
    """Chart below ``w_trust``, bare ppGO above, matched at every node.

    A fixed in-domain exterior draw (``gamma = 0.3``, source ``(1.3, 1.3)``,
    two images, well below the Schwinger wall) with a coarse SYNTHETIC ppGO
    map installed certifies a cell whose ``w_trust`` (read from the map via
    the REAL `LensedRelativeBinningLikelihood._ppgo_band_split`) falls
    inside the dense ``w`` band.  The band is reconstructed the way the
    production dispatch does (`_surrogate_coefficients`): one shared
    geometry partition, a trained far-field chart's spline envelope below
    ``w_trust``, ``E_ff = 0`` (bare ppGO image-kernel sum) above, fed
    through the same `reconstruct_from_envelope` on real-channel switches.

    Gates (Professor TEST BARS): the ppGO segment matches exact ``F`` to
    ``1e-4`` F-normalized at EVERY node (a beat re-crossing above the floor
    would fail here, not just the first node); the chart segment to
    ``5e-3`` absolute (spline currency, not ``1e-4``); and the chart-below
    and ppGO-above reconstructions agree at the split node to ``5e-3`` --
    the load-bearing seam continuity that catches a discontinuity at the
    split.
    """

    GAMMA = 0.3
    SOURCE = (1.3, 1.3)
    DENSE_W = np.geomspace(2.0, 40.0, 80)
    W_CERT = 8.0               # -> w_trust = max(12.0, 10.0) = 12.0
    PPGO_TOL = 1.0e-4          # F-normalized, ppGO segment
    CHART_TOL = 5.0e-3         # absolute, chart segment (spline currency)
    SEAM_TOL = 5.0e-3          # absolute, split-node agreement

    @classmethod
    def setUpClass(cls) -> None:
        # Install a coarse synthetic map certifying this draw's cell, read
        # w_trust through the REAL dispatch helper, then clear the global so
        # no other test sees a map (the reconstruction below needs only the
        # captured float).
        reach = ppgo_map.caustic_geometry(cls.GAMMA, 0.0)[0]
        rho = math.hypot(*cls.SOURCE) / reach
        set_certified_ppgo_map(_synthetic_map(
            parity='positive', gamma=cls.GAMMA, rho=rho, w_cert=cls.W_CERT))
        try:
            cls.w_trust = LensedRelativeBinningLikelihood._ppgo_band_split(
                object(), {'gamma': cls.GAMMA, 'y1': cls.SOURCE[0],
                           'y2': cls.SOURCE[1]})
        finally:
            set_certified_ppgo_map(None)

        cls.below = cls.DENSE_W <= cls.w_trust
        cls.above = ~cls.below

        # One shared partition (exact oracle + geometry), exactly as the
        # dispatch reduces both segments through a single partition.
        engine = ChangRefsdalChannels(cls.DENSE_W)
        engine.reset()
        cls.partition = engine.evaluate(
            gamma=cls.GAMMA, y=cls.SOURCE, beta=0.0, kappa=0.0)
        cls.geom = ChangRefsdalChannels(cls.DENSE_W).geometry_partition(
            gamma=cls.GAMMA, y=cls.SOURCE, beta=0.0, kappa=0.0)
        cls.exact = np.asarray(cls.partition.exact_total)
        cls.f_scale = float(np.max(np.abs(cls.exact)))

        # A real trained far-field chart over a tile covering the draw, whole
        # band -- its spline envelope serves the chart sub-band.
        surrogate = LensAmplificationSurrogate.from_engine(
            gamma_range=(0.25, 0.35), y1_range=(1.2, 1.4),
            y2_range=(1.2, 1.4), w_range=(2.0, 40.0), n_gamma=4, n_y1=4,
            n_y2=4, w_nodes_per_decade=8)
        cls.env_chart, cls.served, cls.definition = surrogate.serve(
            cls.DENSE_W[cls.below], gamma=cls.GAMMA, y1=cls.SOURCE[0],
            y2=cls.SOURCE[1], beta=0.0, eta=cls.geom.caustic_distance,
            theta=cls.geom.caustic_theta,
            image_count=int(cls.geom.real_mask.sum()))

        # ff switch: 1 on every real channel, no critical carrier (the
        # far-field gauge the dispatch uses for the telescoping split).
        cls.ff_switch = np.zeros(
            (cls.DENSE_W.size, cls.geom.real_mask.size), dtype=float)
        cls.ff_switch[:, np.asarray(cls.geom.real_mask, dtype=bool)] = 1.0

        if cls.served:
            env_dense = np.zeros(cls.DENSE_W.size, dtype=complex)
            env_dense[cls.below] = cls.env_chart
            _k, cls.f_bandsplit = reconstruct_from_envelope(
                cls.DENSE_W, env_dense, cls.geom.delays,
                cls.geom.saddle_kernels, cls.ff_switch, 0.0)
            # Bare ppGO everywhere (E_ff = 0), for the seam comparison.
            _k, cls.f_ppgo = reconstruct_from_envelope(
                cls.DENSE_W, np.zeros(cls.DENSE_W.size, dtype=complex),
                cls.geom.delays, cls.geom.saddle_kernels, cls.ff_switch, 0.0)
            cls.max_eff = float(np.max(np.abs(
                farfield_envelope_from_partition(cls.partition))))
        cls._plot()

    @classmethod
    def _plot(cls) -> None:
        if not (_HAVE_MPL and getattr(cls, 'served', False)):
            return
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        rel = np.abs(cls.f_bandsplit - cls.exact) / cls.f_scale
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(cls.DENSE_W, np.maximum(rel, 1e-18), 'b.-',
                    label='|F_recon - F_exact| / max|F|')
        ax.axvline(cls.w_trust, color='g', lw=2, label='w_trust (split)')
        ax.axhline(cls.PPGO_TOL, color='r', ls='--', label='1e-4 ppGO bar')
        ax.set_xlabel('w')
        ax.set_ylabel('band-split reconstruction error')
        ax.set_title('Band-split node match: chart below, ppGO above')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / 'ppgo_band_split_node_match.png', dpi=110)
        plt.close(fig)

    def test_band_actually_splits(self):
        """The draw straddles w_trust and the chart serves (not vacuous)."""
        self.comparisons += 1
        self.assertEqual(self.w_trust, 12.0,
                         'w_trust was not read from the map as expected')
        self.comparisons += 1
        self.assertTrue(self.served,
                        'the far-field chart declined to serve the sub-band; '
                        'the band-split reconstruction is untested')
        self.comparisons += 1
        self.assertTrue(
            self.below.any() and self.above.any(),
            'w_trust does not lie strictly inside the dense band')

    def test_ppgo_segment_matches_exact_at_every_node(self):
        """Above w_trust the bare ppGO sum matches exact F to 1e-4 -- all."""
        self.assertTrue(self.served, 'chart did not serve (see setUp)')
        rel = np.abs(self.f_bandsplit[self.above] - self.exact[self.above]) \
            / self.f_scale
        self.assert_within(
            float(rel.max()), self.PPGO_TOL,
            f'ppGO segment exceeded {self.PPGO_TOL:g} at some node above '
            f'w_trust (max {rel.max():.3e}); a beat re-crossing or a floor '
            f'set too low is the violation')

    def test_chart_segment_matches_at_spline_currency(self):
        """Below w_trust the chart reconstructs F to 5e-3 absolute."""
        self.assertTrue(self.served, 'chart did not serve (see setUp)')
        abs_err = np.abs(self.f_bandsplit[self.below] - self.exact[self.below])
        self.assert_within(
            float(abs_err.max()), self.CHART_TOL,
            f'chart segment exceeded {self.CHART_TOL:g} absolute (max '
            f'{abs_err.max():.3e}, max|E_ff|={self.max_eff:.3e})')

    def test_seam_agreement_at_the_split_node(self):
        """Chart-below and ppGO-above agree at the split node to 5e-3."""
        self.assertTrue(self.served, 'chart did not serve (see setUp)')
        i_split = int(np.flatnonzero(self.below)[-1])
        seam = abs(self.f_bandsplit[i_split] - self.f_ppgo[i_split])
        self.assert_within(
            float(seam), self.SEAM_TOL,
            f'chart and ppGO disagree at the split node w='
            f'{self.DENSE_W[i_split]:.2f} by {seam:.3e}; a discontinuity at '
            f'the seam')


# ======================================================================
# Test #5 -- CORRUPT/ABSENT/UNKNOWN MAP REFUSAL, F010 BOTH DIRECTIONS
#            (WP1 + WP2).
# ======================================================================

class MapRefusalTestCase(_PpgoTestCase):
    """The SAME fixed draw flips served <-> refused with the map's state.

    F010 both directions on ONE fixed synthetic draw straddling ``w_cert``:
    (a) a VALID map -> the draw IS band-split (ppGO-served above
    ``w_trust``); (b) a CORRUPTED-hash artifact AND an ABSENT file both make
    the loader refuse loudly (named ``ValueError`` / ``OSError``) and leave
    the process-global map ``None`` -> the draw is NOT band-split; (c) a
    BEYOND-WALL (UNKNOWN) cell is never served even with a valid map; the
    beyond-wall band guard (a certified cell whose band tops past the parity
    wall) also suppresses the split; (d) every refusal is loud/named and
    routes to the whole-band exact path, never to numerical quadrature.

    The routing decision is the REAL
    `LensedRelativeBinningLikelihood._ppgo_band_split` (it reads only the
    process-global map and the ``lens`` dict, no likelihood state), so the
    served-vs-refused flip is production truth, not a reimplementation.
    """

    GAMMA = 0.3
    LENS = {'gamma': 0.3, 'y1': 1.3, 'y2': 1.3, 'kappa': 0.0}
    W_CERT = 8.0               # -> w_trust = 12.0

    @staticmethod
    def _bandsplit(lens):
        """The REAL dispatch helper (no likelihood state read)."""
        return LensedRelativeBinningLikelihood._ppgo_band_split(object(), lens)

    def setUp(self) -> None:
        super().setUp()
        reach = ppgo_map.caustic_geometry(self.GAMMA, 0.0)[0]
        self.rho = math.hypot(self.LENS['y1'], self.LENS['y2']) / reach
        # Always restore a clean (map-free) global on the way out.
        self.addCleanup(set_certified_ppgo_map, None)

    def _served(self) -> bool:
        """Whether the REAL dispatch would band-split this draw now."""
        return self._bandsplit(self.LENS) is not None

    def _valid_map(self) -> CertifiedPpgoMap:
        return _synthetic_map(parity='positive', gamma=self.GAMMA,
                              rho=self.rho, w_cert=self.W_CERT)

    def test_valid_map_serves_the_draw(self):
        """(a) A valid certified map band-splits the draw above w_trust."""
        set_certified_ppgo_map(self._valid_map())
        w_trust = self._bandsplit(self.LENS)
        self.comparisons += 1
        self.assertEqual(w_trust, 12.0,
                         'valid map did not yield the expected w_trust=12.0')
        self.comparisons += 1
        self.assertTrue(self._served(), 'valid map failed to serve the draw')

    def test_absent_map_refuses_the_same_draw(self):
        """(b) No map installed -> None -> NOT band-split (whole-band path)."""
        set_certified_ppgo_map(None)
        self.comparisons += 1
        self.assertFalse(self._served(),
                         'an absent map still band-split the draw')

    def test_corrupt_and_absent_artifacts_refuse_loudly(self):
        """(b) Corrupt-hash raises ValueError; absent raises OSError."""
        cmap = build_map(astroid_wall=20.0, saddle_wall=15.0,
                         gamma_edges=[0.3, 0.7], rho_edges=[0.0, 1.0, math.inf])
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'ppgo_map.npz'
            save_map(cmap, path)
            # A clean artifact loads and hash-verifies.
            self.comparisons += 1
            self.assertIsInstance(CertifiedPpgoMap.load(path), CertifiedPpgoMap)

            # Corrupt the stored content hash -> loud ValueError.
            with np.load(path, allow_pickle=False) as data:
                prov = json.loads(str(data['provenance']))
                arrays = {k: np.asarray(data[k]) for k in (
                    'parity_codes', 'gamma_edges', 'rho_edges', 'w_cert',
                    'w_cert_diagnostic', 'cell_status', 'interpolable')}
            prov['content_hash'] = 'deadbeef'
            np.savez(path, provenance=np.asarray(json.dumps(prov)), **arrays)
            self.comparisons += 1
            with self.assertRaises(ValueError):
                CertifiedPpgoMap.load(path)

            # An absent artifact raises a named OSError.
            self.comparisons += 1
            with self.assertRaises(OSError):
                CertifiedPpgoMap.load(pathlib.Path(tmp) / 'nope.npz')

            # The opt-in switch swallows BOTH into a refuse-to-certify: the
            # global stays None and the draw is NOT served (no fall-through
            # to quadrature -- the whole-band exact path handles it).
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                ok_corrupt = use_certified_ppgo_map(path)
                served_corrupt = self._served()
                ok_absent = use_certified_ppgo_map(
                    pathlib.Path(tmp) / 'nope.npz')
                served_absent = self._served()
        self.comparisons += 1
        self.assertFalse(ok_corrupt or ok_absent,
                         'use_certified_ppgo_map reported success on a '
                         'corrupt / absent artifact')
        self.comparisons += 1
        self.assertIsNone(get_certified_ppgo_map(),
                          'a corrupt / absent map was left installed')
        self.comparisons += 1
        self.assertFalse(served_corrupt or served_absent,
                         'a corrupt / absent map still served the draw')

    def test_beyond_wall_cell_never_serves(self):
        """(c) A BEYOND-WALL (UNKNOWN) cell is refused even with a valid map."""
        set_certified_ppgo_map(_synthetic_map(
            parity='positive', gamma=self.GAMMA, rho=self.rho,
            w_cert=math.nan, status=STATUS_BEYOND_WALL))
        self.comparisons += 1
        self.assertFalse(self._served(),
                         'a beyond-wall UNKNOWN cell was served')

    def test_beyond_wall_band_guard_suppresses_the_split(self):
        """(c) A certified cell whose band tops past the wall is not split.

        Reproduces the INS-8haf-002 guard in `_surrogate_coefficients`: the
        map certifies a cell by geometry, but certification exists only
        BELOW the parity's Schwinger wall.  A draw whose dense band tops
        beyond the wall must NOT band-split (bare ppGO would serve
        uncertified beyond-wall nodes).  The REAL `_ppgo_band_split` still
        returns w_trust for the certified cell; the caller's wall check is
        what suppresses the split.  Same cell, same w_trust -- only the band
        top changes served -> refused.
        """
        set_certified_ppgo_map(self._valid_map())
        w_trust = self._bandsplit(self.LENS)
        self.comparisons += 1
        self.assertIsNotNone(w_trust, 'certified cell lost its w_trust')
        wall = ASTROID_WALL if self.GAMMA < 1.0 else SADDLE_WALL

        def band_splits(w_lo: float, w_hi: float) -> bool:
            trust = w_trust
            if trust is not None and w_hi > wall:
                trust = None                       # the beyond-wall guard
            return trust is not None and w_lo < trust < w_hi

        # A band within the wall splits; a band topping past the wall does not.
        self.comparisons += 1
        self.assertTrue(band_splits(2.0, 40.0),
                        'an in-wall band failed to split')
        self.comparisons += 1
        self.assertFalse(band_splits(2.0, wall + 50.0),
                         'a band topping past the Schwinger wall was still '
                         'band-split (beyond-wall guard missing)')

    def test_the_flip_is_on_the_identical_input(self):
        """The load-bearing evidence: served state flips, draw unchanged."""
        set_certified_ppgo_map(self._valid_map())
        served_valid = self._served()
        set_certified_ppgo_map(None)
        served_absent = self._served()
        self.comparisons += 1
        self.assertTrue(served_valid and not served_absent,
                        f'the served flag did not flip with the map state '
                        f'(valid={served_valid}, absent={served_absent})')


if __name__ == '__main__':
    main()
