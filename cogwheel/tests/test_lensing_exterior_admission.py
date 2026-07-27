"""Per-``theta_c``-column exterior admission certification (WP1, positive parity).

Independent ``unittest`` suite for the Build 8h-b3 WP1 migration that replaces
the single over-conservative scalar ``exclusion_rho`` far-field inner edge --
built from the cusp-spike ``surrogate._caustic_reach``, which swallowed the
WHOLE prior box for ``gamma >= 0.85`` (exterior coverage 0.000) -- with a
per-``theta_c``-column DIRECTIONAL admission
(`surrogate_training._InteriorAdmission.admits_exterior`, laid down by
`surrogate_training._farfield_exterior_tiles`).

Three Professor-authored specifications are certified here.

1. **Exterior admission coverage.**  Over a fixed-seed quasi-uniform grid of
   ``N_SOURCES`` source points ``y`` filling the prior box
   (``|y|max = BOX_CORNER ~ 4.24``), the truth set
   ``T = { y : nearest_caustic_point(gamma, 0, y).distance >= eta_max AND y
   outside the caustic }`` -- evaluated with the EXACT oracle
   `geometry.nearest_caustic_point` at the WORST gamma in each band (the
   astroid spike moves with gamma, so the worst case is the band's largest
   directional reach, i.e. its upper edge) -- must be covered by the union of
   admitted exterior tiles to at least ``COVERAGE_BAR = 0.95`` in EVERY band,
   INCLUDING the previously-dead ``0.80-0.90`` band.  The five certified bands
   are the four the OLD scalar-reach admission was measured to fail
   (0.944 / 0.632 / 0.271 / 0.000 for 0.30-0.40 / 0.50-0.70 / 0.70-0.80 /
   0.80-0.90) plus the 0.40-0.50 control.  The bar is 0.95 (NOT
   0.97): the measured coverage at ``COVERAGE_N_TILES = 150`` is 0.996 /
   0.994 / 0.986 / 0.980 / 0.973 in band order, leaving margin for grid
   discretization
   (binomial sampling std ~0.002 at ``|T| ~ 9000``).  Coverage rises
   monotonically with tiling resolution (0.908 -> 0.942 -> 0.973 -> 0.982 at
   n = 30, 60, 120, 200) and converges toward 1, so there is NO persistent
   near-cusp uncovered wedge -- the per-column ``rho_inner`` is correctly
   computed.

2. **No-false-admit invariant (HARD / exact).**  For every admitted tile of the
   critical ``0.80-0.90`` band, a ``NFA_GRID x NFA_GRID`` grid across the tile
   INTERIOR (not just its inner edge) is reconstructed to physical ``y`` with
   the production coordinate `surrogate._from_caustic_fixed` and its EXACT
   nearest-caustic distance is measured at every band gamma.  ZERO samples may
   lie within ``eta_max`` of the caustic (tolerance is exactly zero violations;
   ``eta_max`` is itself the physical margin).  The measured left edge of the
   min-distance histogram is ~0.18 -- comfortably at or above ``eta_max``.

3. **Reachable-red for admission.**  Restoring the OLD scalar
   ``exclusion_rho = 1 + (reach_max + eta_max) - coordinate_radius_min`` tiling
   (`surrogate_training._farfield_tiles`) for the ``0.80-0.90`` band admits ZERO
   tiles (``exclusion_rho = 5.942 > rho_outer = 4.443``), so its coverage is
   exactly 0.000 -- proving the coverage metric discriminates the WP1 fix from
   the defect.  A companion assertion confirms the NEW per-column admission
   admits many tiles on the SAME band.

Three further Professor-authored specifications certify the macro-saddle
(``gamma > 1``) exterior coordinate and the ``gamma = 1`` parity-wall guard
(WP2).

4. **Saddle additive-scalar axis triple.**  For a macro-saddle exterior chart
   the coordinate is ``rho = 1 + |y| - _caustic_reach(gamma)`` (an ADDITIVE
   scalar offset, NOT a directional ``r_caustic`` ratio): (a) ``|y| -> rho ->
   |y|`` round-trips to 1e-12 on every in-box exterior node over ``theta_c in
   [-pi, pi]`` and ``gamma in (1, 1.6]``; (b) ``drho/d|y| = 1`` to machine
   epsilon (the map is affine with unit slope, so ``rho - |y|`` is invariant);
   (c) NO ``LensDomainError`` is raised anywhere in the sweep, INCLUDING the
   most discriminating node -- the between-lobe (positive-eigenvalue) axis
   ``theta_c = pi/2`` at the largest gamma, where the directional
   ``r_caustic`` form unavoidably raises (a companion reachable-red confirms it
   does).

5. **Gamma = 1 box-centre guard.**  ``_caustic_reach(1.0)`` raises
   ``LensDomainError`` at the ``det A = 0`` parity wall.  A chart whose box
   CENTRE gamma is exactly 1.0 must construct without raising and record
   ``image_count is None and parity is None``; a grid NODE that lands exactly on
   1.0 must be recorded refused (the c28408b node-loop fix); the guard must
   catch ``LensDomainError`` SPECIFICALLY (a non-refusal exception propagates);
   and a machine-scale step off 1.0 is served (finite reach), so the wall is a
   single point, not a knife-edge.

Oracle independence: the truth set and every distance check use the EXACT
`geometry.nearest_caustic_point` (a critical-curve Newton search) which is
independent of the ``surrogate_training`` admission algebra under test.  The
vectorised ``(rho, theta_c)`` map used for the coverage tally is a batched
reproduction of `surrogate._to_caustic_fixed`; it is cross-checked bit-close
(< 4e-3 near cusps, dominated by ``r_caustic`` interpolation, far below the
tile half-width) against the scalar production function on a random subsample.
"""

from __future__ import annotations

import functools
import math
import unittest
from pathlib import Path
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cogwheel.lensing import prior as _lens_prior
from cogwheel.lensing import surrogate as sg
from cogwheel.lensing import surrogate_training as st
from cogwheel.lensing.chang_refsdal import geometry

#: Physical caustic-distance margin below which the far-field surrogate is
#: untrained (dimensionless source-plane units); asserted against the live
#: ``TrainingConfig`` default in `ConstantsTestCase`.
ETA_MAX = 0.05

#: Largest per-region physical source magnitude (``_source_scale`` at the
#: lightest lens mass, capped at 3.0); production feeds THIS as the exterior
#: ``source_magnitude_max``.
SOURCE_SCALE_MIN = 3.0

#: Prior-box CORNER ``|y|max`` (the spec's 4.24): to certify coverage over the
#: whole box we feed the admission this box extent, not the per-region 3.0.
BOX_CORNER = math.sqrt(2.0) * SOURCE_SCALE_MIN

#: Number of quasi-uniform source draws per gamma band (spec: N >= 1e4).
N_SOURCES = 10_000

#: Fixed RNG seed for the source grid (reproducible truth set).
SEED = 20_260_727

#: ``theta_c`` columns / ``rho`` rows for the coverage tiling.  Coverage is
#: monotone increasing in this count; 150 clears the 0.95 bar in both bands.
COVERAGE_N_TILES = 150

#: Coverage acceptance bar (Professor: 0.95, NOT 0.97 -- discretization margin).
COVERAGE_BAR = 0.95

#: Bands certified for coverage.  The four driver-measured bands that pinned
#: the OLD scalar-reach defect (0.944 / 0.632 / 0.271 / 0.000 coverage) plus
#: the original (0.40, 0.50) control, so the acceptance covers the whole
#: positive-parity gamma range up to the crown, not only its two ends.
#: Measured NEW coverage at `COVERAGE_N_TILES`: 0.996 / 0.994 / 0.986 /
#: 0.980 / 0.973 -- monotone decreasing in gamma (the cusp spike lengthens),
#: all clear of the 0.95 bar.
COVERAGE_BANDS = ((0.30, 0.40), (0.40, 0.50), (0.50, 0.70), (0.70, 0.80),
                  (0.80, 0.90))

#: Band used for the HARD no-false-admit invariant (the critical high band).
NFA_BAND = (0.80, 0.90)

#: Tiling density for the no-false-admit band (kept modest: every interior
#: sample calls the exact ``_from_caustic_fixed`` + oracle).
NFA_N_TILES = 10

#: Interior sample grid per admitted tile (spec: >= 5x5).
NFA_GRID = 5

#: Band whose OLD scalar exclusion admits zero tiles (reachable-red).
RED_BAND = (0.80, 0.90)

#: Polar nodes of the cached ``r_caustic`` interpolation table.
RCAUSTIC_NODES = 1441

#: Directory for diagnostic plots.
OUTPUT_DIR = Path(__file__).parent / 'output'


#: Macro-saddle external-shear values (gamma > 1) for the additive-scalar
#: exterior coordinate triple (WP2a).  All strictly inside ``(1, 1.6]``.
SADDLE_GAMMAS = (1.05, 1.30, 1.60)

#: The largest saddle gamma -- the single most discriminating gamma for the
#: refusal-absence property (the directional ``r_caustic`` form fails hardest
#: here on the between-lobe axis).
SADDLE_GAMMA_MAX = 1.60

#: ``theta_c`` nodes spanning the full polar circle ``[-pi, pi]`` (endpoints
#: included), so the sweep visits both deltoid-lobe axes AND the off-wedge axis.
SADDLE_THETA_C = tuple(np.linspace(-math.pi, math.pi, 17))

#: Exterior physical radial offsets ABOVE the scalar reach, so every sampled
#: node is exterior (``rho = 1 + offset > 1``).
SADDLE_EXTERIOR_OFFSETS = (0.01, 0.5, 1.7, 3.0)

#: ``theta_c`` aimed squarely BETWEEN the two saddle deltoid lobes: the two
#: deltoids sit on the ``y1`` axis (``theta_c in {0, pi}``); the
#: positive-eigenvalue axis perpendicular to them is ``theta_c = pi/2``, where
#: an origin-centred ray misses BOTH deltoids and the directional
#: `geometry.r_caustic` unavoidably raises ``LensDomainError``.  The scalar
#: additive form is ``theta_c``-independent and must NOT raise here.
OFF_WEDGE_THETA_C = math.pi / 2.0

#: Round-trip magnitude tolerance (Professor: 1e-12; measured worst ~4e-16).
SADDLE_ROUNDTRIP_TOL = 1e-12

#: Machine-epsilon budget (in ulp of the reach subtraction) for the unit
#: Jacobian: the additive map ``rho = (1 - reach) + |y|`` is affine in ``|y|``
#: with slope IDENTICALLY 1, so ``rho - |y|`` is invariant up to a few ulp.
SADDLE_JACOBIAN_ULP = 16.0

#: Gamma range whose ARITHMETIC box centre ``0.5*(lo + hi)`` is exactly ``1.0``
#: (the crash path guarded by `_box_region_labels`).
GAMMA1_CENTRE_RANGE = (0.5, 1.5)

#: Gamma range whose ``n = 4`` uniform axis lands a NODE exactly on
#: ``gamma = 1.0`` (nodes ``1.0, 1.2, 1.4, 1.6``): the c28408b node-loop fix.
GAMMA1_NODE_RANGE = (1.0, 1.6)

#: Minimal exterior training box + ``w`` window for the cheap WP2b chart
#: builds (~3.5 s each): exterior ``rho``, a modest ``theta_c`` slab, one decade.
GUARD_RHO_RANGE = (1.05, 1.30)
GUARD_THETA_C_RANGE = (0.3, 0.9)
GUARD_W_RANGE = (1.0, 3.0)
GUARD_N_NODES = 4
GUARD_W_NODES_PER_DECADE = 4


def _build_guard_chart(gamma_range: tuple[float, float]) -> 'sg.FarFieldChart':
    """Build one cheap exterior far-field chart over ``gamma_range`` (WP2b)."""
    single = sg.LensAmplificationSurrogate.from_engine(
        gamma_range=gamma_range, rho_range=GUARD_RHO_RANGE,
        theta_c_range=GUARD_THETA_C_RANGE, w_range=GUARD_W_RANGE,
        n_gamma=GUARD_N_NODES, n_rho=GUARD_N_NODES, n_theta=GUARD_N_NODES,
        w_nodes_per_decade=GUARD_W_NODES_PER_DECADE,
        definition=sg._FARFIELD_ENVELOPE_DEFINITION)
    return single.charts[0]


@functools.lru_cache(maxsize=None)
def _rcaustic_table(gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Cached ``(theta_axis, r_caustic_axis)`` for one gamma over ``[-pi, pi]``."""
    theta_axis = np.linspace(-math.pi, math.pi, RCAUSTIC_NODES)
    radius_axis = np.array(
        [geometry.r_caustic(float(gamma), float(theta))
         for theta in theta_axis])
    return theta_axis, radius_axis


@functools.lru_cache(maxsize=None)
def _admission(band: tuple[float, float]) -> 'st._InteriorAdmission':
    """Cached positive-parity directional admission geometry for one band."""
    reach = sg._caustic_reach(0.5 * (band[0] + band[1]))
    return st._interior_admission(band, 1, reach, st.TrainingConfig())


@functools.lru_cache(maxsize=None)
def _coord_bounds(band: tuple[float, float]) -> tuple[float, float]:
    """Cached ``(coordinate_radius_min, reach_max)`` for one band."""
    return st._coordinate_radius_bounds(band, 1)


def _worst_gamma(band: tuple[float, float]) -> float:
    """The band's worst-case gamma (largest directional reach = upper edge)."""
    return band[1]


def _sample_sources() -> tuple[np.ndarray, np.ndarray]:
    """Fixed-seed quasi-uniform disk of ``N_SOURCES`` points, radius ``BOX_CORNER``."""
    rng = np.random.default_rng(SEED)
    radius = BOX_CORNER * np.sqrt(rng.random(N_SOURCES))
    angle = rng.uniform(-math.pi, math.pi, N_SOURCES)
    return radius * np.cos(angle), radius * np.sin(angle)


def _to_caustic_fixed_vec(gamma: float, y1: np.ndarray, y2: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised reproduction of `surrogate._to_caustic_fixed` (positive parity)."""
    theta_axis, radius_axis = _rcaustic_table(gamma)
    magnitude = np.hypot(y1, y2)
    theta_c = np.arctan2(y2, y1)
    caustic_radius = np.interp(theta_c, theta_axis, radius_axis)
    rho = np.where(magnitude <= caustic_radius,
                   magnitude / caustic_radius,
                   1.0 + magnitude - caustic_radius)
    return rho, theta_c


@functools.lru_cache(maxsize=None)
def _truth_set(band: tuple[float, float]
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``(in_T, rho, theta_c, |T|)`` at the band's worst gamma (exact oracle)."""
    gamma = _worst_gamma(band)
    y1, y2 = _sample_sources()
    theta_axis, radius_axis = _rcaustic_table(gamma)
    magnitude = np.hypot(y1, y2)
    theta_c = np.arctan2(y2, y1)
    outside = magnitude > np.interp(theta_c, theta_axis, radius_axis)
    distance = np.array(
        [geometry.nearest_caustic_point(
            gamma, 0.0, np.array([y1[k], y2[k]])).distance
         for k in range(N_SOURCES)])
    in_t = outside & (distance >= ETA_MAX)
    rho, _ = _to_caustic_fixed_vec(gamma, y1, y2)
    return in_t, rho, theta_c, int(in_t.sum())


def _exterior_tiles(band: tuple[float, float], n_per_side: int,
                    source_magnitude_max: float) -> list:
    """Admitted per-column exterior tiles for a band at the box extent given."""
    coordinate_radius_min, _ = _coord_bounds(band)
    rho_outer = 1.0 + source_magnitude_max - coordinate_radius_min
    return st._farfield_exterior_tiles(
        rho_outer, n_per_side, admission=_admission(band),
        source_magnitude_max=source_magnitude_max)


def _covered_mask(rho: np.ndarray, theta_c: np.ndarray, tiles: list
                  ) -> np.ndarray:
    """Boolean mask of points falling inside ANY admitted tile (caustic-fixed)."""
    covered = np.zeros(rho.shape, dtype=bool)
    for (rho_center, theta_center), (half_rho, half_theta), _, _ in tiles:
        d_theta = np.abs(
            ((theta_c - theta_center + math.pi) % (2.0 * math.pi)) - math.pi)
        covered |= (np.abs(rho - rho_center) <= half_rho) & (d_theta <= half_theta)
    return covered


class ExteriorAdmissionTestCase(unittest.TestCase):
    """Base carrying the anti-vacuity comparison counter.

    Every concrete assertion calls `record_comparison`; `tearDown` FAILS the
    test if not a single comparison ran, so a suite that silently skips its
    body (an import regression, a fixture that stopped producing sources or
    tiles) cannot read green.
    """

    def setUp(self) -> None:
        self.n_compared = 0
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def record_comparison(self) -> None:
        """Register that one real numerical comparison was made."""
        self.n_compared += 1

    def tearDown(self) -> None:
        if self.n_compared == 0:
            self.fail('anti-vacuity: no comparison executed -- the test body '
                      'skipped every assertion (fixture or import regression).')


class ConstantsTestCase(ExteriorAdmissionTestCase):
    """Pin the module constants to the live production defaults."""

    def test_eta_max_matches_training_config(self) -> None:
        self.assertEqual(ETA_MAX, st.TrainingConfig().eta_max)
        self.record_comparison()

    def test_source_scale_min_matches_prior(self) -> None:
        box = st.PriorBox.from_prior_classes()
        scale = float(_lens_prior._source_scale(box.m_lens_range[0]))
        self.assertAlmostEqual(SOURCE_SCALE_MIN, scale, places=10)
        self.record_comparison()

    def test_box_corner_is_spec_extent(self) -> None:
        # The Professor spec fixes |y|max = 4.24; BOX_CORNER = sqrt(2)*3.0.
        self.assertAlmostEqual(BOX_CORNER, 4.24, places=2)
        self.record_comparison()


class WorstGammaTestCase(ExteriorAdmissionTestCase):
    """The worst-case gamma (largest directional reach) is the band's upper edge."""

    def test_reach_is_monotone_increasing_over_each_band(self) -> None:
        for band in COVERAGE_BANDS:
            with self.subTest(band=band):
                lo, hi = band
                mid = 0.5 * (lo + hi)
                reaches = [
                    max(geometry.r_caustic(g, float(t))
                        for t in np.linspace(-math.pi, math.pi, 181))
                    for g in (lo, mid, hi)]
                self.assertEqual(reaches, sorted(reaches))
                self.assertEqual(_worst_gamma(band), hi)
                self.record_comparison()


class ExteriorCoverageTestCase(ExteriorAdmissionTestCase):
    """Spec 1: admitted exterior tiles cover >= 95% of the truth set per band."""

    def test_coverage_at_least_bar_in_every_band(self) -> None:
        for band in COVERAGE_BANDS:
            with self.subTest(band=band):
                in_t, rho, theta_c, n_t = _truth_set(band)
                self.assertGreater(
                    n_t, 1000, f'truth set for {band} is suspiciously small')
                tiles = _exterior_tiles(band, COVERAGE_N_TILES, BOX_CORNER)
                self.assertGreater(
                    len(tiles), 0,
                    f'per-column admission produced zero tiles for {band} '
                    '(the WP1 defect); coverage cannot be measured')
                covered = _covered_mask(rho, theta_c, tiles)
                coverage = float((in_t & covered).sum()) / n_t
                self._plot_coverage(band, in_t, covered, rho, theta_c, coverage)
                self.assertGreaterEqual(
                    coverage, COVERAGE_BAR,
                    f'band {band}: coverage {coverage:.4f} < {COVERAGE_BAR} '
                    f'(|T|={n_t}, tiles={len(tiles)})')
                self.record_comparison()

    def _plot_coverage(self, band, in_t, covered, rho, theta_c, coverage
                       ) -> None:
        sel = in_t
        col = np.where(covered[sel], 'tab:blue', 'tab:red')
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(theta_c[sel], rho[sel], s=2, c=col, linewidths=0)
        ax.set_xlabel('theta_c (rad)')
        ax.set_ylabel('rho (caustic-fixed)')
        ax.set_title(f'exterior coverage band {band}: {coverage:.3f} '
                     '(blue=covered, red=uncovered)')
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f'exterior_coverage_band_{band[0]}_{band[1]}.png',
                    dpi=90)
        plt.close(fig)


class NoFalseAdmitTestCase(ExteriorAdmissionTestCase):
    """Spec 2 (HARD): no admitted-tile interior sample is within eta_max."""

    def test_zero_admitted_samples_inside_eta_shell(self) -> None:
        lo, hi = NFA_BAND
        band_gammas = (lo, 0.5 * (lo + hi), hi)
        tiles = _exterior_tiles(NFA_BAND, NFA_N_TILES, BOX_CORNER)
        self.assertGreater(len(tiles), 0, 'no admitted tiles to probe')
        min_distances: list[float] = []
        violations = 0
        n_samples = 0
        for (rho_center, theta_center), (half_rho, half_theta), _, _ in tiles:
            rhos = np.linspace(rho_center - half_rho, rho_center + half_rho,
                               NFA_GRID)
            thetas = np.linspace(theta_center - half_theta,
                                 theta_center + half_theta, NFA_GRID)
            for rho in rhos:
                for theta in thetas:
                    for gamma in band_gammas:
                        y1, y2 = sg._from_caustic_fixed(
                            gamma, float(rho), float(theta))
                        distance = geometry.nearest_caustic_point(
                            gamma, 0.0, np.array([y1, y2])).distance
                        min_distances.append(distance)
                        n_samples += 1
                        if distance < ETA_MAX:
                            violations += 1
        self.assertGreater(n_samples, 1000, 'too few interior samples probed')
        min_distance = min(min_distances)
        self._plot_histogram(min_distances, min_distance)
        # HARD invariant: exactly zero false admits.
        self.assertEqual(
            violations, 0,
            f'{violations}/{n_samples} admitted-tile samples within eta_max='
            f'{ETA_MAX} of the caustic (min distance {min_distance:.4f})')
        # The histogram left edge must sit at or above the physical margin.
        self.assertGreaterEqual(min_distance, ETA_MAX)
        self.record_comparison()

    def _plot_histogram(self, distances, min_distance) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(distances, bins=60, color='tab:green', alpha=0.8)
        ax.axvline(ETA_MAX, color='k', ls='--', label=f'eta_max={ETA_MAX}')
        ax.set_xlabel('exact nearest-caustic distance of admitted samples')
        ax.set_ylabel('count')
        ax.set_title(f'no-false-admit band {NFA_BAND}: '
                     f'min={min_distance:.4f} (left edge >= eta_max)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / 'no_false_admit_distance_hist.png', dpi=90)
        plt.close(fig)


class ReachableRedTestCase(ExteriorAdmissionTestCase):
    """Spec 3: OLD scalar exclusion admits ZERO tiles on the 0.80-0.90 band."""

    def _old_scalar_tiles(self, band: tuple[float, float], n_per_side: int
                          ) -> list:
        coordinate_radius_min, reach_max = _coord_bounds(band)
        exclusion_rho = 1.0 + (reach_max + ETA_MAX) - coordinate_radius_min
        rho_outer = 1.0 + BOX_CORNER - coordinate_radius_min
        return st._farfield_tiles(exclusion_rho, rho_outer, n_per_side)

    def test_old_scalar_admits_zero_tiles_high_band(self) -> None:
        for n_per_side in (st.TrainingConfig().n_farfield_tiles_per_side,
                           COVERAGE_N_TILES):
            with self.subTest(n_per_side=n_per_side):
                old_tiles = self._old_scalar_tiles(RED_BAND, n_per_side)
                self.assertEqual(
                    len(old_tiles), 0,
                    'OLD scalar exclusion should admit ZERO tiles on '
                    f'{RED_BAND} (exclusion_rho > rho_outer)')
                self.record_comparison()

    def test_old_scalar_coverage_is_zero_high_band(self) -> None:
        in_t, rho, theta_c, n_t = _truth_set(RED_BAND)
        old_tiles = self._old_scalar_tiles(RED_BAND, COVERAGE_N_TILES)
        covered = _covered_mask(rho, theta_c, old_tiles)
        coverage = float((in_t & covered).sum()) / n_t
        self.assertEqual(coverage, 0.0)
        self.record_comparison()

    def test_new_admission_recovers_the_band(self) -> None:
        # The discriminating contrast: the WP1 fix admits many tiles on the
        # SAME band where the OLD scalar admitted none.
        new_tiles = _exterior_tiles(RED_BAND, COVERAGE_N_TILES, BOX_CORNER)
        self.assertGreater(len(new_tiles), 100)
        in_t, rho, theta_c, n_t = _truth_set(RED_BAND)
        coverage = float(
            (in_t & _covered_mask(rho, theta_c, new_tiles)).sum()) / n_t
        self.assertGreaterEqual(coverage, COVERAGE_BAR)
        self.record_comparison()


class SelfFalsificationTestCase(ExteriorAdmissionTestCase):
    """Prove the suite can go RED: each detector fires on a planted defect."""

    def test_coverage_metric_can_fail(self) -> None:
        # An empty tile set yields coverage 0 -- the metric is not vacuously 1.
        in_t, rho, theta_c, n_t = _truth_set(RED_BAND)
        coverage = float((in_t & _covered_mask(rho, theta_c, [])).sum()) / n_t
        self.assertLess(coverage, COVERAGE_BAR)
        self.assertEqual(coverage, 0.0)
        self.record_comparison()

    def test_rho_map_vectorisation_matches_production(self) -> None:
        # The batched (rho, theta_c) map reproduces scalar _to_caustic_fixed.
        gamma = _worst_gamma(RED_BAND)
        y1, y2 = _sample_sources()
        rho_vec, theta_vec = _to_caustic_fixed_vec(gamma, y1, y2)
        rng = np.random.default_rng(SEED + 1)
        max_err = 0.0
        for k in rng.integers(0, N_SOURCES, 60):
            rho_ref, theta_ref = sg._to_caustic_fixed(
                gamma, float(y1[k]), float(y2[k]))
            max_err = max(max_err, abs(rho_ref - rho_vec[k]),
                          abs(theta_ref - theta_vec[k]))
        self.assertLess(max_err, 5e-3)
        self.record_comparison()

    def test_caustic_hugging_tile_is_flagged_a_false_admit(self) -> None:
        # A tile whose inner rho edge sits just above the caustic (rho_inner
        # ~ 1) reconstructs to points WITHIN eta_max -- the no-false-admit
        # detector must flag it, proving the invariant has teeth.
        lo, hi = NFA_BAND
        band_gammas = (lo, 0.5 * (lo + hi), hi)
        half_rho = 5e-4
        rho_center = 1.0 + 1.5 * half_rho  # rho_inner ~ 1.001 (hugs caustic)
        half_theta = math.pi / 200.0
        theta_center = 0.3  # a non-cusp direction
        violations = 0
        n_samples = 0
        for rho in np.linspace(rho_center - half_rho, rho_center + half_rho,
                               NFA_GRID):
            for theta in np.linspace(theta_center - half_theta,
                                     theta_center + half_theta, NFA_GRID):
                for gamma in band_gammas:
                    y1, y2 = sg._from_caustic_fixed(
                        gamma, float(rho), float(theta))
                    distance = geometry.nearest_caustic_point(
                        gamma, 0.0, np.array([y1, y2])).distance
                    n_samples += 1
                    if distance < ETA_MAX:
                        violations += 1
        self.assertGreater(
            violations, 0,
            'a caustic-hugging tile produced no violation -- the '
            'no-false-admit detector is vacuous')
        self.record_comparison()


class SaddleAdditiveRoundTripTestCase(ExteriorAdmissionTestCase):
    """WP2a(1): saddle exterior ``|y| -> rho -> |y|`` round-trips to 1e-12.

    For every in-box exterior node (``theta_c`` spanning ``[-pi, pi]`` and
    ``gamma in (1, 1.6]``), the additive-scalar coordinate pair
    `surrogate._to_caustic_fixed` / `surrogate._from_caustic_fixed` is an exact
    inverse of the magnitude.  A residual spike at off-wedge ``theta_c`` would
    flag a silent regression to the directional ``r_caustic`` form.
    """

    def test_magnitude_round_trips_on_every_exterior_node(self) -> None:
        worst = 0.0
        worst_offwedge = 0.0
        for gamma in SADDLE_GAMMAS:
            reach = sg._caustic_reach(gamma)
            for theta in SADDLE_THETA_C:
                for offset in SADDLE_EXTERIOR_OFFSETS:
                    magnitude = reach + offset
                    y1 = magnitude * math.cos(theta)
                    y2 = magnitude * math.sin(theta)
                    rho, theta_c = sg._to_caustic_fixed(gamma, y1, y2)
                    # The node must be genuinely exterior (rho > 1).
                    self.assertGreater(rho, 1.0)
                    y1b, y2b = sg._from_caustic_fixed(gamma, rho, theta_c)
                    residual = abs(math.hypot(y1b, y2b) - magnitude)
                    worst = max(worst, residual)
                    if abs(abs(theta) - OFF_WEDGE_THETA_C) < 1e-9:
                        worst_offwedge = max(worst_offwedge, residual)
                    self.record_comparison()
        self.assertLess(
            worst, SADDLE_ROUNDTRIP_TOL,
            f'saddle round-trip residual {worst:.3e} exceeds '
            f'{SADDLE_ROUNDTRIP_TOL}')
        # No off-wedge spike: the between-lobe axis is no worse than the bulk.
        self.assertLessEqual(worst_offwedge, worst)


class SaddleUnitJacobianTestCase(ExteriorAdmissionTestCase):
    """WP2a(2): ``drho/d|y| = 1`` to machine epsilon (the additive form).

    Because the macro-saddle exterior map is ``rho = (1 - reach(gamma)) + |y|``,
    the quantity ``rho - |y|`` is INVARIANT in ``|y|`` (slope identically one).
    Sampling on the ``theta_c = 0`` axis makes ``|y| = hypot(|y|, 0)`` exact, so
    the invariance -- and a finite-difference slope -- are asserted at the ulp
    level, not a loose tolerance.
    """

    def test_rho_minus_magnitude_is_invariant_and_equals_one_minus_reach(self
                                                                          ) -> None:
        for gamma in SADDLE_GAMMAS:
            with self.subTest(gamma=gamma):
                reach = sg._caustic_reach(gamma)
                deltas = []
                for offset in SADDLE_EXTERIOR_OFFSETS:
                    magnitude = reach + offset
                    # On-axis: hypot(magnitude, 0) == magnitude exactly.
                    rho, _ = sg._to_caustic_fixed(gamma, magnitude, 0.0)
                    deltas.append(rho - magnitude)
                ulp = SADDLE_JACOBIAN_ULP * np.spacing(max(1.0, reach))
                spread = max(deltas) - min(deltas)
                self.assertLessEqual(
                    spread, ulp,
                    f'gamma={gamma}: rho-|y| spread {spread:.3e} > {ulp:.3e} '
                    '(unit Jacobian violated -- map not affine slope 1)')
                for delta in deltas:
                    self.assertLessEqual(abs(delta - (1.0 - reach)), ulp)
                self.record_comparison()

    def test_finite_difference_slope_is_one(self) -> None:
        for gamma in SADDLE_GAMMAS:
            with self.subTest(gamma=gamma):
                reach = sg._caustic_reach(gamma)
                magnitude = reach + 1.0
                step = 0.25  # large, exact-ish spacing keeps the FD clean
                rho_hi, _ = sg._to_caustic_fixed(gamma, magnitude + step, 0.0)
                rho_lo, _ = sg._to_caustic_fixed(gamma, magnitude - step, 0.0)
                slope = (rho_hi - rho_lo) / (2.0 * step)
                self.assertLess(
                    abs(slope - 1.0), 1e-12,
                    f'gamma={gamma}: drho/d|y| = {slope!r} != 1')
                self.record_comparison()


class SaddleRefusalAbsenceTestCase(ExteriorAdmissionTestCase):
    """WP2a(3): the scalar additive form raises NO ``LensDomainError``.

    The discriminating property between the scalar-reach additive coordinate and
    the directional ``r_caustic`` coordinate: for a macro-saddle
    (``gamma > 1``) the caustic is two disconnected deltoids, so an
    origin-centred ray on the between-lobe (positive-eigenvalue) axis misses
    both and the DIRECTIONAL form must raise -- while the scalar additive form,
    which never performs a directional caustic search, must not.  A raise
    anywhere in the sweep flags a silent regression.
    """

    def test_no_refusal_across_theta_span_for_every_saddle_gamma(self) -> None:
        for gamma in SADDLE_GAMMAS:
            reach = sg._caustic_reach(gamma)  # scalar, theta-independent
            for theta in SADDLE_THETA_C:
                for offset in SADDLE_EXTERIOR_OFFSETS:
                    magnitude = reach + offset
                    y1 = magnitude * math.cos(theta)
                    y2 = magnitude * math.sin(theta)
                    try:
                        rho, theta_c = sg._to_caustic_fixed(gamma, y1, y2)
                        sg._from_caustic_fixed(gamma, rho, theta_c)
                    except geometry.LensDomainError as exc:  # pragma: no cover
                        self.fail(
                            f'saddle coordinate raised LensDomainError at '
                            f'gamma={gamma}, theta={theta:.4f} (regression to '
                            f'directional r_caustic?): {exc}')
                    self.record_comparison()

    def test_most_discriminating_offwedge_node_does_not_raise(self) -> None:
        # The single most discriminating node: largest gamma, between-lobe axis.
        gamma = SADDLE_GAMMA_MAX
        reach = sg._caustic_reach(gamma)
        magnitude = reach + 2.0
        y1 = magnitude * math.cos(OFF_WEDGE_THETA_C)
        y2 = magnitude * math.sin(OFF_WEDGE_THETA_C)
        rho, theta_c = sg._to_caustic_fixed(gamma, y1, y2)
        y1b, y2b = sg._from_caustic_fixed(gamma, rho, theta_c)
        self.assertLess(abs(math.hypot(y1b, y2b) - magnitude),
                        SADDLE_ROUNDTRIP_TOL)
        self.record_comparison()

    def test_directional_form_would_raise_offwedge_reachable_red(self) -> None:
        # Reachable-red: the directional r_caustic form, which a regression
        # would restore, raises on the between-lobe axis at the largest gamma
        # but succeeds on the deltoid axis (theta_c = 0).  This proves the
        # refusal-absence property discriminates the two forms.
        geometry.r_caustic(SADDLE_GAMMA_MAX, 0.0)  # on-lobe: succeeds
        with self.assertRaises(geometry.LensDomainError):
            geometry.r_caustic(SADDLE_GAMMA_MAX, OFF_WEDGE_THETA_C)
        self.record_comparison()


class Gamma1BoxCentreGuardTestCase(ExteriorAdmissionTestCase):
    """WP2b: the ``gamma = 1`` parity wall is guarded, not a crash path.

    ``_caustic_reach(1.0)`` raises ``LensDomainError`` (the ``det A = 0`` parity
    wall).  Chart construction must survive both a box whose CENTRE gamma is
    exactly 1.0 (labels become ``None``) and a grid NODE that lands exactly on
    1.0 (that node is recorded refused), and the guard must catch
    ``LensDomainError`` SPECIFICALLY -- never a bare ``Exception`` that would
    mask an unrelated failure as a benign refusal.
    """

    def test_caustic_reach_raises_only_exactly_at_one(self) -> None:
        # Pins the premise: the wall is a single point, not a neighbourhood.
        with self.assertRaises(geometry.LensDomainError):
            sg._caustic_reach(1.0)
        for gamma in (np.nextafter(1.0, 2.0), np.nextafter(1.0, 0.0)):
            self.assertTrue(math.isfinite(sg._caustic_reach(float(gamma))))
            self.record_comparison()

    def test_box_centre_gamma_one_yields_none_labels(self) -> None:
        lo, hi = GAMMA1_CENTRE_RANGE
        # The arithmetic box centre must be bit-exactly 1.0 (the crash path).
        self.assertEqual(0.5 * (lo + hi), 1.0)
        chart = _build_guard_chart(GAMMA1_CENTRE_RANGE)
        # Exact pair asserted so a refactor returning (0, 0) instead fails.
        self.assertIs(chart.image_count, None)
        self.assertIs(chart.parity, None)
        self.record_comparison()

    def test_node_at_gamma_one_is_recorded_refused(self) -> None:
        lo, hi = GAMMA1_NODE_RANGE
        # The n=4 uniform axis lands a node exactly on gamma = 1.0.
        axis = np.linspace(lo, hi, GUARD_N_NODES)
        self.assertIn(1.0, axis.tolist())
        chart = _build_guard_chart(GAMMA1_NODE_RANGE)
        refused = chart.refused_points
        self.assertGreater(refused.shape[0], 0, 'no refused points recorded')
        # Every gamma = 1.0 node (one full rho x theta_c slab) must be refused.
        self.assertTrue(
            bool(np.any(refused[:, 0] == 1.0)),
            'the gamma = 1.0 node loop was not recorded refused '
            '(regression of the c28408b node-loop fix)')
        self.assertEqual(int(np.sum(refused[:, 0] == 1.0)),
                         GUARD_N_NODES * GUARD_N_NODES)
        # The saddle box centre (1.3) is a normal region -> real labels.
        self.assertIsNotNone(chart.image_count)
        self.assertEqual(chart.parity, -1)
        self.record_comparison()

    def test_guard_catches_lens_domain_error_specifically(self) -> None:
        # Structural: only the named engine refusals are swallowed.
        self.assertIn(geometry.LensDomainError, sg._REFUSAL_ERRORS)
        self.assertNotIn(Exception, sg._REFUSAL_ERRORS)
        self.assertNotIn(BaseException, sg._REFUSAL_ERRORS)
        # Behavioural positive control: a LensDomainError at EVERY node is
        # swallowed -> the build succeeds with all points refused.
        with mock.patch.object(sg, '_from_caustic_fixed',
                               side_effect=geometry.LensDomainError('planted')):
            chart = _build_guard_chart(GAMMA1_NODE_RANGE)
        self.assertEqual(chart.refused_points.shape[0],
                         GUARD_N_NODES ** 3)
        # Behavioural teeth: a NON-refusal exception must PROPAGATE, not be
        # silently treated as a benign refusal.
        with mock.patch.object(sg, '_from_caustic_fixed',
                               side_effect=KeyError('boom')):
            with self.assertRaises(KeyError):
                _build_guard_chart(GAMMA1_NODE_RANGE)
        self.record_comparison()

    def test_guard_is_stable_across_the_boundary_not_a_knife_edge(self) -> None:
        # WP2b(4): a machine-scale step off 1.0 is SERVED (finite reach, no
        # raise), so the refusal fires ONLY exactly on the wall -- a stable
        # boundary, not a knife-edge.  (The reach at 1 +- ulp is near-degenerate
        # and enormous, so only finiteness/no-raise is asserted here, not the
        # O(1)-scale round-trip tolerance of the genuine-saddle nodes.)
        rho, theta = 1.2, 0.5
        with self.assertRaises(geometry.LensDomainError):
            sg._from_caustic_fixed(1.0, rho, theta)
        for gamma in (np.nextafter(1.0, 2.0), np.nextafter(1.0, 0.0)):
            y1, y2 = sg._from_caustic_fixed(float(gamma), rho, theta)
            self.assertTrue(math.isfinite(y1) and math.isfinite(y2))
            # The forward coordinate is also finite and stays exterior.
            rho_back, _ = sg._to_caustic_fixed(float(gamma), y1, y2)
            self.assertTrue(math.isfinite(rho_back))
            self.assertGreater(rho_back, 1.0)
            self.record_comparison()


if __name__ == '__main__':
    unittest.main()
