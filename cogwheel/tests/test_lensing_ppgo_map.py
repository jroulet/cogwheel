"""
Tests for the authoritative ppGO annulus converter `ppgo_map.annulus_rho`
and the monotonic-conservatism invariant of the ppGO exclusion gauge it
feeds (`surrogate_training` WP1, defects D1 + D2).

Two things are certified here.

D2 -- the extraction changed no numbers.  `annulus_rho(gamma, |y|, kappa)`
was lifted verbatim out of an inline expression that read the SCALAR
caustic reach (element 0 of `caustic_geometry`) and divided the physical
source magnitude by it.  `AnnulusRhoByteEquivalenceTestCase` reconstructs
that legacy expression INDEPENDENTLY inside the test -- ``np.hypot(y1, y2)
/ caustic_geometry(gamma, 0.0)[0]`` -- and demands EXACT equality
(``max |diff| == 0.0``), not closeness: a reach-index slip (element 1
instead of 0) or a stray normalisation would leave a nonzero residual that
a tolerance would hide.  Because `caustic_geometry` is deterministic, the
two independent calls return a bit-identical reach, so exact equality is
the right bar.  `AnnulusRhoGuardTestCase` pins the two input guards.

D1 -- the ppGO exclusion read point moved conservatively.  The stale
gauge derived the ppGO annulus coordinate from the PRE-narrowing outer
annulus (``rho = physical_exclusion_radius / reach``); the fix derives it
from the NARROWED served region via `annulus_rho`, feeding a source
magnitude ``|y|_fix = region_exclusion_rho - 1 + coordinate_radius_min``
that never exceeds the HEAD physical radius.  Both gauges divide by the
SAME reach (`_scalar_caustic_reach == caustic_geometry(gamma, 0)[0]`), so
the fix can only move the read point to a SMALLER rho (inner, closer to
the caustic) -- never a larger one.

  A note on the sign.  The Architect brief phrased the invariant as
  ``w_cert <= HEAD`` and "move the read point outward (harder cell)".
  MEASURED against the real helpers, the fix moves the read point INWARD
  (rho_fix = 0.98 < rho_head = 1.19), and the production module's own
  comment states that "the farther-out cell would report an EASIER
  (lower-w_cert) certification".  So near the caustic (small rho) w_cert
  is HIGHER, and the fix reads a HIGHER-or-equal w_cert cell.  This suite
  encodes the measured, code-consistent direction
  (``w_cert(fix) >= w_cert(head)``, i.e. never easier = a higher dispatch
  floor), not the brief's inverted words -- per the standing rule to
  record the measured quantity, not the brief's number.

The geometric half of D1 (``rho_fix <= rho_head``, strict under narrowing)
is sign-unambiguous and certified directly against the real helpers.  The
w_cert half rests on the map's w_cert being non-increasing in rho; that is
a property of the certified artifact (none is shipped), so it is
demonstrated on a SYNTHETIC monotone map built here, isolating the
conservatism MECHANISM the fix relies on.

Two further D1 cases sharpen the fix's guarantees.
`PpgoOrderingReachableRedTestCase` reproduces the exact ORDERING bug
(deriving the ppGO coordinate from the pre-narrowing outer annulus instead
of the narrowed served region) and shows the production "read no easier than
the served inner edge" invariant PASSES under the fixed ordering yet RAISES
under the buggy ordering -- a reachable-red guard, not a vacuous one.
`SaddleBranchByteIdentityTestCase` pins the untouched half: a macro-saddle
band (``parity != 1``, ``gamma > 1``) reads ``physical / reach`` verbatim,
bit-identical to the HEAD gauge expressed through the authoritative
converter, and its ppGO cell fields are unchanged; the saddle branch never
narrows.

`PpgoExclusionMonotonicConservatismTestCase` carries the invariant; the
base `_GaugeTestCase.tearDown` FAILS a test that ran zero comparisons so a
silently-skipping sweep cannot read green.  `SelfFalsificationTestCase`
proves the suite can go red: it reproduces the HEAD gauge and the
no-narrowing degenerate case (both must LOSE the strict "harder" property)
and corrupts the oracle / the map monotonicity to confirm the exact
equality and the conservatism comparison actually detect a fault.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from unittest import TestCase, main, mock

import numpy as np

from cogwheel.lensing.ppgo_map import (
    CertifiedPpgoMap, STATUS_CERTIFIED, STATUS_INVALID, UNKNOWN,
    annulus_rho, caustic_geometry)
from cogwheel.lensing import ppgo_map
from cogwheel.lensing.surrogate_training import (
    _coordinate_radius_bounds, _scalar_caustic_reach, _stratum_ppgo_boundary)

#: Directory for diagnostic plots (created lazily; hook-blocked from
#: directory listings, so generated files are verified via ``glob``).
_OUTPUT_DIR = Path(__file__).parent / 'output'

#: Positive-parity gamma band used for the D1 gauge reproduction.  A real,
#: mid-range band away from the parity boundary at ``gamma = 1``.
_BAND: tuple[float, float] = (0.45, 0.55)

#: Tube-shell half-width ``eta_max`` (dimensionless ``y``); the value the
#: module docstrings cite for a representative training config.  Enters the
#: physical exclusion radius ``reach_max + eta_max``.
_ETA_MAX = 0.05

#: Amount (in additive-exterior rho units) by which the served region is
#: narrowed inside the pre-narrowing outer annulus for the D1 fixture --
#: mimics the per-``theta_c``-column admission pulling the served inner
#: edge closer to the caustic.  Strictly positive so the narrowing is real.
_NARROWING = 0.30

#: (gamma, |y|) grid for the D2 byte-equivalence sweep.  4 gammas x 3
#: magnitudes = 12 ``annulus_rho`` calls, each recomputing
#: ``caustic_geometry`` (~1 s), so ~12 s for the sweep -- well under the
#: 60 s per-test ceiling.
_D2_GAMMAS = (0.3, 0.5, 0.7, 0.9)
_D2_MAGNITUDES = (1.5, 2.5, 4.0)


def _band_gauge_scalars() -> dict:
    """Real intermediate gauge scalars for `_BAND`, positive parity.

    Reproduces exactly the quantities `surrogate_training` computes when it
    builds a positive-parity exterior region: the authoritative scalar
    reach, the per-angle minimum critical-curve radius, the band-maximum
    reach, and the derived physical exclusion radius and outer-annulus
    ``exclusion_rho``.  Cached so the whole suite pays the ~5 s geometry
    sweep once.
    """
    if _band_gauge_scalars._cache is None:
        gamma_mid = 0.5 * (_BAND[0] + _BAND[1])
        reach = _scalar_caustic_reach(gamma_mid)
        coordinate_radius_min, reach_max = _coordinate_radius_bounds(_BAND, 1)
        physical_exclusion_radius = reach_max + _ETA_MAX
        exclusion_rho = 1.0 + physical_exclusion_radius - coordinate_radius_min
        _band_gauge_scalars._cache = {
            'gamma_mid': gamma_mid,
            'reach': reach,
            'coordinate_radius_min': coordinate_radius_min,
            'reach_max': reach_max,
            'physical_exclusion_radius': physical_exclusion_radius,
            'exclusion_rho': exclusion_rho,
        }
    return _band_gauge_scalars._cache


_band_gauge_scalars._cache = None  # type: ignore[attr-defined]


#: Macro-saddle gamma band (``gamma > 1``) for the D1 saddle byte-identity
#: case.  Away from the ``gamma = 1`` parity boundary so the disconnected
#: deltoid caustic is firmly in the macro-saddle regime.
_SADDLE_BAND: tuple[float, float] = (1.5, 1.7)

#: Signed macro-saddle parity code (det < 0).  Any value ``!= 1`` selects
#: the saddle branch in both `_coordinate_radius_bounds` (scalar-reach
#: fallback) and the exclusion gauge (``physical / reach`` verbatim); the
#: census signs a macro saddle ``-2``.
_SADDLE_PARITY = -2


def _saddle_gauge_scalars() -> dict:
    """Real intermediate gauge scalars for `_SADDLE_BAND`, macro-saddle parity.

    Mirrors exactly what `surrogate_training` computes for a macro-saddle
    exterior region: the scalar reach, the band-minimum scalar reach used as
    the caustic-fixed inner radius, the band-maximum reach, the physical
    exclusion radius and the additive outer-annulus ``exclusion_rho``.  For a
    macro saddle there are NO positive-parity exterior tiles, so the served
    region is NOT narrowed: ``region_exclusion_rho == exclusion_rho`` and the
    ppGO gauge collapses to the HEAD scalar-reach expression
    ``physical_exclusion_radius / reach``.  Cached so the geometry sweep is
    paid once.
    """
    if _saddle_gauge_scalars._cache is None:
        gamma_mid = 0.5 * (_SADDLE_BAND[0] + _SADDLE_BAND[1])
        reach = _scalar_caustic_reach(gamma_mid)
        coordinate_radius_min, reach_max = _coordinate_radius_bounds(
            _SADDLE_BAND, _SADDLE_PARITY)
        physical_exclusion_radius = reach_max + _ETA_MAX
        exclusion_rho = 1.0 + physical_exclusion_radius - coordinate_radius_min
        _saddle_gauge_scalars._cache = {
            'gamma_mid': gamma_mid,
            'reach': reach,
            'coordinate_radius_min': coordinate_radius_min,
            'reach_max': reach_max,
            'physical_exclusion_radius': physical_exclusion_radius,
            'exclusion_rho': exclusion_rho,
        }
    return _saddle_gauge_scalars._cache


_saddle_gauge_scalars._cache = None  # type: ignore[attr-defined]


#: Fine rho-band edges for the synthetic ppGO map: rho_fix (~0.98) and
#: rho_head (~1.19) must fall in DIFFERENT cells so the w_cert comparison
#: discriminates.  Last edge ``inf`` mirrors the open outer annulus.
_SYNTH_RHO_EDGES = (0.0, 0.5, 0.9, 1.0, 1.1, 1.3, 1.5, 2.5, 4.0, math.inf)


def _synthetic_ppgo_map(monotone: str = 'decreasing',
                        gamma_max: float = 1.0) -> CertifiedPpgoMap:
    """A fully-certified synthetic map with a known w_cert(rho) profile.

    Every cell is `STATUS_CERTIFIED` with a huge ``rho_measured_max`` (no
    query is cut off) and a w_cert that varies ONLY with the rho-band index
    ``ri``: ``30 - 2 ri`` when ``monotone='decreasing'`` (higher near the
    caustic, the physical direction the fix relies on) or ``14 + 2 ri`` when
    ``'increasing'`` (a corrupted map used only for self-falsification).

    ``gamma_max`` (default ``1.0``) sets the single gamma band ``[0,
    gamma_max]``; pass a value ``> 1`` to cover a macro-saddle band whose
    ``gamma_mid`` exceeds one.  Both parity rows (``'positive'`` code 0.0
    and ``'saddle'`` code 1.0) are populated identically.
    """
    parity_codes = np.array([0.0, 1.0])          # positive, saddle
    gamma_edges = np.array([0.0, gamma_max])     # one gamma band [0, gamma_max]
    rho_edges = np.array(_SYNTH_RHO_EDGES)
    n_rho = rho_edges.size - 1
    ri = np.arange(n_rho, dtype=float)
    if monotone == 'decreasing':
        row = 30.0 - 2.0 * ri
    elif monotone == 'increasing':
        row = 14.0 + 2.0 * ri
    else:
        raise ValueError(f'unknown monotone spec {monotone!r}')
    w_cert = np.broadcast_to(row, (2, 1, n_rho)).astype(float).copy()
    w_ceiling = w_cert + 50.0
    rho_measured_max = np.full((2, 1, n_rho), 1.0e3)
    status = np.full((2, 1, n_rho), STATUS_CERTIFIED)
    interpolable = np.ones((2, 1, n_rho))
    return CertifiedPpgoMap.from_arrays(
        parity_codes, gamma_edges, rho_edges, w_cert, w_cert.copy(),
        w_ceiling, status, interpolable, rho_measured_max,
        {'content_hash': 'synthetic-test-map'})


class _GaugeTestCase(TestCase):
    """Base carrying the anti-vacuity comparison tally."""

    def setUp(self) -> None:
        """Reset the per-test comparison counter used by `tearDown`."""
        self.n_compared = 0

    def tearDown(self) -> None:
        """Fail a test that used the counter yet compared nothing.

        A sweep that skipped every point (all inputs filtered out, or a
        loop that never executed) would otherwise assert nothing and read
        green.  Tests that never touch ``n_compared`` leave it at 0 and are
        exempt only if they never intended to compare -- every comparison
        test in this suite increments it.
        """
        if getattr(self, '_expect_comparisons', False) and not self.n_compared:
            self.fail('the test asserted nothing: zero comparisons ran')


class AnnulusRhoByteEquivalenceTestCase(_GaugeTestCase):
    """D2: `annulus_rho` reproduces the legacy inline expression exactly."""

    def _legacy_rho(self, gamma: float, y1: float, y2: float) -> float:
        """The pre-extraction inline expression, reconstructed here.

        ``rho = |y| / reach`` where ``reach`` is element 0 (the scalar
        maximum caustic radius) of `caustic_geometry`.  Independent of the
        production converter -- it re-derives the reach from scratch.
        """
        reach = caustic_geometry(gamma, 0.0)[0]
        return math.hypot(y1, y2) / reach

    def test_matches_legacy_inline_expression_exactly(self) -> None:
        """``max |legacy - annulus_rho| == 0`` over the (gamma, |y|) grid.

        The magnitude ``|y|`` is realised as a 2-vector ``(y1, y2)`` so the
        test exercises the ``np.hypot`` reconstruction, then fed to
        `annulus_rho` as a scalar magnitude.  Exact (not ``almostEqual``)
        equality: a reach-index or normalisation slip leaves a nonzero
        residual a tolerance would swallow.
        """
        self._expect_comparisons = True
        residuals: list[tuple[float, float]] = []
        for gamma, magnitude in itertools.product(_D2_GAMMAS, _D2_MAGNITUDES):
            # Split the magnitude across both axes at 30 degrees so hypot
            # actually combines two nonzero components.
            y1 = magnitude * math.cos(math.radians(30.0))
            y2 = magnitude * math.sin(math.radians(30.0))
            legacy = self._legacy_rho(gamma, y1, y2)
            produced = annulus_rho(gamma, math.hypot(y1, y2), 0.0)
            with self.subTest(gamma=gamma, magnitude=magnitude):
                self.assertEqual(
                    produced, legacy,
                    f'annulus_rho({gamma}, {magnitude}) = {produced!r} != '
                    f'legacy {legacy!r}')
            residuals.append((gamma, legacy - produced))
            self.n_compared += 1
        self._save_residual_plot(residuals)

    def test_is_pure_scaling_by_reciprocal_reach(self) -> None:
        """``annulus_rho`` is linear in ``|y|``: doubling ``|y|`` doubles rho.

        A cheap structural cross-check that the converter carries no hidden
        offset or nonlinearity -- ``rho`` is exactly ``|y| / reach``.
        """
        self._expect_comparisons = True
        for gamma in _D2_GAMMAS:
            reach = caustic_geometry(gamma, 0.0)[0]
            for magnitude in _D2_MAGNITUDES:
                with self.subTest(gamma=gamma, magnitude=magnitude):
                    self.assertEqual(annulus_rho(gamma, magnitude, 0.0),
                                     magnitude / reach)
                    self.assertEqual(annulus_rho(gamma, 0.0, 0.0), 0.0)
                self.n_compared += 1

    def _save_residual_plot(self, residuals: list[tuple[float, float]]) -> None:
        """Scatter of (legacy - annulus_rho) vs gamma -- the D2 diagnostic."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return
        _OUTPUT_DIR.mkdir(exist_ok=True)
        gammas = [g for g, _ in residuals]
        diffs = [d for _, d in residuals]
        fig, ax = plt.subplots()
        ax.scatter(gammas, diffs, s=40)
        ax.axhline(0.0, color='k', lw=0.5)
        ax.set_xlabel('gamma')
        ax.set_ylabel('legacy - annulus_rho')
        ax.set_title('D2 byte-equivalence residual (must be identically 0)')
        fig.savefig(_OUTPUT_DIR / 'annulus_rho_byte_equivalence_residual.png',
                    dpi=90)
        plt.close(fig)


class AnnulusRhoGuardTestCase(_GaugeTestCase):
    """D2: `annulus_rho` fails loudly on invalid inputs."""

    def test_negative_magnitude_raises_naming_the_argument(self) -> None:
        """A negative ``y_magnitude`` raises ``ValueError`` naming it."""
        with self.assertRaises(ValueError) as caught:
            annulus_rho(0.5, -1.0, 0.0)
        self.assertIn('y_magnitude', str(caught.exception))

    def test_nonpositive_reach_raises_naming_reach(self) -> None:
        """A non-positive caustic reach raises ``ValueError`` naming it.

        Real ``caustic_geometry`` never RETURNS a non-positive reach -- it
        raises `LensDomainError` first -- so this guard is reachable only
        through a degenerate reach.  Patch the module-level
        ``caustic_geometry`` name that `annulus_rho` resolves so a reach of
        ``0.0`` reaches the guard; the message must name the reach and the
        offending gamma.
        """
        stub = mock.Mock(return_value=(0.0, np.array([1.0, 0.0])))
        with mock.patch.object(ppgo_map, 'caustic_geometry', stub):
            with self.assertRaises(ValueError) as caught:
                annulus_rho(0.5, 2.0, 0.0)
        message = str(caught.exception)
        self.assertIn('reach', message)
        self.assertIn('0.5', message)

    def test_zero_magnitude_is_allowed(self) -> None:
        """``|y| = 0`` (source at the caustic centre) is valid, rho = 0."""
        self.assertEqual(annulus_rho(0.5, 0.0, 0.0), 0.0)


class PpgoExclusionMonotonicConservatismTestCase(_GaugeTestCase):
    """D1: the fix moves the ppGO exclusion read point conservatively.

    Reproduces both gauges from the real band scalars and checks, against a
    synthetic monotone map, that the narrowed-region gauge reads a cell that
    is never easier (a higher-or-equal dispatch floor) than the HEAD
    outer-annulus gauge.
    """

    #: Parity string the ppGO map keys positive-parity cells under.
    PARITY = 'positive'

    def _gauges(self, narrowing: float) -> dict:
        """The HEAD and fixed ppGO exclusion rho for a given narrowing.

        ``rho_head`` is the pre-fix outer-annulus gauge
        (``physical_exclusion_radius / reach``); ``rho_fix`` is
        `annulus_rho` of the source magnitude recovered by inverting the
        additive exterior gauge on the NARROWED served region.  Both share
        the same reach, so their ordering follows the source magnitudes.
        """
        scalars = _band_gauge_scalars()
        gamma_mid = scalars['gamma_mid']
        physical = scalars['physical_exclusion_radius']
        exclusion_rho = scalars['exclusion_rho']
        coordinate_radius_min = scalars['coordinate_radius_min']
        # HEAD gauge: outer-annulus scalar reach.
        rho_head = physical / scalars['reach']
        # Fixed gauge: narrowed served region -> annulus_rho.
        region_exclusion_rho = exclusion_rho - narrowing
        y_fix = region_exclusion_rho - 1.0 + coordinate_radius_min
        rho_fix = annulus_rho(gamma_mid, y_fix, 0.0)
        return {
            'gamma_mid': gamma_mid, 'physical': physical,
            'exclusion_rho': exclusion_rho,
            'region_exclusion_rho': region_exclusion_rho,
            'y_fix': y_fix, 'rho_head': rho_head, 'rho_fix': rho_fix}

    def test_head_gauge_equals_annulus_rho_of_full_physical_radius(self) -> None:
        """The HEAD outer-annulus gauge IS `annulus_rho` of the full radius.

        ``physical_exclusion_radius / reach`` is bit-identical to
        ``annulus_rho(gamma_mid, physical_exclusion_radius, 0)`` because
        both divide by the same deterministic reach.  This pins the claim
        that the two gauges differ ONLY in the source magnitude they feed.
        """
        self._expect_comparisons = True
        self.n_compared += 1
        gauges = self._gauges(_NARROWING)
        self.assertEqual(
            gauges['rho_head'],
            annulus_rho(gauges['gamma_mid'], gauges['physical'], 0.0))

    def test_narrowed_region_is_strictly_inside_outer_annulus(self) -> None:
        """Premise: the served region is strictly inside the outer annulus.

        ``region_exclusion_rho < exclusion_rho`` and the fed magnitude is
        strictly below the HEAD physical radius -- otherwise the invariant
        below would be vacuously true.
        """
        self._expect_comparisons = True
        self.n_compared += 1
        gauges = self._gauges(_NARROWING)
        self.assertLess(gauges['region_exclusion_rho'], gauges['exclusion_rho'])
        self.assertLess(gauges['y_fix'], gauges['physical'])

    def test_fixed_gauge_reads_no_larger_rho_than_head(self) -> None:
        """Geometric core: ``rho_fix <= rho_head``, strict under narrowing.

        Sign-unambiguous and independent of any map: the fix can only pull
        the ppGO read point inward (toward the caustic), never outward.
        With zero narrowing the two coincide exactly.
        """
        self._expect_comparisons = True
        self.n_compared += 1
        strict = self._gauges(_NARROWING)
        # Real narrowing (0.30) puts rho_fix ~0.98 well below rho_head ~1.19,
        # so the strict inequality is robust to floating-point noise.
        self.assertLess(strict['rho_fix'], strict['rho_head'])
        # With zero narrowing the two coincide up to the sub-ULP rounding of
        # the additive ``exclusion_rho`` round-trip ((1 + phys - crmin) - 1 +
        # crmin != phys to the last bit); assert closeness, not bit-equality.
        none = self._gauges(0.0)
        self.assertAlmostEqual(none['rho_fix'], none['rho_head'], places=12)

    def test_fixed_gauge_reads_a_not_easier_cell(self) -> None:
        """The narrowed gauge reads a higher-or-equal w_cert / w_trust cell.

        On a map whose w_cert decreases with rho (higher near the caustic,
        the production-documented direction), the inward move to
        ``rho_fix < rho_head`` lands in a STRICTLY harder cell:
        ``w_cert(fix) > w_cert(head)`` and, since ``w_trust`` is monotone in
        ``w_cert``, ``w_trust(fix) > w_trust(head)``.  This is the "never
        easier" property (encoded with its measured, code-consistent sign).
        """
        self._expect_comparisons = True
        self.n_compared += 1
        ppgo = _synthetic_ppgo_map('decreasing')
        gauges = self._gauges(_NARROWING)
        w_cert_fix = ppgo.w_cert(self.PARITY, gauges['gamma_mid'],
                                 gauges['rho_fix'])
        w_cert_head = ppgo.w_cert(self.PARITY, gauges['gamma_mid'],
                                  gauges['rho_head'])
        # Both cells must actually be certified, else the comparison is void.
        self.assertIsNot(w_cert_fix, UNKNOWN)
        self.assertIsNot(w_cert_head, UNKNOWN)
        self.assertGreater(w_cert_fix, w_cert_head)

        w_trust_fix = _stratum_ppgo_boundary(
            1, gauges['gamma_mid'], gauges['rho_fix'], ppgo)
        w_trust_head = _stratum_ppgo_boundary(
            1, gauges['gamma_mid'], gauges['rho_head'], ppgo)
        self.assertIsNotNone(w_trust_fix)
        self.assertIsNotNone(w_trust_head)
        self.assertGreater(w_trust_fix, w_trust_head)
        self._save_w_cert_plot(ppgo, gauges)

    def _save_w_cert_plot(self, ppgo: CertifiedPpgoMap, gauges: dict) -> None:
        """Plot w_cert(rho) with rho_head and rho_fix marked -- D1 diagnostic."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return
        _OUTPUT_DIR.mkdir(exist_ok=True)
        rhos = np.linspace(0.05, 2.4, 200)
        certs = [ppgo.w_cert(self.PARITY, gauges['gamma_mid'], float(rho))
                 for rho in rhos]
        certs = [np.nan if c is UNKNOWN else c for c in certs]
        fig, ax = plt.subplots()
        ax.step(rhos, certs, where='mid')
        ax.axvline(gauges['rho_head'], color='tab:red', ls='--',
                   label=f"rho_head={gauges['rho_head']:.3f} (HEAD, easier)")
        ax.axvline(gauges['rho_fix'], color='tab:green', ls='--',
                   label=f"rho_fix={gauges['rho_fix']:.3f} (fixed, harder)")
        ax.set_xlabel('ppGO annulus rho')
        ax.set_ylabel('w_cert')
        ax.set_title('D1: fixed gauge reads inward -> higher w_cert cell')
        ax.legend()
        fig.savefig(_OUTPUT_DIR / 'ppgo_exclusion_w_cert_read_points.png',
                    dpi=90)
        plt.close(fig)


class PpgoOrderingReachableRedTestCase(_GaugeTestCase):
    """D1 defect-1 reachable-red guard: the ordering bug reads an easier cell.

    The defect was an ORDERING bug: the ppGO annulus coordinate was derived
    from the PRE-narrowing outer annulus (``exclusion_rho``) instead of the
    NARROWED served region (``region_exclusion_rho``) that the positive-parity
    per-``theta_c``-column admission actually covers.  Both orderings feed the
    SAME authoritative converter `annulus_rho` and divide by the SAME reach --
    they differ ONLY in the source magnitude, hence in the read-point ``rho``.

    This case reproduces both orderings from the real band scalars and pins
    the reachable-red property demanded by the Architect: the PRODUCTION
    invariant -- "the ppGO cell the region drops/caps against must be NO
    EASIER than the cell the region's own served inner edge deserves" --
    PASSES under the fixed ordering (it reads exactly at the served inner
    edge) and FAILS (raises) under the buggy ordering (it reads the farther-
    out, lower-``w_cert`` outer cell).  A test that could not go red on the
    buggy ordering would be vacuous; this one demonstrably does.

    The diagnostic is the read-point separation: ``rho_buggy`` (outer, the
    full physical exclusion radius) is strictly LARGER than ``rho_fixed``
    (the narrowed inner edge), so the two orderings genuinely read different
    cells.
    """

    PARITY = 'positive'

    def _served_inner_magnitude(self) -> float:
        """Physical ``|y|`` of the region's TRUE served inner edge.

        Inverts the additive exterior gauge on the narrowed region:
        ``|y| = region_exclusion_rho - 1 + coordinate_radius_min`` with
        ``region_exclusion_rho = exclusion_rho - _NARROWING``.
        """
        scalars = _band_gauge_scalars()
        region_exclusion_rho = scalars['exclusion_rho'] - _NARROWING
        return region_exclusion_rho - 1.0 + scalars['coordinate_radius_min']

    def _outer_magnitude(self) -> float:
        """Physical ``|y|`` of the PRE-narrowing outer annulus (the bug).

        ``|y| = exclusion_rho - 1 + coordinate_radius_min`` which is exactly
        the full ``physical_exclusion_radius`` -- the magnitude the buggy
        ordering consumes because it reads ``exclusion_rho`` before the
        narrowing to ``region_exclusion_rho`` is applied.
        """
        scalars = _band_gauge_scalars()
        return scalars['exclusion_rho'] - 1.0 + scalars['coordinate_radius_min']

    def _read_rho(self, ordering: str) -> float:
        """ppGO annulus read-point for the two orderings (`annulus_rho`)."""
        scalars = _band_gauge_scalars()
        if ordering == 'fixed':
            magnitude = self._served_inner_magnitude()
        elif ordering == 'buggy':
            magnitude = self._outer_magnitude()
        else:
            raise ValueError(f'unknown ordering {ordering!r}')
        return annulus_rho(scalars['gamma_mid'], magnitude, 0.0)

    def _assert_not_easier(self, ordering: str,
                           ppgo: CertifiedPpgoMap) -> tuple[float, float]:
        """Production invariant: read cell no easier than the served edge.

        Reads ``w_cert`` at the ordering's read-point and at the region's
        TRUE served inner edge (``rho_fixed``) and asserts the former is not
        smaller.  Returns ``(w_read, w_region)`` for diagnostics.  Raises
        ``AssertionError`` when the read cell is easier -- the reachable-red
        signal.
        """
        gamma_mid = _band_gauge_scalars()['gamma_mid']
        rho_region = self._read_rho('fixed')
        rho_read = self._read_rho(ordering)
        w_read = ppgo.w_cert(self.PARITY, gamma_mid, rho_read)
        w_region = ppgo.w_cert(self.PARITY, gamma_mid, rho_region)
        self.assertIsNot(w_read, UNKNOWN)
        self.assertIsNot(w_region, UNKNOWN)
        self.assertGreaterEqual(
            w_read, w_region,
            f'{ordering} ordering read an EASIER cell: w_cert(read)={w_read} '
            f'< w_cert(served inner edge)={w_region}')
        return w_read, w_region

    def test_buggy_and_fixed_read_points_differ(self) -> None:
        """Diagnostic: the buggy outer read-point exceeds the fixed inner one.

        Under the real ``_NARROWING`` the served region is pulled strictly
        inside the outer annulus, so ``rho_buggy > rho_fixed`` -- the two
        orderings cannot land in the same cell, which is what makes the
        reachable-red below non-trivial.
        """
        self._expect_comparisons = True
        self.n_compared += 1
        rho_fixed = self._read_rho('fixed')
        rho_buggy = self._read_rho('buggy')
        self.assertGreater(rho_buggy, rho_fixed)
        self.assertNotEqual(rho_buggy, rho_fixed)

    def test_fixed_ordering_reads_a_not_easier_cell(self) -> None:
        """The fixed ordering satisfies the production invariant (green)."""
        self._expect_comparisons = True
        self.n_compared += 1
        ppgo = _synthetic_ppgo_map('decreasing')
        w_read, w_region = self._assert_not_easier('fixed', ppgo)
        # It reads at the served inner edge itself, so the cells coincide.
        self.assertEqual(w_read, w_region)

    def test_buggy_ordering_is_reachable_red(self) -> None:
        """The buggy ordering VIOLATES the invariant -- the test goes red.

        On a map whose ``w_cert`` decreases with ``rho`` (the physical
        direction), the buggy outer read-point lands in a strictly easier
        (lower-``w_cert``) cell than the served inner edge deserves, so the
        ``_assert_not_easier`` invariant raises.  Catching that raise proves
        the guard actually exercises the defect.
        """
        self._expect_comparisons = True
        self.n_compared += 1
        ppgo = _synthetic_ppgo_map('decreasing')
        with self.assertRaises(AssertionError):
            self._assert_not_easier('buggy', ppgo)
        # And, explicitly, the buggy read is the strictly easier one.
        gamma_mid = _band_gauge_scalars()['gamma_mid']
        w_buggy = ppgo.w_cert(self.PARITY, gamma_mid, self._read_rho('buggy'))
        w_region = ppgo.w_cert(self.PARITY, gamma_mid, self._read_rho('fixed'))
        self.assertLess(w_buggy, w_region)
        self._save_ordering_plot(ppgo)

    def _save_ordering_plot(self, ppgo: CertifiedPpgoMap) -> None:
        """Plot w_cert(rho) marking the buggy (outer) and fixed (inner) reads."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return
        _OUTPUT_DIR.mkdir(exist_ok=True)
        gamma_mid = _band_gauge_scalars()['gamma_mid']
        rhos = np.linspace(0.05, 2.4, 200)
        certs = [ppgo.w_cert(self.PARITY, gamma_mid, float(rho))
                 for rho in rhos]
        certs = [np.nan if c is UNKNOWN else c for c in certs]
        fig, ax = plt.subplots()
        ax.step(rhos, certs, where='mid')
        ax.axvline(self._read_rho('buggy'), color='tab:red', ls='--',
                   label=f"rho_buggy={self._read_rho('buggy'):.3f} "
                         f"(outer, easier -- the bug)")
        ax.axvline(self._read_rho('fixed'), color='tab:green', ls='--',
                   label=f"rho_fixed={self._read_rho('fixed'):.3f} "
                         f"(served inner edge)")
        ax.set_xlabel('ppGO annulus rho')
        ax.set_ylabel('w_cert')
        ax.set_title('D1 defect-1: buggy ordering reads the easier outer cell')
        ax.legend()
        fig.savefig(_OUTPUT_DIR / 'ppgo_ordering_reachable_red.png', dpi=90)
        plt.close(fig)


class SaddleBranchByteIdentityTestCase(_GaugeTestCase):
    """D1: the macro-saddle branch reads the HEAD value verbatim.

    The fix touched ONLY the positive-parity branch.  For a macro saddle
    (``parity != 1``, ``gamma > 1``) there are no positive-parity exterior
    tiles, so the served region is not narrowed (``region_exclusion_rho ==
    exclusion_rho``) and the ppGO gauge stays on the HEAD scalar-reach
    expression ``ppgo_exclusion_rho = physical_exclusion_radius / reach``.

    The independent HEAD oracle is `annulus_rho` evaluated on the FULL
    physical exclusion radius: `annulus_rho` divides by
    ``caustic_geometry(gamma, 0)[0]``, which is bit-identical to the scalar
    reach `_scalar_caustic_reach` the branch divides by (verified in the
    suite's probe), so equality is EXACT -- a reach-index slip or a stray
    narrowing would leave a nonzero residual.  The teeth: the same band's
    positive-parity narrowing formula, applied with a real narrowing, would
    read a strictly SMALLER ``rho`` and a different cell.
    """

    PARITY = 'saddle'

    def _saddle_branch_rho(self) -> float:
        """The fixed-code saddle branch value ``physical / reach``, verbatim."""
        scalars = _saddle_gauge_scalars()
        return scalars['physical_exclusion_radius'] / scalars['reach']

    def _head_oracle_rho(self) -> float:
        """HEAD gauge via the authoritative converter on the full radius."""
        scalars = _saddle_gauge_scalars()
        return annulus_rho(scalars['gamma_mid'],
                           scalars['physical_exclusion_radius'], 0.0)

    def _narrowed_foil_rho(self, narrowing: float) -> float:
        """What a saddle would read IF it wrongly narrowed like positive parity.

        Feeds the additive-inverted magnitude of a narrowed region through
        `annulus_rho`; used only to witness that the saddle branch does NOT
        do this (its read-point is strictly larger).
        """
        scalars = _saddle_gauge_scalars()
        region = scalars['exclusion_rho'] - narrowing
        magnitude = region - 1.0 + scalars['coordinate_radius_min']
        return annulus_rho(scalars['gamma_mid'], magnitude, 0.0)

    def test_saddle_branch_is_byte_identical_to_head(self) -> None:
        """``physical / reach`` equals `annulus_rho` of the full radius EXACTLY.

        Exact equality (not ``almostEqual``): the saddle branch must read the
        OLD value to the last bit.
        """
        self._expect_comparisons = True
        self.n_compared += 1
        self.assertEqual(self._saddle_branch_rho(), self._head_oracle_rho())

    def test_saddle_branch_does_not_narrow(self) -> None:
        """The saddle read-point is the full outer radius, not a narrowed edge.

        A ``0.30`` narrowing (the positive-parity fixture's amount) applied
        to this saddle band would pull the read-point strictly inward; the
        saddle branch does no such thing, so its ``rho`` is strictly larger.
        """
        self._expect_comparisons = True
        self.n_compared += 1
        self.assertGreater(self._saddle_branch_rho(),
                           self._narrowed_foil_rho(_NARROWING))

    def test_saddle_cell_fields_match_head_bit_for_bit(self) -> None:
        """Diagnostic: every ppGO cell field at the saddle rho == at HEAD rho.

        Reads ``w_cert``, ``w_trust``, ``w_ceiling`` and the cell status at
        the fixed saddle read-point and at the independent HEAD read-point on
        a macro-saddle-covering synthetic map; the per-field diff is exactly
        zero because the two read-points are the same float.
        """
        self._expect_comparisons = True
        ppgo = _synthetic_ppgo_map('decreasing', gamma_max=2.0)
        gamma_mid = _saddle_gauge_scalars()['gamma_mid']
        rho_fix = self._saddle_branch_rho()
        rho_head = self._head_oracle_rho()
        for field in ('w_cert', 'w_trust', 'w_ceiling'):
            with self.subTest(field=field):
                query = getattr(ppgo, field)
                fixed_value = query(self.PARITY, gamma_mid, rho_fix)
                head_value = query(self.PARITY, gamma_mid, rho_head)
                self.assertIsNot(fixed_value, UNKNOWN)
                self.assertEqual(fixed_value, head_value)
            self.n_compared += 1
        self.assertEqual(
            ppgo.cell_status(self.PARITY, gamma_mid, rho_fix),
            ppgo.cell_status(self.PARITY, gamma_mid, rho_head))
        self.assertEqual(
            ppgo.cell_status(self.PARITY, gamma_mid, rho_fix), 'certified')

    def test_saddle_boundary_matches_head(self) -> None:
        """`_stratum_ppgo_boundary` for the saddle reads the HEAD floor.

        The dispatch floor computed at the saddle branch rho equals the one
        at the HEAD oracle rho (same cell), and both are finite (certified).
        """
        self._expect_comparisons = True
        self.n_compared += 1
        ppgo = _synthetic_ppgo_map('decreasing', gamma_max=2.0)
        gamma_mid = _saddle_gauge_scalars()['gamma_mid']
        floor_fixed = _stratum_ppgo_boundary(
            _SADDLE_PARITY, gamma_mid, self._saddle_branch_rho(), ppgo)
        floor_head = _stratum_ppgo_boundary(
            _SADDLE_PARITY, gamma_mid, self._head_oracle_rho(), ppgo)
        self.assertIsNotNone(floor_fixed)
        self.assertEqual(floor_fixed, floor_head)


class SelfFalsificationTestCase(_GaugeTestCase):
    """Prove the suite can go red.

    A green suite is worth only its ability to fail.  These tests reproduce
    the fault regimes and corrupted oracles/maps and assert that the D1/D2
    checks above actually flag them.
    """

    PARITY = 'positive'

    def test_wrong_reach_index_oracle_is_detected(self) -> None:
        """A reach-index slip yields a nonzero residual the D2 test rejects.

        The exact-equality bar is only meaningful if a wrong reach makes it
        fail.  An oracle that divides by ``reach * (1 + 1e-9)`` (a stand-in
        for a normalisation / reach-index slip) must differ from
        ``annulus_rho`` -- otherwise byte-equality would pass vacuously.
        """
        self._expect_comparisons = True
        differ = 0
        for gamma, magnitude in itertools.product(_D2_GAMMAS, _D2_MAGNITUDES):
            reach = caustic_geometry(gamma, 0.0)[0]
            tainted = magnitude / (reach * (1.0 + 1.0e-9))
            if annulus_rho(gamma, magnitude, 0.0) != tainted:
                differ += 1
            self.n_compared += 1
        self.assertEqual(differ, len(_D2_GAMMAS) * len(_D2_MAGNITUDES),
                         'a corrupted reach slipped past exact equality')

    def test_head_gauge_would_read_an_easier_or_equal_cell(self) -> None:
        """Regressing to the HEAD gauge loses the strict "harder" property.

        If the code reverted to reading the ppGO cell at ``rho_head`` (the
        pre-fix outer-annulus gauge), the "strictly harder" assertion would
        fail: ``w_cert(head)`` is strictly LESS than ``w_cert(fix)`` under
        narrowing, so the HEAD read is easier.  Asserting that the strict
        comparison flips proves the D1 test has teeth against a regression.
        """
        self._expect_comparisons = True
        ppgo = _synthetic_ppgo_map('decreasing')
        gauges = PpgoExclusionMonotonicConservatismTestCase._gauges(
            self, _NARROWING)
        w_cert_head = ppgo.w_cert(self.PARITY, gauges['gamma_mid'],
                                  gauges['rho_head'])
        w_cert_fix = ppgo.w_cert(self.PARITY, gauges['gamma_mid'],
                                 gauges['rho_fix'])
        # The HEAD gauge reads the strictly easier (lower-w_cert) cell.
        self.assertLess(w_cert_head, w_cert_fix)
        # And a test that (wrongly) demanded rho_head be the harder cell
        # would raise -- the falsification.
        with self.assertRaises(AssertionError):
            self.assertGreater(w_cert_head, w_cert_fix)
        self.n_compared += 1

    def test_no_narrowing_removes_the_strict_gap(self) -> None:
        """With zero narrowing both gauges read the SAME cell (no gap).

        Confirms the strict "harder" property is DRIVEN by the narrowing,
        not spuriously always true: identical rho -> identical w_cert, so a
        ``assertGreater`` would fail.
        """
        self._expect_comparisons = True
        ppgo = _synthetic_ppgo_map('decreasing')
        gauges = PpgoExclusionMonotonicConservatismTestCase._gauges(self, 0.0)
        w_cert_fix = ppgo.w_cert(self.PARITY, gauges['gamma_mid'],
                                 gauges['rho_fix'])
        w_cert_head = ppgo.w_cert(self.PARITY, gauges['gamma_mid'],
                                  gauges['rho_head'])
        self.assertEqual(w_cert_fix, w_cert_head)
        with self.assertRaises(AssertionError):
            self.assertGreater(w_cert_fix, w_cert_head)
        self.n_compared += 1

    def test_increasing_map_flips_the_conservatism_comparison(self) -> None:
        """A map whose w_cert rises with rho flips the read-cell ordering.

        Demonstrates the comparison is sensitive to the map's values (not a
        tautology): on an increasing map the inner ``rho_fix`` cell has the
        LOWER w_cert, so ``w_cert(fix) > w_cert(head)`` is false.
        """
        self._expect_comparisons = True
        ppgo = _synthetic_ppgo_map('increasing')
        gauges = PpgoExclusionMonotonicConservatismTestCase._gauges(
            self, _NARROWING)
        w_cert_fix = ppgo.w_cert(self.PARITY, gauges['gamma_mid'],
                                 gauges['rho_fix'])
        w_cert_head = ppgo.w_cert(self.PARITY, gauges['gamma_mid'],
                                  gauges['rho_head'])
        self.assertLess(w_cert_fix, w_cert_head)
        self.n_compared += 1

    def test_uncertified_cell_reads_unknown(self) -> None:
        """An invalid cell yields ``UNKNOWN`` / ``None`` -- no phantom floor.

        Guards the "both cells certified" premise of the D1 test: a map
        whose cells are `STATUS_INVALID` must refuse, so `w_cert` returns
        `UNKNOWN` and `_stratum_ppgo_boundary` returns ``None``.
        """
        self._expect_comparisons = True
        ppgo = _synthetic_ppgo_map('decreasing')
        ppgo.cell_status_grid[...] = STATUS_INVALID
        gauges = PpgoExclusionMonotonicConservatismTestCase._gauges(
            self, _NARROWING)
        self.assertIs(
            ppgo.w_cert(self.PARITY, gauges['gamma_mid'], gauges['rho_fix']),
            UNKNOWN)
        self.assertIsNone(_stratum_ppgo_boundary(
            1, gauges['gamma_mid'], gauges['rho_fix'], ppgo))
        self.n_compared += 1

    def test_narrowing_a_saddle_would_move_the_cell(self) -> None:
        """A saddle that wrongly narrowed would read a DIFFERENT cell (teeth).

        Confirms the saddle byte-identity / cell-field diff test is not
        vacuous: had the saddle branch (wrongly) applied the positive-parity
        narrowing, its read-point would fall in a different ``rho`` cell with
        a different ``w_cert``, so the exact cell-field equality would fail.
        The saddle branch instead reads the full outer radius, keeping the
        HEAD cell.
        """
        self._expect_comparisons = True
        ppgo = _synthetic_ppgo_map('decreasing', gamma_max=2.0)
        scalars = _saddle_gauge_scalars()
        gamma_mid = scalars['gamma_mid']
        head_rho = scalars['physical_exclusion_radius'] / scalars['reach']
        region = scalars['exclusion_rho'] - _NARROWING
        narrowed_rho = annulus_rho(
            gamma_mid, region - 1.0 + scalars['coordinate_radius_min'], 0.0)
        w_head = ppgo.w_cert('saddle', gamma_mid, head_rho)
        w_narrowed = ppgo.w_cert('saddle', gamma_mid, narrowed_rho)
        self.assertLess(narrowed_rho, head_rho)
        self.assertIsNot(w_head, UNKNOWN)
        self.assertIsNot(w_narrowed, UNKNOWN)
        self.assertNotEqual(w_head, w_narrowed)
        self.n_compared += 1


if __name__ == '__main__':
    main()
