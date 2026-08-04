"""
Tests for the authoritative caustic-relative converter `ppgo_map.caustic_rho`
and the monotonic-conservatism invariant of the ppGO exclusion gauge it
feeds (`surrogate_training` WP1, defects D1 + D2).

Two things are certified here.

D2 -- the extraction changed no numbers.  `caustic_rho(gamma, |y|, kappa)`
was lifted verbatim out of an inline expression that read the SCALAR
caustic reach (element 0 of `caustic_geometry`) and divided the physical
source magnitude by it.  `CausticRhoByteEquivalenceTestCase` reconstructs
that legacy expression INDEPENDENTLY inside the test -- ``np.hypot(y1, y2)
/ caustic_geometry(gamma, 0.0)[0]`` -- and demands EXACT equality
(``max |diff| == 0.0``), not closeness: a reach-index slip (element 1
instead of 0) or a stray normalisation would leave a nonzero residual that
a tolerance would hide.  Because `caustic_geometry` is deterministic, the
two independent calls return a bit-identical reach, so exact equality is
the right bar.  `CausticRhoGuardTestCase` pins the two input guards.

D1 -- the ppGO exclusion read point moved conservatively.  The stale
gauge derived the caustic-relative coordinate from the PRE-narrowing outer
rho-band (``rho = physical_exclusion_radius / reach``); the fix derives it
from the NARROWED served region via `caustic_rho`, feeding a source
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
(deriving the ppGO coordinate from the pre-narrowing outer rho-band instead
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
    caustic_rho, caustic_geometry)
from cogwheel.lensing import ppgo_map
from cogwheel.lensing.chang_refsdal import geometry
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
#: narrowed inside the pre-narrowing outer rho-band for the D1 fixture --
#: mimics the per-``theta_c``-column admission pulling the served inner
#: edge closer to the caustic.  Strictly positive so the narrowing is real.
_NARROWING = 0.30

#: (gamma, |y|) grid for the D2 byte-equivalence sweep.  4 gammas x 3
#: magnitudes = 12 ``caustic_rho`` calls, each recomputing
#: ``caustic_geometry`` (~1 s), so ~12 s for the sweep -- well under the
#: 60 s per-test ceiling.
_D2_GAMMAS = (0.3, 0.5, 0.7, 0.9)
_D2_MAGNITUDES = (1.5, 2.5, 4.0)


def _band_gauge_scalars() -> dict:
    """Real intermediate gauge scalars for `_BAND`, positive parity.

    Reproduces exactly the quantities `surrogate_training` computes when it
    builds a positive-parity exterior region: the authoritative scalar
    reach, the per-angle minimum critical-curve radius, the band-maximum
    reach, and the derived physical exclusion radius and outer-rho-band
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
    exclusion radius and the additive outer-rho-band ``exclusion_rho``.  For a
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
#: discriminates.  Last edge ``inf`` mirrors the open outer rho-band.
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


class CausticRhoByteEquivalenceTestCase(_GaugeTestCase):
    """D2: `caustic_rho` reproduces the legacy inline expression exactly."""

    def _legacy_rho(self, gamma: float, y1: float, y2: float) -> float:
        """The pre-extraction inline expression, reconstructed here.

        ``rho = |y| / reach`` where ``reach`` is element 0 (the scalar
        maximum caustic radius) of `caustic_geometry`.  Independent of the
        production converter -- it re-derives the reach from scratch.
        """
        reach = caustic_geometry(gamma, 0.0)[0]
        return math.hypot(y1, y2) / reach

    def test_matches_legacy_inline_expression_exactly(self) -> None:
        """``max |legacy - caustic_rho| == 0`` over the (gamma, |y|) grid.

        The magnitude ``|y|`` is realised as a 2-vector ``(y1, y2)`` so the
        test exercises the ``np.hypot`` reconstruction, then fed to
        `caustic_rho` as a scalar magnitude.  Exact (not ``almostEqual``)
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
            produced = caustic_rho(gamma, math.hypot(y1, y2), 0.0)
            with self.subTest(gamma=gamma, magnitude=magnitude):
                self.assertEqual(
                    produced, legacy,
                    f'caustic_rho({gamma}, {magnitude}) = {produced!r} != '
                    f'legacy {legacy!r}')
            residuals.append((gamma, legacy - produced))
            self.n_compared += 1
        self._save_residual_plot(residuals)

    def test_is_pure_scaling_by_reciprocal_reach(self) -> None:
        """``caustic_rho`` is linear in ``|y|``: doubling ``|y|`` doubles rho.

        A cheap structural cross-check that the converter carries no hidden
        offset or nonlinearity -- ``rho`` is exactly ``|y| / reach``.
        """
        self._expect_comparisons = True
        for gamma in _D2_GAMMAS:
            reach = caustic_geometry(gamma, 0.0)[0]
            for magnitude in _D2_MAGNITUDES:
                with self.subTest(gamma=gamma, magnitude=magnitude):
                    self.assertEqual(caustic_rho(gamma, magnitude, 0.0),
                                     magnitude / reach)
                    self.assertEqual(caustic_rho(gamma, 0.0, 0.0), 0.0)
                self.n_compared += 1

    def _save_residual_plot(self, residuals: list[tuple[float, float]]) -> None:
        """Scatter of (legacy - caustic_rho) vs gamma -- the D2 diagnostic."""
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
        ax.set_ylabel('legacy - caustic_rho')
        ax.set_title('D2 byte-equivalence residual (must be identically 0)')
        fig.savefig(_OUTPUT_DIR / 'caustic_rho_byte_equivalence_residual.png',
                    dpi=90)
        plt.close(fig)


class CausticRhoGuardTestCase(_GaugeTestCase):
    """D2: `caustic_rho` fails loudly on invalid inputs."""

    def test_negative_magnitude_raises_naming_the_argument(self) -> None:
        """A negative ``y_magnitude`` raises ``ValueError`` naming it."""
        with self.assertRaises(ValueError) as caught:
            caustic_rho(0.5, -1.0, 0.0)
        self.assertIn('y_magnitude', str(caught.exception))

    def test_nonpositive_reach_raises_naming_reach(self) -> None:
        """A non-positive caustic reach raises ``ValueError`` naming it.

        Real ``caustic_geometry`` never RETURNS a non-positive reach -- it
        raises `LensDomainError` first -- so this guard is reachable only
        through a degenerate reach.  Patch the module-level
        ``caustic_geometry`` name that `caustic_rho` resolves so a reach of
        ``0.0`` reaches the guard; the message must name the reach and the
        offending gamma.
        """
        stub = mock.Mock(return_value=(0.0, np.array([1.0, 0.0])))
        with mock.patch.object(ppgo_map, 'caustic_geometry', stub):
            with self.assertRaises(ValueError) as caught:
                caustic_rho(0.5, 2.0, 0.0)
        message = str(caught.exception)
        self.assertIn('reach', message)
        self.assertIn('0.5', message)

    def test_zero_magnitude_is_allowed(self) -> None:
        """``|y| = 0`` (source at the caustic centre) is valid, rho = 0."""
        self.assertEqual(caustic_rho(0.5, 0.0, 0.0), 0.0)


class PpgoExclusionMonotonicConservatismTestCase(_GaugeTestCase):
    """D1: the fix moves the ppGO exclusion read point conservatively.

    Reproduces both gauges from the real band scalars and checks, against a
    synthetic monotone map, that the narrowed-region gauge reads a cell that
    is never easier (a higher-or-equal dispatch floor) than the HEAD
    outer-rho-band gauge.
    """

    #: Parity string the ppGO map keys positive-parity cells under.
    PARITY = 'positive'

    def _gauges(self, narrowing: float) -> dict:
        """The HEAD and fixed ppGO exclusion rho for a given narrowing.

        ``rho_head`` is the pre-fix outer-rho-band gauge
        (``physical_exclusion_radius / reach``); ``rho_fix`` is
        `caustic_rho` of the source magnitude recovered by inverting the
        additive exterior gauge on the NARROWED served region.  Both share
        the same reach, so their ordering follows the source magnitudes.
        """
        scalars = _band_gauge_scalars()
        gamma_mid = scalars['gamma_mid']
        physical = scalars['physical_exclusion_radius']
        exclusion_rho = scalars['exclusion_rho']
        coordinate_radius_min = scalars['coordinate_radius_min']
        # HEAD gauge: outer-rho-band scalar reach.
        rho_head = physical / scalars['reach']
        # Fixed gauge: narrowed served region -> caustic_rho.
        region_exclusion_rho = exclusion_rho - narrowing
        y_fix = region_exclusion_rho - 1.0 + coordinate_radius_min
        rho_fix = caustic_rho(gamma_mid, y_fix, 0.0)
        return {
            'gamma_mid': gamma_mid, 'physical': physical,
            'exclusion_rho': exclusion_rho,
            'region_exclusion_rho': region_exclusion_rho,
            'y_fix': y_fix, 'rho_head': rho_head, 'rho_fix': rho_fix}

    def test_head_gauge_equals_caustic_rho_of_full_physical_radius(self) -> None:
        """The HEAD outer-rho-band gauge IS `caustic_rho` of the full radius.

        ``physical_exclusion_radius / reach`` is bit-identical to
        ``caustic_rho(gamma_mid, physical_exclusion_radius, 0)`` because
        both divide by the same deterministic reach.  This pins the claim
        that the two gauges differ ONLY in the source magnitude they feed.
        """
        self._expect_comparisons = True
        self.n_compared += 1
        gauges = self._gauges(_NARROWING)
        self.assertEqual(
            gauges['rho_head'],
            caustic_rho(gauges['gamma_mid'], gauges['physical'], 0.0))

    def test_narrowed_region_is_strictly_inside_outer_rho_band(self) -> None:
        """Premise: the served region is strictly inside the outer rho-band.

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
        ax.set_xlabel('caustic-relative rho')
        ax.set_ylabel('w_cert')
        ax.set_title('D1: fixed gauge reads inward -> higher w_cert cell')
        ax.legend()
        fig.savefig(_OUTPUT_DIR / 'ppgo_exclusion_w_cert_read_points.png',
                    dpi=90)
        plt.close(fig)


class PpgoOrderingReachableRedTestCase(_GaugeTestCase):
    """D1 defect-1 reachable-red guard: the ordering bug reads an easier cell.

    The defect was an ORDERING bug: the caustic-relative coordinate was derived
    from the PRE-narrowing outer rho-band (``exclusion_rho``) instead of the
    NARROWED served region (``region_exclusion_rho``) that the positive-parity
    per-``theta_c``-column admission actually covers.  Both orderings feed the
    SAME authoritative converter `caustic_rho` and divide by the SAME reach --
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
        """Physical ``|y|`` of the PRE-narrowing outer rho-band (the bug).

        ``|y| = exclusion_rho - 1 + coordinate_radius_min`` which is exactly
        the full ``physical_exclusion_radius`` -- the magnitude the buggy
        ordering consumes because it reads ``exclusion_rho`` before the
        narrowing to ``region_exclusion_rho`` is applied.
        """
        scalars = _band_gauge_scalars()
        return scalars['exclusion_rho'] - 1.0 + scalars['coordinate_radius_min']

    def _read_rho(self, ordering: str) -> float:
        """Caustic-relative read-point for the two orderings (`caustic_rho`)."""
        scalars = _band_gauge_scalars()
        if ordering == 'fixed':
            magnitude = self._served_inner_magnitude()
        elif ordering == 'buggy':
            magnitude = self._outer_magnitude()
        else:
            raise ValueError(f'unknown ordering {ordering!r}')
        return caustic_rho(scalars['gamma_mid'], magnitude, 0.0)

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
        inside the outer rho-band, so ``rho_buggy > rho_fixed`` -- the two
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
        ax.set_xlabel('caustic-relative rho')
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

    The independent HEAD oracle is `caustic_rho` evaluated on the FULL
    physical exclusion radius: `caustic_rho` divides by
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
        return caustic_rho(scalars['gamma_mid'],
                           scalars['physical_exclusion_radius'], 0.0)

    def _narrowed_foil_rho(self, narrowing: float) -> float:
        """What a saddle would read IF it wrongly narrowed like positive parity.

        Feeds the additive-inverted magnitude of a narrowed region through
        `caustic_rho`; used only to witness that the saddle branch does NOT
        do this (its read-point is strictly larger).
        """
        scalars = _saddle_gauge_scalars()
        region = scalars['exclusion_rho'] - narrowing
        magnitude = region - 1.0 + scalars['coordinate_radius_min']
        return caustic_rho(scalars['gamma_mid'], magnitude, 0.0)

    def test_saddle_branch_is_byte_identical_to_head(self) -> None:
        """``physical / reach`` equals `caustic_rho` of the full radius EXACTLY.

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
        ``caustic_rho`` -- otherwise byte-equality would pass vacuously.
        """
        self._expect_comparisons = True
        differ = 0
        for gamma, magnitude in itertools.product(_D2_GAMMAS, _D2_MAGNITUDES):
            reach = caustic_geometry(gamma, 0.0)[0]
            tainted = magnitude / (reach * (1.0 + 1.0e-9))
            if caustic_rho(gamma, magnitude, 0.0) != tainted:
                differ += 1
            self.n_compared += 1
        self.assertEqual(differ, len(_D2_GAMMAS) * len(_D2_MAGNITUDES),
                         'a corrupted reach slipped past exact equality')

    def test_head_gauge_would_read_an_easier_or_equal_cell(self) -> None:
        """Regressing to the HEAD gauge loses the strict "harder" property.

        If the code reverted to reading the ppGO cell at ``rho_head`` (the
        pre-fix outer-rho-band gauge), the "strictly harder" assertion would
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
        narrowed_rho = caustic_rho(
            gamma_mid, region - 1.0 + scalars['coordinate_radius_min'], 0.0)
        w_head = ppgo.w_cert('saddle', gamma_mid, head_rho)
        w_narrowed = ppgo.w_cert('saddle', gamma_mid, narrowed_rho)
        self.assertLess(narrowed_rho, head_rho)
        self.assertIsNot(w_head, UNKNOWN)
        self.assertIsNot(w_narrowed, UNKNOWN)
        self.assertNotEqual(w_head, w_narrowed)
        self.n_compared += 1


# ======================================================================
# WP1: caustic_geometry's 720-point polar scan replaced by the closed-form
# reach + direction.  The three specifications below certify the REWRITE:
#   (1) the closed-form reach reproduces a brute high-resolution parametric
#       scan of the caustic radius (an INDEPENDENT oracle, F026 prior art);
#   (2) the maximiser the closed form locates is a genuine STATIONARY point
#       of the caustic radius, confirmed by the independent analytic tangent
#       `geometry.caustic_derivatives` (machine-precision self-check, no scan);
#   (3) `surrogate._caustic_reach` (imported here as `_scalar_caustic_reach`)
#       still routes bit-for-bit through the rewritten `caustic_geometry`, so
#       no second reach copy was introduced.
# ======================================================================

#: (gamma, kappa) grid spanning both parities and the regimes the WP1 brief
#: names: the comfortable positive-parity middle (0.6, 0.9 @ kappa=0), the
#: near-wall macro saddle (1.001..1.2), both sides of the off-axis -> on-axis
#: cusp switch at ``gamma ~ 1.177651`` (1.05 off-axis, 1.3 on-axis), and two
#: ``kappa != 0`` cases whose reduced shear ``e = gamma / (1 - kappa)`` stays
#: in a validated positive-parity range.
_WP1_GRID: tuple[tuple[float, float], ...] = (
    (0.6, 0.0), (0.9, 0.0),
    (1.001, 0.0), (1.005, 0.0), (1.05, 0.0), (1.1, 0.0), (1.2, 0.0),
    (1.3, 0.0),
    (0.5, 0.2), (0.7, 0.2))

#: Number of lens-plane polar samples for the brute parametric caustic scan.
#: The Professor floored this at 11520; the sharpest near-wall spike
#: (gamma=1.001, reach ~ 22) has a raw-scan discretization error of 3.1e-7 at
#: 11520 -- above the 1e-7 bar -- but 1.3e-8 at 46080.  This is 4x the floor:
#: still a PURE brute parametric scan (no local refinement), just dense enough
#: that the discretization error of the oracle itself sits under the bar.
_WP1_SCAN_N_THETA = 46080

#: Relative-agreement bar between the closed-form reach and the brute scan.
#: NOT 1e-9: the scan oracle is itself only ~1e-8 accurate (its own O(h**2)
#: discretization error), so 1e-7 is the right bar; the MEASURED agreement is
#: a few x 1e-8, reported by the test.
_WP1_REACH_RTOL = 1e-7

#: Stationarity bar: ``|d|y|**2 / dtheta| / |y|**2`` at the maximiser.  A
#: genuine stationary point of the caustic radius has ``y . y' = 0``; the
#: smooth off-axis deltoid maximisers clear this at ~1e-13.
_WP1_STATIONARITY_RATIO_BAR = 1e-9

#: Cusp-speed floor for the stationarity DISJUNCTION.  The farthest caustic
#: point is a CUSP (astroid axis cusps at positive parity; the outer deltoid
#: cusp at a macro saddle), where the parametric speed ``|y'|`` is analytically
#: zero.  Its float64 evaluation is a numerical zero -- up to 1.3e-7 at the
#: positive-parity axis cusp (gamma=0.9, where ``sin(pi) != 0`` leaks into the
#: tangent) -- so the ratio arm can miss (gamma=0.9 lands at 4.5e-8).  A floor
#: of 1e-4 is ~800x above that numerical zero yet ~4 orders below any genuine
#: finite caustic speed (median |y'| over a wedge is O(1)), cleanly separating
#: "the maximiser is a cusp" from "the maximiser is a non-stationary point".
_WP1_CUSP_SPEED_FLOOR = 1e-4

#: Wedge-turnaround candidate ``u = sqrt(e**2 - 1)`` (macro saddle): a REGULAR
#: point of the caustic where the theta-parametrization speed DIVERGES (F044),
#: not a stationary point -- excluded from the stationarity check.  It is never
#: the reach maximiser (verified in the suite), so excluding it removes no
#: admitted maximiser.
_WP1_WEDGE_LABEL = 'wedge'

#: gamma grid (both parities, kappa fixed at 0 since `_scalar_caustic_reach`
#: takes only gamma) for the single-source bit-identity check.
_WP1_SCALAR_GAMMAS: tuple[float, ...] = (0.3, 0.6, 0.9, 1.05, 1.2, 1.5)


def _wp1_parametric_radius(theta: np.ndarray, gamma: float, kappa: float,
                           branch: float) -> np.ndarray:
    """Independent F026 |y|(theta) caustic radius, vectorised over ``theta``.

    Evaluates the closed-form Chang--Refsdal caustic curve
    ``y_i = p_i r T_i`` with ``T = (cos theta, sin theta)``,
    ``r = 1 / sqrt(lam u)``, ``p_i = (lam -+ gamma) - lam u`` and
    ``u = e cos 2theta + branch sqrt(1 - e**2 sin**2 2theta)``
    (``lam = 1 - kappa``, ``e = gamma / lam``), returning ``|y|`` where the
    point is a real caustic point (``disc >= 0`` and ``u > 0``) and ``-inf``
    elsewhere so a ``max`` skips invalid samples.

    This is the F026 parametric oracle, generated from the critical curve
    rather than a source-plane uniform-theta ring: a ring can MISS the thin
    near-wall reach spike, whereas the lens-plane parametrization samples the
    whole caustic.  It shares NO code with `caustic_geometry`'s closed-form
    u-candidate extremiser; it is validated against the production caustic
    evaluator `geometry._caustic_source` in the suite before use.
    """
    lam = 1.0 - kappa
    effective_shear = gamma / lam
    sin_2t = np.sin(2.0 * theta)
    cos_2t = np.cos(2.0 * theta)
    discriminant = 1.0 - effective_shear**2 * sin_2t**2
    valid = discriminant >= 0.0
    radial_u = (effective_shear * cos_2t
                + branch * np.sqrt(np.where(valid, discriminant, 0.0)))
    valid = valid & (radial_u > 0.0)
    safe_u = np.where(valid, radial_u, 1.0)
    radius_scale = 1.0 / np.sqrt(lam * safe_u)
    p_x = (lam - gamma) - lam * safe_u
    p_y = (lam + gamma) - lam * safe_u
    y_x = p_x * radius_scale * np.cos(theta)
    y_y = p_y * radius_scale * np.sin(theta)
    return np.where(valid, np.hypot(y_x, y_y), -np.inf)


def _wp1_scan_reach(gamma: float, kappa: float,
                    n_theta: int = _WP1_SCAN_N_THETA) -> float:
    """Brute maximum of the parametric caustic radius over ``[0, 2 pi)``.

    Positive parity (``e < 1``) has a single astroid on the ``+`` branch; a
    macro saddle (``e > 1``) has two deltoid lobes traced by BOTH ``+-``
    branches, so the scan maximises over both.  Independent of
    `caustic_geometry`.
    """
    lam = 1.0 - kappa
    effective_shear = gamma / lam
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    branches = (1.0,) if effective_shear < 1.0 else (1.0, -1.0)
    return max(float(np.max(_wp1_parametric_radius(theta, gamma, kappa, br)))
               for br in branches)


class ClosedFormReachVsParametricScanTestCase(_GaugeTestCase):
    """WP1 spec 1: closed-form reach reproduces a converged parametric scan.

    The reach element of `caustic_geometry` is compared against an INDEPENDENT
    high-resolution parametric scan of the caustic radius (`_wp1_scan_reach`,
    the F026 |y|(theta) parametrization -- NOT a source-plane ring, which can
    miss the thin near-wall spike).  The scan is first validated against the
    production caustic evaluator `geometry._caustic_source` so a transcription
    error in the oracle cannot masquerade as agreement (two-stage oracle).
    """

    def test_parametric_radius_matches_production_caustic_source(self) -> None:
        """Stage 1: the hand-rolled |y|(theta) equals `geometry._caustic_source`.

        Before the scan is trusted as an oracle its per-theta radius must match
        the production caustic point evaluator (a DIFFERENT code path:
        ``macro_matrix @ x - x / |x|**2``) to ~1e-12 relative, at a spread of
        real caustic angles on both parities and both branches.
        """
        self._expect_comparisons = True
        for gamma, kappa in ((0.6, 0.0), (0.9, 0.0), (1.05, 0.0),
                             (1.2, 0.0), (0.7, 0.2)):
            lam = 1.0 - kappa
            effective_shear = gamma / lam
            branches = (1.0,) if effective_shear < 1.0 else (1.0, -1.0)
            for branch in branches:
                for theta in np.linspace(0.02, 0.5, 6):
                    discriminant = (1.0 - effective_shear**2
                                    * math.sin(2.0 * theta)**2)
                    radial_u = (effective_shear * math.cos(2.0 * theta)
                                + branch * math.sqrt(max(discriminant, 0.0)))
                    if discriminant < 0.0 or radial_u <= 0.0:
                        continue
                    production = float(np.linalg.norm(geometry._caustic_source(
                        theta, gamma, 0.0, kappa, branch)))
                    mine = float(_wp1_parametric_radius(
                        np.array([theta]), gamma, kappa, branch)[0])
                    with self.subTest(gamma=gamma, kappa=kappa,
                                      branch=branch, theta=theta):
                        self.assertLess(abs(production - mine),
                                        1e-12 * abs(production) + 1e-14)
                    self.n_compared += 1

    def test_closed_form_reach_matches_parametric_scan(self) -> None:
        """The closed-form reach matches the brute scan to <= 1e-7 relative.

        Over the full WP1 grid -- comfortable middle, the ``e < sqrt(3)/2``
        band, the near-wall saddle spike, both sides of the cusp switch, and
        two ``kappa != 0`` cases -- the relative disagreement is bounded by the
        (measured, few-x-1e-8) discretization error of the scan itself.
        """
        self._expect_comparisons = True
        measured: list[tuple[float, float]] = []
        worst = 0.0
        for gamma, kappa in _WP1_GRID:
            closed_form = caustic_geometry(gamma, kappa)[0]
            scan = _wp1_scan_reach(gamma, kappa)
            rel = abs(closed_form - scan) / abs(closed_form)
            with self.subTest(gamma=gamma, kappa=kappa):
                self.assertLessEqual(
                    rel, _WP1_REACH_RTOL,
                    f'closed-form reach {closed_form!r} vs parametric scan '
                    f'{scan!r} disagree by rel={rel:.3e} > {_WP1_REACH_RTOL} '
                    f'at gamma={gamma}, kappa={kappa}')
            measured.append((gamma, rel))
            worst = max(worst, rel)
            self.n_compared += 1
        # The bar is 1e-7; the measured worst is a few x 1e-8 -- assert it did
        # not silently balloon (e.g. an oracle that lost the near-wall spike
        # would show rel ~ O(1), not ~1e-8).
        self.assertLess(worst, 5e-8,
                        f'measured worst reach agreement {worst:.3e} is far '
                        f'above the expected few-x-1e-8 floor')
        self._save_reach_plot(measured)

    def _save_reach_plot(self, measured: list[tuple[float, float]]) -> None:
        """Diagnostic: reach(gamma) across the wall, coarse ring vs parametric.

        Left panel -- the closed-form reach and the dense parametric scan track
        across the near-wall spike; a DELIBERATELY coarse scan (n_theta=360)
        under-resolves the spike, the failure mode the parametric oracle at
        46080 avoids.  Right panel -- the measured closed-form-vs-scan relative
        agreement over the grid, all far below the 1e-7 bar.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return
        _OUTPUT_DIR.mkdir(exist_ok=True)
        gammas = np.linspace(1.0005, 1.30, 120)
        closed = [caustic_geometry(float(g), 0.0)[0] for g in gammas]
        dense = [_wp1_scan_reach(float(g), 0.0) for g in gammas]
        coarse = [_wp1_scan_reach(float(g), 0.0, n_theta=360) for g in gammas]
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
        ax0.plot(gammas, closed, 'k-', lw=1.5, label='closed-form reach')
        ax0.plot(gammas, dense, 'g.', ms=3,
                 label=f'parametric scan (n={_WP1_SCAN_N_THETA})')
        ax0.plot(gammas, coarse, 'r+', ms=5,
                 label='coarse scan (n=360, misses spike)')
        ax0.set_yscale('log')
        ax0.set_xlabel('gamma')
        ax0.set_ylabel('caustic reach')
        ax0.set_title('near-wall reach spike')
        ax0.legend(fontsize=8)
        gs = [g for g, _ in measured]
        rels = [r for _, r in measured]
        ax1.scatter(gs, rels, s=40)
        ax1.axhline(_WP1_REACH_RTOL, color='r', ls='--',
                    label=f'bar {_WP1_REACH_RTOL:g}')
        ax1.set_yscale('log')
        ax1.set_xlabel('gamma')
        ax1.set_ylabel('|closed-form - scan| / reach')
        ax1.set_title('closed-form vs scan agreement')
        ax1.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / 'wp1_reach_closed_form_vs_scan.png', dpi=90)
        plt.close(fig)

def _wp1_winning_maximiser(gamma: float, kappa: float
                           ) -> tuple[float, float, str, float]:
    """Recover the analytic reach maximiser `caustic_geometry` locates.

    Reproduces `caustic_geometry`'s finite candidate-``u`` extremiser and maps
    the WINNING ``u`` back to a machine-precise ``(theta_win, branch)`` via
    ``cos 2theta = (u**2 - 1 + e**2) / (2 e u)`` and the sign of
    ``u - e cos 2theta`` (the ``+-`` root selector), where ``lam = 1 - kappa``
    and ``e = gamma / lam``.  Recovering ``theta`` from the analytic ``u`` --
    rather than from a radius-maximising scan -- gives the winning angle to
    machine precision, so the stationarity residual is limited by the analytic
    tangent, not by a coarse angular grid.

    Returns
    -------
    theta_win, branch, label, reach
        The first-quadrant winning angle (rad), its ``+-1`` branch, a label
        (``'axis_minus'`` / ``'axis_plus'`` / ``'offaxis'`` / the wedge label),
        and the winning caustic radius.
    """
    lam = 1.0 - kappa
    effective_shear = gamma / lam
    if effective_shear < 1.0:
        candidates = [(1.0 - effective_shear, 'axis_minus'),
                      (1.0 + effective_shear, 'axis_plus')]
    else:
        candidates = [(1.0 + effective_shear, 'axis_plus')]
        radicand = 4.0 * effective_shear**2 - 3.0
        if radicand >= 0.0:
            u_offaxis = (-1.0 + math.sqrt(radicand)) / 2.0
            if u_offaxis > 0.0:
                candidates.append((u_offaxis, 'offaxis'))
        candidates.append((math.sqrt(effective_shear**2 - 1.0),
                           _WP1_WEDGE_LABEL))
    best: tuple[float, float, str] | None = None  # (radius, u, label)
    for u_candidate, label in candidates:
        if u_candidate <= 0.0:
            continue
        cos_2theta = ((u_candidate**2 - 1.0 + effective_shear**2)
                      / (2.0 * effective_shear * u_candidate))
        if abs(cos_2theta) > 1.0 + 1e-12:
            continue
        radius_sq = lam * ((1.0 - u_candidate)**2 * (1.0 + 2.0 * u_candidate)
                           + effective_shear**2 * (2.0 * u_candidate - 1.0)
                           ) / u_candidate**2
        if radius_sq <= 0.0:
            continue
        radius = math.sqrt(radius_sq)
        if best is None or radius > best[0]:
            best = (radius, u_candidate, label)
    if best is None:
        raise AssertionError(f'no maximiser for gamma={gamma}, kappa={kappa}')
    reach, u_win, label = best
    cos_2theta = min(1.0, max(-1.0, (u_win**2 - 1.0 + effective_shear**2)
                              / (2.0 * effective_shear * u_win)))
    theta_win = 0.5 * math.acos(cos_2theta)
    branch = 1.0 if (u_win - effective_shear * cos_2theta) >= 0.0 else -1.0
    return theta_win, branch, label, reach


def _wp1_stationarity_ratio(gamma: float, kappa: float, theta: float,
                            branch: float) -> tuple[float, float]:
    """``|d|y|**2 / dtheta| / |y|**2`` and the caustic speed ``|y'|`` at theta.

    ``y`` is the production caustic point `geometry._caustic_source` and ``y'``
    the INDEPENDENT analytic tangent `geometry.caustic_derivatives` -- a
    different code path from `caustic_geometry`'s extremiser.  With
    ``d|y|**2 / dtheta = 2 (y . y')`` a genuine stationary point of the caustic
    radius has ratio ``= 0``; a cusp additionally has ``|y'| = 0``.
    """
    point = geometry._caustic_source(theta, gamma, 0.0, kappa, branch)
    y_prime, _ = geometry.caustic_derivatives(
        gamma, theta, kappa=kappa, branch=int(branch))
    y_prime = np.asarray(y_prime, dtype=float).reshape(2)
    y_dot_yp = float(point[0] * y_prime[0] + point[1] * y_prime[1])
    y_norm_sq = float(point[0]**2 + point[1]**2)
    speed = float(math.hypot(y_prime[0], y_prime[1]))
    return abs(2.0 * y_dot_yp) / y_norm_sq, speed


class ReachMaximiserStationarityTestCase(_GaugeTestCase):
    """WP1 spec 2: the located maximiser is a stationary point of the radius.

    For each admitted, non-wedge maximiser across the WP1 grid the winning
    ``(theta, branch)`` is recovered analytically and the caustic radius'
    stationarity residual ``|d|y|**2 / dtheta| / |y|**2`` is formed from the
    INDEPENDENT analytic tangent `geometry.caustic_derivatives`.  The bar is a
    DISJUNCTION: the residual is ``<= 1e-9`` (smooth off-axis maximisers) OR
    the caustic speed ``|y'|`` is below a small floor (an axis/on-axis CUSP,
    where ``|y'|`` is analytically zero and the residual is ``0 / 0`` in the
    limit -- its float64 evaluation is a numerical, not exact, zero).  The
    wedge-turnaround candidate ``u = sqrt(e**2 - 1)`` is a REGULAR point where
    the theta-parametrization speed DIVERGES (F044); it is never the maximiser
    (asserted) so its exclusion drops no admitted point.
    """

    def test_maximiser_ties_back_to_caustic_geometry(self) -> None:
        """The analytically recovered maximiser reproduces the WP output.

        The recovered ``(theta_win, branch)`` -- evaluated through the
        production caustic point `geometry._caustic_source` -- must reproduce
        both the reach (radius) and the canonicalised direction that
        `caustic_geometry` returns, confirming the stationarity check probes
        the SAME point the closed form selected.
        """
        self._expect_comparisons = True
        for gamma, kappa in _WP1_GRID:
            reach, direction = caustic_geometry(gamma, kappa)
            theta_win, branch, label, recovered_reach = \
                _wp1_winning_maximiser(gamma, kappa)
            point = np.asarray(geometry._caustic_source(
                theta_win, gamma, 0.0, kappa, branch), dtype=float)
            recovered_direction = point / math.hypot(point[0], point[1])
            # The caustic's 4-fold symmetry makes the quadrant physically
            # irrelevant (the WP docstring canonicalises it away), so the
            # tie-back is PARALLELISM up to sign: |recovered . direction| = 1.
            # Exact-component matching is fragile on axis cusps, where a
            # ~1e-33 numerical x-component flips the canonical sign choice.
            alignment = abs(float(recovered_direction @ direction))
            with self.subTest(gamma=gamma, kappa=kappa, label=label):
                self.assertAlmostEqual(recovered_reach, reach, places=12)
                self.assertAlmostEqual(
                    float(np.linalg.norm(point)), reach,
                    delta=1e-9 * reach + 1e-12)
                self.assertAlmostEqual(alignment, 1.0, places=9)
            self.n_compared += 1

    def test_reach_maximiser_is_stationary(self) -> None:
        """Residual is machine-zero (ratio arm) or the point is a cusp (floor).

        Both arms of the disjunction must be load-bearing over the grid: the
        smooth off-axis maximisers clear the ratio bar (~1e-13) while the
        axis/on-axis cusps are caught only by the speed floor.  The test fails
        if EITHER arm is never exercised (a silently one-armed gate).
        """
        self._expect_comparisons = True
        ratio_arm = 0
        floor_arm = 0
        diagnostic: list[tuple[float, float, float]] = []
        for gamma, kappa in _WP1_GRID:
            theta_win, branch, label, _ = _wp1_winning_maximiser(gamma, kappa)
            self.assertNotEqual(
                label, _WP1_WEDGE_LABEL,
                f'the wedge turnaround unexpectedly won at gamma={gamma}, '
                f'kappa={kappa}; F044 says it is not a stationary point')
            ratio, speed = _wp1_stationarity_ratio(
                gamma, kappa, theta_win, branch)
            passes_ratio = ratio <= _WP1_STATIONARITY_RATIO_BAR
            passes_floor = speed < _WP1_CUSP_SPEED_FLOOR
            with self.subTest(gamma=gamma, kappa=kappa, label=label):
                self.assertTrue(
                    passes_ratio or passes_floor,
                    f'maximiser at gamma={gamma}, kappa={kappa} is neither '
                    f'stationary (ratio={ratio:.3e} > '
                    f'{_WP1_STATIONARITY_RATIO_BAR}) nor a cusp '
                    f'(|y_prime|={speed:.3e} >= {_WP1_CUSP_SPEED_FLOOR})')
            if passes_ratio:
                ratio_arm += 1
            else:
                floor_arm += 1
            diagnostic.append((gamma, ratio, speed))
            self.n_compared += 1
        self.assertGreater(ratio_arm, 0, 'the ratio arm was never exercised')
        self.assertGreater(floor_arm, 0, 'the cusp-floor arm was never '
                           'exercised (the disjunction is one-armed)')
        self._save_stationarity_plot(diagnostic)

    def _save_stationarity_plot(
            self, diagnostic: list[tuple[float, float, float]]) -> None:
        """Diagnostic: stationarity ratio and caustic speed vs gamma."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return
        _OUTPUT_DIR.mkdir(exist_ok=True)
        gammas = [g for g, _, _ in diagnostic]
        ratios = [max(r, 1e-18) for _, r, _ in diagnostic]
        speeds = [max(s, 1e-18) for _, _, s in diagnostic]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(gammas, ratios, s=40, label='stationarity ratio')
        ax.scatter(gammas, speeds, s=40, marker='x', label='caustic speed |y_prime|')
        ax.axhline(_WP1_STATIONARITY_RATIO_BAR, color='r', ls='--',
                   label=f'ratio bar {_WP1_STATIONARITY_RATIO_BAR:g}')
        ax.axhline(_WP1_CUSP_SPEED_FLOOR, color='g', ls=':',
                   label=f'speed floor {_WP1_CUSP_SPEED_FLOOR:g}')
        ax.set_yscale('log')
        ax.set_xlabel('gamma')
        ax.set_ylabel('stationarity ratio / caustic speed')
        ax.set_title('reach maximiser stationarity self-check')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / 'wp1_stationarity_ratio_vs_gamma.png', dpi=90)
        plt.close(fig)


class SingleSourceReachEqualityTestCase(_GaugeTestCase):
    """WP1 spec 3: `_scalar_caustic_reach` still routes through the rewrite.

    `surrogate._caustic_reach` (imported as `_scalar_caustic_reach`) must
    return the reach element of `caustic_geometry(gamma, 0.0)` BIT-FOR-BIT --
    exact float equality, not closeness -- so that no second reach copy has
    been introduced by the WP1 rewrite.  Any nonzero difference means the
    scalar helper computes the reach a different way.
    """

    def test_scalar_reach_is_bit_identical_to_caustic_geometry(self) -> None:
        self._expect_comparisons = True
        for gamma in _WP1_SCALAR_GAMMAS:
            routed = _scalar_caustic_reach(gamma)
            direct = caustic_geometry(gamma, 0.0)[0]
            with self.subTest(gamma=gamma):
                self.assertEqual(
                    routed, direct,
                    f'_scalar_caustic_reach({gamma}) = {routed!r} differs from '
                    f'caustic_geometry({gamma}, 0.0)[0] = {direct!r}: a second '
                    f'reach copy exists')
            self.n_compared += 1


class Wp1SelfFalsificationTestCase(_GaugeTestCase):
    """WP1: the reach, stationarity and single-source gates can go RED.

    A numerical suite that cannot demonstrate its own failure mode is not
    finished.  Each check below corrupts exactly one ingredient and confirms
    the corresponding gate's assertion would fire.
    """

    def test_coarse_scan_misses_the_near_wall_spike(self) -> None:
        """A coarse (n=360) parametric ring under-resolves the wall spike.

        The reach gate's teeth are the resolution of the parametric oracle: at
        the sharpest near-wall config (gamma=1.001) a 360-point scan disagrees
        with the closed form by far more than the 1e-7 bar, whereas the
        shipped 46080-point scan clears it.  This proves the comparison is not
        vacuously satisfied by any scan density.
        """
        self._expect_comparisons = True
        coarse = _wp1_scan_reach(1.001, 0.0, n_theta=360)
        dense = _wp1_scan_reach(1.001, 0.0, n_theta=_WP1_SCAN_N_THETA)
        closed = caustic_geometry(1.001, 0.0)[0]
        rel_coarse = abs(closed - coarse) / abs(closed)
        rel_dense = abs(closed - dense) / abs(closed)
        self.assertGreater(
            rel_coarse, _WP1_REACH_RTOL,
            'a coarse ring should MISS the near-wall spike, but it agreed to '
            f'rel={rel_coarse:.3e} <= {_WP1_REACH_RTOL}')
        self.assertLessEqual(
            rel_dense, _WP1_REACH_RTOL,
            f'the dense scan should clear the bar (rel={rel_dense:.3e})')
        self.n_compared += 1

    def test_offset_angle_breaks_stationarity(self) -> None:
        """Evaluating the tangent off the maximiser fails BOTH arms.

        At ``theta_win + 0.12`` rad the point is neither stationary
        (``y . y' != 0``) nor a cusp (``|y'|`` is O(1)), so the disjunction
        used by `test_reach_maximiser_is_stationary` must reject it -- proving
        the machine-precision gate has teeth.
        """
        self._expect_comparisons = True
        for gamma, kappa in ((0.6, 0.0), (1.2, 0.0)):
            theta_win, branch, _, _ = _wp1_winning_maximiser(gamma, kappa)
            ratio, speed = _wp1_stationarity_ratio(
                gamma, kappa, theta_win + 0.12, branch)
            with self.subTest(gamma=gamma, kappa=kappa):
                self.assertFalse(
                    ratio <= _WP1_STATIONARITY_RATIO_BAR
                    or speed < _WP1_CUSP_SPEED_FLOOR,
                    f'an off-maximiser angle passed the stationarity gate '
                    f'(ratio={ratio:.3e}, speed={speed:.3e})')
            self.n_compared += 1

    def test_one_ulp_breaks_single_source_equality(self) -> None:
        """A single-ULP perturbation defeats the exact-equality gate.

        Confirms `SingleSourceReachEqualityTestCase` demands BIT equality: the
        next representable float above the routed reach is not equal to it.
        """
        self._expect_comparisons = True
        for gamma in (0.6, 1.2):
            routed = _scalar_caustic_reach(gamma)
            perturbed = math.nextafter(routed, math.inf)
            with self.subTest(gamma=gamma):
                self.assertNotEqual(routed, perturbed)
            self.n_compared += 1


# ======================================================================
# WP1 (continued): two further specifications of the closed-form rewrite.
#   (4) SANITY LITERAL + on-axis direction.  At (gamma, kappa) = (0.9, 0)
#       the winning candidate is the positive-parity axis cusp ``u = 1 - e``,
#       whose caustic radius reduces ALGEBRAICALLY to ``2 gamma / sqrt(1 -
#       gamma)`` (derived here, independent of `caustic_geometry`; kappa=0).
#       This equals SPEC.md's recorded cusp radius 5.692100... .  The
#       returned direction is axis-aligned in the shear eigenframe (its
#       projection on the OTHER eigen-axis is ~0); its quadrant/sign is NOT
#       pinned -- the caustic's 4-fold reflection symmetry makes it
#       physically irrelevant (Professor).
#   (5) OFF-AXIS DIRECTION agreement.  In the genuinely diagonal saddle band
#       ``1 < gamma < 1.177651`` the reach maximiser is an off-axis deltoid
#       extremum; the closed-form direction agrees with a converged parametric
#       scan's farthest-point direction, compared as an angle reduced modulo
#       the 4-fold quadrant symmetry (axis-alignment, not raw signed
#       components), to within the scan's angular resolution ~2 pi / 11520.
# ======================================================================

#: Positive-parity gammas (kappa = 0) for the sanity sweep.  For every e < 1
#: the ``u = 1 - e`` axis cusp wins, so the reach is ``2 gamma / sqrt(1 -
#: gamma)`` and the direction is the second eigen-axis ``(0, +-1)`` exactly.
_WP1_SANITY_GAMMAS: tuple[float, ...] = (0.3, 0.6, 0.9)

#: Relative bar for reach == 2 gamma / sqrt(1 - gamma).  Both sides are
#: float64 evaluations of the same algebra along different code paths; the
#: MEASURED agreement is ~1e-16, so 1e-9 (the spec bar) is generous.
_WP1_SANITY_REACH_RTOL = 1e-9

#: SPEC.md's recorded positive-parity cusp radius at gamma = 0.9 (the literal
#: the brief cites).  It is TRUNCATED: it differs from the exact
#: ``2 * 0.9 / sqrt(0.1) = 5.692099788...`` by 2.1e-7, which EXCEEDS 1e-9.  So
#: the tight 1e-9 gate is against the exact closed form; this literal is used
#: only for a loose consistency straddle whose tolerance dwarfs its truncation.
_WP1_SPEC_CUSP_RADIUS = 5.692100

#: Loose tolerance for the SPEC-literal consistency check.  The literal's own
#: truncation error is 2.1e-7; 1e-6 dwarfs it (per the standing rule: a
#: brief's truncated literal is fine only where the margin swamps its error).
_WP1_SPEC_CUSP_ATOL = 1e-6

#: Axis-alignment bar: the direction's projection on the perpendicular
#: eigen-axis, ``min(|d_0|, |d_1|)``, must not exceed this for an on-axis cusp.
#: MEASURED value is 0.0 (the direction is ``(0, 1)`` exactly).
_WP1_AXIS_ALIGN_ATOL = 1e-9

#: Off-axis band gammas (kappa = 0), strictly inside ``(1, 1.177651)`` where
#: the off-axis deltoid extremum is the reach maximiser (MEASURED: the winning
#: direction's smaller eigen-component is 0.138 at 1.05, 0.285 at 1.1).
_WP1_OFFAXIS_GAMMAS: tuple[float, ...] = (1.05, 1.1)

#: Angular-agreement bar between the closed-form and converged-scan directions,
#: reduced modulo the 4-fold symmetry: ~2 pi / 11520, the parametric scan's
#: nominal angular resolution.  MEASURED agreement is < 1.5e-8 (the maximiser
#: direction is a smooth function of gamma, so the converged scan pins it far
#: tighter than one grid step), reported by the test.
_WP1_DIRECTION_ANGLE_BAR = 2.0 * math.pi / 11520.0

#: Floor on the smaller eigen-component of the closed-form direction in the
#: off-axis band: below this the maximiser would be an on-axis cusp (which
#: trivially satisfies any axis-reduced-angle bar), so the test asserts it is
#: exceeded to confirm the diagonal regime is genuinely exercised.
_WP1_OFFAXIS_DIAGONAL_FLOOR = 0.05


def _wp1_scan_direction(gamma: float, kappa: float,
                        n_theta: int = _WP1_SCAN_N_THETA) -> np.ndarray:
    """Unit direction to the farthest caustic point from a brute theta scan.

    Independent of `caustic_geometry`: scans the F026 parametric caustic radius
    (`_wp1_parametric_radius`) over ``[0, 2 pi)`` on both ``+-`` branches, takes
    the global argmax, and returns the unit direction of that farthest caustic
    point evaluated through the production point map `geometry._caustic_source`
    (a DIFFERENT code path from the closed-form direction assembly).  This is
    the scan counterpart of `caustic_geometry`'s ``unit_direction``.
    """
    lam = 1.0 - kappa
    effective_shear = gamma / lam
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    branches = (1.0,) if effective_shear < 1.0 else (1.0, -1.0)
    best_radius = -np.inf
    best_theta = 0.0
    best_branch = 1.0
    for branch in branches:
        radii = _wp1_parametric_radius(theta, gamma, kappa, branch)
        idx = int(np.argmax(radii))
        if radii[idx] > best_radius:
            best_radius = float(radii[idx])
            best_theta = float(theta[idx])
            best_branch = branch
    point = np.asarray(geometry._caustic_source(
        best_theta, gamma, 0.0, kappa, best_branch), dtype=float)
    return point / math.hypot(point[0], point[1])


def _wp1_axis_reduced_angle(dir_a: np.ndarray, dir_b: np.ndarray) -> float:
    """Angle (rad) between two unit directions, reduced modulo 4-fold symmetry.

    The Chang--Refsdal caustic is invariant under reflection across either
    shear eigen-axis (a Klein 4-group of quadrant reflections), so a direction
    and its axis-reflections are physically identical.  Folding both vectors
    into the first quadrant via ``abs`` (norm-preserving) collapses that
    freedom; the returned angle is ``arccos`` of their clipped dot product.
    """
    a = np.abs(np.asarray(dir_a, dtype=float))
    b = np.abs(np.asarray(dir_b, dtype=float))
    return math.acos(min(1.0, max(-1.0, float(a @ b))))


class CausticReachSanityLiteralTestCase(_GaugeTestCase):
    """WP1 spec 4: the sanity cusp radius and on-axis direction.

    At positive parity (``e < 1``) the reach maximiser is the ``u = 1 - e``
    axis cusp, whose caustic radius reduces algebraically to
    ``2 gamma / sqrt(1 - gamma)`` (kappa = 0).  This suite gates the reach
    against that INDEPENDENT closed form to 1e-9 relative, checks the
    canonical case gamma = 0.9 is consistent with SPEC.md's recorded literal
    5.692100..., and confirms the returned direction is axis-aligned in the
    eigenframe -- WITHOUT pinning its quadrant/sign, which the 4-fold caustic
    symmetry renders physically irrelevant.
    """

    def test_reach_matches_axis_cusp_closed_form(self) -> None:
        """reach == 2 gamma / sqrt(1 - gamma) over the positive-parity sweep.

        The right-hand side is derived here from the ``u = 1 - e`` candidate's
        squared radius, ``((1-u)^2 (1+2u) + e^2 (2u-1)) / u^2`` at u = 1 - e,
        which simplifies to ``4 gamma^2 / (1 - gamma)`` -- an independent algebra
        reduction, not a call into `caustic_geometry`.
        """
        self._expect_comparisons = True
        for gamma in _WP1_SANITY_GAMMAS:
            reach, _ = caustic_geometry(gamma, 0.0)
            oracle = 2.0 * gamma / math.sqrt(1.0 - gamma)
            rel = abs(reach - oracle) / abs(oracle)
            with self.subTest(gamma=gamma):
                self.assertLessEqual(
                    rel, _WP1_SANITY_REACH_RTOL,
                    f'reach {reach!r} at gamma={gamma} disagrees with the '
                    f'axis-cusp closed form {oracle!r} by rel={rel:.3e}')
            self.n_compared += 1

    def test_gamma_point_nine_matches_spec_literal(self) -> None:
        """The gamma = 0.9 reach equals the exact form and SPEC.md's literal.

        Two bars: (a) the tight 1e-9 relative gate against the EXACT
        ``2 * 0.9 / sqrt(0.1)`` (the SPEC literal 5.692100 is truncated and
        would MISS 1e-9 by 2.1e-7); (b) a loose consistency straddle showing
        the exact value rounds to SPEC's recorded 5.692100 within 1e-6.
        """
        self._expect_comparisons = True
        reach, direction = caustic_geometry(0.9, 0.0)
        exact = 2.0 * 0.9 / math.sqrt(1.0 - 0.9)
        rel = abs(reach - exact) / abs(exact)
        self.assertLessEqual(
            rel, _WP1_SANITY_REACH_RTOL,
            f'reach {reach!r} at gamma=0.9 disagrees with exact {exact!r} '
            f'by rel={rel:.3e} > {_WP1_SANITY_REACH_RTOL}')
        self.assertLessEqual(
            abs(exact - _WP1_SPEC_CUSP_RADIUS), _WP1_SPEC_CUSP_ATOL,
            f'exact cusp radius {exact!r} inconsistent with SPEC literal '
            f'{_WP1_SPEC_CUSP_RADIUS} beyond {_WP1_SPEC_CUSP_ATOL}')
        # Direction is axis-aligned (its projection on the other eigen-axis is
        # ~0); quadrant/sign deliberately NOT asserted.
        perpendicular_projection = min(abs(direction[0]), abs(direction[1]))
        self.assertLessEqual(
            perpendicular_projection, _WP1_AXIS_ALIGN_ATOL,
            f'direction {direction} is not axis-aligned: perpendicular '
            f'projection {perpendicular_projection:.3e} > {_WP1_AXIS_ALIGN_ATOL}')
        self.n_compared += 1

    def test_positive_parity_direction_is_axis_aligned(self) -> None:
        """Every positive-parity cusp direction lies on an eigen-axis.

        The direction is a UNIT vector (norm 1) whose projection on the
        perpendicular eigen-axis vanishes -- i.e. it is ``(0, +-1)`` or
        ``(+-1, 0)``.  The quadrant/sign is not pinned.
        """
        self._expect_comparisons = True
        for gamma in _WP1_SANITY_GAMMAS:
            _, direction = caustic_geometry(gamma, 0.0)
            with self.subTest(gamma=gamma):
                self.assertAlmostEqual(
                    float(np.hypot(direction[0], direction[1])), 1.0, places=12)
                self.assertLessEqual(
                    min(abs(direction[0]), abs(direction[1])),
                    _WP1_AXIS_ALIGN_ATOL,
                    f'direction {direction} at gamma={gamma} is not '
                    f'axis-aligned')
            self.n_compared += 1


class OffAxisDirectionAgreementTestCase(_GaugeTestCase):
    """WP1 spec 5: off-axis direction agrees with the converged scan.

    In the genuinely diagonal saddle band ``1 < gamma < 1.177651`` the reach
    maximiser is an off-axis deltoid extremum.  The closed-form direction is
    compared to the converged parametric scan's farthest-point direction as an
    angle reduced modulo the 4-fold quadrant symmetry (`_wp1_axis_reduced_angle`
    folds both into the first quadrant, so a sign/quadrant difference is not a
    disagreement).  The bar is the scan's nominal angular resolution
    ~2 pi / 11520; the measured agreement is far tighter.
    """

    def test_offaxis_direction_matches_converged_scan(self) -> None:
        self._expect_comparisons = True
        measured: list[tuple[float, float]] = []
        worst = 0.0
        for gamma in _WP1_OFFAXIS_GAMMAS:
            _, closed_direction = caustic_geometry(gamma, 0.0)
            scan_direction = _wp1_scan_direction(gamma, 0.0)
            angle = _wp1_axis_reduced_angle(closed_direction, scan_direction)
            smaller_component = min(abs(closed_direction[0]),
                                    abs(closed_direction[1]))
            with self.subTest(gamma=gamma):
                # Confirm we are genuinely in the diagonal regime, not at an
                # on-axis cusp that would satisfy the angle bar trivially.
                self.assertGreater(
                    smaller_component, _WP1_OFFAXIS_DIAGONAL_FLOOR,
                    f'direction {closed_direction} at gamma={gamma} is not '
                    f'genuinely off-axis (smaller component '
                    f'{smaller_component:.3e})')
                self.assertLessEqual(
                    angle, _WP1_DIRECTION_ANGLE_BAR,
                    f'closed-form direction {closed_direction} and converged '
                    f'scan direction {scan_direction} disagree by '
                    f'{angle:.3e} rad > {_WP1_DIRECTION_ANGLE_BAR:.3e} at '
                    f'gamma={gamma}')
            measured.append((gamma, angle))
            worst = max(worst, angle)
            self.n_compared += 1
        # Canary: the measured agreement is < 1.5e-8; assert it did not
        # silently balloon toward the (much looser) resolution bar.
        self.assertLess(
            worst, 1e-3,
            f'measured worst direction disagreement {worst:.3e} is far above '
            f'the expected sub-1e-7 floor')
        self._save_direction_plot(measured)

    def _save_direction_plot(self, measured: list[tuple[float, float]]) -> None:
        """Diagnostic: overlay closed-form and scan directions on the caustic."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return
        _OUTPUT_DIR.mkdir(exist_ok=True)
        gamma = _WP1_OFFAXIS_GAMMAS[-1]
        reach, closed_direction = caustic_geometry(gamma, 0.0)
        scan_direction = _wp1_scan_direction(gamma, 0.0)
        theta = np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False)
        caustic = np.array([geometry._caustic_source(float(t), gamma, 0.0, 0.0,
                                                     branch)
                            for branch in (1.0, -1.0) for t in theta])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(caustic[:, 0], caustic[:, 1], s=1, color='0.7',
                   label='caustic')
        ax.plot([0, reach * closed_direction[0]],
                [0, reach * closed_direction[1]], 'k-', lw=2,
                label='closed-form direction')
        ax.plot([0, reach * scan_direction[0]],
                [0, reach * scan_direction[1]], 'r--', lw=1.5,
                label='scan direction')
        ax.set_aspect('equal')
        ax.set_xlabel('y_1 (eigenframe)')
        ax.set_ylabel('y_2 (eigenframe)')
        ax.set_title(f'off-axis farthest-point direction (gamma={gamma})')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(_OUTPUT_DIR / 'wp1_offaxis_direction_overlay.png', dpi=90)
        plt.close(fig)


class Wp1DirectionSelfFalsificationTestCase(_GaugeTestCase):
    """WP1 specs 4-5: the sanity and direction gates can go RED.

    Each check corrupts exactly one ingredient and confirms the corresponding
    assertion would fire, so neither new gate is vacuously green.
    """

    def test_wrong_reach_breaks_sanity_gate(self) -> None:
        """A 0.1%-perturbed reach fails the 1e-9 axis-cusp closed-form gate."""
        self._expect_comparisons = True
        gamma = 0.9
        oracle = 2.0 * gamma / math.sqrt(1.0 - gamma)
        perturbed = oracle * 1.001
        rel = abs(perturbed - oracle) / abs(oracle)
        self.assertGreater(
            rel, _WP1_SANITY_REACH_RTOL,
            'a 0.1%-wrong reach should fail the sanity gate')
        self.n_compared += 1

    def test_diagonal_direction_fails_axis_alignment(self) -> None:
        """A genuinely diagonal direction is NOT axis-aligned.

        Confirms the axis-alignment gate has teeth: the perpendicular
        projection of a 45-degree direction is ~0.707, far above 1e-9.
        """
        self._expect_comparisons = True
        diagonal = np.array([1.0, 1.0]) / math.sqrt(2.0)
        perpendicular_projection = min(abs(diagonal[0]), abs(diagonal[1]))
        self.assertGreater(
            perpendicular_projection, _WP1_AXIS_ALIGN_ATOL,
            'a diagonal direction should fail the axis-alignment gate')
        self.n_compared += 1

    def test_rotated_direction_breaks_angle_bar(self) -> None:
        """Rotating the closed-form direction by 0.1 rad breaks the angle bar.

        The axis-reduced angle between a 0.1-rad-rotated closed-form direction
        and the converged scan direction is ~0.1 rad -- far above the
        ~5.5e-4 bar -- proving the off-axis agreement gate is not vacuous.
        """
        self._expect_comparisons = True
        for gamma in _WP1_OFFAXIS_GAMMAS:
            _, closed_direction = caustic_geometry(gamma, 0.0)
            scan_direction = _wp1_scan_direction(gamma, 0.0)
            cos_a, sin_a = math.cos(0.1), math.sin(0.1)
            rotated = np.array([
                cos_a * closed_direction[0] - sin_a * closed_direction[1],
                sin_a * closed_direction[0] + cos_a * closed_direction[1]])
            angle = _wp1_axis_reduced_angle(rotated, scan_direction)
            with self.subTest(gamma=gamma):
                self.assertGreater(
                    angle, _WP1_DIRECTION_ANGLE_BAR,
                    f'a 0.1-rad-rotated direction passed the angle bar '
                    f'(angle={angle:.3e}) at gamma={gamma}')
            self.n_compared += 1


if __name__ == '__main__':
    main()
