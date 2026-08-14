"""Tests for the region the near-cusp Pearcey arm actually COVERS (F074).

WHAT THIS SUITE USED TO ASSERT, AND WHY THAT IS GONE
----------------------------------------------------
Before F074 the cusp arm was gated on ``radius >= (c_P/bar)^(2/3)``, a
bound on the error of ``pearcey_asymptotic`` -- the object the uniform
form REPLACES.  That gate refused the near-vertex neighbourhood, which
made a purely angular statement true by accident: every served source sat
at least a fixed angular coverage margin (0.07 rad) away from the cusp
vertex in image-theta.  This file pinned that bound, its transition, and a
self-falsification of it.  (That margin -- a module constant in
``surrogate.py`` -- has since been retired outright; see the RETIRED block
at the bottom of this file for the F079 record.)

F074 replaced the gate with a bound on the error of what is actually
SERVED::

    est = _K_UNIFORM / sqrt(w) + |ghost(w)| e^{-w Im tau_c}  <=  envelope_bar

which is FREQUENCY-driven, not radius-driven.  The near-vertex region --
where the uniform Pearcey form is at its best -- now serves.  Measured at
this build over 50 served near-cusp sources: *every one* violates the old
bound, and the minimum image-theta offset of a served source is 0.0.  The
old coverage bound is not merely loose, it is false everywhere it used to
be checked, so the classes that asserted it are deleted rather than
retuned (see the RETIRED block at the bottom of this file).

WHAT THIS SUITE ASSERTS NOW
---------------------------
Value claims about the SHAPE of the served region and the ACCURACY of what
is served -- never a re-pin of the routing predicate itself:

  1. The served band in ``w`` is UPWARD-CLOSED: once the arm serves a node
     it serves it at every higher frequency.  (Where the flip happens is
     pinned once, in `test_lensing_airy_fold`; this asserts the band has
     no holes, which that pin cannot see.)
  2. The served region is CONTIGUOUS in both source coordinates -- inward
     depth from the vertex and transverse offset -- so the arm covers a
     connected neighbourhood rather than scattered islands.
  3. The EXACT cusp vertex is refused at every frequency, up to ``w =
     1000``.  This survives F074 and is not the frequency floor doing the
     work: the vertex sits on the caustic, where the arm's local normal
     form does not exist.
  4. The served VALUE beats ``envelope_bar`` against an independent exact
     oracle.

The oracle in (4) is the point of the F074 floor.  The arm serves from
``w = (_K_UNIFORM / envelope_bar)^2 = 49`` upward; the exact double-double
Schwinger engine runs for ``w <= W_CEILING_SCHWINGER = 60``.  Those two
windows now OVERLAP in ``w`` in [49, 60], so `operator.F_op` there returns
the exact wave value without ever consulting the uniform arms (they are
offered only above 60), giving a cheap, genuinely independent oracle.
Pre-F074 the arm refused below ~49 and no such overlap existed.

Tolerance rationale:
  The served value is compared against the exact engine at ``envelope_bar``
  (0.05) -- the arm's own advertised error budget, not a fitted number.
  Measured worst case over the fixture set: 0.0473 (gamma = 0.2, dp = 0.01,
  w = 60), i.e. the assertion runs with ~6% headroom and real teeth.

Cost estimate:
  `cusp_amplification` is ~5-20 ms; an exact `F_op` node at ``w <= 60`` is
  ~0.3 s.  Sixteen oracle comparisons (~5 s) plus the scan tests (~3 s)
  keep the file under ~10 s.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from cogwheel.lensing.chang_refsdal import _pearcey_cusp, _schwinger, geometry
from cogwheel.lensing.chang_refsdal._pearcey_cusp import (
    cusp_amplification,
    use_pearcey_table,
)
from cogwheel.lensing.chang_refsdal.operator import (
    F_op,
    geometric_amplification,
)


# ---------------------------------------------------------------------------
# Constants -- derived from production, never pinned as literals
# ---------------------------------------------------------------------------

#: The arm's subleading-uniform error constant and its envelope budget.
_K_UNIFORM = _pearcey_cusp._K_UNIFORM
_ENVELOPE_BAR = _pearcey_cusp._DEFAULT_ENVELOPE_BAR

#: Lowest ``w`` at which the arm's own error estimate can clear the bar:
#: ``_K_UNIFORM / sqrt(w) <= bar``  =>  ``w >= (K / bar)^2``.  Used ONLY to
#: centre the scans below on the interesting region -- the serve/refuse flip
#: at this frequency is pinned once, in `test_lensing_airy_fold`.
_CUSP_UNIFORM_W_FLOOR = (_K_UNIFORM / _ENVELOPE_BAR) ** 2

#: Upper edge of the exact double-double Schwinger engine.  Together with
#: the floor above this brackets the overlap window in which `F_op` is an
#: independent exact oracle for the arm (the uniform arms are offered only
#: above ``W_CEILING_SCHWINGER_QD``, far above this).
_W_EXACT_CEILING = _schwinger.W_CEILING_SCHWINGER

#: Frequencies inside the overlap window used for the oracle comparison.
_ORACLE_WS = (55.0, _W_EXACT_CEILING)

#: Shear magnitudes spanning weak to strong (all positive-parity astroid).
_GAMMAS = (0.2, 0.3, 0.5, 0.7)

#: Phase of the cusp the fixtures sit on, in the shear-aligned frame.
_CUSP_AXIS_PHASE = 0.5 * math.pi

#: Fractional inward offsets from the cusp vertex for the oracle fixtures.
#: ``dp = 0.01`` is the tightest corner (worst measured error) and is kept
#: deliberately -- it is where the assertion has the most teeth.
_ORACLE_DPS = (0.01, 0.05)

#: Default inward offset for the single-node scans.
_SCAN_DP = 0.02

#: A frequency far above the floor, used to show the vertex refusal is not
#: the frequency gate in disguise.
_HIGH_W = 1000.0

#: Frequency for the region-shape scans (above the QD ceiling).
_SHAPE_W = 151.0

#: Minimum resolvable difference between the arm and the exact engine.  If
#: the two ever agreed to better than this the "accuracy" comparison would
#: be a self-comparison rather than an independent check.
_ORACLE_DISTINCTNESS_MIN = 1e-3


def _cusp_vertex_source(gamma: float, phase: float = _CUSP_AXIS_PHASE,
                        beta: float = 0.0, kappa: float = 0.0,
                        branch: int = 1) -> np.ndarray:
    """The source at the cusp vertex itself (exactly on the caustic)."""
    cusp = geometry.critical_point(gamma, phase + beta, beta, kappa, branch)
    return np.asarray(cusp.source, dtype=float)


def _near_cusp_source(gamma: float, phase: float = _CUSP_AXIS_PHASE, *,
                      beta: float = 0.0, kappa: float = 0.0,
                      dp: float = _SCAN_DP, dperp: float = 0.0,
                      branch: int = 1) -> np.ndarray:
    """A source ``dp`` (as a fraction of the cusp-vertex distance) inward
    from the cusp vertex, plus ``dperp`` along the hard axis.

    Mirrors the fixture builder the F074 rewrite installed in
    `test_lensing_airy_fold`: the uniform Pearcey form is a local expansion
    about the vertex, so the arm's home -- and this suite's fixtures -- live
    here.  Everything is derived from `geometry.critical_point`, so the
    fixtures track the caustic if the geometry moves.
    """
    cusp = geometry.critical_point(gamma, phase + beta, beta, kappa, branch)
    return ((1.0 - dp) * np.asarray(cusp.source, dtype=float)
            + dperp * np.asarray(cusp.hard_axis, dtype=float))


def _serves(w: float, source: np.ndarray, gamma: float) -> bool:
    """Whether the cusp arm certifies a finite value at this node."""
    return cusp_amplification(float(w), source, gamma) is not None


def _runs(flags: list[bool]) -> int:
    """Number of maximal runs of ``True`` in ``flags`` (island count)."""
    return sum(1 for i, f in enumerate(flags)
               if f and (i == 0 or not flags[i - 1]))


# ---------------------------------------------------------------------------
# Module fixtures: install the Pearcey table, and PUT IT BACK
# ---------------------------------------------------------------------------

#: Global Pearcey-table state saved by `setUpModule`, restored by
#: `tearDownModule`.
_SAVED_PEARCEY_TABLE = None


def setUpModule() -> None:
    """Install the tabulated Pearcey primitive for this file only.

    The table approximates the direct quadrature to ~6e-7 relative -- far
    inside every tolerance asserted here -- and makes the scans below much
    cheaper.  But `_pearcey_cusp` holds it in a MODULE GLOBAL, so it must
    be put back: leaving it installed silently changes what
    `cusp_amplification` returns for every later test in the same worker
    process.  Measured: the byte-identity goldens in
    `test_lensing_airy_fold` are frozen against the direct quadrature, and
    they go red by exactly that 6e-7 whenever xdist happens to schedule
    this file onto their worker first.  Save/restore makes the file's
    result independent of scheduling.
    """
    global _SAVED_PEARCEY_TABLE
    _SAVED_PEARCEY_TABLE = _pearcey_cusp.get_pearcey_table()
    use_pearcey_table()


def tearDownModule() -> None:
    """Restore the global Pearcey-table state (see `setUpModule`)."""
    _pearcey_cusp.set_pearcey_table(_SAVED_PEARCEY_TABLE)


# ---------------------------------------------------------------------------
# Helper base
# ---------------------------------------------------------------------------

class _CuspArmCoverageTestCase(unittest.TestCase):
    """Base carrying the anti-vacuity tearDown."""

    def setUp(self) -> None:
        self.n_checks = 0

    def tearDown(self) -> None:
        self.assertGreater(
            self.n_checks, 0,
            'Anti-vacuity: zero comparisons ran — the test is silently empty.')


# ---------------------------------------------------------------------------
# Test 1: the served band in w has no holes
# ---------------------------------------------------------------------------

class CuspArmWBandTestCase(_CuspArmCoverageTestCase):
    """The served band in ``w`` is upward-closed -- no holes above the floor.

    F074's gate is monotone in ``w`` by construction (``_K_UNIFORM /
    sqrt(w)`` falls, and the ghost term is exactly zero on the interior),
    but the certificate and normal-form steps that follow it are not
    obviously monotone in anything.  This asserts the OBSERVED band shape:
    once the arm serves a fixed node it never stops.  The location of the
    flip is deliberately NOT asserted here -- that predicate has one home,
    in `test_lensing_airy_fold`.
    """

    def test_serve_band_is_upward_closed_in_w(self) -> None:
        """Once served at a node, served at every higher ``w``."""
        w_grid = list(np.linspace(0.4 * _CUSP_UNIFORM_W_FLOOR,
                                  8.0 * _CUSP_UNIFORM_W_FLOOR, 24))
        for gamma in _GAMMAS:
            source = _near_cusp_source(gamma)
            with self.subTest(gamma=gamma):
                flags = [_serves(w, source, gamma) for w in w_grid]
                # Anti-vacuity: the scan must straddle the transition, or
                # upward-closure is trivially true.
                self.assertTrue(any(flags),
                                f'gamma={gamma}: arm never served on the '
                                f'scan — fixture is stale')
                self.assertFalse(all(flags),
                                 f'gamma={gamma}: arm served everywhere on '
                                 f'the scan — scan no longer straddles the '
                                 f'transition')
                self.n_checks += 1
                first = flags.index(True)
                self.assertTrue(
                    all(flags[first:]),
                    f'gamma={gamma}: served band has a hole — first served '
                    f'at w={w_grid[first]:.3f}, but the band above it is '
                    f'{"".join("S" if f else "." for f in flags[first:])}')


# ---------------------------------------------------------------------------
# Test 2: the served region is a connected neighbourhood of the vertex
# ---------------------------------------------------------------------------

class CuspArmRegionShapeTestCase(_CuspArmCoverageTestCase):
    """The arm covers a CONNECTED neighbourhood of the cusp, and the vertex
    itself is excluded from it at every frequency.

    Contiguity is the property the deleted coverage bound was reaching for
    -- "the arm serves an outer region and refuses an inner one" -- stated
    as something still true: whatever the served set is, it is one piece,
    not scattered islands.  An arm that served on a speckled set would be
    routing on noise.
    """

    def test_served_region_contiguous_in_depth(self) -> None:
        """Sources swept inward from the vertex serve on ONE interval."""
        dps = list(np.linspace(0.0, 0.4, 41))
        for gamma in _GAMMAS:
            with self.subTest(gamma=gamma):
                flags = [_serves(_SHAPE_W, _near_cusp_source(gamma,
                                                             dp=float(dp)),
                                 gamma)
                         for dp in dps]
                self.assertTrue(any(flags),
                                f'gamma={gamma}: no depth served — stale')
                self.n_checks += 1
                self.assertEqual(
                    _runs(flags), 1,
                    f'gamma={gamma}: served depths are not contiguous — '
                    f'{"".join("S" if f else "." for f in flags)}')

    def test_served_region_contiguous_in_transverse_offset(self) -> None:
        """Sources swept across the cusp axis serve on ONE interval."""
        offsets = list(np.linspace(-0.2, 0.2, 41))
        for gamma in _GAMMAS:
            with self.subTest(gamma=gamma):
                flags = [_serves(_SHAPE_W,
                                 _near_cusp_source(gamma, dperp=float(o)),
                                 gamma)
                         for o in offsets]
                self.assertTrue(any(flags),
                                f'gamma={gamma}: no offset served — stale')
                # Teeth: the sweep must also leave the served region, or
                # "one run" says nothing.
                self.assertFalse(all(flags),
                                 f'gamma={gamma}: every offset served — the '
                                 f'sweep no longer leaves the region')
                self.n_checks += 1
                self.assertEqual(
                    _runs(flags), 1,
                    f'gamma={gamma}: served offsets are not contiguous — '
                    f'{"".join("S" if f else "." for f in flags)}')

    def test_exact_cusp_vertex_refused_at_every_frequency(self) -> None:
        """The vertex source is refused from the floor up to ``w = 1000``.

        Survives F074 unchanged, and the high-``w`` end proves it is not the
        frequency gate doing the work: at ``w = 1000`` the error estimate is
        ~0.011, comfortably under the bar, and the node is still refused --
        the vertex lies ON the caustic, where the arm's local normal form
        does not exist.
        """
        for gamma in _GAMMAS:
            source = _cusp_vertex_source(gamma)
            for w in (_CUSP_UNIFORM_W_FLOOR, _SHAPE_W, _HIGH_W):
                with self.subTest(gamma=gamma, w=w):
                    self.n_checks += 1
                    self.assertIsNone(
                        cusp_amplification(float(w), source, gamma),
                        f'gamma={gamma}: the exact cusp vertex was served '
                        f'at w={w}')


# ---------------------------------------------------------------------------
# Test 3: what is served is as accurate as the gate promises
# ---------------------------------------------------------------------------

class CuspArmServedAccuracyTestCase(_CuspArmCoverageTestCase):
    """The served value beats ``envelope_bar`` against the exact engine.

    This is the claim the F074 gate makes and the one the deleted coverage
    bound could not make: the gate bounds the error of what is SERVED, so
    the served value must actually land inside that bound.  The oracle is
    `operator.F_op` at ``w`` in the overlap window [49, 60], where it runs
    the exact double-double Schwinger engine and never consults the uniform
    arms.
    """

    def _oracle_fixtures(self):
        for gamma in _GAMMAS:
            for dp in _ORACLE_DPS:
                source = _near_cusp_source(gamma, dp=dp)
                for w in _ORACLE_WS:
                    yield gamma, dp, w, source

    def test_served_value_beats_envelope_bar_against_exact_engine(
            self) -> None:
        """Every served node agrees with the exact engine within the bar."""
        worst = 0.0
        for gamma, dp, w, source in self._oracle_fixtures():
            arm = cusp_amplification(w, source, gamma)
            self.assertIsNotNone(
                arm,
                f'gamma={gamma} dp={dp} w={w}: fixture is inside the overlap '
                f'window but the arm refused — fixture is stale')
            exact = complex(F_op(w, source, gamma)[0])
            rel = abs(complex(arm) - exact) / abs(exact)
            worst = max(worst, rel)
            with self.subTest(gamma=gamma, dp=dp, w=w):
                self.n_checks += 1
                self.assertLessEqual(
                    rel, _ENVELOPE_BAR,
                    f'gamma={gamma} dp={dp} w={w}: served value is '
                    f'{rel:.4g} from the exact engine, over the arm\'s own '
                    f'envelope_bar={_ENVELOPE_BAR}')
        # Teeth: the bound must not be passing because the two agree to
        # machine precision (that would mean F_op served the arm's own
        # value, making this a self-comparison, not an oracle check).
        self.n_checks += 1
        self.assertGreater(
            worst, _ORACLE_DISTINCTNESS_MIN,
            f'worst arm-vs-exact disagreement is {worst:.4g} — the "exact" '
            f'oracle is returning the arm\'s own value, so this test is a '
            f'self-comparison')

    def test_oracle_is_the_wave_engine_not_the_geometric_sum(self) -> None:
        """`F_op` in the overlap window is the exact wave value, not ppGO.

        Near the cusp the stationary-phase (ppGO) sum diverges -- the images
        are nearly degenerate -- so if `F_op` were serving the geometric
        branch here the comparison above would be meaningless.  Measured:
        the geometric value is O(1e3)-O(1e4) relative away, versus the arm's
        ~0.04.
        """
        w = _ORACLE_WS[0]
        for gamma in _GAMMAS:
            source = _near_cusp_source(gamma)
            exact = complex(F_op(w, source, gamma)[0])
            geometric = complex(geometric_amplification(w, source, gamma))
            with self.subTest(gamma=gamma):
                self.n_checks += 1
                self.assertGreater(
                    abs(exact - geometric) / abs(exact), 1.0,
                    f'gamma={gamma}: F_op agrees with the divergent ppGO sum '
                    f'at a near-cusp node — it is not the wave engine')


# ---------------------------------------------------------------------------
# RETIRED (F074 re-derivation, 2026-08-13)
# ---------------------------------------------------------------------------
# The following classes asserted the PRE-F074 served/refused band structure
# and are deleted rather than retuned, because their subject -- the
# `radius >= (c_P/bar)^(2/3)` gate and the angular coverage bound it
# implied -- no longer exists in production:
#
# * `CoverageConstantTestCase` -- asserted only that the cusp-arm coverage
#   constant is a 2-decimal number in (0, 1).  No oracle, and the derivation
#   it documented ("minimum image-theta offset at which cusp_amplification
#   serves") is now false: measured minimum over served near-cusp sources
#   is 0.0.
#
# * `CuspVertexRefusalTestCase::test_near_vertex_offsets_refused` --
#   asserted that small perturbations of the vertex are refused.  F074
#   inverts this by design: the near-vertex region is where the uniform
#   form is best, and it now serves for `w >= (K/bar)^2`.  Its sibling
#   `test_cusp_vertex_source_refused` survives, rewritten and strengthened
#   as `CuspArmRegionShapeTestCase::
#   test_exact_cusp_vertex_refused_at_every_frequency`.
#
# * `ServedSourceCoverageTestCase` (both methods) -- asserted every served
#   source sits at least the cusp-arm coverage margin away in image-theta.
#   Measured at this build: all 50 served near-cusp sources violate it.  Not
#   retunable -- the arm's admission no longer reads an angle at all.
#
# * `TransitionMonotonicityTestCase::
#   test_refused_then_served_crosses_coverage` -- the same dead bound,
#   applied to the first served angle.  Its sibling
#   `test_served_band_contiguous` asserted a real property (the served set
#   is one piece) on a fixture ray that no longer serves anywhere; the
#   property is kept and generalised in `CuspArmRegionShapeTestCase`, and
#   extended to `w` in `CuspArmWBandTestCase`.
#
# * `SelfFalsificationTestCase` -- proved a deliberately inflated coverage
#   constant would trip the (now dead) bound.  Falsifying a deleted
#   assertion has no teeth; the suite's teeth now live in the anti-vacuity
#   straddle assertions above and in
#   `CuspArmServedAccuracyTestCase::test_oracle_is_the_wave_engine_not_the_
#   geometric_sum`, which shows the accuracy oracle is a genuinely
#   different object from the arm.
#
#
# ---------------------------------------------------------------------------
# RETIRED (F079 constant removal, 2026-08-14)
# ---------------------------------------------------------------------------
# The F074 note above recorded the coverage constant as "still live in
# `surrogate.py` as a tube-window shrink".  It is no longer: both the
# positive-parity and saddle-parity cusp-arm coverage constants have been
# DELETED from `surrogate.py` in this build.  They were measured dead for
# two independent reasons:
#
#   (a) WRONG UNITS.  The constant was subtracted, in `_tube_serves`, from
#       the chart's cusp-exclusion window half-width.  But the window is a
#       critical-curve PARAMETER angle, while the coverage constant was
#       calibrated as an image-plane POLAR offset.  Subtracting one from the
#       other mixed two different angular coordinates -- the "correction"
#       had no defined meaning for its consumer.
#
#   (b) ZERO BEHAVIOURAL EFFECT.  Post-F074, sweeping the constant over its
#       whole range changed the serve/refuse decision on 0 of 64 production
#       tube windows: the eta-floor and w-floor gates already decide every
#       cusp-region query, so the angular shrink never flipped an outcome it
#       was the last word on.
#
# `_tube_serves` now excludes the tube over the FULL cusp-exclusion window
# (the chart's own `(theta_cusp, delta_theta)` schema, used with no angular
# arm-coverage subtraction); a query inside the window falls through to the
# serving ladder (Pearcey arm, then the exact engine).  There is
# consequently NO live constant left for this suite to reference, and the
# monkey-patch-window pin the F074 note pointed at (in
# `test_lensing_surrogate_training`) is retired with the constant.  This
# file is deliberately kept free of the deleted identifiers so the
# post-removal grep stays clean.


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
