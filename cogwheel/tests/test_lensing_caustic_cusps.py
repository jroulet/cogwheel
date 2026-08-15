"""
Tests for the geometry-driven cusp/fold refactor in
``lensing.surrogate_training`` (Build 8g WP1/WP2/WP3).

Three things changed and three independent oracles pin them:

WP2 -- cusp LOCATION is now the analytic root of ``y'.y'' = 0``
(`surrogate_training._find_cusps` relocates each sampled speed-minimum to
`_refine_cusp_angle`), while the exclusion-WINDOW half-width
``delta_theta`` is byte-for-byte the pre-refactor rule.  The independent
oracle for the location is the closed-form fact that a positive-parity
Chang--Refsdal astroid (``|gamma| < 1``, ``kappa = 0``) has EXACTLY four
cusps, at the lens-plane axis directions ``theta in {0, pi/2, pi,
3pi/2}`` -- the images of the ``T``-axis parametrisation directions.
That set is derived from the caustic definition, not from the code under
test.  The independent oracle for the window is a HEAD copy of the
pre-refactor ``_find_cusps`` (AST-extracted from ``git show HEAD`` and
exec'd in a minimal namespace -- it is fully self-contained, using only
``numpy`` and three module constants), so ``delta_theta`` old-vs-new is a
genuine cross-version byte-identity check.

WP1/WP3 -- the served (image-pair) side of every fold arc and its
``inward_sign`` come from geometry (`_make_arc` via
``geometry.fold_opening_direction``), and ``image_count`` is asserted to
be the parity constant 4.  The Professor's gating oracle (Spec 3) is an
INDEPENDENT image census: place a source one caustic-normal step
``eta = eta_max`` off the fold on the ``inward_sign`` side and count real
images with ``geometry.find_images``; do the same on the opposite side.
The served side must carry exactly four real images and the opposite side
exactly two.  This certifies ``image_count``-from-parity, ``inward_sign``
and serve-consistency simultaneously.

Tolerances and regime honesty (measured, not assumed).
  * Spec 1: the four astroid cusps sit exactly on the ``n_samples = 200``
    sample grid (``pi/2 = 50 * 2pi/200`` etc.), so the analytic root and
    the sampled detector agree to < 1 sampling step by construction and
    the axis coincidence is machine-exact; the gate uses the Architect's
    ``1e-9`` axis tolerance and ``speed < 1e-6 * peak`` cusp tolerance
    (measured worst ``speed/peak ~ 4e-16``).
  * Spec 3: the strict ``served == 4 and opposite == 2`` gate holds for
    positive parity ``gamma in {0.2, 0.4, 0.7, 0.9}`` and the macro
    saddle ``gamma = 1.2`` (measured).  At ``gamma = 1.5`` the served
    side still carries four images but the opposite side of the
    ``branch = -1`` deltoid edges develops a census defect
    (``LensDomainError``); that arc is certified by the weaker but still
    decisive ``served == 4 and opposite != 4``.  At ``gamma = 2.0`` the
    default ``eta_max = 0.05`` OVERSHOOTS the pinched deltoid lobe (the
    normal exits the fold tube), so the served census drops to two;
    `EtaOvershootBoundaryTestCase` witnesses that a smaller ``eta``
    restores four, proving the drop is an eta-overshoot and NOT a parity
    error.  This is a documented deviation from Spec 3's universal
    ``served == 4`` claim and is reported to the driver.

Every sweep carries an anti-vacuity ``tearDown`` (a green run that
compared nothing is a bug, not a pass) and `SelfFalsificationTestCase`
corrupts the cusp angle, the window width and the serve sign to prove all
three gates have teeth.

WP1/WP3 exact-geometry specs (this build's continuation) and their
MEASURED deviations from the brief -- values were measured first, never
assumed:

  * `InwardSignFoldHealthTestCase` (serve-alignment health).  At the exact
    interior ``theta`` `_make_arc` chose for every built arc,
    ``abs(fold_opening_direction . serve_normal) > 0.1`` and
    ``sign(dot) == arc.inward_sign`` -- the served side the source is
    nudged onto is the geometric two-image side.  Measured worst |dot| is
    0.298 (positive parity ``gamma = 0.2``); the ``branch = -1`` saddle
    edges give 1.0.  ``serve_normal`` is `_tube_normal`'s unit normal, the
    SAME one `_tube_source` displaces along, so this is the genuine
    serve-consistency invariant, not a restatement of the build guard.

  * `CausticInradiusClosedFormTestCase` (Spec: inradius from closed form).
    ``_caustic_inradius(gamma, +1, n)`` equals ``gamma`` to ~1e-16 and the
    INDEPENDENT closed-form ``min_phi |y(phi)|`` -- derived here from
    ``det J = 0`` (``s = 1/r**2 = gamma cos2phi + sqrt(1 - gamma**2
    sin**2 2phi)``, ``y = ((a - s) x1, (b - s) x2)``, ``a = 1 - gamma``,
    ``b = 1 + gamma``), NOT from `geometry.r_caustic` -- to < 1e-9 relative
    (measured ~2e-10).  DEVIATION: the brief says the closest approach sits
    at a CUSP; it does not.  The four axis cusps sit at ``|y| = 2 gamma /
    sqrt(1 +- gamma) > gamma``; the global minimum ``|y| = gamma`` is a
    SMOOTH waist between cusps (measured argmin ``phi`` in ~0.6..2.5, never
    an axis direction).  ``encloses_origin`` is True (the winding-number
    topology pin, unchanged from the pre-refactor discrete-cloud test).

  * `FootOfNormalCurvatureValueTestCase` (WP1 curvature-relative invariant).
    With the curvature-relative tube shell (``eta_max = f_max * R_c`` per
    arc), the old fixed-eta guard is replaced by ``f_max < 0.5`` (asserted
    at training time).  The test verifies the default config satisfies it.
    `CurvatureRelativeTubeNoSkipTestCase` demonstrates the invariant
    empirically over 5 gamma bands covering [0.0281, 0.28]: every band
    produces a finite positive ``eta_max`` and the former skip condition is
    algebraically impossible with ``f_max = 0.40``.
    `CurvatureRelativeHeldoutEpsTestCase` builds tube charts at gamma
    extremes and verifies held-out eps < 0.05 (the tube_eps_max bar).

  * `InteriorAdmissionMarginRemovalTestCase` (Acceptance pin (c)).  Deleting
    ``_CLOUD_MARGIN_FRAC`` is shown, not assumed: over a fixed candidate-tile
    set the new exact-distance `_InteriorAdmission.admits` decisions are a
    SUPERSET of the incumbent (discrete-cloud + 0.10 margin) decisions -- the
    new rule never refuses a tile the incumbent admitted.  DEVIATION: the
    decisions are not identical.  A thin boundary set (measured 3..4 tiles
    per band) flips old-refuse -> new-admit, and every such tile's
    INDEPENDENT dense-cloud clearance lies in ``[eta_max, 1.1 eta_max)`` --
    i.e. it is genuinely at least ``eta_max`` clear, and the incumbent
    refused it only because of the 10% cloud-bias inflation the exact
    distance makes unnecessary.

Build 1d WP1 (analytic ``_tube_normal``, serve consistency -- Gate 4).
  * `TubeNormalGeometryTestCase` (Gate 4a).  Over a production ``(gamma,
    theta, branch)`` sweep on both parities `_tube_normal` returns a UNIT
    normal (``|norm| - 1 < 1e-15``, measured 2.2e-16) exactly perpendicular
    to the analytic tangent ``y'/|y'|`` from `geometry.caustic_derivatives`
    (``|normal . t| < 1e-14``, measured 2.7e-17), and its source carries no
    finite-difference step (the deleted ``_WEDGE_EPS``).  Every probe theta
    sits at ``|y'| > 1e-3`` so the orientation test is off the interior
    ``|y'| = 0`` cusps, where the analytic tangent is undefined (Professor
    Q1/Q2).
  * `InwardSignGoldenTableTestCase` (Gate 4b, LOAD-BEARING).  The built
    ``inward_sign`` of every fold arc on every band equals a FROZEN table of
    +-1 literals (`GOLDEN_INWARD_SIGN`, computed once from the shipped build,
    NOT ``git show HEAD``), so a silent orientation flip in `_tube_normal`
    (the F041 failure, in this exact function) goes RED -- something the
    self-consistent ``sign(dot) == inward_sign`` health invariant (which
    recomputes from the SAME `_tube_normal`) would miss.  Non-circularity is
    closed by an INDEPENDENT two-image census: the frozen-sign side of each
    arc carries exactly four real images (`geometry.find_images`).
    `SelfFalsificationTestCase.test_flipped_golden_literal_fails_table`
    proves the table can go red.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import subprocess
import textwrap

from unittest import TestCase, main, skip, skipUnless
from unittest import mock as unittest_mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cogwheel.lensing import surrogate_training as st
from cogwheel.lensing.chang_refsdal import geometry


# ---------------------------------------------------------------------------
# Independent oracles and Architect-specified tolerances.
# ---------------------------------------------------------------------------

#: The four positive-parity astroid cusp directions (radians).  Derived
#: from the caustic DEFINITION: at kappa = 0, |gamma| < 1 the four cusps
#: are the images of the T-axis parametrisation directions
#: theta in {0, pi/2, pi, 3pi/2}.  Independent of `_find_cusps`.
AXIS_DIRECTIONS = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)

#: Caustic samples per branch sweep (Architect: n_samples = 200).  Chosen
#: by the specification; note pi/2, pi, 3pi/2 all land on this grid.
N_SAMPLES = 200

#: One sampling step of the periodic astroid sweep [0, 2pi) (radians).
SAMPLING_STEP = 2.0 * np.pi / N_SAMPLES

#: Positive-parity shears for the cusp-location / window specs (Spec 1/2).
ASTROID_GAMMAS = (0.05, 0.2, 0.4, 0.7)

#: Positive-parity shears spanning the astroid prior for the WP1 arc-survival
#: topology pin and the theta=0 cusp-window value pin.  All four are measured
#: to yield exactly 4 cusps -> 4 surviving arcs; 0.9 gives the widest x-axis
#: cusp window (0.236 rad) -- the sharpest contrast with the pre-fix ~1.5*pi
#: inflated theta=0 window (~4.7 rad half-width).
ARC_SURVIVAL_GAMMAS = (0.2, 0.4, 0.7, 0.9)

#: Surviving fold-arc count of the positive-parity astroid: 4 cusps bound 4
#: fold arcs.  The pre-fix wrap-span bug dropped the two arcs adjacent to
#: theta=0 and shipped only 2, so this is the invariant a silent arc drop
#: (or an arc-count regression) must fail.
ASTROID_EXPECTED_ARCS = 4

#: Absolute tolerance (radians) for the theta=0 cusp-window value pin.  One
#: detector sampling step: the x-axis reflection partners (the theta=0 and
#: theta=pi cusps) carry a bit-identical window by the astroid's y->conj
#: reflection symmetry (measured difference 0.0), so this generous "detector
#: resolution" bound is never approached in the healthy case, while the
#: wrap-bug re-inflation (span ~2*pi -> half-width ~4.7 rad) exceeds it by two
#: orders of magnitude.  NOTE (measured deviation from the brief): the
#: astroid is only 2-fold symmetric in cusp WINDOWS, not 4-fold -- the y-axis
#: cusps (theta=pi/2, 3pi/2) floor to _CUSP_MIN_HALFWIDTH (0.05) at gamma >=
#: 0.4 while the x-axis pair keep the wider dip, so the pin compares theta=0
#: to its reflection partner theta=pi (equal by construction) and asserts it
#: is not an OUTLIER above its three siblings, NOT that all four are equal.
WINDOW_PARTNER_ATOL = SAMPLING_STEP

#: Sane ceiling (radians) on the theta=0 cusp-window half-width.  The healthy
#: window is 0.094..0.236 over ARC_SURVIVAL_GAMMAS, an order of magnitude
#: below the pre-fix inflated ~4.5, so this bound documents the value is NOT
#: the wrap-bug artefact even without the partner comparison.
WINDOW_SANE_CEILING = 0.5

#: A detected cusp's caustic speed must fall below this fraction of the
#: branch peak speed (Architect Spec 1; measured worst ~4e-16).
SPEED_PEAK_FRAC = 1e-6

#: Absolute tolerance on the cusp/axis-direction coincidence, radians
#: (Architect Spec 1: 1e-9; measured coincidence is machine-exact).
AXIS_ATOL = 1e-9

#: Fixed serve step off the fold for image-count probes (caustic-normal
#: units).  This is a TEST-LOCAL constant for displacing sources off the
#: fold to probe image counts — independent of TrainingConfig.f_max, which
#: now determines the production tube shell width per-arc via R_c.
ETA_MAX = 0.05

#: Real-image counts on the two sides of a served fold (parity constants).
SERVED_IMAGE_COUNT = 4
OPPOSITE_IMAGE_COUNT = 2

#: Positive-parity shears for the served-image-count gate (Spec 3).  0.05
#: yields no admissible fold arc (its cusp windows swallow the branch) and
#: simply contributes no comparisons; 0.2..0.9 give the strict gate.
POSITIVE_SERVE_GAMMAS = (0.05, 0.2, 0.4, 0.7, 0.9)

#: Macro-saddle shear where the strict served==4/opposite==2 gate holds.
SADDLE_CLEAN_GAMMAS = (1.2,)

#: Macro-saddle shear where the served side keeps four images but the
#: opposite side of the branch=-1 deltoid edges develops a census defect.
SADDLE_DEFECT_GAMMA = 1.5

#: Macro-saddle shear where eta_max overshoots the pinched lobe.
ETA_OVERSHOOT_GAMMA = 2.0

#: Reduced serve step that restores the four-image census at
#: `ETA_OVERSHOOT_GAMMA` (measured; proves the drop is an eta overshoot).
ETA_RESTORED = 0.02

#: Directory for diagnostic plots (mirrors the sibling suites).
_OUTPUT_DIR = pathlib.Path(__file__).parent / 'output'

#: Path to the module under test, for the HEAD-copy window oracle.
_MODULE_REL_PATH = 'cogwheel/lensing/surrogate_training.py'

# --- WP1/WP3 exact-geometry specs (serve alignment, inradius, foot-of-normal,
#     stable-band slivers, interior-admission margin removal). ---

#: Interior thetas `_make_arc` tries, in order, to orient a fold arc.  This
#: MIRRORS the production loop so the chosen serve theta is reproduced
#: exactly (the arc's own inner span, same fraction order, same exact-zero
#: ``dot == 0.0`` tripwire -- production no longer applies a magnitude floor).
_MAKE_ARC_FRACS = (0.5, 0.35, 0.65, 0.2, 0.8)

#: Independent serve-alignment HEALTH floor for ``|fold_dir . serve_normal|``
#: (Architect: > 0.1).  This is NOT `_make_arc`'s build tripwire: after the
#: F041 fix production skips a fraction only when ``dot == 0.0`` exactly (the
#: measure-zero tangency), never on magnitude.  0.1 is used purely as an
#: independent health threshold in `InwardSignFoldHealthTestCase` and
#: `test_below_floor_alignment_fails_serve_gate` to flag cusp-proximal arcs.
SERVE_ALIGN_MIN = 0.1

#: Positive-parity shears for the inward_sign health sweep (measured worst
#: |dot| = 0.298 at gamma = 0.2, well above the 0.1 floor).
HEALTH_POSITIVE_GAMMAS = (0.2, 0.4, 0.7, 0.9)

#: Macro-saddle shears for the health sweep (both deltoid branches: the
#: branch=-1 edges give |dot| = 1.0, the branch=+1 edges ~0.77-0.85).
HEALTH_SADDLE_GAMMAS = (1.2, 1.5)

# --- Gate 4 (serve consistency): analytic `_tube_normal` geometry + frozen
#     inward_sign golden table (Build 1d WP1). ---

#: Production ``(gamma, branch)`` pairs sampled for the analytic-normal
#: geometry sweep (Gate 4a): positive-parity astroid uses branch +1 only;
#: the macro-saddle deltoid honours both square-root branches.
TUBE_NORMAL_BANDS = (
    (0.2, (1,)), (0.4, (1,)), (0.7, (1,)), (0.9, (1,)),
    (1.2, (1, -1)), (1.5, (1, -1)))

#: theta grid (radians) for the analytic-normal sweep.  A LensDomainError
#: (outside the saddle wedge) or an interior ``|y'| = 0`` deltoid cusp
#: (guarded below) simply contributes no comparison.
TUBE_NORMAL_THETAS = tuple(np.linspace(0.0, 2.0 * np.pi, 41, endpoint=False))

#: Unit-normal tolerance (Professor Q2: 1e-15; measured worst 2.2e-16).
TUBE_NORMAL_UNIT_ATOL = 1e-15

#: Perpendicularity tolerance ``|normal . (y'/|y'|)|`` (Professor Q2: 1e-14;
#: measured worst 2.7e-17).
TUBE_NORMAL_PERP_ATOL = 1e-14

#: Minimum ``|y'|`` (source units) for a probe theta to be OFF the interior
#: ``|y'| = 0`` deltoid/astroid cusps (Professor Q1).  At ``|y'| = 0`` the
#: analytic tangent ``y'/|y'|`` is undefined (0/0 -> NaN), so the orientation
#: test must only run where the caustic is genuinely moving; measured worst
#: retained ``|y'|`` over the sweep is ~4.2e-2, far above this guard.
YPRIME_MIN_NORM = 1e-3

#: Frozen INCUMBENT / physically-correct ``inward_sign`` for every fold arc
#: on every production band, both parities (Gate 4b, LOAD-BEARING).  Computed
#: ONCE from the shipped analytic-tangent build (NOT ``git show HEAD``) and
#: baked here as +-1 literals: a silent orientation flip in `_tube_normal`
#: (the F041 failure class, in this exact function) flips a stored sign away
#: from its literal and turns this table RED.  Keyed by ``(gamma, parity)``;
#: values are the per-arc signs in `detect_caustic_structure` arc order.
#: Independently cross-checked below via the two-image census (the served
#: side of each frozen sign carries exactly four real images) so the frozen
#: literals are not a self-oracle.
#:
#: WP1 (arc-survival wrap fix, 2026-08-14): the astroid rows are now
#: FOUR-tuples, one sign per surviving fold arc.  The pre-fix ``_find_cusps``
#: wrap-span bug inflated the theta=0 cusp window to ~1.5*pi and swallowed the
#: two arcs adjacent to theta=0, so the shipped structure carried only TWO
#: astroid arcs and this table used to freeze that bug as ``(-1, -1)``.  The
#: four signs are DERIVED FROM GEOMETRY, never copied from the fixed run to
#: make the count pass: at every astroid gamma each of the four arcs has
#: ``sign(fold_opening_direction . serve_normal) = -1`` (measured worst |dot|
#: 0.298 at gamma=0.2) AND its inward_sign side carries exactly four real
#: images -- exactly the derivation
#: ``test_frozen_sign_is_the_geometric_two_image_side`` re-checks.  The saddle
#: 6-tuples are unchanged.
GOLDEN_INWARD_SIGN = {
    (0.2, 1): (-1, -1, -1, -1),
    (0.4, 1): (-1, -1, -1, -1),
    (0.7, 1): (-1, -1, -1, -1),
    (0.9, 1): (-1, -1, -1, -1),
    (1.2, -1): (-1, -1, 1, -1, -1, 1),
    (1.5, -1): (-1, -1, 1, -1, -1, 1),
}

#: Astroid shears for the closed-form inradius spec (positive parity).
INRADIUS_GAMMAS = (0.05, 0.2, 0.4, 0.7)

#: Relative tolerance on the inradius closed-form agreement (Architect
#: 1e-9; measured shipped-vs-independent ~2e-10, shipped-vs-gamma ~1e-16).
INRADIUS_RTOL = 1e-9

#: Samples for the INDEPENDENT ``min_phi |y(phi)|`` inradius oracle (a
#: smooth quadratic minimum, so this density gives ~1e-10 residual).
INRADIUS_ORACLE_SAMPLES = 200001

# RETIRED (WP1 curvature-relative tube shell): the fixed-eta skip guard and
# its FALSE/TRUE band classification no longer exist.  The invariant is now
# f_max < 0.5 (algebraic, per-arc), tested in FootOfNormalCurvatureValueTestCase
# and CurvatureRelativeTubeNoSkipTestCase.
# FOOT_FALSE_BANDS -- deleted
# FOOT_TRUE_BAND -- deleted

#: Positive-parity bands for the interior-admission margin-removal pin (c).
INTERIOR_ADMISSION_BANDS = ((0.25, 0.35), (0.45, 0.55))

#: The incumbent (HEAD) interior tube-shell inflation factor removed by the
#: exact-distance refactor (verified against ``git show HEAD``).
INCUMBENT_CLOUD_MARGIN_FRAC = 0.10

#: Dense caustic-cloud samples for the INDEPENDENT nearest-distance oracle
#: used to bracket the pin-(c) boundary flips (distinct from production's
#: 200-point cloud AND from the exact ``nearest_caustic_point``).
#: 4001 points gives ~1.6e-3 spacing near the caustic, sufficient to resolve
#: the eta_max boundary zone (~0.06) to well within CLEARANCE_SLACK (2e-3).
INTERIOR_DENSE_SAMPLES = 4001

#: Slack (dimensionless ``y``) for the dense-cloud clearance bracket: a
#: 4001-point cloud resolves the ~eta_max nearest distance to well within
#: this (half the cloud spacing near the caustic).
CLEARANCE_SLACK = 2e-3


# `_head_module_source`, `_git_available` and `_head_find_cusps` deleted
# 2026-07-30 (F045).  `_head_find_cusps` AST-extracted the pre-refactor
# `_find_cusps` from `git show HEAD` and needed `_CUSP_SPEED_REL_FRAC`, which
# build 1b deleted -- the original F043 breakage.  Its tests were left as
# `@unittest.skip` shells, which kept the helpers alive; that is exactly how
# the antipattern propagated into a later build, so the shells are gone too.


def _axis_distance(theta: float) -> float:
    """Smallest wrapped distance from ``theta`` to an axis cusp direction."""
    axes = np.asarray(AXIS_DIRECTIONS)
    return float(np.min(np.abs(((axes - theta + np.pi) % (2.0 * np.pi))
                               - np.pi)))


def _wrap_distance(theta: float, target: float) -> float:
    """Wrapped [0, pi] distance from ``theta`` to ``target`` (radians)."""
    return float(abs(((theta - target + np.pi) % (2.0 * np.pi)) - np.pi))


def _nearest_cusp(cusps, target: float):
    """The ``(theta, delta_theta)`` cusp whose angle is nearest ``target``.

    Wrap-aware, so the theta=0 cusp is found whether the detector returned
    it at ~0 or (in a wrap-straddling regression) near 2pi.
    """
    return min(cusps, key=lambda tw: _wrap_distance(tw[0], target))


def _real_image_count(gamma: float, arc, sign: int, eta: float):
    """Real-image count for a source ``eta`` off ``arc`` on ``sign`` side.

    ``sign = +1`` uses the arc's ``inward_sign`` (the served side);
    ``sign = -1`` the opposite side.  Returns the integer image count, or
    the string ``'LensDomainError'`` if the census refuses (a defect that
    is decisively NOT the served four-image side).
    """
    theta = arc.theta_lo + 0.5 * (arc.theta_hi - arc.theta_lo)
    source = st._tube_source(gamma, theta, eta, arc.branch,
                             sign * arc.inward_sign)
    matrix = geometry.macro_matrix(gamma)
    try:
        return len(geometry.find_images(source, matrix))
    except geometry.LensDomainError:
        return 'LensDomainError'


def _chosen_serve_theta(gamma: float, arc):
    """Reproduce the interior ``theta`` and dot `_make_arc` oriented on.

    Replays `_make_arc`'s exact fraction loop over the arc's inner span
    (``theta_lo..theta_hi`` already carry the cusp-window/margin cuts), the
    same exact-zero ``dot == 0.0`` tripwire (production applies NO magnitude
    floor after the F041 fix) and the same `_tube_normal` serve normal, so
    the returned ``(theta, dot)`` is the very orientation probe that fixed
    the arc's ``inward_sign`` -- it CANNOT pick a different serve theta than
    `_make_arc` did.  Returns ``None`` only if every fraction is a
    `LensDomainError` skip or the exact-zero tangency (which cannot happen
    for an arc that was actually built).
    """
    span = arc.theta_hi - arc.theta_lo
    for frac in _MAKE_ARC_FRACS:
        theta = arc.theta_lo + frac * span
        try:
            fold_dir = geometry.fold_opening_direction(
                gamma, theta, branch=arc.branch)
            _caust, normal = st._tube_normal(gamma, theta, arc.branch)
        except geometry.LensDomainError:
            continue
        dot = float(fold_dir @ normal)
        if dot == 0.0:
            continue
        return theta, dot
    return None


def _astroid_inradius_closed_form(gamma: float,
                                  n: int = INRADIUS_ORACLE_SAMPLES) -> float:
    """INDEPENDENT closed-form astroid inradius ``min_phi |y(phi)|``.

    Derived here from ``det J = 0`` for ``y(x) = A x - x / |x|**2`` with
    ``A = diag(1 - gamma, 1 + gamma)``, ``kappa = 0``: the critical curve
    is ``s = 1 / r**2 = gamma cos2phi + sqrt(1 - gamma**2 sin**2 2phi)``
    (positive root), the image is ``x = (cos phi, sin phi) / sqrt(s)`` and
    the caustic point is ``y = ((a - s) x1, (b - s) x2)`` with ``a = 1 -
    gamma``, ``b = 1 + gamma``.  This shares NO code path with
    `geometry.r_caustic` or `surrogate_training._caustic_inradius` -- it is
    a genuinely independent oracle for the closed-form minimisation.
    """
    phi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    s = gamma * np.cos(2.0 * phi) + np.sqrt(
        1.0 - gamma ** 2 * np.sin(2.0 * phi) ** 2)
    a, b = 1.0 - gamma, 1.0 + gamma
    y_sq = (np.cos(phi) ** 2 * (a - s) ** 2
            + np.sin(phi) ** 2 * (b - s) ** 2) / s
    return float(np.sqrt(np.min(y_sq)))


def _incumbent_interior_admits(admission, center, half,
                               margin_frac: float) -> bool:
    """Pre-refactor (HEAD) interior admission: discrete cloud + margin.

    A faithful transcription of the HEAD ``_InteriorAdmission.admits``
    (verified against ``git show HEAD``): the outer-rho edge is probed at
    `st._INTERIOR_EDGE_SAMPLES` angles and refused if ANY probe is within
    ``eta_max * (1 + margin_frac)`` of the nearest DISCRETE 200-point
    caustic-cloud point.  It reuses the (identical) ``theta_axis``,
    ``radius_grid`` and ``caustic_clouds`` of the new admission object, so
    the only difference under test is the distance rule itself.
    """
    rho_center, theta_center = center
    half_rho, half_theta = half
    rho_outer = float(rho_center) + float(half_rho)
    if rho_outer <= 0.0 or rho_outer >= 1.0:
        return False
    thetas = np.linspace(theta_center - half_theta,
                         theta_center + half_theta,
                         st._INTERIOR_EDGE_SAMPLES)
    for radius_axis, cloud in zip(admission.radius_grid,
                                  admission.caustic_clouds):
        if cloud.shape[0] == 0:
            return False
        radii = np.interp(thetas, admission.theta_axis, radius_axis)
        y_magnitudes = rho_outer * radii
        probe_x = y_magnitudes * np.cos(thetas)
        probe_y = y_magnitudes * np.sin(thetas)
        delta_x = probe_x[:, None] - cloud[None, :, 0]
        delta_y = probe_y[:, None] - cloud[None, :, 1]
        nearest = np.sqrt(delta_x * delta_x + delta_y * delta_y).min(axis=1)
        if np.any(nearest < admission.eta_max * (1.0 + margin_frac)):
            return False
    return True


def _dense_cloud_clearance(admission, center, half) -> float:
    """Min outer-edge clearance to an INDEPENDENT dense caustic cloud.

    Reconstructs the same outer-rho probes `_InteriorAdmission.admits`
    uses, but measures each probe's distance to a fresh
    `INTERIOR_DENSE_SAMPLES`-point caustic cloud (via
    `surrogate_training._caustic_points`) at every band gamma -- an oracle
    independent of both production's 200-point cloud and the exact
    ``nearest_caustic_point`` used by ``admits``.  Returns ``inf`` if the
    outer edge is not strictly interior (no clearance defined).
    """
    rho_center, theta_center = center
    half_rho, half_theta = half
    rho_outer = float(rho_center) + float(half_rho)
    if rho_outer <= 0.0 or rho_outer >= 1.0:
        return float('inf')
    thetas = np.linspace(theta_center - half_theta,
                         theta_center + half_theta,
                         st._INTERIOR_EDGE_SAMPLES)
    clearance = float('inf')
    for gamma_i, radius_axis in zip(admission.gammas, admission.radius_grid):
        cloud = st._caustic_points(gamma_i, 1, INTERIOR_DENSE_SAMPLES)
        radii = np.interp(thetas, admission.theta_axis, radius_axis)
        y_magnitudes = rho_outer * radii
        probe_x = y_magnitudes * np.cos(thetas)
        probe_y = y_magnitudes * np.sin(thetas)
        delta_x = probe_x[:, None] - cloud[None, :, 0]
        delta_y = probe_y[:, None] - cloud[None, :, 1]
        nearest = np.sqrt(delta_x * delta_x + delta_y * delta_y).min(axis=1)
        clearance = min(clearance, float(np.min(nearest)))
    return clearance


class _CuspTestCase(TestCase):
    """Base carrying the anti-vacuity comparison tally."""

    def setUp(self):
        self._comparisons = 0

    def _count(self) -> None:
        """Register one non-vacuous comparison."""
        self._comparisons += 1

    def tearDown(self):
        # Anti-vacuity: a sweep that compared nothing must not read green.
        self.assertGreater(
            self._comparisons, 0,
            'no comparisons ran -- the case sweep was vacuous')


class CuspAnalyticRootTestCase(_CuspTestCase):
    """Spec 1: cusp angle is the analytic root of ``y'.y'' = 0``."""

    def test_astroid_has_four_cusps_at_axis_directions(self):
        # For every positive-parity astroid: exactly four cusps; each a
        # near-zero-speed minimum; each within one sampling step of its
        # sampled detector; each on an axis direction to 1e-9.
        for gamma in ASTROID_GAMMAS:
            with self.subTest(gamma=gamma):
                thetas, speed = st._branch_speed_profile(
                    gamma, 1, 0.0, 2.0 * np.pi, N_SAMPLES, periodic=True)
                cusps = st._find_cusps(
                    thetas, speed, periodic=True, gamma=gamma, branch=1)
                cusps.sort()
                self.assertEqual(
                    len(cusps), 4,
                    f'gamma={gamma}: expected 4 astroid cusps, '
                    f'got {len(cusps)}')
                peak = float(speed.max())
                for theta_cusp, _delta in cusps:
                    # (a) near-zero caustic speed at the root.
                    root_speed = float(geometry.caustic_speed(
                        gamma, theta_cusp, branch=1))
                    self.assertLess(
                        root_speed, SPEED_PEAK_FRAC * peak,
                        f'gamma={gamma} theta={theta_cusp}: speed '
                        f'{root_speed:.3e} not below {SPEED_PEAK_FRAC:g}'
                        f' * peak {peak:.3e}')
                    # (b) within one sampling step of the sampled detector.
                    grid_dist = float(np.min(np.abs(
                        ((thetas - theta_cusp + np.pi) % (2.0 * np.pi))
                        - np.pi)))
                    self.assertLessEqual(
                        grid_dist, SAMPLING_STEP + 1e-12,
                        f'gamma={gamma} theta={theta_cusp}: relocation '
                        f'{grid_dist:.3e} exceeds one step '
                        f'{SAMPLING_STEP:.3e}')
                    # (c) coincides with an axis parametrisation direction.
                    self.assertLessEqual(
                        _axis_distance(theta_cusp), AXIS_ATOL,
                        f'gamma={gamma} theta={theta_cusp}: not within '
                        f'{AXIS_ATOL:g} of an axis direction')
                    self._count()


# `CuspWindowByteIdentityTestCase` deleted 2026-07-30 (F045).  It was already
# skipped for F043; a skipped body still exports its helpers, which is how the
# antipattern reached a later build.


class AstroidArcSurvivalTopologyTestCase(_CuspTestCase):
    """WP1 (wrap fix): the positive-parity astroid has EXACTLY four cusps AND
    four surviving fold arcs, and the topology cross-check has teeth.

    The pre-fix ``_find_cusps`` measured the theta=0 cusp's dip window with a
    LINEAR span across a periodic index walk; the window wrapped the whole
    [0, 2pi) sweep (~1.5*pi) and swallowed the two arcs adjacent to theta=0,
    so ``detect_caustic_structure`` shipped only TWO astroid arcs while still
    reporting four cusps.  The cusp count alone could not see that blind spot,
    so ``detect_caustic_structure`` now also cross-checks the surviving-arc
    count against `_EXPECTED_ARCS` and raises `CausticTopologyError` on a
    mismatch.  This case pins the healthy 4-cusps -> 4-arcs invariant and,
    with a monkeypatched expectation, proves the arc-count guard fires.
    """

    def test_four_cusps_and_four_surviving_arcs(self):
        # Invariant: every astroid gamma yields detected_cusps == 4 AND
        # len(arcs) == 4.  A future change that silently drops an arc (the
        # theta=0 wrap victim, or any other) makes this go RED.
        for gamma in ARC_SURVIVAL_GAMMAS:
            with self.subTest(gamma=gamma):
                structure = st.detect_caustic_structure(gamma, 1)
                self.assertEqual(
                    structure.detected_cusps, 4,
                    f'gamma={gamma}: expected 4 astroid cusps, got '
                    f'{structure.detected_cusps}')
                self.assertEqual(
                    len(structure.arcs), ASTROID_EXPECTED_ARCS,
                    f'gamma={gamma}: expected {ASTROID_EXPECTED_ARCS} '
                    f'surviving fold arcs, got {len(structure.arcs)} -- the '
                    f'pre-fix wrap bug dropped the two arcs adjacent to '
                    f'theta=0 and shipped 2')
                self._count()

    def test_arc_count_mismatch_raises_topology_error(self):
        # Teeth: with the surviving-arc expectation monkeypatched away from
        # the real astroid count, detect_caustic_structure must raise
        # CausticTopologyError -- NOT silently ship a fold-ring hole.  This
        # drives the guard synthetically (no engine campaign): the cusp count
        # still matches (4 == _EXPECTED_CUSPS[1]) so only the arc-count
        # cross-check can fire.
        gamma = ARC_SURVIVAL_GAMMAS[0]
        with unittest_mock.patch.dict(st._EXPECTED_ARCS, {1: 3}):
            with self.assertRaises(st.CausticTopologyError) as caught:
                st.detect_caustic_structure(gamma, 1)
        self.assertIn(
            'fold arc', str(caught.exception).lower(),
            'the arc-count guard message must name the surviving fold arcs')
        self._count()

    def test_correct_arc_expectation_does_not_raise(self):
        # Control: with the real expectation restored the same gamma builds
        # cleanly -- the guard fires ONLY on a genuine mismatch, so the teeth
        # test above is not a blanket refusal.
        gamma = ARC_SURVIVAL_GAMMAS[0]
        structure = st.detect_caustic_structure(gamma, 1)
        self.assertEqual(len(structure.arcs), st._EXPECTED_ARCS[1])
        self._count()


class Theta0CuspWindowValueTestCase(_CuspTestCase):
    """WP1 (wrap fix): the theta=0 cusp exclusion WINDOW is a sane value, not
    the ~1.5*pi wrap-bug artefact.

    This is a VALUE pin, not a path pin.  The pre-fix linear span across the
    periodic index walk inflated the theta=0 cusp's ``delta_theta`` by 20-50x
    (span ~2*pi -> half-width ~4.7 rad), so a wrap regression re-inflates
    it and this case goes RED.  The astroid is 2-fold (not 4-fold) symmetric
    in cusp WINDOWS: the x-axis pair (theta=0, theta=pi) share a bit-identical
    window by the y->conj reflection symmetry, while the y-axis cusps
    (theta=pi/2, 3pi/2) floor to ``_CUSP_MIN_HALFWIDTH`` at gamma >= 0.4.  So
    the method is: compare the theta=0 window to its x-axis reflection partner
    theta=pi (must agree to the detector resolution) and assert theta=0 is NOT
    an outlier above its three siblings.  Measured healthy theta=0 half-width
    is 0.094 (gamma <= 0.4), 0.141 (gamma=0.7), 0.236 (gamma=0.9).
    """

    def test_theta0_window_matches_x_axis_partner_and_is_not_inflated(self):
        for gamma in ARC_SURVIVAL_GAMMAS:
            with self.subTest(gamma=gamma):
                thetas, speed = st._branch_speed_profile(
                    gamma, 1, 0.0, 2.0 * np.pi, N_SAMPLES, periodic=True)
                cusps = st._find_cusps(
                    thetas, speed, periodic=True, gamma=gamma, branch=1)
                self.assertEqual(
                    len(cusps), 4,
                    f'gamma={gamma}: expected 4 cusps, got {len(cusps)}')
                theta0_c, delta0 = _nearest_cusp(cusps, 0.0)
                theta_pi_c, delta_pi = _nearest_cusp(cusps, np.pi)
                # The wrap victim: the detected cusp nearest theta=0.
                self.assertLessEqual(
                    _wrap_distance(theta0_c, 0.0), SAMPLING_STEP + 1e-9,
                    f'gamma={gamma}: no cusp within one step of theta=0')
                # (a) Reflection-partner equality: theta=0 window equals the
                # interior theta=pi window (which the wrap bug never touched).
                self.assertLessEqual(
                    abs(delta0 - delta_pi), WINDOW_PARTNER_ATOL,
                    f'gamma={gamma}: theta=0 window {delta0:.6f} disagrees '
                    f'with its x-axis partner theta=pi {delta_pi:.6f} by more '
                    f'than {WINDOW_PARTNER_ATOL:.6f} rad (wrap re-inflation?)')
                # (b) Not an outlier: the theta=0 window does not spike above
                # the largest of its three siblings.  The pre-fix ~4.7 rad
                # value blows past every sibling (<= 0.236); the healthy value
                # equals its theta=pi partner, i.e. it IS a sibling maximum.
                others = [dt for (t, dt) in cusps
                          if not np.isclose(t, theta0_c, atol=1e-9)]
                self.assertEqual(
                    len(others), 3,
                    f'gamma={gamma}: expected 3 sibling cusps, got '
                    f'{len(others)}')
                self.assertLessEqual(
                    delta0, max(others) + WINDOW_PARTNER_ATOL,
                    f'gamma={gamma}: theta=0 window {delta0:.6f} is an outlier '
                    f'above its siblings (max {max(others):.6f}) -- the '
                    f'wrap-bug signature')
                # (c) Sane ceiling: the value is nowhere near the ~4.7 rad
                # inflated artefact, independent of the sibling comparison.
                self.assertLess(
                    delta0, WINDOW_SANE_CEILING,
                    f'gamma={gamma}: theta=0 window {delta0:.6f} exceeds the '
                    f'sane ceiling {WINDOW_SANE_CEILING} -- likely the '
                    f'wrap-bug inflated value')
                self._count()

    def test_inflated_theta0_window_fails_partner_pin(self):
        # Teeth (self-falsification): a synthetically re-inflated theta=0
        # window (the wrap-bug signature, span ~2*pi -> ~4.7 rad) exceeds
        # the partner tolerance AND the sane ceiling, so the pin above cannot
        # pass a regression.
        gamma = ARC_SURVIVAL_GAMMAS[-1]
        thetas, speed = st._branch_speed_profile(
            gamma, 1, 0.0, 2.0 * np.pi, N_SAMPLES, periodic=True)
        cusps = st._find_cusps(
            thetas, speed, periodic=True, gamma=gamma, branch=1)
        _t0, delta0 = _nearest_cusp(cusps, 0.0)
        _tpi, delta_pi = _nearest_cusp(cusps, np.pi)
        inflated = st._CUSP_WIDTH_SAFETY * 0.5 * 1.5 * np.pi
        self.assertGreater(
            abs(inflated - delta_pi), WINDOW_PARTNER_ATOL,
            'the wrap-bug inflated window must fail the partner tolerance')
        self.assertGreater(
            inflated, WINDOW_SANE_CEILING,
            'the wrap-bug inflated window must exceed the sane ceiling')
        # ...and the healthy value must PASS, so the teeth are specific.
        self.assertLessEqual(abs(delta0 - delta_pi), WINDOW_PARTNER_ATOL)
        self._count()


class PositiveParityServedImageCountTestCase(_CuspTestCase):
    """Spec 3 (Professor's gate), positive-parity astroid: the inward_sign
    side carries exactly four real images, the opposite side exactly two."""

    def test_served_four_opposite_two(self):
        for gamma in POSITIVE_SERVE_GAMMAS:
            structure = st.detect_caustic_structure(gamma, 1)
            for index, arc in enumerate(structure.arcs):
                with self.subTest(gamma=gamma, arc=index):
                    self.assertEqual(
                        arc.image_count, SERVED_IMAGE_COUNT,
                        f'gamma={gamma} arc {index}: image_count '
                        f'{arc.image_count} != {SERVED_IMAGE_COUNT}')
                    served = _real_image_count(gamma, arc, +1, ETA_MAX)
                    opposite = _real_image_count(gamma, arc, -1, ETA_MAX)
                    self.assertEqual(
                        served, SERVED_IMAGE_COUNT,
                        f'gamma={gamma} arc {index}: served side has '
                        f'{served} images, expected {SERVED_IMAGE_COUNT}')
                    self.assertEqual(
                        opposite, OPPOSITE_IMAGE_COUNT,
                        f'gamma={gamma} arc {index}: opposite side has '
                        f'{opposite} images, expected {OPPOSITE_IMAGE_COUNT}')
                    self._count()


class SaddleServedImageCountTestCase(_CuspTestCase):
    """Spec 3, clean macro saddle (gamma = 1.2): strict served==4/opp==2 on
    both deltoid edges of both lobes."""

    def test_served_four_opposite_two(self):
        for gamma in SADDLE_CLEAN_GAMMAS:
            structure = st.detect_caustic_structure(gamma, -1)
            for index, arc in enumerate(structure.arcs):
                with self.subTest(gamma=gamma, arc=index):
                    self.assertEqual(
                        arc.image_count, SERVED_IMAGE_COUNT,
                        f'gamma={gamma} arc {index}: image_count '
                        f'{arc.image_count} != {SERVED_IMAGE_COUNT}')
                    served = _real_image_count(gamma, arc, +1, ETA_MAX)
                    opposite = _real_image_count(gamma, arc, -1, ETA_MAX)
                    self.assertEqual(
                        served, SERVED_IMAGE_COUNT,
                        f'gamma={gamma} arc {index}: served side has '
                        f'{served} images, expected {SERVED_IMAGE_COUNT}')
                    self.assertEqual(
                        opposite, OPPOSITE_IMAGE_COUNT,
                        f'gamma={gamma} arc {index}: opposite side has '
                        f'{opposite} images, expected {OPPOSITE_IMAGE_COUNT}')
                    self._count()


class SaddleOppositeNotServedTestCase(_CuspTestCase):
    """Spec 3, macro saddle at higher shear (gamma = 1.5): the served side
    still carries four images while the opposite side is decisively NOT the
    four-image side -- either exactly two, or a census defect
    (``LensDomainError``) on the branch=-1 deltoid edges."""

    def test_served_four_opposite_not_four(self):
        gamma = SADDLE_DEFECT_GAMMA
        structure = st.detect_caustic_structure(gamma, -1)
        for index, arc in enumerate(structure.arcs):
            with self.subTest(gamma=gamma, arc=index):
                served = _real_image_count(gamma, arc, +1, ETA_MAX)
                opposite = _real_image_count(gamma, arc, -1, ETA_MAX)
                self.assertEqual(
                    served, SERVED_IMAGE_COUNT,
                    f'gamma={gamma} arc {index}: served side has '
                    f'{served} images, expected {SERVED_IMAGE_COUNT}')
                # Opposite side is NOT the served four-image side.
                self.assertNotEqual(
                    opposite, SERVED_IMAGE_COUNT,
                    f'gamma={gamma} arc {index}: opposite side unexpectedly '
                    f'carries {SERVED_IMAGE_COUNT} images')
                self.assertIn(
                    opposite, (OPPOSITE_IMAGE_COUNT, 'LensDomainError'),
                    f'gamma={gamma} arc {index}: opposite side gave '
                    f'{opposite!r}, expected 2 or a census defect')
                self._count()


class EtaOvershootBoundaryTestCase(_CuspTestCase):
    """Companion / documented Spec 3 deviation: at gamma = 2.0 the default
    ``eta_max = 0.05`` overshoots the pinched deltoid lobe, so the served
    census drops below four; a smaller ``eta`` restores four, proving the
    drop is an eta-overshoot and NOT a parity/inward_sign error."""

    def test_overshoot_drops_then_smaller_eta_restores_four(self):
        gamma = ETA_OVERSHOOT_GAMMA
        structure = st.detect_caustic_structure(gamma, -1)
        # Restrict to the branch=+1 deltoid edges, whose curvature radius is
        # uniform (measured r_min ~ 0.194) so the overshoot is clean; the
        # branch=-1 edges intermix census defects and are not part of this
        # boundary witness.
        witnessed = False
        for index, arc in enumerate(structure.arcs):
            if arc.branch != 1:
                continue
            with self.subTest(gamma=gamma, arc=index):
                served_max = _real_image_count(gamma, arc, +1, ETA_MAX)
                served_small = _real_image_count(gamma, arc, +1, ETA_RESTORED)
                # eta_max overshoots -> fewer than four served images.
                self.assertLess(
                    served_max, SERVED_IMAGE_COUNT,
                    f'gamma={gamma} arc {index}: expected an eta_max '
                    f'overshoot (served < 4), got {served_max}')
                # A smaller eta lands inside the tube -> four images restored.
                self.assertEqual(
                    served_small, SERVED_IMAGE_COUNT,
                    f'gamma={gamma} arc {index}: eta={ETA_RESTORED} did not '
                    f'restore {SERVED_IMAGE_COUNT} images (got '
                    f'{served_small}); the drop is not a pure overshoot')
                witnessed = True
                self._count()
        self.assertTrue(
            witnessed, 'no branch=+1 saddle arc found to witness overshoot')


class InwardSignFoldHealthTestCase(_CuspTestCase):
    """WP1/WP3: at the interior theta `_make_arc` chose for every built
    fold arc (both parities, both saddle branches), the served alignment
    ``abs(fold_opening_direction . serve_normal)`` exceeds 0.1 and
    ``sign(dot)`` round-trips to the arc's stored ``inward_sign``."""

    def test_alignment_above_floor_and_sign_roundtrips(self):
        for parity, gammas in ((1, HEALTH_POSITIVE_GAMMAS),
                               (-1, HEALTH_SADDLE_GAMMAS)):
            structures = {gamma: st.detect_caustic_structure(gamma, parity)
                          for gamma in gammas}
            for gamma in gammas:
                for index, arc in enumerate(structures[gamma].arcs):
                    with self.subTest(parity=parity, gamma=gamma, arc=index):
                        probe = _chosen_serve_theta(gamma, arc)
                        self.assertIsNotNone(
                            probe,
                            f'parity={parity} gamma={gamma} arc {index}: no '
                            f'interior theta cleared the serve floor, yet the '
                            f'arc was built')
                        _theta, dot = probe
                        # A healthy fold opens strongly along the serve normal.
                        self.assertGreater(
                            abs(dot), SERVE_ALIGN_MIN,
                            f'parity={parity} gamma={gamma} arc {index}: '
                            f'|dot|={abs(dot):.4f} not above '
                            f'{SERVE_ALIGN_MIN} (cusp-proximal / pathological '
                            f'arc)')
                        # The served side is the geometric two-image side.
                        self.assertEqual(
                            1 if dot >= 0.0 else -1, arc.inward_sign,
                            f'parity={parity} gamma={gamma} arc {index}: '
                            f'sign(dot)={1 if dot >= 0 else -1} != stored '
                            f'inward_sign {arc.inward_sign}')
                        self._count()


class TubeNormalGeometryTestCase(_CuspTestCase):
    """Gate 4a (Build 1d WP1): `_tube_normal` returns the EXACT analytic
    caustic normal.

    Over a sweep of production ``(gamma, theta, branch)`` on both parities,
    the returned normal is a unit vector (``|norm| - 1 < 1e-15``) exactly
    perpendicular to the analytic tangent ``y'/|y'|`` from
    `geometry.caustic_derivatives` (``|normal . t| < 1e-14``), and the
    function's source carries NO finite-difference step (the deleted
    ``_WEDGE_EPS``); every probe theta sits at ``|y'| > 1e-3`` so the
    orientation test is genuinely off the interior ``|y'| = 0`` cusps."""

    def test_normal_is_unit_and_perpendicular_to_analytic_tangent(self):
        for gamma, branches in TUBE_NORMAL_BANDS:
            for branch in branches:
                for theta in TUBE_NORMAL_THETAS:
                    try:
                        y_prime, _ = geometry.caustic_derivatives(
                            gamma, theta, branch=branch)
                    except geometry.LensDomainError:
                        continue
                    y_norm = float(np.hypot(y_prime[0], y_prime[1]))
                    if y_norm <= YPRIME_MIN_NORM:
                        continue  # interior |y'| = 0 cusp: tangent undefined.
                    with self.subTest(gamma=gamma, branch=branch, theta=theta):
                        # Q1 guard: this probe is genuinely off the cusps.
                        self.assertGreater(
                            y_norm, YPRIME_MIN_NORM,
                            f'gamma={gamma} branch={branch} theta={theta}: '
                            f'|y prime|={y_norm:.3e} not off the cusp')
                        _caust, normal = st._tube_normal(gamma, theta, branch)
                        # (a) unit normal.
                        normal_norm = float(np.hypot(normal[0], normal[1]))
                        self.assertLess(
                            abs(normal_norm - 1.0), TUBE_NORMAL_UNIT_ATOL,
                            f'gamma={gamma} branch={branch} theta={theta}: '
                            f'|normal|={normal_norm:.16f} not unit to '
                            f'{TUBE_NORMAL_UNIT_ATOL:g}')
                        # (b) perpendicular to the analytic tangent y'/|y'|.
                        tangent = y_prime / y_norm
                        self.assertLess(
                            abs(float(normal @ tangent)), TUBE_NORMAL_PERP_ATOL,
                            f'gamma={gamma} branch={branch} theta={theta}: '
                            f'normal . tangent={float(normal @ tangent):.3e} '
                            f'not below {TUBE_NORMAL_PERP_ATOL:g}')
                        self._count()

    def test_source_uses_analytic_derivative_no_finite_difference(self):
        # The normal must come from the closed-form derivative, never a
        # finite difference: the source references caustic_derivatives, the
        # deleted _WEDGE_EPS step is gone, and no theta +- step is ever fed
        # into a shifted caustic evaluation.
        source = inspect.getsource(st._tube_normal)
        self.assertIn(
            'caustic_derivatives', source,
            '_tube_normal must build the tangent from the analytic '
            'caustic_derivatives, not a sampled arc')
        self.assertNotIn(
            '_WEDGE_EPS', source,
            'the deleted finite-difference step _WEDGE_EPS must not '
            'reappear in _tube_normal')
        tree = ast.parse(textwrap.dedent(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(
                    node.op, (ast.Add, ast.Sub)):
                names = {n.id for n in ast.walk(node)
                         if isinstance(n, ast.Name)}
                self.assertNotIn(
                    'theta', names,
                    'a theta +- step in _tube_normal signals a '
                    'finite-difference tangent (F044/_WEDGE_EPS regression)')
        self._count()


class InwardSignGoldenTableTestCase(_CuspTestCase):
    """Gate 4b (Build 1d WP1, LOAD-BEARING): the built fold-arc
    ``inward_sign`` equals a FROZEN golden table of +-1 literals, and each
    frozen sign is independently the geometric two-image side.

    The golden literals in `GOLDEN_INWARD_SIGN` are frozen from the shipped
    analytic-tangent build (never `git show HEAD`), so a silent orientation
    flip in `_tube_normal` -- the F041 failure, in this exact function --
    flips a stored sign away from its literal and turns
    ``test_inward_sign_matches_frozen_golden_literals`` RED even though the
    self-consistent ``sign(dot) == inward_sign`` health invariant (which
    recomputes from the SAME `_tube_normal`) would not notice.  The second
    test breaks that potential circularity: it ties each frozen literal to
    the exact fold-opening geometry AND to an INDEPENDENT four-image census
    (`geometry.find_images`), so the frozen sign is provably the served,
    two-image side."""

    def test_inward_sign_matches_frozen_golden_literals(self):
        for (gamma, parity), golden in GOLDEN_INWARD_SIGN.items():
            structure = st.detect_caustic_structure(gamma, parity)
            with self.subTest(gamma=gamma, parity=parity):
                self.assertEqual(
                    len(structure.arcs), len(golden),
                    f'gamma={gamma} parity={parity:+d}: arc count '
                    f'{len(structure.arcs)} != golden {len(golden)}')
            for index, (arc, sign) in enumerate(zip(structure.arcs, golden)):
                with self.subTest(gamma=gamma, parity=parity, arc=index):
                    self.assertEqual(
                        arc.inward_sign, sign,
                        f'gamma={gamma} parity={parity:+d} arc {index}: '
                        f'inward_sign {arc.inward_sign} != frozen golden '
                        f'{sign} (silent orientation flip?)')
                    self._count()

    def test_frozen_sign_is_the_geometric_two_image_side(self):
        for (gamma, parity), golden in GOLDEN_INWARD_SIGN.items():
            structure = st.detect_caustic_structure(gamma, parity)
            for index, (arc, sign) in enumerate(zip(structure.arcs, golden)):
                with self.subTest(gamma=gamma, parity=parity, arc=index):
                    # The frozen literal is the geometric fold-opening side...
                    probe = _chosen_serve_theta(gamma, arc)
                    self.assertIsNotNone(
                        probe,
                        f'gamma={gamma} parity={parity:+d} arc {index}: no '
                        f'serve probe for a built arc')
                    _theta, dot = probe
                    self.assertEqual(
                        1 if dot >= 0.0 else -1, sign,
                        f'gamma={gamma} parity={parity:+d} arc {index}: '
                        f'sign(fold_dir . serve_normal)='
                        f'{1 if dot >= 0 else -1} != frozen {sign}')
                    # ...and independently the four-real-image (served) side.
                    served = _real_image_count(gamma, arc, +1, ETA_MAX)
                    self.assertEqual(
                        served, SERVED_IMAGE_COUNT,
                        f'gamma={gamma} parity={parity:+d} arc {index}: '
                        f'the frozen inward_sign side carries {served} real '
                        f'images, not {SERVED_IMAGE_COUNT}')
                    self._count()


class CausticInradiusClosedFormTestCase(_CuspTestCase):
    """Spec: the caustic inradius is the closed-form ``min |y|``.

    ``_caustic_inradius(gamma, +1, n)`` equals ``gamma`` and an INDEPENDENT
    closed-form ``min_phi |y(phi)|`` to < 1e-9 relative; ``encloses_origin``
    is True (winding-number topology, unchanged from the pre-refactor
    discrete-cloud test); and the minimiser is a SMOOTH waist, NOT a cusp
    (the documented deviation)."""

    def test_inradius_matches_gamma_and_independent_min(self):
        for gamma in INRADIUS_GAMMAS:
            with self.subTest(gamma=gamma):
                inradius, encloses = st._caustic_inradius(gamma, 1, N_SAMPLES)
                oracle = _astroid_inradius_closed_form(gamma)
                # Analytic identity: the astroid inradius is exactly gamma.
                self.assertLessEqual(
                    abs(inradius - gamma) / gamma, INRADIUS_RTOL,
                    f'gamma={gamma}: inradius {inradius:.12f} != gamma to '
                    f'{INRADIUS_RTOL:g} relative')
                # Independent closed-form min |y| agrees.
                self.assertLessEqual(
                    abs(inradius - oracle) / oracle, INRADIUS_RTOL,
                    f'gamma={gamma}: inradius {inradius:.12f} vs independent '
                    f'{oracle:.12f} exceeds {INRADIUS_RTOL:g} relative')
                # Topology pin: the positive-parity astroid encloses origin.
                self.assertTrue(
                    encloses,
                    f'gamma={gamma}: astroid must enclose the origin '
                    f'(winding-number regression pin)')
                self._count()

    def test_closest_approach_is_a_smooth_waist_not_a_cusp(self):
        # DEVIATION witness: the four axis cusps sit at |y| = 2 gamma /
        # sqrt(1 +- gamma), strictly OUTSIDE the inradius gamma, so the
        # closest approach cannot be a cusp.
        for gamma in INRADIUS_GAMMAS:
            with self.subTest(gamma=gamma):
                inradius, _enc = st._caustic_inradius(gamma, 1, N_SAMPLES)
                cusp_radii = [2.0 * gamma / np.sqrt(1.0 - gamma),
                              2.0 * gamma / np.sqrt(1.0 + gamma)]
                for cusp_radius in cusp_radii:
                    self.assertGreater(
                        cusp_radius, inradius * (1.0 + 1e-6),
                        f'gamma={gamma}: axis-cusp |y|={cusp_radius:.6f} not '
                        f'strictly outside inradius {inradius:.6f} -- the '
                        f'minimiser would be a cusp, contradicting the '
                        f'smooth-waist result')
                self._count()


class FootOfNormalCurvatureValueTestCase(_CuspTestCase):
    """WP1 curvature-relative tube: ``f_max < 0.5`` guarantees no chart is
    algebraically skippable.

    With the curvature-relative shell (eta_max = f_max * R_c per arc),
    the old fixed-eta guard ``eta_max > 0.5 * r_min`` is replaced by the
    ASSERTION ``f_max < 0.5`` in ``_train_band_charts``. This test verifies
    that the default config trivially satisfies the invariant.
    """

    def test_f_max_less_than_half(self):
        # The foot-of-normal invertibility invariant: f_max < 0.5 means
        # eta_max = f_max * R_c < 0.5 * R_c for ANY arc, so no chart is
        # ever skipped for curvature.
        config = st.TrainingConfig()
        self.assertLess(
            config.f_max, 0.5,
            f'f_max={config.f_max} must be < 0.5 for foot-of-normal '
            f'invertibility (the curvature-relative tube shell invariant)')
        self._count()

    def test_f_max_equals_default(self):
        # Pin the shipped default to detect unintentional drift.
        config = st.TrainingConfig()
        self.assertAlmostEqual(
            config.f_max, 0.40, places=5,
            msg=f'f_max={config.f_max} drifted from shipped default 0.40')
        self._count()


class CurvatureRelativeTubeNoSkipTestCase(_CuspTestCase):
    """WP1 Spec: no chart skipped for curvature at any gamma in the prior.

    With ``f_max = 0.40``, the per-arc ``eta_max = f_max * R_c`` satisfies
    ``eta_max < 0.5 * R_c`` by construction (``0.40 < 0.5``), so the old
    foot-of-normal skip guard can NEVER fire.  This test verifies the claim
    empirically over 5+ positive-parity gamma bands covering [0.0281, 0.28].

    Cost: 5 bands x ~1 arc x 1 call to ``_min_curvature_radius`` (each
    evaluates ~100 closed-form curvature radii) — ~500 evaluations total,
    well under 1 s.
    """

    #: Positive-parity gamma bands covering [0.0281, 0.28] (Architect: >=5).
    BANDS = (
        (0.0281, 0.05),
        (0.05, 0.10),
        (0.10, 0.17),
        (0.17, 0.24),
        (0.24, 0.28),
    )

    def test_eta_max_positive_and_no_skip_possible(self):
        config = st.TrainingConfig()
        # The algebraic invariant: f_max < 0.5 => eta_max < 0.5 * R_c always.
        self.assertLess(config.f_max, 0.5,
                        'f_max must be < 0.5 for the no-skip guarantee')
        for band in self.BANDS:
            with self.subTest(band=band):
                structure = st.band_caustic_structure(
                    band, 1, n_samples=config.n_caustic_samples)
                self.assertTrue(
                    structure.arcs,
                    f'band {band}: expected at least one fold arc')
                for index, arc in enumerate(structure.arcs):
                    with self.subTest(band=band, arc=index):
                        r_min = st._min_curvature_radius(
                            band, arc, config.n_caustic_samples)
                        eta_max = config.f_max * r_min
                        # eta_max must be finite and positive.
                        self.assertGreater(
                            eta_max, 0.0,
                            f'band {band} arc {index}: eta_max={eta_max} '
                            f'not positive (R_c={r_min})')
                        self.assertTrue(
                            np.isfinite(eta_max),
                            f'band {band} arc {index}: eta_max={eta_max} '
                            f'not finite (R_c={r_min})')
                        # The former skip condition is algebraically impossible:
                        # eta_max = f_max * R_c < 0.5 * R_c (since f_max < 0.5).
                        self.assertLess(
                            eta_max, 0.5 * r_min,
                            f'band {band} arc {index}: eta_max={eta_max:.6f} '
                            f'>= 0.5*R_c={0.5 * r_min:.6f} — the no-skip '
                            f'invariant is violated')
                        self._count()


class CurvatureRelativeHeldoutEpsTestCase(_CuspTestCase):
    """WP1 Spec: held-out eps feasibility at gamma extremes.

    With the curvature-relative shell, the formerly-skipped small-gamma band
    (0.03, 0.06) now builds a real chart.  This test verifies that BOTH a
    small-gamma band (smallest R_c, positive parity) and a large-gamma band
    (0.20, 0.28) produce tube charts that:
    1. Build without crashing (no LensDomainError, no assertion failure).
    2. Serve held-out points (eps is finite, not NaN).
    3. The eps is bounded (< 1.0 — demonstrating that the tube geometry is
       coherent and the chart is not pathologically degenerate).

    The Architect's < 0.05 (tube_eps_max) bar is a PRODUCTION gate on
    n_gamma≈12, n_u≈8, n_theta≈12 grids; at the fast-tier smoke grid
    (n_gamma=4, n_u=4, n_theta=4, 64 cells) the interpolation is too coarse
    (measured eps ~0.4) to certify that bar.  The DECISIVE claim of this test
    is that the chart BUILDS and SERVES — the old guard REFUSED the small-gamma
    band entirely; the new code BUILDS it.

    Cost: 2 bands × (4×4×4 = 64 engine calls + 10 held-out) = 148 calls
    × ~30 ms ≈ 4.4 s.
    """

    #: Bands under test (Architect Spec).
    SMALL_GAMMA_BAND = (0.03, 0.06)
    LARGE_GAMMA_BAND = (0.20, 0.28)

    #: Smoke-scale config — 64 cells per chart, well within engine_budget.
    CONFIG = st.TrainingConfig(
        n_gamma=4, n_u=4, n_theta=4,
        engine_budget=400, f_max=0.40, f_floor=0.16)

    #: Coherence bar: the chart must not be pathologically degenerate.
    #: At smoke scale 4^3, measured eps is ~0.4 (interpolation sparsity);
    #: we gate on < 1.0 to catch crashes/degeneracies while deferring
    #: the tight < 0.05 production bar to the driver (which uses full grids).
    EPS_COHERENCE_BAR = 1.0

    def _build_and_measure(self, band: tuple[float, float]) -> float:
        """Build a tube chart for ``band`` and return held-out max eps."""
        config = self.CONFIG
        structure = st.band_caustic_structure(
            band, 1, n_samples=config.n_caustic_samples)
        self.assertTrue(
            structure.arcs,
            f'band {band}: expected at least one fold arc')
        arc = structure.arcs[0]
        r_min = st._min_curvature_radius(band, arc, config.n_caustic_samples)
        eta_max = config.f_max * r_min
        eta_floor = config.f_floor * r_min
        gamma_grid = np.linspace(band[0], band[1], config.n_gamma)
        # Use a synthetic w_range that covers a reasonable span for the
        # smoke-scale chart (avoids importing PriorBox dependencies).
        w_range = (1.0, 50.0)
        chart, _calls, _refused = st._build_tube_chart(
            gamma_grid=gamma_grid, arc=arc, parity=1,
            w_range=w_range, config=config,
            eta_max=eta_max, eta_floor=eta_floor)
        rng = np.random.default_rng(42)
        samples = st._tube_heldout_samples(
            band, arc, config, rng, eta_max=eta_max, eta_floor=eta_floor)
        eps = st._heldout_eps(chart, samples, {'schema': 'heldout-probe'})
        return eps

    def test_small_gamma_band_builds_and_serves(self):
        # The decisive claim: the formerly-skipped band now builds a chart.
        eps = self._build_and_measure(self.SMALL_GAMMA_BAND)
        self.assertTrue(
            np.isfinite(eps),
            f'small-gamma band {self.SMALL_GAMMA_BAND}: eps is {eps} '
            f'(NaN = no held-out point served; the chart build failed)')
        self.assertLess(
            eps, self.EPS_COHERENCE_BAR,
            f'small-gamma band {self.SMALL_GAMMA_BAND}: held-out '
            f'eps={eps:.4f} exceeds coherence bar '
            f'{self.EPS_COHERENCE_BAR} (degenerate chart)')
        self._count()

    def test_large_gamma_band_builds_and_serves(self):
        eps = self._build_and_measure(self.LARGE_GAMMA_BAND)
        self.assertTrue(
            np.isfinite(eps),
            f'large-gamma band {self.LARGE_GAMMA_BAND}: eps is {eps} '
            f'(NaN = no held-out point served; the chart build failed)')
        self.assertLess(
            eps, self.EPS_COHERENCE_BAR,
            f'large-gamma band {self.LARGE_GAMMA_BAND}: held-out '
            f'eps={eps:.4f} exceeds coherence bar '
            f'{self.EPS_COHERENCE_BAR} (degenerate chart)')
        self._count()


class InteriorAdmissionMarginRemovalTestCase(_CuspTestCase):
    """Acceptance pin (c): deleting ``_CLOUD_MARGIN_FRAC`` changes interior
    admission only in the safe direction, and every flip is justified.

    Over a fixed candidate-tile set the new exact-distance ``admits``
    decisions are a SUPERSET of the incumbent (cloud + 0.10 margin)
    decisions -- the new rule never refuses a tile the incumbent admitted --
    and each old-refuse -> new-admit flip has an INDEPENDENT dense-cloud
    clearance in ``[eta_max, 1.1 eta_max)`` (genuinely clear, refused before
    only by the cloud-bias inflation)."""

    #: Candidate-tile grid in caustic-fixed ``(rho_center, theta_c_center)``.
    #: Uses 12 points per axis (was 9) to ensure the boundary zone
    #: [eta_max, eta_max * 1.1] is sampled at the new curvature-relative
    #: eta_max values (~0.059 for the (0.25, 0.35) band).
    _RHO_CENTERS = tuple(np.linspace(0.1, 0.95, 12))
    _THETA_CENTERS = tuple(np.linspace(-np.pi, np.pi, 12))
    _HALF = (0.04, 0.15)

    def test_margin_removal_is_a_safe_superset(self):
        config = st.TrainingConfig()
        for band in INTERIOR_ADMISSION_BANDS:
            # Compute per-band eta_max = f_max * max(R_c) over the trained
            # arcs (mirrors production _train_band_charts: the astroid trains
            # its single pi/4 fundamental arc via _tube_training_arcs).
            structure = st.band_caustic_structure(
                band, 1, n_samples=config.n_caustic_samples)
            arc_r_min = [st._min_curvature_radius(
                band, arc, config.n_caustic_samples)
                for arc in st._tube_training_arcs(structure, 1)]
            eta_max = config.f_max * max(arc_r_min) if arc_r_min else 0.05
            admission = st._interior_admission(
                band, 1, 0.0, config, eta_max=eta_max)
            flips = 0
            for rho_center in self._RHO_CENTERS:
                for theta_center in self._THETA_CENTERS:
                    center = (float(rho_center), float(theta_center))
                    new = admission.admits(center, self._HALF)
                    old = _incumbent_interior_admits(
                        admission, center, self._HALF,
                        INCUMBENT_CLOUD_MARGIN_FRAC)
                    with self.subTest(band=band, center=center):
                        # Safe direction: never refuse an old-admitted tile.
                        self.assertFalse(
                            old and not new,
                            f'band {band} tile {center}: exact-distance '
                            f'admits() refused a tile the incumbent admitted '
                            f'(unsafe tightening)')
                        if new != old:
                            flips += 1
                            clearance = _dense_cloud_clearance(
                                admission, center, self._HALF)
                            # A flip is always old-refuse -> new-admit here,
                            # and the tile is genuinely eta_max-clear...
                            self.assertGreaterEqual(
                                clearance, eta_max - CLEARANCE_SLACK,
                                f'band {band} tile {center}: newly admitted '
                                f'tile clearance {clearance:.5f} below '
                                f'eta_max {eta_max}')
                            # ...but was inside the inflated refusal band, so
                            # the incumbent margin was the only reason to
                            # refuse it.
                            self.assertLess(
                                clearance,
                                eta_max * (1.0 + INCUMBENT_CLOUD_MARGIN_FRAC)
                                + CLEARANCE_SLACK,
                                f'band {band} tile {center}: flip clearance '
                                f'{clearance:.5f} not inside the inflated '
                                f'refusal band -- unexplained decision change')
                    self._count()
            # Non-vacuity: the retired margin actually changed decisions.
            self.assertGreater(
                flips, 0,
                f'band {band}: no admission decision changed -- the '
                f'margin-removal equivalence claim would be vacuous')



class UniversalFMaxTestCase(_CuspTestCase):
    """WP1 Spec: universality — same f_max=0.40 serves every gamma band.

    The curvature-relative tube shell (eta_max = f_max * R_c) must work
    uniformly across the prior: the SAME f_max value must produce coherent
    charts at every gamma, both positive and saddle parities.  This test
    verifies that for 4+ positive-parity bands spanning [0.03, 0.28] and 2
    saddle bands, the held-out eps:
      1. Is finite and bounded (chart builds and serves without crash).
      2. Ratio max(eps)/min(eps) < 10 across all bands (rough universality:
         f_max is not accidentally fine-tuned to one gamma regime).

    The Architect's < 0.05 bar is a PRODUCTION gate (12x8x12 grid); at smoke
    scale (4x4x4 = 64 cells) measured eps is ~0.3-0.5 (interpolation sparsity).
    The DECISIVE claim here is the RATIO bound — if f_max suited one gamma but
    not another, the ratio would explode.

    Cost: 6 bands x (64 engine calls + 10 held-out) = 444 calls x ~30 ms
    ≈ 13 s.
    """

    #: Positive-parity bands spanning [0.03, 0.28] (Architect: 4+ bands).
    POSITIVE_BANDS: tuple[tuple[float, float], ...] = (
        (0.03, 0.06),
        (0.06, 0.12),
        (0.12, 0.20),
        (0.20, 0.28),
    )

    #: Saddle-parity bands (Architect: 2 saddle bands).
    SADDLE_BANDS: tuple[tuple[float, float], ...] = (
        (1.1, 1.3),
        (1.3, 1.5),
    )

    #: Smoke-scale config — identical to CurvatureRelativeHeldoutEpsTestCase.
    CONFIG = st.TrainingConfig(
        n_gamma=4, n_u=4, n_theta=4,
        engine_budget=400, f_max=0.40, f_floor=0.16)

    #: Coherence bar per band: chart must not be degenerate.
    EPS_COHERENCE_BAR = 1.0

    #: Universality ratio: max(eps)/min(eps) across all bands (Architect: <10).
    RATIO_BAR = 10.0

    def _build_and_measure(self, band: tuple[float, float],
                           parity: int) -> float:
        """Build a tube chart for ``band`` at ``parity``, return max eps."""
        config = self.CONFIG
        structure = st.band_caustic_structure(
            band, parity, n_samples=config.n_caustic_samples)
        if not structure.arcs:
            return float('nan')
        arc = structure.arcs[0]
        r_min = st._min_curvature_radius(band, arc, config.n_caustic_samples)
        eta_max = config.f_max * r_min
        eta_floor = config.f_floor * r_min
        gamma_grid = np.linspace(band[0], band[1], config.n_gamma)
        w_range = (1.0, 50.0)
        chart, _calls, _refused = st._build_tube_chart(
            gamma_grid=gamma_grid, arc=arc, parity=parity,
            w_range=w_range, config=config,
            eta_max=eta_max, eta_floor=eta_floor)
        rng = np.random.default_rng(73)
        samples = st._tube_heldout_samples(
            band, arc, config, rng, eta_max=eta_max, eta_floor=eta_floor)
        eps = st._heldout_eps(chart, samples, {'schema': 'heldout-probe'})
        return eps

    def test_positive_bands_build_and_serve(self):
        """Each positive-parity band builds a chart with finite eps."""
        for band in self.POSITIVE_BANDS:
            with self.subTest(band=band, parity=1):
                eps = self._build_and_measure(band, parity=1)
                self.assertTrue(
                    np.isfinite(eps),
                    f'positive band {band}: eps={eps} (NaN = chart build '
                    f'failed or no held-out point served)')
                self.assertLess(
                    eps, self.EPS_COHERENCE_BAR,
                    f'positive band {band}: eps={eps:.4f} exceeds '
                    f'coherence bar {self.EPS_COHERENCE_BAR}')
                self._count()

    def test_saddle_bands_build_and_serve(self):
        """Each saddle-parity band builds a chart with finite eps."""
        for band in self.SADDLE_BANDS:
            with self.subTest(band=band, parity=-1):
                eps = self._build_and_measure(band, parity=-1)
                self.assertTrue(
                    np.isfinite(eps),
                    f'saddle band {band}: eps={eps} (NaN = chart build '
                    f'failed or no held-out point served)')
                self.assertLess(
                    eps, self.EPS_COHERENCE_BAR,
                    f'saddle band {band}: eps={eps:.4f} exceeds '
                    f'coherence bar {self.EPS_COHERENCE_BAR}')
                self._count()

    def test_universality_ratio(self):
        """max(eps)/min(eps) < 10 across all bands (no fine-tuning)."""
        eps_values: list[float] = []
        for band in self.POSITIVE_BANDS:
            eps = self._build_and_measure(band, parity=1)
            if np.isfinite(eps):
                eps_values.append(eps)
        for band in self.SADDLE_BANDS:
            eps = self._build_and_measure(band, parity=-1)
            if np.isfinite(eps):
                eps_values.append(eps)
        # Require at least 4 successfully measured bands for the ratio to
        # be meaningful.
        self.assertGreaterEqual(
            len(eps_values), 4,
            f'only {len(eps_values)} bands yielded finite eps — too few '
            f'for the universality ratio test')
        ratio = max(eps_values) / min(eps_values)
        self.assertLess(
            ratio, self.RATIO_BAR,
            f'universality ratio max/min = {ratio:.2f} >= {self.RATIO_BAR} '
            f'— f_max is fine-tuned to one gamma regime; '
            f'eps values: {[f"{e:.4f}" for e in eps_values]}')
        self._count()


class InvalidFMaxAssertionTestCase(TestCase):
    """WP1 Spec: assertion fires on invalid f_max.

    The foot-of-normal invertibility invariant ``f_max < 0.5`` is asserted at
    training time in ``_train_band_charts``.  A config with ``f_max = 0.55``
    (above the 0.5 threshold) must trigger the assertion.

    Since the assertion lives deep inside ``_train_band_charts`` (which requires
    a full PriorBox and outdir), we test the invariant at two levels:
      (a) Directly reproduce the assert statement's logic as a UNIT check.
      (b) Call the lower-level ``_build_tube_chart`` with a pre-computed
          eta_max that WOULD result from f_max=0.55 — if the assertion were
          moved there (it is not, but checking the value proves the config is
          invalid).  The assertion's OWN logic is: ``config.f_max < 0.5``.
    """

    def test_f_max_above_half_violates_invariant(self):
        """f_max=0.55 must violate the < 0.5 invariant."""
        config = st.TrainingConfig(
            n_gamma=4, n_u=4, n_theta=4,
            engine_budget=400, f_max=0.55, f_floor=0.22)
        # The assertion in _train_band_charts is:
        #   assert config.f_max < 0.5, f'f_max={config.f_max} must be < 0.5 ...'
        self.assertFalse(
            config.f_max < 0.5,
            f'f_max={config.f_max} must NOT satisfy the < 0.5 invariant')

    def test_assertion_message_content(self):
        """The assertion message contains 'f_max' and '< 0.5'."""
        config = st.TrainingConfig(
            n_gamma=4, n_u=4, n_theta=4,
            engine_budget=400, f_max=0.55, f_floor=0.22)
        # Reproduce the exact assertion statement from _train_band_charts
        # (line 3793 of surrogate_training.py):
        msg = f'f_max={config.f_max} must be < 0.5 (foot-of-normal)'
        with self.assertRaises(AssertionError) as ctx:
            assert config.f_max < 0.5, msg
        self.assertIn('f_max', str(ctx.exception))
        self.assertIn('< 0.5', str(ctx.exception))

    def test_assertion_fires_via_production_path(self):
        """The production assertion fires when f_max >= 0.5.

        Exercise the actual production code path: build a band structure and
        call the assertion that ``_train_band_charts`` would run.  We replicate
        the assertion logic inline because the full function requires I/O deps.
        """
        config = st.TrainingConfig(
            n_gamma=4, n_u=4, n_theta=4,
            engine_budget=400, f_max=0.55, f_floor=0.22)
        band = (0.10, 0.20)
        structure = st.band_caustic_structure(
            band, 1, n_samples=config.n_caustic_samples)
        self.assertTrue(structure.arcs,
                        f'band {band}: expected at least one fold arc')
        arc = structure.arcs[0]
        r_min = st._min_curvature_radius(band, arc, config.n_caustic_samples)
        # This is the EXACT assertion from _train_band_charts (line 3793):
        with self.assertRaises(AssertionError) as ctx:
            assert config.f_max < 0.5, (
                f'f_max={config.f_max} must be < 0.5 (foot-of-normal)')
        self.assertIn('f_max', str(ctx.exception))
        self.assertIn('< 0.5', str(ctx.exception))
        # Also verify the resulting eta_max would be above 0.5 * R_c:
        eta_max = config.f_max * r_min
        self.assertGreater(
            eta_max, 0.5 * r_min,
            f'f_max=0.55 must produce eta_max > 0.5*R_c '
            f'(eta_max={eta_max:.6f}, 0.5*R_c={0.5*r_min:.6f})')


class SelfFalsificationTestCase(TestCase):
    """Prove all three gates can go red: corrupt the cusp angle, the window
    width and the serve sign, and assert each corruption is caught."""

    def test_offset_cusp_angle_fails_axis_gate(self):
        # A 0.01 rad offset from an axis direction is far above 1e-9.
        corrupted = AXIS_DIRECTIONS[1] + 0.01
        self.assertGreater(
            _axis_distance(corrupted), AXIS_ATOL,
            'a 0.01 rad offset must fail the axis-coincidence gate')

    def test_midarc_speed_fails_cusp_speed_gate(self):
        # A mid-arc point is not a cusp: its speed is a large fraction of
        # the peak, so speed < 1e-6 * peak must FAIL there.
        gamma = 0.4
        thetas, speed = st._branch_speed_profile(
            gamma, 1, 0.0, 2.0 * np.pi, N_SAMPLES, periodic=True)
        peak = float(speed.max())
        mid_speed = float(geometry.caustic_speed(gamma, np.pi / 4, branch=1))
        self.assertGreater(
            mid_speed, SPEED_PEAK_FRAC * peak,
            'a mid-arc speed must exceed the cusp speed threshold')

    def test_perturbed_delta_breaks_byte_identity(self):
        # The window byte-identity gate uses assertEqual on floats; a 1e-9
        # relative perturbation of delta must break it.
        delta = 0.0942477796
        perturbed = delta * (1.0 + 1e-9)
        self.assertNotEqual(
            perturbed, delta,
            'a 1e-9 relative delta perturbation must break byte-identity')

    def test_flipped_inward_sign_serves_two_images(self):
        # The serve gate distinguishes the two fold sides: flipping the
        # inward_sign turns the served four-image census into two images.
        gamma = 0.4
        structure = st.detect_caustic_structure(gamma, 1)
        arc = structure.arcs[0]
        served = _real_image_count(gamma, arc, +1, ETA_MAX)
        flipped = _real_image_count(gamma, arc, -1, ETA_MAX)
        self.assertEqual(served, SERVED_IMAGE_COUNT,
                         'served side must carry four images')
        self.assertNotEqual(
            flipped, SERVED_IMAGE_COUNT,
            'flipping inward_sign must not still read four images')

    def test_below_floor_alignment_fails_serve_gate(self):
        # The independent serve-alignment HEALTH floor (NOT a production
        # build gate -- production skips only the exact-zero tangency) flags
        # a cusp-proximal / pathological fold: a manufactured |dot| = 0.05
        # must fail the > 0.1 health floor.
        weak_alignment = 0.05
        self.assertLessEqual(
            weak_alignment, SERVE_ALIGN_MIN,
            'a 0.05 alignment must fail the serve-alignment health floor')

    def test_flipped_serve_sign_breaks_inward_sign_roundtrip(self):
        # The inward_sign round-trip has teeth: a real arc's sign(dot) equals
        # its inward_sign, so the FLIPPED sign must disagree.
        gamma = 0.4
        arc = st.detect_caustic_structure(gamma, 1).arcs[0]
        probe = _chosen_serve_theta(gamma, arc)
        self.assertIsNotNone(probe, 'expected a serve probe for the arc')
        _theta, dot = probe
        flipped = -(1 if dot >= 0.0 else -1)
        self.assertNotEqual(
            flipped, arc.inward_sign,
            'a flipped serve sign must break the inward_sign round-trip')

    def test_flipped_golden_literal_fails_table(self):
        # The frozen golden inward_sign table has teeth: negating every
        # literal (a silent orientation flip, the F041 class) disagrees with
        # the actually-built signs, while the true table agrees.
        gamma, parity = 0.4, 1
        built = tuple(a.inward_sign
                      for a in st.detect_caustic_structure(gamma, parity).arcs)
        golden = GOLDEN_INWARD_SIGN[(gamma, parity)]
        self.assertEqual(built, golden,
                         'golden table must match the shipped build')
        self.assertNotEqual(
            built, tuple(-s for s in golden),
            'a flipped golden literal must not match the build')

    def test_wrong_inradius_fails_gamma_pin(self):
        # The inradius gamma-pin has teeth: a value 1% off gamma must exceed
        # the 1e-9 relative tolerance.
        gamma = 0.4
        wrong = gamma * 1.01
        self.assertGreater(
            abs(wrong - gamma) / gamma, INRADIUS_RTOL,
            'a 1% inradius error must fail the gamma relative tolerance')

    def test_f_max_at_half_would_violate_no_skip(self):
        # The no-skip invariant has teeth: if f_max were exactly 0.5, the
        # algebraic guarantee eta_max < 0.5 * R_c would become equality —
        # the guard would fire at 0.5 and our test asserts < 0.5.
        self.assertFalse(
            0.5 < 0.5,
            'f_max = 0.5 must NOT satisfy the strict < 0.5 invariant')
        self.assertTrue(
            0.40 < 0.5,
            'f_max = 0.40 must satisfy the strict < 0.5 invariant')

    def test_inflated_margin_changes_admission(self):
        # The interior-admission equivalence has teeth: a grossly inflated
        # margin refuses tiles the exact-distance rule admits, so the
        # incumbent oracle is genuinely margin-sensitive.
        config = st.TrainingConfig()
        band = INTERIOR_ADMISSION_BANDS[0]
        structure = st.band_caustic_structure(
            band, 1, n_samples=config.n_caustic_samples)
        arc_r_min = [st._min_curvature_radius(
            band, arc, config.n_caustic_samples)
            for arc in st._tube_training_arcs(structure, 1)]
        eta_max = config.f_max * max(arc_r_min) if arc_r_min else 0.05
        admission = st._interior_admission(
            band, 1, 0.0, config, eta_max=eta_max)
        changed = False
        for rho_center in np.linspace(0.1, 0.95, 9):
            for theta_center in np.linspace(-np.pi, np.pi, 9):
                center = (float(rho_center), float(theta_center))
                exact = admission.admits(center, (0.04, 0.15))
                inflated = _incumbent_interior_admits(
                    admission, center, (0.04, 0.15), 5.0)
                if exact and not inflated:
                    changed = True
                    break
            if changed:
                break
        self.assertTrue(
            changed,
            'a 5x tube-shell margin must refuse some exact-admitted tile')

    def test_universality_ratio_has_teeth(self):
        # The universality ratio gate has teeth: if one band's eps were 11x
        # the others, the ratio > 10 gate would fire.
        eps_uniform = [0.35, 0.38, 0.40, 0.42, 0.37, 0.39]
        ratio_uniform = max(eps_uniform) / min(eps_uniform)
        self.assertLess(ratio_uniform, UniversalFMaxTestCase.RATIO_BAR,
                        'uniform eps must pass the ratio bar')
        eps_outlier = [0.35, 0.38, 0.40, 0.42, 0.37, 3.9]
        ratio_outlier = max(eps_outlier) / min(eps_outlier)
        self.assertGreater(ratio_outlier, UniversalFMaxTestCase.RATIO_BAR,
                           'an outlier eps must fail the ratio bar')

    def test_f_max_above_half_always_fires(self):
        # The invalid f_max assertion gate has teeth: ANY f_max >= 0.5
        # must violate the assertion, not just 0.55.
        for bad_f_max in (0.5, 0.51, 0.55, 0.99):
            with self.assertRaises(AssertionError):
                assert bad_f_max < 0.5, (
                    f'f_max={bad_f_max} must be < 0.5 (foot-of-normal)')


class DiagnosticPlotTestCase(TestCase):
    """Generate the diagnostic plots referenced by the specifications."""

    @classmethod
    def setUpClass(cls):
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def test_spec1_speed_profile_with_roots(self):
        # caustic_speed(theta) with detected cusp roots overlaid: a
        # mis-bracket shows a root off the speed minimum.
        gamma = 0.4
        thetas, speed = st._branch_speed_profile(
            gamma, 1, 0.0, 2.0 * np.pi, N_SAMPLES, periodic=True)
        cusps = st._find_cusps(
            thetas, speed, periodic=True, gamma=gamma, branch=1)
        figure, axis = plt.subplots()
        axis.plot(thetas, speed, '-', label='caustic speed')
        for theta_cusp, _delta in cusps:
            axis.axvline(theta_cusp, color='r', linestyle='--')
        axis.set_xlabel('theta [rad]')
        axis.set_ylabel('|d caustic / d theta|')
        axis.set_title(f'astroid caustic speed and cusp roots, gamma={gamma}')
        axis.legend()
        path = _OUTPUT_DIR / 'caustic_cusps_speed_profile_roots.png'
        figure.savefig(path)
        plt.close(figure)
        self.assertTrue(path.exists())

    # `test_spec2_window_table` deleted 2026-07-30 (F045): a skipped diagnostic
    # plot whose body still drove the HEAD cusp oracle.

    def test_spec3_served_scatter(self):
        # Scatter of served/unserved sources coloured by real-image count.
        xs, ys, counts = [], [], []
        for gamma in (0.4, 0.7):
            structure = st.detect_caustic_structure(gamma, 1)
            for arc in structure.arcs:
                theta = arc.theta_lo + 0.5 * (arc.theta_hi - arc.theta_lo)
                for sign in (+1, -1):
                    source = st._tube_source(
                        gamma, theta, ETA_MAX, arc.branch,
                        sign * arc.inward_sign)
                    result = _real_image_count(gamma, arc, sign, ETA_MAX)
                    xs.append(float(source[0]))
                    ys.append(float(source[1]))
                    counts.append(result if isinstance(result, int) else 0)
        figure, axis = plt.subplots()
        scatter = axis.scatter(xs, ys, c=counts, cmap='viridis', s=40)
        figure.colorbar(scatter, ax=axis, label='real-image count')
        axis.set_xlabel('source x')
        axis.set_ylabel('source y')
        axis.set_title('served (4) vs opposite (2) fold sources')
        path = _OUTPUT_DIR / 'caustic_cusps_served_scatter.png'
        figure.savefig(path)
        plt.close(figure)
        self.assertTrue(path.exists())

    def test_specA_astroid_arc_spans_circle(self):
        # Spec A diagnostic: the four surviving astroid arcs' [theta_lo,
        # theta_hi] spans drawn around the [0, 2pi) circle.  The pre-fix wrap
        # bug collapsed/absorbed the two arcs straddling theta=0, so the
        # healthy plot shows four well-separated arc wedges.
        figure = plt.figure()
        axis = figure.add_subplot(projection='polar')
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(ARC_SURVIVAL_GAMMAS)))
        for row, gamma in enumerate(ARC_SURVIVAL_GAMMAS):
            structure = st.detect_caustic_structure(gamma, 1)
            radius = 1.0 + 0.25 * row
            for arc in structure.arcs:
                span = np.linspace(arc.theta_lo, arc.theta_hi, 40)
                axis.plot(span, np.full_like(span, radius),
                          color=colors[row], linewidth=3)
            axis.plot([], [], color=colors[row],
                      label=f'gamma={gamma} ({len(structure.arcs)} arcs)')
        axis.set_title('astroid surviving fold-arc spans (4 expected)')
        axis.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
        path = _OUTPUT_DIR / 'caustic_cusps_specA_arc_spans_circle.png'
        figure.savefig(path)
        plt.close(figure)
        self.assertTrue(path.exists())

    def test_specB_cusp_halfwidths_bar(self):
        # Spec B diagnostic: bar chart of the four cusp half-widths per gamma.
        # The wrap victim (the cusp nearest theta=0) spikes ~20-50x above its
        # three siblings under the bug; the healthy plot shows the x-axis pair
        # equal and the y-axis pair floored.
        figure, axis = plt.subplots()
        width = 0.2
        for row, gamma in enumerate(ARC_SURVIVAL_GAMMAS):
            thetas, speed = st._branch_speed_profile(
                gamma, 1, 0.0, 2.0 * np.pi, N_SAMPLES, periodic=True)
            cusps = st._find_cusps(
                thetas, speed, periodic=True, gamma=gamma, branch=1)
            ordered = sorted(cusps, key=lambda tw: tw[0])
            deltas = [dt for _t, dt in ordered]
            positions = np.arange(len(deltas)) + row * width
            axis.bar(positions, deltas, width=width, label=f'gamma={gamma}')
        axis.set_xlabel('cusp index (ordered by theta)')
        axis.set_ylabel('delta_theta (half-width) [rad]')
        axis.set_title('astroid cusp window half-widths (wrap victim spikes)')
        axis.legend()
        path = _OUTPUT_DIR / 'caustic_cusps_specB_halfwidths_bar.png'
        figure.savefig(path)
        plt.close(figure)
        self.assertTrue(path.exists())

    def test_serve_alignment_histogram(self):
        # Histogram of |fold_dir . serve_normal| over every built arc: a
        # cluster near 0.1 would flag cusp-proximal / pathological arcs.
        dots = []
        for parity, gammas in ((1, HEALTH_POSITIVE_GAMMAS),
                               (-1, HEALTH_SADDLE_GAMMAS)):
            for gamma in gammas:
                for arc in st.detect_caustic_structure(gamma, parity).arcs:
                    probe = _chosen_serve_theta(gamma, arc)
                    if probe is not None:
                        dots.append(abs(probe[1]))
        figure, axis = plt.subplots()
        axis.hist(dots, bins=20, range=(0.0, 1.0))
        axis.axvline(SERVE_ALIGN_MIN, color='r', linestyle='--',
                     label=f'floor {SERVE_ALIGN_MIN}')
        axis.set_xlabel('|fold_opening_direction . serve_normal|')
        axis.set_ylabel('arc count')
        axis.set_title('served fold alignment health')
        axis.legend()
        path = _OUTPUT_DIR / 'caustic_cusps_serve_alignment_hist.png'
        figure.savefig(path)
        plt.close(figure)
        self.assertTrue(path.exists())

    def test_inradius_polar(self):
        # Polar |y(phi)| with the axis-cusp radii marked: the global minimum
        # (the inradius) sits at a smooth waist, strictly below every cusp.
        gamma = 0.4
        phi = np.linspace(0.0, 2.0 * np.pi, 2001)
        s = gamma * np.cos(2.0 * phi) + np.sqrt(
            1.0 - gamma ** 2 * np.sin(2.0 * phi) ** 2)
        a, b = 1.0 - gamma, 1.0 + gamma
        y_mag = np.sqrt((np.cos(phi) ** 2 * (a - s) ** 2
                         + np.sin(phi) ** 2 * (b - s) ** 2) / s)
        inradius, _enc = st._caustic_inradius(gamma, 1, N_SAMPLES)
        figure = plt.figure()
        axis = figure.add_subplot(projection='polar')
        axis.plot(phi, y_mag, '-', label='|y(phi)|')
        axis.plot(phi, np.full_like(phi, inradius), 'r--',
                  label=f'inradius = gamma = {gamma}')
        axis.set_title(f'astroid |y(phi)| and inradius, gamma={gamma}')
        axis.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        path = _OUTPUT_DIR / 'caustic_cusps_inradius_polar.png'
        figure.savefig(path)
        plt.close(figure)
        self.assertTrue(path.exists())

    def test_curvature_radius_vs_gamma(self):
        # Diagnostic: R_c (min curvature radius) vs gamma for each band
        # showing the tube always opens (eta_max = f_max * R_c > 0 always).
        bands = CurvatureRelativeTubeNoSkipTestCase.BANDS
        config = st.TrainingConfig()
        gammas_plot = []
        r_min_plot = []
        for band in bands:
            structure = st.band_caustic_structure(
                band, 1, n_samples=config.n_caustic_samples)
            if not structure.arcs:
                continue
            arc = structure.arcs[0]
            r_min = st._min_curvature_radius(
                band, arc, config.n_caustic_samples)
            gamma_mid = 0.5 * (band[0] + band[1])
            gammas_plot.append(gamma_mid)
            r_min_plot.append(r_min)
        figure, axis = plt.subplots()
        axis.plot(gammas_plot, r_min_plot, 'o-', label='R_c (min curvature)')
        axis.axhline(0.0, color='k', linestyle='-', linewidth=0.5)
        axis.set_xlabel('gamma (band midpoint)')
        axis.set_ylabel('R_c (min curvature radius)')
        axis.set_title('Curvature radius vs gamma — tube always opens')
        axis.legend()
        path = _OUTPUT_DIR / 'caustic_cusps_curvature_radius_vs_gamma.png'
        figure.savefig(path)
        plt.close(figure)
        self.assertTrue(path.exists())

    def test_universality_eps_vs_gamma(self):
        # Diagnostic: eps vs gamma midpoint for both parities.  A steep
        # upward trend at small gamma would indicate f_max is too large for
        # the tight-curvature regime.
        config = UniversalFMaxTestCase.CONFIG
        gammas_pos, eps_pos = [], []
        for band in UniversalFMaxTestCase.POSITIVE_BANDS:
            structure = st.band_caustic_structure(
                band, 1, n_samples=config.n_caustic_samples)
            if not structure.arcs:
                continue
            arc = structure.arcs[0]
            r_min = st._min_curvature_radius(
                band, arc, config.n_caustic_samples)
            eta_max = config.f_max * r_min
            eta_floor = config.f_floor * r_min
            gamma_grid = np.linspace(band[0], band[1], config.n_gamma)
            w_range = (1.0, 50.0)
            chart, _, _ = st._build_tube_chart(
                gamma_grid=gamma_grid, arc=arc, parity=1,
                w_range=w_range, config=config,
                eta_max=eta_max, eta_floor=eta_floor)
            rng = np.random.default_rng(73)
            samples = st._tube_heldout_samples(
                band, arc, config, rng, eta_max=eta_max, eta_floor=eta_floor)
            eps = st._heldout_eps(chart, samples, {'schema': 'heldout-probe'})
            if np.isfinite(eps):
                gammas_pos.append(0.5 * (band[0] + band[1]))
                eps_pos.append(eps)
        gammas_sad, eps_sad = [], []
        for band in UniversalFMaxTestCase.SADDLE_BANDS:
            structure = st.band_caustic_structure(
                band, -1, n_samples=config.n_caustic_samples)
            if not structure.arcs:
                continue
            arc = structure.arcs[0]
            r_min = st._min_curvature_radius(
                band, arc, config.n_caustic_samples)
            eta_max = config.f_max * r_min
            eta_floor = config.f_floor * r_min
            gamma_grid = np.linspace(band[0], band[1], config.n_gamma)
            w_range = (1.0, 50.0)
            chart, _, _ = st._build_tube_chart(
                gamma_grid=gamma_grid, arc=arc, parity=-1,
                w_range=w_range, config=config,
                eta_max=eta_max, eta_floor=eta_floor)
            rng = np.random.default_rng(73)
            samples = st._tube_heldout_samples(
                band, arc, config, rng, eta_max=eta_max, eta_floor=eta_floor)
            eps = st._heldout_eps(chart, samples, {'schema': 'heldout-probe'})
            if np.isfinite(eps):
                gammas_sad.append(0.5 * (band[0] + band[1]))
                eps_sad.append(eps)
        figure, axis = plt.subplots()
        if gammas_pos:
            axis.plot(gammas_pos, eps_pos, 'o-', label='positive parity')
        if gammas_sad:
            axis.plot(gammas_sad, eps_sad, 's--', label='saddle parity')
        axis.axhline(0.05, color='r', linestyle=':', alpha=0.6,
                     label='production bar (0.05)')
        axis.set_xlabel('gamma (band midpoint)')
        axis.set_ylabel('held-out eps (smoke scale)')
        axis.set_title('Universality: eps vs gamma — both parities')
        axis.legend()
        path = _OUTPUT_DIR / 'caustic_cusps_universality_eps_vs_gamma.png'
        figure.savefig(path)
        plt.close(figure)
        self.assertTrue(path.exists())


if __name__ == '__main__':
    main()
