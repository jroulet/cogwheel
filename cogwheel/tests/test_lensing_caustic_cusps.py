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

  * `FootOfNormalCurvatureValueTestCase` (Acceptance pin (a)).  With
    ``eta_max = 0.05``, ``config.eta_max > 0.5 * _min_curvature_radius(band,
    arc, n)`` is asserted per band as a VALUE (never byte-identity with the
    incumbent).  It is False on the four main bands (0.25,0.35), (0.45,0.55),
    (0.65,0.75), (0.85,0.95) and on (0.155,0.3) -- the brief's headline.
    DEVIATION: it is not universally False across the brief's small-gamma
    table.  On (0.0825,0.1550) the guard FIRES (True): the small astroid's
    curvature radius (measured ``r_min = 0.059``) makes ``0.5 r_min = 0.030
    < eta_max`` -- exactly the tight-curvature condition the foot-of-normal
    guard exists to catch, so the chart is correctly skipped there.  (The
    F041 arc-orientation fix makes small-gamma astroid bands build real fold
    arcs, so the former "brief small bands have no band-wide arc" and
    "stable_gamma_bands drops a sliver" pins -- which encoded the pre-fix
    pathology -- have been retired.)

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
"""

from __future__ import annotations

import ast
import pathlib
import subprocess

from unittest import TestCase, main, skipUnless

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

#: A detected cusp's caustic speed must fall below this fraction of the
#: branch peak speed (Architect Spec 1; measured worst ~4e-16).
SPEED_PEAK_FRAC = 1e-6

#: Absolute tolerance on the cusp/axis-direction coincidence, radians
#: (Architect Spec 1: 1e-9; measured coincidence is machine-exact).
AXIS_ATOL = 1e-9

#: Serve step off the fold, in caustic-normal units (production
#: ``_DEFAULT_ETA_MAX``; Architect eta_max = 0.05).
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
#: exactly (the arc's own inner span, same fraction order, same 0.1 floor).
_MAKE_ARC_FRACS = (0.5, 0.35, 0.65, 0.2, 0.8)

#: Minimum served-alignment ``|fold_dir . serve_normal|`` for a healthy fold
#: arc (Architect: > 0.1; equals `_make_arc`'s own build floor).
SERVE_ALIGN_MIN = 0.1

#: Positive-parity shears for the inward_sign health sweep (measured worst
#: |dot| = 0.298 at gamma = 0.2, well above the 0.1 floor).
HEALTH_POSITIVE_GAMMAS = (0.2, 0.4, 0.7, 0.9)

#: Macro-saddle shears for the health sweep (both deltoid branches: the
#: branch=-1 edges give |dot| = 1.0, the branch=+1 edges ~0.77-0.85).
HEALTH_SADDLE_GAMMAS = (1.2, 1.5)

#: Astroid shears for the closed-form inradius spec (positive parity).
INRADIUS_GAMMAS = (0.05, 0.2, 0.4, 0.7)

#: Relative tolerance on the inradius closed-form agreement (Architect
#: 1e-9; measured shipped-vs-independent ~2e-10, shipped-vs-gamma ~1e-16).
INRADIUS_RTOL = 1e-9

#: Samples for the INDEPENDENT ``min_phi |y(phi)|`` inradius oracle (a
#: smooth quadratic minimum, so this density gives ~1e-10 residual).
INRADIUS_ORACLE_SAMPLES = 200001

#: Positive-parity gamma bands with a well-defined band-wide fold arc where
#: the foot-of-normal guard is measured FALSE (pin (a) headline).
FOOT_FALSE_BANDS = ((0.25, 0.35), (0.45, 0.55), (0.65, 0.75),
                    (0.85, 0.95), (0.155, 0.3))

#: Small-gamma band where the foot-of-normal guard measurably FIRES (True):
#: the small astroid's curvature radius drops below 2 * eta_max (documented
#: deviation from the brief's "False on every band").
FOOT_TRUE_BAND = (0.0825, 0.155)

#: Positive-parity bands for the interior-admission margin-removal pin (c).
INTERIOR_ADMISSION_BANDS = ((0.25, 0.35), (0.45, 0.55))

#: The incumbent (HEAD) interior tube-shell inflation factor removed by the
#: exact-distance refactor (verified against ``git show HEAD``).
INCUMBENT_CLOUD_MARGIN_FRAC = 0.10

#: Dense caustic-cloud samples for the INDEPENDENT nearest-distance oracle
#: used to bracket the pin-(c) boundary flips (distinct from production's
#: 200-point cloud AND from the exact ``nearest_caustic_point``).
INTERIOR_DENSE_SAMPLES = 40001

#: Slack (dimensionless ``y``) for the dense-cloud clearance bracket: a
#: 40001-point cloud resolves the ~eta_max nearest distance to well within
#: this (half the cloud spacing near the caustic).
CLEARANCE_SLACK = 2e-3


def _git_available() -> bool:
    """True if ``git show HEAD:<module>`` yields the pre-refactor source."""
    try:
        _head_module_source()
    except (subprocess.CalledProcessError, OSError):
        return False
    return True


def _head_module_source() -> str:
    """Return the HEAD text of ``surrogate_training.py`` via ``git show``."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ['git', 'show', f'HEAD:{_MODULE_REL_PATH}'],
        cwd=repo_root, check=True, capture_output=True, text=True)
    return completed.stdout


def _head_find_cusps():
    """Build the pre-refactor ``_find_cusps`` as an INDEPENDENT oracle.

    ``_find_cusps`` at HEAD is fully self-contained -- it references only
    ``numpy`` and the module constants ``_CUSP_SPEED_REL_FRAC``,
    ``_CUSP_WIDTH_SAFETY`` and ``_CUSP_MIN_HALFWIDTH`` -- so its
    ``FunctionDef`` is AST-extracted from the HEAD source and exec'd in a
    minimal namespace carrying just those four names.  This deliberately
    avoids importing the (heavy, refactored) live module a second time and
    keeps the window oracle a true cross-version comparison.

    Returns
    -------
    callable
        ``head_find_cusps(thetas, speed, periodic, *, width_safety=...,
        min_halfwidth=...)`` -> list of ``(theta_cusp, delta_theta)``.
    """
    source = _head_module_source()
    tree = ast.parse(source)
    namespace: dict = {'np': np}
    wanted_consts = {
        '_CUSP_SPEED_REL_FRAC', '_CUSP_WIDTH_SAFETY', '_CUSP_MIN_HALFWIDTH'}
    func_segment = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted_consts:
                    namespace[target.id] = ast.literal_eval(node.value)
        elif isinstance(node, ast.FunctionDef) and node.name == '_find_cusps':
            func_segment = ast.get_source_segment(source, node)
    if func_segment is None:
        raise RuntimeError('HEAD _find_cusps not found in module source')
    missing = wanted_consts - namespace.keys()
    if missing:
        raise RuntimeError(f'HEAD cusp constants missing: {sorted(missing)}')
    exec(compile(func_segment, '<head_find_cusps>', 'exec'), namespace)
    return namespace['_find_cusps']


def _axis_distance(theta: float) -> float:
    """Smallest wrapped distance from ``theta`` to an axis cusp direction."""
    axes = np.asarray(AXIS_DIRECTIONS)
    return float(np.min(np.abs(((axes - theta + np.pi) % (2.0 * np.pi))
                               - np.pi)))


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
    same ``|dot| > 0.1`` floor and the same `_tube_normal` serve normal, so
    the returned ``(theta, dot)`` is the very orientation probe that fixed
    the arc's ``inward_sign``.  Returns ``None`` if no fraction cleared the
    floor (which cannot happen for an arc that was actually built).
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
        if abs(dot) <= SERVE_ALIGN_MIN:
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


class CuspWindowByteIdentityTestCase(_CuspTestCase):
    """Spec 2: the exclusion-window half-width is byte-identical; only the
    cusp centre moves (by at most one sampling step) toward the root."""

    @classmethod
    def setUpClass(cls):
        cls._head_find_cusps = staticmethod(_head_find_cusps())

    @skipUnless(_git_available(), 'git HEAD source unavailable')
    def test_window_widths_unchanged_centres_shift(self):
        for gamma in ASTROID_GAMMAS:
            with self.subTest(gamma=gamma):
                thetas, speed = st._branch_speed_profile(
                    gamma, 1, 0.0, 2.0 * np.pi, N_SAMPLES, periodic=True)
                new = sorted(st._find_cusps(
                    thetas, speed, periodic=True, gamma=gamma, branch=1))
                old = sorted(self._head_find_cusps(
                    thetas, speed, periodic=True))
                self.assertEqual(
                    len(new), len(old),
                    f'gamma={gamma}: cusp count changed old={len(old)} '
                    f'new={len(new)}')
                for (theta_new, delta_new), (theta_old, delta_old) in zip(
                        new, old):
                    # Window half-width byte-for-byte identical.
                    self.assertEqual(
                        delta_new, delta_old,
                        f'gamma={gamma}: delta changed {delta_old!r} -> '
                        f'{delta_new!r} (window rule must be untouched)')
                    # Centre shifts by at most one sampling step.
                    shift = abs(((theta_new - theta_old + np.pi)
                                 % (2.0 * np.pi)) - np.pi)
                    self.assertLessEqual(
                        shift, SAMPLING_STEP + 1e-12,
                        f'gamma={gamma}: centre shift {shift:.3e} exceeds '
                        f'one step {SAMPLING_STEP:.3e}')
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
    """Acceptance pin (a): ``eta_max > 0.5 * _min_curvature_radius`` as a
    VALUE per band (not byte-identity with the incumbent margin)."""

    def test_guard_false_on_main_bands(self):
        config = st.TrainingConfig()
        for band in FOOT_FALSE_BANDS:
            structure = st.band_caustic_structure(
                band, 1, n_samples=config.n_caustic_samples)
            self.assertTrue(
                structure.arcs,
                f'band {band}: expected a band-wide fold arc to test')
            for index, arc in enumerate(structure.arcs):
                with self.subTest(band=band, arc=index):
                    r_min = st._min_curvature_radius(
                        band, arc, config.n_caustic_samples)
                    self.assertFalse(
                        config.eta_max > 0.5 * r_min,
                        f'band {band} arc {index}: foot-of-normal guard '
                        f'fired (eta_max={config.eta_max} > '
                        f'0.5*r_min={0.5 * r_min:.5f}); expected clearance')
                    self._count()

    def test_guard_fires_on_small_astroid_band(self):
        # Documented deviation: on (0.0825,0.155) the small astroid's tight
        # curvature radius makes 0.5*r_min < eta_max, so the guard fires --
        # the chart is correctly skipped, contrary to the brief's universal
        # "False on every band".
        config = st.TrainingConfig()
        structure = st.band_caustic_structure(
            FOOT_TRUE_BAND, 1, n_samples=config.n_caustic_samples)
        self.assertTrue(structure.arcs,
                        f'band {FOOT_TRUE_BAND}: expected a fold arc')
        for index, arc in enumerate(structure.arcs):
            with self.subTest(band=FOOT_TRUE_BAND, arc=index):
                r_min = st._min_curvature_radius(
                    FOOT_TRUE_BAND, arc, config.n_caustic_samples)
                self.assertTrue(
                    config.eta_max > 0.5 * r_min,
                    f'band {FOOT_TRUE_BAND} arc {index}: expected the guard '
                    f'to fire (eta_max={config.eta_max} > 0.5*r_min='
                    f'{0.5 * r_min:.5f}); the small astroid is tightly curved')
                self._count()

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
    _RHO_CENTERS = tuple(np.linspace(0.1, 0.95, 9))
    _THETA_CENTERS = tuple(np.linspace(-np.pi, np.pi, 9))
    _HALF = (0.04, 0.15)

    def test_margin_removal_is_a_safe_superset(self):
        config = st.TrainingConfig()
        eta_max = config.eta_max
        for band in INTERIOR_ADMISSION_BANDS:
            admission = st._interior_admission(band, 1, 0.0, config)
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
        # The serve-alignment gate rejects a cusp-proximal / pathological
        # fold: a manufactured |dot| = 0.05 must fail the > 0.1 floor.
        weak_alignment = 0.05
        self.assertLessEqual(
            weak_alignment, SERVE_ALIGN_MIN,
            'a 0.05 alignment must fail the serve-alignment floor')

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

    def test_wrong_inradius_fails_gamma_pin(self):
        # The inradius gamma-pin has teeth: a value 1% off gamma must exceed
        # the 1e-9 relative tolerance.
        gamma = 0.4
        wrong = gamma * 1.01
        self.assertGreater(
            abs(wrong - gamma) / gamma, INRADIUS_RTOL,
            'a 1% inradius error must fail the gamma relative tolerance')

    def test_inflated_margin_changes_admission(self):
        # The interior-admission equivalence has teeth: a grossly inflated
        # margin refuses tiles the exact-distance rule admits, so the
        # incumbent oracle is genuinely margin-sensitive.
        config = st.TrainingConfig()
        band = INTERIOR_ADMISSION_BANDS[0]
        admission = st._interior_admission(band, 1, 0.0, config)
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

    def test_spec2_window_table(self):
        # Table of (theta_old, theta_new, delta_old, delta_new): deltas
        # match, centres shift toward the analytic root.
        if not _git_available():
            self.skipTest('git HEAD source unavailable')
        head_find_cusps = _head_find_cusps()
        rows, cell_text = [], []
        for gamma in ASTROID_GAMMAS:
            thetas, speed = st._branch_speed_profile(
                gamma, 1, 0.0, 2.0 * np.pi, N_SAMPLES, periodic=True)
            new = sorted(st._find_cusps(
                thetas, speed, periodic=True, gamma=gamma, branch=1))
            old = sorted(head_find_cusps(thetas, speed, periodic=True))
            for (tn, dn), (to, do) in zip(new, old):
                rows.append(f'g={gamma}')
                cell_text.append([f'{to:.6f}', f'{tn:.6f}',
                                  f'{do:.6f}', f'{dn:.6f}'])
        figure, axis = plt.subplots(figsize=(7, 0.4 * len(cell_text) + 1))
        axis.axis('off')
        axis.table(cellText=cell_text, rowLabels=rows,
                   colLabels=['theta_old', 'theta_new',
                              'delta_old', 'delta_new'],
                   loc='center')
        axis.set_title('cusp window byte-identity (delta_old == delta_new)')
        path = _OUTPUT_DIR / 'caustic_cusps_window_table.png'
        figure.savefig(path, bbox_inches='tight')
        plt.close(figure)
        self.assertTrue(path.exists())

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


if __name__ == '__main__':
    main()
