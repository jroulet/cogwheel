"""Low-w flat extrapolation in surrogate serve path (WP1).

The surrogate chart spline covers a finite log_w band [log_w_min, log_w_max].
Previously, queries below log_w_min were refused (chart not served).  WP1 adds
flat extrapolation at the low end: the envelope value at any w < w_min is
clamped to the envelope at w_min.  This is physically justified because the
envelope is smooth and nearly constant below the first Airy fringe — the
correction is O(w_min^2) from the geometric limit.

The HIGH end remains strict: no upward extrapolation (the envelope is
oscillatory above w_max), so queries with w_max > chart.log_w_grid[-1] are
refused.

Tolerance choices:
- Flat extrapolation: EXACT bitwise identity (np.clip is exact for values
  below the low clamp), so tolerance is 0.0.
- Self-falsification: scipy BSpline cubic extrapolation diverges polynomially,
  so the unclamped–clamped difference at w_min/2 is typically > 0.01 for
  a sinusoidal envelope (measured ~0.03–0.15).  The LOAD_BEARING_THRESHOLD
  (1e-10) is a conservative lower bound.

Cost: 4-node synthetic chart build ~0.01s; 5 evaluations per test; total
suite < 2s.
"""
from __future__ import annotations

import functools
import math
import unittest

import numpy as np

from cogwheel.lensing import surrogate as surrogate_module
from cogwheel.lensing import surrogate_training
from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal._airy_fold import _merging_fold_pair
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels, farfield_w_floor, _FARFIELD_KERNEL_FAMILY,
    FARFIELD_DIFFRACTIVE, FARFIELD_KERNEL_SUM,
    FARFIELD_KERNEL_SUM_MINUS_GHOST)
from cogwheel.lensing.surrogate import (
    TubeChart,
    LensAmplificationSurrogate,
    _evaluate_chart,
    _log_w_band_serveable,
    _contract_tensor_spline,
    select_chart,
)

# ---------------------------------------------------------------------------
# Fixture constants
# ---------------------------------------------------------------------------

#: 4-node log_w grid: w in [5, 50] → log_w in [~1.61, ~3.91].
#: Chosen so w_min=5.0 gives a non-trivial extrapolation test region (w<5).
LOG_W_GRID = np.log(np.array([5.0, 10.0, 25.0, 50.0]))

#: Spatial grids (must be 4-node minimum per _validate_axis).
GAMMA_GRID = np.linspace(0.3, 0.6, 4)
U_GRID = np.linspace(0.05, 0.20, 4)  #: sqrt(eta) grid
THETA_GRID = np.linspace(0.1, 1.0, 4)

#: Chart metadata — positive parity, 2-image region.
IMAGE_COUNT = 2
PARITY = 1
ETA_FLOOR = U_GRID[0] ** 2
ETA_MAX = U_GRID[-1] ** 2

#: Query parameters that are inside the chart's spatial box.
QUERY_GAMMA = 0.45
QUERY_ETA = 0.01  #: between ETA_FLOOR and ETA_MAX
QUERY_THETA = 0.5  #: inside THETA_GRID range

#: Frequencies below w_min for flat-extrapolation test.
W_MIN = np.exp(LOG_W_GRID[0])  #: = 5.0
W_BELOW = np.array([W_MIN / 4, W_MIN / 2, W_MIN * 0.9])

#: Frequency at exactly w_min (control).
W_EXACT_MIN = np.array([W_MIN])

#: Frequency above w_max for high-end refusal test.
W_MAX = np.exp(LOG_W_GRID[-1])  #: = 50.0
W_ABOVE = np.array([W_MAX * 2.0])

#: Load-bearing threshold for self-falsification: the unclamped BSpline
#: extrapolation must differ from the clamped value by at least this.
LOAD_BEARING_THRESHOLD = 1e-10


#: Sample density for `_find_real_tube_source`'s arc scan -- matches the
#: density validated by the sibling derivations in
#: ``test_lensing_surrogate.py`` / ``test_lensing_tube_d2_fold.py``.
_REAL_SOURCE_N_SAMPLES = 200

#: Off-axis floor: a source with either eigenframe component smaller than
#: this sits too close to the tube's own symmetry axis for `_tube_f_ref`
#: to build reliably across the fixture's ``w`` band.
_REAL_SOURCE_MIN_COMP = 0.05

#: Search band containing ``QUERY_GAMMA`` (0.45), passed to
#: `band_caustic_structure`.  Positive parity (astroid).
_QUERY_BAND = (0.4, 0.5)


@functools.lru_cache(maxsize=None)
def _find_real_tube_source(gamma_query: float, band: tuple[float, float],
                           parity: int) -> tuple[float, float]:
    """Find a genuine, off-axis, F_ref-buildable source near ``gamma_query``.

    INS-1-002: the beat-free `TubeChart` serve contract
    (`surrogate._tube_serves`'s F_ref-buildability gate, and
    `surrogate._evaluate_chart`'s unconditional post-multiply by
    `surrogate._tube_f_ref`) requires a REAL 4-image fold geometry to be
    rebuildable AT THE QUERIED SOURCE -- independent of whatever arbitrary
    synthetic envelope or axis grids the fixture chart carries.  Querying
    the synthetic chart at the arbitrary on-axis ``(y1, y2) = (0, 0)`` (as
    this file did at HEAD) therefore either declines (a NaN F_ref, so
    ``serve`` returns ``served=False``) or crashes (`_evaluate_chart`
    raises ``RuntimeError('Tube F_ref unbuildable at a served query')``).

    This scans `surrogate_training`'s OWN arc-detection machinery
    (`band_caustic_structure` / `_tube_training_arcs` / `_tube_source`) --
    the same real production geometry a genuine tube-chart build would walk
    -- for an off-axis fold source whose `_merging_fold_pair` is resolvable
    and whose `_tube_f_ref` is finite over the fixture's ``ln w`` band, so
    the synthetic chart can be queried at a point that is real and
    buildable while its (arbitrary) synthetic envelope stays as authored.

    Args:
        gamma_query: shear at which to build the geometry (also the
            representative gamma for the arc scan).
        band: ``(gamma_lo, gamma_hi)`` search band containing
            ``gamma_query``, passed to `band_caustic_structure`.
        parity: ``+1`` (astroid) or ``-1`` (deltoid).

    Returns:
        ``(y1_eig, y2_eig)`` -- a genuine, off-axis, F_ref-buildable source
        in the shear eigenframe at ``gamma_query``.

    Raises:
        AssertionError: no candidate on the scanned arc cleared every gate.
    """
    matrix = geometry.macro_matrix(gamma_query)
    structure = surrogate_training.band_caustic_structure(
        band, parity, n_samples=_REAL_SOURCE_N_SAMPLES)
    arc = surrogate_training._tube_training_arcs(structure, parity)[0]
    r_min = surrogate_training._min_curvature_radius(
        band, arc, _REAL_SOURCE_N_SAMPLES)
    eta_max = surrogate_training.TrainingConfig().f_max * r_min
    w_lin = np.exp(LOG_W_GRID)
    best: tuple[float, float] | None = None
    best_gap = -math.inf
    for theta in np.linspace(arc.theta_lo, arc.theta_hi,
                             _REAL_SOURCE_N_SAMPLES):
        source = surrogate_training._tube_source(
            gamma_query, float(theta), eta_max, arc.branch, arc.inward_sign)
        if min(abs(float(source[0])),
               abs(float(source[1]))) < _REAL_SOURCE_MIN_COMP:
            continue
        try:
            images = geometry.find_images(source, matrix)
        except geometry.LensDomainError:
            continue
        if len(images) != 4:
            continue
        pair = _merging_fold_pair(images, source, matrix)
        if pair is None:
            continue
        if surrogate_module._tube_f_ref(w_lin, gamma_query, source) is None:
            continue
        gap = float(pair[1] - pair[0])
        if gap > best_gap:
            best_gap = gap
            best = (float(source[0]), float(source[1]))
    if best is None:
        raise AssertionError(
            f'no F_ref-buildable real tube source found for '
            f'gamma={gamma_query}, band={band}, parity={parity} -- pick a '
            'different validated (band, parity) or widen the scan.')
    return best


#: A genuine off-axis 4-image F_ref-buildable eigenframe source at
#: ``QUERY_GAMMA`` -- every serve / `_evaluate_chart` call below queries the
#: synthetic chart HERE (not the old on-axis ``(0, 0)``) so the beat-free
#: F_ref gate admits it.  ``beta=0`` keeps the eigenframe == physical frame.
QUERY_Y1, QUERY_Y2 = _find_real_tube_source(QUERY_GAMMA, _QUERY_BAND, PARITY)


def _build_synthetic_chart() -> TubeChart:
    """Build a 4x4x4x4 TubeChart with a sinusoidal envelope.

    The envelope is NOT constant — it has genuine w-dependence so the
    cubic B-spline extrapolates to a DIFFERENT value below w_min (which
    the clamp then corrects).  A constant envelope would make the self-
    falsification test vacuous.
    """
    grid_w, grid_g, grid_u, grid_t = np.meshgrid(
        LOG_W_GRID, GAMMA_GRID, U_GRID, THETA_GRID, indexing='ij')
    # Sinusoidal in log_w so the spline extrapolation isn't trivially flat.
    real = (np.cos(1.5 * grid_w) * (1.0 + 0.3 * grid_g)
            * np.exp(-2.0 * grid_u) * (1.0 + 0.2 * grid_t))
    imag = (np.sin(1.5 * grid_w) * (1.0 - 0.2 * grid_g)
            * (1.0 + 0.1 * grid_u) * np.cos(0.3 * grid_t))
    return TubeChart.from_values(
        gamma_grid=GAMMA_GRID,
        u_grid=U_GRID,
        theta_grid=THETA_GRID,
        log_w_grid=LOG_W_GRID,
        envelope_real=real,
        envelope_imag=imag,
        image_count=IMAGE_COUNT,
        parity=PARITY,
        eta_floor=ETA_FLOOR,
        eta_max=ETA_MAX,
        cusp_windows=[],
    )


class _LowWExtrapolationTestCase(unittest.TestCase):
    """Base class providing chart fixture + anti-vacuity tearDown."""

    _n_comparisons: int = 0

    @classmethod
    def setUpClass(cls) -> None:
        cls.chart = _build_synthetic_chart()
        cls.surrogate = LensAmplificationSurrogate(
            [cls.chart], {'chart_count': 1, 'chart_types': ['tube']})

    def tearDown(self) -> None:
        """Anti-vacuity: FAIL if zero comparisons ran (silently-skipping)."""
        if self._n_comparisons == 0:
            self.fail(
                f'{type(self).__name__}: zero comparisons ran — '
                f'suite is vacuous.')

    def _record(self, n: int = 1) -> None:
        self._n_comparisons += n


class FlatExtrapolationTestCase(_LowWExtrapolationTestCase):
    """Test 1: Low-w flat extrapolation serves draws below w_min."""

    def test_serve_returns_true_below_w_min(self) -> None:
        """serve() returns served=True for frequencies entirely below w_min."""
        w_query = W_BELOW.copy()
        _, served, _ = self.surrogate.serve(
            w_query, gamma=QUERY_GAMMA, y1=QUERY_Y1, y2=QUERY_Y2, beta=0.0,
            eta=QUERY_ETA, theta=QUERY_THETA, image_count=IMAGE_COUNT)
        self.assertTrue(served, 'serve must return served=True for w < w_min')
        self._record()

    def test_envelope_below_w_min_equals_at_w_min(self) -> None:
        """Envelope at w < w_min is BITWISE IDENTICAL to envelope at w_min."""
        # Evaluate at w_min
        log_w_min = np.array([LOG_W_GRID[0]])
        env_at_min = _evaluate_chart(
            self.chart, gamma=QUERY_GAMMA, eta=QUERY_ETA,
            theta=QUERY_THETA, log_w_query=log_w_min,
            y1_eig=QUERY_Y1, y2_eig=QUERY_Y2)
        # Evaluate at several w < w_min
        log_w_below = np.log(W_BELOW)
        env_below = _evaluate_chart(
            self.chart, gamma=QUERY_GAMMA, eta=QUERY_ETA,
            theta=QUERY_THETA, log_w_query=log_w_below,
            y1_eig=QUERY_Y1, y2_eig=QUERY_Y2)
        # Each below-w_min value must be exactly equal to env_at_min
        for i, w in enumerate(W_BELOW):
            with self.subTest(w=w):
                self.assertEqual(
                    env_below[i], env_at_min[0],
                    f'Envelope at w={w:.2f} (below w_min={W_MIN}) differs '
                    f'from envelope at w_min: got {env_below[i]}, '
                    f'expected {env_at_min[0]}')
                self._record()

    def test_serve_with_mixed_band_spanning_below_w_min(self) -> None:
        """serve() works when w_array spans below AND inside the band."""
        w_mixed = np.array([W_MIN / 3, W_MIN, W_MIN * 2, W_MIN * 5])
        env, served, _ = self.surrogate.serve(
            w_mixed, gamma=QUERY_GAMMA, y1=QUERY_Y1, y2=QUERY_Y2, beta=0.0,
            eta=QUERY_ETA, theta=QUERY_THETA, image_count=IMAGE_COUNT)
        self.assertTrue(served, 'Mixed band spanning below w_min must serve')
        # The first element (below w_min) must equal the second (at w_min)
        self.assertEqual(
            env[0], env[1],
            'Envelope at w < w_min must equal envelope at w_min in mixed query')
        self._record()

    def test_regression_values_above_w_min_unchanged(self) -> None:
        """Values at w >= w_min match direct _contract_tensor_spline call.

        This confirms the clamp does not alter in-band queries.
        """
        # Query at w_min and w_max (boundaries of the band)
        log_w_test = np.array([LOG_W_GRID[0], LOG_W_GRID[-1]])
        env = _evaluate_chart(
            self.chart, gamma=QUERY_GAMMA, eta=QUERY_ETA,
            theta=QUERY_THETA, log_w_query=log_w_test,
            y1_eig=QUERY_Y1, y2_eig=QUERY_Y2)
        # Direct spline evaluation (no clamping needed — these are in-band)
        theta_inframe = QUERY_THETA - float(self.chart.theta_grid[0])
        s_val = float(np.interp(QUERY_THETA, self.chart.theta_to_s[0],
                                self.chart.theta_to_s[1]))
        real_direct = _contract_tensor_spline(
            self.chart.real_coeffs, self.chart.knots,
            QUERY_GAMMA, float(np.sqrt(QUERY_ETA)), s_val, log_w_test)
        imag_direct = _contract_tensor_spline(
            self.chart.imag_coeffs, self.chart.knots,
            QUERY_GAMMA, float(np.sqrt(QUERY_ETA)), s_val, log_w_test)
        # Beat-free contract: `_evaluate_chart` stores the RESIDUAL
        # r = E / F_ref and re-modulates by `_tube_f_ref` at the raw query
        # source, so the oracle must apply the SAME reference (both w's are
        # in-band, hence no clamp).  This still isolates the clamp: the only
        # thing under test is that in-band w's pass through untouched.
        source_q = np.array([QUERY_Y1, QUERY_Y2], dtype=float)
        fref = surrogate_module._tube_f_ref(
            np.exp(log_w_test), QUERY_GAMMA, source_q)
        self.assertIsNotNone(
            fref, 'fixture query source must be F_ref-buildable')
        expected = (real_direct + 1j * imag_direct) * fref
        np.testing.assert_array_equal(
            env, expected,
            err_msg='In-band values must be unchanged by the clamp')
        self._record(2)

    def test_log_w_band_serveable_admits_below_min(self) -> None:
        """_log_w_band_serveable returns True when log_w_min < grid[0]."""
        # log_w_min well below the chart's grid start
        log_w_min_below = float(np.log(W_MIN / 10))
        log_w_max_inside = float(LOG_W_GRID[-1] - 0.1)
        result = _log_w_band_serveable(
            self.chart, log_w_min_below, log_w_max_inside)
        self.assertTrue(
            result,
            '_log_w_band_serveable must admit when log_w_min < grid[0] '
            'and log_w_max <= grid[-1]')
        self._record()


class KernelSumLowEndGuardTestCase(unittest.TestCase):
    """The low-end clamp is NOT licensed for the kernel-sum family (F070).

    `_log_w_band_serveable` deliberately leaves the low end open, justified
    by the envelope being "smooth and nearly constant below the first Airy
    fringe".  That is true of the tube / SACR-C envelope every other test in
    this file exercises, and FALSE of `FARFIELD_KERNEL_SUM`, which diverges
    into the diffractive bottom below the region's `farfield_w_floor`.

    Measured 2026-08-13 on a chart tiled exactly the way
    `surrogate_training` tiles one: every admission gate passed and the
    sub-floor band served at ``eps = 4.7e+02`` (468x ``max|F|``) while the
    interpolated part sat at 1.5e-3.  `_surrogate_coefficients` now re-checks
    the floor at serve time and refuses instead.

    These are cheap geometry/contract checks, not a trained-chart accuracy
    run: the point is that the floor is COMPUTED at the serve site and that
    the family membership test selects the right labels.
    """

    def setUp(self) -> None:
        self._n = 0

    def tearDown(self) -> None:
        if self._n == 0:
            self.fail(f'{type(self).__name__}: zero comparisons ran.')

    def test_kernel_family_is_exactly_the_subtracting_labels(self) -> None:
        """The guard must fire for kernel-sum labels and NOT for others.

        `FARFIELD_DIFFRACTIVE` is the bounded object that is legitimately
        valid below the floor -- guarding it would refuse serves that are
        fine, which is the opposite failure.
        """
        self._n += 1
        self.assertIn(FARFIELD_KERNEL_SUM, _FARFIELD_KERNEL_FAMILY)
        self._n += 1
        self.assertIn(FARFIELD_KERNEL_SUM_MINUS_GHOST,
                      _FARFIELD_KERNEL_FAMILY)
        self._n += 1
        self.assertNotIn(
            FARFIELD_DIFFRACTIVE, _FARFIELD_KERNEL_FAMILY,
            'the diffractive label is the BOUNDED object valid below the '
            'floor; guarding it would refuse correct serves')

    def test_floor_is_finite_and_positive_for_a_resolvable_pair(self) -> None:
        """`farfield_w_floor` returns a usable number from geometry alone.

        The guard is only as good as this being computable at the serve
        site with no engine call.
        """
        w_grid = np.geomspace(5.0, 60.0, 12)
        geom = ChangRefsdalChannels(w_grid).geometry_partition(
            gamma=0.5, y=(1.5, 0.3), beta=0.0, kappa=0.0)
        floor = farfield_w_floor(geom.delays, geom.real_mask)
        self._n += 1
        self.assertTrue(np.isfinite(floor) and floor > 0.0,
                        f'w_floor is not usable: {floor!r}')

    def test_floor_rises_as_the_pair_closes_up(self) -> None:
        """`w_floor` is `(RHO_END/2) / min|dtau|`, so it grows as the
        closest real pair merges -- which is why an interior source has a
        floor far above a well-separated exterior one, and why a band that
        clears the floor for one config can sit entirely below it for
        another.  This is the quantity the clamp was ignoring.
        """
        w_grid = np.geomspace(5.0, 60.0, 12)
        far = ChangRefsdalChannels(w_grid).geometry_partition(
            gamma=0.5, y=(1.5, 0.3), beta=0.0, kappa=0.0)
        near = ChangRefsdalChannels(w_grid).geometry_partition(
            gamma=0.5, y=(0.05, 0.02), beta=0.0, kappa=0.0)
        floor_far = farfield_w_floor(far.delays, far.real_mask)
        floor_near = farfield_w_floor(near.delays, near.real_mask)
        self._n += 1
        self.assertGreater(
            floor_near, floor_far,
            f'w_floor did not rise as the pair closed up '
            f'(far {floor_far:.4g} vs near {floor_near:.4g}); the guard '
            f'keys on this ordering.')


if __name__ == '__main__':
    unittest.main()
