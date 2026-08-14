"""D2 gauge-image tube serving + single-arc astroid training (WP1).

The Chang-Refsdal amplification is exactly D2-symmetric --
``F(w; y1, y2) = F(w; +/-y1, +/-y2)`` -- and the gauge-angle images of the
four source reflections are exactly ``theta``, ``pi - theta``
(``y1_eig -> -y1_eig``), ``-theta`` (``y2_eig -> -y2_eig``) and
``pi + theta`` (both).  The tube serve path therefore matches a query
against a chart by trying all four gauge images against the chart's
trained ``theta_grid`` frame, identity image first
(`surrogate._tube_theta_inframe`); the training campaign charges only ONE
canonical astroid arc (`surrogate_training._tube_training_arcs`) and the
image search recovers the three mirror copies (closes the F079 half-ring
hole).

WHY gauge images and not a sign-keyed fold: the gauge<->source map is
orientation-reversing (source angle ``~ pi - theta``), so the source-sign
octant does NOT identify the gauge arc, and near-cusp queries at caustic
distance eta sit slightly across the source axes, so a fixed-sign source
region spans slivers of THREE arcs.  A fold keyed on
``sign(y1_eig), sign(y2_eig)`` reflected charts' own queries off their
trained arcs (measured 2026-08-14: 0/10 held-out served, eps NaN).  The
frame-matched image search needs neither source signs nor a privileged
"fundamental" arc.

Pinned invariants (one pin each):

1. **Closed form of the image search.**  Identity image returned when in
   frame; each mirror image lands back on the same in-frame angle; a gauge
   angle none of whose images touch the frame returns ``None``.
2. **Incumbent bit-equality.**  A fundamental-domain query (identity
   image) serves BIT-IDENTICALLY to the pre-fold incumbent lookup
   (`_theta_into_frame` unwrap alone) -- the identity image is tried
   first, so the spline is evaluated at the exact same float coordinate.
3. **D2 serve equality.**  One physical query presented at all four
   eigenframe sign octants (each handed the gauge angle its geometry
   would report) serves the SAME amplification to a stated near-machine
   bound.  NOT bit-exact by design: a mirror query reaches the spline
   through a reflected float angle (``math.pi - theta`` and the ``% 2*pi``
   unwrap each round by ~1 ULP), so the interpolant input differs in the
   last bits.  The bound is ~10 orders tighter than the O(0.1) divergence
   a reflection-sign bug produces.
4. **Half-ring hole closure + teeth.**  All four octants serve through
   the real path; patching the image search back to identity-only
   (the incumbent) reopens the hole for the three mirrors while the
   fundamental octant still serves -- proving the closure comes from the
   image search and the pins have teeth.
5. **Training-arc selection.**  Astroid: exactly one arc, the canonical
   gauge arc bracketing ``pi/4`` (selected by theta-interval predicate,
   not slice position).  Saddle: the incumbent ``arcs[:max_tube_arcs]``
   slice -- the knob still governs the deltoid because ``max_eta_max``
   sized over ALL deltoid arcs (r_min ~3.5 outer vs ~0.28 lobe-edge)
   balloons the tube shell and starves the lobe admissions.
"""
import math
import unittest
from unittest import mock

import numpy as np

from cogwheel.lensing import surrogate as surrogate_module
from cogwheel.lensing.surrogate import (
    LensAmplificationSurrogate,
    TubeChart,
    _theta_into_frame,
    _tube_theta_inframe,
)
from cogwheel.lensing.surrogate_training import (
    band_caustic_structure,
    _tube_training_arcs,
)

#: Caustic-distance band served by the fixture tubes ``[eta_floor, eta_max]``.
ETA_FLOOR = 0.005
ETA_MAX = 0.05

#: A caustic distance strictly inside the served band (D2-invariant; the SAME
#: value passes for all four octants -- eta is never reflected).
QUERY_ETA = 0.02

#: Fixture ``ln w`` training band (``w in [0.5, 20]``) and the query
#: frequencies, both interior to it so the log-w band guard never fires.
LOG_W_GRID = np.log(np.geomspace(0.5, 20.0, 5))
W_ARRAY = np.geomspace(0.7, 15.0, 12)

#: Generic fundamental-frame gauge angles: interior to each fixture arc,
#: NOT on a diagonal (pi/4) and NOT in any cusp window.
ASTROID_THETA0 = 0.6            # inside the astroid arc [0.2, 1.2]
SADDLE_THETA0 = -0.24           # inside the saddle wedge arc [-0.39, -0.09]

#: The four eigenframe sign octants ``(sign y1_eig, sign y2_eig)``.
OCTANTS = ((+1.0, +1.0), (+1.0, -1.0), (-1.0, +1.0), (-1.0, -1.0))

#: Near-machine bound for octant serve equality.  A mirror query reaches
#: the spline through ``math.pi - theta`` / ``pi + theta`` and the
#: ``% 2*pi`` unwrap, each rounding by <= 1 ULP (~2.2e-16 rad); through the
#: smooth arc-length interp + tensor spline that moves the served value by
#: O(1e-13) at most (measured ~1e-13 on real trained charts, 2026-08-14).
OCTANT_RTOL = 1e-12
OCTANT_ATOL = 1e-14

#: Query gammas interior to each fixture tube's gamma band.
ASTROID_GAMMA_QUERY = 0.35     # inside [0.2, 0.5]
SADDLE_GAMMA_QUERY = 1.25      # inside [1.1, 1.4]

#: Real caustic-structure bands for the training-arc selection pins.
ASTROID_BAND = (0.35, 0.45)
SADDLE_BAND = (1.1, 1.15)
N_CAUSTIC_SAMPLES = 200


def _smooth_tensor(gamma_grid: np.ndarray, u_grid: np.ndarray,
                   theta_grid: np.ndarray, log_w_grid: np.ndarray,
                   phase: float) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic smooth ``(n_w, n_gamma, n_u, n_theta)`` real/imag tensors.

    A closed-form analytic surface (products of low-frequency sinusoids and
    exponentials) that the tensor-cubic spline fits stably.  ``phase``
    decorrelates the two fixture charts.  The absolute values carry no
    physical meaning -- these gates pin the D2 serve structure, not
    certified reconstruction accuracy.
    """
    grid_w, grid_g, grid_u, grid_t = np.meshgrid(
        log_w_grid, gamma_grid, u_grid, theta_grid, indexing='ij')
    real = (np.cos(0.5 * grid_w + phase) * (1.0 + 0.3 * grid_g)
            * np.exp(-0.4 * grid_u) * (1.0 + 0.2 * grid_t))
    imag = (np.sin(0.5 * grid_w + phase) * (1.0 - 0.2 * grid_g)
            * (1.0 + 0.1 * grid_u) * np.cos(0.3 * grid_t))
    return real, imag


def _astroid_surrogate() -> LensAmplificationSurrogate:
    """A single positive-parity astroid TubeChart surrogate (no engine call).

    Arc ``theta in [0.2, 1.2]`` bracketing the generic query angle
    ``ASTROID_THETA0``; no cusp windows so the generic query is never
    excluded.
    """
    gamma = np.linspace(0.2, 0.5, 4)
    theta = np.linspace(0.2, 1.2, 4)
    u_grid = np.linspace(np.sqrt(ETA_FLOOR), np.sqrt(ETA_MAX), 4)
    real, imag = _smooth_tensor(gamma, u_grid, theta, LOG_W_GRID, 0.0)
    tube = TubeChart.from_values(
        gamma_grid=gamma, u_grid=u_grid, theta_grid=theta,
        log_w_grid=LOG_W_GRID, envelope_real=real, envelope_imag=imag,
        image_count=2, parity=1, eta_floor=ETA_FLOOR, eta_max=ETA_MAX)
    return LensAmplificationSurrogate([tube], {'chart_count': 1})


def _saddle_surrogate() -> LensAmplificationSurrogate:
    """A single saddle-parity deltoid TubeChart surrogate (no engine call).

    NEGATIVE-wedge arc ``theta in [-0.39, -0.09]`` so a ``[0, 2*pi)``
    gauge angle routes through the ``_theta_into_frame`` unwrap to select it.
    """
    gamma = np.linspace(1.1, 1.4, 4)
    theta = np.linspace(-0.39, -0.09, 4)
    u_grid = np.linspace(np.sqrt(ETA_FLOOR), np.sqrt(ETA_MAX), 4)
    real, imag = _smooth_tensor(gamma, u_grid, theta, LOG_W_GRID, 1.0)
    tube = TubeChart.from_values(
        gamma_grid=gamma, u_grid=u_grid, theta_grid=theta,
        log_w_grid=LOG_W_GRID, envelope_real=real, envelope_imag=imag,
        image_count=4, parity=-1, eta_floor=ETA_FLOOR, eta_max=ETA_MAX)
    return LensAmplificationSurrogate([tube], {'chart_count': 1})


def _octant_physical_theta(theta0: float, sign_y1: float,
                           sign_y2: float) -> float:
    """The physical caustic gauge angle an octant's geometry would report.

    The gauge image of the source reflection: ``y1 -> -y1`` maps
    ``theta -> pi - theta`` and ``y2 -> -y2`` maps ``theta -> -theta``
    (orientation-reversing caustic map).  A real caller's
    `geometry.nearest_caustic_point` reports this mirrored angle; the serve
    path's image search must map all four back onto the same trained arc.
    """
    if sign_y1 > 0.0 and sign_y2 > 0.0:
        return theta0
    if sign_y1 > 0.0 and sign_y2 < 0.0:
        return -theta0
    if sign_y1 < 0.0 and sign_y2 > 0.0:
        return math.pi - theta0
    return math.pi + theta0


def _serve_at_octants(surrogate: LensAmplificationSurrogate, *, gamma: float,
                      theta0: float, image_count: int
                      ) -> dict[tuple[float, float], tuple[np.ndarray, bool]]:
    """Serve the SAME physical query at all four eigenframe sign octants.

    ``beta = 0`` so the eigenframe signs equal ``sign(y1), sign(y2)``; each
    octant is handed the physical gauge angle its geometry would report
    (`_octant_physical_theta`).  ``eta`` is D2-invariant and passes as-is.
    Returns ``{(sign_y1, sign_y2): (E_array, served)}``.
    """
    out: dict[tuple[float, float], tuple[np.ndarray, bool]] = {}
    for sign_y1, sign_y2 in OCTANTS:
        theta_phys = _octant_physical_theta(theta0, sign_y1, sign_y2)
        env, served, _definition = surrogate.serve(
            W_ARRAY, gamma=gamma, y1=sign_y1 * 1.0, y2=sign_y2 * 1.0,
            beta=0.0, eta=QUERY_ETA, theta=theta_phys, image_count=image_count)
        out[(sign_y1, sign_y2)] = (env, served)
    return out


def _identity_only_inframe(chart, theta):
    """The incumbent (pre-image-search) lookup: identity unwrap or decline.

    Used to (a) pin bit-equality of the fundamental-domain path and (b)
    reopen the half-ring hole, proving the closure comes from the image
    search.
    """
    frame_lo = float(chart.theta_grid[0])
    theta_inframe = _theta_into_frame(theta, frame_lo)
    if frame_lo <= theta_inframe <= float(chart.theta_grid[-1]):
        return theta_inframe
    return None


class _TubeD2TestCase(unittest.TestCase):
    """Base carrying the anti-vacuity comparison tally."""

    def setUp(self) -> None:
        self._n_comparisons = 0

    def _count(self, n: int = 1) -> None:
        self._n_comparisons += n

    def tearDown(self) -> None:
        if self._n_comparisons == 0:
            self.fail('anti-vacuity: no comparisons executed in this test')


class TubeThetaInframeClosedFormTestCase(_TubeD2TestCase):
    """Pin 1: closed form of the D2 gauge-image search."""

    def setUp(self) -> None:
        super().setUp()
        self.chart = _astroid_surrogate().charts[0]
        self.frame_lo = float(self.chart.theta_grid[0])

    def test_identity_image_when_in_frame(self) -> None:
        """An in-frame gauge angle returns its plain unwrap, bit-equal."""
        base = _tube_theta_inframe(self.chart, ASTROID_THETA0)
        self._count()
        self.assertEqual(base,
                         _theta_into_frame(ASTROID_THETA0, self.frame_lo))

    def test_mirror_images_land_on_the_same_angle(self) -> None:
        """Each of the three mirror gauge images maps back in-frame onto the
        base angle to near-machine (the reflection arithmetic rounds by
        ~1 ULP, so exact float equality is not claimed here)."""
        base = _tube_theta_inframe(self.chart, ASTROID_THETA0)
        for mirrored in (math.pi - ASTROID_THETA0, -ASTROID_THETA0,
                         math.pi + ASTROID_THETA0):
            got = _tube_theta_inframe(self.chart, mirrored)
            self._count()
            self.assertIsNotNone(
                got, f'image search declined mirrored angle {mirrored}')
            self.assertTrue(
                math.isclose(got, base, rel_tol=1e-12),
                f'mirrored {mirrored} -> {got}, expected ~{base}')

    def test_returns_none_when_no_image_touches_frame(self) -> None:
        """A gauge angle in the inter-arc gap (all four images outside the
        trained frame) declines with ``None``."""
        # 1.5 rad: images {1.5, pi-1.5, -1.5, pi+1.5} all avoid [0.2, 1.2].
        self._count()
        self.assertIsNone(_tube_theta_inframe(self.chart, 1.5))


class IncumbentIdentityBitEqualityTestCase(_TubeD2TestCase):
    """Pin 2: fundamental-domain queries serve bit-identically to the
    incumbent.

    The identity image is tried FIRST in `_tube_theta_inframe`, so any
    query the pre-image-search code served reaches the spline at the exact
    same float coordinate: the served arrays must be BIT-identical.
    """

    def test_fundamental_serve_is_bit_identical_to_incumbent(self) -> None:
        for name, surrogate, gamma, theta0, image_count in (
                ('astroid', _astroid_surrogate(), ASTROID_GAMMA_QUERY,
                 ASTROID_THETA0, 2),
                ('saddle', _saddle_surrogate(), SADDLE_GAMMA_QUERY,
                 SADDLE_THETA0, 4)):
            env_now, served_now, _ = surrogate.serve(
                W_ARRAY, gamma=gamma, y1=1.0, y2=1.0, beta=0.0,
                eta=QUERY_ETA, theta=theta0, image_count=image_count)
            with mock.patch.object(surrogate_module, '_tube_theta_inframe',
                                   _identity_only_inframe):
                env_old, served_old, _ = surrogate.serve(
                    W_ARRAY, gamma=gamma, y1=1.0, y2=1.0, beta=0.0,
                    eta=QUERY_ETA, theta=theta0, image_count=image_count)
            self._count(2)
            self.assertTrue(served_now and served_old,
                            f'{name}: fundamental query must serve both ways')
            self.assertTrue(np.array_equal(env_now, env_old),
                            f'{name}: identity-image serve is not '
                            f'bit-identical to the incumbent lookup')


class TubeD2ServeEqualityTestCase(_TubeD2TestCase):
    """Pin 3: the four sign octants serve the same amplification.

    Near-machine, NOT bit-exact: mirror octants evaluate the spline at a
    REFLECTED float angle (``math.pi - theta`` / ``pi + theta`` and the
    ``% 2*pi`` unwrap each round by <= 1 ULP), so the interpolant inputs
    differ in the last bits.  ``OCTANT_RTOL`` absorbs that yet sits ~10
    orders below a reflection-sign bug's O(0.1) divergence.
    """

    def _check(self, name: str, surrogate: LensAmplificationSurrogate,
               gamma: float, theta0: float, image_count: int) -> None:
        results = _serve_at_octants(surrogate, gamma=gamma, theta0=theta0,
                                    image_count=image_count)
        base, base_served = results[(+1.0, +1.0)]
        self._count()
        self.assertTrue(base_served, f'{name}: fundamental octant must serve')
        for octant in OCTANTS[1:]:
            env, served = results[octant]
            self._count(2)
            self.assertTrue(served, f'{name}: octant {octant} not served')
            self.assertTrue(
                np.allclose(env, base, rtol=OCTANT_RTOL, atol=OCTANT_ATOL),
                f'{name}: octant {octant} served value diverges from the '
                f'fundamental octant beyond rtol={OCTANT_RTOL} '
                f'(max abs diff {np.max(np.abs(env - base)):.3e})')

    def test_astroid_d2_serve_equality(self) -> None:
        self._check('astroid', _astroid_surrogate(), ASTROID_GAMMA_QUERY,
                    ASTROID_THETA0, 2)

    def test_saddle_d2_serve_equality(self) -> None:
        self._check('saddle', _saddle_surrogate(), SADDLE_GAMMA_QUERY,
                    SADDLE_THETA0, 4)


class HalfRingHoleClosureTestCase(_TubeD2TestCase):
    """Pin 4: the image search closes F079, and the pin has teeth.

    With the real serve path all four octants serve (checked in Pin 3);
    here the image search is patched back to identity-only (the incumbent
    lookup) and the three mirror octants must STOP serving while the
    fundamental octant still serves -- the closure is the image search,
    not an accident of the fixtures.
    """

    def _check(self, name: str, surrogate: LensAmplificationSurrogate,
               gamma: float, theta0: float, image_count: int) -> None:
        with mock.patch.object(surrogate_module, '_tube_theta_inframe',
                               _identity_only_inframe):
            results = _serve_at_octants(surrogate, gamma=gamma,
                                        theta0=theta0,
                                        image_count=image_count)
        _env, base_served = results[(+1.0, +1.0)]
        self._count()
        self.assertTrue(
            base_served,
            f'{name}: control failed -- fundamental octant must serve even '
            f'with the incumbent identity-only lookup')
        for octant in OCTANTS[1:]:
            _env, served = results[octant]
            self._count()
            self.assertFalse(
                served,
                f'{name}: octant {octant} served WITHOUT the image search '
                f'-- the F079 closure pin has no teeth')

    def test_astroid_hole_reopens_without_image_search(self) -> None:
        self._check('astroid', _astroid_surrogate(), ASTROID_GAMMA_QUERY,
                    ASTROID_THETA0, 2)

    def test_saddle_hole_reopens_without_image_search(self) -> None:
        self._check('saddle', _saddle_surrogate(), SADDLE_GAMMA_QUERY,
                    SADDLE_THETA0, 4)


class TubeTrainingArcSelectionTestCase(_TubeD2TestCase):
    """Pin 5: training-arc selection on REAL band caustic structures."""

    def test_astroid_selects_exactly_the_pi_over_four_arc(self) -> None:
        """Astroid: exactly ONE arc, the one bracketing pi/4 in gauge angle,
        selected by theta-interval predicate regardless of max_tube_arcs."""
        structure = band_caustic_structure(
            ASTROID_BAND, 1, n_samples=N_CAUSTIC_SAMPLES)
        quarter_pi = 0.25 * math.pi
        for knob in (1, 20):
            selected = _tube_training_arcs(structure, 1, knob)
            self._count(2)
            self.assertEqual(
                len(selected), 1,
                f'astroid must select exactly one arc (got {len(selected)} '
                f'with max_tube_arcs={knob})')
            arc = selected[0]
            self.assertTrue(
                arc.theta_lo <= quarter_pi <= arc.theta_hi,
                f'selected arc [{arc.theta_lo:.4f}, {arc.theta_hi:.4f}] '
                f'does not bracket pi/4')

    def test_saddle_reconsumes_max_tube_arcs_slice(self) -> None:
        """Saddle: the incumbent ``arcs[:max_tube_arcs]`` slice -- the knob
        must still bound the deltoid arc set because ``max_eta_max`` sized
        over all arcs balloons the tube shell (outer-arc r_min ~3.5 vs
        lobe-edge ~0.28) and starves the lobe admissions."""
        structure = band_caustic_structure(
            SADDLE_BAND, -1, n_samples=N_CAUSTIC_SAMPLES)
        self._count()
        self.assertGreater(len(structure.arcs), 1,
                           'fixture band must expose multiple deltoid arcs')
        self._count(2)
        self.assertEqual(_tube_training_arcs(structure, -1, 1),
                         list(structure.arcs[:1]))
        self.assertEqual(_tube_training_arcs(structure, -1, 20),
                         list(structure.arcs))


if __name__ == '__main__':
    unittest.main()
