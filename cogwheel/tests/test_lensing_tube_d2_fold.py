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
5. **Training-arc selection (astroid).**  Exactly one arc, the canonical
   gauge arc bracketing ``pi/4`` (selected by a theta-interval predicate,
   not slice position).  The retired ``max_tube_arcs`` knob no longer
   appears in the signature (``_tube_training_arcs(structure, parity)``).
6. **Saddle orbit-partition selection.**  The six detected deltoid arcs
   collapse to ONE representative per D2 gauge orbit derived from the fold
   law ``{theta, pi - theta, -theta, pi + theta}``; the retained count is
   COMPUTED from the partition, never hard-coded (measured ``6 -> 2`` on
   the fixture band -- the four branch ``+1`` lobe-edge arcs share one
   orbit, the two branch ``-1`` arcs at gauge ``0`` / ``pi`` the other).
   Every detected arc is D2-equivalent to exactly one representative and
   the representatives are pairwise non-equivalent.  Teeth: defeating the
   D2 coincidence test degenerates the trim to the identity (``6 -> 6``).
7. **Serve-coverage preservation under D2 folding** (the symmetry equality
   pin).  Sweeping a dense gauge ring, every theta the all-six-arc
   incumbent tube set serves is still served by the trimmed fundamental
   set through `_tube_theta_inframe` -- the fundamental served-theta set
   is a SUPERSET of the incumbent's, so the trim introduces NO new
   unserved band.  Teeth: dropping any single representative strands a
   band the folding cannot recover.
8. **Per-arc lobe-edge shell, NOT band-wide max** (F081 Part B; the
   anisotropic-shell pin).  The macro-saddle tube shell excluded from the
   lobe interior (and added to the far-field inner edge ``exclusion_rho``)
   is sized by the SMALLEST arc curvature radius over the band --
   ``min_eta_max = f_max * min(arc_r_min)`` (the tightly-curved lobe-edge
   arc) -- NOT the band-wide maximum ``f_max * max(arc_r_min)`` (the nearly
   straight outer arc).  An isotropic band-wide-max shell would over-exclude
   a wide annulus of genuinely served interior around the lobe-edge caustic.
   Witness: a synthetic point at caustic distance ``d`` with
   ``min_eta_max < d < max_eta_max`` is ADMITTED by `_SaddleLobeAdmission`
   under the shipped ``min_eta_max`` shell and clears the far-field inner
   edge, yet FLIPS to excluded the instant the shell is widened to
   ``max_eta_max`` (self-falsification: the reverted band-wide-max code
   wrongly drops it).  Also: `_saddle_lobe_admissions` fed the shipped shell
   builds ``corridor_half == _INTERLOBE_CORRIDOR_ETA_SCALE * f_max *
   min(arc_r_min)``, distinct from the ``* max`` value.  Engine-free real
   saddle-band geometry; no census-count or coverage test pins it precisely.
"""
import dataclasses
import functools
import math
import pathlib
import unittest
from unittest import mock

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - diagnostics are best-effort
    _HAVE_MPL = False

from cogwheel.lensing import surrogate as surrogate_module
from cogwheel.lensing import surrogate_training as surrogate_training_module
from cogwheel.lensing.surrogate import (
    LensAmplificationSurrogate,
    TubeChart,
    _theta_into_frame,
    _tube_f_ref,
    _tube_theta_inframe,
)
from cogwheel.lensing.surrogate_training import (
    TrainingConfig,
    band_caustic_structure,
    _INTERLOBE_CORRIDOR_ETA_SCALE,
    _coordinate_radius_bounds,
    _min_curvature_radius,
    _saddle_lobe_admissions,
    _SaddleLobeAdmission,
    _tube_source,
    _tube_training_arcs,
)
from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal._airy_fold import _merging_fold_pair

#: Directory for diagnostic plots (created lazily; plotting is best-effort
#: and never gates an assertion).
_OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / 'output'

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

#: Real caustic bands used SOLELY to derive a genuine 4-image interior
#: source for the F_ref-buildability gate (`surrogate._tube_serves` /
#: `_evaluate_chart`) -- independent of each fixture chart's synthetic
#: (arbitrary sinusoidal) envelope, and of its gamma/theta/eta training box,
#: which the F_ref gate never consults (`_tube_f_ref` depends only on
#: ``(gamma, source)``).  Centred on the respective query gamma so the
#: derived source is valid there.
_ASTROID_SOURCE_BAND = (0.30, 0.40)
_SADDLE_SOURCE_BAND = (1.20, 1.30)

#: Minimum magnitude of BOTH eigenframe source components (mirrors
#: ``test_lensing_tube_beat_free._OFFAXIS_MIN_COMP``): an on-axis source
#: makes a D2 sign reflection coincide with the raw source, which would let
#: an octant-equality test pass vacuously even with a reflection-sign bug.
_SOURCE_MIN_COMP = 0.05


@functools.lru_cache(maxsize=None)
def _find_d2_source(gamma_query: float, band: tuple[float, float],
                    parity: int) -> tuple[float, float]:
    """A genuine 4-image interior source, off both eigenframe axes, whose
    ``F_ref`` builds across ``LOG_W_GRID`` at ``gamma_query``.

    The new `surrogate._tube_serves` buildability gate and the
    `_evaluate_chart` residual-to-envelope multiplication (checklist-5b)
    both require the eigenframe query source to correspond to a real
    4-image geometry with a valid merging fold pair; this file's tube
    fixtures carry synthetic envelopes with no such constraint on their own
    query source, so the D2 octant tests need an independently-derived
    physical source.  Scans a real fold arc detected on ``band`` (the same
    detection machinery `surrogate_training._tube_training_arcs` uses for
    training) for the largest-gap node clearing `_SOURCE_MIN_COMP` on BOTH
    eigenframe components.  Memoised: the same physical source is reused by
    every test in this module that needs it.
    """
    matrix = geometry.macro_matrix(gamma_query)
    structure = band_caustic_structure(band, parity,
                                       n_samples=N_CAUSTIC_SAMPLES)
    arc = _tube_training_arcs(structure, parity)[0]
    r_min = _min_curvature_radius(band, arc, N_CAUSTIC_SAMPLES)
    eta_max = TrainingConfig().f_max * r_min
    w_lin = np.exp(LOG_W_GRID)
    best: tuple[float, float] | None = None
    best_gap = -math.inf
    for theta in np.linspace(arc.theta_lo, arc.theta_hi, N_CAUSTIC_SAMPLES):
        source = _tube_source(gamma_query, float(theta), eta_max,
                              arc.branch, arc.inward_sign)
        if min(abs(float(source[0])), abs(float(source[1]))) < _SOURCE_MIN_COMP:
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
        if _tube_f_ref(w_lin, gamma_query, source) is None:
            continue
        gap = float(pair[1] - pair[0])
        if gap > best_gap:
            best_gap = gap
            best = (float(source[0]), float(source[1]))
    if best is None:
        raise AssertionError(
            'fixture premise lost: no off-axis 4-image interior source with'
            f' buildable F_ref found for gamma={gamma_query} on {band}.')
    return best


def _astroid_d2_source() -> tuple[float, float]:
    """Real ``(y1, y2)`` eigenframe source for `ASTROID_GAMMA_QUERY`."""
    return _find_d2_source(ASTROID_GAMMA_QUERY, _ASTROID_SOURCE_BAND, 1)


def _saddle_d2_source() -> tuple[float, float]:
    """Real ``(y1, y2)`` eigenframe source for `SADDLE_GAMMA_QUERY`."""
    return _find_d2_source(SADDLE_GAMMA_QUERY, _SADDLE_SOURCE_BAND, -1)


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
                      theta0: float, image_count: int, y1_mag: float,
                      y2_mag: float
                      ) -> dict[tuple[float, float], tuple[np.ndarray, bool]]:
    """Serve the SAME physical query at all four eigenframe sign octants.

    ``beta = 0`` so the eigenframe signs equal ``sign(y1), sign(y2)``; each
    octant is handed the physical gauge angle its geometry would report
    (`_octant_physical_theta`).  ``eta`` is D2-invariant and passes as-is.
    ``(y1_mag, y2_mag)`` are the ABSOLUTE component magnitudes of a genuine
    4-image interior source (`_find_d2_source`) -- the F_ref-buildability
    gate needs a real fold geometry, and D2 symmetry of the lens equation
    guarantees every sign reflection of a valid source is itself a valid
    4-image source at the same gamma (checklist-5b).
    Returns ``{(sign_y1, sign_y2): (E_array, served)}``.
    """
    out: dict[tuple[float, float], tuple[np.ndarray, bool]] = {}
    for sign_y1, sign_y2 in OCTANTS:
        theta_phys = _octant_physical_theta(theta0, sign_y1, sign_y2)
        env, served, _definition = surrogate.serve(
            W_ARRAY, gamma=gamma, y1=sign_y1 * y1_mag, y2=sign_y2 * y2_mag,
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


#: Circular tolerance (rad) for the INDEPENDENT orbit oracle's D2 midpoint
#: coincidence test.  Comfortably above the exact-to-float D2 coincidences
#: measured on the fixture band (partner midpoints agree to <1e-12) yet far
#: below the ~0.6 rad gap between distinct orbits, so the partition is not
#: sensitive to this choice.
_ORBIT_TOL = 0.05

#: Number of gauge angles swept over ``[0, 2*pi)`` for the serve-coverage
#: superset pin.  Dense enough to enter every fixture arc's frame yet trivial
#: to evaluate (pure ``_tube_theta_inframe`` geometry, no engine).
N_SERVE_RING = 720


def _circular_gap(a: float, b: float) -> float:
    """Independent oracle for the shortest angular distance between ``a`` and
    ``b`` on the circle.

    Re-derived here (NOT the production ``_circular_angular_distance``) so the
    orbit oracle does not gate the trim against its own helper.
    """
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _d2_gauge_images(theta: float) -> tuple[float, float, float, float]:
    """The four D2 gauge images of ``theta`` from the fold law:
    ``{theta, pi - theta, -theta, pi + theta}`` (identity first)."""
    return (theta, math.pi - theta, -theta, math.pi + theta)


def _arc_midpoint(arc) -> float:
    """Caustic-segment midpoint (gauge angle) of a detected fold arc."""
    return 0.5 * (arc.theta_lo + arc.theta_hi)


def _independent_orbit_labels(midpoints: list[float],
                              tol: float = _ORBIT_TOL) -> list[int]:
    """Union-find D2 orbit labels for arc ``midpoints`` (independent oracle).

    Two midpoints share an orbit iff one lands on the other under some D2
    gauge image within ``tol``.  Returns a canonical label per input (equal
    labels iff same orbit).  Built from scratch -- it does NOT call
    `_tube_training_arcs` or its production helper -- so it is a genuine
    cross-check of the trim's partition.
    """
    n = len(midpoints)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if any(_circular_gap(image, midpoints[j]) <= tol
                   for image in _d2_gauge_images(midpoints[i])):
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    canonical = {root: label for label, root in enumerate(sorted(set(roots)))}
    return [canonical[root] for root in roots]


def _d2_equivalent(theta_a: float, theta_b: float,
                   tol: float = _ORBIT_TOL) -> bool:
    """True iff ``theta_a`` coincides with a D2 gauge image of ``theta_b``
    (independent oracle, symmetric within ``tol``)."""
    return any(_circular_gap(image, theta_b) <= tol
               for image in _d2_gauge_images(theta_a))


def _tube_chart_for_arc(arc, *, parity: int, image_count: int,
                        gamma_lo: float, gamma_hi: float,
                        phase: float = 0.0) -> TubeChart:
    """A minimal serve-ready TubeChart whose ``theta_grid`` spans ``arc``.

    Only the trained ``theta_grid`` frame matters for serve-coverage
    geometry (`_tube_theta_inframe` reads the frame endpoints); the envelope
    is an arbitrary smooth surface.
    """
    gamma = np.linspace(gamma_lo, gamma_hi, 4)
    theta = np.linspace(arc.theta_lo, arc.theta_hi, 4)
    u_grid = np.linspace(np.sqrt(ETA_FLOOR), np.sqrt(ETA_MAX), 4)
    real, imag = _smooth_tensor(gamma, u_grid, theta, LOG_W_GRID, phase)
    return TubeChart.from_values(
        gamma_grid=gamma, u_grid=u_grid, theta_grid=theta,
        log_w_grid=LOG_W_GRID, envelope_real=real, envelope_imag=imag,
        image_count=image_count, parity=parity,
        eta_floor=ETA_FLOOR, eta_max=ETA_MAX)


def _set_serves(charts: list[TubeChart], theta: float) -> bool:
    """True iff ANY chart in the set has ``theta`` (via a D2 gauge image)
    inside its trained frame -- i.e. the tube set serves ``theta``."""
    return any(_tube_theta_inframe(chart, theta) is not None
               for chart in charts)


def _served_ring(charts: list[TubeChart],
                 ring: np.ndarray) -> np.ndarray:
    """Boolean served-flag over ``ring`` for a tube set."""
    return np.array([_set_serves(charts, float(theta)) for theta in ring])


def _save_orbit_plot(name: str, midpoints: list[float], labels: list[int],
                     rep_midpoints: list[float]) -> None:
    """Diagnostic: arc midpoints on the theta circle coloured by orbit, with
    retained representatives ringed.  Best-effort; never gates an assertion.
    """
    if not _HAVE_MPL:
        return
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(subplot_kw={'projection': 'polar'})
    mids = np.asarray(midpoints)
    axis.scatter(mids, np.ones_like(mids), c=labels, cmap='tab10', s=90,
                 zorder=3)
    reps = np.asarray(rep_midpoints)
    axis.scatter(reps, np.ones_like(reps), facecolors='none',
                 edgecolors='k', s=220, linewidths=1.8, zorder=4,
                 label='retained representative')
    axis.set_title(f'{name}: detected arc midpoints coloured by D2 orbit')
    axis.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    fig.savefig(_OUTPUT_DIR / f'{name}_orbit_partition.png', dpi=110,
                bbox_inches='tight')
    plt.close(fig)


def _save_serve_plot(name: str, ring: np.ndarray, incumbent: np.ndarray,
                     fundamental: np.ndarray) -> None:
    """Diagnostic: theta-vs-served overlay for the all-arc incumbent and the
    trimmed fundamental set.  A regression shows as a theta band the
    incumbent serves but the fundamental set does not.  Best-effort.
    """
    if not _HAVE_MPL:
        return
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 3))
    axis.fill_between(ring, 0, incumbent.astype(float), step='mid',
                      alpha=0.35, label='all-arc incumbent served')
    axis.step(ring, fundamental.astype(float) * 0.9, where='mid',
              color='C3', label='trimmed fundamental served (x0.9)')
    axis.set_xlabel('gauge theta [rad]')
    axis.set_ylabel('served flag')
    axis.set_title(f'{name}: serve-coverage preservation under D2 folding')
    axis.legend(loc='upper right')
    fig.savefig(_OUTPUT_DIR / f'{name}_serve_coverage.png', dpi=110,
                bbox_inches='tight')
    plt.close(fig)


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
        for name, surrogate, gamma, theta0, image_count, (y1, y2) in (
                ('astroid', _astroid_surrogate(), ASTROID_GAMMA_QUERY,
                 ASTROID_THETA0, 2, _astroid_d2_source()),
                ('saddle', _saddle_surrogate(), SADDLE_GAMMA_QUERY,
                 SADDLE_THETA0, 4, _saddle_d2_source())):
            env_now, served_now, _ = surrogate.serve(
                W_ARRAY, gamma=gamma, y1=y1, y2=y2, beta=0.0,
                eta=QUERY_ETA, theta=theta0, image_count=image_count)
            with mock.patch.object(surrogate_module, '_tube_theta_inframe',
                                   _identity_only_inframe):
                env_old, served_old, _ = surrogate.serve(
                    W_ARRAY, gamma=gamma, y1=y1, y2=y2, beta=0.0,
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
               gamma: float, theta0: float, image_count: int,
               y1_mag: float, y2_mag: float) -> None:
        results = _serve_at_octants(surrogate, gamma=gamma, theta0=theta0,
                                    image_count=image_count, y1_mag=y1_mag,
                                    y2_mag=y2_mag)
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
        y1, y2 = _astroid_d2_source()
        self._check('astroid', _astroid_surrogate(), ASTROID_GAMMA_QUERY,
                    ASTROID_THETA0, 2, abs(y1), abs(y2))

    def test_saddle_d2_serve_equality(self) -> None:
        y1, y2 = _saddle_d2_source()
        self._check('saddle', _saddle_surrogate(), SADDLE_GAMMA_QUERY,
                    SADDLE_THETA0, 4, abs(y1), abs(y2))


class TubeRawSourceRemodulationSpyTestCase(_TubeD2TestCase):
    """New pin: ``F_ref`` is recomputed at the RAW eigenframe query source.

    Distinct from -- and narrower than -- the D2 output-equality pin
    (`TubeD2ServeEqualityTestCase`, which asserts the four octants agree on
    the reconstructed physical ``E``).  This pin spies on the
    ``surrogate._tube_f_ref`` binding the serve path calls and asserts, for a
    query in a NON-fundamental D2 octant (negative ``y1_eig``), that
    ``_tube_f_ref`` is invoked with the RAW signed eigenframe
    ``(y1_eig, y2_eig)`` -- NOT the theta-folded ``(u, s)`` interpolation
    coordinate and NOT the fundamental-octant reflection
    ``(|y1_eig|, |y2_eig|)``.  Because ``F_ref`` is exactly D2-invariant
    (even in each source component), the theta-fold is applied ONLY to the
    residual interpolation coordinate, while the demodulation reference is
    rebuilt at the raw D2-invariant source.  A regression that folded the
    source BEFORE ``_tube_f_ref`` would still round-trip at the fundamental
    octant but silently corrupt off-fundamental serves; the captured-source
    assertion catches exactly that (its teeth are the negative-vs-folded
    first component).

    The second method proves the rebuilt ``F_ref`` is actually CONSUMED
    (``E = r * F_ref``) rather than computed and discarded: scaling the
    reference by a constant scales the served envelope by the same constant.
    """

    #: Non-fundamental octant with a negative FIRST eigenframe component.
    _OCTANT = (-1.0, +1.0)

    def _serve_nonfundamental(self, f_ref_impl):
        """Serve the astroid fixture at ``_OCTANT`` with ``surrogate._tube_f_ref``
        replaced by ``f_ref_impl`` (a spy/wrapper).  Returns
        ``(env, served, y1_mag, y2_mag)``.  The physical source is derived
        (and its ``F_ref`` cached) BEFORE the patch so the derivation never
        goes through the spy.
        """
        y1_mag = abs(_astroid_d2_source()[0])
        y2_mag = abs(_astroid_d2_source()[1])
        theta_phys = _octant_physical_theta(ASTROID_THETA0, *self._OCTANT)
        surrogate = _astroid_surrogate()
        with mock.patch.object(surrogate_module, '_tube_f_ref', f_ref_impl):
            env, served, _definition = surrogate.serve(
                W_ARRAY, gamma=ASTROID_GAMMA_QUERY,
                y1=self._OCTANT[0] * y1_mag, y2=self._OCTANT[1] * y2_mag,
                beta=0.0, eta=QUERY_ETA, theta=theta_phys, image_count=2)
        return env, served, y1_mag, y2_mag

    def test_tube_f_ref_called_with_raw_eigenframe_source(self) -> None:
        spy = mock.MagicMock(side_effect=_tube_f_ref)
        env, served, y1_mag, y2_mag = self._serve_nonfundamental(spy)
        self.assertTrue(served, 'non-fundamental octant must serve')
        # Isolate the `_evaluate_chart` re-modulation call: it passes the
        # query band (W_ARRAY, size 12), whereas the `_tube_serves` gate
        # probes the chart's own 5-node w grid -- both, per the shipped code,
        # pass the same raw source, but the spec pins the serve-layer call.
        eval_calls = [c for c in spy.call_args_list
                      if np.asarray(c.args[0]).size == W_ARRAY.size]
        self.assertTrue(
            eval_calls,
            '_tube_f_ref never called over the query w band -- the '
            '_evaluate_chart re-modulation did not run')
        raw = np.array([self._OCTANT[0] * y1_mag, self._OCTANT[1] * y2_mag])
        folded = np.abs(raw)
        # Fixture premise: the octant is genuinely non-fundamental.
        self.assertLess(raw[0], 0.0,
                        'fixture premise lost: octant is not non-fundamental')
        for call in eval_calls:
            self._count()
            source = np.asarray(call.args[2], dtype=float)
            self.assertTrue(
                np.array_equal(source, raw),
                f'F_ref rebuilt at {source}, not the RAW eigenframe {raw}')
            self.assertLess(
                source[0], 0.0,
                'F_ref reference lost the negative eigenframe component '
                '(source was folded/absoluted before demodulation)')
            self.assertFalse(
                np.array_equal(source, folded),
                'F_ref rebuilt at the folded fundamental image '
                f'{folded} -- source was folded before demodulation')

    def test_tube_f_ref_return_is_consumed_multiplicatively(self) -> None:
        # Baseline serve through the real reference (wrapped, not altered).
        base_spy = mock.MagicMock(side_effect=_tube_f_ref)
        env_base, served_base, _y1, _y2 = self._serve_nonfundamental(base_spy)
        self.assertTrue(served_base, 'baseline non-fundamental serve failed')

        scale = 3.0

        def scaled(w_grid, gamma, source):
            # Calls the top-level imported real `_tube_f_ref` (the module
            # attribute is patched, this name is not -> no recursion).
            fref = _tube_f_ref(w_grid, gamma, source)
            return None if fref is None else scale * fref

        env_scaled, served_scaled, _y1b, _y2b = \
            self._serve_nonfundamental(scaled)
        self._count()
        self.assertTrue(
            served_scaled,
            'scaling F_ref must not change the serve decision (still non-None)')
        # E = r * F_ref: the residual r is unaffected by scaling F_ref, so the
        # served envelope must scale by exactly `scale`.
        self._count()
        self.assertTrue(
            np.allclose(env_scaled, scale * env_base, rtol=1e-10, atol=0.0),
            'served envelope did not scale with F_ref -- E = r * F_ref '
            'consumption is broken (max rel dev '
            f'{np.max(np.abs(env_scaled - scale * env_base)):.3e})')
        # Teeth: had F_ref been discarded (E = r), the two serves would be
        # identical.
        self._count()
        self.assertFalse(
            np.allclose(env_scaled, env_base, rtol=1e-6, atol=0.0),
            'scaling F_ref left the envelope unchanged -- F_ref is not '
            'consumed on the serve path')


class HalfRingHoleClosureTestCase(_TubeD2TestCase):
    """Pin 4: the image search closes F079, and the pin has teeth.

    With the real serve path all four octants serve (checked in Pin 3);
    here the image search is patched back to identity-only (the incumbent
    lookup) and the three mirror octants must STOP serving while the
    fundamental octant still serves -- the closure is the image search,
    not an accident of the fixtures.
    """

    def _check(self, name: str, surrogate: LensAmplificationSurrogate,
               gamma: float, theta0: float, image_count: int,
               y1_mag: float, y2_mag: float) -> None:
        with mock.patch.object(surrogate_module, '_tube_theta_inframe',
                               _identity_only_inframe):
            results = _serve_at_octants(surrogate, gamma=gamma,
                                        theta0=theta0,
                                        image_count=image_count,
                                        y1_mag=y1_mag, y2_mag=y2_mag)
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
        y1, y2 = _astroid_d2_source()
        self._check('astroid', _astroid_surrogate(), ASTROID_GAMMA_QUERY,
                    ASTROID_THETA0, 2, abs(y1), abs(y2))

    def test_saddle_hole_reopens_without_image_search(self) -> None:
        y1, y2 = _saddle_d2_source()
        self._check('saddle', _saddle_surrogate(), SADDLE_GAMMA_QUERY,
                    SADDLE_THETA0, 4, abs(y1), abs(y2))


class TubeTrainingArcSelectionTestCase(_TubeD2TestCase):
    """Pin 5: astroid training-arc selection on a REAL band structure.

    The retired ``max_tube_arcs`` knob no longer appears in the signature:
    `_tube_training_arcs` is called ``(structure, parity)`` and the astroid
    choice is a pure theta-interval predicate.
    """

    def test_astroid_selects_exactly_the_pi_over_four_arc(self) -> None:
        """Astroid: exactly ONE arc, the one bracketing pi/4 in gauge angle,
        selected by a theta-interval predicate (2-arg signature)."""
        structure = band_caustic_structure(
            ASTROID_BAND, 1, n_samples=N_CAUSTIC_SAMPLES)
        quarter_pi = 0.25 * math.pi
        self._count()
        self.assertGreater(
            len(structure.arcs), 1,
            'fixture astroid band must expose multiple gauge-image arcs so '
            'the single-arc trim is non-trivial')
        selected = _tube_training_arcs(structure, 1)
        self._count()
        self.assertEqual(
            len(selected), 1,
            f'astroid must select exactly one arc (got {len(selected)})')
        arc = selected[0]
        self._count()
        self.assertTrue(
            arc.theta_lo <= quarter_pi <= arc.theta_hi,
            f'selected arc [{arc.theta_lo:.4f}, {arc.theta_hi:.4f}] '
            f'does not bracket pi/4')

    def test_astroid_rejects_the_third_positional_argument(self) -> None:
        """The retired knob is gone: a third positional argument now raises
        ``TypeError`` -- proving the signature really is 2-arg and no dead
        knob path lingers."""
        structure = band_caustic_structure(
            ASTROID_BAND, 1, n_samples=N_CAUSTIC_SAMPLES)
        self._count()
        with self.assertRaises(TypeError):
            _tube_training_arcs(structure, 1, 20)


class SaddleOrbitPartitionSelectionTestCase(_TubeD2TestCase):
    """Pin 6: the saddle trim keeps exactly one arc per D2 gauge orbit.

    The retained count is COMPUTED from an independent union-find partition
    of the detected arc midpoints under the fold law, never hard-coded.
    Every detected arc is D2-equivalent to exactly one retained
    representative and the representatives are pairwise non-equivalent.
    Engine-free (band caustic-structure geometry only).
    """

    def setUp(self) -> None:
        super().setUp()
        self.structure = band_caustic_structure(
            SADDLE_BAND, -1, n_samples=N_CAUSTIC_SAMPLES)
        self.midpoints = [_arc_midpoint(arc) for arc in self.structure.arcs]
        self.labels = _independent_orbit_labels(self.midpoints)
        self.reps = _tube_training_arcs(self.structure, -1)
        self.rep_midpoints = [_arc_midpoint(arc) for arc in self.reps]

    def test_multiple_arcs_detected(self) -> None:
        """Premise: the fixture band exposes several deltoid arcs so the
        collapse is non-trivial (guards against a degenerate band silently
        making the trim a no-op)."""
        self._count()
        self.assertGreater(
            len(self.structure.arcs), 1,
            'fixture saddle band must expose multiple deltoid arcs')

    def test_retained_count_equals_independent_orbit_count(self) -> None:
        """The number of retained representatives equals the orbit count from
        the INDEPENDENT partition -- computed, not hard-coded."""
        expected_orbits = len(set(self.labels))
        self._count()
        self.assertEqual(
            len(self.reps), expected_orbits,
            f'trim kept {len(self.reps)} arcs but the independent D2 '
            f'partition of midpoints {[round(m, 4) for m in self.midpoints]} '
            f'has {expected_orbits} orbits')
        # The collapse must be genuine (fewer reps than detected arcs), else
        # the pin would pass vacuously on a band with no redundancy.
        self._count()
        self.assertLess(len(self.reps), len(self.structure.arcs),
                        'expected a genuine 6 -> fewer collapse')

    def test_every_arc_maps_to_exactly_one_representative(self) -> None:
        """Each detected arc is D2-equivalent to exactly ONE retained
        representative -- no arc is orphaned or double-covered."""
        for midpoint in self.midpoints:
            matches = [rep_mid for rep_mid in self.rep_midpoints
                       if _d2_equivalent(midpoint, rep_mid)]
            self._count()
            self.assertEqual(
                len(matches), 1,
                f'arc midpoint {midpoint:.4f} is D2-equivalent to '
                f'{len(matches)} representatives {matches}, expected exactly '
                f'one')

    def test_representatives_pairwise_non_equivalent(self) -> None:
        """No two retained representatives are D2 gauge images of each other
        (the orbits are distinct)."""
        for i in range(len(self.rep_midpoints)):
            for j in range(i + 1, len(self.rep_midpoints)):
                self._count()
                self.assertFalse(
                    _d2_equivalent(self.rep_midpoints[i],
                                   self.rep_midpoints[j]),
                    f'representatives {self.rep_midpoints[i]:.4f} and '
                    f'{self.rep_midpoints[j]:.4f} are D2-equivalent -- the '
                    f'trim kept two members of one orbit')
        _save_orbit_plot('saddle', self.midpoints, self.labels,
                         self.rep_midpoints)


class SaddleOrbitPartitionSelfFalsificationTestCase(_TubeD2TestCase):
    """Pin 6 teeth: defeating the D2 coincidence test degenerates the trim to
    the identity.

    Patching the production ``_circular_angular_distance`` to always report a
    large distance makes every arc look like a fresh orbit, so the trim
    retains ALL detected arcs (``6 -> 6``).  This proves the collapse is
    driven by the D2 coincidence test, not by an incidental slice.
    """

    def test_no_merging_retains_every_arc(self) -> None:
        structure = band_caustic_structure(
            SADDLE_BAND, -1, n_samples=N_CAUSTIC_SAMPLES)
        n_detected = len(structure.arcs)
        with mock.patch.object(surrogate_training_module,
                               '_circular_angular_distance',
                               lambda a, b: 1.0e9):
            reps = _tube_training_arcs(structure, -1)
        self._count()
        self.assertEqual(
            len(reps), n_detected,
            f'with the coincidence test defeated the trim must keep all '
            f'{n_detected} arcs (got {len(reps)}) -- otherwise the real '
            f'6 -> 2 collapse has no teeth')


class SaddleServeCoveragePreservationTestCase(_TubeD2TestCase):
    """Pin 7: the trim introduces no new unserved theta band.

    The trimmed fundamental tube set's served-theta set is a SUPERSET of the
    all-six-arc incumbent's over a dense gauge ring -- every query the
    incumbent served is still served through `_tube_theta_inframe`'s D2 gauge
    images.  The symmetry equality pin: folding recovers exactly what the
    redundant mirror charts covered.  Engine-free serve-side geometry.
    """

    def setUp(self) -> None:
        super().setUp()
        self.structure = band_caustic_structure(
            SADDLE_BAND, -1, n_samples=N_CAUSTIC_SAMPLES)
        gamma_lo, gamma_hi = SADDLE_BAND
        self.incumbent = [
            _tube_chart_for_arc(arc, parity=-1, image_count=4,
                                gamma_lo=gamma_lo, gamma_hi=gamma_hi)
            for arc in self.structure.arcs]
        self.fundamental = [
            _tube_chart_for_arc(arc, parity=-1, image_count=4,
                                gamma_lo=gamma_lo, gamma_hi=gamma_hi)
            for arc in _tube_training_arcs(self.structure, -1)]
        self.ring = np.linspace(0.0, 2.0 * math.pi, N_SERVE_RING,
                                endpoint=False)
        self.incumbent_served = _served_ring(self.incumbent, self.ring)
        self.fundamental_served = _served_ring(self.fundamental, self.ring)

    def test_incumbent_serves_a_nontrivial_band(self) -> None:
        """Premise: the incumbent set serves a non-empty theta band, so the
        superset claim is not vacuous."""
        self._count()
        self.assertGreater(
            int(self.incumbent_served.sum()), 0,
            'incumbent tube set served nothing -- superset pin would be '
            'vacuous')

    def test_fundamental_served_is_superset_of_incumbent(self) -> None:
        """No theta the incumbent serves is left unserved by the trimmed
        fundamental set (SUPERSET; zero new unserved band)."""
        violations = np.logical_and(self.incumbent_served,
                                    np.logical_not(self.fundamental_served))
        n_violations = int(violations.sum())
        self._count(len(self.ring))
        self.assertEqual(
            n_violations, 0,
            f'{n_violations} of {int(self.incumbent_served.sum())} '
            f'incumbent-served angles became unserved after the trim -- the '
            f'fold dropped a gauge orbit it cannot recover')
        _save_serve_plot('saddle', self.ring, self.incumbent_served,
                         self.fundamental_served)


class SaddleServeCoverageSelfFalsificationTestCase(_TubeD2TestCase):
    """Pin 7 teeth: dropping any single representative strands a band.

    If a genuine fundamental representative is removed, the D2 folding can no
    longer recover its orbit and a band the incumbent served becomes
    unserved -- so the superset pin above is discriminating, not automatic.
    """

    def test_dropping_each_representative_creates_violations(self) -> None:
        structure = band_caustic_structure(
            SADDLE_BAND, -1, n_samples=N_CAUSTIC_SAMPLES)
        gamma_lo, gamma_hi = SADDLE_BAND
        incumbent = [
            _tube_chart_for_arc(arc, parity=-1, image_count=4,
                                gamma_lo=gamma_lo, gamma_hi=gamma_hi)
            for arc in structure.arcs]
        reps = _tube_training_arcs(structure, -1)
        fundamental = [
            _tube_chart_for_arc(arc, parity=-1, image_count=4,
                                gamma_lo=gamma_lo, gamma_hi=gamma_hi)
            for arc in reps]
        ring = np.linspace(0.0, 2.0 * math.pi, N_SERVE_RING, endpoint=False)
        incumbent_served = _served_ring(incumbent, ring)
        self.assertGreater(len(fundamental), 1,
                           'need >1 representative to test dropping one')
        for drop in range(len(fundamental)):
            reduced = [c for k, c in enumerate(fundamental) if k != drop]
            reduced_served = _served_ring(reduced, ring)
            violations = int(np.logical_and(
                incumbent_served,
                np.logical_not(reduced_served)).sum())
            self._count()
            self.assertGreater(
                violations, 0,
                f'dropping representative {drop} left the incumbent coverage '
                f'fully served -- that representative was redundant, so the '
                f'superset pin would not detect its loss')


def _saddle_shell_derivation() -> dict:
    """Real saddle-band tube-shell sizing quantities (engine-free).

    Reproduces the production derivation
    (`surrogate_training._train_band_charts`): detect the topology-stable
    macro-saddle band structure, select the D2-orbit tube-arc representatives,
    and size the per-arc curvature-relative tube shell.  The lobe-edge shell
    ``min_eta_max = f_max * min(arc_r_min)`` (the tightly-curved lobe-edge arc)
    is what the shipped code feeds `_saddle_lobe_admissions` and the far-field
    ``exclusion_rho``; ``max_eta_max = f_max * max(arc_r_min)`` (the nearly
    straight outer arc) is the retired isotropic band-wide-max shell.  Also
    returns the caustic coordinate bounds feeding ``exclusion_rho``.
    """
    config = TrainingConfig()
    structure = band_caustic_structure(
        SADDLE_BAND, -1, n_samples=N_CAUSTIC_SAMPLES)
    tube_arcs = _tube_training_arcs(structure, -1)
    arc_r_min = [
        _min_curvature_radius(SADDLE_BAND, arc, config.n_caustic_samples)
        for arc in tube_arcs]
    min_eta_max = config.f_max * min(arc_r_min)
    max_eta_max = config.f_max * max(arc_r_min)
    coordinate_radius_min, reach_max = _coordinate_radius_bounds(
        SADDLE_BAND, -1)
    return {
        'config': config, 'tube_arcs': tube_arcs, 'arc_r_min': arc_r_min,
        'min_eta_max': min_eta_max, 'max_eta_max': max_eta_max,
        'coordinate_radius_min': coordinate_radius_min,
        'reach_max': reach_max,
    }


def _circular_lobe(radius: float, n: int = 1440) -> np.ndarray:
    """Ordered CCW closed ring of ``n`` points on a circle of ``radius``.

    A synthetic single macro-saddle deltoid lobe standing in for both the
    winding loop and the caustic cloud.  ``n`` includes an exact sample at
    gauge angle ``0`` (``endpoint=False`` from ``0``), so an interior probe on
    the ``+x`` axis has an EXACT nearest caustic distance ``radius - r_probe``.
    """
    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])


def _build_lobe_witness(min_eta_max: float, max_eta_max: float):
    """Synthetic lobe admission + a witness at distance ``d`` between shells.

    Builds a circular synthetic lobe and a single interior probe whose nearest
    caustic distance ``d`` satisfies ``min_eta_max < d < max_eta_max`` (the
    geometric mean when ``max_eta_max`` is finite; ``4 * min_eta_max`` if the
    outer arc is straight, ``r_min = inf``).  The probe sits on the ``+x`` axis
    at an EXACT ring-sample angle so ``d = radius - r_probe`` exactly.  The
    other lobe is placed far away and the corridor half-width is small, so
    admission is decided PURELY by the tube-shell distance test -- the point
    flips from admitted (``eta_max = min_eta_max``) to excluded
    (``eta_max = max_eta_max``) on the shell size alone.

    Returns ``(admission, center, half, witness_xy, d, radius)`` with the
    admission carrying the shipped ``min_eta_max`` shell.
    """
    d = (math.sqrt(min_eta_max * max_eta_max)
         if math.isfinite(max_eta_max) else 4.0 * min_eta_max)
    radius = d + 2.0
    ring = _circular_lobe(radius)
    boundary_theta = np.linspace(-math.pi, math.pi, 65)[1:]
    boundary_r = np.full_like(boundary_theta, radius)
    admission = _SaddleLobeAdmission(
        centroid=np.zeros(2), other_centroid=np.array([1000.0, 0.0]),
        reach=radius, eta_max=min_eta_max, corridor_half=min_eta_max,
        loops=(ring,), caustic_cloud=ring,
        boundary_theta=boundary_theta, boundary_r=boundary_r)
    r_probe = radius - d
    center = (r_probe / radius, 0.0)   # (rho_lobe, theta_local)
    half = (0.0, 0.0)                  # collapse the 9 probes to the witness
    witness_xy = np.array([r_probe, 0.0])
    return admission, center, half, witness_xy, d, radius


def _save_shell_witness_plot(radius: float, min_eta_max: float,
                             max_eta_max: float,
                             witness_xy: np.ndarray) -> None:
    """Diagnostic: the lobe caustic ring, the lobe-edge (min) shell, the old
    band-wide-max shell, and the admitted witness (best-effort)."""
    if not _HAVE_MPL:  # pragma: no cover - diagnostics are best-effort
        return
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ang = np.linspace(0.0, 2.0 * math.pi, 400)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(radius * np.cos(ang), radius * np.sin(ang), 'k-',
            label='lobe caustic')
    for shell, style, lbl in (
            (min_eta_max, 'g--', 'lobe-edge shell (min, shipped)'),
            (max_eta_max if math.isfinite(max_eta_max) else radius,
             'r:', 'band-wide-max shell (retired)')):
        inner = radius - shell
        if inner > 0:
            ax.plot(inner * np.cos(ang), inner * np.sin(ang), style,
                    label=lbl)
    ax.plot(witness_xy[0], witness_xy[1], 'bo', ms=8, label='witness')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('F081 per-arc lobe-edge shell, not band-wide max')
    fig.savefig(_OUTPUT_DIR / 'saddle_lobe_edge_shell_witness.png', dpi=90)
    plt.close(fig)


class SaddleLobeEdgeShellTestCase(_TubeD2TestCase):
    """Pin 8: the macro-saddle tube shell tracks the SMALLEST arc curvature
    radius (the lobe-edge arc), not the band-wide maximum.

    Real saddle-band geometry sizes ``min_eta_max`` (lobe-edge) and
    ``max_eta_max`` (outer, nearly straight); a synthetic witness at a caustic
    distance strictly between them is served by the shipped ``min_eta_max``
    shell -- through both the per-lobe interior admission and the far-field
    inner-edge ``exclusion_rho`` -- and is WRONGLY dropped by the retired
    ``max_eta_max`` shell.  Engine-free.
    """

    def setUp(self) -> None:
        super().setUp()
        self.deriv = _saddle_shell_derivation()

    def test_arc_radii_span_min_strictly_below_max(self) -> None:
        """Premise (vacuity guard): the two tube arcs have DISTINCT curvature
        radii, so min-vs-max is a real anisotropy and every downstream flip is
        non-trivial."""
        arc_r_min = self.deriv['arc_r_min']
        self._count()
        self.assertGreaterEqual(
            len(arc_r_min), 2,
            'saddle band trimmed to <2 tube arcs -- the per-arc shell '
            'distinction is vacuous; the fixture band must retain the '
            'lobe-edge and outer orbits')
        self.assertLess(
            self.deriv['min_eta_max'], self.deriv['max_eta_max'],
            'lobe-edge and outer arc curvature radii coincide -- the '
            'band-wide-max reversion would be indistinguishable, so the pin '
            'has no teeth on this band')

    def test_corridor_half_uses_min_arc_radius_not_max(self) -> None:
        """`_saddle_lobe_admissions` fed the shipped lobe-edge shell builds
        ``corridor_half == _INTERLOBE_CORRIDOR_ETA_SCALE * f_max *
        min(arc_r_min)`` -- distinct from the band-wide-max value."""
        min_eta_max = self.deriv['min_eta_max']
        max_eta_max = self.deriv['max_eta_max']
        admissions = _saddle_lobe_admissions(
            SADDLE_BAND, self.deriv['config'], eta_max=min_eta_max)
        self.assertEqual(len(admissions), 2,
                         'expected exactly two macro-saddle lobe admissions')
        for adm in admissions:
            self._count()
            self.assertAlmostEqual(
                adm.corridor_half,
                _INTERLOBE_CORRIDOR_ETA_SCALE * min_eta_max, places=12,
                msg='corridor half-width must equal one lobe-edge tube shell')
            self.assertAlmostEqual(adm.eta_max, min_eta_max, places=12)
            # ... and NOT the retired isotropic band-wide-max shell.
            self.assertNotAlmostEqual(
                adm.corridor_half,
                _INTERLOBE_CORRIDOR_ETA_SCALE * max_eta_max, places=6,
                msg='corridor half-width tracked the band-wide MAX arc '
                    'radius -- the isotropic reversion')

    def test_witness_admitted_under_min_shell_excluded_under_max(self) -> None:
        """A point at caustic distance ``d`` with ``min < d < max`` is a served
        lobe interior under the shipped ``min_eta_max`` shell, and flips to
        excluded under the band-wide ``max_eta_max`` shell (self-falsifying)."""
        min_eta_max = self.deriv['min_eta_max']
        max_eta_max = self.deriv['max_eta_max']
        admission, center, half, witness_xy, d, radius = _build_lobe_witness(
            min_eta_max, max_eta_max)
        # Premise: the witness distance really lies between the two shells.
        nearest = float(np.hypot(
            admission.caustic_cloud[:, 0] - witness_xy[0],
            admission.caustic_cloud[:, 1] - witness_xy[1]).min())
        self.assertAlmostEqual(nearest, d, places=9)
        self.assertLess(min_eta_max, nearest)
        self.assertLess(nearest, max_eta_max)
        # Shipped lobe-edge shell: the interior point is served.
        self._count()
        self.assertTrue(
            admission.admits(center, half),
            'lobe-edge (min) shell wrongly excluded a served interior point')
        # Band-wide-max shell: the SAME point is dropped (the reversion bug).
        reverted = dataclasses.replace(admission, eta_max=max_eta_max)
        self._count()
        self.assertFalse(
            reverted.admits(center, half),
            'band-wide-max shell admitted the witness -- the two shells did '
            'not produce distinct admission, so the pin lacks teeth')
        _save_shell_witness_plot(radius, min_eta_max, max_eta_max, witness_xy)

    def test_farfield_inner_edge_admits_under_min_shell_excludes_max(
            self) -> None:
        """The far-field inner-edge ``exclusion_rho`` uses ``min_eta_max``:
        a child tile whose inner rho edge lies between the min-shell and
        max-shell ``exclusion_rho`` clears the shipped inner edge but not the
        band-wide-max one.

        Mirrors the production predicate ``child_rho - child_half_r >=
        exclusion_rho`` (`surrogate_training._subdivide_farfield_tile`).
        """
        min_eta_max = self.deriv['min_eta_max']
        max_eta_max = self.deriv['max_eta_max']
        base = 1.0 + self.deriv['reach_max'] - self.deriv[
            'coordinate_radius_min']
        exclusion_rho_min = base + min_eta_max        # shipped inner edge
        capped_max = (max_eta_max if math.isfinite(max_eta_max)
                      else min_eta_max + 4.0)
        exclusion_rho_max = base + capped_max         # band-wide-max edge
        self.assertLess(exclusion_rho_min, exclusion_rho_max)
        # A child tile inner edge strictly between the two exclusion radii.
        child_half_r = 0.0
        child_rho = 0.5 * (exclusion_rho_min + exclusion_rho_max)
        inner_edge = child_rho - child_half_r
        self._count()
        self.assertGreaterEqual(
            inner_edge, exclusion_rho_min,
            'the shipped lobe-edge exclusion_rho wrongly rejected an '
            'exterior tile')
        self._count()
        self.assertLess(
            inner_edge, exclusion_rho_max,
            'band-wide-max exclusion_rho admitted the tile -- the inner edge '
            'did not tighten with the larger shell, so the pin has no teeth')


class SaddleLobeEdgeShellSelfFalsificationTestCase(_TubeD2TestCase):
    """Pin 8 teeth: the flip is caused by shell ANISOTROPY, not by noise.

    Under a SINGLE isotropic shell the witness admission does not flip -- it is
    admitted at the small shell and (if the shell is grown to the max) excluded
    at the large one, but never differs BETWEEN two equal shells.  This proves
    the main pin's admit/exclude split is a genuine consequence of
    ``min(arc_r_min) < max(arc_r_min)``, not an unrelated artefact.
    """

    def setUp(self) -> None:
        super().setUp()
        self.deriv = _saddle_shell_derivation()

    def test_equal_shells_never_flip_the_witness(self) -> None:
        """With the SAME shell on both admissions the witness decision is
        identical -- so the main test's flip must come from the min/max
        difference, not from the witness construction."""
        min_eta_max = self.deriv['min_eta_max']
        max_eta_max = self.deriv['max_eta_max']
        admission, center, half, _witness, _d, _radius = _build_lobe_witness(
            min_eta_max, max_eta_max)
        both_min = dataclasses.replace(admission, eta_max=min_eta_max)
        self._count()
        self.assertEqual(
            admission.admits(center, half), both_min.admits(center, half),
            'two equal (min) shells disagreed on the witness -- admission is '
            'not a pure function of the shell size')
        both_max = dataclasses.replace(admission, eta_max=max_eta_max)
        reverted = dataclasses.replace(admission, eta_max=max_eta_max)
        self._count()
        self.assertEqual(
            reverted.admits(center, half), both_max.admits(center, half),
            'two equal (max) shells disagreed on the witness')

    def test_reverted_derivation_signature_excludes_and_widens_corridor(
            self) -> None:
        """The reverted band-wide-max code path (feeding ``max_eta_max`` to
        `_saddle_lobe_admissions`) both WIDENS the corridor to the max shell
        and, on the witness, excludes it -- the exact double signature the
        shipped ``min_eta_max`` wiring avoids."""
        min_eta_max = self.deriv['min_eta_max']
        max_eta_max = self.deriv['max_eta_max']
        reverted = _saddle_lobe_admissions(
            SADDLE_BAND, self.deriv['config'], eta_max=max_eta_max)
        for adm in reverted:
            self._count()
            self.assertAlmostEqual(
                adm.corridor_half,
                _INTERLOBE_CORRIDOR_ETA_SCALE * max_eta_max, places=12,
                msg='the reverted wiring must widen the corridor to the max '
                    'shell (this is the bug the pin guards against)')
        admission, center, half, _w, _d, _r = _build_lobe_witness(
            min_eta_max, max_eta_max)
        self._count()
        self.assertFalse(
            dataclasses.replace(admission, eta_max=max_eta_max).admits(
                center, half),
            'reverted band-wide-max shell served the witness -- no exclusion '
            'to falsify against')


if __name__ == '__main__':
    unittest.main()
