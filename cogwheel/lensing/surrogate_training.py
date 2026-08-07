"""Offline training driver for the global multi-chart lensing surrogate.

Builds the Build-8c `LensAmplificationSurrogate` artifact from the sampled
prior box read from the lens prior classes.  For each parity (astroid
``gamma < 1`` and macro-saddle ``gamma > 1``) and each external-shear band it
derives the caustic structure FROM THE GEOMETRY ENGINE -- cusps are located as
minima of the caustic speed ``|d caustic / d theta|`` and cross-checked against
the expected topology (4 cusps on the positive-parity astroid; 6 = 2 lobes x 3
on the saddle deltoid) -- then builds near-caustic TUBE charts per inter-cusp
fold arc (in caustic-adapted ``(log w, gamma, u = sqrt(eta), theta)``) and
raw-eigenframe FAR-FIELD charts per image-count region, packs them into the WP1
single-npz artifact via `LensAmplificationSurrogate.save`, and emits a
machine-usable JSON training report.

WHY this shape:

- The exact SACR-C engine costs milliseconds per envelope; a shipped surrogate
  serves the same envelope in microseconds.  The chart set is caustic-adapted
  because the envelope's parameter derivatives carry a sqrt-type fold
  singularity AT the caustic; the tube's ``u = sqrt(eta)`` coordinate
  linearizes it so a cubic spline stays smooth through the transition.
- The chart-building routine is PARITY-AGNOSTIC: the same code builds an
  astroid fold tube and a deltoid fold tube, differing only in the geometry
  (branch and ``theta`` arc) fed in.
- Ranges are NOT hard-coded: the prior box (``gamma`` range, the
  mass-conditioned source box, the ``w`` band from the mass range) is read from
  the lens prior classes.  Cusp counts are NOT hard-coded: they are measured
  from the engine's caustic speed and cross-checked against the topology.

The FULL-box production training run is a DEFERRED post-build driver step; the
in-build entry point runs only at smoke scale (a reduced grid).  Training is
resumable at per-chart-file granularity: each chart is written to its own
``.npz`` and skipped at loop start if that file already exists (a plain
existence check -- no within-chart progress manifest).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from cogwheel.lensing import prior as _lens_prior
from cogwheel.lensing.waveform import dimensionless_frequency
from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels, farfield_envelope_from_partition, farfield_w_floor,
    INTERIOR_SACR_C, FARFIELD_KERNEL_SUM)
from cogwheel.lensing.chang_refsdal._hyp1f1 import HypergeometricDomainError
from cogwheel.lensing.ppgo_map import (
    CertifiedPpgoMap, UNKNOWN, caustic_rho, get_certified_ppgo_map)
from cogwheel.lensing.surrogate import (
    ExteriorPolarChart, TubeChart, LensAmplificationSurrogate,
    _REFUSAL_ERRORS, _log_w_grid, _uniform_axis, _log_reach_gamma_axis,
    _caustic_reach as _scalar_caustic_reach, _from_caustic_fixed,
    _from_lobe_fixed, _lobe_boundary_radius, LobeInteriorChart,
    InteriorWedgeChart, _from_wedge_fixed,
    _wedge_theta_waist, _wedge_cusp_axis_map,
    CarrierDiscontinuityError)

#: Engine refusals treated conservatively as "do not serve here" during
#: training.  Extends the 8a surrogate refusal vocabulary with the point-mass
#: kernel's double-double ceiling error, which the engine can raise directly
#: from `ChangRefsdalChannels.evaluate` when ``w * |y|`` exceeds ~60.
_ENGINE_REFUSALS = tuple(_REFUSAL_ERRORS) + (HypergeometricDomainError,)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default detector frequency band (Hz) used to map the lens-mass range onto
#: the dimensionless-frequency band ``w``.  The band is a DATA property (not a
#: prior-class range), so it is a driver parameter with a standard default.
DEFAULT_F_LO_HZ = 20.0
DEFAULT_F_HI_HZ = 1024.0

#: Per-parity ``w`` ceilings with margin below the engine's hard limits
#: (operator wave branch ``w <= 500``; saddle Schwinger QD ``w <= 150``).
_POSITIVE_W_CEILING = 480.0
# 2 below W_CEILING_SCHWINGER_QD (150); the mpmath path handles w ∈ (60, 150].
_SADDLE_W_CEILING = 148.0

#: Tube shell sizing as dimensionless fractions of the LOCAL curvature radius
#: ``R_c``.  The absolute tube band ``[eta_floor, eta_max]`` is computed
#: per-arc as ``f * R_c`` in ``_train_band_charts``, so the shell scales with
#: geometry: tight where cusps sharpen, broad where the fold is gentle.
#: ``f_floor / f_max = 0.4`` preserves the original 0.02/0.05 design ratio.
_DEFAULT_F_FLOOR = 0.16
#: Outer fraction of curvature radius.  The foot-of-normal invertibility
#: invariant requires ``f_max < 0.5`` (asserted at training time).
_DEFAULT_F_MAX = 0.40
#: Minimum caustic distance a far-field chart serves at (tube/far-field seam).
_DEFAULT_FARFIELD_OVERLAP = 0.05

#: Cusp-window half-width = safety factor x measured dip half-width, floored.
_CUSP_WIDTH_SAFETY = 1.5
_CUSP_MIN_HALFWIDTH = 0.05
#: Inward nudge (rad) keeping the analytic-root brentq bracket strictly inside
#: the sampled interval, so it never lands on the diverging saddle wedge edge.
_CUSP_BRACKET_EPS = 1e-9
#: Saddle-only cusp-exclusion widening (Build 8g WP3).  The macro-saddle
#: deltoid lobes have shallow interior cusps and, crucially, wedge-edge
#: turnaround walls whose foot-of-normal map is near-singular; the astroid
#: siblings fit at ~1e-2 with the values above, but three saddle deltoid-arc
#: tube charts fit at eps 0.4..2.2 because their arcs are clipped to these
#: least-guarded ends.  Widening the saddle cusp windows (and guarding the
#: wedge edges, see `_saddle_arcs`) refuses the polluted near-vertex core so
#: the arm/ladder serves it -- a refusal-conservative narrowing.  These are
#: threaded ONLY through the saddle path; the astroid path keeps the constants
#: above verbatim so its charts stay byte-identical.
_SADDLE_CUSP_WIDTH_SAFETY = 2.5
_SADDLE_CUSP_MIN_HALFWIDTH = 0.08
#: Minimum distance (physical source-plane units) from a tile corner to an
#: astroid cusp vertex below which the tile is excluded from exterior charting.
_CUSP_EXCLUSION_DISTANCE = 0.2
#: Fractional shrink of each fold arc away from its bounding walls.
_ARC_MARGIN_FRAC = 0.03
#: Number of theta samples used to integrate the tube's arc-length axis map
#: ``s = integral |y'| dtheta`` across a fold arc (see `_tube_arc_length_map`).
_TUBE_ARC_MAP_SIZE = 2001
#: Margin below the double-double product ceiling ``w * |y| <= 60`` used to cap
#: each chart's ``w`` grid.  Mirrors the prior's mass coupling, which keeps
#: ``w * |y| <= ~55`` by construction (the mass-conditioned source scale), so a
#: chart never samples the (large-w, large-|y|) corner the engine refuses.
_DD_PRODUCT_MARGIN = 58.0
#: Innermost radial floor of the positive-parity wedge-interior tiler
#: (`_wedge_interior_tiles`), in caustic-relative ``r`` units (``r = |y| /
#: r_caustic``).  The degenerate astroid centre ``r -> 0`` -- where the wedge
#: angle ``theta_wedge`` is undefined and the four folds are equidistant (a
#: carrier-continuity trap) -- is excluded from every trained wedge chart and
#: served by the exact engine.  One percent of the caustic reach is a
#: negligible coverage sliver.
_WEDGE_R_MIN = 1e-2

#: Expected cusp counts by parity (astroid / deltoid, both lobes summed).
_EXPECTED_CUSPS = {1: 4, -1: 6}
#: S2-1 interior directional admission (frozen WP6).  Polar-angle nodes of the
#: band-minimum directional caustic boundary ``rho_boundary`` (`r_caustic`
#: sampled per band gamma); the outer edge of each candidate interior tile is
#: probed at ``_INTERIOR_EDGE_SAMPLES`` angles across its ``theta_c`` span so a
#: smooth diagonal minimum interior to a cusp-aligned tile is not missed.
_INTERIOR_BOUNDARY_NODES = 181
_INTERIOR_EDGE_SAMPLES = 5
#: S2-2 per-lobe saddle interior (frozen WP7).  Lens-plane angular centres of
#: the two macro-saddle deltoid lobes on the negative-eigenvalue (shear) axis
#: at ``beta = 0``; each lobe is swept over its critical wedge ``|sin 2 theta|
#: <= (1 - kappa) / |gamma|`` about its centre.
_SADDLE_LOBE_CENTERS = (0.0, math.pi)
#: Half-width of the excluded inter-lobe corridor, as a multiple of
#: ``eta_max``.  A source within ``_INTERLOBE_CORRIDOR_ETA_SCALE * eta_max``
#: (dimensionless ``y``) of the lobe-equidistance (perpendicular-bisector) line
#: has an ambiguous lobe assignment, so no tile is admitted there and no tile
#: straddles the inter-lobe line.  The width is tied to the tube-shell
#: half-width ``eta_max`` -- one shell around the symmetry line, the same
#: length scale that gates the near-caustic exclusion.
_INTERLOBE_CORRIDOR_ETA_SCALE = 1.0


class CausticTopologyError(ValueError):
    """Detected cusp count disagrees with the expected caustic topology.

    Raised when the number of caustic-speed minima found for a parity does not
    match the analytic expectation (4 astroid / 6 deltoid).  A mismatch means
    the caustic sampling or the engine geometry is wrong; it is a flagged
    error, never a silent pass.
    """


# ---------------------------------------------------------------------------
# Prior box (read from the prior classes -- no hard-coded ranges)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PriorBox:
    """Sampled prior box, read verbatim from the lens prior classes.

    Attributes
    ----------
    gamma_range : tuple[float, float]
        Reduced-shear range ``(low, high)`` spanning both parities.
    ln_m_lens_range : tuple[float, float]
        Log redshifted lens-mass range ``(low, high)`` in ``ln(Msun)``.
    u1_range, u2_range : tuple[float, float]
        Unit source-box ranges (scaled by the mass-conditioned factor).
    f_lo_hz, f_hi_hz : float
        Detector frequency band bounds (Hz) used for the ``w`` band.
    """

    gamma_range: tuple[float, float]
    ln_m_lens_range: tuple[float, float]
    u1_range: tuple[float, float]
    u2_range: tuple[float, float]
    f_lo_hz: float
    f_hi_hz: float

    @classmethod
    def from_prior_classes(cls, *, f_lo_hz: float = DEFAULT_F_LO_HZ,
                           f_hi_hz: float = DEFAULT_F_HI_HZ,
                           m_lens_range: tuple[float, float] | None = None
                           ) -> 'PriorBox':
        """Read the box from the lens prior classes.

        Parameters
        ----------
        f_lo_hz, f_hi_hz : float, optional
            Detector frequency band bounds (Hz); defaults 20 / 1024.
        m_lens_range : (float, float), optional
            Restrict the lens-mass range to ``(m_lo, m_hi)`` Msun instead of
            the full prior.  Used by per-region probes to train a single
            mass/w stratum (the DRY single-source alternative to hand-rolled
            probe pipelines); ``None`` uses the full prior mass range.
        """
        gamma_range = tuple(
            _lens_prior.UniformReducedShearPrior.range_dic['gamma'])
        ln_m = tuple(
            _lens_prior.UniformLensMassPrior.range_dic['ln_m_lens_msun'])
        if m_lens_range is not None:
            ln_m = (math.log(m_lens_range[0]), math.log(m_lens_range[1]))
        u1 = tuple(_lens_prior.UniformSourcePositionPrior.range_dic['u1'])
        u2 = tuple(_lens_prior.UniformSourcePositionPrior.range_dic['u2'])
        return cls(gamma_range=(float(gamma_range[0]), float(gamma_range[1])),
                   ln_m_lens_range=(float(ln_m[0]), float(ln_m[1])),
                   u1_range=(float(u1[0]), float(u1[1])),
                   u2_range=(float(u2[0]), float(u2[1])),
                   f_lo_hz=float(f_lo_hz), f_hi_hz=float(f_hi_hz))

    @property
    def m_lens_range(self) -> tuple[float, float]:
        """Redshifted lens-mass range ``(low, high)`` in solar masses."""
        return (float(np.exp(self.ln_m_lens_range[0])),
                float(np.exp(self.ln_m_lens_range[1])))

    @property
    def y_reach(self) -> float:
        """Maximum eigenframe source displacement ``max|y|`` over the box.

        The source-position scale ``Y(m) = min(_Y_SCALE / m, _Y_SCALE_CAP)``
        is largest at the smallest lens mass, so the eigenframe box reaches
        ``u_max * Y(m_min)``.
        """
        u_max = max(abs(v) for v in self.u1_range + self.u2_range)
        return float(u_max * _lens_prior._source_scale(self.m_lens_range[0]))

    def w_range(self, parity: int) -> tuple[float, float]:
        """Dimensionless-frequency band ``(w_min, w_max)`` for a parity.

        ``w_min`` maps the lowest frequency at the lightest lens; ``w_max`` the
        highest frequency at the heaviest lens, clipped to the parity's engine
        ceiling.
        """
        m_lo, m_hi = self.m_lens_range
        w_min = float(dimensionless_frequency(self.f_lo_hz, m_lo, 0.0))
        w_max = float(dimensionless_frequency(self.f_hi_hz, m_hi, 0.0))
        ceiling = _POSITIVE_W_CEILING if parity == 1 else _SADDLE_W_CEILING
        return (w_min, min(w_max, ceiling))


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    """Grid sizing and budgets for a training run (smoke defaults).

    The defaults build a small multi-chart fixture (one tube fold arc + one
    far-field region per parity, coarse grids).  A production run raises the
    node counts, ``w`` density, and the per-chart engine budget, and tiles the
    gamma bands / fold arcs / regions.
    """

    n_gamma: int = 4
    n_u: int = 4
    n_theta: int = 4
    # Caustic-fixed far-field TILE counts: production proposes exterior
    # regions in ``(rho, theta_c)`` before bridging each accepted tile to the
    # ExteriorPolarChart's ``(rho, theta_c)`` spline axes.
    # ``n_theta_c`` is DISTINCT from the tube's along-caustic ``n_theta``.
    n_rho: int = 4
    n_theta_c: int = 4
    w_nodes_per_decade: int = 4
    # Interior (inside-caustic) tiles need higher w-axis density because the
    # SACR-C envelope carries more oscillation cycles per decade than the
    # exterior far-field remainder (Build interior-SACRC brief).
    interior_w_nodes_per_decade: int = 15
    f_floor: float = _DEFAULT_F_FLOOR
    f_max: float = _DEFAULT_F_MAX
    farfield_overlap: float = _DEFAULT_FARFIELD_OVERLAP
    gamma_band_halfwidth: float = 0.1
    min_gamma_band: float = 1e-6
    engine_budget: int = 400
    max_tube_arcs: int = 1
    # ``None`` = no cap (the production default: the tiling itself bounds the
    # count); an int caps admitted tiles with a loud truncation record.
    max_farfield_regions: int | None = None
    # Tile-grid side for the mass-stratified far-field tiling in caustic-fixed
    # coordinates (Build 8h-b3): each stratum's exterior region uses the
    # additive physical radial offset arm of ``rho`` and
    # ``theta_c in (-pi, pi]``.  It is split into
    # ``n_farfield_tiles_per_side^2`` rectangular tiles (the inner rho edge
    # already lies outside the caustic + tube shell, so every admitted tile is
    # exterior by construction; theta_c tile edges land on
    # +-pi so no tile straddles the branch cut).  ``max_farfield_regions``
    # then caps the total admitted tiles.
    n_farfield_tiles_per_side: int = 5
    n_heldout: int = 10
    # Held-out envelope-eps bars for chart registration: a chart above its
    # bar (or with NaN eps -- zero held-out points served) is recorded as
    # gated in the report and NOT packed into the artifact, so its window
    # falls through to the serving ladder.  The two bars use DIFFERENT error
    # currencies: the tube bar is max-normalized on ``max|E|`` (unchanged),
    # while the far-field bar is F-NORMALIZED on ``max|F| = max|exact_total|``
    # (Build 8g-b), because the redefined far-field label
    # ``E_ff = F - sum_a H_a e^{1j w tau_a}`` has ``max|E_ff| ~ 1e-4`` -- too
    # tiny and unstable a denominator to normalize against.  Defaults:
    # tube median is 3.8e-2, so 5e-2 separates the 0.43/1.15/2.15 saddle tail
    # (and the five >=0.09 charts) from healthy ~1e-2 siblings; the far-field
    # bar is 1e-3, the Professor 8g-b campaign-start value against the new
    # F-normalized currency (production re-gate to ~1e-4 with a caustic-edge
    # margin is a driver deferral, mirroring the 8a Q5 deferral).
    tube_eps_max: float = 5e-2
    farfield_eps_max: float = 1e-3
    # Held-out envelope-eps bar for the SACR-C interior charts (Build S2-3).
    # An interior chart stores the caustic-region ``tau_c``-demodulated
    # envelope, so its error is in the SAME max-|E| currency as the tube bar
    # (NOT the far-field F-normalized currency); the interior bar therefore
    # mirrors the tube default ``5e-2``.  Whether the crown reaches ``1e-3``
    # or plateaus at ``~1e-1`` is a measured post-build question (Professor
    # R4), so this bar is the campaign-start value, not a certified target.
    interior_eps_max: float = 5e-2
    # Finer interior gamma-bands near the parity boundary ``gamma = 1``
    # (Build S2-3 hygiene, NOT the fix -- the interior eps failure is a
    # fixed-gamma conditioning error, not gamma-interpolation).  A band whose
    # nearer edge lies within ``gamma_refine_near_one_window`` of ``gamma = 1``
    # is split into sub-bands no wider than ``gamma_refine_near_one_width`` so
    # the rapidly-varying near-merge geometry is sampled at more gamma nodes.
    # ``gamma_refine_near_one_window = 0`` disables the refinement.
    gamma_refine_near_one_window: float = 0.15
    gamma_refine_near_one_width: float = 0.05
    n_caustic_samples: int = 200
    seed: int = 0


# ---------------------------------------------------------------------------
# Caustic structure: cusp detection + fold arcs (from the geometry engine)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FoldArc:
    """One inter-cusp fold arc on a single caustic branch.

    Attributes
    ----------
    branch : int
        Square-root branch of ``critical_point`` (``+1`` always at positive
        parity; ``+-1`` for the two edges of a saddle deltoid lobe).
    theta_lo, theta_hi : float
        Lens-plane polar-angle bounds of the arc (cusp windows already
        excluded), radians.
    inward_sign : int
        Sign of the caustic normal pointing to the image-pair-present side.
    image_count : int
        Real-image count on the served (image-pair) side.
    cusp_windows : tuple of (float, float)
        ``(theta_cusp, delta_theta)`` exclusion windows bounding the arc.
        Interior cusps always contribute one; on the saddle deltoid the
        wedge-edge turnarounds also contribute a `_SADDLE_CUSP_MIN_HALFWIDTH`
        guard window (Build 8g WP3), whereas the astroid arcs have no
        turnaround walls (their branch is periodic).
    """

    branch: int
    theta_lo: float
    theta_hi: float
    inward_sign: int
    image_count: int
    cusp_windows: tuple


@dataclass(frozen=True)
class CausticStructure:
    """Detected caustic structure for one parity at one gamma.

    Attributes
    ----------
    parity : int
        ``+1`` astroid / ``-1`` saddle.
    gamma : float
        The shear the structure was measured at.
    cusp_thetas : tuple of float
        Detected cusp lens-plane angles.
    detected_cusps, expected_cusps : int
        Measured and topology-expected cusp counts (equal, else the caller
        raised `CausticTopologyError`).
    caustic_reach : float
        Maximum source-plane radius of the caustic (sizes the far-field box).
    arcs : tuple of FoldArc
        The fold arcs available for tube charts.
    """

    parity: int
    gamma: float
    cusp_thetas: tuple
    detected_cusps: int
    expected_cusps: int
    caustic_reach: float
    arcs: tuple


def _tube_normal(gamma: float, theta: float, branch: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Caustic point and unit source-plane normal at ``(gamma, theta,
    branch)``.

    The normal is the unit perpendicular to the EXACT analytic caustic
    tangent ``y' / |y'|`` (no finite difference), where ``y'`` is the
    closed-form theta-derivative from
    :func:`~cogwheel.lensing.chang_refsdal.geometry.caustic_derivatives`.
    Both ``y'`` and a forward difference point along increasing ``theta``,
    so the ``(-t_y, t_x)`` rotation preserves the previous orientation.  A
    `LensDomainError` from ``caustic_derivatives`` at the wedge edge (where
    ``critical_point`` still succeeds) is left to propagate to the caller.
    """
    caust = np.asarray(
        geometry.critical_point(gamma, theta, 0.0, 0.0, branch).source,
        dtype=float)
    y_prime, _ = geometry.caustic_derivatives(gamma, theta, branch=branch)
    tangent = y_prime / np.hypot(y_prime[0], y_prime[1])
    normal = np.array([-tangent[1], tangent[0]])
    return caust, normal


def _tube_source(gamma: float, theta: float, eta: float, branch: int,
                 sign: int) -> np.ndarray:
    """Source at caustic distance ``eta`` off the ``branch`` fold at
    ``theta``."""
    caust, normal = _tube_normal(gamma, theta, branch)
    return caust + sign * eta * normal


def _branch_speed_profile(gamma: float, branch: int, theta_lo: float,
                          theta_hi: float, n: int, periodic: bool
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Caustic ``theta`` samples and speed ``|d caustic / d theta|`` on a
    branch.

    Exact closed-form parametric speed
    (:func:`geometry.caustic_speed`).  Points outside the branch's domain
    (the saddle wedge) are dropped: a whole-array ``caustic_derivatives``
    call refuses if ANY theta is off-wedge, so each theta is evaluated
    individually and off-domain angles are skipped.
    """
    thetas = (np.linspace(theta_lo, theta_hi, n, endpoint=False) if periodic
              else np.linspace(theta_lo, theta_hi, n))
    good_theta, speeds = [], []
    for theta in thetas:
        try:
            speed = float(
                geometry.caustic_speed(gamma, float(theta), branch=branch))
        except geometry.LensDomainError:
            continue
        speeds.append(speed)
        good_theta.append(float(theta))
    good_theta = np.asarray(good_theta)
    if good_theta.shape[0] < 4:
        return good_theta, np.array([])
    return good_theta, np.asarray(speeds)


def _speed_slope(gamma: float, branch: int, theta: float) -> float:
    """Slope of the squared caustic speed: ``g(theta) = y'(theta) . y''(theta)``.

    Equals ``(1/2) d|y'|**2 / dtheta``.  It is real-analytic in ``theta``
    through a cusp (the caustic's non-smoothness lives in arc length, not in
    the angular parameter -- Professor) and crosses zero upward (``g' > 0``)
    at each speed minimum, so its root pins the cusp angle.  Uses the exact
    analytic derivatives (`geometry.caustic_derivatives`); no finite difference.
    """
    y_prime, y_double_prime = geometry.caustic_derivatives(
        gamma, theta, branch=branch)
    return float(y_prime[0] * y_double_prime[0]
                 + y_prime[1] * y_double_prime[1])


def _radial_slope(gamma: float, branch: int, theta: float) -> float:
    """Slope of the squared caustic radius: ``h(theta) = y(theta) . y'(theta)``.

    Equals ``(1/2) d|y|**2 / dtheta``.  Its UPWARD zero crossings mark the
    smooth local minima of the source-plane distance ``|y|`` from the origin.
    A cusp is a distinct kind of ``|y|`` minimum -- there ``y' -> 0`` so ``h``
    need not vanish and a root solver degenerates; cusp angles are handled by
    their own closed-form set, not this slope.  Uses the exact caustic point
    and first derivative (`geometry.critical_point` / `geometry.caustic_derivatives`);
    no finite difference.
    """
    y = np.asarray(
        geometry.critical_point(gamma, theta, 0.0, 0.0, branch).source,
        dtype=float)
    y_prime, _ = geometry.caustic_derivatives(gamma, theta, branch=branch)
    return float(y[0] * y_prime[0] + y[1] * y_prime[1])


def _refine_cusp_angle(gamma: float, branch: int,
                       theta_lo: float, theta_hi: float) -> float:
    """Analytic cusp angle: the root of ``y'.y'' = 0`` in ``[theta_lo, theta_hi]``.

    A caustic cusp is the point where the parametric speed ``|y'(theta)|``
    reaches zero, i.e. where ``g = _speed_slope`` crosses zero.  A single
    ``brentq`` bracketed strictly inside the sampled interval pins the angle
    to ~1e-10.

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    branch : int
        Square-root branch ``+-1`` of the caustic parametrisation.
    theta_lo, theta_hi : float
        Bracket endpoints (radians); ``g`` must change sign across them.

    Returns
    -------
    float
        The angle ``theta`` where ``y'.y'' = 0``.
    """
    eps = np.finfo(float).eps
    return brentq(lambda theta: _speed_slope(gamma, branch, theta),
                  theta_lo, theta_hi, xtol=4.0 * eps)


def _find_cusps(thetas: np.ndarray, speed: np.ndarray, periodic: bool, *,
                gamma: float, branch: int,
                width_safety: float = _CUSP_WIDTH_SAFETY,
                min_halfwidth: float = _CUSP_MIN_HALFWIDTH
                ) -> list[tuple[float, float]]:
    """Cusp ``(theta, delta_theta)`` pairs from caustic-speed minima.

    A cusp is a local minimum of ``speed``; its ANGLE is relocated to the
    exact analytic root of ``y'.y'' = 0`` (`_refine_cusp_angle`, ``brentq``
    bracketed strictly inside the sampled interval), and ``delta_theta`` is
    ``width_safety`` times the half-width of the dip that falls below
    ``window_dip_frac`` of the median speed, floored at ``min_halfwidth``.
    The astroid path uses the module defaults (`_CUSP_WIDTH_SAFETY`,
    `_CUSP_MIN_HALFWIDTH`); the saddle path passes its wider
    `_SADDLE_CUSP_WIDTH_SAFETY` / `_SADDLE_CUSP_MIN_HALFWIDTH` (Build 8g WP3).

    The analytic root is accepted only under the Professor TWIN GATE: ``g``
    must cross zero upward across the bracket (``g(lo) < 0 < g(hi)``, so the
    root is a speed *minimum* not a maximum) AND the caustic speed at the root
    must be below ``1e-6`` of the peak speed.  If the gate fails (or the
    bracket is degenerate / off-domain) the sampled minimum ``thetas[i]`` is
    kept -- the detector never invents a cusp.
    """
    if speed.size < 4:
        return []
    # RELATIVE dip fraction: sizes the carved-out exclusion WINDOW around a
    # detected cusp only.  It plays no part in locating the cusp angle (that
    # is the analytic root of y'.y'' = 0 below).
    window_dip_frac = 0.2
    threshold = window_dip_frac * float(np.median(speed))
    n = speed.size
    speed_peak = float(speed.max())
    step = float(np.median(np.diff(thetas)))
    theta_min = float(thetas.min())
    theta_max = float(thetas.max())
    cusps: list[tuple[float, float]] = []
    for i in range(n):
        left = speed[(i - 1) % n] if periodic else speed[max(i - 1, 0)]
        right = speed[(i + 1) % n] if periodic else speed[min(i + 1, n - 1)]
        is_edge_min = (not periodic) and (i == 0 or i == n - 1)
        if is_edge_min or not (speed[i] <= left and speed[i] < right):
            continue
        if speed[i] >= threshold:
            continue
        lo = i
        while speed[(lo - 1) % n if periodic else max(lo - 1, 0)] < threshold \
                and (periodic or lo > 0):
            lo = (lo - 1) % n if periodic else lo - 1
            if lo == i:
                break
        hi = i
        while speed[(hi + 1) % n if periodic else min(hi + 1, n - 1)] < \
                threshold and (periodic or hi < n - 1):
            hi = (hi + 1) % n if periodic else hi + 1
            if hi == i:
                break
        span = abs(thetas[i] - thetas[lo]) + abs(thetas[hi] - thetas[i])
        delta = max(min_halfwidth, width_safety * 0.5 * span)
        # Relocate to the analytic root of y'.y'' = 0, bracketed strictly
        # inside the sampled interval (keeps brentq off the diverging saddle
        # wedge edge).  Twin gate: upward zero crossing (speed minimum) AND
        # near-zero speed at the root; else keep the sampled minimum.
        theta_cusp = float(thetas[i])
        bracket_lo = max(theta_cusp - step, theta_min + _CUSP_BRACKET_EPS)
        bracket_hi = min(theta_cusp + step, theta_max - _CUSP_BRACKET_EPS)
        if bracket_lo < bracket_hi:
            try:
                g_lo = _speed_slope(gamma, branch, bracket_lo)
                g_hi = _speed_slope(gamma, branch, bracket_hi)
                if g_lo < 0.0 < g_hi:
                    root = _refine_cusp_angle(
                        gamma, branch, bracket_lo, bracket_hi)
                    root_speed = float(geometry.caustic_speed(
                        gamma, root, branch=branch))
                    if root_speed < 1e-6 * speed_peak:
                        theta_cusp = root
            except geometry.LensDomainError:
                theta_cusp = float(thetas[i])
        cusps.append((theta_cusp, float(delta)))
    return cusps


def _astroid_arcs(gamma: float, n: int
                  ) -> tuple[list[tuple[float, float]], list[FoldArc], float]:
    """Cusps and fold arcs of the positive-parity astroid (single branch)."""
    thetas, speed = _branch_speed_profile(
        gamma, 1, 0.0, 2.0 * np.pi, n, periodic=True)
    cusps = _find_cusps(thetas, speed, periodic=True, gamma=gamma, branch=1)
    cusps.sort()
    reach = _caustic_reach(gamma, 1, 0.0, 2.0 * np.pi, n)
    arcs: list[FoldArc] = []
    n_c = len(cusps)
    for i in range(n_c):
        (tc_lo, w_lo) = cusps[i]
        (tc_hi, w_hi) = cusps[(i + 1) % n_c]
        hi = tc_hi + (2.0 * np.pi if i == n_c - 1 else 0.0)
        arc = _make_arc(gamma, 1, tc_lo, w_lo, hi, w_hi,
                        [(tc_lo, w_lo), (tc_hi, w_hi)])
        if arc is not None:
            arcs.append(arc)
    return cusps, arcs, reach


def _saddle_arcs(gamma: float, n: int
                 ) -> tuple[list[tuple[float, float]], list[FoldArc], float]:
    """Cusps and fold arcs of the macro-saddle deltoid (two lobes, two
    branches).

    Each lobe (centred at lens-plane ``theta = 0`` and ``pi``) is bounded by
    the critical wedge ``|sin 2 theta| <= (1 - kappa) / |gamma|``; its two
    square-root branches each carry interior cusps, and the wedge edges are
    smooth turnarounds (walls, but not cusps).
    """
    theta_max = 0.5 * np.arcsin(1.0 / abs(gamma))
    all_cusps: list[tuple[float, float]] = []
    arcs: list[FoldArc] = []
    reach = 0.0
    for center in (0.0, np.pi):
        lo_edge = center - theta_max
        hi_edge = center + theta_max
        for branch in (1, -1):
            thetas, speed = _branch_speed_profile(
                gamma, branch, lo_edge, hi_edge, n, periodic=False)
            reach = max(reach, _caustic_reach(
                gamma, branch, lo_edge, hi_edge, n))
            cusps = _find_cusps(
                thetas, speed, periodic=False, gamma=gamma, branch=branch,
                width_safety=_SADDLE_CUSP_WIDTH_SAFETY,
                min_halfwidth=_SADDLE_CUSP_MIN_HALFWIDTH)
            cusps.sort()
            all_cusps.extend(cusps)
            # Guard the wedge-edge turnarounds (Build 8g WP3).  These walls
            # are smooth (not cusps) but the foot-of-normal map is near-
            # singular there and today they emit no exclusion, leaving the
            # arc's high-curvature end unguarded -- the measured root cause of
            # the saddle tube-tail (saddle_b*_tube_2/5 at eps 0.4..2.2).
            # Attaching a `_SADDLE_CUSP_MIN_HALFWIDTH` window makes
            # `_tube_serves` fall through near the turnaround.
            edge_hw = _SADDLE_CUSP_MIN_HALFWIDTH
            walls = [(lo_edge, edge_hw)] + cusps + [(hi_edge, edge_hw)]
            for (t_lo, w_lo), (t_hi, w_hi) in zip(walls[:-1], walls[1:]):
                windows = [(t, w) for (t, w) in ((t_lo, w_lo), (t_hi, w_hi))
                           if w > 0.0]
                arc = _make_arc(gamma, branch, t_lo, w_lo, t_hi, w_hi, windows)
                if arc is not None:
                    arcs.append(arc)
    return all_cusps, arcs, reach


def _make_arc(gamma: float, branch: int, t_lo: float, w_lo: float,
              t_hi: float, w_hi: float,
              windows: Sequence[tuple[float, float]]) -> FoldArc | None:
    """Assemble a `FoldArc` if its interior carries a faithful image pair."""
    inner_lo = t_lo + w_lo
    inner_hi = t_hi - w_hi
    margin = _ARC_MARGIN_FRAC * (inner_hi - inner_lo)
    inner_lo += margin
    inner_hi -= margin
    if inner_hi - inner_lo < _CUSP_MIN_HALFWIDTH:
        return None
    # Orient the arc from geometry, not a census probe: the image-pair side is
    # the sign of the exact fold-opening direction (points toward the two-image
    # side) projected onto the SAME serve normal `_tube_source` applies at
    # `_tube_normal` -- so an admitted source is nudged onto the served side by
    # construction (serve-consistency, Professor Q3).  Only the SIGN of `dot`
    # matters: it fixes the served two-image side and carries ~12 orders of
    # float64 margin (the minimum |dot| over the whole prior is 4.4e-3).  The
    # magnitude |dot| measures fold-opening transversality, which scales as
    # ~1.5*gamma; it is NOT a cusp-proximity proxy, so it must NOT be
    # magnitude-filtered -- doing so was the F041 regression, the same category
    # error as the retired _PROBE_ETA.  The exact-zero tripwire below only skips
    # the measure-zero pathology where the fold-opening direction is exactly
    # tangent to the serve normal (sign undefined).  The fallback fractions
    # exist solely to step past LensDomainError skips; the two-image side is a
    # global property of the fold arc, so the sign is invariant across them.
    span = inner_hi - inner_lo
    sign: int | None = None
    for frac in (0.5, 0.35, 0.65, 0.2, 0.8):
        theta = inner_lo + frac * span
        try:
            fold_dir = geometry.fold_opening_direction(
                gamma, theta, branch=branch)
            _caust, normal = _tube_normal(gamma, theta, branch)
        except geometry.LensDomainError:
            continue
        dot = float(fold_dir @ normal)
        if dot == 0.0:
            continue
        sign = 1 if dot >= 0.0 else -1
        break
    if sign is None:
        return None
    # The served/image-pair side carries exactly four real images for BOTH the
    # positive-parity astroid interior and the macro-saddle deltoid lobe at
    # kappa = 0 -- a constant of parity, so no `find_images` probe is needed
    # (Professor Q2).
    return FoldArc(branch=branch, theta_lo=float(inner_lo),
                   theta_hi=float(inner_hi), inward_sign=int(sign),
                   image_count=4,
                   cusp_windows=tuple((float(t), float(w))
                                      for t, w in windows))


def _caustic_reach(gamma: float, branch: int, theta_lo: float,
                   theta_hi: float, n: int) -> float:
    """Maximum source-plane radius of the caustic over a branch sweep."""
    reach = 0.0
    for theta in np.linspace(theta_lo, theta_hi, n):
        try:
            src = geometry.critical_point(
                gamma, theta, 0.0, 0.0, branch).source
        except geometry.LensDomainError:
            continue
        reach = max(reach, float(np.hypot(src[0], src[1])))
    return reach


def detect_caustic_structure(gamma: float, parity: int, *,
                             n_samples: int = 200) -> CausticStructure:
    """Detect cusps and fold arcs for one parity, cross-checking the topology.

    Parameters
    ----------
    gamma : float
        External shear; must be on the requested parity's side of ``1``.
    parity : int
        ``+1`` astroid (``gamma < 1``) or ``-1`` saddle (``gamma > 1``).
    n_samples : int, optional
        Caustic samples per branch sweep (default 200).

    Returns
    -------
    CausticStructure
        Detected cusps, fold arcs, and caustic reach.

    Raises
    ------
    CausticTopologyError
        If the detected cusp count does not match the expected topology.
    """
    if parity == 1:
        cusps, arcs, reach = _astroid_arcs(gamma, n_samples)
    else:
        cusps, arcs, reach = _saddle_arcs(gamma, n_samples)
    expected = _EXPECTED_CUSPS[parity]
    detected = len(cusps)
    if detected != expected:
        raise CausticTopologyError(
            f'Detected {detected} caustic cusps at gamma={gamma} '
            f'(parity {parity:+d}) but the topology expects {expected} '
            f'({"astroid" if parity == 1 else "deltoid, 2 lobes x 3"}). '
            f'The caustic sampling or geometry is inconsistent.')
    return CausticStructure(
        parity=parity, gamma=float(gamma),
        cusp_thetas=tuple(float(t) for t, _ in cusps),
        detected_cusps=detected, expected_cusps=expected,
        caustic_reach=float(reach), arcs=tuple(arcs))


def _merge_windows(windows: list[tuple[float, float]]
                   ) -> tuple[tuple[float, float], ...]:
    """Merge ``(theta, halfwidth)`` windows into disjoint covering windows."""
    if not windows:
        return ()
    spans = sorted((t - w, t + w) for t, w in windows)
    merged = [list(spans[0])]
    for lo, hi in spans[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return tuple((0.5 * (lo + hi), 0.5 * (hi - lo)) for lo, hi in merged)


def band_caustic_structure(band: tuple[float, float], parity: int, *,
                           n_samples: int = 200) -> CausticStructure:
    """Caustic structure whose arcs are valid across a whole gamma band.

    A tube chart samples a rectangular ``(gamma, theta)`` grid, but arc
    bounds are gamma-dependent -- the saddle critical wedge
    ``|sin 2 theta| <= 1 / |gamma|`` NARROWS as ``gamma`` grows, and cusps
    migrate with ``gamma`` on both parities.  Structure detected at a single
    anchor gamma is therefore not a valid grid domain for the band.  This
    detects at the band's two edges and center, matches arcs by position in
    the deterministic detection order, and returns arcs with

    - ``theta`` bounds = the INTERSECTION of the three matched intervals,
    - ``cusp_windows`` = the conservative union (merged covering windows),
    - ``caustic_reach`` = the max over the band (conservative for w caps).

    Any structural disagreement across the band (arc count, branch order,
    served side, or image count) raises `CausticTopologyError` -- a band
    that changes topology must be split by the caller, never papered over.
    """
    gammas = (band[0], 0.5 * (band[0] + band[1]), band[1])
    structs = [detect_caustic_structure(g, parity, n_samples=n_samples)
               for g in gammas]
    center = structs[1]
    counts = {len(s.arcs) for s in structs}
    if len(counts) != 1:
        raise CausticTopologyError(
            f'Fold-arc count changes across gamma band {band} '
            f'(parity {parity:+d}): {[len(s.arcs) for s in structs]}. '
            f'Split the band.')
    arcs = []
    for triplet in zip(*(s.arcs for s in structs)):
        if len({a.branch for a in triplet}) != 1:
            raise CausticTopologyError(
                f'Arc branch order changes across gamma band {band} '
                f'(parity {parity:+d}). Split the band.')
        sides = {(a.inward_sign, a.image_count) for a in triplet}
        if len(sides) != 1:
            raise CausticTopologyError(
                f'Arc served side / image count changes across gamma band '
                f'{band} (parity {parity:+d}): {sorted(sides)}. '
                f'Split the band.')
        theta_lo = max(a.theta_lo for a in triplet)
        theta_hi = min(a.theta_hi for a in triplet)
        if theta_hi - theta_lo < _CUSP_MIN_HALFWIDTH:
            continue
        windows = _merge_windows(
            [w for a in triplet for w in a.cusp_windows])
        arcs.append(FoldArc(
            branch=triplet[0].branch, theta_lo=float(theta_lo),
            theta_hi=float(theta_hi), inward_sign=triplet[0].inward_sign,
            image_count=triplet[0].image_count, cusp_windows=windows))
    return CausticStructure(
        parity=parity, gamma=center.gamma,
        cusp_thetas=center.cusp_thetas,
        detected_cusps=center.detected_cusps,
        expected_cusps=center.expected_cusps,
        caustic_reach=float(max(s.caustic_reach for s in structs)),
        arcs=tuple(arcs))


def stable_gamma_bands(band: tuple[float, float], parity: int, *,
                       n_samples: int = 200, min_width: float = 1e-6,
                       refine_near_one_window: float = 0.0,
                       refine_near_one_width: float = 0.05
                       ) -> tuple[list[tuple[tuple[float, float],
                                             CausticStructure]],
                                  list[tuple[float, float]]]:
    """Bisect a gamma band into topology-stable sub-bands.

    The saddle deltoid's fold-arc partition changes at discrete gamma
    values (cusps migrate through wedge walls), so a single rectangular
    tube grid cannot span such a metamorphosis.  Bands failing the
    `band_caustic_structure` consistency guard are bisected; slivers
    narrower than ``min_width`` that still straddle a change are DROPPED
    (refusal-conservative: those gammas receive no chart and fall through
    to the exact engine, mirroring the ``gamma = 1`` guard band) and returned in the
    second list so the caller can record them.

    When ``refine_near_one_window > 0``, every topology-stable band whose
    nearer edge lies within that window of the parity boundary ``gamma = 1``
    is additionally split into uniform sub-bands no wider than
    ``refine_near_one_width`` (Build S2-3 interior hygiene): near ``gamma =
    1`` the near-merged image geometry the SACR-C interior charts sample
    varies rapidly in ``gamma``, so more gamma nodes are warranted.  This is
    a NODE-DENSITY refinement, not a topology split -- each sub-band inherits
    the parent's (already consistent) `CausticStructure` recomputed at the
    sub-band, so no sliver is ever dropped by refinement.

    Returns
    -------
    (stable, dropped)
        ``stable`` -- ordered ``((lo, hi), CausticStructure)`` pairs;
        ``dropped`` -- ordered excluded ``(lo, hi)`` slivers.
    """
    stable: list[tuple[tuple[float, float], CausticStructure]] = []
    dropped: list[tuple[float, float]] = []
    stack = [(float(band[0]), float(band[1]))]
    while stack:
        sub = stack.pop()
        try:
            structure = band_caustic_structure(
                sub, parity, n_samples=n_samples)
        except CausticTopologyError:
            if sub[1] - sub[0] < min_width:
                dropped.append(sub)
                continue
            mid = 0.5 * (sub[0] + sub[1])
            stack.extend([(sub[0], mid), (mid, sub[1])])
            continue
        stable.extend(_refine_band_near_one(
            sub, structure, parity, n_samples,
            refine_near_one_window, refine_near_one_width))
    return sorted(stable), sorted(dropped)


def _refine_band_near_one(band: tuple[float, float],
                          structure: CausticStructure, parity: int,
                          n_samples: int, window: float, width: float
                          ) -> list[tuple[tuple[float, float],
                                          CausticStructure]]:
    """Split a near-``gamma = 1`` topology-stable band into finer sub-bands.

    Returns ``[(band, structure)]`` unchanged when refinement is disabled
    (``window <= 0``), the band is already narrow enough, or its nearer edge
    is farther than ``window`` from ``gamma = 1``.  Otherwise returns uniform
    sub-bands no wider than ``width``, each with its own `CausticStructure`
    recomputed at the sub-band; a sub-band that unexpectedly fails the
    topology guard falls back to the parent structure, so refinement never
    drops coverage.
    """
    lo, hi = band
    edge_distance = min(abs(lo - 1.0), abs(hi - 1.0))
    if window <= 0.0 or edge_distance > window or (hi - lo) <= width:
        return [(band, structure)]
    n_sub = int(math.ceil((hi - lo) / width))
    edges = np.linspace(lo, hi, n_sub + 1)
    refined: list[tuple[tuple[float, float], CausticStructure]] = []
    for i in range(n_sub):
        sub = (float(edges[i]), float(edges[i + 1]))
        try:
            sub_structure = band_caustic_structure(
                sub, parity, n_samples=n_samples)
        except CausticTopologyError:
            sub_structure = structure
        refined.append((sub, sub_structure))
    return refined


def _min_curvature_radius(band: tuple[float, float], arc: FoldArc,
                          n_samples: int) -> float:
    """Minimum caustic curvature radius over an arc, worst gamma in band.

    Exact closed-form caustic curvature radius
    (:func:`geometry.caustic_curvature_radius`), sampled across the arc's
    ``theta`` span with endpoints included, at the band's edge gammas
    (curvature is worst where the caustic is smallest). Conservative floor
    for the foot-of-normal assertion. A genuinely straight caustic point
    returns ``inf`` -- the physical infinite radius, i.e. no curvature
    constraint -- so no collinearity guard is needed.
    """
    thetas = np.linspace(arc.theta_lo, arc.theta_hi, max(n_samples // 2, 32))
    r_min = np.inf
    for gamma in (band[0], band[1]):
        radii = geometry.caustic_curvature_radius(
            float(gamma), thetas, branch=arc.branch)
        r_min = min(r_min, float(np.min(radii)))
    return float(r_min)


# ---------------------------------------------------------------------------
# Chart builders (engine sampling)
# ---------------------------------------------------------------------------

def _gamma_band(box: PriorBox, parity: int, halfwidth: float
                ) -> tuple[float, float]:
    """A narrow gamma band on the requested parity's side of ``gamma = 1``.

    Derived from the prior box (never hard-coded): the band is centred in the
    parity's gamma sub-range and half `halfwidth` wide, clipped to the box.
    """
    lo, hi = box.gamma_range
    guard = 1e-2
    if parity == 1:
        sub = (lo, min(1.0 - guard, hi))
    else:
        sub = (max(1.0 + guard, lo), hi)
    center = 0.5 * (sub[0] + sub[1])
    return (max(sub[0], center - halfwidth), min(sub[1], center + halfwidth))


def _upper_w_cap(w_max: float, parity: int, y_magnitude: float) -> float:
    """Lower an uncapped ``w`` top by the parity ceiling and the DD product
    cap.

    The double-double point-mass kernel refuses ``w * |y|`` above its ceiling,
    so the largest ``w`` a chart may sample is ``_DD_PRODUCT_MARGIN /
    y_magnitude`` where ``y_magnitude`` is the largest source MAGNITUDE
    (``|y|``, not a per-axis coordinate) the chart reaches.  One authoritative
    place for the ceiling + DD arithmetic shared by `_capped_w_range`
    (tube-shell radius) and `_stratum_w_range` (far-field square-box corner).

    Parameters
    ----------
    w_max : float
        The stratum's uncapped band top ``w(f_hi, m_hi)``.
    parity : int
        ``+1`` astroid / ``-1`` saddle (selects the ``w`` ceiling).
    y_magnitude : float
        Largest source magnitude ``|y|`` the chart samples (a radius, already
        including any geometric corner factor).
    """
    ceiling = _POSITIVE_W_CEILING if parity == 1 else _SADDLE_W_CEILING
    dd_cap = _DD_PRODUCT_MARGIN / max(y_magnitude, 1e-3)
    return min(w_max, ceiling, dd_cap)


def _capped_w_range(box: PriorBox, parity: int, y_max: float
                    ) -> tuple[float, float]:
    """Chart ``w`` band, capped so ``w_max * y_max`` stays below the DD
    ceiling.

    Starts from the prior's mass-derived ``w`` band (the full prior mass range)
    and lowers the upper edge to ``_DD_PRODUCT_MARGIN / y_max`` when the
    chart's largest source magnitude would otherwise push ``w * |y|`` past the
    point-mass kernel's ceiling.  This mirrors the prior, where the
    mass-conditioned source scale keeps ``w * |y| <= ~55`` by construction, so
    the (large-w, large-|y|) corner the engine refuses is never sampled.

    ``y_max`` here is a source MAGNITUDE (the tube-shell radius
    ``caustic_reach + eta_max``), so it feeds `_upper_w_cap` directly with no
    corner factor.
    """
    m_lo, m_hi = box.m_lens_range
    w_min = float(dimensionless_frequency(box.f_lo_hz, m_lo, 0.0))
    w_max = float(dimensionless_frequency(box.f_hi_hz, m_hi, 0.0))
    return (w_min, _upper_w_cap(w_max, parity, y_max))


def _stratum_w_range(box: PriorBox, parity: int, m_lo: float, m_hi: float,
                     y_half_width: float) -> tuple[float, float]:
    """Whole-band ``w`` range of one mass stratum, with the parity + DD caps.

    Returns ``[w(f_lo, m_lo), w(f_hi, m_hi)]`` -- the band that contains every
    in-stratum draw's ``[w(20, m), w(1024, m)]`` (whole-band containment is the
    serving contract) -- then lowers the upper edge via `_upper_w_cap`: the
    parity engine ceiling and the double-double product cap.  Unlike
    `_capped_w_range` it does NOT start from ``box.w_range`` (which spans the
    whole prior mass range); it brackets a single stratum.  Where the cap
    truncates ``w_max`` below ``w(f_hi, m_hi)`` the high-mass corner of the
    stratum is beyond the cap and the caller records it.

    ``y_half_width`` is the far-field square-box PER-AXIS half-width ``Y``, but
    `_farfield_tiles` admits tiles out to the box CORNER at ``|y| = Y *
    sqrt(2)`` (INS-1-001).  The DD cap must bound ``w * |y|`` at that corner --
    the exact node the tiling is built to cover -- so the corner magnitude
    ``y_half_width * sqrt(2)`` (not the per-axis half-width) is what feeds
    `_upper_w_cap`; the largest ``w`` a tile samples then keeps
    ``w * (Y * sqrt(2)) <= _DD_PRODUCT_MARGIN`` and the engine never refuses
    the outer corner.

    Parameters
    ----------
    box : PriorBox
        Supplies the detector frequency band bounds.
    parity : int
        ``+1`` astroid / ``-1`` saddle (selects the ``w`` ceiling).
    m_lo, m_hi : float
        Stratum mass edges (solar masses), ``m_lo <= m_hi``.
    y_half_width : float
        Per-axis half-width ``Y`` of the stratum's square y-support box; the DD
        cap uses the box-corner magnitude ``Y * sqrt(2)``.
    """
    w_min = float(dimensionless_frequency(box.f_lo_hz, m_lo, 0.0))
    w_max = float(dimensionless_frequency(box.f_hi_hz, m_hi, 0.0))
    y_corner = y_half_width * math.sqrt(2.0)
    return (w_min, _upper_w_cap(w_max, parity, y_corner))


def _mass_strata(box: PriorBox, parity: int
                 ) -> tuple[list[tuple[float, float]], dict | None]:
    """Partition the parity's REACHABLE lens-mass range into log strata.

    A stratum's mass ratio is fixed at ``R = sqrt(f_hi / f_lo)`` (~7.16 for the
    20-1024 Hz band), so each chart's log-``w`` range is ~1.5x a single draw's
    ``f_hi/f_lo`` band -- the spline-accuracy sweet spot (Professor 8g Q1).

    The reachable top is ``min(m_hi_prior, m_ceiling)`` where ``m_ceiling`` is
    the mass at which ``w(f_hi, m)`` reaches the parity's engine ceiling
    (astroid ``_POSITIVE_W_CEILING``, saddle ``_SADDLE_W_CEILING`` -- the
    Schwinger wall, so the saddle tops out near ~458 Msun today).  Mass above
    the reachable top cannot satisfy whole-band containment and is returned as
    a loud ``beyond_w_cap`` record, never silently dropped.

    Parameters
    ----------
    box : PriorBox
        Supplies the prior mass range and the detector frequency band.
    parity : int
        ``+1`` astroid / ``-1`` saddle.

    Returns
    -------
    tuple[list[tuple[float, float]], dict | None]
        ``(strata, beyond)`` -- ``strata`` is the list of ``(m_lo, m_hi)``
        stratum mass edges; ``beyond`` is ``None`` when the whole prior mass
        range is reachable, else a record of the un-tileable high-mass tail.
    """
    m_lo_prior, m_hi_prior = box.m_lens_range
    ceiling = _POSITIVE_W_CEILING if parity == 1 else _SADDLE_W_CEILING
    w_per_msun_at_fhi = float(dimensionless_frequency(box.f_hi_hz, 1.0, 0.0))
    m_ceiling = ceiling / w_per_msun_at_fhi
    m_reachable_hi = min(m_hi_prior, m_ceiling)

    beyond: dict | None = None
    if m_reachable_hi < m_hi_prior:
        beyond = {'m_lo': float(m_reachable_hi), 'm_hi': float(m_hi_prior),
                  'ceiling': float(ceiling)}

    if m_reachable_hi <= m_lo_prior:
        # No mass is reachable at this parity; the whole prior range is beyond
        # the w-cap.  (Not expected for the real parities, guarded for safety.)
        return [], {'m_lo': float(m_lo_prior), 'm_hi': float(m_hi_prior),
                    'ceiling': float(ceiling)}

    band_factor = box.f_hi_hz / box.f_lo_hz
    log_ratio = math.sqrt(band_factor)
    n_strata = max(1, math.ceil(
        math.log10(m_reachable_hi / m_lo_prior) / math.log10(log_ratio)))
    edges = np.logspace(math.log10(m_lo_prior), math.log10(m_reachable_hi),
                        n_strata + 1)
    strata = [(float(edges[k]), float(edges[k + 1])) for k in range(n_strata)]
    return strata, beyond


def _coordinate_radius_bounds(
        band: tuple[float, float], parity: int,
) -> tuple[float, float]:
    """Minimum directional caustic radius and maximum reach over a gamma band.

    The first value maps a physical support or exclusion disk to one
    conservative chart-rho bound. Positive parity takes the minimum actual
    critical-curve radius over polar angle and the band's edges/midpoint;
    macro-saddle charts use their scalar-reach fallback. The second value
    bounds the whole physical caustic independently of direction.
    """
    gamma_lo, gamma_hi = band
    gammas = (gamma_lo, 0.5 * (gamma_lo + gamma_hi), gamma_hi)
    scalar_reaches = [_scalar_caustic_reach(gamma) for gamma in gammas]
    if parity == 1:
        thetas = np.linspace(
            -math.pi, math.pi, _INTERIOR_BOUNDARY_NODES,
        )
        coordinate_radii = [
            geometry.r_caustic(gamma, float(theta))
            for gamma in gammas
            for theta in thetas
        ]
    else:
        coordinate_radii = scalar_reaches
    radius_min = float(min(coordinate_radii))
    reach_max = float(max(scalar_reaches))
    if not (math.isfinite(radius_min) and radius_min > 0.0):
        raise geometry.LensDomainError(
            f'Invalid caustic coordinate radius {radius_min} for band {band}.')
    return radius_min, reach_max


def _farfield_tiles(rho_inner: float, rho_outer: float, n_per_side: int
                    ) -> list[tuple[tuple[float, float],
                                    tuple[float, float], int, int]]:
    """Rectangular exterior tiles of a caustic-fixed ``(rho, theta_c)``
    region.

    Lays a uniform ``n_per_side x n_per_side`` grid over the exterior region
    ``rho in [rho_inner, rho_outer]`` x ``theta_c in [-pi, pi]`` and returns
    one tile per grid cell. The caller derives ``rho_inner`` from the physical
    ``reach_max + eta_max`` exclusion disk and the band's minimum coordinate
    radius. Therefore every emitted tile is in the single 2-image exterior
    region even though positive-parity rho uses a directional radius.

    The ``theta_c`` axis is tiled over ``[-pi, pi]`` so tile edges fall exactly
    on ``+-pi``; no tile spans the ``atan2`` branch cut at ``theta_c = +-pi``
    (the serve side derives ``theta_c = atan2(y2, y1) in (-pi, pi]``).  When
    the region is empty (``rho_outer <= rho_inner`` -- a high-mass stratum
    whose whole ``y``-support box lies inside the caustic disk) no tile is
    emitted; those near-caustic draws are served by the tube + serving ladder.

    Parameters
    ----------
    rho_inner : float
        Conservative inner exterior-admission radius in chart-rho units.
    rho_outer : float
        Outer prior-support radius in chart-rho units.
    n_per_side : int
        Number of tiles along each axis (``rho`` and ``theta_c``).

    Returns
    -------
    list[tuple[tuple[float, float], tuple[float, float], int, int]]
        ``((rho_center, theta_c_center), (half_rho, half_theta_c), i, j)`` for
        each tile, in row-major grid order (deterministic).  ``i`` indexes the
        ``rho`` axis, ``j`` the ``theta_c`` axis.  Empty when the region is
        empty.
    """
    if rho_outer <= rho_inner:
        return []
    half_rho = 0.5 * (rho_outer - rho_inner) / n_per_side
    half_theta = math.pi / n_per_side  # ((pi) - (-pi)) / n_per_side / 2
    rho_centers = [rho_inner + half_rho * (2 * k + 1)
                   for k in range(n_per_side)]
    theta_centers = [-math.pi + half_theta * (2 * k + 1)
                     for k in range(n_per_side)]
    tiles: list[tuple[tuple[float, float], tuple[float, float], int, int]] = []
    for i, rho_c in enumerate(rho_centers):
        for j, theta_c in enumerate(theta_centers):
            tiles.append(((float(rho_c), float(theta_c)),
                          (float(half_rho), float(half_theta)), i, j))
    return tiles


def _farfield_region_w_floor(box: PriorBox, band: tuple[float, float],
                             exclusion_rho: float, config: TrainingConfig
                             ) -> tuple[float, dict]:
    """Conservative far-field ``w_floor`` for the whole exterior region.

    The exterior far-field kernel-sum label (`FARFIELD_KERNEL_SUM`) is the
    bounded MID-BAND object, valid only at and above the S1-2 physics threshold
    ``w_floor = (RHO_END / 2) / min_{a != b real} |tau_a - tau_b|``
    (`channels.farfield_w_floor`): below it no real pair separates and the
    label is the divergent diffractive-bottom object instead.  A SINGLE chart
    trained over the whole ``(gamma, rho, theta_c)`` box must have its label
    valid at EVERY node, so the region ``w_floor`` is the MAXIMUM of the local
    physics floor over the region -- and the local floor is largest where the
    two exterior images are closest, i.e. at the INNER admission edge
    ``rho = exclusion_rho`` (just outside the caustic + tube shell).  The probe
    therefore evaluates the local floor at ``rho = exclusion_rho`` across the
    band's gamma edges/midpoint and the ``theta_c`` tile centres and returns
    the max finite value.

    When every probe refuses (`_ENGINE_REFUSALS`) or yields a non-finite floor
    (fewer than two real images resolved at the inner edge), the routine falls
    back to the prior band's lowest ``w`` edge ``w(f_lo, m_lo)`` -- the
    smallest frequency any in-region draw reaches -- so the window degenerates
    to the whole prior band (no diffractive exclusion) rather than guessing a
    floor.  The fallback is recorded loudly.

    Returns
    -------
    tuple[float, dict]
        ``(w_floor, report)`` where ``report`` records the probe count, the max
        local floor found, and whether the fallback was used.
    """
    gammas = (band[0], 0.5 * (band[0] + band[1]), band[1])
    n_theta = max(1, config.n_farfield_tiles_per_side)
    half_theta = math.pi / n_theta
    thetas = [-math.pi + half_theta * (2 * k + 1) for k in range(n_theta)]
    # The physics floor reads only ``partition.delays`` /
    # ``partition.real_mask`` (both w-independent), but the channels engine
    # requires a >=2-point w grid; a fixed 2-node dummy grid suffices (its
    # values never enter the floor).
    probe_channels = ChangRefsdalChannels(np.array([1.0, 2.0]))
    floors: list[float] = []
    for gamma in gammas:
        for theta in thetas:
            y1_eig, y2_eig = _from_caustic_fixed(gamma, exclusion_rho, theta)
            try:
                partition = probe_channels.evaluate(
                    gamma=gamma, y=(y1_eig, y2_eig), beta=0.0, kappa=0.0)
            except _ENGINE_REFUSALS:
                continue
            floor = farfield_w_floor(partition.delays, partition.real_mask)
            if math.isfinite(floor):
                floors.append(float(floor))
    n_probe = len(gammas) * len(thetas)
    if floors:
        return max(floors), {
            'w_floor_source': 'physics_threshold',
            'w_floor': round(max(floors), 6),
            'n_probe': n_probe, 'n_floor_finite': len(floors)}
    fallback = float(dimensionless_frequency(
        box.f_lo_hz, box.m_lens_range[0], 0.0))
    return fallback, {
        'w_floor_source': 'prior_band_low_edge_fallback',
        'w_floor': round(fallback, 6),
        'n_probe': n_probe, 'n_floor_finite': 0}


def _farfield_region_window(box: PriorBox, parity: int,
                            band: tuple[float, float], exclusion_rho: float,
                            rho_outer: float, coordinate_radius_max: float,
                            ppgo_boundary: float | None,
                            ppgo_ceiling: float | None,
                            config: TrainingConfig,
                            source_magnitude_max: float | None = None,
                            ) -> tuple[tuple[float, float] | None, str, dict]:
    """Fixed ``[w_floor, w_trust]`` w-window for the exterior far-field region.

    Replaces the per-mass-stratum ``w`` partitioning (`_stratum_w_range`) with
    a SINGLE fixed window for the whole exterior region (Build S1-3):

    - ``w_floor`` is the region's conservative S1-2 physics threshold
      (`_farfield_region_w_floor`);
    - the uncapped top is the prior's highest ``w`` edge ``w(f_hi, m_hi)``,
      lowered by the parity engine ceiling and the double-double product cap at
      the region's largest physical source magnitude (`_upper_w_cap`);
    - ``w_trust`` is that top trimmed against the certified-ppGO hand-off floor
      via the SAME `_apply_ppgo_trim` the strata path used (band-split serving
      is live above ``w_trust``), so a certified region whose whole band sits
      above the floor is ``'drop'`` (ppGO serves it, no chart) and one whose
      top exceeds the floor is ``'cap'`` (the tail hands to ppGO).

    Returns
    -------
    tuple[tuple[float, float] | None, str, dict]
        ``(window, action, report)``.  ``window`` is ``None`` with
        ``action in {'drop', 'empty'}`` when the region needs no exterior chart
        (whole band ppGO-served, or a degenerate ``w_floor >= w_trust``
        window); otherwise ``window = (w_floor, w_trust)`` with ``action in
        {'keep', 'cap'}``.
    """
    w_floor, floor_report = _farfield_region_w_floor(
        box, band, exclusion_rho, config)
    w_top_uncapped = float(dimensionless_frequency(
        box.f_hi_hz, box.m_lens_range[1], 0.0))
    y_magnitude = (
        float(rho_outer) * float(coordinate_radius_max)
        if source_magnitude_max is None else float(source_magnitude_max)
    )
    base_top = _upper_w_cap(w_top_uncapped, parity, y_magnitude)
    trimmed, action = _apply_ppgo_trim(
        (w_floor, base_top), ppgo_boundary, ppgo_ceiling)
    report = {
        'w_top_uncapped': round(w_top_uncapped, 6),
        'base_top': round(float(base_top), 6),
        'y_magnitude': round(y_magnitude, 6),
        'ppgo_action': action,
        'ppgo_boundary': (None if ppgo_boundary is None
                          else round(float(ppgo_boundary), 6)),
        'ppgo_ceiling': (None if ppgo_ceiling is None
                         else round(float(ppgo_ceiling), 6)),
        **floor_report}
    if action == 'drop':
        report['w_range'] = [round(w_floor, 6), round(float(base_top), 6)]
        return None, 'drop', report
    w_lo, w_hi = trimmed
    report['w_range'] = [round(float(w_lo), 6), round(float(w_hi), 6)]
    if not w_lo < w_hi:
        # A degenerate window (the conservative physics floor exceeds the
        # capped/trusted top): the exterior far-field label is valid nowhere in
        # the region's served band, so no chart is trained -- those draws fall
        # to the tube / interior / serving ladder.  Recorded loudly.
        report['empty_reason'] = 'w_floor_ge_w_trust'
        return None, 'empty', report
    return (float(w_lo), float(w_hi)), action, report


def _farfield_window_contains_draws(box: PriorBox, window: tuple[float, float],
                                    *, n_mass: int = 8, tol: float = 1e-12
                                    ) -> tuple[bool, dict]:
    """Range-check that in-region draws' chart w-segments lie within
    ``window``.

    Replaces the mass-strata whole-band containment BOOKKEEPING with a direct
    subset test (Build S1-3): for a geometric sweep of lens masses across the
    prior range, each draw's full detector band ``[w(f_lo, m), w(f_hi, m)]`` is
    intersected with the fixed window to form the CHART SEGMENT the surrogate
    is responsible for (band-split serving hands the sub-``w_floor``
    diffractive bottom and the super-``w_trust`` tail to other labels / bare
    ppGO).  The check asserts every non-empty chart segment is a subset of
    ``[w_floor, w_trust]`` to ``tol`` -- true by construction of the clip, so a
    violation signals a window/clip inconsistency, and a zero-overlap count
    signals a window that covers no draw (a coverage note, not a violation).

    Parameters
    ----------
    box : PriorBox
        Supplies the prior mass range and detector band.
    window : tuple[float, float]
        The fixed ``(w_floor, w_trust)`` region window.
    n_mass : int, optional
        Number of log-spaced masses swept across the prior range.
    tol : float, optional
        Subset tolerance (default ``1e-12``).

    Returns
    -------
    tuple[bool, dict]
        ``(contained, report)`` -- ``contained`` is the subset verdict;
        ``report`` records the max subset violation, the overlap count, and the
        probe count.
    """
    w_floor, w_trust = window
    m_lo, m_hi = box.m_lens_range
    masses = np.geomspace(m_lo, m_hi, max(2, n_mass))
    max_violation = 0.0
    n_overlap = 0
    for mass in masses:
        w_lo = float(dimensionless_frequency(box.f_lo_hz, float(mass), 0.0))
        w_hi = float(dimensionless_frequency(box.f_hi_hz, float(mass), 0.0))
        seg_lo = max(w_lo, w_floor)
        seg_hi = min(w_hi, w_trust)
        if seg_lo > seg_hi:
            continue  # this draw's band does not overlap the chart window
        n_overlap += 1
        max_violation = max(max_violation,
                            w_floor - seg_lo, seg_hi - w_trust)
    contained = bool(max_violation <= tol)
    return contained, {
        'containment_ok': contained,
        'max_subset_violation': float(max_violation),
        'n_overlap': int(n_overlap),
        'n_probe': int(len(masses)),
        'subset_tol': tol}


def _caustic_points(gamma: float, parity: int, n: int) -> np.ndarray:
    """All sampled caustic source-plane points for one parity, shape ``(k,
    2)``.

    Sweeps the astroid (single periodic branch, ``theta`` over ``2 pi``) or the
    saddle deltoid (two lobes centred at ``theta = 0, pi``, each with two
    square-root branches over the critical wedge).  Points outside a branch's
    domain (the saddle wedge) are dropped.  Mirrors the branch enumeration of
    `_astroid_arcs` / `_saddle_arcs` but returns only the caustic points.
    """
    points: list[np.ndarray] = []
    if parity == 1:
        segments = [(1, np.linspace(0.0, 2.0 * np.pi, n, endpoint=False))]
    else:
        theta_max = 0.5 * np.arcsin(1.0 / abs(gamma))
        segments = []
        for center in (0.0, np.pi):
            thetas = np.linspace(center - theta_max,
                                 center + theta_max, n)
            for branch in (1, -1):
                segments.append((branch, thetas))
    for branch, thetas in segments:
        for theta in thetas:
            try:
                src = geometry.critical_point(
                    gamma, theta, 0.0, 0.0, branch).source
            except geometry.LensDomainError:
                continue
            points.append(np.asarray(src, dtype=float))
    return np.asarray(points) if points else np.empty((0, 2))


def _winding_number(points: np.ndarray) -> float:
    """Signed winding number of an ORDERED closed curve about the origin.

    ``points`` must trace the curve in traversal order (shape ``(k, 2)``); the
    loop is closed back to the first point.  Sums the signed angular increments
    seen from the origin, each wrapped to ``(-pi, pi]``, and divides by
    ``2 pi``.  A curve enclosing the origin returns ``+-1``; one that does not
    returns ``0``.  Only meaningful for a genuinely ordered single loop: the
    positive-parity astroid sweep, and each disjoint saddle lobe via the
    ordered boundary that `_lobe_winding_loop` builds -- it is applied to the
    saddle lobes at the `_SaddleLobeAdmission` interior probe test.
    """
    angles = np.arctan2(points[:, 1], points[:, 0])
    increments = np.diff(np.concatenate([angles, angles[:1]]))
    increments = (increments + np.pi) % (2.0 * np.pi) - np.pi
    return float(increments.sum() / (2.0 * np.pi))


def _branch_inradius_candidates(gamma: float, branch: int, theta_lo: float,
                                theta_hi: float, n: int, periodic: bool
                                ) -> list[float]:
    """Closed-form ``|y|`` candidates for the caustic inradius on one branch.

    The closest source-plane approach to the origin on a caustic branch is
    either a cusp (astroid cusps ARE the points nearest the origin) or a smooth
    interior minimum of ``|y|(theta)``.  Both are located from exact geometry:

    * refined cusp angles (`_find_cusps` -> `geometry.critical_point`), where a
      naive ``brentq`` on ``h = y . y'`` degenerates because ``y' -> 0``;
    * smooth interior minima -- an UPWARD zero crossing of ``h`` between two
      adjacent in-domain samples whose caustic speed stays clear of a cusp dip,
      refined with ``brentq`` and evaluated in closed form.

    Returns the list of candidate ``|y|`` values (empty if the branch carries
    fewer than four in-domain samples).
    """
    thetas, speed = _branch_speed_profile(
        gamma, branch, theta_lo, theta_hi, n, periodic=periodic)
    if speed.size < 4:
        return []
    candidates: list[float] = []
    # (a) Refined cusp angles: evaluate |y| in closed form at each cusp.
    for theta, _delta in _find_cusps(thetas, speed, periodic=periodic,
                                     gamma=gamma, branch=branch):
        try:
            src = geometry.critical_point(
                gamma, theta, 0.0, 0.0, branch).source
        except geometry.LensDomainError:
            continue
        candidates.append(float(np.hypot(src[0], src[1])))
    # (b) Smooth interior minima: upward zero crossings of h = y . y' that are
    #     clear of a cusp dip (both endpoint speeds above 0.2 * median, the same
    #     dip fraction `_find_cusps` uses to size cusp windows).
    median_speed = float(np.median(speed))
    h_vals: list[float] = []
    for theta in thetas:
        try:
            h_vals.append(_radial_slope(gamma, branch, float(theta)))
        except geometry.LensDomainError:
            h_vals.append(math.nan)
    m = thetas.shape[0]
    n_brackets = m if periodic else m - 1
    for k in range(n_brackets):
        i, j = k, (k + 1) % m
        lo, hi = float(thetas[i]), float(thetas[j])
        if hi <= lo:
            # Periodic wrap bracket (theta_hi -> theta_lo) is not a monotone
            # interval; any |y| minimum at the seam is a cusp already in (a).
            continue
        if not (h_vals[i] < 0.0 <= h_vals[j]):
            continue
        if min(float(speed[i]), float(speed[j])) < 0.2 * median_speed:
            continue
        try:
            root = brentq(lambda t: _radial_slope(gamma, branch, t), lo, hi)
            src = geometry.critical_point(
                gamma, root, 0.0, 0.0, branch).source
        except (geometry.LensDomainError, ValueError):
            continue
        candidates.append(float(np.hypot(src[0], src[1])))
    return candidates


def _closed_form_inradius(gamma: float, parity: int, n: int) -> float:
    """Caustic inradius (closest approach to the origin) from exact geometry.

    The minimum closed-form ``|y|`` over the refined cusp angles and refined
    smooth interior minima of every caustic branch (`_branch_inradius_candidates`):
    the astroid is a single periodic branch over ``[0, 2 pi)``; the macro-saddle
    deltoid is two lobes, each with two square-root branches over its critical
    wedge.  The quadratic curvature at a minimum gives ~1e-9 relative accuracy,
    replacing the discrete-sample ``min |y|`` which biases high by the sample
    spacing.  Returns ``0.0`` only if no branch carries enough in-domain samples
    (the caller's cloud guard already rejects that degenerate case).
    """
    candidates: list[float] = []
    if parity == 1:
        candidates += _branch_inradius_candidates(
            gamma, 1, 0.0, 2.0 * np.pi, n, periodic=True)
    else:
        theta_max = 0.5 * np.arcsin(1.0 / abs(gamma))
        for center in (0.0, np.pi):
            lo_edge = center - theta_max
            hi_edge = center + theta_max
            for branch in (1, -1):
                candidates += _branch_inradius_candidates(
                    gamma, branch, lo_edge, hi_edge, n, periodic=False)
    return min(candidates) if candidates else 0.0


def _caustic_inradius(gamma: float, parity: int, n: int) -> tuple[float, bool]:
    """Minimum caustic radius and whether the caustic encloses the origin.

    Returns ``(inradius, encloses_origin)``.  ``inradius`` is the smallest
    source-plane radius any caustic point reaches -- the radius of the largest
    origin-centred disk that fits inside the caustic curve, which the interior
    far-field tiles must stay within.  It is the minimum CLOSED-FORM ``|y|``
    over the refined cusp angles and refined smooth interior minima of the
    caustic branches (`_closed_form_inradius`), accurate to ~1e-9 (a quadratic
    minimum), not a discrete-sample minimum biased high by the sample spacing.

    ``encloses_origin`` keys the interior admission off the caustic TOPOLOGY,
    never a bare image count (Professor 8h-a), and is still computed from the
    winding number of the ordered discrete cloud:

    * The positive-parity astroid is a single closed 4-cusped curve swept over
      one continuous ``theta`` branch; it winds once around the origin, so an
      origin-centred interior disk is a genuine 4-image region.  Enclosure is
      the winding number of that ordered sweep about the origin (robust across
      the whole band; an angular-gap heuristic misclassifies near ``gamma = 1``
      where the astroid and the saddle deltoids merge).
    * The macro-saddle caustic (``gamma > 1``) is TWO disjoint deltoid lobes
      sitting OFF the origin on the shear axis.  The origin is a 2-image saddle
      region enclosed by neither lobe at any ``gamma > 1`` (each lobe's winding
      number about the origin is 0), so there is no valid origin-centred
      interior disk -- the interior loop records a loud skip and admits nothing
      (the tube charts and the serving ladder cover the near-lobe regions).
      The lobes are not a single ordered loop, so their winding number is not
      computed from the point order (which is a segment-enumeration artifact);
      enclosure is False by the two-lobe topology.
    """
    points = _caustic_points(gamma, parity, n)
    if points.shape[0] < 4:
        return 0.0, False
    inradius = _closed_form_inradius(gamma, parity, n)
    if parity != 1:
        return inradius, False
    return inradius, abs(_winding_number(points)) >= 0.5


def _cusp_source_angles(gamma: float, n: int) -> list[float]:
    """Source-plane polar angles (rad, sorted, in ``[-pi, pi]``) of the astroid
    cusps.

    The four positive-parity astroid cusps are the caustic-speed minima of the
    branch sweep (`_find_cusps` -- the SAME detector the fold-arc tiler uses);
    each cusp's LENS-plane angle is mapped through `critical_point` to its
    SOURCE-plane image and reported as an ``atan2`` direction.  The directional
    caustic radius `geometry.r_caustic` has slope KINKS exactly along these
    rays, so the interior tiler aligns its ``theta_c`` tile edges to them and
    no tile straddles a kink (Professor caveat i, frozen WP6).

    Returns an empty list when the sweep resolves no cusp (a degenerate band);
    the caller then falls back to a uniform ``theta_c`` tiling and relies on
    the held-out eps gate as the safety net.
    """
    thetas, speed = _branch_speed_profile(
        gamma, 1, 0.0, 2.0 * np.pi, n, periodic=True)
    cusps = _find_cusps(thetas, speed, periodic=True, gamma=gamma, branch=1)
    angles: list[float] = []
    for theta_lens, _delta in cusps:
        try:
            src = geometry.critical_point(
                gamma, float(theta_lens), 0.0, 0.0, 1).source
        except geometry.LensDomainError:
            continue
        angles.append(float(np.arctan2(src[1], src[0])))
    return sorted(angles)


def _cusp_aligned_theta_tiles(cusp_angles: list[float], n_per_side: int
                              ) -> list[tuple[float, float]]:
    """``theta_c`` tiles aligned to the cusp rays and the ``+-pi`` branch cut.

    Partitions ``[-pi, pi]`` into sectors bounded by the cusp rays (and by
    ``+-pi``), then splits each sector into ``n_per_side`` uniform sub-tiles,
    so NO tile straddles a cusp-ray kink (Professor caveat i) or the ``atan2``
    branch cut at ``theta_c = +-pi`` (the serve side derives ``theta_c =
    atan2(y2, y1) in (-pi, pi]``).  Falls back to a single sector (uniform
    tiling) when no cusp ray is supplied.

    Returns ``[(theta_center, half_theta), ...]`` in ascending-angle order.
    """
    edges = {-math.pi, math.pi}
    for angle in cusp_angles:
        edges.add((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)
    sorted_edges = sorted(edges)
    tiles: list[tuple[float, float]] = []
    for edge_lo, edge_hi in zip(sorted_edges[:-1], sorted_edges[1:]):
        span = edge_hi - edge_lo
        if span <= 0.0:
            continue
        sub = span / n_per_side
        half_theta = 0.5 * sub
        for k in range(n_per_side):
            tiles.append((edge_lo + sub * k + half_theta, half_theta))
    return tiles

def _exclude_near_cusp(gamma: float, center: tuple[float, float],
                       half: tuple[float, float],
                       cusp_angles: list[float],
                       d_exclude: float = _CUSP_EXCLUSION_DISTANCE) -> bool:
    """Return True if any tile corner is within ``d_exclude`` of an astroid cusp.

    The four positive-parity astroid cusps are the caustic-speed minima
    whose source-plane directions ``cusp_angles`` come from
    `_cusp_source_angles`.  Each cusp's source-plane position is
    reconstructed as ``(r_caustic * cos(phi), r_caustic * sin(phi))``
    because the cusp lies ON the caustic and the directional caustic
    radius `geometry.r_caustic(gamma, phi)` is the outward magnitude.
    The tile's four corners are mapped to eigenframe source coordinates
    via `_from_caustic_fixed`.  The exclusion distance ``d_exclude`` is
    the minimum Euclidean source-plane separation below which the tile
    is considered too close to a near-cusp singularity for a smooth
    polar chart.

    Only astroid (positive-parity, ``gamma < 1``) cusps are checked;
    saddle (``gamma >= 1``) deltoid cusps are off-axis and not relevant
    for the caustic-centre-fixed exterior polar tiling.  A domain
    refusal from `geometry.r_caustic` (e.g. a cusp-angle drift at a
    band edge) is treated conservatively as excluded.
    """
    if not cusp_angles:
        return False
    rho_c, theta_c = center
    half_rho, half_theta = half
    rho_corners = (rho_c - half_rho, rho_c + half_rho)
    theta_corners = (theta_c - half_theta, theta_c + half_theta)
    corner_points = np.array([
        _from_caustic_fixed(gamma, cr, ct)
        for cr in rho_corners for ct in theta_corners
    ], dtype=float)
    cusp_positions: list[tuple[float, float]] = []
    for angle in cusp_angles:
        try:
            r = geometry.r_caustic(gamma, float(angle))
        except geometry.LensDomainError:
            continue
        cusp_positions.append((float(r * math.cos(angle)),
                               float(r * math.sin(angle))))
    if not cusp_positions:
        return False
    cusp_points = np.array(cusp_positions, dtype=float)
    delta = corner_points[:, None, :] - cusp_points[None, :, :]
    min_dist = float(np.sqrt((delta * delta).sum(axis=2)).min())
    return min_dist < d_exclude


@dataclass(frozen=True, eq=False)
class _InteriorAdmission:
    """Directional caustic-radius interior admission across a gamma band
    (S2-1).

    A candidate tile is admitted only when its outer rho edge is strictly below
    one and remains at least ``eta_max`` from the caustic at every sampled
    gamma in the band. Because positive-parity rho is normalized by the
    directional radius at the same ``(gamma, theta_c)``, ``rho = 1`` is the
    caustic for every direction and gamma. The physical nearest-distance check
    is still required: near a cusp the nearest caustic point is off the radial
    ray.

    Attributes
    ----------
    eta_max : float
        Tube-shell half-width excluded from the interior (dimensionless ``y``).
    theta_axis : np.ndarray
        Sorted polar-angle nodes in ``[-pi, pi]``.
    radius_grid : np.ndarray
        Shape ``(n_gamma, n_theta)`` directional physical caustic radii.
    caustic_clouds : tuple[np.ndarray, ...]
        Per-gamma ``(K, 2)`` eigenframe caustic point clouds (used by
        ``admits_exterior``).
    gammas : tuple
        Per-gamma shear magnitudes, aligned row-for-row with
        ``radius_grid`` and ``caustic_clouds``; the interior distance test
        queries the exact caustic at each of these gammas.
    """

    eta_max: float
    theta_axis: np.ndarray
    radius_grid: np.ndarray
    caustic_clouds: tuple[np.ndarray, ...]
    gammas: tuple

    def admits(self, center: tuple[float, float],
               half: tuple[float, float]) -> bool:
        """Whether the tile's outer edge is interior AND clear of the tube
        shell.

        ``center`` is ``(rho_center, theta_c_center)`` and ``half`` is
        ``(half_rho, half_theta_c)`` in caustic-fixed coordinates.  The outer
        ``rho`` edge is probed at `_INTERIOR_EDGE_SAMPLES` polar angles across
        the tile's ``theta_c`` span. Every physical probe is reconstructed with
        the same gamma- and angle-dependent radius as the chart, then its
        clearance is the EXACT nearest-caustic distance
        (:func:`geometry.nearest_caustic_point`) at that gamma -- no discrete
        cloud, no margin: a probe within ``eta_max`` of the caustic refuses the
        tile. A domain refusal from the geometry (e.g. the parity boundary) is
        treated conservatively as non-admission.
        """
        rho_center, theta_center = center
        half_rho, half_theta = half
        rho_outer = float(rho_center) + float(half_rho)
        if rho_outer <= 0.0 or rho_outer >= 1.0:
            return False
        thetas = np.linspace(theta_center - half_theta,
                             theta_center + half_theta, _INTERIOR_EDGE_SAMPLES)
        for gamma_i, radius_axis in zip(self.gammas, self.radius_grid):
            radii = np.interp(thetas, self.theta_axis, radius_axis)
            y_magnitudes = rho_outer * radii
            probe_x = y_magnitudes * np.cos(thetas)
            probe_y = y_magnitudes * np.sin(thetas)
            for px, py in zip(probe_x, probe_y):
                try:
                    nearest = geometry.nearest_caustic_point(
                        gamma_i, 0.0, np.array([px, py]), kappa=0.0).distance
                except geometry.LensDomainError:
                    return False
                if nearest < self.eta_max:
                    return False
        return True

    def admits_exterior(self, center: tuple[float, float],
                        half: tuple[float, float],
                        source_magnitude_max: float) -> bool:
        """Whether the exterior tile clears the caustic and tube shell.

        Positive-parity EXTERIOR companion to `admits` (WP1).  ``center`` is
        ``(rho_center, theta_c_center)`` and ``half`` is
        ``(half_rho, half_theta_c)`` in caustic-fixed coordinates.  The INNER
        ``rho`` edge ``rho_inner = rho_center - half_rho`` -- the point of the
        tile CLOSEST to the caustic, hence the hardest to admit -- is probed at
        `_INTERIOR_EDGE_SAMPLES` polar angles across the tile's ``theta_c``
        span.  Each physical probe is reconstructed with the additive
        positive-parity exterior form
        ``y_mag = r_caustic(gamma, theta_c) + rho_inner - 1`` (the ``rho > 1``
        arm of `surrogate._from_caustic_fixed`) at every sampled gamma in the
        band.

        A tile is admitted iff (1) its inner ``rho`` edge is strictly outside
        the caustic (``rho_inner > 1``); (2) EVERY probe (every band gamma,
        every angle) is at least ``eta_max`` from the nearest per-gamma
        caustic-cloud point -- the SAME cloud test `admits` uses, because near
        a cusp the nearest caustic point is off the radial ray; and (3) the
        tile-CENTRE direction stays inside the prior source box for every band
        gamma (``r_caustic(gamma, theta_center) + rho_inner - 1 <=
        source_magnitude_max``).  The box (usefulness) gate is evaluated at the
        centre direction ONLY (WP1 defect 2): columns are cusp-aligned, so the
        centre is representative and an off-centre angular probe poking out of
        the box no longer discards a tile that still admits useful in-box
        sources; the caustic-distance (correctness) gate is independent and
        remains an ``np.any`` over all 5 probes and every gamma.  This
        per-direction test replaces the single over-conservative scalar
        ``exclusion_rho`` built from the cusp-spike ``_caustic_reach``, which
        excluded the whole prior box for ``gamma >= 0.85`` (exterior coverage
        0.000).

        Parameters
        ----------
        center : tuple[float, float]
            Tile centre ``(rho_center, theta_c_center)``.
        half : tuple[float, float]
            Tile half-widths ``(half_rho, half_theta_c)``.
        source_magnitude_max : float
            Largest physical source magnitude in the prior box (the union
            extent ``y_outer_region``); a probe beyond it is out of the box.
        """
        rho_center, theta_center = center
        half_rho, half_theta = half
        rho_inner = float(rho_center) - float(half_rho)
        if rho_inner <= 1.0:
            return False
        thetas = np.linspace(theta_center - half_theta,
                             theta_center + half_theta, _INTERIOR_EDGE_SAMPLES)
        for radius_axis, caustic_cloud in zip(
                self.radius_grid, self.caustic_clouds):
            if caustic_cloud.shape[0] == 0:
                return False
            radii = np.interp(thetas, self.theta_axis, radius_axis)
            y_magnitudes = radii + rho_inner - 1.0
            # Box (usefulness) gate on the tile-CENTRE direction only (WP1
            # defect 2).  Columns are cusp-aligned, so the centre is
            # representative and an off-centre angular probe poking out of the
            # prior box no longer discards the whole tile.  The
            # caustic-distance (correctness) gate below is INDEPENDENT and
            # still spans all 5 probes and every band gamma, so relaxing the
            # box test cannot admit a near-caustic tile -- only one whose
            # angular edge sees out-of-box (large ``|y|``) sources.
            if (np.interp(theta_center, self.theta_axis, radius_axis)
                    + rho_inner - 1.0 > source_magnitude_max):
                return False
            probe_x = y_magnitudes * np.cos(thetas)
            probe_y = y_magnitudes * np.sin(thetas)
            delta_x = probe_x[:, None] - caustic_cloud[None, :, 0]
            delta_y = probe_y[:, None] - caustic_cloud[None, :, 1]
            nearest = np.sqrt(
                delta_x * delta_x + delta_y * delta_y,
            ).min(axis=1)
            if np.any(nearest < self.eta_max):
                return False
        return True


def _interior_admission(band: tuple[float, float], parity: int, reach: float,
                        config: 'TrainingConfig',
                        eta_max: float) -> _InteriorAdmission:
    """Precompute the directional interior-admission geometry for one band.

    The retained ``reach`` argument is ignored for call-site compatibility:
    positive-parity chart rho now uses the directional radius at each gamma and
    angle. The physical radius grid and matching caustic cloud are stored per
    sampled gamma so the tube-shell distance test never mixes coordinate scales
    from different members of the band.
    """
    del reach
    if parity != 1:
        raise ValueError(
            'Origin-centred interior admission is defined only for the '
            'positive-parity astroid.')
    gamma_lo, gamma_hi = band
    gamma_mid = 0.5 * (gamma_lo + gamma_hi)
    band_gammas = (gamma_lo, gamma_mid, gamma_hi)
    theta_axis = np.linspace(-math.pi, math.pi, _INTERIOR_BOUNDARY_NODES)
    radius_grid = np.array([
        [geometry.r_caustic(float(gamma), float(theta))
         for theta in theta_axis]
        for gamma in band_gammas
    ])
    caustic_clouds = tuple(
        _caustic_points(gamma, parity, config.n_caustic_samples)
        for gamma in band_gammas
    )
    return _InteriorAdmission(
        eta_max=eta_max, theta_axis=theta_axis,
        radius_grid=radius_grid, caustic_clouds=caustic_clouds,
        gammas=tuple(float(g) for g in band_gammas))


def _farfield_exterior_tiles(rho_outer: float, n_per_side: int, *,
                             admission: '_InteriorAdmission',
                             source_magnitude_max: float,
                             cusp_angles: list[float] | None = None,
                             gamma: float | None = None
                             ) -> list[tuple[tuple[float, float],
                                             tuple[float, float], int, int]]:
    """Per-``theta_c``-column exterior tiles of the caustic-fixed region.

    Positive-parity companion to `_farfield_tiles` that replaces the single
    over-conservative scalar ``exclusion_rho`` inner edge (built from the
    cusp-spike ``_caustic_reach``, which swallowed the whole prior box for
    ``gamma >= 0.85``) with a per-column DIRECTIONAL admission
    (`_InteriorAdmission.admits_exterior`).  Per column, ``n_per_side`` ``rho``
    rows over ``[1, rho_outer]`` (``rho = 1`` is the caustic in every
    direction).  The ``theta_c`` columns follow the same cusp-alignment
    convention as the interior tiler: when ``cusp_angles`` is
    supplied the columns come from `_cusp_aligned_theta_tiles` so no admitted
    tile straddles an astroid cusp ray (an ``r_caustic`` slope kink) or the
    ``+-pi`` branch cut -- the positive-parity exterior ``rho > 1`` arm of
    `surrogate._from_caustic_fixed` is a ``theta_c``-independent affine
    push-out of ``r_caustic(gamma, theta_c)``, so it inherits the same four
    source-plane cusp rays as the interior.  When ``cusp_angles`` is None/empty
    the columns fall back to the byte-identical UNIFORM ``theta_c`` grid over
    ``[-pi, pi]`` (edges pinned on ``+-pi``) so existing callers/tests are
    unaffected.  Before the admission test, a tile whose corners are within
    ``_CUSP_EXCLUSION_DISTANCE`` of an astroid cusp vertex (when ``gamma`` is
    supplied and ``cusp_angles`` is non-empty) is silently dropped -- near-
    cusp tiles induce oscillatory ``E_ff`` labels that a polar chart cannot
    resolve, so the tube/cusp-arm serves them instead.  A tile is kept iff
    ``admission.admits_exterior`` is True: its INNER ``rho`` edge stays
    outside the caustic, at least ``eta_max`` from the nearest caustic point
    (over all probes and band gammas), and its centre direction is inside the
    prior source box, for every gamma in the band.  The kept tiles form a
    per-column band whose inner radius emerges from the direction's true
    caustic distance.

    Parameters
    ----------
    rho_outer : float
        Outer prior-support radius in caustic-fixed ``rho`` units.
    n_per_side : int
        Number of ``rho`` rows and of ``theta_c`` sub-tiles per cusp sector
        (or over the whole ``[-pi, pi]`` sector when no cusp ray is supplied).
    admission : _InteriorAdmission
        The band's directional-admission geometry (`_interior_admission`),
        reused via its exterior probe `admits_exterior`.
    source_magnitude_max : float
        Largest physical source magnitude in the region (the union extent
        ``y_outer_region``); a centre-direction probe beyond it lies outside
        the prior box.
    cusp_angles : list of float, optional
        Source-plane cusp-ray angles (`_cusp_source_angles`).  When non-empty
        the ``theta_c`` tile edges are aligned to them (and the ``+-pi`` branch
        cut) so no tile straddles a cusp kink; when None/empty the columns fall
        back to the byte-identical uniform ``theta_c`` grid.
    gamma : float, optional
        Representative shear magnitude for the cusp-position computation.
        When supplied and ``cusp_angles`` is non-empty, tiles within
        ``_CUSP_EXCLUSION_DISTANCE`` of a cusp vertex are silently dropped.
        When None the cusp-exclusion step is skipped (backward-compatible).

    Returns
    -------
    list[tuple[tuple[float, float], tuple[float, float], int, int]]
        ``((rho_center, theta_c_center), (half_rho, half_theta_c), i, j)`` for
        each admitted tile, in row-major order (``i`` the ``rho`` row from the
        caustic outward, ``j`` the ``theta_c`` column).  Empty when no column
        admits any tile.  Because ``i`` runs inner-first, ``tiles[0]`` is the
        tile with the SMALLEST admitted ``rho_inner`` (closest to the caustic,
        hardest to fit) -- the reprovision probe and the region ``w_floor``.
    """
    rho_inner_floor = 1.0
    if rho_outer <= rho_inner_floor:
        return []
    half_rho = 0.5 * (rho_outer - rho_inner_floor) / n_per_side
    rho_centers = [rho_inner_floor + half_rho * (2 * k + 1)
                   for k in range(n_per_side)]
    if cusp_angles:
        theta_tiles = _cusp_aligned_theta_tiles(cusp_angles, n_per_side)
    else:
        half_theta = math.pi / n_per_side
        theta_tiles = [(-math.pi + half_theta * (2 * k + 1), half_theta)
                       for k in range(n_per_side)]
    tiles: list[tuple[tuple[float, float], tuple[float, float], int, int]] = []
    for i, rho_c in enumerate(rho_centers):
        for j, (theta_c, half_theta) in enumerate(theta_tiles):
            center = (float(rho_c), float(theta_c))
            half = (float(half_rho), float(half_theta))
            if (gamma is not None and cusp_angles
                    and _exclude_near_cusp(gamma, center, half, cusp_angles)):
                continue
            if admission.admits_exterior(center, half, source_magnitude_max):
                tiles.append((center, half, i, j))
    return tiles


def _lobe_caustic_points(gamma: float, lens_center: float,
                         n: int) -> np.ndarray:
    """Source-plane caustic points of ONE macro-saddle deltoid lobe, ``(k,
    2)``.

    Sweeps the single lobe centred at lens-plane angle ``lens_center`` (0 or
    pi) over its critical wedge ``|sin 2 theta| <= 1 / |gamma|`` (kappa = 0),
    both square-root branches, dropping wedge-forbidden angles.  Companion to
    `_caustic_points` restricted to one lobe -- the per-lobe cloud the S2-2
    interior admission needs for its centroid, winding loop, and
    nearest-caustic distance.
    """
    theta_max = 0.5 * np.arcsin(1.0 / abs(gamma))
    thetas = np.linspace(lens_center - theta_max,
                         lens_center + theta_max, n)
    points: list[np.ndarray] = []
    for branch in (1, -1):
        for theta in thetas:
            try:
                src = geometry.critical_point(
                    gamma, float(theta), 0.0, 0.0, branch).source
            except geometry.LensDomainError:
                continue
            points.append(np.asarray(src, dtype=float))
    return np.asarray(points) if points else np.empty((0, 2))


def _lobe_winding_loop(gamma: float, lens_center: float,
                       n: int) -> np.ndarray:
    """Ordered closed boundary of ONE deltoid lobe, shape ``(k, 2)``.

    Traverses the lobe's ``+`` square-root branch across its wedge (low to high
    lens angle) then the ``-`` branch back (high to low), so the concatenated
    source-plane points trace the closed 3-cusp deltoid boundary IN ORDER --
    the two branches merge at the wedge turnarounds where the discriminant
    vanishes.  The ordered loop is what `_winding_number` needs to test whether
    a candidate source ``p`` lies inside the lobe (translate the loop by ``-p``
    and read the winding about the origin).  Wedge-forbidden angles are
    dropped.  With the wedge endpoints now sampled at the true edge, the
    endpoint discriminant clamps to zero and both branches meet at the same
    source-plane turnaround point, so the loop closes exactly (bit-exact
    ``0.0`` separation between its first and last vertices); a point well
    inside the lobe still winds ``+-1`` with margin against the ``0.5``
    interior threshold.
    """
    theta_max = 0.5 * np.arcsin(1.0 / abs(gamma))
    lo = lens_center - theta_max
    hi = lens_center + theta_max
    thetas = np.linspace(lo, hi, n)
    loop: list[np.ndarray] = []
    for branch, sweep in ((1, thetas), (-1, thetas[::-1])):
        for theta in sweep:
            try:
                src = geometry.critical_point(
                    gamma, float(theta), 0.0, 0.0, branch).source
            except geometry.LensDomainError:
                continue
            loop.append(np.asarray(src, dtype=float))
    return np.asarray(loop) if len(loop) >= 3 else np.empty((0, 2))


def _lobe_cusp_source_angles(gamma: float, lens_center: float,
                             centroid: np.ndarray, n: int) -> list[float]:
    """Lobe-local polar angles (rad, sorted) of one deltoid lobe's three cusps.

    The lobe's cusps are the caustic-speed minima of its two square-root branch
    sweeps (`_find_cusps` -- the SAME detector the fold-arc tiler and the
    astroid interior use, with the wider saddle windows); each cusp's
    lens-plane angle is mapped through `critical_point` to its source-plane
    image and reported as an ``atan2`` direction MEASURED FROM THE LOBE
    CENTROID (the lobe-local frame).  The lobe's directional radius
    ``r_deltoid`` has slope kinks exactly along these rays, so the interior
    tiler aligns its lobe-local ``theta`` tile edges to them and no tile
    straddles a kink (frozen WP7, same cusp-kink fix as S2-1).  An empty list
    (a degenerate band resolving no cusp) makes the tiler fall back to a
    uniform lobe-local tiling.
    """
    theta_max = 0.5 * np.arcsin(1.0 / abs(gamma))
    lo = lens_center - theta_max
    hi = lens_center + theta_max
    angles: list[float] = []
    for branch in (1, -1):
        thetas, speed = _branch_speed_profile(
            gamma, branch, lo, hi, n, periodic=False)
        for theta_lens, _delta in _find_cusps(
                thetas, speed, periodic=False, gamma=gamma, branch=branch,
                width_safety=_SADDLE_CUSP_WIDTH_SAFETY,
                min_halfwidth=_SADDLE_CUSP_MIN_HALFWIDTH):
            try:
                src = geometry.critical_point(
                    gamma, float(theta_lens), 0.0, 0.0, branch).source
            except geometry.LensDomainError:
                continue
            angles.append(float(np.arctan2(src[1] - centroid[1],
                                           src[0] - centroid[0])))
    return sorted(angles)


def _directional_lobe_boundary(points: np.ndarray, centroid: np.ndarray,
                               n_bins: int = _INTERIOR_BOUNDARY_NODES
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Lobe-local directional boundary radius ``r_deltoid(theta_local)``
    (S2-2).

    Bins the lobe caustic ``points`` by their polar angle about ``centroid``
    and takes, per angular bin, the largest ``|point - centroid|`` -- the
    boundary radius in that direction (the deltoid is star-shaped about its
    symmetry centre, so each direction has a single outer crossing).  Empty
    bins are filled by periodic interpolation from populated neighbours.  The
    returned ``(centers, radii)`` (both shape ``(n_bins,)``, ``centers``
    ascending on ``(-pi, pi]``) normalise the lobe-local radial coordinate
    ``rho_lobe = |y - centroid| / r_deltoid(theta_local)`` so ``rho_lobe = 1``
    tracks the deltoid boundary in EVERY direction -- unlike a scalar reach,
    which overshoots the near-cusp directions of an elongated (sheared) lobe
    and leaves its interior untileable.  ``r_deltoid`` has slope kinks exactly
    on the three per-lobe cusp rays (the deltoid's 3-fold structure); the tiler
    aligns its lobe-local ``theta`` edges to those rays so no tile straddles a
    kink.

    Returns ``(centers, radii)`` with all-zero ``radii`` when ``points`` is
    empty (a degenerate lobe the caller then admits nothing into).
    """
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if points.shape[0] == 0:
        return centers, np.zeros(n_bins)
    rel = points - centroid
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    rad = np.hypot(rel[:, 0], rel[:, 1])
    bin_idx = np.clip(np.digitize(ang, edges) - 1, 0, n_bins - 1)
    radii = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = rad[bin_idx == b]
        if sel.size:
            radii[b] = sel.max()
    valid = ~np.isnan(radii)
    if not valid.any():
        return centers, np.zeros(n_bins)
    radii = np.interp(centers, centers[valid], radii[valid],
                      period=2.0 * np.pi)
    return centers, radii


@dataclass(frozen=True, eq=False)
class _SaddleLobeAdmission:
    """Per-lobe interior admission for one macro-saddle deltoid lobe (S2-2).

    Frozen WP7.  For ``gamma > 1`` the caustic is two disjoint 3-cusp deltoid
    lobes sitting off the origin on the shear axis; neither encloses the
    origin, so the origin-centred astroid admission (`_InteriorAdmission`) does
    not apply.  Each lobe instead gets its OWN interior family in a lobe-local
    frame centred on the lobe's SOURCE-PLANE deltoid centroid ``centroid`` -- a
    regular interior source point (the deltoid's symmetry centre on the shear
    axis), estimated as the mean of the densely sampled lobe caustic points.

    A candidate eigenframe source ``p`` is admitted into this lobe iff
    simultaneously

    * it lies INSIDE the lobe for EVERY gamma in the band -- the winding number
      of the ordered lobe boundary about ``p`` is ``+-1`` for each band loop
      (`_winding_number` on ``loop - p``; the topological interior test the WP
      pins, robust where a directional-radius threshold is fragile near the
      three shallow deltoid cusps); AND
    * it is at least ``eta_max`` (dimensionless ``y``) from the nearest point
      of the band's lobe caustic cloud -- the tube-shell exclusion, off-radial
      near a cusp, the same nearest-distance test as S2-1; AND
    * it is strictly nearer THIS lobe's centroid than the other lobe's by at
      least ``corridor_half``:
      ``|p - centroid| + corridor_half <= |p - other_centroid|``.  This assigns
      each inter-lobe source to exactly one lobe and EXCLUDES the thin corridor
      on the lobe-equidistance (perpendicular-bisector) line where the
      assignment flips, so no admitted tile straddles the inter-lobe line.

    The lobe-local radial coordinate ``rho_lobe = |y - centroid| /
    r_deltoid(theta_local)`` is normalised by the DIRECTIONAL boundary radius
    ``r_deltoid`` (`boundary_theta` / `boundary_r`, `_r_deltoid`), so
    ``rho_lobe = 1`` tracks the deltoid boundary in every direction -- a scalar
    reach overshoots the near-cusp directions of an elongated (sheared) lobe
    and leaves its interior untileable.  ``reach`` is retained only as the
    scalar lobe extent (max ``|caustic - centroid|``) for reporting; per-lobe
    membership is decided by the winding number, not the normalisation.

    Attributes
    ----------
    centroid : np.ndarray
        ``(2,)`` source-plane deltoid centroid -- the lobe-local frame origin.
    other_centroid : np.ndarray
        ``(2,)`` centroid of the OTHER lobe, for the inter-lobe corridor test.
    reach : float
        Scalar lobe extent (max ``|caustic - centroid|``, dimensionless ``y``),
        for reporting only; ``rho_lobe`` is normalised by ``r_deltoid``.
    eta_max : float
        Tube-shell half-width excluded from the interior (dimensionless ``y``).
    corridor_half : float
        Inter-lobe corridor half-width (dimensionless ``y``).
    loops : tuple[np.ndarray, ...]
        Ordered lobe boundaries (one per band gamma) for the winding test.
    caustic_cloud : np.ndarray
        ``(K, 2)`` lobe caustic points across the band for the nearest-distance
        tube-shell test.
    boundary_theta : np.ndarray
        Ascending lobe-local angular nodes on ``(-pi, pi]`` for ``r_deltoid``.
    boundary_r : np.ndarray
        Directional boundary radius ``r_deltoid(boundary_theta)`` normalising
        ``rho_lobe`` (`_directional_lobe_boundary`).
    """

    centroid: np.ndarray
    other_centroid: np.ndarray
    reach: float
    eta_max: float
    corridor_half: float
    loops: tuple
    caustic_cloud: np.ndarray
    boundary_theta: np.ndarray
    boundary_r: np.ndarray

    def _r_deltoid(self, theta: np.ndarray) -> np.ndarray:
        """Directional lobe boundary radius at lobe-local angle(s) ``theta``.

        Delegates to the authoritative ``surrogate._lobe_boundary_radius`` so
        that the deltoid-boundary convention (periodic linear interpolation of
        ``boundary_r`` over ``boundary_theta``; ``rho_lobe = 1`` traces the
        deltoid boundary) has a single source shared with the lobe coordinate
        maps and ``from_lobe_engine``.
        """
        return _lobe_boundary_radius(theta, self.boundary_theta,
                                     self.boundary_r)

    def _probe_points(self, center: tuple[float, float],
                      half: tuple[float, float]) -> np.ndarray:
        """Eigenframe positions of a lobe-local tile's corners, edges and
        centre.

        ``center`` is ``(rho_lobe_center, theta_local_center)`` and ``half`` is
        ``(half_rho, half_theta)`` in the lobe-local polar frame.  Nine probes
        (the 3x3 outer product of the ``rho`` and ``theta`` extremes and their
        midpoints) cover the whole tile so admission tests the tile, not just
        its centre -- a deltoid is non-convex, so a tile may exit the lobe on
        any side.
        """
        rho_c, theta_c = center
        half_rho, half_theta = half
        rhos = np.clip(
            np.array([rho_c - half_rho, rho_c, rho_c + half_rho]), 0.0, None)
        thetas = np.array(
            [theta_c - half_theta, theta_c, theta_c + half_theta])
        rr, tt = np.meshgrid(rhos, thetas)
        rr = rr.ravel()
        tt = tt.ravel()
        radius = rr * self._r_deltoid(tt)
        probe_x = self.centroid[0] + radius * np.cos(tt)
        probe_y = self.centroid[1] + radius * np.sin(tt)
        return np.column_stack([probe_x, probe_y])

    def admits(self, center: tuple[float, float],
               half: tuple[float, float]) -> bool:
        """Whether every probe of the lobe-local tile is a served lobe
        interior.

        The tile is admitted iff each of its nine probes lies inside the lobe
        for every band gamma (winding), at least ``eta_max`` from the nearest
        caustic point, and strictly nearer this lobe's centroid than the
        other's by the corridor half-width.  Any failing probe rejects the
        tile.
        """
        if self.reach <= 0.0 or self.caustic_cloud.shape[0] == 0 \
                or not self.loops:
            return False
        for probe in self._probe_points(center, half):
            for loop in self.loops:
                if loop.shape[0] < 3 \
                        or abs(_winding_number(loop - probe)) < 0.5:
                    return False
            nearest = float(np.hypot(
                self.caustic_cloud[:, 0] - probe[0],
                self.caustic_cloud[:, 1] - probe[1]).min())
            if nearest < self.eta_max:
                return False
            near_this = math.hypot(probe[0] - self.centroid[0],
                                   probe[1] - self.centroid[1])
            near_other = math.hypot(probe[0] - self.other_centroid[0],
                                    probe[1] - self.other_centroid[1])
            if near_this + self.corridor_half > near_other:
                return False
        return True


def _saddle_lobe_admissions(band: tuple[float, float],
                            config: 'TrainingConfig',
                            eta_max: float
                            ) -> list[_SaddleLobeAdmission]:
    """Build the two per-lobe interior admissions for a macro-saddle band
    (S2-2).

    For each of the two deltoid lobes (lens-plane centres 0 and pi) collects
    the band's lobe caustic cloud and ordered winding loops (at the two band
    edges and the midpoint), estimates the lobe's source-plane centroid as the
    mean of the midpoint-gamma lobe caustic points, and its scalar reach as the
    largest caustic distance from that centroid over the band.  Each lobe is
    paired with the OTHER lobe's centroid for the inter-lobe corridor test; the
    corridor half-width is ``_INTERLOBE_CORRIDOR_ETA_SCALE * eta_max`` (one
    tube shell around the lobe-equidistance line).  Returns one
    `_SaddleLobeAdmission` per lobe in lens-centre order.
    """
    gamma_lo, gamma_hi = band
    gamma_mid = 0.5 * (gamma_lo + gamma_hi)
    band_gammas = (gamma_lo, gamma_mid, gamma_hi)
    n = config.n_caustic_samples
    centroids: list[np.ndarray] = []
    clouds: list[np.ndarray] = []
    loops_per_lobe: list[tuple] = []
    boundaries: list[tuple[np.ndarray, np.ndarray]] = []
    for lens_center in _SADDLE_LOBE_CENTERS:
        cloud_parts = [pts for pts in
                       (_lobe_caustic_points(g, lens_center, n)
                        for g in band_gammas) if pts.shape[0] > 0]
        cloud = np.vstack(cloud_parts) if cloud_parts else np.empty((0, 2))
        mid_points = _lobe_caustic_points(gamma_mid, lens_center, n)
        centroid = (mid_points.mean(axis=0) if mid_points.shape[0] > 0
                    else np.zeros(2))
        loops = tuple(loop for loop in
                      (_lobe_winding_loop(g, lens_center, n)
                       for g in band_gammas) if loop.shape[0] >= 3)
        # Directional boundary ``r_deltoid(theta_local)`` from the
        # midpoint-gamma deltoid (a single clean lobe); normalises ``rho_lobe``
        # per direction so the elongated near-cusp directions stay tileable.
        # Winding over the band loops still guards membership at the band
        # edges.
        centroid = np.asarray(centroid, dtype=float)
        boundaries.append(_directional_lobe_boundary(mid_points, centroid))
        centroids.append(centroid)
        clouds.append(cloud)
        loops_per_lobe.append(loops)
    corridor_half = _INTERLOBE_CORRIDOR_ETA_SCALE * eta_max
    admissions: list[_SaddleLobeAdmission] = []
    for k in range(len(_SADDLE_LOBE_CENTERS)):
        centroid = centroids[k]
        cloud = clouds[k]
        reach = (float(np.hypot(cloud[:, 0] - centroid[0],
                                cloud[:, 1] - centroid[1]).max())
                 if cloud.shape[0] > 0 else 0.0)
        boundary_theta, boundary_r = boundaries[k]
        admissions.append(_SaddleLobeAdmission(
            centroid=centroid, other_centroid=centroids[1 - k],
            reach=reach, eta_max=eta_max,
            corridor_half=corridor_half, loops=loops_per_lobe[k],
            caustic_cloud=cloud, boundary_theta=boundary_theta,
            boundary_r=boundary_r))
    return admissions


def _lobe_interior_tiles(admission: _SaddleLobeAdmission,
                         cusp_angles: list[float], n_per_side: int
                         ) -> list[tuple[tuple[float, float],
                                         tuple[float, float], int, int]]:
    """Cusp-aligned lobe-local interior tiles of one deltoid lobe (S2-2).

    Lays ``n_per_side`` uniform ``rho_lobe`` rows over ``rho_lobe in [0, 1]``
    (centroid to lobe reach) and, on the lobe-local polar angle, cusp-aligned
    sub-tiles (`_cusp_aligned_theta_tiles`) so no tile straddles one of the
    lobe's three cusp rays or the lobe-local ``+-pi`` seam.  A tile is ADMITTED
    iff `_SaddleLobeAdmission.admits` -- inside the lobe by winding for every
    band gamma, clear of the ``eta_max`` tube shell, and out of the inter-lobe
    corridor.  Returns
    ``((rho_lobe_center, theta_local_center), (half_rho, half_theta), i, j)``
    for each admitted tile in row-major order (deterministic); ``i`` indexes
    the ``rho_lobe`` row, ``j`` the cusp-aligned lobe-local angular sub-tile.
    """
    if admission.reach <= 0.0:
        return []
    half_rho = 0.5 / n_per_side
    rho_centers = [half_rho * (2 * k + 1) for k in range(n_per_side)]
    theta_tiles = _cusp_aligned_theta_tiles(cusp_angles, n_per_side)
    tiles: list[tuple[tuple[float, float], tuple[float, float], int, int]] = []
    for i, rho_c in enumerate(rho_centers):
        for j, (theta_c, half_theta) in enumerate(theta_tiles):
            center = (float(rho_c), float(theta_c))
            half = (float(half_rho), float(half_theta))
            if admission.admits(center, half):
                tiles.append((center, half, i, j))
    return tiles


def _wedge_interior_tiles(gamma: float, r_extent: float, n_per_side: int
                          ) -> list[tuple[tuple[float, float],
                                          tuple[float, float], int, int, str]]:
    """Radial-row x waist-split angular-column tiles of the astroid interior.

    The wedge-caustic counterpart of `_lobe_interior_tiles`, in WEDGE-FIXED
    caustic-relative coordinates ``(r, theta_wedge)`` (``r = |y| /
    r_caustic(gamma, theta_wedge)`` in ``[0, 1)``; ``theta_wedge =
    atan2(|y2|, |y1|)`` in ``[0, pi/2]``).  Because ``r_caustic`` is exactly
    four-fold symmetric, the ``[0, pi/2]`` wedge is one quadrant of the
    interior and the D2 fold serves the other three by symmetry.

    The astroid's two cusps sit at the wedge EDGES ``theta_wedge = 0`` and
    ``pi/2`` (where ``r_caustic`` is largest); its regular WAIST -- the angular
    minimum ``theta_waist = argmin_theta r_caustic(gamma, theta)`` -- sits
    between them.  The chart's cusp-adapted angular spline axis ``u =
    d**(2/3)`` (``d`` = distance to the NEAR cusp) is per-tile monotone, so
    each angular column must lie entirely on ONE side of the waist and carry
    the matching near-cusp origin.  This helper therefore emits TWO angular
    columns per radial row, split at ``theta_waist`` -- NOT at ``pi/4``: the
    external shear stretches the astroid, so the two cusps are inequivalent and
    the waist migrates up to ~30% away from ``pi/4`` as ``gamma`` grows (worst
    exactly where the asymmetry is largest; see `_wedge_theta_waist`).  The low
    column ``theta_wedge in [0, theta_waist]`` carries ``axis_origin='low'``
    (near cusp at ``0``, ``d = theta``); the high column ``theta_wedge in
    [theta_waist, pi/2]`` carries ``axis_origin='high'`` (near cusp at ``pi/2``,
    ``d = pi/2 - theta``).

    The angular columns span the cusp EDGES with NO exclusion strip: unlike the
    degenerate astroid CENTRE (``r = 0``, where ``theta_wedge`` is undefined --
    excluded by `_WEDGE_R_MIN` and served by the exact engine), the cusp edges
    ARE chartable because ``u = d**(2/3)`` absorbs the ``r_caustic ~ const -
    c * d**(2/3)`` cusp scaling that otherwise makes the raw-``theta`` spline
    diverge as ``d**(-1/3)`` -- there is no angular analogue of the centre
    exclusion.  ``n_per_side`` UNIFORM radial rows tile ``r in [_WEDGE_R_MIN,
    r_extent]`` (``r_min`` strictly positive); the caller caps ``r_extent``
    below one so the Airy caustic edge (``r -> 1``) is left to the tube chart.

    Parameters
    ----------
    gamma : float
        Band-representative external shear, used ONLY to locate the caustic
        waist ``theta_waist`` at which the two angular columns split.  Callers
        pass the SAME representative `from_wedge_engine` uses internally (the
        ``median`` of the log-reach gamma grid) so the tiler's split boundary
        and the engine's per-tile near-cusp classification agree exactly (no
        train/serve skew, this repo's #1 bug class).
    r_extent : float
        Outer radial bound in caustic-relative ``r`` units (capped below one
        by the caller).
    n_per_side : int
        Number of radial rows.

    Returns
    -------
    list[tuple[tuple[float, float], tuple[float, float], int, int, str]]
        ``((r_center, theta_wedge_center), (half_r, half_theta_wedge), i, j,
        axis_origin)`` for each tile in ``(radial row, angular column)`` order
        (deterministic).  ``i`` indexes the radial row; ``j`` is ``0`` for the
        low column and ``1`` for the high column; ``axis_origin`` is ``'low'``
        or ``'high'`` (the near-cusp side, threaded unchanged into
        `_build_wedge_chart` -> `from_wedge_engine`).
    """
    if r_extent <= _WEDGE_R_MIN:
        return []
    half_r = 0.5 * (r_extent - _WEDGE_R_MIN) / n_per_side
    r_centers = [_WEDGE_R_MIN + half_r * (2 * k + 1)
                 for k in range(n_per_side)]
    # Split the wedge at the true caustic waist (NOT pi/4): the two cusps are
    # inequivalent under the shear and the waist migrates with gamma.
    theta_waist = _wedge_theta_waist(float(gamma))
    half_pi = 0.5 * np.pi
    # (theta_lo, theta_hi, axis_origin, j) per angular column.  No cusp-edge
    # exclusion strip: the u = d**(2/3) spline axis absorbs the cusp scaling.
    columns = ((0.0, theta_waist, 'low', 0),
               (theta_waist, half_pi, 'high', 1))
    tiles: list[tuple[tuple[float, float], tuple[float, float],
                      int, int, str]] = []
    for i, r_c in enumerate(r_centers):
        for theta_lo, theta_hi, axis_origin, j in columns:
            theta_center = 0.5 * (theta_lo + theta_hi)
            half_theta = 0.5 * (theta_hi - theta_lo)
            center = (float(r_c), float(theta_center))
            half = (float(half_r), float(half_theta))
            tiles.append((center, half, i, j, axis_origin))
    return tiles


def _stratum_ppgo_boundary(parity: int, gamma: float, rho: float,
                           ppgo_map: CertifiedPpgoMap | None
                           ) -> float | None:
    """Certified-ppGO dispatch floor ``w_trust`` for a region, or ``None``.

    Returns the margin-inflated handoff floor (a ``float``) only when the map
    certifies the ``(parity, gamma, rho)`` cell; ``None`` when no map is
    installed or the cell is `UNKNOWN` (out-of-grid / beyond-wall /
    uncertified).  The caller trims a stratum's chart w-range against this
    floor: whole band above it -> ppGO serves the stratum, drop the chart; top
    above it -> cap the chart at the floor (band-split serving hands the tail
    to ppGO).  ``None`` -> no trim, keep the chart intact.

    The floor is ``w_trust`` (margin-inflated), NOT the raw ``w_cert``: the
    ``[w_cert, w_trust]`` band is the dispatch hand-off margin and must stay
    with the chart, else that band routes to a dropped / capped chart and
    leaves a serving gap.
    """
    if ppgo_map is None:
        return None
    parity_str = 'positive' if parity == 1 else 'saddle'
    floor = ppgo_map.w_trust(parity_str, float(gamma), float(rho))
    if floor is UNKNOWN:
        return None
    return float(floor)


def _stratum_ppgo_ceiling(parity: int, gamma: float, rho: float,
                          ppgo_map: CertifiedPpgoMap | None
                          ) -> float | None:
    """Measured ppGO ``w`` ceiling for a region, or ``None``.

    Reads ``w_ceiling`` from the SAME ``(parity, gamma, rho)`` cell as
    `_stratum_ppgo_boundary` reads ``w_trust`` (Build 8h-b).  Returns the
    measured ceiling (a ``float``) only when the map certifies the cell;
    ``None`` when no map is installed or the cell is `UNKNOWN` (out-of-grid /
    beyond-wall / uncertified).  `_apply_ppgo_trim` trims a stratum only when
    this ceiling covers the stratum top; above it the reference is UNKNOWN, so
    the tail stays charted / refused, never handed to ppGO.  A certified cell
    always carries a finite ceiling, so a non-``None`` ``w_trust`` from
    `_stratum_ppgo_boundary` implies a non-``None`` ceiling here (both gate on
    the same certified cell).
    """
    if ppgo_map is None:
        return None
    parity_str = 'positive' if parity == 1 else 'saddle'
    ceiling = ppgo_map.w_ceiling(parity_str, float(gamma), float(rho))
    if ceiling is UNKNOWN:
        return None
    return float(ceiling)


def _apply_ppgo_trim(w_range: tuple[float, float], boundary: float | None,
                     ceiling: float | None = None
                     ) -> tuple[tuple[float, float], str]:
    """Trim a stratum ``w`` range against the ppGO hand-off floor.

    Returns ``(new_w_range, action)`` with ``action`` one of ``'drop'`` (the
    whole band lies above the floor -- ppGO serves it, no chart needed),
    ``'cap'`` (the top is lowered to the floor) or ``'keep'`` (unchanged).  A
    ``None`` boundary (no map / `UNKNOWN` cell) always keeps the range.

    ``ceiling`` is the cell's MEASURED ``w`` ceiling (`_stratum_ppgo_ceiling`,
    Build 8h-b).  A stratum is trimmed (``'drop'`` / ``'cap'``) only when the
    ceiling covers the stratum top (``w_max <= ceiling``); when the top exceeds
    the ceiling the exact reference is UNKNOWN there, so the chart is kept
    intact (``'keep'``) and its tail routes to the loud whole-band refusal
    rather than to ppGO.  A ``None`` ceiling (UNKNOWN cell) imposes no ceiling
    constraint -- byte-identical to HEAD.
    """
    if boundary is None:
        return w_range, 'keep'
    w_min, w_max = w_range
    if ceiling is not None and w_max > ceiling:
        return w_range, 'keep'
    if w_min >= boundary:
        return w_range, 'drop'
    if w_max > boundary:
        return (w_min, boundary), 'cap'
    return w_range, 'keep'


def _budget_check(n_points: int, budget: int, name: str) -> None:
    """Fail fast if a chart's grid exceeds its per-chart engine-call budget."""
    if n_points > budget:
        raise ValueError(
            f'Chart {name!r} needs {n_points} engine calls but the per-chart '
            f'budget is {budget}. Reduce the grid or raise --engine-budget.')


def _tube_arc_length_map(gamma: float, arc: FoldArc,
                         n_map: int = _TUBE_ARC_MAP_SIZE
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Arc-length axis map ``theta -> s`` for one fold arc at fixed gamma.

    Returns ``(theta_fine, s_fine)`` where ``theta_fine`` is a uniform,
    strictly ascending grid of ``n_map`` points over ``[arc.theta_lo,
    arc.theta_hi]`` (the arc's wedge frame) and ``s_fine`` is the cumulative
    arc length ``s = integral |y'| dtheta`` from ``0`` at ``theta_lo``.

    The exact caustic parametric speed ``|y'(theta)|`` is evaluated with
    :func:`geometry.caustic_speed` on the arc's own ``branch`` and integrated
    by the trapezoidal rule (`scipy.integrate.cumulative_trapezoid`); no
    finite difference is used.  Because the cusp windows exclude the
    ``|y'| -> 0`` caustic cusps, the speed stays positive over the arc, so
    ``s_fine`` is finite and strictly increasing -- both are checked and
    raise :class:`ValueError` on violation.

    Parameters
    ----------
    gamma : float
        Convergence ratio at which to evaluate the caustic speed.
    arc : FoldArc
        The fold arc supplying ``theta_lo``, ``theta_hi`` and ``branch``.
    n_map : int
        Number of theta samples (map resolution).

    Returns
    -------
    theta_fine, s_fine : np.ndarray
        The arc-length axis map rows (each shape ``(n_map,)``).
    """
    theta_fine = np.linspace(arc.theta_lo, arc.theta_hi, n_map)
    speed = geometry.caustic_speed(gamma, theta_fine, branch=arc.branch)
    s_fine = cumulative_trapezoid(speed, theta_fine, initial=0.0)
    if not np.isfinite(s_fine).all():
        raise ValueError(
            f'Tube arc-length map is non-finite for gamma={gamma}, '
            f'branch={arc.branch} over [{arc.theta_lo}, {arc.theta_hi}].')
    if not np.all(np.diff(s_fine) > 0.0):
        raise ValueError(
            f'Tube arc-length map is not strictly increasing for gamma='
            f'{gamma}, branch={arc.branch}; the caustic speed vanishes inside '
            f'the arc (cusp windows should exclude the |y\'|->0 cusps).')
    return theta_fine, s_fine


def _build_tube_chart(*, gamma_grid: np.ndarray, arc: FoldArc, parity: int,
                      w_range: tuple[float, float], config: TrainingConfig,
                      eta_max: float, eta_floor: float
                      ) -> tuple[TubeChart, int, int]:
    """Build one tube chart over ``(log w, gamma, u=sqrt(eta), s)``.

    The fourth axis is ARC LENGTH ``s = integral |y'| dtheta`` (not raw
    theta): the ``theta`` nodes are placed as the images of a UNIFORM ``s``
    grid so the envelope is sampled uniformly in the physically meaningful
    coordinate.  The ``theta -> s`` map is built once at the band's
    representative (median) gamma via `_tube_arc_length_map` and stored on
    the chart (``theta_to_s``); the same map is read at serve time.

    Returns the chart, the number of engine calls, and the number of refused
    grid points (left as zeros in the value tensor).
    """
    log_w_grid = _log_w_grid(w_range, config.w_nodes_per_decade)
    w_grid = np.exp(log_w_grid)
    # --- Eta-axis uniformizing coordinate: u = sqrt(eta) ---
    # The Airy fold's uniformizing control is xi = (3 w DeltaTau / 4)^{2/3}.
    # At fixed eta (caustic distance), xi ~ eta^{3/2} * w^{2/3}: the
    # demodulated envelope (carrier removed) is smooth in xi because the
    # Airy function's oscillatory/decay structure is parameterized by xi.
    #
    # Since the tube chart splines over MULTIPLE w values on a separate log-w
    # axis, the eta axis must use a w-INDEPENDENT projection of xi.  The
    # correct choice is u = sqrt(eta): the fold's singular sqrt-branch
    # (magnification ~ 1/sqrt(eta)) is smooth in u, and u^2 = eta linearizes
    # the Airy transition region.  Concretely, the envelope's dependence on
    # eta enters through DeltaTau ~ eta^{3/2}, giving xi ~ (w * eta^{3/2})^{2/3}
    # = w^{2/3} * eta; at fixed w, xi is linear in eta = u^2, so uniform-in-u
    # places nodes quadratically in eta -- denser near the caustic (small eta)
    # where the Airy fringe structure varies fastest.
    #
    # Near CUSPS this breaks: the Pearcey catastrophe's control (x, y) takes
    # over and u = sqrt(eta) is no longer the correct uniformizing coordinate.
    # The cusp-window exclusion (cusp_windows on the chart) handles this by
    # excising theta intervals where the Pearcey regime dominates.
    u_grid = np.linspace(np.sqrt(eta_floor), np.sqrt(eta_max),
                         config.n_u)

    # Arc-length node placement: build the theta -> s map at the band's
    # representative gamma, then invert a uniform s grid back to theta so the
    # theta nodes cluster where the fold turns fastest.  Endpoints are forced
    # exactly onto the arc bounds (np.interp already lands there up to fp).
    rep_gamma = float(np.median(gamma_grid))
    theta_fine, s_fine = _tube_arc_length_map(rep_gamma, arc)
    s_total = float(s_fine[-1])
    s_grid = np.linspace(0.0, s_total, config.n_theta)
    theta_grid = np.interp(s_grid, s_fine, theta_fine)
    theta_grid[0] = arc.theta_lo
    theta_grid[-1] = arc.theta_hi
    theta_to_s = np.vstack([theta_fine, s_fine])

    n_points = gamma_grid.size * u_grid.size * theta_grid.size
    _budget_check(n_points, config.engine_budget, 'tube')

    shape = (log_w_grid.size, gamma_grid.size, u_grid.size, theta_grid.size)
    env_real = np.zeros(shape, dtype=float)
    env_imag = np.zeros(shape, dtype=float)
    calls = refused = 0
    for i_g, gamma in enumerate(gamma_grid):
        for i_u, u in enumerate(u_grid):
            eta = float(u * u)
            for i_t, theta in enumerate(theta_grid):
                source = _tube_source(float(gamma), float(theta), eta,
                                      arc.branch, arc.inward_sign)
                env = _engine_envelope(w_grid, float(gamma), source)
                calls += 1
                if env is None:
                    refused += 1
                    continue
                env_real[:, i_g, i_u, i_t] = env.real
                env_imag[:, i_g, i_u, i_t] = env.imag
    chart = TubeChart.from_values(
        gamma_grid=gamma_grid, u_grid=u_grid, theta_grid=theta_grid,
        log_w_grid=log_w_grid, envelope_real=env_real, envelope_imag=env_imag,
        image_count=arc.image_count, parity=parity,
        eta_floor=eta_floor, eta_max=eta_max,
        cusp_windows=arc.cusp_windows, s_grid=s_grid, theta_to_s=theta_to_s)
    return chart, calls, refused

def _build_farfield_chart(*, gamma_band: tuple[float, float], parity: int,
                          box_center: tuple[float, float],
                          half: tuple[float, float],
                          w_range: tuple[float, float], config: TrainingConfig,
                          w_nodes_per_decade: int | None = None
                          ) -> tuple[ExteriorPolarChart, int, int]:
    """Build one exterior-polar chart in caustic-fixed ``(rho, theta_c)``.

    The production tiler lays out exterior tiles in caustic-fixed
    ``(rho, theta_c)``; ``box_center`` is ``(rho_center, theta_c_center)`` and
    ``half`` is ``(half_rho, half_theta_c)``.  The chart interpolates directly
    in ``(rho, theta_c)`` -- the polar coordinate is self-contained,
    single-valued, and cusp-safe by construction.  Each grid node
    ``(gamma, rho, theta_c)`` is mapped to a physical eigenframe source via
    `_from_caustic_fixed` at that node's OWN gamma before the engine call.

    The chart is always trained on the exterior far-field kernel-sum label
    (`FARFIELD_KERNEL_SUM`); the interior is charted in wedge caustic-relative
    coordinates by `_build_wedge_chart` / `InteriorWedgeChart`.

    Both positive-parity astroid (``gamma < 1``) and macro-saddle
    (``gamma >= 1``) exterior regions are chartable in polar coordinates.
    A tile that cannot be charted (e.g. carrier-phase exceeds the Nyquist
    equivalent) raises `CarrierDiscontinuityError`, so the existing caller
    path records it as a ladder-served gap and defers to the exact engine
    at serve.

    ``w_nodes_per_decade`` overrides the ``w``-axis node density for THIS chart
    only; ``None`` falls back to ``config.w_nodes_per_decade``.
    """
    nodes_per_decade = (config.w_nodes_per_decade
                        if w_nodes_per_decade is None
                        else int(w_nodes_per_decade))
    n_points = config.n_gamma * config.n_rho * config.n_theta_c
    _budget_check(n_points, config.engine_budget, 'farfield')
    rho_center, theta_c_center = box_center
    half_rho, half_theta_c = half
    rho_range = (float(rho_center - half_rho),
                 float(rho_center + half_rho))
    theta_c_range = (float(theta_c_center - half_theta_c),
                     float(theta_c_center + half_theta_c))
    try:
        single = LensAmplificationSurrogate.from_engine(
            gamma_range=gamma_band, rho_range=rho_range,
            theta_c_range=theta_c_range, w_range=w_range,
            n_gamma=config.n_gamma, n_rho=config.n_theta_c,
            n_theta_c=config.n_rho,
            w_nodes_per_decade=nodes_per_decade,
            definition=FARFIELD_KERNEL_SUM)
    except CarrierDiscontinuityError as exc:
        raise CarrierDiscontinuityError(
            'Exterior-polar tile label winds faster than the Nyquist '
            f'equivalent ({exc}); recorded as a ladder-served gap.') from exc
    chart = single.charts[0]
    refused = int(chart.refused_points.shape[0])
    return chart, n_points, refused


def _build_lobe_chart(*, gamma_band: tuple[float, float], parity: int,
                      lobe: '_SaddleLobeAdmission',
                      box_center: tuple[float, float],
                      half: tuple[float, float],
                      w_range: tuple[float, float], config: TrainingConfig,
                      w_nodes_per_decade: int | None = None
                      ) -> tuple['LobeInteriorChart', int, int]:
    """Build one macro-saddle lobe-interior chart in lobe-local coordinates.

    The lobe-interior counterpart of `_build_farfield_chart`.  ``box_center``
    is ``(rho_lobe_center, theta_local_center)`` and ``half`` is
    ``(half_rho, half_theta)``; the chart is trained on the axis-aligned
    lobe-local box ``rho_lobe in [rho_lobe_center +- half_rho]`` x
    ``theta_local in [theta_local_center +- half_theta]`` via
    `LensAmplificationSurrogate.from_lobe_engine`, which maps each
    ``(gamma, rho_lobe, theta_local)`` node to a physical eigenframe source
    through the lobe frame (`_from_lobe_fixed`, NOT the origin-centred
    `_from_caustic_fixed`) and stores the ``tau_c``-demodulated
    `INTERIOR_SACR_C` envelope on a `LobeInteriorChart`.

    The lobe frame is read straight off ``lobe`` (`_SaddleLobeAdmission`):
    `centroid`, `other_centroid`, `corridor_half`, `boundary_theta`,
    `boundary_r` are all carried by `from_lobe_engine` and persisted on the
    chart so a served node maps back to its true physical source.  Building a
    lobe chart on a tile that straddles a critical-basin flip raises
    `CarrierDiscontinuityError` (the caller records the ladder-served gap).

    Only macro-saddle (``parity != 1``) bands have lobe interiors; a
    positive-parity call is a programming error.  ``w_nodes_per_decade``
    overrides the ``w``-axis node density for THIS chart only; ``None`` falls
    back to ``config.w_nodes_per_decade``.

    Returns
    -------
    tuple[LobeInteriorChart, int, int]
        The built `LobeInteriorChart` itself (unwrapped from the
        single-chart surrogate `from_lobe_engine` returns), the engine node
        count, and the number of refused nodes.

    Raises
    ------
    ValueError
        If ``parity == 1`` (positive-parity bands have no lobe interior).
    CarrierDiscontinuityError
        If the tile straddles a critical-basin flip (caller records the gap).
    """
    if parity == 1:
        raise ValueError(
            'lobe-interior charts exist only for macro-saddle (parity != 1) '
            f'bands; got parity={parity}.')
    nodes_per_decade = (config.w_nodes_per_decade
                        if w_nodes_per_decade is None
                        else int(w_nodes_per_decade))
    n_points = config.n_gamma * config.n_rho * config.n_theta_c
    _budget_check(n_points, config.engine_budget, 'lobe')
    rho_lobe_c, theta_local_c = box_center
    half_rho, half_theta = half
    rho_lobe_range = (rho_lobe_c - half_rho, rho_lobe_c + half_rho)
    theta_local_range = (theta_local_c - half_theta, theta_local_c + half_theta)
    single = LensAmplificationSurrogate.from_lobe_engine(
        admission=lobe, gamma_range=gamma_band,
        rho_lobe_range=rho_lobe_range, theta_local_range=theta_local_range,
        w_range=w_range, n_gamma=config.n_gamma, n_rho=config.n_rho,
        n_theta=config.n_theta_c, w_nodes_per_decade=nodes_per_decade)
    chart = single.charts[0]
    refused = int(chart.refused_points.shape[0])
    return chart, n_points, refused


def _build_wedge_chart(*, gamma_band: tuple[float, float], parity: int,
                       box_center: tuple[float, float],
                       half: tuple[float, float],
                       w_range: tuple[float, float], config: TrainingConfig,
                       w_nodes_per_decade: int | None = None,
                       axis_origin: str | None = None
                       ) -> tuple['InteriorWedgeChart', int, int]:
    """Build one positive-parity astroid-interior chart in wedge coordinates.

    The wedge-interior counterpart of `_build_lobe_chart`, for the
    positive-parity (``parity == 1``) astroid interior.  ``box_center`` is
    ``(r_center, theta_wedge_center)`` and ``half`` is
    ``(half_r, half_theta_wedge)``; the chart is trained on the axis-aligned
    wedge-fixed box ``r in [r_center +- half_r]`` x ``theta_wedge in
    [theta_wedge_center +- half_theta_wedge]`` via
    `LensAmplificationSurrogate.from_wedge_engine`, which maps each
    ``(gamma, r, theta_wedge)`` node to a physical eigenframe source through
    the wedge frame (`_from_wedge_fixed`, canonical first quadrant) and stores
    the ``tau_c``-demodulated `INTERIOR_SACR_C` envelope on an
    `InteriorWedgeChart`.  ``from_wedge_engine`` applies the DD-product
    ``w``-ceiling (``w * r * r_caustic <= _DD_PRODUCT_MARGIN``) and builds the
    cusp-adapted angular (``u = d**(2/3)``) ``theta_wedge -> u`` map INTERNALLY
    -- neither is re-derived here.

    Only positive-parity (``parity == 1``) bands have an origin-enclosing
    astroid interior; a macro-saddle call is a programming error (the saddle
    interior is charted per lobe by `_build_lobe_chart`).
    ``w_nodes_per_decade`` overrides the ``w``-axis node density for THIS chart
    only; ``None`` falls back to ``config.w_nodes_per_decade``.  ``axis_origin``
    (``'low'`` / ``'high'`` / ``None``) is the near-cusp side for the chart's
    cusp-adapted angular map; it is single-sourced from the tile (the waist-split
    tiler / subdivider) and threaded UNCHANGED into `from_wedge_engine`, which
    asserts it agrees with its own midpoint-vs-waist classification (guarding
    train/serve skew).  ``None`` lets the engine derive the origin itself.

    Returns
    -------
    tuple[InteriorWedgeChart, int, int]
        The built `InteriorWedgeChart` (unwrapped from the single-chart
        surrogate `from_wedge_engine` returns), the engine node count, and the
        number of refused nodes.

    Raises
    ------
    ValueError
        If ``parity != 1`` (macro-saddle interiors have no astroid wedge).
    CarrierDiscontinuityError
        If the tile straddles a critical-basin flip (caller records the gap).
    """
    if parity != 1:
        raise ValueError(
            'wedge-interior charts exist only for the positive-parity '
            f'(parity == 1) astroid interior; got parity={parity}.')
    nodes_per_decade = (config.w_nodes_per_decade
                        if w_nodes_per_decade is None
                        else int(w_nodes_per_decade))
    n_points = config.n_gamma * config.n_rho * config.n_theta_c
    _budget_check(n_points, config.engine_budget, 'wedge')
    r_c, theta_wedge_c = box_center
    half_r, half_theta = half
    r_range = (r_c - half_r, r_c + half_r)
    theta_wedge_range = (theta_wedge_c - half_theta,
                         theta_wedge_c + half_theta)
    single = LensAmplificationSurrogate.from_wedge_engine(
        gamma_range=gamma_band, r_range=r_range,
        theta_wedge_range=theta_wedge_range, w_range=w_range,
        n_gamma=config.n_gamma, n_r=config.n_rho,
        n_theta_wedge=config.n_theta_c, w_nodes_per_decade=nodes_per_decade,
        definition=INTERIOR_SACR_C, axis_origin=axis_origin)
    chart = single.charts[0]
    refused = int(chart.refused_points.shape[0])
    return chart, n_points, refused


def _engine_envelope(w_grid: np.ndarray, gamma: float, source: np.ndarray
                     ) -> np.ndarray | None:
    """Exact SACR-C envelope ``E(w)`` at a point, or ``None`` if refused.

    A non-finite envelope is treated conservatively as a refusal (F005): the
    surrogate never serves a value the engine could not certify.
    """
    channels = ChangRefsdalChannels(w_grid)
    try:
        partition = channels.evaluate(
            gamma=gamma, y=(float(source[0]), float(source[1])),
            beta=0.0, kappa=0.0)
    except _ENGINE_REFUSALS:
        return None
    env = np.asarray(partition.envelope)
    if not np.all(np.isfinite(env)):
        return None
    return env


# ---------------------------------------------------------------------------
# Held-out accuracy
# ---------------------------------------------------------------------------

def _heldout_eps(chart: TubeChart | ExteriorPolarChart | LobeInteriorChart
                 | InteriorWedgeChart,
                 samples: Sequence[tuple[float, float, float]],
                 provenance: dict) -> float:
    """Max relative envelope error of a chart over held-out geometry points.

    Serves each ``(gamma, y1, y2)`` through the full guard stack of a one-chart
    surrogate and compares to a fresh engine reference; unserved points are
    skipped.  Returns ``nan`` when no held-out point is served.

    The reference envelope and its normalization depend on the chart's
    ENVELOPE LABEL, matching the label each chart is trained on (Build 8g-b):

    - a far-field `ExteriorPolarChart` (a far-field-tag `envelope_definition`) is
      trained on its window-class far-field label, so the reference is
      `farfield_envelope_from_partition` called with the chart's OWN
      ``envelope_definition`` (kernel-sum, diffractive-bottom, or
      kernel-sum-minus-ghost -- never the default), F-normalized by
      ``max|exact_total|`` (``max|E_ff| ~ 1e-4`` is too tiny a denominator);
      a held-out point the ghost gate refuses (kernel-sum-minus-ghost only)
      is skipped, mirroring the training-time gate;
    - an `InteriorWedgeChart` (positive-parity astroid interior in wedge
      caustic-relative coordinates, WP1) is trained on the caustic-region
      ``partition.envelope`` (the ``tau_c``-demodulated SACR-C envelope), so
      its reference is that envelope normalized by ``max|E|`` -- the same
      currency as a tube / lobe-interior chart;
    - a `TubeChart` (and a `LobeInteriorChart`) keeps the caustic-region
      ``partition.envelope`` reference normalized by ``max|E|``.
    """
    surrogate = LensAmplificationSurrogate([chart], provenance)
    w_grid = np.exp(chart.log_w_grid)
    # Only a far-field `ExteriorPolarChart` uses the far-field reference /
    # normalization; every other chart type (tube, lobe-interior, and the
    # WP1 wedge-interior) uses the caustic-region ``partition.envelope``
    # (``max|E|`` currency).  After WP1 no ExteriorPolarChart carries the
    # interior SACR-C label, so a bare isinstance check suffices.
    is_farfield_label = isinstance(chart, ExteriorPolarChart)
    errors: list[float] = []
    for gamma, y1, y2 in samples:
        channels = ChangRefsdalChannels(w_grid)
        try:
            partition = channels.evaluate(
                gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
        except _ENGINE_REFUSALS:
            continue
        if is_farfield_label:
            # Build the held-out reference with the chart's OWN window-class
            # tag (Build S1-2), never the FARFIELD_KERNEL_SUM default: a
            # diffractive-bottom / kernel-sum-minus-ghost chart must be probed
            # against the label it was trained on, else the LOO count is set
            # against the wrong-F reference.
            try:
                env_true = farfield_envelope_from_partition(
                    partition, chart.envelope_definition)
            except geometry.GhostDomainError:
                # FARFIELD_KERNEL_SUM_MINUS_GHOST only: the ghost gate
                # (w_min * Im tau_c >= 2, off the principal axes) refused this
                # point.  Drop it from the LOO accumulation exactly as the
                # training-time gate would, never propagate or substitute.
                continue
            denom = float(np.max(np.abs(partition.exact_total))) or 1.0
        else:
            env_true = np.asarray(partition.envelope)
            denom = float(np.max(np.abs(env_true))) or 1.0
        if not np.all(np.isfinite(env_true)):
            continue
        emulated, served, _definition = surrogate.serve(
            w_grid, gamma=gamma, y1=y1, y2=y2, beta=0.0,
            eta=partition.caustic_distance, theta=partition.critical_theta,
            image_count=int(partition.real_mask.sum()))
        if not served:
            continue
        errors.append(float(np.max(np.abs(emulated - env_true)) / denom))
    return max(errors) if errors else float('nan')


def _reprovision_w_nodes(*, band: tuple[float, float], parity: int,
                         tile: dict, window: tuple[float, float],
                         config: TrainingConfig, rng: np.random.Generator
                         ) -> tuple[int, dict]:
    """Minimal ``w``-node density that still clears the far-field eps bar.

    Per-window adaptive node reprovision on the windowed remainder (Build
    S1-3).  With the exterior far-field ``w`` axis now confined to a fixed
    ``[w_floor, w_trust]`` window (a single detector-band-ish decade rather
    than a whole mass stratum), the surrogate's ``w`` spline needs FEWER nodes
    to resolve the smoothed envelope; this routine finds how many.

    Starting from the full ``config.w_nodes_per_decade`` it retrains the ONE
    probe tile on the windowed remainder at descending densities, recomputing
    the SAME leave-one-out held-out F-normalized eps (`_heldout_eps`, no new
    metric) against the ``config.farfield_eps_max`` (``1e-3``) bar, and returns
    the MINIMAL density ``N_rec`` still clearing the bar -- confirmed minimal
    by ``eps(N_rec) <= bar`` while ``eps(N_rec - 1) > bar`` (the probe decides;
    the caller never guesses the count).

    Only the ``w`` axis is reprovisioned: ``config.n_rho`` /
    ``config.n_theta_c`` (the spatial tiling density) are HELD fixed --
    windowing smooths the ``w`` dependence, not the ``(rho, theta_c)``
    dependence, so the same held-out sample set is reused at every density
    (differences in eps come only from the ``w`` node count, a controlled
    probe).

    Parameters
    ----------
    band : tuple[float, float]
        The band's ``(gamma_lo, gamma_hi)``.
    parity : int
        ``+1`` astroid / ``-1`` saddle.
    tile : dict
        The probe tile (``center`` = ``(rho_c, theta_c)``,
        ``half`` = ``(half_rho, half_theta_c)``); the innermost admitted
        exterior tile (largest ``w_floor``, hardest fit).
    window : tuple[float, float]
        The fixed ``(w_floor, w_trust)`` region window the tile is trained on.
    config : TrainingConfig
        Supplies ``w_nodes_per_decade`` (the descent start) and
        ``farfield_eps_max`` (the bar).
    rng : np.random.Generator
        Draws the shared held-out sample set once (reused at every density).

    Returns
    -------
    tuple[int, dict]
        ``(n_rec, report)``.  ``report`` records the descent trace, the bar,
        the decision (``eps`` at ``N_rec`` and ``N_rec - 1``), and a loud
        status when the bar is never cleared (``'bar_not_cleared'``), the
        engine refused (``'engine_refused'``), or the density floor was reached
        (``'floor_reached'``).  On any non-decision status the FULL
        ``config.w_nodes_per_decade`` is returned (never a guessed reduction).
    """
    center = tile['center']
    half = tile['half']
    bar = float(config.farfield_eps_max)
    n_start = int(config.w_nodes_per_decade)
    # Draw the held-out set ONCE so eps differences across densities reflect
    # the w-node count alone (spatial sampling held constant).
    samples = _farfield_heldout_samples(band, center, half, config, rng)

    trace: list[dict] = []
    eps_at: dict[int, float] = {}

    def _eps_for(n_w: int) -> float | None:
        try:
            chart, _calls, _refused = _build_farfield_chart(
                gamma_band=band, parity=parity, box_center=center, half=half,
                w_range=window, config=config,
                w_nodes_per_decade=n_w)
        except _ENGINE_REFUSALS:
            trace.append({'n_w_per_decade': int(n_w), 'eps': None,
                          'status': 'engine_refused'})
            return None
        except CarrierDiscontinuityError as exc:
            trace.append({'n_w_per_decade': int(n_w), 'eps': None,
                          'status': 'carrier_discontinuity',
                          'detail': str(exc)})
            return None
        eps = _heldout_eps(chart, samples, {'schema': 'heldout-probe'})
        finite = bool(math.isfinite(eps))
        eps_at[int(n_w)] = eps if finite else float('nan')
        trace.append({
            'n_w_per_decade': int(n_w),
            'eps': (None if not finite else round(float(eps), 8)),
            'clears': bool(finite and eps <= bar)})
        return eps if finite else None

    base = {'n_start': n_start, 'bar': bar, 'trace': trace,
            'n_rho_held': int(config.n_rho),
            'n_theta_c_held': int(config.n_theta_c)}

    eps_start = _eps_for(n_start)
    if eps_start is None:
        # No held-out point served / engine refused at full density: cannot
        # probe; keep the full density (loud), never guess a reduction.
        return n_start, {**base, 'status': 'engine_refused', 'n_rec': n_start}
    if eps_start > bar:
        # Even the full density fails the bar; the windowed tile is genuinely
        # hard (the eps gate + subdivision handle it downstream).  Keep full.
        return n_start, {**base, 'status': 'bar_not_cleared',
                         'n_rec': n_start, 'eps_at_n_rec': round(eps_start, 8)}

    # Descend while the bar still clears; stop at the first failing density.
    n_rec = n_start
    for n_w in range(n_start - 1, 0, -1):
        eps = _eps_for(n_w)
        if eps is None or eps > bar:
            # n_w fails (or engine refused): N_rec is the last clearing count,
            # and eps(N_rec - 1) > bar confirms minimality.
            return n_rec, {
                **base, 'status': 'ok', 'n_rec': int(n_rec),
                'eps_at_n_rec': round(float(eps_at[n_rec]), 8),
                'eps_at_n_rec_minus_1': (None if eps is None
                                         else round(float(eps), 8)),
                'decision_confirmed': bool(eps is not None and eps > bar)}
        n_rec = n_w

    # Reached the density floor (n_w = 1 still clears); cannot reduce further.
    return n_rec, {**base, 'status': 'floor_reached', 'n_rec': int(n_rec),
                   'eps_at_n_rec': round(float(eps_at[n_rec]), 8),
                   'decision_confirmed': False}


def _chart_gated(kind: str, eps: float, config: TrainingConfig
                 ) -> tuple[bool, str | None]:
    """Decide whether a chart's held-out eps disqualifies it from registration.

    A chart is *gated* -- excluded from the packed artifact and recorded in the
    report -- when its max-normalized held-out envelope error is NaN (zero
    held-out points served, e.g. an all-refused far-field chart) or exceeds the
    per-kind bar.  Passing charts are registered unchanged (no serve-time
    behavior change).

    Parameters
    ----------
    kind : {'tube', 'farfield', 'interior'}
        Which registration bar to apply.
    eps : float
        The chart's max-normalized held-out envelope error (may be NaN).
    config : TrainingConfig
        Supplies ``tube_eps_max``, ``farfield_eps_max`` and
        ``interior_eps_max``.

    Returns
    -------
    tuple[bool, str | None]
        ``(gated, reason)`` where ``reason`` is ``'nan_eps'``,
        ``'eps_above_bar'``, or ``None`` when the chart passes.

    Raises
    ------
    ValueError
        If ``kind`` is not 'tube', 'farfield' or 'interior'.
    """
    bars = {'tube': config.tube_eps_max,
            'farfield': config.farfield_eps_max,
            'interior': config.interior_eps_max}
    if kind not in bars:
        raise ValueError(
            f"kind must be 'tube', 'farfield' or 'interior'; got {kind!r}.")
    if math.isnan(eps):
        return True, 'nan_eps'
    if eps > bars[kind]:
        return True, 'eps_above_bar'
    return False, None


def _gate_chart(kind: str, report: dict, config: TrainingConfig
                ) -> tuple[bool, str | None]:
    """Decide whether a fresh-or-resumed chart is gated from registration.

    Thin wrapper around `_chart_gated` that additionally honors the
    ``legacy_no_eps`` marker `_load_or_build` sets when a resumed chart's
    provenance predates the ``heldout_eps`` key (pre-8g trainer).  Such
    charts are passed through un-gated rather than being gated on a
    manufactured NaN eps, so a mixed-version resume never silently drops a
    previously-registered chart.

    Parameters
    ----------
    kind : {'tube', 'farfield', 'interior'}
        Which registration bar to apply.
    report : dict
        The per-chart report returned by `_load_or_build`.
    config : TrainingConfig
        Supplies ``tube_eps_max``, ``farfield_eps_max`` and
        ``interior_eps_max``.

    Returns
    -------
    tuple[bool, str | None]
        ``(gated, reason)``, see `_chart_gated`.
    """
    if report.get('legacy_no_eps'):
        return False, None
    eps = float(report.get('heldout_eps', float('nan')))
    return _chart_gated(kind, eps, config)


def _tube_heldout_samples(gamma_band: tuple[float, float], arc: FoldArc,
                          config: TrainingConfig, rng: np.random.Generator,
                          eta_max: float, eta_floor: float
                          ) -> list[tuple[float, float, float]]:
    """Random served-interior held-out sources for a tube chart."""
    samples: list[tuple[float, float, float]] = []
    for _ in range(config.n_heldout):
        gamma = float(rng.uniform(*gamma_band))
        eta = float(rng.uniform(eta_floor, eta_max))
        theta = float(rng.uniform(arc.theta_lo, arc.theta_hi))
        source = _tube_source(gamma, theta, eta, arc.branch, arc.inward_sign)
        samples.append((gamma, float(source[0]), float(source[1])))
    return samples


def _farfield_heldout_samples(gamma_band: tuple[float, float],
                              box_center: tuple[float, float],
                              half: tuple[float, float],
                              config: TrainingConfig,
                              rng: np.random.Generator
                              ) -> list[tuple[float, float, float]]:
    """Random held-out sources inside a candidate tile's proposal box.

    Production still proposes exterior regions in caustic-fixed
    ``(rho, theta_c)`` tile coordinates. It maps each draw to a PHYSICAL
    eigenframe source ``(y1, y2)`` through `_from_caustic_fixed`; the full
    far-field serve guard then maps that source into the chart's current
    gamma-resolved ``(s, d)`` spline coordinates. The returned
    ``(gamma, y1, y2)`` points therefore validate the placement-to-serve
    bridge, not a retired chart-axis round trip.
    """
    rho_c, theta_c = box_center
    half_rho, half_theta = half
    samples: list[tuple[float, float, float]] = []
    for _ in range(config.n_heldout):
        gamma = float(rng.uniform(*gamma_band))
        rho = float(rng.uniform(rho_c - half_rho, rho_c + half_rho))
        theta = float(rng.uniform(theta_c - half_theta, theta_c + half_theta))
        y1_eig, y2_eig = _from_caustic_fixed(gamma, rho, theta)
        samples.append((gamma, float(y1_eig), float(y2_eig)))
    return samples


def _lobe_heldout_samples(gamma_band: tuple[float, float],
                          box_center: tuple[float, float],
                          half: tuple[float, float],
                          config: TrainingConfig,
                          rng: np.random.Generator, *,
                          lobe: '_SaddleLobeAdmission'
                          ) -> list[tuple[float, float, float]]:
    """Random held-out sources inside a lobe chart's lobe-local box.

    The lobe-interior counterpart of `_farfield_heldout_samples`.  Draws
    ``(gamma, rho_lobe, theta_local)`` uniformly inside the chart's lobe-local
    box and maps each draw to a PHYSICAL eigenframe source ``(y1, y2)`` through
    the lobe frame (`_from_lobe_fixed` with the passed ``lobe``'s ``centroid``,
    ``boundary_theta``, ``boundary_r``), matching the per-``gamma`` mapping the
    lobe trainer applied at each grid node.  Using the origin-centred
    `_from_caustic_fixed` here would silently place the probe at the wrong
    physical source, so the lobe-local forward map is used instead; the
    returned ``(gamma, y1, y2)`` points are what `_heldout_eps` serves through
    the full guard stack.
    """
    rho_lobe_c, theta_local_c = box_center
    half_rho, half_theta = half
    centroid = np.ascontiguousarray(lobe.centroid, dtype=float).reshape(2)
    samples: list[tuple[float, float, float]] = []
    for _ in range(config.n_heldout):
        gamma = float(rng.uniform(*gamma_band))
        rho_lobe = float(rng.uniform(rho_lobe_c - half_rho,
                                     rho_lobe_c + half_rho))
        theta_local = float(rng.uniform(theta_local_c - half_theta,
                                        theta_local_c + half_theta))
        y1_eig, y2_eig = _from_lobe_fixed(
            centroid, lobe.boundary_theta, lobe.boundary_r, rho_lobe,
            theta_local)
        samples.append((gamma, float(y1_eig), float(y2_eig)))
    return samples


# ---------------------------------------------------------------------------
# Per-chart resumability
# ---------------------------------------------------------------------------

def _load_or_build(path: Path, build_fn: Callable[[], tuple],
                   provenance: dict
                   ) -> tuple[TubeChart | ExteriorPolarChart | LobeInteriorChart
                              | InteriorWedgeChart, dict, bool]:
    """Load a per-chart file if present, else build it and save it.

    Returns ``(chart, chart_report, reused)``.  Resumability is a plain file
    existence check -- no within-chart progress manifest.  The chart's
    ``heldout_eps`` is persisted into the saved per-chart provenance and read
    back on reuse, so the registration gate fires identically on freshly-built
    and resumed charts WITHOUT recomputing eps (kept deterministic) --
    *except* for charts written by a pre-8g trainer, whose provenance has no
    ``heldout_eps`` key.  Treating that absence as eps=NaN would silently gate
    out (drop) a previously-registered chart across a mixed-version resume,
    so the returned report instead carries a loud ``legacy_no_eps: True``
    marker; callers (via ``_gate_chart``) pass such charts through un-gated
    rather than gating them on a manufactured NaN.
    """
    if path.exists():
        try:
            loaded = LensAmplificationSurrogate.load(path)
        except ValueError:
            # Stale per-chart artifact with an incompatible axis schema
            # (e.g. old (s,d) far-field-exterior chart that predates the
            # polar ExteriorPolarChart).  Delete it and rebuild rather
            # than returning a silently-wrong serve.
            path.unlink()
        else:
            # Surface the persisted held-out eps so the registration gate is
            # applied consistently on reuse (do NOT recompute -- deterministic).
            # Pre-8g charts predate the heldout_eps provenance key; flag that
            # explicitly rather than silently defaulting eps to NaN downstream.
            report: dict = {}
            if 'heldout_eps' in loaded.provenance:
                report['heldout_eps'] = loaded.provenance['heldout_eps']
            else:
                report['legacy_no_eps'] = True
            return loaded.charts[0], report, True
    start = time.perf_counter()
    chart, calls, refused, report_extra = build_fn()
    report = {'engine_calls': int(calls), 'refused_points': int(refused),
              'build_seconds': round(time.perf_counter() - start, 3),
              **report_extra}
    # Persist the held-out eps into the per-chart provenance so a later resume
    # can gate the reused chart without re-running the engine.
    chart_provenance = dict(provenance)
    if 'heldout_eps' in report:
        chart_provenance['heldout_eps'] = report['heldout_eps']
    LensAmplificationSurrogate([chart], chart_provenance).save(path)
    return chart, report, False


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def _artifact_size_bytes(path: Path) -> int:
    """Size in bytes of the packed artifact (``.npz`` appended by numpy)."""
    if path.exists():
        return int(path.stat().st_size)
    appended = path.with_name(path.name + '.npz')
    return int(appended.stat().st_size) if appended.exists() else 0


def _build_provenance(box: PriorBox, config: TrainingConfig,
                      charts: Sequence,
                      dropped_gamma_slivers: Sequence[Sequence[float]] = ()
                      ) -> dict:
    """Provenance dict packed into the artifact (box, config, train hash).

    Parameters
    ----------
    dropped_gamma_slivers : sequence of (lo, hi), optional
        Metamorphosis gamma slivers dropped by `stable_gamma_bands` during
        training, flattened across parities as ``[[lo, hi], ...]``.  Consumed
        by `cogwheel.lensing.surrogate_census` to attribute fall-through
        samples to the ``dropped-sliver`` bucket by default.
    """
    hasher = hashlib.sha1()
    for chart in charts:
        hasher.update(np.ascontiguousarray(chart.real_coeffs).tobytes())
        hasher.update(np.ascontiguousarray(chart.imag_coeffs).tobytes())
    chart_types = ['tube' if isinstance(c, TubeChart) else 'farfield'
                   for c in charts]
    return {
        'schema': 'build8c-multichart',
        'prior_box': {
            'gamma_range': list(box.gamma_range),
            'ln_m_lens_range': list(box.ln_m_lens_range),
            'u1_range': list(box.u1_range),
            'u2_range': list(box.u2_range),
            'f_lo_hz': box.f_lo_hz, 'f_hi_hz': box.f_hi_hz},
        'config': asdict(config),
        'beta': 0.0, 'kappa': 0.0,
        'chart_count': len(charts),
        'chart_types': chart_types,
        'dropped_gamma_slivers': [list(s) for s in dropped_gamma_slivers],
        'training_hash': hasher.hexdigest()[:12]}


#: Fast-tier wall-clock budget [s] for one in-build heavy operation.
#: Far above a fast test / probe (seconds to a couple of minutes) and far
#: below a production sweep (tens of minutes to hours), mirroring the
#: conftest fast-tier per-test ceiling rationale (F061).
_FAST_TIER_BUDGET_S = 900.0

#: Environment names that mark an opt-in slow tier (same set the conftest
#: uses).  Any set means long work is expected and the judge does not fire.
_SLOW_TIER_ENV_VARS = (
    "COGWHEEL_BRUTE_ACCURACY",
    "COGWHEEL_TRAIN_TIER",
    "COGWHEEL_STRICT_TIMING",
    "COGWHEEL_RUN_TIMING_SMOKE",
)


def guard_slow_operation(
    est_seconds: float,
    *,
    what: str,
    budget_s: float = _FAST_TIER_BUDGET_S,
) -> None:
    """Deterministic admission judge for a potentially slow operation.

    The CALLER supplies the honest wall-clock estimate -- the agent knows
    what it is about to run, and tests can be arbitrarily imaginative, so a
    fixed per-function cost model cannot cover them.  This judge is the
    single programmatic gate: it refuses the run when the caller's estimate
    exceeds the fast-tier budget AND no slow tier is enabled.  No prompt
    level instructs an agent to "be fast" -- the judge enforces it, so a
    caller that intends a multi-hour sweep in a build gets a loud refusal
    instead of silently running.

    Call before any heavy invocation with your best runtime estimate:

        guard_slow_operation(est_seconds=2400, what='engine sweep')

    The judge is context-aware: slow tiers are pinned OFF inside builds
    (SDK agents.py), so it refuses there; the driver's post-build sweeps
    enable the tiers and pass through.
    """
    if est_seconds <= budget_s:
        return
    if any(os.environ.get(v) for v in _SLOW_TIER_ENV_VARS):
        return
    raise ValueError(
        f"{what}: estimated {est_seconds/60:.0f} min exceeds the in-build "
        f"fast-tier budget ({budget_s/60:.0f} min).  Slow tiers are pinned "
        f"OFF inside builds; run this as a post-build driver step "
        f"(.claude/sdk/post_build_sweeps.sh) or set a slow-tier env var."
    )


def _self_estimate(
    config: "TrainingConfig",
    regions: tuple[str, ...] | None,
) -> float:
    """Conservative wall-clock proxy [s] for a ``train()`` call.

    Not a tuned cost model -- the general judge accepts an agent-supplied
    estimate for arbitrary operations (see `guard_slow_operation`).  This is
    only train()'s own self-defense: a rough upper bound from the grid it is
    about to build, so a production-scale call is refused in-build even if
    the caller forgot to gate it.  A smoke/probe config stays under the
    budget; a production config (many w nodes, full region set) exceeds it.
    """
    regions = regions or ("tube", "exterior", "wedge_interior", "lobe_interior")
    w_nodes = int(config.w_nodes_per_decade * 2.0)
    # Per-region engine-eval count at the config's grid.  A single-region
    # probe pays only that region's grid, not the full 4-D union.
    per_region = {
        "tube": config.n_theta * config.n_u,
        "exterior": config.n_rho * config.n_theta_c,
        "wedge_interior": 1,
        "lobe_interior": 1,
    }
    n_evals = sum(per_region[r] for r in regions) * config.n_gamma * w_nodes
    # Tiling/subdivision expands the nominal grid; be conservative.
    return n_evals * 8 * 0.09
def train(*, outdir: str | Path,
          artifact_path: str | Path | None = None,
          config: TrainingConfig | None = None,
          f_lo_hz: float = DEFAULT_F_LO_HZ,
          f_hi_hz: float = DEFAULT_F_HI_HZ,
          report_path: str | Path | None = None,
          ppgo_map: CertifiedPpgoMap | None = None,
          regions: tuple[str, ...] | None = None,
          m_lens_range: tuple[float, float] | None = None
          ) -> tuple[LensAmplificationSurrogate, dict]:
    """Build the multi-chart surrogate artifact from the prior box.

    Reads the prior box from the lens prior classes, detects the caustic
    structure per parity from the geometry engine, builds tube + far-field
    charts (resumable per-chart), packs them into a single ``.npz`` artifact
    via `LensAmplificationSurrogate.save`, and emits a JSON training report.

    Parameters
    ----------
    outdir : str or Path
        Directory for per-chart files and (default) the packed artifact.
    artifact_path : str or Path, optional
        Packed-artifact destination (default ``outdir/<default name>``).
    config : TrainingConfig, optional
        Grid sizing and budgets (default smoke scale).
    f_lo_hz, f_hi_hz : float, optional
        Detector frequency band bounds (Hz) for the ``w`` band.
    report_path : str or Path, optional
        If given, the JSON report is written here.
    regions : tuple[str, ...], optional
        Restrict training to the given chart regions
        (``tube`` / ``exterior`` / ``wedge_interior`` / ``lobe_interior``);
        ``None`` trains every region.
    m_lens_range : (float, float), optional
        Restrict the lens-mass range to ``(m_lo, m_hi)`` Msun instead of the
        full prior, so a per-region probe trains a single mass/w stratum via
        the production path rather than reimplementing it.

    Returns
    -------
    LensAmplificationSurrogate
        The packed surrogate.
    dict
        The training report (also written to ``report_path`` if given).
    """
    box = PriorBox.from_prior_classes(f_lo_hz=f_lo_hz, f_hi_hz=f_hi_hz,
                                      m_lens_range=m_lens_range)
    config = config or TrainingConfig()
    # Slow-operation admission judge (programmatic, not prompt-level).
    # train() self-reports a conservative estimate from its own grid so a
    # production-scale call is refused in-build even if no agent thinks to
    # call `guard_slow_operation` first.  The general judge is exported for
    # ANY heavy operation an agent runs, with the agent's own estimate.
    guard_slow_operation(
        est_seconds=_self_estimate(config, regions),
        what="train()",
    )
    # The certified-ppGO map trims mass strata that ppGO already serves.  Fall
    # back to the process-global map (opt-in switch); absent -> None -> no trim
    # and interior/exterior tiling proceeds under the ceiling caps unchanged.
    if ppgo_map is None:
        ppgo_map = get_certified_ppgo_map()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    start = time.perf_counter()

    charts: list = []
    chart_reports: list[dict] = []
    parity_reports: dict[str, dict] = {}
    all_dropped_slivers: list[list[float]] = []

    for parity in (1, -1):
        label = 'astroid' if parity == 1 else 'saddle'
        band = _gamma_band(box, parity, config.gamma_band_halfwidth)
        w_range = box.w_range(parity)
        # The fold-arc partition can change at discrete gammas (cusp/wall
        # metamorphoses on the deltoid); tube grids are rectangular, so the
        # band is bisected into topology-stable sub-bands.  With min_gamma_band=0.0,
        # bisection continues to float resolution and no slivers are dropped; any
        # topology-straddling band resolves once narrow enough for all three sample
        # gammas to agree.
        sub_bands, dropped = stable_gamma_bands(
            band, parity, n_samples=config.n_caustic_samples,
            min_width=config.min_gamma_band,
            refine_near_one_window=config.gamma_refine_near_one_window,
            refine_near_one_width=config.gamma_refine_near_one_width)
        all_dropped_slivers.extend([float(lo), float(hi)]
                                    for lo, hi in dropped)
        parity_reports[label] = {
            'gamma_band': list(band), 'w_range': list(w_range),
            'n_stable_sub_bands': len(sub_bands),
            'dropped_gamma_slivers': [list(s) for s in dropped],
            'sub_bands': [{
                'gamma_range': list(sub),
                'detected_cusps': structure.detected_cusps,
                'cusp_thetas': [round(t, 5)
                                for t in structure.cusp_thetas],
                'caustic_reach': round(structure.caustic_reach, 5),
                'n_fold_arcs': len(structure.arcs)}
                for sub, structure in sub_bands]}

        for i_band, (sub, structure) in enumerate(sub_bands):
            _train_band_charts(
                box=box, config=config, rng=rng, outdir=outdir,
                parity=parity, label=f'{label}_b{i_band}', band=sub,
                structure=structure, charts=charts,
                chart_reports=chart_reports, ppgo_map=ppgo_map,
                regions=regions)

    provenance = _build_provenance(box, config, charts, all_dropped_slivers)
    surrogate = LensAmplificationSurrogate(charts, provenance)
    artifact_path = Path(artifact_path) if artifact_path is not None else (
        outdir / 'lens_amplification_surrogate.npz')
    surrogate.save(artifact_path)

    # Per-parity + total counts of gated (unregistered) charts by reason, so
    # the driver census can see coverage holes the eps gate opened.
    def _gated_counts(reports: list[dict]) -> dict:
        counts = {'total': 0, 'nan_eps': 0, 'eps_above_bar': 0}
        for chart_report in reports:
            counts['total'] += 1
            reason = chart_report.get('gate_reason')
            if reason in counts:
                counts[reason] += 1
        return counts

    gated_reports = [r for r in chart_reports if r.get('gated')]
    gated_charts = {
        'total': _gated_counts(gated_reports),
        'astroid': _gated_counts(
            [r for r in gated_reports if r['parity'] == 1]),
        'saddle': _gated_counts(
            [r for r in gated_reports if r['parity'] == -1])}

    report = {
        'prior_box': provenance['prior_box'],
        'config': provenance['config'],
        'parities': parity_reports,
        'charts': chart_reports,
        'gated_charts': gated_charts,
        'artifact': {
            'path': str(artifact_path),
            'size_bytes': _artifact_size_bytes(artifact_path),
            'n_charts': len(charts),
            'training_hash': provenance['training_hash']},
        'total_seconds': round(time.perf_counter() - start, 3)}
    if report_path is not None:
        Path(report_path).write_text(json.dumps(report, indent=2))
    return surrogate, report


#: Bounded corrective-subdivision depth for `_subdivide_tile`.
#:
#: The Professor's measured astroid-interior case (band 0, ``gamma_mid =
#: 0.495``) needs TWO halvings to clear the ``5e-2`` interior bar: the single
#: shipping halving leaves three marginal children at ``1.19-1.34x`` the bar,
#: and one further level clears them.  ``depth == 1`` is that first halving
#: (the historic single-level behaviour), ``depth == 2`` is the level the
#: measurement shows is required, and ``depth == 3`` is one safety level for a
#: tile still marginal at depth 2.  Depth 4 is deliberately excluded: it spends
#: 4x the leaf count chasing a per-halving error-decay law that has ALREADY
#: broken down on exactly the caustic-straddle tiles that would reach it, so
#: the extra level buys resolution where the smooth-decay premise no longer
#: holds.
MAX_SUBDIVISION_DEPTH: int = 3


def _farfield_child_boxes(
        tile: dict) -> tuple[list[tuple[tuple[float, float],
                                        tuple[float, float]]], float, float]:
    """Compute the up-to-four far-field child boxes and the shared child half.

    Halves the caustic-fixed ``(rho, theta_c)`` box: four children at
    ``(rho_c +- half_rho/2, theta_c +- half_theta/2)`` in a fixed row-major
    order (``s_rho`` outer, ``s_theta`` inner), each carrying the SAME half
    ``(half_rho/2, half_theta/2)`` (a quarter box).

    Returns ``(boxes, child_half_rho, child_half_theta)`` where ``boxes`` is a
    list of ``(center, half)`` tuples.  Single-sourced so the wrapper's return
    dict (``child_half``) and the generic subdivider's splitter agree exactly.
    """
    rho_c, theta_c = tile['center']
    half_rho, half_theta = tile['half']
    child_half_rho = 0.5 * float(half_rho)
    child_half_theta = 0.5 * float(half_theta)
    boxes: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for s_rho in (-1.0, 1.0):
        for s_theta in (-1.0, 1.0):
            child_rho = float(rho_c) + s_rho * child_half_rho
            child_theta = float(theta_c) + s_theta * child_half_theta
            boxes.append(((child_rho, child_theta),
                          (child_half_rho, child_half_theta)))
    return boxes, child_half_rho, child_half_theta


def _wedge_child_boxes(
        tile: dict) -> tuple[list[tuple[tuple[float, float],
                                        tuple[float, float]]], float, float]:
    """Compute the up-to-four wedge child boxes, the u-midpoint theta split,
    and the shared radial child half.

    Radial split at the plain ``r`` midpoint; angular split at the
    ``u``-MIDPOINT mapped back to ``theta_wedge`` on the parent's own
    cusp-adapted map (`_wedge_cusp_axis_map`, the SAME map `from_wedge_engine`
    fits and serves) -- NEVER the ``theta`` midpoint.  Four children in a fixed
    row-major order (radial sub-row outer, angular sub-column inner); the two
    angular children have UNEQUAL ``theta`` widths (the near-cusp child is
    narrower) and share the radial half.

    Returns ``(boxes, theta_split, child_half_r)`` where ``boxes`` is a list of
    ``(center, half)`` tuples.  Single-sourced so the wrapper's return dict
    (``theta_split``, ``child_half_r``) and the generic subdivider's splitter
    agree exactly.
    """
    r_c, theta_wedge_c = tile['center']
    half_r, half_theta = tile['half']
    axis_origin = tile['axis_origin']
    theta_lo = float(theta_wedge_c) - float(half_theta)
    theta_hi = float(theta_wedge_c) + float(half_theta)
    r_lo = float(r_c) - float(half_r)
    r_hi = float(r_c) + float(half_r)
    child_half_r = 0.5 * float(half_r)

    # Angular split at the u-midpoint mapped back to theta on the parent's own
    # cusp-adapted map (u_fine[0] == 0 by construction), NOT the theta midpoint.
    theta_fine, u_fine = _wedge_cusp_axis_map(theta_lo, theta_hi, axis_origin)
    u_mid = 0.5 * (float(u_fine[0]) + float(u_fine[-1]))
    theta_split = float(np.interp(u_mid, u_fine, theta_fine))

    r_children = ((r_lo, 0.5 * (r_lo + r_hi)),
                  (0.5 * (r_lo + r_hi), r_hi))
    theta_children = ((theta_lo, theta_split), (theta_split, theta_hi))
    boxes: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for child_r_lo, child_r_hi in r_children:
        for child_theta_lo, child_theta_hi in theta_children:
            child_r_c = 0.5 * (child_r_lo + child_r_hi)
            child_theta_c = 0.5 * (child_theta_lo + child_theta_hi)
            child_half_theta = 0.5 * (child_theta_hi - child_theta_lo)
            boxes.append(((float(child_r_c), float(child_theta_c)),
                          (float(child_half_r), float(child_half_theta))))
    return boxes, theta_split, child_half_r


def _subdivide_tile(
        *, tile: dict, parent_tag: str, band: tuple[float, float],
        parity: int, config: TrainingConfig, rng: np.random.Generator,
        outdir: Path, charts: list, chart_reports: list[dict],
        split_children: Callable[[dict], list], build_child: Callable[..., tuple],
        gate_kind: str, eps_bar: float,
        admit_child: Callable[[tuple, tuple], bool] | None = None,
        max_depth: int = MAX_SUBDIVISION_DEPTH, depth: int = 1) -> dict:
    """Generic bounded-recursion tile subdivider shared by both regions.

    Extracts the single skeleton that `_subdivide_farfield_tile` and
    `_subdivide_wedge_tile` duplicated (iterate candidate children,
    `_load_or_build` each, `_gate_chart` it, pack-or-record, accumulate a
    ``children_summary``) and adds the ONE piece both lacked: bounded recursion.
    A child that STILL fails the eps bar is itself subdivided -- until it clears
    or the halving chain reaches ``max_depth`` (`MAX_SUBDIVISION_DEPTH`), at
    which point it is recorded as a ladder-served gap exactly as before.  The
    achieved subdivision depth is reported per child and rolled up into
    ``max_achieved_depth`` so a runaway is visible and the census can attribute
    cleared-vs-still-gated windows.

    The two regions differ ONLY in parameters, never in control flow:

    - ``split_children(tile)`` returns the up-to-four ``(center, half)`` child
      boxes (far-field halves caustic-fixed ``(rho, theta_c)``; the wedge
      halves ``(r, u)`` with the angular split at the u-midpoint);
    - ``build_child(center, half, tile)`` calls the region's chart builder plus
      held-out probe and returns ``(chart, calls, refused, report)`` for
      `_load_or_build`;
    - ``admit_child(center, half)`` is the far-field re-admission predicate
      (`_InteriorAdmission.admits_exterior` or the scalar ``exclusion_rho``
      floor); ``None`` for the wedge, whose every child is a sub-box of an
      already-admitted interior tile and is always built.

    The report STYLE is keyed on whether an admission predicate is supplied: a
    subdivider WITHOUT one is the wedge-interior style (no per-child
    ``'admission'`` key; gated/flip children carry the explicit
    ``'subdivided'`` / ``'ladder_served_gap'`` markers and a carrier-flip child
    gets its own ``chart_reports`` entry).  A subdivider WITH one is the
    far-field style (per-child ``'admission'``; a carrier-flip child recorded
    only in the summary).  A future lobe subdivider supplies its own admission
    predicate and inherits the far-field style -- the unification does not
    preclude wiring it, which is explicitly out of scope here.

    The `CarrierDiscontinuityError` branch is preserved exactly: a child that
    straddles a critical-basin (``tau_c``) flip is recorded as a carrier-flip
    gap and is NEVER recursed -- halving cannot fix a phase discontinuity.

    Parameters
    ----------
    tile : dict
        The gated parent tile record (``center``, ``half``, ``region``,
        ``w_range``, ``si``, ``m_lo``, ``m_hi``, and -- for the wedge --
        ``axis_origin``; ``w_nodes_per_decade`` optional).
    parent_tag : str
        The parent chart's tag; children are named ``{parent_tag}_c{ci}`` (so a
        recursed grandchild is ``{parent_tag}_c{ci}_c{cj}``).
    band, parity, config, rng, outdir
        Threaded through unchanged from `_train_band_charts`.
    charts, chart_reports : list
        In-place accumulators for packed charts and per-chart reports.
    split_children, build_child, admit_child
        Region parameters described above.
    gate_kind : str
        ``_gate_chart`` kind (``'farfield'`` or ``'interior'``).
    eps_bar : float
        The registration bar recorded in each child summary entry.
    max_depth : int
        Recursion cap (`MAX_SUBDIVISION_DEPTH`).
    depth : int
        Current 1-based subdivision depth; the initial call is depth 1.

    Returns
    -------
    dict
        ``{'children': [...], 'packed': int, 'max_achieved_depth': int}``.
        ``packed`` counts every packed chart in the FULL recursion subtree so
        the wedge call site's ``ladder_served_gap`` reflects grandchildren too.
    """
    region = tile['region']
    # A subdivider without an admission predicate is the wedge-interior style;
    # with one it is the far-field style (see docstring).
    wedge_style = admit_child is None

    children_summary: list[dict] = []
    total_packed = 0
    max_achieved = depth
    ci = 0
    for center, half in split_children(tile):
        entry_center = [round(float(center[0]), 6), round(float(center[1]), 6)]
        entry_half = [round(float(half[0]), 6), round(float(half[1]), 6)]

        if admit_child is not None and not admit_child(center, half):
            # The parent's edge straddles the caustic/shell boundary; a
            # disk-excluded child is correct geometry, dropped silently
            # (recorded here for the census, packed nowhere).
            children_summary.append({
                'ci': ci, 'center': entry_center, 'half': entry_half,
                'admission': 'disk_excluded', 'result': 'disk_excluded',
                'achieved_depth': depth})
            ci += 1
            continue

        child_tag = f'{parent_tag}_c{ci}'
        child_path = outdir / f'{child_tag}.npz'

        try:
            chart, report, reused = _load_or_build(
                child_path,
                lambda center=center, half=half: build_child(
                    center, half, tile),
                {'schema': 'build8c-chart', 'parity': parity})
        except CarrierDiscontinuityError as exc:
            # A subdivided child STILL straddles a basin flip: record as a
            # carrier-flip gap served by the ladder, NEVER recursed.
            entry = {'ci': ci, 'center': entry_center, 'half': entry_half}
            if wedge_style:
                chart_reports.append({
                    'name': child_tag, 'parity': parity,
                    'file': str(child_path), 'region': region,
                    'subdivided_from': parent_tag, 'carrier_flip': True,
                    'carrier_flip_detail': str(exc),
                    'subdivided': False, 'ladder_served_gap': True})
                entry['result'] = 'carrier_flip'
            else:
                entry['admission'] = 'admitted'
                entry['result'] = 'carrier_flip'
                entry['carrier_flip_detail'] = str(exc)
            entry['achieved_depth'] = depth
            children_summary.append(entry)
            ci += 1
            continue

        gated, gate_reason = _gate_chart(gate_kind, report, config)
        child_eps = float(report.get('heldout_eps', float('nan')))
        base_report = {'name': child_tag, 'parity': parity,
                       'file': str(child_path), 'reused': reused,
                       'subdivided_from': parent_tag, **report}

        entry = {'ci': ci, 'center': entry_center, 'half': entry_half}
        if not wedge_style:
            entry['admission'] = 'admitted'

        if not gated:
            charts.append(chart)
            chart_reports.append(base_report)
            total_packed += 1
            entry['eps'] = (None if math.isnan(child_eps)
                            else round(child_eps, 8))
            entry['bar'] = eps_bar
            entry['gate_reason'] = gate_reason
            entry['result'] = 'packed'
            entry['achieved_depth'] = depth
            children_summary.append(entry)
            ci += 1
            continue

        if depth < max_depth:
            # Still gated with recursion budget left: record the gated child
            # chart, then subdivide IT one level deeper on the same bar.
            gated_report = {**base_report, 'gated': True,
                            'gate_reason': gate_reason, 'subdivided': True}
            chart_reports.append(gated_report)
            child_tile = {
                'center': tuple(center), 'half': tuple(half),
                'axis_origin': tile.get('axis_origin'),
                'w_range': tile['w_range'], 'region': region,
                'si': tile['si'], 'm_lo': tile['m_lo'], 'm_hi': tile['m_hi'],
                'w_nodes_per_decade': tile.get('w_nodes_per_decade')}
            sub = _subdivide_tile(
                tile=child_tile, parent_tag=child_tag, band=band,
                parity=parity, config=config, rng=rng, outdir=outdir,
                charts=charts, chart_reports=chart_reports,
                split_children=split_children, build_child=build_child,
                gate_kind=gate_kind, eps_bar=eps_bar, admit_child=admit_child,
                max_depth=max_depth, depth=depth + 1)
            total_packed += sub['packed']
            max_achieved = max(max_achieved, sub['max_achieved_depth'])
            gated_report['subdivision'] = sub
            if wedge_style:
                gated_report['ladder_served_gap'] = sub['packed'] == 0
            entry['eps'] = (None if math.isnan(child_eps)
                            else round(child_eps, 8))
            entry['bar'] = eps_bar
            entry['gate_reason'] = gate_reason
            entry['result'] = 'subdivided'
            entry['achieved_depth'] = sub['max_achieved_depth']
            children_summary.append(entry)
            ci += 1
            continue

        # Gated at the recursion cap: terminal ladder-served gap (the historic
        # single-level outcome).
        gated_report = {**base_report, 'gated': True,
                        'gate_reason': gate_reason}
        if wedge_style:
            gated_report['subdivided'] = False
            gated_report['ladder_served_gap'] = True
        chart_reports.append(gated_report)
        entry['eps'] = (None if math.isnan(child_eps)
                        else round(child_eps, 8))
        entry['bar'] = eps_bar
        entry['gate_reason'] = gate_reason
        entry['result'] = 'recorded_gated'
        entry['achieved_depth'] = depth
        children_summary.append(entry)
        ci += 1

    return {'children': children_summary, 'packed': total_packed,
            'max_achieved_depth': max_achieved}


def _subdivide_farfield_tile(
        *, tile: dict, parent_tag: str, band: tuple[float, float],
        parity: int, config: TrainingConfig, rng: np.random.Generator,
        outdir: Path, exclusion_rho: float,
        interior_admission: '_InteriorAdmission | None',
        charts: list, chart_reports: list[dict],
        exterior_admission: '_InteriorAdmission | None' = None,
        source_magnitude_max: float | None = None) -> dict:
    """Halve one eps-gated far-field tile into up to four children (Build 8h-a
    WP4), now a thin wrapper over the shared `_subdivide_tile` skeleton.

    Caustic-fixed ``(rho, theta_c)`` corrective subdivision.  A far-field tile
    whose held-out eps failed the ``farfield_eps_max`` bar is split into up to
    four quarter boxes at ``(rho_c +- half_rho/2, theta_c +- half_theta/2)``
    (row-major ``s_rho`` outer, ``s_theta`` inner).  Each child is re-admitted
    through the PARENT's OWN region predicate (carried verbatim in
    ``tile['region']``): a positive-parity exterior parent re-runs the SAME
    per-``theta_c``-column directional `admits_exterior` test (``exterior_
    admission`` with ``source_magnitude_max``); a macro-saddle exterior parent
    the scalar-reach floor ``rho_c_child - half_rho/2 >= exclusion_rho``.  A
    disk-excluded child is DROPPED silently (correct geometry, not a training
    failure).  Admitted children inherit the parent's ppGO-trimmed ``w_range``
    verbatim, retrain via `_build_farfield_chart`, and re-gate via
    `_gate_chart`.

    The bounded recursion now lives once in `_subdivide_tile`: a still-gated
    child is itself halved until it clears or the chain reaches
    `MAX_SUBDIVISION_DEPTH`, at which point it is a ladder-served gap exactly as
    the historic single-level behavior recorded it.  A far-field tile whose
    children ALL pass at the first halving triggers no recursion and produces a
    depth-1 report byte-identical to the pre-refactor output plus the additive
    ``achieved_depth`` / ``max_achieved_depth`` fields.

    Parameters mirror the pre-refactor signature exactly (call sites unchanged).
    ``interior_admission`` remains vestigial since WP1 (interior tiles are no
    longer subdivided) and is never dereferenced.  ``exterior_admission`` /
    ``source_magnitude_max`` select the directional exterior re-admission path;
    ``None`` (default) keeps the byte-identical scalar-reach floor.

    Returns
    -------
    dict
        ``{'parent_tag', 'region', 'child_half', 'children'}`` -- every
        pre-refactor key -- plus the additive ``'max_achieved_depth'``.
    """
    _, child_half_rho, child_half_theta = _farfield_child_boxes(tile)
    region = tile['region']

    def split_children(subtile: dict) -> list:
        return _farfield_child_boxes(subtile)[0]

    def admit_child(center: tuple, half: tuple) -> bool:
        # Re-admit through the PARENT's region predicate (Professor guard (e)):
        # positive-parity exterior children re-run the SAME per-column
        # directional `admits_exterior`; macro-saddle exterior children the
        # scalar-rho exclusion floor.  Always supplied, so the shared skeleton
        # uses the far-field report style.
        child_rho, child_theta = center
        child_half_r, _ = half
        if (exterior_admission is not None
                and source_magnitude_max is not None):
            return exterior_admission.admits_exterior(
                (child_rho, child_theta), half, source_magnitude_max)
        return child_rho - child_half_r >= exclusion_rho

    def build_child(center: tuple, half: tuple, subtile: dict) -> tuple:
        # Children inherit the parent's reprovisioned w-node density verbatim
        # (raw ``w_nodes`` into `_build_farfield_chart`); the resolved 3-way
        # ``eff_w_nodes`` is only reported in ``node_counts`` (mirrors the main
        # tiler; interior parents fall back to interior density, INS-2-001).
        w_nodes = subtile.get('w_nodes_per_decade')
        region_t = subtile['region']
        if w_nodes is not None:
            eff_w_nodes = int(w_nodes)
        elif region_t in ('interior', 'lobe_interior'):
            eff_w_nodes = config.interior_w_nodes_per_decade
        else:
            eff_w_nodes = config.w_nodes_per_decade
        chart, calls, refused = _build_farfield_chart(
            gamma_band=band, parity=parity, box_center=center, half=half,
            w_range=subtile['w_range'], config=config,
            w_nodes_per_decade=w_nodes)
        samples = _farfield_heldout_samples(band, center, half, config, rng)
        eps = _heldout_eps(chart, samples, {'schema': 'heldout-probe'})
        return chart, calls, refused, {
            'kind': 'farfield', 'region': region_t,
            'image_count': chart.image_count,
            'stratum_index': subtile['si'],
            'stratum_mass_range': [round(subtile['m_lo'], 3),
                                   round(subtile['m_hi'], 3)],
            'rho_theta_box': [list(center), list(half)],
            'w_range': [round(subtile['w_range'][0], 6),
                        round(subtile['w_range'][1], 6)],
            'node_counts': {'n_gamma': config.n_gamma,
                            'n_rho': config.n_rho,
                            'n_theta_c': config.n_theta_c,
                            'n_w_per_decade': int(eff_w_nodes)},
            'heldout_eps': eps}

    summary = _subdivide_tile(
        tile=tile, parent_tag=parent_tag, band=band, parity=parity,
        config=config, rng=rng, outdir=outdir, charts=charts,
        chart_reports=chart_reports, split_children=split_children,
        build_child=build_child, gate_kind='farfield',
        eps_bar=config.farfield_eps_max, admit_child=admit_child)

    return {'parent_tag': parent_tag, 'region': region,
            'child_half': [round(child_half_rho, 6),
                           round(child_half_theta, 6)],
            'children': summary['children'],
            'max_achieved_depth': summary['max_achieved_depth']}


def _subdivide_wedge_tile(
        *, tile: dict, parent_tag: str, band: tuple[float, float],
        parity: int, config: TrainingConfig, rng: np.random.Generator,
        outdir: Path, charts: list, chart_reports: list[dict]) -> dict:
    """Halve one eps-gated wedge-interior tile into up to four children (WP2),
    now a thin wrapper over the shared `_subdivide_tile` skeleton.

    The astroid-interior counterpart of `_subdivide_farfield_tile`, in the
    WEDGE chart's own caustic-relative ``(r, u)`` coordinates.  The radial split
    is at the plain ``r`` midpoint; the ANGULAR split is at the ``u``-MIDPOINT
    mapped back to ``theta_wedge`` on the parent's own cusp-adapted map
    (`_wedge_cusp_axis_map`, the SAME map `from_wedge_engine` fits and serves)
    -- NEVER the ``theta`` midpoint -- so the two angular children have UNEQUAL
    ``theta`` widths (the near-cusp child narrower), which is the point of the
    ``u`` axis.  Each child inherits the parent's ``axis_origin`` verbatim,
    rebuilds via `_build_wedge_chart`, and re-gates on ``config.interior_eps_max``
    via `_gate_chart('interior', ...)`.  There is NO admission predicate: every
    child is a sub-box of an already-admitted interior tile and is always built,
    which is what selects the wedge-interior report style in the shared
    skeleton (``admit_child=None``).

    The bounded recursion now lives once in `_subdivide_tile`: a still-gated
    child is itself halved (u-midpoint angular split preserved at every level)
    until it clears or the chain reaches `MAX_SUBDIVISION_DEPTH`, then recorded
    as a ladder-served gap.  ``packed`` counts the FULL recursion subtree, so
    the call site's ``ladder_served_gap = subdivision['packed'] == 0`` reflects
    grandchildren too.  A carrier-flip (`CarrierDiscontinuityError`) child is
    recorded as a ladder-served gap and NEVER recursed (halving cannot fix a
    phase discontinuity).

    Parameters mirror the pre-refactor signature exactly (call site unchanged).

    Returns
    -------
    dict
        ``{'parent_tag', 'region', 'axis_origin', 'theta_split',
        'child_half_r', 'packed', 'children'}`` -- every pre-refactor key --
        plus the additive ``'max_achieved_depth'``.
    """
    _, theta_split, child_half_r = _wedge_child_boxes(tile)
    axis_origin = tile['axis_origin']
    region = tile['region']

    def split_children(subtile: dict) -> list:
        return _wedge_child_boxes(subtile)[0]

    def build_child(center: tuple, half: tuple, subtile: dict) -> tuple:
        # Interior children inherit the interior w-node density (3-way: tile
        # override -> config.interior_w_nodes_per_decade -> config.w_nodes_per_
        # decade), mirroring the main tiler (INS-2-001).  axis_origin threads
        # into from_wedge_engine, whose DD-product w-ceiling caps the band.
        w_nodes = subtile.get('w_nodes_per_decade')
        region_t = subtile['region']
        if w_nodes is not None:
            eff_w_nodes = int(w_nodes)
        elif region_t in ('interior', 'lobe_interior', 'wedge_interior'):
            eff_w_nodes = config.interior_w_nodes_per_decade
        else:
            eff_w_nodes = config.w_nodes_per_decade
        chart, calls, refused = _build_wedge_chart(
            gamma_band=band, parity=parity, box_center=center, half=half,
            w_range=subtile['w_range'], config=config,
            w_nodes_per_decade=eff_w_nodes,
            axis_origin=subtile['axis_origin'])
        # Held-out probe INLINE (mirrors the main wedge branch): draw
        # (gamma, r, theta_wedge) uniformly inside the child's wedge-fixed box
        # and map each draw to a PHYSICAL eigenframe source through the child
        # chart's OWN wedge_map.
        r_cen, theta_cen = center
        half_r_c, half_theta_c = half
        samples: list[tuple[float, float, float]] = []
        for _ in range(config.n_heldout):
            gamma = float(rng.uniform(*band))
            r = float(rng.uniform(r_cen - half_r_c, r_cen + half_r_c))
            theta_wedge = float(rng.uniform(
                theta_cen - half_theta_c, theta_cen + half_theta_c))
            y1_eig, y2_eig = _from_wedge_fixed(
                gamma, r, theta_wedge, chart.wedge_map)
            samples.append((gamma, float(y1_eig), float(y2_eig)))
        eps = _heldout_eps(chart, samples, {'schema': 'heldout-probe'})
        return chart, calls, refused, {
            'kind': 'interior', 'region': region_t,
            'image_count': chart.image_count,
            'stratum_index': subtile['si'],
            'stratum_mass_range': [round(subtile['m_lo'], 3),
                                   round(subtile['m_hi'], 3)],
            'rho_theta_box': [list(center), list(half)],
            'w_range': [round(subtile['w_range'][0], 6),
                        round(subtile['w_range'][1], 6)],
            'node_counts': {'n_gamma': config.n_gamma,
                            'n_rho': config.n_rho,
                            'n_theta_c': config.n_theta_c,
                            'n_w_per_decade': int(eff_w_nodes)},
            'heldout_eps': eps}

    summary = _subdivide_tile(
        tile=tile, parent_tag=parent_tag, band=band, parity=parity,
        config=config, rng=rng, outdir=outdir, charts=charts,
        chart_reports=chart_reports, split_children=split_children,
        build_child=build_child, gate_kind='interior',
        eps_bar=config.interior_eps_max, admit_child=None)

    return {'parent_tag': parent_tag, 'region': region,
            'axis_origin': axis_origin,
            'theta_split': round(theta_split, 6),
            'child_half_r': round(child_half_r, 6),
            'packed': summary['packed'], 'children': summary['children'],
            'max_achieved_depth': summary['max_achieved_depth']}


def _train_band_charts(*, box: 'PriorBox', config: TrainingConfig,
                       rng: np.random.Generator, outdir: Path, parity: int,
                       label: str, band: tuple[float, float],
                       structure: CausticStructure, charts: list,
                       chart_reports: list[dict],
                       ppgo_map: CertifiedPpgoMap | None = None,
                       regions: tuple[str, ...] | None = None) -> None:
    """Build the tube + far-field charts of one topology-stable gamma band."""
    if regions is None:
        regions = ('tube', 'exterior', 'wedge_interior', 'lobe_interior')
    gamma_grid = _log_reach_gamma_axis(band, config.n_gamma, f'gamma_{label}')

    # -- Tube charts (per fold arc, resumable) --
    # Pre-compute per-arc minimum curvature radii (worst over gamma band).
    # The absolute tube band [eta_floor, eta_max] is f * R_c per arc.
    arc_r_min = [_min_curvature_radius(band, arc, config.n_caustic_samples)
                 for arc in structure.arcs[:config.max_tube_arcs]]
    max_eta_max = (config.f_max * max(arc_r_min)
                   if arc_r_min else config.f_max * 0.05)
    # Cap the tube w grid by the largest source magnitude it samples
    # (caustic reach plus the outer eta wall), so w * |y| stays below the
    # engine's double-double ceiling -- mirroring the prior's mass coupling.
    if 'tube' in regions:
        tube_w_range = _capped_w_range(
            box, parity, structure.caustic_reach + max_eta_max)
    else:
        tube_w_range = (0.0, 0.0)
    for idx, arc in enumerate(structure.arcs[:config.max_tube_arcs]
                               if 'tube' in regions else ()):
        # Per-arc curvature-relative tube shell sizing.
        r_min = arc_r_min[idx]
        assert config.f_max < 0.5, (
            f'f_max={config.f_max} must be < 0.5 (foot-of-normal)')
        eta_max = config.f_max * r_min
        eta_floor = config.f_floor * r_min
        assert eta_max >= 1e-3, (
            f'eta_max={eta_max} too small (R_c={r_min})')
        tag = f'chart_{label}_tube_{idx}'
        path = outdir / f'{tag}.npz'

        def build_tube(arc=arc, band=band, gamma_grid=gamma_grid,
                       w_range=tube_w_range, eta_max=eta_max,
                       eta_floor=eta_floor):
            chart, calls, refused = _build_tube_chart(
                gamma_grid=gamma_grid, arc=arc, parity=parity,
                w_range=w_range, config=config,
                eta_max=eta_max, eta_floor=eta_floor)
            samples = _tube_heldout_samples(band, arc, config, rng,
                                            eta_max=eta_max,
                                            eta_floor=eta_floor)
            eps = _heldout_eps(chart, samples,
                               {'schema': 'heldout-probe'})
            # Single-gamma-map adequacy diagnostic (NOT a gate, per the
            # Professor's caveat): the arc-length map is built at one
            # representative gamma, so record how much the NORMALIZED profile
            # s/s_total drifts between the band's gamma endpoints.  A small
            # value means one map suffices for the whole band; a large value
            # flags that a gamma-dependent map may be needed.
            _, s_lo = _tube_arc_length_map(float(gamma_grid[0]), arc)
            _, s_hi = _tube_arc_length_map(float(gamma_grid[-1]), arc)
            s_map_dev = float(np.max(np.abs(
                s_hi / s_hi[-1] - s_lo / s_lo[-1])))
            return chart, calls, refused, {
                'kind': 'tube', 'branch': arc.branch,
                'image_count': arc.image_count,
                'theta_range': [round(arc.theta_lo, 5),
                                round(arc.theta_hi, 5)],
                'node_counts': {
                    'n_w': len(_log_w_grid(
                        w_range, config.w_nodes_per_decade)),
                    'n_gamma': config.n_gamma, 'n_u': config.n_u,
                    'n_theta': config.n_theta},
                'heldout_eps': eps,
                's_map_gamma_endpoint_dev': s_map_dev}

        chart, report, reused = _load_or_build(
            path, build_tube, {'schema': 'build8c-chart', 'parity': parity})
        gated, gate_reason = _gate_chart('tube', report, config)
        chart_report = {'name': tag, 'parity': parity, 'file': str(path),
                        'reused': reused, **report}
        if gated:
            chart_report['gated'] = True
            chart_report['gate_reason'] = gate_reason
            chart_reports.append(chart_report)
            continue
        charts.append(chart)
        chart_reports.append(chart_report)

    # -- Far-field charts (resumable) --
    # Build 8g WP2 partitioned the parity's REACHABLE mass range into log
    # strata so each stratum's whole ``[w(20, m), w(1024, m)]`` band fit one
    # chart w range.  Build S1-3 RETIRES that outer partitioning for the
    # EXTERIOR: it is now tiled ONCE under a single fixed ``[w_floor,
    # w_trust]`` region window (see below).  The mass strata REMAIN for the
    # INTERIOR ``w`` ranges (and the high-mass ``beyond_w_cap`` record), so
    # ``_mass_strata`` is still called.
    strata, beyond = _mass_strata(box, parity)
    if beyond is not None:
        # Mass above the parity w-ceiling cannot satisfy whole-band containment
        # today (saddle Schwinger wall ~458 Msun); record it loudly, never drop
        # it silently.  Build 8h moves the wall.
        chart_reports.append({
            'name': f'chart_{label}_farfield_beyond_w_cap',
            'parity': parity, 'beyond_w_cap': True,
            'mass_range': [round(beyond['m_lo'], 3), round(beyond['m_hi'], 3)],
            'w_ceiling': round(beyond['ceiling'], 3),
            'reason': 'lens mass above the parity w-ceiling is not tileable'})

    # Far-field tiles live in caustic-fixed ``(rho, theta_c)`` coordinates.
    # Positive-parity exterior rho is one plus the physical radial offset from
    # the directional caustic. Subtracting the band's MINIMUM caustic radius
    # from the physical exclusion disk yields a rho floor safe for every chart
    # node. The ppGO map remains scalar-reach based and receives its own
    # rho coordinate below. Macro saddles retain scalar-reach rho.
    gamma_mid = 0.5 * (band[0] + band[1])
    reach_scalar = _scalar_caustic_reach(gamma_mid)
    coordinate_radius_min, reach_max = _coordinate_radius_bounds(band, parity)
    physical_exclusion_radius = reach_max + max_eta_max
    # Additive scalar/directional caustic-fixed inner edge for BOTH parities:
    # ``rho = 1 + |y| - coordinate_radius_min`` is the exact inverse of the
    # serve map's exterior arm (`_from_caustic_fixed`), giving ``rho = 1`` at
    # the caustic and ``drho/d|y| = 1``.  The parity difference lives entirely
    # in ``coordinate_radius_min`` (per-angle minimum critical-curve radius for
    # positive parity; band-minimum scalar ``_caustic_reach`` for macro
    # saddles, whose disconnected deltoids miss most origin-centred rays).
    exclusion_rho = 1.0 + physical_exclusion_radius - coordinate_radius_min
    admitted: list[dict] = []
    dropped_strata: list[dict] = []
    if strata:
        m_lo_region, m_hi_region = strata[0][0], strata[-1][1]
    else:
        m_lo_region, m_hi_region = box.m_lens_range
    # Union spatial extent: the largest physical source magnitude mapped with
    # the band's minimum directional radius covers every gamma and theta.
    y_outer_region = float(_lens_prior._source_scale(m_lo_region))
    # Additive outer edge, same convention as ``exclusion_rho`` above and the
    # serve map: mutual inverse of `_from_caustic_fixed` for both parities.
    rho_outer_region = 1.0 + y_outer_region - coordinate_radius_min
    # Exterior admission (WP1).  Positive parity: per-``theta_c``-column
    # directional admission (`_farfield_exterior_tiles` via `admits_exterior`)
    # replaces the single scalar ``exclusion_rho``.  Each column's inner rho
    # edge is admitted on the TRUE per-gamma nearest-caustic distance, so the
    # ``gamma >= 0.85`` prior box (swallowed whole by the cusp-spike scalar
    # reach) is covered again.  The window / w-floor machinery is fed the
    # MINIMUM admitted per-column rho_inner (the column closest to the caustic,
    # hence the largest local w_floor), so the ppGO / physics-floor
    # certification stays conservative.  Macro saddles keep the scalar
    # ``_farfield_tiles(exclusion_rho, ...)`` path unchanged (parity != 1).
    exterior_admission = None
    if 'exterior' in regions:
        exterior_tiles: list | None = None
        if parity == 1:
            exterior_admission = _interior_admission(
                band, 1, reach_scalar, config, eta_max=max_eta_max)
            # Cusp-align the exterior ``theta_c`` columns to the SAME source-plane
            # astroid cusp rays as the interior (WP1 defect 1): the exterior
            # ``rho > 1`` arm is a theta-independent affine push-out of
            # ``r_caustic``, so it inherits the interior's slope kinks.  Reuse the
            # band's ``gamma_mid`` (already computed above); cusp rays are
            # gamma-dependent, so never recompute at a band edge.
            cusp_angles = _cusp_source_angles(gamma_mid, config.n_caustic_samples)
            exterior_tiles = _farfield_exterior_tiles(
                rho_outer_region, config.n_farfield_tiles_per_side,
                admission=exterior_admission,
                source_magnitude_max=y_outer_region,
                cusp_angles=cusp_angles, gamma=gamma_mid)
            region_exclusion_rho = (
                min(center[0] - half[0] for center, half, _, _ in exterior_tiles)
                if exterior_tiles else exclusion_rho)
        else:
            region_exclusion_rho = exclusion_rho
        # caustic-relative inner edge (WP1 defect 1).  Derive it from the NARROWED
        # served region ``region_exclusion_rho`` -- NOT the pre-narrowing outer
        # rho-band -- so the certified-ppGO trim below reads ``w_trust`` /
        # ``w_ceiling`` from the rho-band cell the region actually covers.  The
        # positive-parity per-column admission (above) can pull the served inner
        # edge closer to the caustic than the scalar exclusion disk; reading the
        # farther-out cell would report an easier (lower-``w_cert``) certification
        # than the inner columns actually enjoy, capping/dropping charts where ppGO
        # is not in fact certified.  ``caustic_rho`` is the ONE authoritative
        # converter into the scalar-reach ppGO gauge; feed it the physical ``|y|``
        # recovered by inverting the additive exterior gauge
        # (``rho = 1 + |y| - coordinate_radius_min`` => ``|y| = rho - 1 +
        # coordinate_radius_min``).  Deriving from the narrowed region gives a
        # smaller/closer inner edge and hence a not-easier ppGO cell --
        # conservatism
        # compounds, never over-accepts.  Macro saddles (parity != 1) keep the HEAD
        # scalar-reach edge: their additive exterior gauge collapses to scalar
        # reach, so ``region_exclusion_rho == exclusion_rho`` and the physical
        # exclusion radius is already the authoritative inner edge.  Both branches
        # nonetheless obtain ``rho`` through ``caustic_rho`` so the ppGO gauge
        # lives
        # in exactly ONE place (``_scalar_caustic_reach == caustic_geometry(gamma,
        # 0)[0]`` bit-exact, so the saddle result is byte-identical to the former
        # hand-rolled ``physical_exclusion_radius / reach_scalar``).
        if parity == 1:
            ppgo_exclusion_rho = caustic_rho(
                gamma_mid,
                region_exclusion_rho - 1.0 + coordinate_radius_min,
                kappa=0.0)
        else:
            ppgo_exclusion_rho = caustic_rho(
                gamma_mid, physical_exclusion_radius, kappa=0.0)
        # -- Exterior far-field: ONE fixed [w_floor, w_trust] window (S1-3) --
        # Build S1-3 replaces the per-mass-stratum ``w`` partitioning of the
        # exterior with a SINGLE fixed window ``[w_floor(region),
        # w_trust(region)]`` that contains every in-region draw's chart w-segment
        # by construction (band-split serving is live).  ``w_floor`` is the S1-2
        # physics threshold (`_farfield_region_w_floor`); ``w_trust`` is the
        # ppGO-trimmed top (`_farfield_region_window`).  The geometry-admitted tile
        # loop (`_farfield_tiles`) is UNCHANGED -- tiles are admitted by geometry,
        # never by mass -- but the whole exterior region is now tiled ONCE over
        # the union source extent (largest ``|y|`` at the smallest reachable lens
        # mass), not once per stratum.  The certified-ppGO trim uses
        # ``ppgo_exclusion_rho`` derived (above) from the region's OWN served inner
        # edge, so the drop certifies the rho-band the region actually covers and
        # never over-clears.
        ext_boundary = _stratum_ppgo_boundary(
            parity, gamma_mid, ppgo_exclusion_rho, ppgo_map)
        ext_ceiling = _stratum_ppgo_ceiling(
            parity, gamma_mid, ppgo_exclusion_rho, ppgo_map)
        exterior_region_report: dict = {
            'name': f'chart_{label}_farfield_region',
            'parity': parity, 'exterior_region_summary': True,
            'exclusion_rho': round(float(exclusion_rho), 6),
            'region_exclusion_rho': round(float(region_exclusion_rho), 6),
            'ppgo_exclusion_rho': round(float(ppgo_exclusion_rho), 6),
            'coordinate_radius_min': round(float(coordinate_radius_min), 6),
            'reach_scalar': round(float(reach_scalar), 6),
            'reach_max': round(float(reach_max), 6),
            'rho_outer': round(float(rho_outer_region), 6),
            'mass_range': [round(float(m_lo_region), 3),
                           round(float(m_hi_region), 3)],
            'n_rho': config.n_rho, 'n_theta_c': config.n_theta_c}
        if parity == 1 and not exterior_tiles:
            # Loud zero-admission (WP1): no ``theta_c`` column clears the caustic +
            # tube shell inside the prior box, so no exterior chart is built and
            # those draws fall to the tube / interior / serving ladder.  (Restoring
            # the scalar test makes the gamma 0.80-0.90 band collapse HERE -- the
            # coverage defect this WP repairs.)
            exterior_region_report.update(
                {'window': None, 'admitted_tiles': 0,
                 'window_action': 'zero_admission',
                 'zero_admission': True,
                 'zero_admission_reason': 'no_exterior_column_admits_tile'})
        else:
            window, window_action, window_report = _farfield_region_window(
                box, parity, band, region_exclusion_rho, rho_outer_region,
                reach_max, ext_boundary, ext_ceiling, config,
                source_magnitude_max=y_outer_region)
            exterior_region_report['window_action'] = window_action
            if window is None:
                # No exterior chart: 'drop' (ppGO serves the whole band) or 'empty'
                # (degenerate w_floor >= w_trust window).  Loud record; those draws
                # fall to the tube / interior / serving ladder.
                exterior_region_report.update(
                    {'window': None, 'admitted_tiles': 0, **window_report})
            else:
                w_floor, w_trust = window
                # Containment RANGE CHECK (S1-3): every in-region draw's chart
                # w-segment must lie within [w_floor, w_trust] to 1e-12 -- a
                # self-consistency invariant replacing the strata whole-band
                # containment bookkeeping.
                contained, containment_report = _farfield_window_contains_draws(
                    box, window)
                if parity == 1:
                    # Per-column admitted set (`_farfield_exterior_tiles`); every
                    # tile's inner edge already clears the caustic + tube shell and
                    # lies inside the prior box for its direction.
                    tiles = exterior_tiles
                else:
                    # Macro saddle: unchanged scalar-reach exterior tiler.
                    tiles = _farfield_tiles(
                        exclusion_rho, rho_outer_region,
                        config.n_farfield_tiles_per_side)
                # Per-window node reprovision (w-axis ONLY): probe the innermost
                # tile (largest w_floor, hardest fit) for the minimal w-node
                # density N_rec still clearing the eps bar; the rho/theta_c tiling
                # density is HELD.
                n_rec = int(config.w_nodes_per_decade)
                reprovision_report: dict = {'status': 'no_admitted_tile',
                                            'n_rec': n_rec}
                if tiles:
                    probe_center, probe_half = tiles[0][0], tiles[0][1]
                    probe_tile = {'center': probe_center, 'half': probe_half,
                                  'si': 0, 'i': tiles[0][2], 'j': tiles[0][3],
                                  'm_lo': m_lo_region, 'm_hi': m_hi_region,
                                  'w_range': window, 'region': 'exterior'}
                    n_rec, reprovision_report = _reprovision_w_nodes(
                        band=band, parity=parity, tile=probe_tile, window=window,
                        config=config, rng=rng)
                for center, half, i, j in tiles:
                    admitted.append({
                        'si': 0, 'i': i, 'j': j, 'center': center, 'half': half,
                        'm_lo': m_lo_region, 'm_hi': m_hi_region,
                        'w_range': window, 'w_nodes_per_decade': int(n_rec),
                        'region': 'exterior'})
                exterior_region_report.update({
                    'window': [round(float(w_floor), 6),
                               round(float(w_trust), 6)],
                    'admitted_tiles': len(tiles),
                    'n_w_per_decade': int(n_rec),
                    'containment_ok': bool(contained),
                    'containment': containment_report,
                    'reprovision': reprovision_report, **window_report})
                if not contained:
                    # True by construction of the clip; a violation signals a
                    # window/clip inconsistency -- flagged loudly, not silently.
                    exterior_region_report['containment_violation'] = True
        chart_reports.append(exterior_region_report)
    else:
        exterior_tiles = None
        region_exclusion_rho = exclusion_rho
    # -- Interior (4-image) far-field tiles (frozen WP6, S2-1) --
    # The astroid interior is a single 4-image region enclosing the origin, so
    # an interior tile carries the SAME E_ff / far-field label (the subtraction
    # runs over the morse-sign real_mask, so an interior box subtracts four
    # kernels automatically -- no code change to `_build_farfield_chart`).
    # Admission is DIRECTIONAL (frozen WP6): a tile is interior iff its
    # farthest point has ``rho = |y| / r_caustic(gamma, theta_y) < 1`` for
    # every gamma in the band (`_InteriorAdmission`), replacing the isotropic
    # inscribed disk ``caustic_inradius - eta_max`` that discarded the
    # anisotropic interior between the inradius and the directional radius.
    # The eta_max tube shell is excluded with the NEAREST-caustic distance (not
    # the radial gap -- off-radial near a cusp), and theta_c tile edges align
    # to the four cusp rays so no tile straddles a kink.  The saddle deltoid's
    # lobes do NOT enclose the origin; there is no origin-centred interior
    # disk, so the interior loop records a loud skip and admits nothing (the
    # eps gate is the final safety net; per-lobe saddle interiors are S2-2).
    inradius, encloses = _caustic_inradius(
        gamma_mid, parity, config.n_caustic_samples)
    interior_records: list[dict] = []
    interior_admitted = 0
    interior_skip: str | None = None
    admission: '_InteriorAdmission | None' = None
    cusp_angles: list[float] = []
    lobe_records: list[dict] = []
    if parity != 1:
        if 'lobe_interior' in regions:
            # --- Saddle (gamma > 1): per-lobe deltoid interiors
            # (frozen WP7, S2-2).
            # The macro-saddle caustic is two disjoint 3-cusp deltoid lobes off the
            # origin on the shear axis; neither encloses the origin, so the
            # origin-centred astroid admission does not apply.  Each lobe gets its
            # OWN interior family in a lobe-local frame centred on that lobe's
            # source-plane deltoid centroid, admitted by per-lobe winding number,
            # tube-shell nearest-distance, and the inter-lobe corridor exclusion
            # (`_SaddleLobeAdmission`); the lobe-local theta tile edges align to
            # the lobe's three cusp rays and no tile straddles the inter-lobe
            # equidistance (perpendicular-bisector) line.
            #
            # These lobe-local tiles are packed into ``admitted`` with
            # ``region='lobe_interior'`` and their owning ``_SaddleLobeAdmission``
            # (S2-3 serve wiring): the build loop trains each through
            # ``_build_lobe_chart`` / ``from_lobe_engine`` in lobe-local
            # ``(rho_lobe, theta_local)`` coordinates and the persisted lobe frame
            # (centroid, boundary) maps a served node back to its true physical
            # source, so the lobe interiors are now served (not just recorded).
            # Saddle eta_max for lobe corridor: use the band's max eta_max
            # (widest tube shell among all fold arcs in this band).
            saddle_eta_max = max_eta_max
            lobe_admissions = _saddle_lobe_admissions(band, config,
                                                      eta_max=saddle_eta_max)
            for lobe_index, (lens_center, lobe) in enumerate(
                    zip(_SADDLE_LOBE_CENTERS, lobe_admissions)):
                lobe_cusps = _lobe_cusp_source_angles(
                    gamma_mid, lens_center, lobe.centroid,
                    config.n_caustic_samples)
                lobe_tiles = _lobe_interior_tiles(
                    lobe, lobe_cusps, config.n_farfield_tiles_per_side)
                interior_admitted += len(lobe_tiles)
                # Pack each admitted lobe tile into the served build set carrying
                # its owning ``_SaddleLobeAdmission`` (S2-3 serve wiring): the build
                # loop routes ``region == 'lobe_interior'`` tiles through
                # ``_build_lobe_chart`` / ``from_lobe_engine`` in lobe-local
                # ``(rho_lobe, theta_local)`` coordinates, so the lobe-centroid
                # offset now flows through the serve pipeline.  ``si = lobe_index``
                # disambiguates the two lobes' per-chart tags.
                centroid_mag = float(np.hypot(lobe.centroid[0], lobe.centroid[1]))
                r_deltoid_max = float(np.max(lobe.boundary_r))
                for center, half, i, j in lobe_tiles:
                    rho_lobe_max = float(center[0]) + float(half[0])
                    # Union spatial extent for the frequency cap: the farthest
                    # physical source magnitude reachable inside this lobe-local
                    # tile (centroid offset + outer directional boundary radius).
                    y_max_tile = centroid_mag + rho_lobe_max * r_deltoid_max
                    admitted.append({
                        'si': lobe_index, 'i': i, 'j': j,
                        'center': center, 'half': half,
                        'm_lo': m_lo_region, 'm_hi': m_hi_region,
                        'w_range': _capped_w_range(box, parity, y_max_tile),
                        'region': 'lobe_interior', 'lobe': lobe})
                lobe_records.append({
                    'lens_center': round(float(lens_center), 6),
                    'centroid': [round(float(lobe.centroid[0]), 6),
                                 round(float(lobe.centroid[1]), 6)],
                    'reach': round(float(lobe.reach), 6),
                    'corridor_half': round(float(lobe.corridor_half), 6),
                    'n_cusp_rays': len(lobe_cusps),
                    'cusp_angles': [round(float(a), 6) for a in lobe_cusps],
                    'admitted_tiles': len(lobe_tiles)})
            if interior_admitted == 0:
                interior_skip = 'saddle_lobes_zero_admission'
    elif not encloses:
        interior_skip = 'caustic_not_origin_enclosing'
    elif reach_scalar <= max_eta_max:
        # The eta_max tube shell fills the whole caustic (the reach is within
        # one shell of the origin); no interior point clears the shell.
        interior_skip = 'tube_shell_fills_interior'
    else:
        if 'wedge_interior' in regions:
            # Positive-parity astroid interior in WEDGE caustic-relative
            # coordinates (WP1): the origin-enclosing astroid interior is charted
            # by ``InteriorWedgeChart`` (built via ``from_wedge_engine``) instead
            # of the retired far-field ``ffin`` tiling.  ``r`` is normalised by the
            # directional caustic reach and ``theta_wedge = atan2(|y2|, |y1|)``
            # spans one canonical quadrant ``[0, pi/2]`` (the astroid D2 fold maps
            # the other three quadrants onto it).  The wedge tiler is a MINIMAL
            # single-angular-column, uniform-radial-rows family; DD-product
            # ``w``-capping and the arc-length ``theta_wedge -> s`` map are applied
            # INSIDE ``from_wedge_engine`` per tile, and no cusp-alignment or
            # directional admission geometry is needed -- the caustic-relative
            # frame absorbs the caustic shape.  ``coordinate_radius_min`` /
            # ``reach_max`` are the band-level bounds already computed above
            # (bit-identical to ``np.min(admission.radius_grid)``, so no per-band
            # ``_interior_admission`` object is built here).
            int_rho = 0.0  # near-origin: the hardest interior region (Build 8h-a)
            int_boundary = _stratum_ppgo_boundary(
                parity, gamma_mid, int_rho, ppgo_map)
            int_ceiling = _stratum_ppgo_ceiling(
                parity, gamma_mid, int_rho, ppgo_map)
            for si, (m_lo, m_hi) in enumerate(strata):
                y_extent = float(_lens_prior._source_scale(m_lo))
                # The directional interior reaches the full caustic at ``rho=1``,
                # capped conservatively by the stratum source support divided by
                # the smallest physical directional radius in the band.
                grid_rho_extent = min(
                    1.0, float(y_extent) / coordinate_radius_min,
                )
                grid_extent = grid_rho_extent * reach_max
                int_w_range = _stratum_w_range(
                    box, parity, m_lo, m_hi, grid_extent)
                trimmed_w_range, action = _apply_ppgo_trim(
                    int_w_range, int_boundary, int_ceiling)
                if action == 'drop':
                    dropped_strata.append({
                        'stratum_index': si, 'region': 'wedge_interior',
                        'mass_range': [round(m_lo, 3), round(m_hi, 3)],
                        'w_range': [round(int_w_range[0], 6),
                                    round(int_w_range[1], 6)],
                        'w_trust': round(float(int_boundary), 6),
                        'reason': 'ppGO certified over the whole stratum w-band'})
                    continue
                int_w_range = trimmed_w_range
                # Cap the wedge radial extent one tube-shell inside the caustic so
                # the Airy caustic edge (r -> 1) is left to the tube chart; a
                # non-positive extent yields no tiles (ladder-served interior).
                r_extent = min(
                    grid_rho_extent, 1.0 - max_eta_max / coordinate_radius_min)
                # Locate the caustic waist at the SAME band-representative gamma
                # `from_wedge_engine` uses internally (median of the log-reach
                # gamma grid over this band), so the tiler's angular split boundary
                # and the engine's per-tile near-cusp classification agree exactly
                # (no train/serve skew -- the engine asserts on disagreement).
                gamma_rep = float(np.median(
                    _log_reach_gamma_axis(band, config.n_gamma, 'gamma')))
                tiles = _wedge_interior_tiles(
                    gamma_rep, r_extent, config.n_farfield_tiles_per_side)
                interior_admitted += len(tiles)
                interior_records.append({
                    'stratum_index': si,
                    'mass_range': [round(m_lo, 3), round(m_hi, 3)],
                    'grid_extent': round(float(grid_extent), 6),
                    'grid_rho_extent': round(float(grid_rho_extent), 6),
                    'r_extent': round(float(r_extent), 6),
                    'w_range': [round(int_w_range[0], 6),
                                round(int_w_range[1], 6)],
                    'ppgo_capped': bool(action == 'cap'),
                    'admitted_tiles': len(tiles)})
                for center, half, i, j, axis_origin in tiles:
                    admitted.append({
                        'si': si, 'i': i, 'j': j, 'center': center, 'half': half,
                        'm_lo': m_lo, 'm_hi': m_hi, 'w_range': int_w_range,
                        'region': 'wedge_interior', 'axis_origin': axis_origin})

    # Loud interior summary.  Where geometry permits an interior region (origin
    # enclosed, reach clears the tube shell) admission MUST be non-empty; a
    # zero count there is a coverage defect and is flagged loudly (mirrors the
    # exterior n_per_side admission finding).
    interior_report: dict = {
        'name': f'chart_{label}_farfield_interior',
        'parity': parity, 'interior_summary': True,
        'origin_enclosed': bool(encloses),
        'admission': ('per_lobe_winding' if parity != 1
                      else 'wedge_caustic_relative'),
        'caustic_inradius': round(float(inradius), 6),
        'caustic_reach': round(float(reach_scalar), 6),
        'n_cusp_rays': len(cusp_angles),
        'cusp_angles': [round(float(a), 6) for a in cusp_angles],
        'interior_admitted_tiles': int(interior_admitted),
        'strata': interior_records}
    if parity != 1:
        # Per-lobe saddle interior (S2-3): record the lobe frames + admitted
        # lobe-local tile counts.  The lobe-local tiles are now PACKED into the
        # served build set and trained via ``from_lobe_engine`` in lobe-local
        # ``(rho_lobe, theta_local)`` coordinates; the persisted lobe frame
        # (centroid + directional boundary) carries the lobe-centroid offset
        # through the serve pipeline, so lobe interiors are served.
        interior_report['lobes'] = lobe_records
        interior_report['served'] = interior_admitted > 0
        interior_report['serve_note'] = (
            'lobe-local tiles admitted + cusp-aligned and PACKED into served '
            'charts; trained via from_lobe_engine in (rho_lobe, theta_local) '
            'coordinates with the persisted lobe frame (centroid + directional '
            'boundary) mapping served nodes to their true physical source')
    if interior_skip is not None:
        interior_report['interior_skipped'] = interior_skip
    elif interior_admitted == 0:
        interior_report['interior_zero_admission'] = True
    chart_reports.append(interior_report)

    # Loud mass-strata summary.  The EXTERIOR is no longer mass-stratified
    # (Build S1-3: it uses one fixed [w_floor, w_trust] window, reported in
    # ``chart_{label}_farfield_region``); the mass strata now partition only
    # the INTERIOR ``w`` ranges, so ``strata`` carries the interior per-stratum
    # records and ``dropped_strata`` the interior ppGO drops (the ladder census
    # attributes the cleared budget).
    chart_reports.append({
        'name': f'chart_{label}_farfield_strata',
        'parity': parity, 'strata_summary': True, 'region': 'interior',
        'n_strata': len(strata),
        'exclusion_rho': round(float(exclusion_rho), 6),
        'reach_scalar': round(float(reach_scalar), 6),
        'strata': interior_records,
        'dropped_strata': dropped_strata})

    # ``max_farfield_regions`` is a TRUE cap on distinct admitted tiles; a
    # truncation is recorded loudly with the dropped count.
    if (config.max_farfield_regions is not None
            and len(admitted) > config.max_farfield_regions):
        chart_reports.append({
            'name': f'chart_{label}_farfield_truncated',
            'parity': parity, 'truncated': True,
            'admitted_tiles': len(admitted),
            'cap': config.max_farfield_regions,
            'dropped': len(admitted) - config.max_farfield_regions})
        admitted = admitted[:config.max_farfield_regions]

    for tile in admitted:
        si, i, j = tile['si'], tile['i'], tile['j']
        center, half, w_range = tile['center'], tile['half'], tile['w_range']
        m_lo, m_hi = tile['m_lo'], tile['m_hi']
        region = tile['region']
        # Near-cusp side of the wedge angular map (None for non-wedge tiles);
        # single-sourced from the waist-split tiler, threaded unchanged into
        # `_build_wedge_chart` -> `from_wedge_engine`.
        axis_origin = tile.get('axis_origin')
        # Exterior tiles carry a per-window reprovisioned ``w``-node density
        # (Build S1-3, ``N_rec``); interior tiles have no such key and fall
        # back to ``config.interior_w_nodes_per_decade`` (higher density for
        # the SACR-C envelope oscillations) or ``config.w_nodes_per_decade``
        # for exterior remainder.  Only the ``w`` axis is affected -- the
        # spatial ``(n_gamma, n_rho, n_theta_c)`` density is untouched.
        tile_w_nodes = tile.get('w_nodes_per_decade')
        if tile_w_nodes is not None:
            eff_w_nodes = int(tile_w_nodes)
        elif region in ('interior', 'lobe_interior', 'wedge_interior'):
            eff_w_nodes = config.interior_w_nodes_per_decade
        else:
            eff_w_nodes = config.w_nodes_per_decade
        # Interior (4-image) tiles reuse the identical build/serve path, but
        # store the SACR-C ``tau_c``-demodulated envelope label
        # (`INTERIOR_SACR_C`) instead of the divergent-kernel-subtracting
        # far-field label (Build S2-3): inside the caustic the far-field label
        # subtracts near-merged image kernels that individually blow up, so it
        # is fitted poorly; the SACR-C label switches that pair INTO the
        # bounded envelope.  The tag infix (``ffin`` vs ``ff``) and the
        # registration ``kind`` (interior eps bar vs far-field) also switch.
        if region == 'lobe_interior':
            # Macro-saddle lobe-interior tile (S2-3): trained in lobe-local
            # (rho_lobe, theta_local) coordinates via ``from_lobe_engine`` on
            # the tile's owning ``_SaddleLobeAdmission`` (carried on the tile),
            # storing the INTERIOR_SACR_C tau_c-demodulated envelope on a
            # ``LobeInteriorChart``.  Gated on the SAME interior eps bar as the
            # origin-centred interior path, but the held-out probe maps through
            # the LOBE frame (`_lobe_heldout_samples`), never the origin-centred
            # `_from_caustic_fixed`.  ``si`` disambiguates the two lobes' tags.
            lobe = tile['lobe']
            lobe_tag = f'chart_{label}_s{si}_fflobe_{i}_{j}'
            lobe_path = outdir / f'{lobe_tag}.npz'

            def build_lobe(band=band, center=center, half=half,
                           w_range=w_range, si=si, m_lo=m_lo, m_hi=m_hi,
                           region=region, lobe=lobe, eff_w_nodes=eff_w_nodes,
                           w_nodes=eff_w_nodes):
                chart, calls, refused = _build_lobe_chart(
                    gamma_band=band, parity=parity, lobe=lobe,
                    box_center=center, half=half, w_range=w_range,
                    config=config, w_nodes_per_decade=w_nodes)
                samples = _lobe_heldout_samples(
                    band, center, half, config, rng, lobe=lobe)
                eps = _heldout_eps(chart, samples,
                                   {'schema': 'heldout-probe'})
                return chart, calls, refused, {
                    'kind': 'interior', 'region': region,
                    'image_count': chart.image_count,
                    'stratum_index': si,
                    'stratum_mass_range': [round(m_lo, 3), round(m_hi, 3)],
                    'rho_theta_box': [list(center), list(half)],
                    'w_range': [round(w_range[0], 6), round(w_range[1], 6)],
                    'node_counts': {'n_gamma': config.n_gamma,
                                    'n_rho': config.n_rho,
                                    'n_theta_c': config.n_theta_c,
                                    'n_w_per_decade': int(eff_w_nodes)},
                    'heldout_eps': eps}

            try:
                chart, report, reused = _load_or_build(
                    lobe_path, build_lobe,
                    {'schema': 'build8c-chart', 'parity': parity})
            except CarrierDiscontinuityError as exc:
                # The lobe tile straddles a critical-basin (``tau_c``) flip.
                # The far-field subdivider is origin-centred (scalar
                # ``exclusion_rho`` + ``_from_caustic_fixed`` samples) and
                # CANNOT resubdivide a lobe-local box, so the tile is recorded
                # as a ladder-served gap (not subdivided); lobe-aware
                # subdivision is owed follow-on work.
                chart_reports.append({
                    'name': lobe_tag, 'parity': parity,
                    'file': str(lobe_path), 'region': region,
                    'carrier_flip': True, 'carrier_flip_detail': str(exc),
                    'subdivided': False, 'ladder_served_gap': True})
                continue
            gated, gate_reason = _gate_chart('interior', report, config)
            chart_report = {'name': lobe_tag, 'parity': parity,
                            'file': str(lobe_path), 'reused': reused, **report}
            if gated:
                # A gated lobe tile is a ladder-served gap: the far-field
                # subdivider cannot halve a lobe-local box (see above), so the
                # window is served by the ladder, never numerical quadrature.
                chart_report['gated'] = True
                chart_report['gate_reason'] = gate_reason
                chart_report['subdivided'] = False
                chart_report['ladder_served_gap'] = True
                chart_reports.append(chart_report)
                continue
            charts.append(chart)
            chart_reports.append(chart_report)
            continue

        if region == 'wedge_interior':
            # Positive-parity astroid-interior tile (WP1): trained in WEDGE
            # caustic-relative ``(r, theta_wedge)`` coordinates via
            # ``from_wedge_engine`` (inside ``_build_wedge_chart``), storing the
            # ``INTERIOR_SACR_C`` ``tau_c``-demodulated envelope on an
            # ``InteriorWedgeChart``.  Gated on the SAME interior eps bar as the
            # origin-centred / lobe interior paths; the held-out probe maps
            # through the chart's own wedge frame (``chart.wedge_map`` +
            # ``_from_wedge_fixed``), never the retired ``_from_caustic_fixed``.
            # A gated or carrier-flipped wedge tile is a ladder-served gap (NO
            # subdivision -- the far-field subdivider is origin-centred and
            # cannot resubdivide a caustic-relative box; mirrors the lobe path).
            wedge_tag = f'chart_{label}_s{si}_ffwedge_{i}_{j}'
            wedge_path = outdir / f'{wedge_tag}.npz'

            def build_wedge(band=band, center=center, half=half,
                            w_range=w_range, si=si, m_lo=m_lo, m_hi=m_hi,
                            region=region, axis_origin=axis_origin,
                            eff_w_nodes=eff_w_nodes, w_nodes=eff_w_nodes):
                chart, calls, refused = _build_wedge_chart(
                    gamma_band=band, parity=parity, box_center=center,
                    half=half, w_range=w_range, config=config,
                    w_nodes_per_decade=w_nodes, axis_origin=axis_origin)
                # Held-out probe INLINE (task step D): draw
                # ``(gamma, r, theta_wedge)`` uniformly inside the tile's
                # wedge-fixed box and map each draw to a PHYSICAL eigenframe
                # source through the chart's OWN ``wedge_map`` -- transcribes
                # ``_lobe_heldout_samples`` for the wedge frame.
                r_c, theta_wedge_c = center
                half_r, half_theta = half
                samples: list[tuple[float, float, float]] = []
                for _ in range(config.n_heldout):
                    gamma = float(rng.uniform(*band))
                    r = float(rng.uniform(r_c - half_r, r_c + half_r))
                    theta_wedge = float(rng.uniform(
                        theta_wedge_c - half_theta,
                        theta_wedge_c + half_theta))
                    y1_eig, y2_eig = _from_wedge_fixed(
                        gamma, r, theta_wedge, chart.wedge_map)
                    samples.append((gamma, float(y1_eig), float(y2_eig)))
                eps = _heldout_eps(chart, samples,
                                   {'schema': 'heldout-probe'})
                return chart, calls, refused, {
                    'kind': 'interior', 'region': region,
                    'image_count': chart.image_count,
                    'stratum_index': si,
                    'stratum_mass_range': [round(m_lo, 3), round(m_hi, 3)],
                    'rho_theta_box': [list(center), list(half)],
                    'w_range': [round(w_range[0], 6), round(w_range[1], 6)],
                    'node_counts': {'n_gamma': config.n_gamma,
                                    'n_rho': config.n_rho,
                                    'n_theta_c': config.n_theta_c,
                                    'n_w_per_decade': int(eff_w_nodes)},
                    'heldout_eps': eps}

            try:
                chart, report, reused = _load_or_build(
                    wedge_path, build_wedge,
                    {'schema': 'build8c-chart', 'parity': parity})
            except CarrierDiscontinuityError as exc:
                # The wedge tile straddles a critical-basin (``tau_c``) flip;
                # recorded as a ladder-served gap (not subdivided).
                chart_reports.append({
                    'name': wedge_tag, 'parity': parity,
                    'file': str(wedge_path), 'region': region,
                    'carrier_flip': True, 'carrier_flip_detail': str(exc),
                    'subdivided': False, 'ladder_served_gap': True})
                continue
            gated, gate_reason = _gate_chart('interior', report, config)
            chart_report = {'name': wedge_tag, 'parity': parity,
                            'file': str(wedge_path), 'reused': reused,
                            **report}
            if gated:
                # WP2: a gated wedge tile is halved ONCE (single level, no
                # recursion) in (r, u) into up to four children -- the angular
                # split at the u-MIDPOINT mapped back to theta (equal steps in
                # the cusp-adapted u the spline sees, NEVER the cusp-singular
                # theta), each rebuilt via `_build_wedge_chart` carrying the
                # parent's axis_origin and re-gated on the SAME interior eps
                # bar.  Only if NO child clears the bar does the window fall
                # back to a ladder-served gap.  Restoring this eps feedback
                # loop is the point: a tiler with no feedback cannot discover
                # it needs more tiles (removing it hid the axis-adjacent
                # under-resolution for a day -- see the build brief).
                chart_report['gated'] = True
                chart_report['gate_reason'] = gate_reason
                chart_report['subdivided'] = True
                chart_reports.append(chart_report)
                subdivision = _subdivide_wedge_tile(
                    tile=tile, parent_tag=wedge_tag, band=band, parity=parity,
                    config=config, rng=rng, outdir=outdir,
                    charts=charts, chart_reports=chart_reports)
                chart_report['subdivision'] = subdivision
                chart_report['ladder_served_gap'] = subdivision['packed'] == 0
                continue
            charts.append(chart)
            chart_reports.append(chart_report)
            continue

        # Exterior far-field remainder tile: the ``FARFIELD_KERNEL_SUM``
        # envelope in ``(s, d)`` coordinates.  After WP1 the astroid interior
        # is charted by the wedge branch above, so this final branch is
        # exterior-only (``region == 'exterior'``) -- no ``INTERIOR_SACR_C``
        # far-field chart is ever built.
        kind = 'farfield'
        tag = f'chart_{label}_s{si}_ff_{i}_{j}'
        path = outdir / f'{tag}.npz'

        def build_ff(band=band, center=center, half=half, w_range=w_range,
                     si=si, m_lo=m_lo, m_hi=m_hi, region=region, kind=kind,
                     w_nodes=eff_w_nodes, eff_w_nodes=eff_w_nodes):
            chart, calls, refused = _build_farfield_chart(
                gamma_band=band, parity=parity, box_center=center,
                half=half, w_range=w_range, config=config,
                w_nodes_per_decade=w_nodes)
            samples = _farfield_heldout_samples(
                band, center, half, config, rng)
            eps = _heldout_eps(chart, samples,
                               {'schema': 'heldout-probe'})
            return chart, calls, refused, {
                'kind': kind, 'region': region,
                'image_count': chart.image_count,
                'stratum_index': si,
                'stratum_mass_range': [round(m_lo, 3), round(m_hi, 3)],
                'rho_theta_box': [list(center), list(half)],
                'w_range': [round(w_range[0], 6), round(w_range[1], 6)],
                'node_counts': {'n_gamma': config.n_gamma,
                                'n_rho': config.n_rho,
                                'n_theta_c': config.n_theta_c,
                                'n_w_per_decade': int(eff_w_nodes)},
                'heldout_eps': eps}

        try:
            chart, report, reused = _load_or_build(
                path, build_ff, {'schema': 'build8c-chart', 'parity': parity})
        except CarrierDiscontinuityError as exc:
            # Interpolator hygiene: the exterior far-field tile straddles a
            # critical-basin (``tau_c``) flip, so a single spline cannot
            # represent the phase-kinked envelope.  Resolve by
            # reseat-via-SUBDIVISION -- halve the tile so each sub-tile lands in
            # one nearest-caustic basin -- recorded loudly.  A flip is
            # generically absent for well-separated exterior tiles, so this is
            # the exceptional path.
            flip_report = {'name': tag, 'parity': parity, 'file': str(path),
                           'region': region, 'carrier_flip': True,
                           'carrier_flip_detail': str(exc), 'subdivided': True}
            chart_reports.append(flip_report)
            flip_report['subdivision'] = _subdivide_farfield_tile(
                tile=tile, parent_tag=tag, band=band, parity=parity,
                config=config, rng=rng, outdir=outdir,
                exclusion_rho=exclusion_rho,
                interior_admission=None,
                charts=charts, chart_reports=chart_reports,
                exterior_admission=(exterior_admission if parity == 1
                                    else None),
                source_magnitude_max=(y_outer_region if parity == 1
                                      else None))
            continue
        gated, gate_reason = _gate_chart(kind, report, config)
        chart_report = {'name': tag, 'parity': parity, 'file': str(path),
                        'reused': reused, **report}
        if gated:
            chart_report['gated'] = True
            chart_report['gate_reason'] = gate_reason
            # Build 8h-a WP4: a gated far-field tile is halved once (single
            # level, no recursion) into up to four children, each re-admitted
            # through the parent's own region predicate, retrained on the
            # inherited w_range and re-gated against the same bar.  Passing
            # children are packed; still-failing (and disk-excluded) children
            # are recorded so the serving ladder / ladder census can attribute
            # their windows.  A still-failing child is a recorded chart gap
            # served by the ladder -- never numerical quadrature.
            chart_report['subdivided'] = True
            chart_reports.append(chart_report)
            chart_report['subdivision'] = _subdivide_farfield_tile(
                tile=tile, parent_tag=tag, band=band, parity=parity,
                config=config, rng=rng, outdir=outdir,
                exclusion_rho=exclusion_rho,
                interior_admission=None,
                charts=charts, chart_reports=chart_reports,
                exterior_admission=(exterior_admission if parity == 1
                                    else None),
                source_magnitude_max=(y_outer_region if parity == 1
                                      else None))
            continue
        charts.append(chart)
        chart_reports.append(chart_report)
