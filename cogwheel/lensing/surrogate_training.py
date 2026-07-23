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
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from cogwheel.lensing import prior as _lens_prior
from cogwheel.lensing.waveform import dimensionless_frequency
from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal.channels import (
    ChangRefsdalChannels, farfield_envelope_from_partition)
from cogwheel.lensing.chang_refsdal._hyp1f1 import HypergeometricDomainError
from cogwheel.lensing.ppgo_map import (
    CertifiedPpgoMap, UNKNOWN, get_certified_ppgo_map)
from cogwheel.lensing.surrogate import (
    FarFieldChart, TubeChart, LensAmplificationSurrogate,
    _REFUSAL_ERRORS, _log_w_grid, _uniform_axis)

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
#: (operator wave branch ``w <= 500``; saddle Schwinger ``w <= 60``).
_POSITIVE_W_CEILING = 480.0
_SADDLE_W_CEILING = 58.0

#: Tube caustic-distance band ``[eta_floor, eta_max]`` (source-plane units).
#: Below ``eta_floor`` the fold sharpens and queries fall through to the exact
#: engine; above ``eta_max`` the far-field charts take over.
_DEFAULT_ETA_FLOOR = 0.02
#: The tube's outer eta wall.  MUST stay well inside the local caustic
#: curvature radius: the tube coordinate map (theta, eta) -> source =
#: caustic(theta) + eta * normal(theta) is only inverted by the query-time
#: nearest-caustic projection (theta* = theta, eta* = eta, the
#: foot-of-normal property) for eta below that radius -- and the radius
#: collapses toward cusps.  At 0.30 the map leaves its validity tube
#: (measured: astroid held-out eps 0.52, saddle queries land on foreign
#: arcs and never serve); 0.05 is the design value from the build plan.
_DEFAULT_ETA_MAX = 0.05
#: Minimum caustic distance a far-field chart serves at (tube/far-field seam).
_DEFAULT_FARFIELD_OVERLAP = 0.05

#: A cusp is a local caustic-speed minimum below this fraction of the median
#: speed along the sampled caustic.  A RELATIVE threshold is used rather than
#: the brief's nominal absolute ``1e-6`` because the measured dip depth scales
#: as ``~ caustic_size / n_samples`` (a semicubical cusp has speed -> 0
#: linearly in arc index), so at ~100-200 samples genuine cusps read
#: ``1e-2..1e-5`` -- an absolute cut would miss them.  The relative cut tracks
#: the same "speed collapses at a cusp" signal robustly across gamma.
_CUSP_SPEED_REL_FRAC = 0.2
#: Cusp-window half-width = safety factor x measured dip half-width, floored.
_CUSP_WIDTH_SAFETY = 1.5
_CUSP_MIN_HALFWIDTH = 0.05
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
#: Fractional shrink of each fold arc away from its bounding walls.
_ARC_MARGIN_FRAC = 0.03
#: Small offset off the saddle wedge edges when sampling a branch.
_WEDGE_EPS = 1e-3
#: Caustic distance used to probe which side of a fold carries the image pair.
_PROBE_ETA = 0.05
#: Margin below the double-double product ceiling ``w * |y| <= 60`` used to cap
#: each chart's ``w`` grid.  Mirrors the prior's mass coupling, which keeps
#: ``w * |y| <= ~55`` by construction (the mass-conditioned source scale), so a
#: chart never samples the (large-w, large-|y|) corner the engine refuses.
_DD_PRODUCT_MARGIN = 58.0

#: Expected cusp counts by parity (astroid / deltoid, both lobes summed).
_EXPECTED_CUSPS = {1: 4, -1: 6}


class CausticTopologyError(ValueError):
    """Detected cusp count disagrees with the expected caustic topology.

    Raised when the number of caustic-speed minima found for a parity does not
    match the analytic expectation (4 astroid / 6 deltoid).  A mismatch means
    the caustic sampling or the engine geometry is wrong; it is a flagged error,
    never a silent pass.
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
                           f_hi_hz: float = DEFAULT_F_HI_HZ) -> 'PriorBox':
        """Read the box from the lens prior classes.

        Parameters
        ----------
        f_lo_hz, f_hi_hz : float, optional
            Detector frequency band bounds (Hz); defaults 20 / 1024.
        """
        gamma_range = tuple(
            _lens_prior.UniformReducedShearPrior.range_dic['gamma'])
        ln_m = tuple(
            _lens_prior.UniformLensMassPrior.range_dic['ln_m_lens_msun'])
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
    n_y1: int = 4
    n_y2: int = 4
    w_nodes_per_decade: int = 4
    eta_floor: float = _DEFAULT_ETA_FLOOR
    eta_max: float = _DEFAULT_ETA_MAX
    farfield_overlap: float = _DEFAULT_FARFIELD_OVERLAP
    gamma_band_halfwidth: float = 0.1
    min_gamma_band: float = 0.02
    engine_budget: int = 400
    max_tube_arcs: int = 1
    # ``None`` = no cap (the production default: the tiling itself bounds the
    # count); an int caps admitted tiles with a loud truncation record.
    max_farfield_regions: int | None = None
    # Cartesian grid side for the mass-stratified far-field tiling (Build 8g
    # WP2): each stratum's shear-frame y-support box ``[-Y(m_lo), Y(m_lo)]^2``
    # is split into ``n_farfield_tiles_per_side^2`` square tiles (tile half
    # ``Y(m_lo) / n``); only tiles lying wholly outside the caustic disk are
    # admitted.  ``max_farfield_regions`` then caps the total admitted tiles.
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
    """Caustic point and unit source-plane normal at ``(gamma, theta, branch)``.

    The normal is the unit perpendicular to the caustic tangent (finite-
    differenced along ``theta`` on the same branch).
    """
    caust = np.asarray(
        geometry.critical_point(gamma, theta, 0.0, 0.0, branch).source,
        dtype=float)
    dth = 1e-6
    caust2 = np.asarray(
        geometry.critical_point(gamma, theta + dth, 0.0, 0.0, branch).source,
        dtype=float)
    tangent = caust2 - caust
    tangent /= np.hypot(tangent[0], tangent[1])
    normal = np.array([-tangent[1], tangent[0]])
    return caust, normal


def _tube_source(gamma: float, theta: float, eta: float, branch: int,
                 sign: int) -> np.ndarray:
    """Source at caustic distance ``eta`` off the ``branch`` fold at ``theta``."""
    caust, normal = _tube_normal(gamma, theta, branch)
    return caust + sign * eta * normal


def _probe_arc_side(gamma: float, theta: float, branch: int
                    ) -> tuple[int, int] | None:
    """Choose the image-pair side of a fold arc, returning ``(sign, n_img)``.

    Places a test source at ``_PROBE_ETA`` on each side of the fold and keeps
    the side whose nearest caustic point faithfully reconstructs the intended
    ``(distance, theta)`` (so the source really sits on this fold), preferring
    the side with more real images (where the fold image pair is present).
    Returns ``None`` if neither side reconstructs faithfully.
    """
    caust, normal = _tube_normal(gamma, theta, branch)
    matrix = geometry.macro_matrix(gamma, 0.0, 0.0)
    best: tuple[int, int] | None = None
    for sign in (1, -1):
        source = caust + sign * _PROBE_ETA * normal
        try:
            near = geometry.nearest_caustic_point(gamma, 0.0, source, kappa=0.0)
            dtheta = abs((near.theta - theta + np.pi) % (2.0 * np.pi) - np.pi)
            if abs(near.distance - _PROBE_ETA) > 0.25 * _PROBE_ETA \
                    or dtheta > 0.1:
                continue
            n_img = len(geometry.find_images(source, matrix))
        except geometry.LensDomainError:
            # A refused probe (near-degenerate census, F012) means this side is
            # not cleanly on the fold; skip it conservatively.
            continue
        if best is None or n_img > best[1]:
            best = (sign, n_img)
    return best


def _branch_speed_profile(gamma: float, branch: int, theta_lo: float,
                          theta_hi: float, n: int, periodic: bool
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Caustic ``theta`` samples and speed ``|d caustic / d theta|`` on a branch.

    Points outside the branch's domain (saddle wedge) are dropped.
    """
    thetas = (np.linspace(theta_lo, theta_hi, n, endpoint=False) if periodic
              else np.linspace(theta_lo, theta_hi, n))
    good_theta, points = [], []
    for theta in thetas:
        try:
            points.append(np.asarray(
                geometry.critical_point(gamma, theta, 0.0, 0.0, branch).source,
                dtype=float))
            good_theta.append(theta)
        except geometry.LensDomainError:
            continue
    good_theta = np.asarray(good_theta)
    points = np.asarray(points)
    if points.shape[0] < 4:
        return good_theta, np.array([])
    if periodic:
        deriv = 0.5 * (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0))
        step = float(thetas[1] - thetas[0])
        speed = np.hypot(deriv[:, 0], deriv[:, 1]) / step
    else:
        deriv = np.gradient(points, good_theta, axis=0)
        speed = np.hypot(deriv[:, 0], deriv[:, 1])
    return good_theta, speed


def _find_cusps(thetas: np.ndarray, speed: np.ndarray, periodic: bool, *,
                width_safety: float = _CUSP_WIDTH_SAFETY,
                min_halfwidth: float = _CUSP_MIN_HALFWIDTH
                ) -> list[tuple[float, float]]:
    """Cusp ``(theta, delta_theta)`` pairs from caustic-speed minima.

    A cusp is a local minimum of ``speed`` below `_CUSP_SPEED_REL_FRAC` of the
    median speed; ``delta_theta`` is ``width_safety`` times the half-width of
    the below-threshold dip around it, floored at ``min_halfwidth``.  The
    astroid path uses the module defaults (`_CUSP_WIDTH_SAFETY`,
    `_CUSP_MIN_HALFWIDTH`); the saddle path passes its wider
    `_SADDLE_CUSP_WIDTH_SAFETY` / `_SADDLE_CUSP_MIN_HALFWIDTH` (Build 8g WP3).
    """
    if speed.size < 4:
        return []
    threshold = _CUSP_SPEED_REL_FRAC * float(np.median(speed))
    n = speed.size
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
        cusps.append((float(thetas[i]), float(delta)))
    return cusps


def _astroid_arcs(gamma: float, n: int
                  ) -> tuple[list[tuple[float, float]], list[FoldArc], float]:
    """Cusps and fold arcs of the positive-parity astroid (single branch)."""
    thetas, speed = _branch_speed_profile(
        gamma, 1, 0.0, 2.0 * np.pi, n, periodic=True)
    cusps = _find_cusps(thetas, speed, periodic=True)
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
    """Cusps and fold arcs of the macro-saddle deltoid (two lobes, two branches).

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
        lo_edge = center - theta_max + _WEDGE_EPS
        hi_edge = center + theta_max - _WEDGE_EPS
        for branch in (1, -1):
            thetas, speed = _branch_speed_profile(
                gamma, branch, lo_edge, hi_edge, n, periodic=False)
            reach = max(reach, _caustic_reach(
                gamma, branch, lo_edge, hi_edge, n))
            cusps = _find_cusps(
                thetas, speed, periodic=False,
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
    # Probe several interior thetas, not just the midpoint: a mid-lobe arc's
    # midpoint can sit in the near-axial F012 census dead zone at isolated
    # gammas, and a single refused probe would silently drop a physically
    # present arc (measured: deltoid branch -1 arcs vanished at gamma =
    # 1.245/1.265/1.305/1.315 only, flickering the arc count 6 -> 4 and
    # shredding the band splitter's stable sub-bands).
    span = inner_hi - inner_lo
    side = None
    for frac in (0.5, 0.35, 0.65, 0.2, 0.8):
        side = _probe_arc_side(gamma, inner_lo + frac * span, branch)
        if side is not None:
            break
    if side is None:
        return None
    sign, image_count = side
    return FoldArc(branch=branch, theta_lo=float(inner_lo),
                   theta_hi=float(inner_hi), inward_sign=int(sign),
                   image_count=int(image_count),
                   cusp_windows=tuple((float(t), float(w)) for t, w in windows))


def _caustic_reach(gamma: float, branch: int, theta_lo: float,
                   theta_hi: float, n: int) -> float:
    """Maximum source-plane radius of the caustic over a branch sweep."""
    reach = 0.0
    for theta in np.linspace(theta_lo, theta_hi, n):
        try:
            src = geometry.critical_point(gamma, theta, 0.0, 0.0, branch).source
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
                       n_samples: int = 200, min_width: float = 0.02
                       ) -> tuple[list[tuple[tuple[float, float],
                                             CausticStructure]],
                                  list[tuple[float, float]]]:
    """Bisect a gamma band into topology-stable sub-bands.

    The saddle deltoid's fold-arc partition changes at discrete gamma
    values (cusps migrate through wedge walls), so a single rectangular
    tube grid cannot span such a metamorphosis.  Bands failing the
    `band_caustic_structure` consistency guard are bisected; slivers
    narrower than ``min_width`` that still straddle a change are DROPPED
    (refusal-conservative: those gammas fall through to far-field/exact
    serving, mirroring the ``gamma = 1`` guard band) and returned in the
    second list so the caller can record them.

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
        stable.append((sub, structure))
    return sorted(stable), sorted(dropped)


def _min_curvature_radius(band: tuple[float, float], arc: FoldArc,
                          n_samples: int) -> float:
    """Minimum caustic curvature radius over an arc, worst gamma in band.

    Three-point circumradius over densely sampled caustic points at the
    band's edge gammas (curvature is worst where the caustic is
    smallest). Conservative floor for the foot-of-normal assertion.
    """
    r_min = np.inf
    thetas = np.linspace(arc.theta_lo, arc.theta_hi, max(n_samples // 2, 32))
    for gamma in (band[0], band[1]):
        pts = np.array([
            geometry.critical_point(float(gamma), float(t), 0.0, 0.0,
                                    arc.branch).source
            for t in thetas])
        for i in range(1, len(pts) - 1):
            a, b, c = pts[i - 1], pts[i], pts[i + 1]
            ab, bc, ca = (np.linalg.norm(b - a), np.linalg.norm(c - b),
                          np.linalg.norm(a - c))
            area2 = abs((b[0] - a[0]) * (c[1] - a[1])
                        - (b[1] - a[1]) * (c[0] - a[0]))
            if area2 < 1e-30:
                continue  # collinear: infinite radius, not a constraint
            r_min = min(r_min, ab * bc * ca / (2.0 * area2))
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
    """Lower an uncapped ``w`` top by the parity ceiling and the DD product cap.

    The double-double point-mass kernel refuses ``w * |y|`` above its ceiling,
    so the largest ``w`` a chart may sample is ``_DD_PRODUCT_MARGIN /
    y_magnitude`` where ``y_magnitude`` is the largest source MAGNITUDE (``|y|``,
    not a per-axis coordinate) the chart reaches.  One authoritative place for
    the ceiling + DD arithmetic shared by `_capped_w_range` (tube-shell radius)
    and `_stratum_w_range` (far-field square-box corner).

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
    """Chart ``w`` band, capped so ``w_max * y_max`` stays below the DD ceiling.

    Starts from the prior's mass-derived ``w`` band (the full prior mass range)
    and lowers the upper edge to ``_DD_PRODUCT_MARGIN / y_max`` when the chart's
    largest source magnitude would otherwise push ``w * |y|`` past the
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


def _farfield_tiles(y_extent: float, exclusion_radius: float, n_per_side: int
                    ) -> list[tuple[tuple[float, float], float, int, int]]:
    """Square exterior tiles of the ``[-y_extent, y_extent]^2`` support box.

    Lays a uniform ``n_per_side x n_per_side`` Cartesian grid of square tiles
    (tile half ``y_extent / n_per_side``) over the shear-frame y-support box and
    ADMITS a tile iff its axis-aligned box lies WHOLLY OUTSIDE the disk of
    radius ``exclusion_radius`` centered at the origin.  The minimum L2 distance
    from the origin to a tile box ``[cx-h, cx+h] x [cy-h, cy+h]`` is
    ``hypot(max(0, |cx|-h), max(0, |cy|-h))``.  Because the caustic lies inside
    that disk, an admitted tile is entirely in the single 2-image exterior
    region, so the one-image-count-per-box constraint holds by construction and
    no per-point engine probing is needed (Professor 8g Q2).

    Parameters
    ----------
    y_extent : float
        Half-width of the (square) y-support box, ``Y(m_lo)``.
    exclusion_radius : float
        Caustic-disk radius ``caustic_reach + eta_max``; tiles touching this
        disk are dropped (covered by the tube shell + serving ladder).
    n_per_side : int
        Number of tiles along each axis.

    Returns
    -------
    list[tuple[tuple[float, float], float, int, int]]
        ``(tile_center, half, i, j)`` for each admitted tile, in row-major grid
        order (deterministic).
    """
    half = y_extent / n_per_side
    centers = [-y_extent + half * (2 * k + 1) for k in range(n_per_side)]
    tiles: list[tuple[tuple[float, float], float, int, int]] = []
    for i, cx in enumerate(centers):
        for j, cy in enumerate(centers):
            dx = max(0.0, abs(cx) - half)
            dy = max(0.0, abs(cy) - half)
            if math.hypot(dx, dy) >= exclusion_radius:
                tiles.append(((float(cx), float(cy)), float(half), i, j))
    return tiles


def _caustic_points(gamma: float, parity: int, n: int) -> np.ndarray:
    """All sampled caustic source-plane points for one parity, shape ``(k, 2)``.

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
            thetas = np.linspace(center - theta_max + _WEDGE_EPS,
                                 center + theta_max - _WEDGE_EPS, n)
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
    returns ``0``.  Only meaningful for a genuinely ordered single loop (the
    positive-parity astroid sweep); the disjoint saddle lobes are NOT such a
    loop, so this is never applied to them.
    """
    angles = np.arctan2(points[:, 1], points[:, 0])
    increments = np.diff(np.concatenate([angles, angles[:1]]))
    increments = (increments + np.pi) % (2.0 * np.pi) - np.pi
    return float(increments.sum() / (2.0 * np.pi))


def _caustic_inradius(gamma: float, parity: int, n: int) -> tuple[float, bool]:
    """Minimum caustic radius and whether the caustic encloses the origin.

    Returns ``(inradius, encloses_origin)``.  ``inradius`` is the smallest
    source-plane radius any caustic point reaches -- the radius of the largest
    origin-centred disk that fits inside the caustic curve, which the interior
    far-field tiles must stay within.

    ``encloses_origin`` keys the interior admission off the caustic TOPOLOGY,
    never a bare image count (Professor 8h-a):

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
    radii = np.hypot(points[:, 0], points[:, 1])
    inradius = float(radii.min())
    if parity != 1:
        return inradius, False
    return inradius, abs(_winding_number(points)) >= 0.5


def _farfield_interior_tiles(grid_extent: float, admit_radius: float,
                             n_per_side: int
                             ) -> list[tuple[tuple[float, float], float,
                                             int, int]]:
    """Square interior tiles wholly inside the caustic disk minus the tube shell.

    Companion to `_farfield_tiles` with the admission test INVERTED: lays the
    same uniform ``n_per_side x n_per_side`` grid over
    ``[-grid_extent, grid_extent]^2`` (tile half ``grid_extent / n_per_side``)
    and ADMITS a tile iff its axis-aligned box lies WHOLLY INSIDE the disk of
    radius ``admit_radius`` centred at the origin.  The MAXIMUM L2 distance from
    the origin to a tile box ``[cx-h, cx+h] x [cy-h, cy+h]`` is the corner
    FARTHEST from the origin, ``hypot(|cx| + h, |cy| + h)``; when that stays
    below ``admit_radius`` the whole tile is inside the caustic interior
    (single 4-image region) with no caustic crossing and no overlap with the
    tube shell.  So the one-image-count-per-box constraint holds by
    construction, exactly as the exterior predicate enforces 2-image by
    geometry -- no per-point engine image-count probing (Professor 8h-a).

    Parameters
    ----------
    grid_extent : float
        Half-width of the (square) grid, ``min(interior_admit_radius,
        Y(m_lo))`` (the intersection of the interior disk and the prior
        y-support box).
    admit_radius : float
        Interior admission radius ``caustic_inradius - eta_max`` (the caustic
        disk minus the tube shell); tiles whose farthest corner exceeds this
        are dropped (they would straddle the tube shell or the caustic).
    n_per_side : int
        Number of tiles along each axis.

    Returns
    -------
    list[tuple[tuple[float, float], float, int, int]]
        ``(tile_center, half, i, j)`` for each admitted tile, in row-major grid
        order (deterministic).
    """
    half = grid_extent / n_per_side
    centers = [-grid_extent + half * (2 * k + 1) for k in range(n_per_side)]
    tiles: list[tuple[tuple[float, float], float, int, int]] = []
    for i, cx in enumerate(centers):
        for j, cy in enumerate(centers):
            far = math.hypot(abs(cx) + half, abs(cy) + half)
            if far <= admit_radius:
                tiles.append(((float(cx), float(cy)), float(half), i, j))
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


def _build_tube_chart(*, gamma_grid: np.ndarray, arc: FoldArc, parity: int,
                      w_range: tuple[float, float], config: TrainingConfig
                      ) -> tuple[TubeChart, int, int]:
    """Build one tube chart over ``(log w, gamma, u=sqrt(eta), theta)``.

    Returns the chart, the number of engine calls, and the number of refused
    grid points (left as zeros in the value tensor).
    """
    log_w_grid = _log_w_grid(w_range, config.w_nodes_per_decade)
    w_grid = np.exp(log_w_grid)
    u_grid = np.linspace(np.sqrt(config.eta_floor), np.sqrt(config.eta_max),
                         config.n_u)
    theta_grid = np.linspace(arc.theta_lo, arc.theta_hi, config.n_theta)

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
        eta_floor=config.eta_floor, eta_max=config.eta_max,
        cusp_windows=arc.cusp_windows)
    return chart, calls, refused


def _build_farfield_chart(*, gamma_band: tuple[float, float], parity: int,
                          box_center: tuple[float, float], half: float,
                          w_range: tuple[float, float], config: TrainingConfig
                          ) -> tuple[FarFieldChart, int, int]:
    """Build one far-field chart via the reused 8a `from_engine` trainer."""
    n_points = config.n_gamma * config.n_y1 * config.n_y2
    _budget_check(n_points, config.engine_budget, 'farfield')
    y1_range = (box_center[0] - half, box_center[0] + half)
    y2_range = (box_center[1] - half, box_center[1] + half)
    single = LensAmplificationSurrogate.from_engine(
        gamma_range=gamma_band, y1_range=y1_range, y2_range=y2_range,
        w_range=w_range, n_gamma=config.n_gamma, n_y1=config.n_y1,
        n_y2=config.n_y2, w_nodes_per_decade=config.w_nodes_per_decade)
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

def _heldout_eps(chart: TubeChart | FarFieldChart,
                 samples: Sequence[tuple[float, float, float]],
                 provenance: dict) -> float:
    """Max relative envelope error of a chart over held-out geometry points.

    Serves each ``(gamma, y1, y2)`` through the full guard stack of a one-chart
    surrogate and compares to a fresh engine reference; unserved points are
    skipped.  Returns ``nan`` when no held-out point is served.

    The reference envelope and its normalization depend on the chart type,
    matching the label each chart is trained on (Build 8g-b):

    - a `FarFieldChart` is trained on the far-field label
      ``E_ff = F - sum_{a real} H_a e^{1j w tau_a}``
      (`farfield_envelope_from_partition`), so the reference is that SAME
      helper and the error is F-normalized by ``max|exact_total|`` (``max|E_ff|
      ~ 1e-4`` is too tiny a denominator);
    - a `TubeChart` keeps the caustic-region ``partition.envelope`` reference
      normalized by ``max|E|`` -- byte-identical to HEAD.
    """
    surrogate = LensAmplificationSurrogate([chart], provenance)
    w_grid = np.exp(chart.log_w_grid)
    is_farfield = isinstance(chart, FarFieldChart)
    errors: list[float] = []
    for gamma, y1, y2 in samples:
        channels = ChangRefsdalChannels(w_grid)
        try:
            partition = channels.evaluate(
                gamma=gamma, y=(y1, y2), beta=0.0, kappa=0.0)
        except _ENGINE_REFUSALS:
            continue
        if is_farfield:
            env_true = farfield_envelope_from_partition(partition)
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
    kind : {'tube', 'farfield'}
        Which registration bar to apply.
    eps : float
        The chart's max-normalized held-out envelope error (may be NaN).
    config : TrainingConfig
        Supplies ``tube_eps_max`` and ``farfield_eps_max``.

    Returns
    -------
    tuple[bool, str | None]
        ``(gated, reason)`` where ``reason`` is ``'nan_eps'``,
        ``'eps_above_bar'``, or ``None`` when the chart passes.

    Raises
    ------
    ValueError
        If ``kind`` is neither 'tube' nor 'farfield'.
    """
    bars = {'tube': config.tube_eps_max, 'farfield': config.farfield_eps_max}
    if kind not in bars:
        raise ValueError(
            f"kind must be 'tube' or 'farfield'; got {kind!r}.")
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
    kind : {'tube', 'farfield'}
        Which registration bar to apply.
    report : dict
        The per-chart report returned by `_load_or_build`.
    config : TrainingConfig
        Supplies ``tube_eps_max`` and ``farfield_eps_max``.

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
                          config: TrainingConfig, rng: np.random.Generator
                          ) -> list[tuple[float, float, float]]:
    """Random served-interior held-out sources for a tube chart."""
    samples: list[tuple[float, float, float]] = []
    for _ in range(config.n_heldout):
        gamma = float(rng.uniform(*gamma_band))
        eta = float(rng.uniform(config.eta_floor, config.eta_max))
        theta = float(rng.uniform(arc.theta_lo, arc.theta_hi))
        source = _tube_source(gamma, theta, eta, arc.branch, arc.inward_sign)
        samples.append((gamma, float(source[0]), float(source[1])))
    return samples


def _farfield_heldout_samples(gamma_band: tuple[float, float],
                              box_center: tuple[float, float], half: float,
                              config: TrainingConfig,
                              rng: np.random.Generator
                              ) -> list[tuple[float, float, float]]:
    """Random held-out sources inside a far-field chart's raw box."""
    return [(float(rng.uniform(*gamma_band)),
             float(rng.uniform(box_center[0] - half, box_center[0] + half)),
             float(rng.uniform(box_center[1] - half, box_center[1] + half)))
            for _ in range(config.n_heldout)]


# ---------------------------------------------------------------------------
# Per-chart resumability
# ---------------------------------------------------------------------------

def _load_or_build(path: Path, build_fn: Callable[[], tuple],
                   provenance: dict
                   ) -> tuple[TubeChart | FarFieldChart, dict, bool]:
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
        loaded = LensAmplificationSurrogate.load(path)
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


def train(*, outdir: str | Path,
          artifact_path: str | Path | None = None,
          config: TrainingConfig | None = None,
          f_lo_hz: float = DEFAULT_F_LO_HZ,
          f_hi_hz: float = DEFAULT_F_HI_HZ,
          report_path: str | Path | None = None,
          ppgo_map: CertifiedPpgoMap | None = None
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

    Returns
    -------
    LensAmplificationSurrogate
        The packed surrogate.
    dict
        The training report (also written to ``report_path`` if given).
    """
    box = PriorBox.from_prior_classes(f_lo_hz=f_lo_hz, f_hi_hz=f_hi_hz)
    config = config or TrainingConfig()
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
        # band is bisected into topology-stable sub-bands and metamorphosis
        # slivers are dropped (they fall through to far-field/exact serving).
        sub_bands, dropped = stable_gamma_bands(
            band, parity, n_samples=config.n_caustic_samples,
            min_width=config.min_gamma_band)
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
                chart_reports=chart_reports, ppgo_map=ppgo_map)

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


def _subdivide_farfield_tile(
        *, tile: dict, parent_tag: str, band: tuple[float, float],
        parity: int, config: TrainingConfig, rng: np.random.Generator,
        outdir: Path, exclusion_radius: float, interior_admit_radius: float,
        charts: list, chart_reports: list[dict]) -> dict:
    """Halve one eps-gated far-field tile into up to four children (Build 8h-a WP4).

    Single-level corrective subdivision, no recursion: a far-field tile whose
    held-out eps failed the registration bar is split into up to four children
    at ``(cx +/- h/2, cy +/- h/2)`` each with half ``h/2``.  A smaller tile
    carries less envelope oscillation content, so re-fitting the same
    far-field label on a quarter box is a strictly easier fit against the SAME
    (tile-size-invariant, absolute-``max|E_ff|``) ``farfield_eps_max`` bar --
    halving is the corrective lever.

    Each child is re-admitted through the PARENT's OWN region predicate
    (carried verbatim in ``tile['region']`` -- never re-derived from geometry):

    - an exterior parent admits a child iff its min corner satisfies
      ``hypot(max(0, |ccx|-h/2), max(0, |ccy|-h/2)) >= exclusion_radius``
      (mirrors `_farfield_tiles`);
    - an interior parent admits a child iff its max corner satisfies
      ``hypot(|ccx|+h/2, |ccy|+h/2) <= interior_admit_radius``
      (mirrors `_farfield_interior_tiles`).

    A child the disk excludes is DROPPED silently -- it is correct geometry
    (the parent's edge straddles the disk boundary), not a training failure, so
    it is packed into neither ``charts`` nor the still-failing chart reports
    (its outcome is recorded only in the returned subdivision summary).  Each
    admitted child inherits the parent's already-ppGO-trimmed ``w_range``
    verbatim (no per-child ``_stratum_w_range`` / ``_apply_ppgo_trim``
    recompute), retrains via `_build_farfield_chart`, and re-gates via
    `_gate_chart`.  A passing child is appended to ``charts`` and recorded in
    ``chart_reports`` (tag ``{parent_tag}_c{ci}``, ``subdivided_from`` field)
    exactly like a normal admitted tile; a still-failing child is recorded in
    ``chart_reports`` with its ``gate_reason`` but NOT packed -- its windows
    fall to the serving ladder, which the ladder census attributes.  A
    ``nan_eps`` parent whose child re-nans (a genuine engine cancellation in
    the same parity/gamma cell) is EXPECTED to still fail; it is recorded, not
    special-cased -- halving cannot fix a cancellation.

    Children are iterated in a fixed row-major order over the ``+/-h/2`` signs
    (``sx`` outer, ``sy`` inner) so the report is reproducible.

    Parameters
    ----------
    tile : dict
        The gated parent tile record (``center``, ``half``, ``w_range``,
        ``region``, ``si``, ``m_lo``, ``m_hi``).
    parent_tag : str
        The parent chart's tag, used to name children ``{parent_tag}_c{ci}``.
    band, parity, config, rng, outdir
        Threaded through unchanged from `_train_band_charts`.
    exclusion_radius : float
        Exterior admission radius ``caustic_reach + eta_max``.
    interior_admit_radius : float
        Interior admission radius ``caustic_inradius - eta_max``.
    charts : list
        Packed-chart accumulator; passing children are appended in place.
    chart_reports : list of dict
        Per-chart report accumulator; every admitted child (packed or
        still-gated) is appended in place.

    Returns
    -------
    dict
        Subdivision summary (parent tag, region, per-child admission result,
        per-child eps vs bar, packed/recorded) for the ladder census to
        attribute cleared-vs-still-gated windows.
    """
    cx, cy = tile['center']
    child_half = 0.5 * float(tile['half'])
    region = tile['region']
    w_range = tile['w_range']
    si, m_lo, m_hi = tile['si'], tile['m_lo'], tile['m_hi']

    children_summary: list[dict] = []
    ci = 0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            ccx = float(cx) + sx * child_half
            ccy = float(cy) + sy * child_half
            # Re-admit through the PARENT's region predicate (carried
            # verbatim, never re-derived from geometry -- Professor guard (e)).
            if region == 'interior':
                far = math.hypot(abs(ccx) + child_half, abs(ccy) + child_half)
                admitted_child = far <= interior_admit_radius
            else:  # exterior
                dx = max(0.0, abs(ccx) - child_half)
                dy = max(0.0, abs(ccy) - child_half)
                admitted_child = math.hypot(dx, dy) >= exclusion_radius
            if not admitted_child:
                # The parent's edge lies partly across the disk boundary; a
                # disk-excluded child is correct geometry, dropped silently
                # (recorded here for the census, packed nowhere).
                children_summary.append({
                    'ci': ci, 'center': [round(ccx, 6), round(ccy, 6)],
                    'half': round(child_half, 6),
                    'admission': 'disk_excluded', 'result': 'disk_excluded'})
                ci += 1
                continue

            child_center = (ccx, ccy)
            child_tag = f'{parent_tag}_c{ci}'
            child_path = outdir / f'{child_tag}.npz'

            def build_child(center=child_center, half=child_half,
                            w_range=w_range, si=si, m_lo=m_lo, m_hi=m_hi,
                            region=region):
                chart, calls, refused = _build_farfield_chart(
                    gamma_band=band, parity=parity, box_center=center,
                    half=half, w_range=w_range, config=config)
                samples = _farfield_heldout_samples(
                    band, center, half, config, rng)
                eps = _heldout_eps(chart, samples,
                                   {'schema': 'heldout-probe'})
                return chart, calls, refused, {
                    'kind': 'farfield', 'region': region,
                    'image_count': chart.image_count,
                    'stratum_index': si,
                    'stratum_mass_range': [round(m_lo, 3), round(m_hi, 3)],
                    'y_box': [list(center), half],
                    'w_range': [round(w_range[0], 6), round(w_range[1], 6)],
                    'node_counts': {'n_gamma': config.n_gamma,
                                    'n_y1': config.n_y1, 'n_y2': config.n_y2},
                    'heldout_eps': eps}

            chart, report, reused = _load_or_build(
                child_path, build_child,
                {'schema': 'build8c-chart', 'parity': parity})
            gated, gate_reason = _gate_chart('farfield', report, config)
            child_eps = float(report.get('heldout_eps', float('nan')))
            child_report = {'name': child_tag, 'parity': parity,
                            'file': str(child_path), 'reused': reused,
                            'subdivided_from': parent_tag, **report}
            if gated:
                child_report['gated'] = True
                child_report['gate_reason'] = gate_reason
                chart_reports.append(child_report)
                result = 'recorded_gated'
            else:
                charts.append(chart)
                chart_reports.append(child_report)
                result = 'packed'
            children_summary.append({
                'ci': ci, 'center': [round(ccx, 6), round(ccy, 6)],
                'half': round(child_half, 6), 'admission': 'admitted',
                'eps': (None if math.isnan(child_eps)
                        else round(child_eps, 8)),
                'bar': config.farfield_eps_max,
                'gate_reason': gate_reason, 'result': result})
            ci += 1

    return {'parent_tag': parent_tag, 'region': region,
            'child_half': round(child_half, 6), 'children': children_summary}


def _train_band_charts(*, box: 'PriorBox', config: TrainingConfig,
                       rng: np.random.Generator, outdir: Path, parity: int,
                       label: str, band: tuple[float, float],
                       structure: CausticStructure, charts: list,
                       chart_reports: list[dict],
                       ppgo_map: CertifiedPpgoMap | None = None) -> None:
    """Build the tube + far-field charts of one topology-stable gamma band."""
    gamma_grid = _uniform_axis(band, config.n_gamma, f'gamma_{label}')

    # -- Tube charts (per fold arc, resumable) --
    # Cap the tube w grid by the largest source magnitude it samples
    # (caustic reach plus the outer eta wall), so w * |y| stays below the
    # engine's double-double ceiling -- mirroring the prior's mass coupling.
    tube_w_range = _capped_w_range(
        box, parity, structure.caustic_reach + config.eta_max)
    for idx, arc in enumerate(structure.arcs[:config.max_tube_arcs]):
        # FOOT-OF-NORMAL ASSERTION (owner-mandated, checked not
        # remembered): the tube map (theta, eta) -> caustic + eta*normal
        # is invertible by the query-time nearest-point projection only
        # for eta below the local caustic curvature radius. A band whose
        # minimum curvature radius over the arc cannot support eta_max is
        # SKIPPED with a loud record (far-field charts + the serving
        # ladder cover it) -- never trained wrongly (the eta_max=0.3
        # failure class, size-induced at small gamma).
        r_min = _min_curvature_radius(band, arc, config.n_caustic_samples)
        if config.eta_max > 0.5 * r_min:
            chart_reports.append({
                'name': f'chart_{label}_tube_{idx}',
                'parity': parity, 'skipped': 'foot_of_normal',
                'min_curvature_radius': round(float(r_min), 6),
                'eta_max': config.eta_max,
                'theta_range': [round(arc.theta_lo, 5),
                                round(arc.theta_hi, 5)]})
            continue
        tag = f'chart_{label}_tube_{idx}'
        path = outdir / f'{tag}.npz'

        def build_tube(arc=arc, band=band, gamma_grid=gamma_grid,
                       w_range=tube_w_range):
            chart, calls, refused = _build_tube_chart(
                gamma_grid=gamma_grid, arc=arc, parity=parity,
                w_range=w_range, config=config)
            samples = _tube_heldout_samples(band, arc, config, rng)
            eps = _heldout_eps(chart, samples,
                               {'schema': 'heldout-probe'})
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
                'heldout_eps': eps}

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

    # -- Far-field charts (mass-stratified exterior tiling, resumable) --
    # Build 8g WP2: partition the parity's REACHABLE mass range into log strata
    # so each stratum's whole ``[w(20, m), w(1024, m)]`` band fits one chart w
    # range (whole-band containment is the serving contract), then tile each
    # stratum's shear-frame y-support box ``[-Y(m_lo), Y(m_lo)]^2`` with
    # DISTINCT square tiles lying wholly outside the caustic disk.  This
    # replaces the legacy single hard-coded box that was rebuilt under
    # different filenames, giving real prior coverage.
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

    # The caustic (and the eta_max tube shell around it) lies inside this disk,
    # so a tile wholly outside it is entirely in the single 2-image exterior
    # region -- no per-point engine/geometry probing needed.
    exclusion_radius = structure.caustic_reach + config.eta_max
    caustic_reach = structure.caustic_reach
    gamma_mid = 0.5 * (band[0] + band[1])
    admitted: list[dict] = []
    stratum_records: list[dict] = []
    dropped_strata: list[dict] = []
    # Certified-ppGO strata trimming (Build 8h-a WP3): where the map certifies a
    # region, ppGO serves the high-``w`` tail, so a stratum whose whole band is
    # above the hand-off floor needs no chart (dropped) and one whose top
    # exceeds it is capped (band-split serving hands the tail to ppGO).  The
    # representative rho is the exterior INNER edge (closest to the caustic,
    # highest w_cert): certifying THERE implies the easier outer regions are
    # covered too, so the drop never over-clears.
    ext_rho = (exclusion_radius / caustic_reach
               if caustic_reach > 0.0 else float('inf'))
    ext_boundary = _stratum_ppgo_boundary(parity, gamma_mid, ext_rho, ppgo_map)
    ext_ceiling = _stratum_ppgo_ceiling(parity, gamma_mid, ext_rho, ppgo_map)
    for si, (m_lo, m_hi) in enumerate(strata):
        y_extent = float(_lens_prior._source_scale(m_lo))
        stratum_w_range = _stratum_w_range(box, parity, m_lo, m_hi, y_extent)
        # The high-mass corner of a stratum is beyond the w-cap when the DD /
        # ceiling cap truncates the stratum w range below ``w(f_hi, m_hi)``.
        w_max_uncapped = float(dimensionless_frequency(box.f_hi_hz, m_hi, 0.0))
        corner_beyond_cap = stratum_w_range[1] < w_max_uncapped * (1.0 - 1e-9)
        trimmed_w_range, action = _apply_ppgo_trim(
            stratum_w_range, ext_boundary, ext_ceiling)
        if action == 'drop':
            dropped_strata.append({
                'stratum_index': si, 'region': 'exterior',
                'mass_range': [round(m_lo, 3), round(m_hi, 3)],
                'w_range': [round(stratum_w_range[0], 6),
                            round(stratum_w_range[1], 6)],
                'w_trust': round(float(ext_boundary), 6),
                'reason': 'ppGO certified over the whole stratum w-band'})
            continue
        stratum_w_range = trimmed_w_range
        tiles = _farfield_tiles(
            y_extent, exclusion_radius, config.n_farfield_tiles_per_side)
        stratum_records.append({
            'stratum_index': si,
            'mass_range': [round(m_lo, 3), round(m_hi, 3)],
            'y_extent': round(y_extent, 6),
            'w_range': [round(stratum_w_range[0], 6),
                        round(stratum_w_range[1], 6)],
            'w_max_uncapped': round(w_max_uncapped, 6),
            'high_w_corner_beyond_cap': bool(corner_beyond_cap),
            'ppgo_capped': bool(action == 'cap'),
            'admitted_tiles': len(tiles)})
        for center, half, i, j in tiles:
            admitted.append({
                'si': si, 'i': i, 'j': j, 'center': center, 'half': half,
                'm_lo': m_lo, 'm_hi': m_hi, 'w_range': stratum_w_range,
                'region': 'exterior'})

    # -- Interior (4-image) far-field tiles (Build 8h-a WP3) --
    # The astroid interior is a single 4-image region enclosing the origin, so
    # tiles wholly inside ``caustic_inradius - eta_max`` carry the SAME E_ff /
    # far-field label (the subtraction runs over the morse-sign real_mask, so an
    # interior box subtracts four kernels automatically -- no code change to
    # `_build_farfield_chart`).  The saddle deltoid's lobes do NOT enclose the
    # origin; there is no origin-centred interior disk, so the interior loop
    # records a loud skip and admits nothing (the eps gate is the final safety
    # net for any mis-admission).
    inradius, encloses = _caustic_inradius(
        gamma_mid, parity, config.n_caustic_samples)
    interior_admit_radius = inradius - config.eta_max
    interior_records: list[dict] = []
    interior_admitted = 0
    interior_skip: str | None = None
    if not encloses:
        interior_skip = 'caustic_not_origin_enclosing'
    elif interior_admit_radius <= 0.0:
        interior_skip = 'tube_shell_fills_interior'
    else:
        int_rho = 0.0  # near-origin: the hardest interior region (Build 8h-a)
        int_boundary = _stratum_ppgo_boundary(
            parity, gamma_mid, int_rho, ppgo_map)
        int_ceiling = _stratum_ppgo_ceiling(
            parity, gamma_mid, int_rho, ppgo_map)
        for si, (m_lo, m_hi) in enumerate(strata):
            y_extent = float(_lens_prior._source_scale(m_lo))
            grid_extent = min(interior_admit_radius, y_extent)
            int_w_range = _stratum_w_range(box, parity, m_lo, m_hi, grid_extent)
            trimmed_w_range, action = _apply_ppgo_trim(
                int_w_range, int_boundary, int_ceiling)
            if action == 'drop':
                dropped_strata.append({
                    'stratum_index': si, 'region': 'interior',
                    'mass_range': [round(m_lo, 3), round(m_hi, 3)],
                    'w_range': [round(int_w_range[0], 6),
                                round(int_w_range[1], 6)],
                    'w_trust': round(float(int_boundary), 6),
                    'reason': 'ppGO certified over the whole stratum w-band'})
                continue
            int_w_range = trimmed_w_range
            tiles = _farfield_interior_tiles(
                grid_extent, interior_admit_radius,
                config.n_farfield_tiles_per_side)
            interior_admitted += len(tiles)
            interior_records.append({
                'stratum_index': si,
                'mass_range': [round(m_lo, 3), round(m_hi, 3)],
                'grid_extent': round(float(grid_extent), 6),
                'w_range': [round(int_w_range[0], 6), round(int_w_range[1], 6)],
                'ppgo_capped': bool(action == 'cap'),
                'admitted_tiles': len(tiles)})
            for center, half, i, j in tiles:
                admitted.append({
                    'si': si, 'i': i, 'j': j, 'center': center, 'half': half,
                    'm_lo': m_lo, 'm_hi': m_hi, 'w_range': int_w_range,
                    'region': 'interior'})

    # Loud interior summary.  Where geometry permits an interior disk (origin
    # enclosed, admit radius positive) admission MUST be non-empty; a zero count
    # there is a coverage defect and is flagged loudly (mirrors the exterior
    # n_per_side admission finding).
    interior_report: dict = {
        'name': f'chart_{label}_farfield_interior',
        'parity': parity, 'interior_summary': True,
        'origin_enclosed': bool(encloses),
        'caustic_inradius': round(float(inradius), 6),
        'interior_admit_radius': round(float(interior_admit_radius), 6),
        'interior_admitted_tiles': int(interior_admitted),
        'strata': interior_records}
    if interior_skip is not None:
        interior_report['interior_skipped'] = interior_skip
    elif interior_admitted == 0:
        interior_report['interior_zero_admission'] = True
    chart_reports.append(interior_report)

    # Loud per-stratum summary (0-count strata are recorded too: a high-mass
    # stratum whose whole y-box lies inside the caustic disk admits no exterior
    # tile -- those draws are near-caustic, served by tube + ladder).  Strata
    # dropped by ppGO trimming are listed separately so the ladder census can
    # attribute the cleared budget.
    chart_reports.append({
        'name': f'chart_{label}_farfield_strata',
        'parity': parity, 'strata_summary': True,
        'n_strata': len(strata),
        'exclusion_radius': round(float(exclusion_radius), 6),
        'strata': stratum_records,
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
        # Interior (4-image) tiles reuse the identical far-field build/serve
        # path -- only the tag infix distinguishes them (``ffin`` vs ``ff``).
        infix = 'ffin' if region == 'interior' else 'ff'
        tag = f'chart_{label}_s{si}_{infix}_{i}_{j}'
        path = outdir / f'{tag}.npz'

        def build_ff(band=band, center=center, half=half, w_range=w_range,
                     si=si, m_lo=m_lo, m_hi=m_hi, region=region):
            chart, calls, refused = _build_farfield_chart(
                gamma_band=band, parity=parity, box_center=center,
                half=half, w_range=w_range, config=config)
            samples = _farfield_heldout_samples(
                band, center, half, config, rng)
            eps = _heldout_eps(chart, samples,
                               {'schema': 'heldout-probe'})
            return chart, calls, refused, {
                'kind': 'farfield', 'region': region,
                'image_count': chart.image_count,
                'stratum_index': si,
                'stratum_mass_range': [round(m_lo, 3), round(m_hi, 3)],
                'y_box': [list(center), half],
                'w_range': [round(w_range[0], 6), round(w_range[1], 6)],
                'node_counts': {'n_gamma': config.n_gamma,
                                'n_y1': config.n_y1, 'n_y2': config.n_y2},
                'heldout_eps': eps}

        chart, report, reused = _load_or_build(
            path, build_ff, {'schema': 'build8c-chart', 'parity': parity})
        gated, gate_reason = _gate_chart('farfield', report, config)
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
                exclusion_radius=exclusion_radius,
                interior_admit_radius=interior_admit_radius,
                charts=charts, chart_reports=chart_reports)
            continue
        charts.append(chart)
        chart_reports.append(chart_report)
