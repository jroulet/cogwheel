"""
Fast multi-chart tensor-cubic-spline emulator of the Chang-Refsdal
envelope ``E(w)``.

WHAT
----
`LensAmplificationSurrogate` is an offline-trained emulator of the SACR-C
transition envelope ``E(w)`` -- the single beat-free, smooth object the
lensed relative-binning likelihood already interpolates
(`cogwheel.lensing.chang_refsdal.channels.ChangRefsdalPartition.envelope`).
Real and imaginary parts of ``E`` are interpolated *separately* (never
magnitude/phase, which aliases under phase wrap) by tensor-product cubic
B-splines with not-a-knot boundary conditions.

A GLOBAL surrogate is a flat collection of two kinds of chart (Build 8c):

- `FarFieldChart` -- a raw-eigenframe-coordinate chart over
  ``(log w, gamma, y1_eig, y2_eig)``, one per image-count region, valid
  away from the caustic.  This is exactly the single-box interpolant the
  8a surrogate shipped; a global surrogate carries several of them.
- `TubeChart` -- a near-caustic chart in the caustic-adapted coordinates
  ``(gamma, u = sqrt(eta), theta, log w)``, where ``eta`` is the
  source-plane distance to the caustic and ``theta`` its arc position
  (`geometry.NearestCausticPoint`).  Fitting in ``u = sqrt(eta)``
  linearizes the fold's square-root branch so the interpolant is smooth
  through the near-caustic transition; ``theta`` is BOUNDED and
  NON-PERIODIC (a single inter-cusp fold arc), with cusp neighbourhoods
  excluded and served by the exact engine.

At query time a deterministic guard stack (`select_chart`) picks at most
one chart -- keying only on the certified physical quantities ``gamma``,
the caustic distance ``eta`` and the image count, never on the gauge
angle ``theta`` except for the cusp-window exclusion test (FINDINGS
F017) -- or falls through (``served=False``) so the caller re-evaluates
with the exact engine.

WHY
---
The exact engine is the ground truth and the fallback: it is certified
(FINDINGS F005/F013) but costs tens of milliseconds per envelope node.
The surrogate is a purely *additive* speed layer -- it never overrides a
refusal and never serves outside its validated domain.  A surrogate
answer where the engine would refuse would be the F005 failure mode and
is guarded against by the per-chart exclusion balls, the gamma guard
band near the ``det A = 0`` parity boundary, and the image-count guard.

Evaluation mechanism (precomputed tensor cubic B-spline)
--------------------------------------------------------
Each chart stores real/imag cubic B-spline coefficient tensors plus one
knot vector per axis, built ONCE at construction by successive 1-D
`scipy.interpolate.make_interp_spline` fits along each of the four axes
(`_fit_tensor_spline`).  A query fixes the three parameter axes
(``gamma`` and either ``(y1_eig, y2_eig)`` or ``(u, theta)``) and varies
only ``ln w``: the coefficient tensor is contracted at the three fixed
coordinates down to a single 1-D B-spline in ``ln w``
(`_contract_tensor_spline`), evaluated at every ``w`` node.  This is a
handful of de Boor evaluations -- no per-call linear solve -- so a served
query is deterministic and well under 0.1 ms even for hundreds of ``w``
nodes.

Coordinate conventions
----------------------
- ``w`` is the dimensionless lensing frequency
  ``w = 8*pi*G*M_lens*(1+z_lens)*f/c^3``; the log-w training axis is
  ``ln w`` (natural log).  Delays ``tau`` are in seconds and are not axes.
- ``kappa`` is fixed at 0 (sampled space): the mass-sheet degeneracy is
  eliminated upstream, so there is no convergence axis.  A candidate with
  ``kappa != 0`` is served the exact engine by the likelihood (the
  surrogate never sees it).
- BETA ELIMINATION (exact): the engine reduces the source into the shear
  eigenframe via ``z_eig = exp(-i*beta) * (y1 + i*y2)``, i.e. a rotation
  ``R(-beta)``.  The envelope is invariant under this rotation, so
  training is done at ``beta = 0`` and every query rotates its source by
  ``-beta`` into the eigenframe before lookup.

Serialization
-------------
A single ``.npz`` holds every chart flat -- ``chart{i}_re_coeffs`` /
``chart{i}_im_coeffs`` / ``chart{i}_knots_*`` / ``chart{i}_axis*`` plus a
JSON-encoded ``chart{i}_meta`` (kind, image count, parity, exclusion
data) -- alongside a single JSON-encoded ``provenance`` scalar and the
chart count.  There is NO bespoke manifest file.  `save`/`load`
round-trip the interpolant bit-for-bit.  An 8a single-box ``.npz`` (no
``n_charts`` key) loads as a one-chart `FarFieldChart` for backward
compatibility.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

from cogwheel.lensing.chang_refsdal import (
    ChangRefsdalChannels, farfield_envelope_from_partition, geometry)
from cogwheel.lensing.chang_refsdal.channels import (
    FARFIELD_KERNEL_SUM, KNOWN_FARFIELD_DEFINITIONS, INTERIOR_SACR_C,
    KNOWN_INTERIOR_DEFINITIONS)
from cogwheel.lensing.chang_refsdal.geometry import LensDomainError
from cogwheel.lensing.chang_refsdal.operator import CancellationError
from cogwheel.lensing.chang_refsdal._schwinger import (
    SchwingerCertificationError)

if TYPE_CHECKING:  # typing-only; NEVER a runtime import (surrogate is the
    # lower module -- ``surrogate_training`` imports it, not vice versa).
    from cogwheel.lensing.surrogate_training import _SaddleLobeAdmission

# The engine's named refusals.  Any of these at ANY w node marks the whole
# parameter grid point refused (per-w refusal propagation, Professor Q4).
_REFUSAL_ERRORS = (LensDomainError, CancellationError,
                   SchwingerCertificationError)

# Default training resolutions (Professor Q2 sizing).
_DEFAULT_W_NODES_PER_DECADE = 15
_DEFAULT_PARAM_NODES = 7

# Source-plane cusp angles of the astroid caustic (eigenframe polar angle
# ``theta_c``, radians).  Closed form and gamma-INDEPENDENT: only the cusp
# MAGNITUDE (reach) scales with the external shear ``gamma``, not the angle
# (Professor ruling, Build 8h-d2 WP3/D4).  A positive-parity far-field
# chart places an exact spline node on each in-range cusp angle so a cubic
# fit sits a node ON the C2 curvature kink instead of smoothing across it.
_ASTROID_CUSP_ANGLES: tuple[float, ...] = (0.0, np.pi / 2, -np.pi / 2, np.pi)

# Absolute tolerance (radians) below which a cusp angle coincident with an
# existing uniform ``theta_c`` node is treated as the same node and NOT
# doubled, keeping the augmented axis strictly increasing.
_CUSP_NODE_DEDUP_TOL = 1e-9

# Exclusion-ball radius in units of the parameter-grid spacing: a query
# within one grid cell (normalized Euclidean) of any refused training
# point is refused conservatively.
_EXCLUSION_RADIUS = 1.0

# Interpolant degree: cubic tensor-product B-spline with not-a-knot
# boundary conditions -- the exact interpolant `make_interp_spline`
# produces, precomputed once here.
_SPLINE_DEGREE = 3

# Half-width of the gamma guard band around the ``det A = 0`` parity
# boundary ``gamma = 1``.  Inside it the image-count / parity
# classification is fragile (the analog of INS-8a-001), so the guard
# stack falls through to the exact engine.
_GAMMA_GUARD_BAND = 1e-3

# Minimum source-plane caustic distance a far-field chart serves at.
# Nearer the caustic the envelope surface sharpens and its image-count
# region can flip; the tube charts cover the near-caustic band instead.
# Legacy single-box charts adopt this as their ``eta_overlap_min`` so the
# 8a caustic-floor behaviour is preserved after rewiring.
_DEFAULT_CAUSTIC_FLOOR = 0.05

# Envelope-definition tag persisted in each far-field chart's npz meta.
# Each far-field chart is trained on ONE w-windowed window-class label
# (`channels.farfield_envelope_from_partition`, Build 8h-b3-fin S1-2);
# `FARFIELD_KERNEL_SUM` is the historical mid-band kernel-sum label
# ``E_ff = F - sum_{a real} H_a e^{1j w tau_a}`` (the full
# post-geometric-optics remainder with the criticality switch forced to 1
# on every real channel and NO ``tau_c`` demodulation carrier).  The
# serving side must mirror the tag's label EXACTLY (see
# `channels.reconstruct_farfield` / `channels.farfield_ghost_term`) or the
# reconstructed ``F`` would not match the label.  The single authoritative
# tag frozenset lives in `channels` (`KNOWN_FARFIELD_DEFINITIONS`), extended
# atomically with its reconstruction dispatch; the loader hard-refuses a
# far-field chart whose tag is absent or unknown (the v1/v2 partial
# artifacts predate the tag and were trained on the OLD, lobe-flipping
# caustic-region envelope, so reconstructing them under the new definition
# would be finite-but-wrong).  Mixed-tag charts are legal.
_FARFIELD_ENVELOPE_DEFINITION = FARFIELD_KERNEL_SUM
_KNOWN_FARFIELD_DEFINITIONS = KNOWN_FARFIELD_DEFINITIONS

# Interior (inside-the-caustic) envelope-definition tag persisted in each
# interior chart's npz meta (Build S2-3, frozen WP8 amended to whole-interior).
# The whole astroid/deltoid interior is trained on the SACR-C
# ``tau_c``-demodulated envelope ``E`` (`channels.INTERIOR_SACR_C`,
# reconstructed by `channels.reconstruct_from_envelope` with the geometry's
# OWN switch and critical delay) rather than the far-field-style label: the
# far-field label subtracts individually-divergent near-merged image kernels
# and fails generically inside the caustic (eps ~ 6e-2 at mid-gamma), whereas
# the SACR-C label switches the near-merged pair INTO the bounded envelope and
# carries no ``1/(tau_a - tau_c)`` or ``Im tau_c`` denominator.  Interior
# charts stay in the SAME caustic-fixed ``(rho, theta_c)`` coordinate the
# far-field charts use; only the ENVELOPE LABEL differs, so they are still
# ``FarFieldChart`` objects distinguished purely by this tag.  The single
# authoritative interior tag frozenset lives in `channels`
# (`KNOWN_INTERIOR_DEFINITIONS`), extended atomically with its reconstruction
# dispatch.
_INTERIOR_ENVELOPE_DEFINITION = INTERIOR_SACR_C
_KNOWN_INTERIOR_DEFINITIONS = KNOWN_INTERIOR_DEFINITIONS

# Every envelope-definition tag a chart may legally carry: the union of the
# far-field window-class labels and the interior SACR-C label.  The loader
# validates a chart's tag against this union (a far-field-region chart still
# hard-refused if it carries no tag or an unknown one).
_KNOWN_ENVELOPE_DEFINITIONS = (
    KNOWN_FARFIELD_DEFINITIONS | KNOWN_INTERIOR_DEFINITIONS)

# Axis-schema tag persisted in each far-field chart's npz meta (Build
# 8h-b3). A positive-parity far-field chart stores CAUSTIC-FIXED spatial axes
# ``(rho, theta_c)``. Inside the caustic rho is the directional radius ratio;
# outside it is one plus the physical radial offset from the caustic. Thus the
# caustic is exactly rho=1 without coupling the far exterior coordinate to a
# multiplicative gamma-dependent scale. The certified-ppGO map retains its
# separate scalar annulus coordinate. Charts trained before this build stored
# raw
# eigenframe axes ``(y1_eig, y2_eig)``; reconstructing them under the
# caustic-fixed serve mirror would query the spline at the wrong
# coordinate and return a finite-but-wrong ``F``.  The loader hard-refuses
# a far-field chart whose axis-schema tag is absent or unknown (mirroring
# the 8g-b envelope-definition hard-refuse): a stale raw-coordinate,
# scalar-reach, or multiplicative-directional artifact fails loudly.
#
# Build 8h-d2 additionally makes the STORED far-field label frame-invariant
# (`channels.farfield_envelope_from_partition` demodulates by
# ``exp(+1j w t_min)``); a chart trained under the OLD frame-dependent label
# stores incompatible values, so the tag carries a ``_framewinv`` suffix and
# the loader hard-refuses any pre-8h-d2 artifact rather than serving a
# finite-but-wrong reconstruction.
_FARFIELD_AXIS_SCHEMA = 'caustic_radial_offset_rho_theta_framewinv'
_KNOWN_FARFIELD_AXIS_SCHEMAS = frozenset({_FARFIELD_AXIS_SCHEMA})

# Axis-schema tag persisted in each macro-saddle LOBE-INTERIOR chart's npz
# meta (Build S2 saddle-lobe serve).  A lobe-interior chart (``gamma > 1``)
# stores LOBE-LOCAL spatial axes ``(rho_lobe, theta_local)`` centred on the
# lobe's source-plane deltoid centroid, where ``rho_lobe = |y - centroid| /
# r_deltoid(theta_local)`` normalises by the DIRECTIONAL boundary radius so
# ``rho_lobe = 1`` traces the deltoid boundary in every direction (a scalar
# reach overshoots the near-cusp directions of an elongated/sheared lobe and
# leaves its interior untileable).  These axes are meaningless without the
# lobe frame (centroid + directional boundary), which is why they carry their
# OWN schema tag: a far-field or tube chart reconstructed under the lobe
# mirror -- or a lobe chart reconstructed under the far-field mirror -- would
# be queried at the wrong coordinate and return a finite-but-wrong ``F``.  The
# loader hard-refuses a lobe chart whose tag is absent or unknown, exactly as
# the far-field loader does.  The ``_framewinv`` suffix mirrors the far-field
# frame-invariant-label convention: the stored envelope is the ``tau_c``-
# demodulated INTERIOR_SACR_C label.
_LOBE_AXIS_SCHEMA = 'lobe_local_offset_rholobe_thetalocal_framewinv'
_KNOWN_LOBE_AXIS_SCHEMAS = frozenset({_LOBE_AXIS_SCHEMA})

# Real-image count of a macro-saddle deltoid-lobe INTERIOR (``gamma > 1``).
# A candidate strictly inside one lobe images into four real geometric-optics
# images; a lobe-interior training node whose engine partition reports a
# different count straddles a region boundary and is recorded refused rather
# than fitted (Professor Q2).
_MACRO_SADDLE_IMAGE_COUNT = 4

# Angular half-width (radians) of the certified Pearcey-cusp arm coverage,
# subtracted from each TubeChart cusp-exclusion window in `_tube_serves`.
# Each 8c window ``(theta_cusp, delta_theta)`` excludes the tube where the
# sqrt(eta) fold model is invalid near a cusp; the near-cusp uniform
# Pearcey arm now covers the OUTER part of that neighbourhood (far enough
# from the cusp vertex that ``R = hypot(x, y)`` clears the arm's radius
# gate), so the window shrinks to ``max(0, delta_theta - coverage)`` -- the
# complement of the arm's certified reach.  A query still inside the
# shrunken window falls through to the arm, then the exact engine.
# Default 0.0 keeps the 8c windows byte-identical and enables no new
# serving by default; a nonzero value must be pinned by the corner census
# against the arm's measured angular coverage (UNVERIFIED until then).
_CUSP_ARM_COVERAGE = 0.0

# Default package-data artifact name (under ``cogwheel/data/``).  The
# trained global artifact is shipped here once training lands; until then
# `load()` with no argument raises a clear FileNotFoundError.
_DEFAULT_ARTIFACT_NAME = 'lens_amplification_surrogate.npz'


def _rotate_to_eigenframe(y1: float, y2: float,
                          beta: float) -> tuple[float, float]:
    """Rotate a source position into the shear eigenframe.

    Applies ``R(-beta)`` (equivalently ``exp(-i*beta)*(y1 + 1j*y2)``),
    the exact reduction the engine performs, so a query at orientation
    ``beta`` maps onto the ``beta = 0`` training box.

    Parameters
    ----------
    y1, y2 : float
        Source position (dimensionless) in the shear frame at
        orientation ``beta``.
    beta : float
        External shear orientation, radians.

    Returns
    -------
    tuple[float, float]
        The eigenframe source position ``(y1_eig, y2_eig)``.
    """
    cos_b, sin_b = np.cos(beta), np.sin(beta)
    y1_eig = cos_b * y1 + sin_b * y2
    y2_eig = -sin_b * y1 + cos_b * y2
    return float(y1_eig), float(y2_eig)


def _caustic_reach(gamma: float) -> float:
    """Maximum source-plane caustic reach used by conservative disk guards.

    Returns the SAME authoritative ``kappa = 0`` caustic reach the
    certified-ppGO map uses (`ppgo_map.caustic_geometry`). The map's annulus
    coordinate, saddle chart fallback, and physical exterior-disk guards
    remain scalar-reach based. Positive-parity charts use the actual
    directional boundary from `geometry.r_caustic`.

    The reach is the MAXIMUM caustic radius over polar angle (a single
    scalar per ``gamma``), so a physical source beyond ``reach + eta`` lies
    outside the whole caustic and its ``eta`` shell.

    Parameters
    ----------
    gamma : float
        External shear magnitude.

    Returns
    -------
    float
        The scalar source-plane caustic reach (dimensionless ``y`` units).

    Raises
    ------
    LensDomainError
        Propagated from `caustic_geometry` at the ``det A = 0`` parity
        boundary or an over-critical convergence.
    """
    from cogwheel.lensing.ppgo_map import caustic_geometry
    reach, _direction = caustic_geometry(float(gamma), 0.0)
    return float(reach)


def _to_caustic_fixed(gamma: float, y1_eig: float, y2_eig: float
                      ) -> tuple[float, float]:
    """Caustic-fixed ``(rho, theta_c)`` of an eigenframe source position.

    For positive parity, ``rho = |y| / r_caustic`` inside the caustic and
    ``rho = 1 + |y| - r_caustic`` outside it. The piecewise map is continuous
    at the exact critical-curve image ``rho = 1``; its additive exterior arm
    avoids a persistent gamma/radius interpolation coupling. Macro-saddle
    exterior charts use an ADDITIVE scalar-reach offset
    (rho = 1 + |y| - _caustic_reach) -- still scalar because the disconnected
    deltoids do not intersect every origin-centred ray, but additive to remove
    the reach-stretch gamma/radius coupling; drho/d|y| = 1.
    ``theta_c = atan2(y2_eig, y1_eig)`` in ``(-pi, pi]``.

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    y1_eig, y2_eig : float
        Eigenframe source position (dimensionless).

    Returns
    -------
    tuple[float, float]
        ``(rho, theta_c)``.
    """
    source_radius = float(np.hypot(y1_eig, y2_eig))
    theta_c = float(np.arctan2(y2_eig, y1_eig))
    if abs(float(gamma)) < 1.0:
        caustic_radius = geometry.r_caustic(float(gamma), theta_c)
        rho = (source_radius / caustic_radius
               if source_radius <= caustic_radius
               else 1.0 + source_radius - caustic_radius)
    else:
        rho = 1.0 + source_radius - _caustic_reach(gamma)
    return rho, theta_c


def _from_caustic_fixed(gamma: float, rho: float, theta_c: float
                        ) -> tuple[float, float]:
    """Eigenframe source position of a caustic-fixed ``(rho, theta_c)`` node.

    Positive parity uses ``rho * r_caustic`` for ``rho <= 1`` and
    ``r_caustic + rho - 1`` for ``rho > 1``. Macro-saddle exterior charts use
    an ADDITIVE scalar-reach offset (|y| = _caustic_reach + rho - 1) -- still
    scalar because the disconnected deltoids do not intersect every
    origin-centred ray, but additive to remove the reach-stretch gamma/radius
    coupling; drho/d|y| = 1. The map is used at train time before each engine
    evaluation and is the exact inverse of the serve coordinate.

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    rho : float
        Piecewise caustic-fixed radial coordinate for positive parity.
    theta_c : float
        Caustic-fixed polar angle, radians.

    Returns
    -------
    tuple[float, float]
        The eigenframe source position ``(y1_eig, y2_eig)``.
    """
    rho = float(rho)
    if rho < 0.0:
        raise ValueError(f'rho must be non-negative; got {rho}.')
    if abs(float(gamma)) < 1.0:
        caustic_radius = geometry.r_caustic(float(gamma), float(theta_c))
        y_mag = (rho * caustic_radius if rho <= 1.0
                 else caustic_radius + rho - 1.0)
    else:
        y_mag = _caustic_reach(gamma) + rho - 1.0
    return y_mag * float(np.cos(theta_c)), y_mag * float(np.sin(theta_c))


def _lobe_boundary_radius(theta, boundary_theta: np.ndarray,
                          boundary_r: np.ndarray):
    """Directional macro-saddle lobe boundary radius ``r_deltoid(theta)``.

    THE single authoritative definition of the lobe deltoid boundary:
    periodic linear interpolation of ``boundary_r`` over ``boundary_theta``
    with period ``2*pi``.  ``rho_lobe = |y - centroid| / r_deltoid(theta)``
    so ``rho_lobe = 1`` tracks the deltoid boundary in EVERY lobe-local
    direction -- a scalar reach overshoots the near-cusp directions of an
    elongated (sheared) lobe and leaves its interior untileable.  The lobe
    coordinate maps (`_from_lobe_fixed` / `_to_lobe_fixed`) and the
    training-side admission predicate
    (`surrogate_training._SaddleLobeAdmission._r_deltoid`, routed here in
    WP3) BOTH use this helper so the convention has exactly one home.

    Parameters
    ----------
    theta : float or np.ndarray
        Lobe-local polar angle(s), radians.
    boundary_theta : np.ndarray
        Ascending lobe-local angular nodes on ``(-pi, pi]``.
    boundary_r : np.ndarray
        Directional boundary radius at ``boundary_theta`` (dimensionless
        ``y`` units), strictly positive.

    Returns
    -------
    float or np.ndarray
        The directional boundary radius at ``theta``.
    """
    return np.interp(theta, boundary_theta, boundary_r, period=2.0 * np.pi)


def _from_lobe_fixed(centroid: np.ndarray, boundary_theta: np.ndarray,
                     boundary_r: np.ndarray, rho_lobe: float,
                     theta_local: float) -> tuple[float, float]:
    """Eigenframe source of a lobe-local ``(rho_lobe, theta_local)`` node.

    Macro-saddle lobe-interior forward map (train time, before each engine
    call): ``radius = rho_lobe * r_deltoid(theta_local)`` and
    ``y = centroid + radius * (cos theta_local, sin theta_local)``, the
    exact inverse of `_to_lobe_fixed`.  Uses the single authoritative
    `_lobe_boundary_radius`; NEVER a private ``np.interp`` copy.

    Parameters
    ----------
    centroid : np.ndarray
        ``(2,)`` source-plane lobe centroid (lobe-local frame origin).
    boundary_theta, boundary_r : np.ndarray
        Directional-boundary nodes for `_lobe_boundary_radius`.
    rho_lobe : float
        Lobe-local radial coordinate (``rho_lobe = 1`` on the deltoid
        boundary); must be non-negative.
    theta_local : float
        Lobe-local polar angle, radians.

    Returns
    -------
    tuple[float, float]
        The eigenframe source position ``(y1_eig, y2_eig)``.

    Raises
    ------
    ValueError
        If ``rho_lobe`` is negative.
    """
    rho_lobe = float(rho_lobe)
    if rho_lobe < 0.0:
        raise ValueError(f'rho_lobe must be non-negative; got {rho_lobe}.')
    theta_local = float(theta_local)
    radius = rho_lobe * float(
        _lobe_boundary_radius(theta_local, boundary_theta, boundary_r))
    y1_eig = float(centroid[0]) + radius * float(np.cos(theta_local))
    y2_eig = float(centroid[1]) + radius * float(np.sin(theta_local))
    return y1_eig, y2_eig


def _to_lobe_fixed(centroid: np.ndarray, boundary_theta: np.ndarray,
                   boundary_r: np.ndarray, y1_eig: float, y2_eig: float
                   ) -> tuple[float, float]:
    """Lobe-local ``(rho_lobe, theta_local)`` of an eigenframe source.

    Macro-saddle lobe-interior inverse map (serve time): with
    ``rel = y_eig - centroid``, ``theta_local = atan2(rel_y, rel_x)`` and
    ``rho_lobe = |rel| / r_deltoid(theta_local)``.  Uses the single
    authoritative `_lobe_boundary_radius`; NEVER a private ``np.interp``
    copy.  The ``+-pi`` angular seam is handled by the ``period = 2*pi``
    interpolation and round-trips through `_from_lobe_fixed`.

    Parameters
    ----------
    centroid : np.ndarray
        ``(2,)`` source-plane lobe centroid (lobe-local frame origin).
    boundary_theta, boundary_r : np.ndarray
        Directional-boundary nodes for `_lobe_boundary_radius`.
    y1_eig, y2_eig : float
        Eigenframe source position (dimensionless).

    Returns
    -------
    tuple[float, float]
        ``(rho_lobe, theta_local)``.

    Raises
    ------
    ValueError
        If the query lies EXACTLY at the centroid, where ``theta_local``
        is undefined (``atan2(0, 0)``); such a degenerate query is refused
        rather than served an arbitrary angle (Professor Q4).
    """
    rel_x = float(y1_eig) - float(centroid[0])
    rel_y = float(y2_eig) - float(centroid[1])
    if rel_x == 0.0 and rel_y == 0.0:
        raise ValueError(
            'Lobe-local coordinate is undefined at the lobe centroid '
            '(theta_local = atan2(0, 0)); this degenerate query is refused.')
    theta_local = float(np.arctan2(rel_y, rel_x))
    r_dir = float(
        _lobe_boundary_radius(theta_local, boundary_theta, boundary_r))
    rho_lobe = float(np.hypot(rel_x, rel_y)) / r_dir
    return rho_lobe, theta_local


def _log_w_grid(w_range: tuple[float, float],
                nodes_per_decade: int) -> np.ndarray:
    """Build a natural-log-uniform ``ln w`` grid over ``w_range``.

    Parameters
    ----------
    w_range : tuple[float, float]
        ``(w_min, w_max)``, both strictly positive with ``w_min < w_max``.
    nodes_per_decade : int
        Number of grid nodes per decade in ``w``.

    Returns
    -------
    np.ndarray
        1-D strictly increasing ``ln w`` grid.

    Raises
    ------
    ValueError
        If the range is not strictly positive and increasing, or the
        node density is non-positive.
    """
    w_min, w_max = float(w_range[0]), float(w_range[1])
    if not (0.0 < w_min < w_max):
        raise ValueError(
            f'w_range must satisfy 0 < w_min < w_max; got {w_range}.')
    if nodes_per_decade <= 0:
        raise ValueError(
            f'nodes_per_decade must be positive; got {nodes_per_decade}.')
    n_decades = np.log10(w_max / w_min)
    n_nodes = max(4, int(np.ceil(nodes_per_decade * n_decades)) + 1)
    return np.linspace(np.log(w_min), np.log(w_max), n_nodes)


def _uniform_axis(value_range: tuple[float, float], n_nodes: int,
                  name: str) -> np.ndarray:
    """Build a uniform 1-D parameter axis, validating the request.

    Parameters
    ----------
    value_range : tuple[float, float]
        ``(low, high)`` with ``low < high``.
    n_nodes : int
        Number of nodes; must be at least 4 for cubic interpolation.
    name : str
        Axis name, used in error messages.

    Returns
    -------
    np.ndarray
        1-D strictly increasing uniform axis.

    Raises
    ------
    ValueError
        If the range is not increasing or ``n_nodes < 4``.
    """
    low, high = float(value_range[0]), float(value_range[1])
    if not low < high:
        raise ValueError(
            f'{name} range must satisfy low < high; got {value_range}.')
    if n_nodes < 4:
        raise ValueError(
            f'{name} needs at least 4 nodes for cubic interpolation; '
            f'got {n_nodes}.')
    return np.linspace(low, high, n_nodes)


def _validate_axis(axis: np.ndarray, name: str) -> np.ndarray:
    """Return ``axis`` as a validated strictly-increasing 1-D array."""
    arr = np.ascontiguousarray(axis, dtype=float)
    if arr.ndim != 1 or arr.size < 4:
        raise ValueError(
            f'{name} must be a 1-D array with at least 4 nodes; '
            f'got shape {arr.shape}.')
    if not np.all(np.diff(arr) > 0.0):
        raise ValueError(f'{name} must be strictly increasing.')
    return arr


def _union_cusp_nodes(theta_c_grid: np.ndarray,
                      theta_c_range: tuple[float, float]) -> np.ndarray:
    """Union the astroid cusp angles into a positive-parity theta_c axis.

    Adds an exact spline node at every source-plane cusp angle
    (`_ASTROID_CUSP_ANGLES`) that lies within ``theta_c_range`` so a cubic
    chart places a node ON each C2 curvature kink rather than smoothing
    across it (a cusp column is a curvature discontinuity, and a cubic
    spline needs a node on the kink).  A cusp angle coincident with an
    existing uniform node (within `_CUSP_NODE_DEDUP_TOL`) is dropped so the
    axis stays strictly increasing for the spline fit.

    Parameters
    ----------
    theta_c_grid : np.ndarray
        Strictly increasing uniform ``theta_c`` axis to augment.
    theta_c_range : tuple[float, float]
        ``(low, high)`` chart bounds; only cusp angles inside are unioned.

    Returns
    -------
    np.ndarray
        Strictly increasing ``theta_c`` axis with the in-range cusp angles
        unioned in and sorted ascending.
    """
    low, high = float(theta_c_range[0]), float(theta_c_range[1])
    cusp_angles = [a for a in _ASTROID_CUSP_ANGLES if low <= a <= high]
    if not cusp_angles:
        return theta_c_grid
    merged = np.sort(np.concatenate(
        [theta_c_grid, np.array(cusp_angles, dtype=float)]))
    # Keep the first node of every near-coincident cluster: a cusp angle
    # within the dedup tolerance of a uniform node must not double it.
    keep = np.concatenate(
        ([True], np.diff(merged) > _CUSP_NODE_DEDUP_TOL))
    return merged[keep]


def _fit_tensor_spline(axis_grids: tuple[np.ndarray, ...],
                       value_real: np.ndarray, value_imag: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Fit real/imag tensor cubic B-splines over a 4-D value tensor.

    Successive 1-D `make_interp_spline` fits (not-a-knot boundary
    conditions) along each axis produce the SAME tensor-product cubic
    interpolant the per-call `RegularGridInterpolator(method='cubic')`
    evaluated, but computed once.  Each 1-D fit returns its coefficient
    axis leading, so it is rotated to the back; after four fits the layout
    returns to the original axis order.

    Parameters
    ----------
    axis_grids : tuple of np.ndarray
        The four axis grids, in the SAME order as the value-tensor axes
        (leading axis first).
    value_real, value_imag : np.ndarray
        Real and imaginary value tensors, shaped like the axis grids.

    Returns
    -------
    real_coeffs, imag_coeffs : np.ndarray
        Coefficient tensors in the original axis order.
    knots : list of np.ndarray
        One knot vector per axis, in axis order.
    """
    real_c = np.ascontiguousarray(value_real, dtype=float)
    imag_c = np.ascontiguousarray(value_imag, dtype=float)
    knot_list: list[np.ndarray] = []
    for axis_grid in axis_grids:
        spl_r = make_interp_spline(axis_grid, real_c, k=_SPLINE_DEGREE, axis=0)
        spl_i = make_interp_spline(axis_grid, imag_c, k=_SPLINE_DEGREE, axis=0)
        real_c = np.moveaxis(spl_r.c, 0, -1)
        imag_c = np.moveaxis(spl_i.c, 0, -1)
        knot_list.append(np.ascontiguousarray(spl_r.t, dtype=float))
    return (np.ascontiguousarray(real_c, dtype=float),
            np.ascontiguousarray(imag_c, dtype=float), knot_list)


def _contract_tensor_spline(coeffs: np.ndarray, knots: tuple[np.ndarray, ...],
                            v0: float, v1: float, v2: float,
                            log_w_query: np.ndarray) -> np.ndarray:
    """Evaluate a tensor cubic B-spline at fixed params over ``ln w``.

    The three parameter axes are fixed for one query, so the 4-D
    coefficient tensor (axis order ``(log w, p0, p1, p2)``) is contracted
    at ``(v0, v1, v2)`` -- collapsing ``p0``, then ``p1``, then ``p2`` --
    down to a single 1-D B-spline in ``ln w``, evaluated at
    ``log_w_query``.

    Parameters
    ----------
    coeffs : np.ndarray
        Coefficient tensor, axes ``(log w, p0, p1, p2)``.
    knots : tuple of np.ndarray
        Knot vectors ``(t_logw, t_p0, t_p1, t_p2)``.
    v0, v1, v2 : float
        The three fixed parameter coordinates.
    log_w_query : np.ndarray
        ``ln w`` coordinates to evaluate at.

    Returns
    -------
    np.ndarray
        The 1-D spline value at every ``log_w_query`` node.
    """
    t_w, t0, t1, t2 = knots
    cc = BSpline(t0, coeffs, _SPLINE_DEGREE, axis=1)(v0)
    cc = BSpline(t1, cc, _SPLINE_DEGREE, axis=1)(v1)
    cc = BSpline(t2, cc, _SPLINE_DEGREE, axis=1)(v2)
    return BSpline(t_w, cc, _SPLINE_DEGREE, axis=0)(log_w_query)


def _normalize_refused(refused_points: np.ndarray | None) -> np.ndarray:
    """Return ``refused_points`` as a validated ``(n, 3)`` float array.

    ``None`` (no refused training points) normalizes to an empty
    ``(0, 3)`` array.
    """
    if refused_points is None:
        return np.empty((0, 3), dtype=float)
    refused = np.asarray(refused_points, dtype=float)
    if refused.size == 0:
        return np.empty((0, 3), dtype=float)
    if refused.ndim != 2 or refused.shape[1] != 3:
        raise ValueError(
            f'refused_points must have shape (n, 3); got {refused.shape}.')
    return np.ascontiguousarray(refused, dtype=float)


# ---- Interior carrier-continuity guard --------------------------------

#: Fraction of the local caustic reach a node-to-node jump in the parked
#: critical carrier ``critical_source`` must exceed to be read as a
#: critical-basin FLIP (a discontinuous hop of the nearest-caustic point to
#: a different caustic arc).  Away from a medial ridge the nearest-caustic
#: map is Lipschitz, so adjacent dense-grid nodes move the carrier by
#: ``O(node spacing) << reach``; a jump of order the caustic reach itself is
#: unambiguously a basin flip.  A conservative single threshold (not a fit
#: knob): well above smooth motion, well below a genuine flip.
_CARRIER_FLIP_FRACTION = 0.5

#: Maximum phase (radians) the frame-invariant far-field label ``E_tilde``
#: may wind between adjacent spatial nodes at the top of the band before a
#: cubic spline can no longer represent it (Build 8h-d2).  A Nyquist
#: quarter-turn ``pi/2``: beyond it the real/imag components each swing more
#: than the spline tracks, so the tile must be subdivided.  Checked ONLY on
#: the demodulated (frame-invariant) label, whose dominant ``w * t_min``
#: frame phase has already been removed; a surviving violation is genuine
#: physical oscillation the tile is too coarse for.
#:
#: Expressed as a normalized COMPLEX INCREMENT, not a phase step: a violation
#: means the label changes by more than the whole chart's peak ``|E_tilde|``
#: across one adjacent-node gap.  Phase is the wrong observable because the
#: chart splines ``re`` and ``im`` separately, so an ``arg`` swing at an
#: amplitude null is smooth in the fields actually being fitted (F022).
#: Calibrated 2026-07-28: worst must-pass fixture 0.1997, must-raise
#: (synthetic 2.5 rad flip at unit magnitude) 1.8980.  At full amplitude 1.0
#: corresponds to ``pi/3`` of winding -- stricter than the retired ``pi/2``
#: where the label is strong, permissive where it has decayed to noise.
_FARFIELD_CARRIER_STEP_MAX = 1.0


class CarrierDiscontinuityError(ValueError):
    """A SACR-C interior tile straddles a critical-basin (carrier) flip.

    Raised by `_assert_carrier_continuity` when the parked critical
    carrier ``tau_c`` (`ChangRefsdalPartition.critical_delay`, via its
    ``critical_source``) hops discontinuously between adjacent nodes of an
    interior chart -- i.e. the tile crosses a medial ridge of the caustic
    where the nearest-caustic point flips arc.  The ``tau_c``-demodulated
    envelope ``E = e^{-i w tau_c}(...)`` then has a phase KINK across that
    ridge and a single cubic spline over the tile cannot represent it.

    The interior tiler resolves this by SUBDIVISION (the assignment-
    convention reseat realised geometrically: each sub-tile lands in a
    single basin, restoring a consistent carrier reference).  Serve
    consistency is unaffected because the carrier is recomputed fresh from
    the query position at serve and is NEVER interpolated -- only ``E`` is.
    """


def _assert_carrier_continuity(critical_sources: np.ndarray,
                               gamma_grid: np.ndarray,
                               shape: tuple[int, int, int]) -> None:
    """Assert the interior carrier ``tau_c`` is basin-continuous over a tile.

    Interpolator hygiene for the SACR-C interior label (Build S2-3): the
    ``tau_c``-demodulated envelope is smooth in position ONLY within a
    single nearest-caustic basin.  This checks that no pair of adjacent
    nodes along any spatial axis hops the parked critical carrier
    ``critical_source`` by more than `_CARRIER_FLIP_FRACTION` of the local
    caustic reach.  Cusp-aligned S2-1 interior tiles are single-basin by
    construction, so this is generically a no-op; a violation means the
    tile straddles a medial ridge and must be subdivided.

    Parameters
    ----------
    critical_sources : np.ndarray
        Shape ``(n_gamma, n_rho, n_theta, 2)`` parked-carrier
        ``critical_source`` per node; ``NaN`` rows mark refused nodes and
        are skipped (a refused neighbour cannot certify continuity but is
        not itself a flip).
    gamma_grid : np.ndarray
        The ``n_gamma`` gamma axis, for the per-gamma caustic reach.
    shape : tuple[int, int, int]
        The ``(n_gamma, n_rho, n_theta)`` node-grid shape.

    Raises
    ------
    CarrierDiscontinuityError
        If a basin flip is detected between adjacent nodes.
    """
    n_gamma, n_rho, n_theta = shape
    grid = np.asarray(critical_sources, dtype=float).reshape(*shape, 2)
    # Per-gamma caustic reach, broadcast to the full node grid (the reach
    # varies with gamma only).
    reach = np.array([_caustic_reach(float(g)) for g in gamma_grid])
    reach_grid = np.broadcast_to(
        reach[:, None, None], (n_gamma, n_rho, n_theta))
    # Compare adjacent nodes along each spatial axis (gamma, rho, theta_c).
    for axis in range(3):
        n_axis = shape[axis]
        if n_axis < 2:
            continue
        lead = np.take(grid, range(1, n_axis), axis=axis)
        trail = np.take(grid, range(0, n_axis - 1), axis=axis)
        jump = np.linalg.norm(lead - trail, axis=-1)
        # A node pair certifies continuity only when both carriers are
        # finite; the smaller reach of the pair is the conservative scale.
        reach_pair = np.minimum(
            np.take(reach_grid, range(1, n_axis), axis=axis),
            np.take(reach_grid, range(0, n_axis - 1), axis=axis))
        finite = np.isfinite(jump)
        flip = finite & (jump > _CARRIER_FLIP_FRACTION * reach_pair)
        if np.any(flip):
            raise CarrierDiscontinuityError(
                'SACR-C interior tile crosses a critical-basin flip along '
                f'axis {axis} (max carrier jump {np.nanmax(jump[finite]):.3g} '
                f'vs reach scale {float(np.max(reach_pair)):.3g}); subdivide '
                'the tile so each sub-tile lands in a single nearest-caustic '
                'basin.')


def _assert_farfield_carrier_continuity(env_grid: np.ndarray,
                                        w_max: float,
                                        gamma_grid: np.ndarray,
                                        shape: tuple[int, int, int]) -> None:
    """Assert the frame-invariant far-field label is spline-representable.

    Interpolator hygiene for the EXTERIOR far-field label, the far-field twin
    of `_assert_carrier_continuity`.  This is a cheap GROSS-ALIASING SCREEN,
    not an accuracy check -- the held-out eps gate (`_gate_chart`) is the real
    falsifier.  Its job is to catch a tile whose label jumps so violently
    between adjacent nodes that fitting it would alias, and reject it for
    subdivision rather than serve a phase-aliased envelope.

    WHAT IS MEASURED, and why it is not the phase (F022).  `FarFieldChart`
    stores ``envelope_real`` and ``envelope_imag`` and splines them as
    SEPARATE REAL FIELDS.  The representability question is therefore about
    the increments of ``re`` and ``im``, not about ``arg``.  These differ
    exactly where it matters: at an amplitude NULL the label passes close to
    the origin, so ``arg`` swings by ``pi`` while ``re`` and ``im`` pass
    smoothly through zero.  An earlier version of this guard measured ``arg``
    and consequently false-positived on every null -- and because nulls are
    generic in an interference pattern, while `surrogate_training` responds to
    this error by subdividing ONCE and then recording the child as a
    ladder-served gap, that turned a benign feature into silent coverage loss.
    Refinement cannot fix a null (the step pins at ``pi`` as nodes are added
    instead of shrinking like ``1/n``), so the subdivision never converged.

    The measured quantity is the complex increment ``|E_lead - E_trail|``
    between adjacent nodes on the top-of-band slice, normalized by the peak
    ``|E_tilde|`` over the WHOLE grid (every ``w``, not just this slice).
    Normalizing by the whole grid is load-bearing: where the label has decayed
    with ``w``, the top slice can be entirely floating-point noise, and
    noise-relative-to-noise is O(1) while noise-relative-to-the-chart is zero.

    CALIBRATION (measured 2026-07-28 across every known fixture).  Must-pass:
    synthetic continuous 0.1997, synthetic zeroed-flip 0.1997, the two
    ``gamma``-wall guard boxes 0.1160 and 0.1556, the band-split box 0.0000,
    the census dense box 0.0000.  Must-raise: the synthetic pathological grid
    (a 2.5 rad flip at unit magnitude) 1.8980.  The bound sits at 1.0 --
    5x above the worst must-pass and 1.9x below the must-raise.  Two rejected
    alternatives, both recorded so they are not re-proposed: normalizing by the
    top-SLICE peak collapses that margin to 1.24x (must-pass reaches 1.5289),
    and scanning ALL ``w`` slices collapses it to 1.38x (must-pass reaches
    1.3703) because accurate charts genuinely carry large mid-band increments.

    The bound has a tuning-free reading: a violation means the label changes
    by more than the entire chart's peak magnitude across ONE node gap.  At
    full amplitude that corresponds to ``pi/3`` of phase winding
    (``2 sin(phi/2) = 1``), i.e. STRICTER than the retired ``pi/2`` bound
    where the label is strong, and correctly permissive where it has decayed.

    Parameters
    ----------
    env_grid : np.ndarray
        Complex far-field label per node, shape ``(n_w, n_gamma, n_rho,
        n_theta)``.  Refused/unfilled nodes are exactly zero (`from_engine`
        leaves the value arrays zero there) and are skipped: a refused
        neighbour is a hole in the grid, not a discontinuity.
    w_max : float
        Top-of-band dimensionless frequency; the check is applied on this
        (last, highest) ``w`` slice and the value is reported in the error.
    gamma_grid : np.ndarray
        The ``n_gamma`` gamma axis, carried for parallelism with the interior
        guard; length-checked against ``shape``.
    shape : tuple[int, int, int]
        The ``(n_gamma, n_rho, n_theta)`` spatial node-grid shape.

    Raises
    ------
    CarrierDiscontinuityError
        If the normalized adjacent-node increment reaches
        `_FARFIELD_CARRIER_STEP_MAX` along any spatial axis.
    ValueError
        If ``gamma_grid`` length disagrees with ``shape[0]``.
    """
    n_gamma, _n_rho, _n_theta = shape
    if gamma_grid.shape[0] != n_gamma:
        raise ValueError(
            f'gamma_grid length {gamma_grid.shape[0]} does not match '
            f'shape[0] = {n_gamma}.')
    grid = np.asarray(env_grid)
    # Reference scale: the peak over the WHOLE grid, so a decayed top slice
    # is measured against the chart it belongs to rather than against itself.
    all_magnitude = np.abs(grid)
    finite = np.isfinite(all_magnitude)
    scale = float(np.max(all_magnitude[finite], initial=0.0))
    if scale <= 0.0:
        # An all-zero (fully refused) grid carries no label to check.
        return
    top = grid[-1]
    magnitude = np.abs(top)
    # Compare adjacent nodes along each spatial axis (gamma, rho, theta_c).
    for axis in range(3):
        n_axis = shape[axis]
        if n_axis < 2:
            continue
        lead = np.take(top, range(1, n_axis), axis=axis)
        trail = np.take(top, range(0, n_axis - 1), axis=axis)
        mag_lead = np.take(magnitude, range(1, n_axis), axis=axis)
        mag_trail = np.take(magnitude, range(0, n_axis - 1), axis=axis)
        # Only node pairs with a finite, non-zero label on BOTH sides certify
        # (a refused neighbour cannot certify continuity but is not a jump).
        both = ((mag_lead > 0.0) & (mag_trail > 0.0)
                & np.isfinite(mag_lead) & np.isfinite(mag_trail))
        step = np.abs(lead - trail) / scale
        bad = both & (step >= _FARFIELD_CARRIER_STEP_MAX)
        if np.any(bad):
            worst = int(np.argmax(np.where(bad, step, -np.inf)))
            rel = float(np.minimum(mag_lead, mag_trail).flat[worst] / scale)
            raise CarrierDiscontinuityError(
                'Frame-invariant far-field label jumps by more than the '
                f'bound {_FARFIELD_CARRIER_STEP_MAX:.3g} x peak |E| along '
                f'axis {axis} at w_max = {float(w_max):.3g} (max step '
                f'{float(step.flat[worst]):.3g} x peak, at relative amplitude '
                f'{rel:.3g}); subdivide the tile so the demodulated envelope '
                'is spline-representable.')


# ---- Charts -----------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FarFieldChart:
    """Caustic-fixed-coordinate envelope chart, valid away from a caustic.

    Interpolates ``E(w)`` over ``(log w, gamma, rho, theta_c)`` for one
    image-count region, where the two spatial axes are the CAUSTIC-FIXED
    coordinates: a directional radius ratio inside the caustic and a physical
    radial offset outside it, with ``rho = 1`` on the caustic. The second axis
    is ``theta_c = atan2(y2_eig, y1_eig)`` (Build 8h-b3). Serve only where
    ``eta > eta_overlap_min`` (bounded away from the caustic) and the
    candidate matches ``image_count``.

    Attributes
    ----------
    gamma_grid, rho_grid, theta_c_grid, log_w_grid : np.ndarray
        1-D strictly increasing training axes.  ``rho_grid`` /
        ``theta_c_grid`` are the caustic-fixed spatial axes.
    real_coeffs, imag_coeffs : np.ndarray
        Cubic B-spline coefficient tensors, axes ``(log w, gamma, rho,
        theta_c)``.
    knots : tuple of np.ndarray
        Knot vectors ``(t_logw, t_gamma, t_rho, t_theta)``.
    image_count : int or None
        Real-image count of the chart's region (``None`` for a legacy
        single-box chart whose region label was not recorded; then the
        image-count guard is skipped).
    parity : int or None
        Macro-image parity ``+1`` (``gamma < 1``) or ``-1``
        (``gamma > 1``); ``None`` if unrecorded.
    eta_overlap_min : float
        Minimum caustic distance the chart serves at.
    refused_points : np.ndarray
        Shape ``(n, 3)`` caustic-fixed ``(gamma, rho, theta_c)`` training
        points the engine refused; the exclusion-ball gate rejects
        queries within one grid spacing of any of them.
    param_spacing : np.ndarray
        Shape ``(3,)`` mean spacing of ``(gamma, rho, theta_c)`` for the
        exclusion-ball normalization.
    envelope_definition : str
        Tag naming the label the chart's envelope encodes (Build 8g-b).
        Persisted in the npz meta and checked on load; the serving side
        dispatches the reconstruction on it.  Fresh charts default to
        `_FARFIELD_ENVELOPE_DEFINITION`.
    """

    gamma_grid: np.ndarray
    rho_grid: np.ndarray
    theta_c_grid: np.ndarray
    log_w_grid: np.ndarray
    real_coeffs: np.ndarray
    imag_coeffs: np.ndarray
    knots: tuple
    image_count: int | None
    parity: int | None
    eta_overlap_min: float
    refused_points: np.ndarray
    param_spacing: np.ndarray
    envelope_definition: str

    @classmethod
    def from_values(cls, *, gamma_grid: np.ndarray, rho_grid: np.ndarray,
                    theta_c_grid: np.ndarray, log_w_grid: np.ndarray,
                    envelope_real: np.ndarray, envelope_imag: np.ndarray,
                    image_count: int | None, parity: int | None,
                    eta_overlap_min: float = _DEFAULT_CAUSTIC_FLOOR,
                    refused_points: np.ndarray | None = None,
                    envelope_definition: str = _FARFIELD_ENVELOPE_DEFINITION
                    ) -> 'FarFieldChart':
        """Build a far-field chart by fitting splines to a value tensor.

        Parameters
        ----------
        gamma_grid, rho_grid, theta_c_grid, log_w_grid : np.ndarray
            1-D strictly increasing training axes (the two spatial axes
            are the caustic-fixed ``rho`` and ``theta_c``).
        envelope_real, envelope_imag : np.ndarray
            Shape ``(n_w, n_gamma, n_rho, n_theta)`` real/imag envelope
            values.
        image_count, parity : int or None
            Region labels (``None`` if unrecorded).
        eta_overlap_min : float, optional
            Minimum caustic distance served (default the caustic floor).
        refused_points : np.ndarray, optional
            Refused caustic-fixed ``(gamma, rho, theta_c)`` training
            points.
        envelope_definition : str, optional
            Tag naming the label the chart's envelope encodes (default the
            far-field kernel-sum label).  Interior charts pass the SACR-C
            interior tag so the serve side dispatches the SACR-C
            reconstruction (Build S2-3).
        """
        gamma_grid = _validate_axis(gamma_grid, 'gamma_grid')
        rho_grid = _validate_axis(rho_grid, 'rho_grid')
        theta_c_grid = _validate_axis(theta_c_grid, 'theta_c_grid')
        log_w_grid = _validate_axis(log_w_grid, 'log_w_grid')
        expected = (log_w_grid.size, gamma_grid.size, rho_grid.size,
                    theta_c_grid.size)
        _check_value_shape(envelope_real, envelope_imag, expected)
        real_c, imag_c, knots = _fit_tensor_spline(
            (log_w_grid, gamma_grid, rho_grid, theta_c_grid),
            envelope_real, envelope_imag)
        return cls._assemble(
            gamma_grid, rho_grid, theta_c_grid, log_w_grid, real_c, imag_c,
            knots, image_count, parity, eta_overlap_min, refused_points,
            envelope_definition=envelope_definition)

    @classmethod
    def _assemble(cls, gamma_grid, rho_grid, theta_c_grid, log_w_grid,
                  real_coeffs, imag_coeffs, knots, image_count, parity,
                  eta_overlap_min, refused_points,
                  envelope_definition=_FARFIELD_ENVELOPE_DEFINITION
                  ) -> 'FarFieldChart':
        """Assemble a chart from prebuilt coefficient tensors and knots."""
        param_spacing = np.array([
            float(np.mean(np.diff(gamma_grid))),
            float(np.mean(np.diff(rho_grid))),
            float(np.mean(np.diff(theta_c_grid)))])
        return cls(
            gamma_grid=_validate_axis(gamma_grid, 'gamma_grid'),
            rho_grid=_validate_axis(rho_grid, 'rho_grid'),
            theta_c_grid=_validate_axis(theta_c_grid, 'theta_c_grid'),
            log_w_grid=_validate_axis(log_w_grid, 'log_w_grid'),
            real_coeffs=np.ascontiguousarray(real_coeffs, dtype=float),
            imag_coeffs=np.ascontiguousarray(imag_coeffs, dtype=float),
            knots=tuple(np.ascontiguousarray(t, dtype=float) for t in knots),
            image_count=None if image_count is None else int(image_count),
            parity=None if parity is None else int(parity),
            eta_overlap_min=float(eta_overlap_min),
            refused_points=_normalize_refused(refused_points),
            param_spacing=param_spacing,
            envelope_definition=str(envelope_definition))


@dataclass(frozen=True, eq=False)
class LobeInteriorChart:
    """Lobe-local-coordinate interior envelope chart for a macro-saddle lobe.

    Interpolates ``E(w)`` over ``(log w, gamma, rho_lobe, theta_local)`` for
    ONE macro-saddle (``gamma > 1``) deltoid lobe, where the two spatial axes
    are the LOBE-LOCAL polar coordinates centred on the lobe's source-plane
    deltoid centroid: ``rho_lobe = |y - centroid| / r_deltoid(theta_local)``
    (so ``rho_lobe = 1`` traces the deltoid boundary in every direction) and
    ``theta_local = atan2(y - centroid)``.  This mirrors `FarFieldChart` but
    the spatial frame is lobe-local rather than origin-centred, so the chart
    ALSO carries the lobe frame (`centroid`, `other_centroid`, `corridor_half`,
    `boundary_theta`, `boundary_r`) needed to place a query at its true
    physical source at serve time.  The envelope is the ``tau_c``-demodulated
    INTERIOR_SACR_C label (`_INTERIOR_ENVELOPE_DEFINITION`), reconstructed by
    the interior serve mirror; the coordinate convention is stamped as
    `_LOBE_AXIS_SCHEMA` so a mislabeled artifact hard-refuses at load.

    Attributes
    ----------
    gamma_grid, rho_lobe_grid, theta_local_grid, log_w_grid : np.ndarray
        1-D strictly increasing training axes.  ``rho_lobe_grid`` /
        ``theta_local_grid`` are the lobe-local spatial axes.
    real_coeffs, imag_coeffs : np.ndarray
        Cubic B-spline coefficient tensors, axes ``(log w, gamma, rho_lobe,
        theta_local)``.
    knots : tuple of np.ndarray
        Knot vectors ``(t_logw, t_gamma, t_rho_lobe, t_theta_local)``.
    image_count : int or None
        Real-image count of the lobe interior (`_MACRO_SADDLE_IMAGE_COUNT`).
    parity : int or None
        Macro-image parity ``-1`` for a ``gamma > 1`` lobe interior.
    eta_overlap_min : float
        Minimum caustic distance the chart serves at.
    refused_points : np.ndarray
        Shape ``(n, 3)`` lobe-local ``(gamma, rho_lobe, theta_local)``
        training points the engine refused.
    param_spacing : np.ndarray
        Shape ``(3,)`` mean spacing of ``(gamma, rho_lobe, theta_local)``
        for the exclusion-ball normalization.
    envelope_definition : str
        Tag naming the label the chart's envelope encodes (default the
        interior SACR-C label); the serve side dispatches on it.
    centroid : np.ndarray
        ``(2,)`` source-plane lobe centroid (lobe-local frame origin), read
        off the `_SaddleLobeAdmission` frame.
    other_centroid : np.ndarray
        ``(2,)`` centroid of the OTHER lobe, for the serve-time inter-lobe
        corridor refusal.
    corridor_half : float
        Inter-lobe corridor half-width (dimensionless ``y``).
    boundary_theta, boundary_r : np.ndarray
        Directional lobe boundary nodes normalising ``rho_lobe``
        (`_lobe_boundary_radius`).
    """

    gamma_grid: np.ndarray
    rho_lobe_grid: np.ndarray
    theta_local_grid: np.ndarray
    log_w_grid: np.ndarray
    real_coeffs: np.ndarray
    imag_coeffs: np.ndarray
    knots: tuple
    image_count: int | None
    parity: int | None
    eta_overlap_min: float
    refused_points: np.ndarray
    param_spacing: np.ndarray
    envelope_definition: str
    centroid: np.ndarray
    other_centroid: np.ndarray
    corridor_half: float
    boundary_theta: np.ndarray
    boundary_r: np.ndarray

    @classmethod
    def from_lobe_values(cls, *, gamma_grid: np.ndarray,
                         rho_lobe_grid: np.ndarray,
                         theta_local_grid: np.ndarray, log_w_grid: np.ndarray,
                         envelope_real: np.ndarray, envelope_imag: np.ndarray,
                         image_count: int | None, parity: int | None,
                         centroid: np.ndarray, other_centroid: np.ndarray,
                         corridor_half: float, boundary_theta: np.ndarray,
                         boundary_r: np.ndarray,
                         eta_overlap_min: float = _DEFAULT_CAUSTIC_FLOOR,
                         refused_points: np.ndarray | None = None,
                         envelope_definition: str
                         = _INTERIOR_ENVELOPE_DEFINITION
                         ) -> 'LobeInteriorChart':
        """Build a lobe-interior chart by fitting splines to a value tensor.

        Parameters
        ----------
        gamma_grid, rho_lobe_grid, theta_local_grid, log_w_grid : np.ndarray
            1-D strictly increasing training axes (the two spatial axes are
            the lobe-local ``rho_lobe`` and ``theta_local``).
        envelope_real, envelope_imag : np.ndarray
            Shape ``(n_w, n_gamma, n_rho_lobe, n_theta_local)`` real/imag
            envelope values (the ``tau_c``-demodulated interior label).
        image_count, parity : int or None
            Region labels (``None`` if unrecorded).
        centroid, other_centroid : np.ndarray
            ``(2,)`` this-lobe and other-lobe source-plane centroids.
        corridor_half : float
            Inter-lobe corridor half-width.
        boundary_theta, boundary_r : np.ndarray
            Directional lobe boundary nodes normalising ``rho_lobe``.
        eta_overlap_min : float, optional
            Minimum caustic distance served (default the caustic floor).
        refused_points : np.ndarray, optional
            Refused lobe-local ``(gamma, rho_lobe, theta_local)`` training
            points.
        envelope_definition : str, optional
            Tag naming the label the chart's envelope encodes (default the
            interior SACR-C label).
        """
        gamma_grid = _validate_axis(gamma_grid, 'gamma_grid')
        rho_lobe_grid = _validate_axis(rho_lobe_grid, 'rho_lobe_grid')
        theta_local_grid = _validate_axis(theta_local_grid,
                                          'theta_local_grid')
        log_w_grid = _validate_axis(log_w_grid, 'log_w_grid')
        expected = (log_w_grid.size, gamma_grid.size, rho_lobe_grid.size,
                    theta_local_grid.size)
        _check_value_shape(envelope_real, envelope_imag, expected)
        real_c, imag_c, knots = _fit_tensor_spline(
            (log_w_grid, gamma_grid, rho_lobe_grid, theta_local_grid),
            envelope_real, envelope_imag)
        return cls._assemble(
            gamma_grid, rho_lobe_grid, theta_local_grid, log_w_grid,
            real_c, imag_c, knots, image_count, parity, eta_overlap_min,
            refused_points, centroid, other_centroid, corridor_half,
            boundary_theta, boundary_r,
            envelope_definition=envelope_definition)

    @classmethod
    def _assemble(cls, gamma_grid, rho_lobe_grid, theta_local_grid, log_w_grid,
                  real_coeffs, imag_coeffs, knots, image_count, parity,
                  eta_overlap_min, refused_points, centroid, other_centroid,
                  corridor_half, boundary_theta, boundary_r,
                  envelope_definition=_INTERIOR_ENVELOPE_DEFINITION
                  ) -> 'LobeInteriorChart':
        """Assemble a lobe chart from prebuilt coefficient tensors and knots.

        Load-bearing for `_chart_from_npz`: rebuilds the frozen chart from
        the persisted axes, coefficients, knots and lobe frame without
        re-fitting.
        """
        param_spacing = np.array([
            float(np.mean(np.diff(gamma_grid))),
            float(np.mean(np.diff(rho_lobe_grid))),
            float(np.mean(np.diff(theta_local_grid)))])
        return cls(
            gamma_grid=_validate_axis(gamma_grid, 'gamma_grid'),
            rho_lobe_grid=_validate_axis(rho_lobe_grid, 'rho_lobe_grid'),
            theta_local_grid=_validate_axis(theta_local_grid,
                                            'theta_local_grid'),
            log_w_grid=_validate_axis(log_w_grid, 'log_w_grid'),
            real_coeffs=np.ascontiguousarray(real_coeffs, dtype=float),
            imag_coeffs=np.ascontiguousarray(imag_coeffs, dtype=float),
            knots=tuple(np.ascontiguousarray(t, dtype=float) for t in knots),
            image_count=None if image_count is None else int(image_count),
            parity=None if parity is None else int(parity),
            eta_overlap_min=float(eta_overlap_min),
            refused_points=_normalize_refused(refused_points),
            param_spacing=param_spacing,
            envelope_definition=str(envelope_definition),
            centroid=np.ascontiguousarray(centroid, dtype=float).reshape(2),
            other_centroid=np.ascontiguousarray(
                other_centroid, dtype=float).reshape(2),
            corridor_half=float(corridor_half),
            boundary_theta=np.ascontiguousarray(boundary_theta, dtype=float),
            boundary_r=np.ascontiguousarray(boundary_r, dtype=float))


@dataclass(frozen=True, eq=False)
class TubeChart:
    """Near-caustic envelope chart in caustic-adapted coordinates.

    Interpolates ``E(w)`` over ``(log w, gamma, u = sqrt(eta), theta)``,
    where ``eta`` is the source-plane distance to the caustic and
    ``theta`` its arc position.  Fitting in ``u = sqrt(eta)`` linearizes
    the fold's square-root branch so the interpolant is smooth through the
    near-caustic transition; the tube covers only the image-pair-present
    side ``eta > 0``.  ``theta`` is BOUNDED and NON-PERIODIC (a single
    inter-cusp fold arc); cusp neighbourhoods are excluded.

    Attributes
    ----------
    gamma_grid, u_grid, theta_grid, log_w_grid : np.ndarray
        1-D strictly increasing training axes (``u = sqrt(eta)``).
    real_coeffs, imag_coeffs : np.ndarray
        Cubic B-spline coefficient tensors, axes ``(log w, gamma, u,
        theta)``.
    knots : tuple of np.ndarray
        Knot vectors ``(t_logw, t_gamma, t_u, t_theta)``.
    image_count : int or None
        Real-image count of the chart's region.
    parity : int or None
        Macro-image parity ``+1``/``-1``.
    eta_floor, eta_max : float
        Caustic-distance band the tube serves ``[eta_floor, eta_max]``;
        below ``eta_floor`` the fold cusps sharpen and the query falls
        through to the exact engine.
    cusp_windows : tuple of (float, float)
        ``(theta_cusp, delta_theta)`` exclusion windows: a query with
        ``|theta - theta_cusp| < delta_theta`` falls through.
    """

    gamma_grid: np.ndarray
    u_grid: np.ndarray
    theta_grid: np.ndarray
    log_w_grid: np.ndarray
    real_coeffs: np.ndarray
    imag_coeffs: np.ndarray
    knots: tuple
    image_count: int | None
    parity: int | None
    eta_floor: float
    eta_max: float
    cusp_windows: tuple

    @classmethod
    def from_values(cls, *, gamma_grid: np.ndarray, u_grid: np.ndarray,
                    theta_grid: np.ndarray, log_w_grid: np.ndarray,
                    envelope_real: np.ndarray, envelope_imag: np.ndarray,
                    image_count: int | None, parity: int | None,
                    eta_floor: float, eta_max: float,
                    cusp_windows: tuple | None = None) -> 'TubeChart':
        """Build a tube chart by fitting splines to a value tensor.

        Parameters
        ----------
        gamma_grid, u_grid, theta_grid, log_w_grid : np.ndarray
            1-D strictly increasing training axes (``u = sqrt(eta)``).
        envelope_real, envelope_imag : np.ndarray
            Shape ``(n_w, n_gamma, n_u, n_theta)`` real/imag envelope
            values.
        image_count, parity : int or None
            Region labels.
        eta_floor, eta_max : float
            Caustic-distance band served ``[eta_floor, eta_max]``.
        cusp_windows : tuple of (float, float), optional
            ``(theta_cusp, delta_theta)`` exclusion windows.
        """
        gamma_grid = _validate_axis(gamma_grid, 'gamma_grid')
        u_grid = _validate_axis(u_grid, 'u_grid')
        theta_grid = _validate_axis(theta_grid, 'theta_grid')
        log_w_grid = _validate_axis(log_w_grid, 'log_w_grid')
        expected = (log_w_grid.size, gamma_grid.size, u_grid.size,
                    theta_grid.size)
        _check_value_shape(envelope_real, envelope_imag, expected)
        real_c, imag_c, knots = _fit_tensor_spline(
            (log_w_grid, gamma_grid, u_grid, theta_grid),
            envelope_real, envelope_imag)
        return cls._assemble(
            gamma_grid, u_grid, theta_grid, log_w_grid, real_c, imag_c, knots,
            image_count, parity, eta_floor, eta_max, cusp_windows)

    @classmethod
    def _assemble(cls, gamma_grid, u_grid, theta_grid, log_w_grid,
                  real_coeffs, imag_coeffs, knots, image_count, parity,
                  eta_floor, eta_max, cusp_windows) -> 'TubeChart':
        """Assemble a chart from prebuilt coefficient tensors and knots."""
        windows = tuple((float(tc), float(dt))
                        for tc, dt in (cusp_windows or ()))
        return cls(
            gamma_grid=_validate_axis(gamma_grid, 'gamma_grid'),
            u_grid=_validate_axis(u_grid, 'u_grid'),
            theta_grid=_validate_axis(theta_grid, 'theta_grid'),
            log_w_grid=_validate_axis(log_w_grid, 'log_w_grid'),
            real_coeffs=np.ascontiguousarray(real_coeffs, dtype=float),
            imag_coeffs=np.ascontiguousarray(imag_coeffs, dtype=float),
            knots=tuple(np.ascontiguousarray(t, dtype=float) for t in knots),
            image_count=None if image_count is None else int(image_count),
            parity=None if parity is None else int(parity),
            eta_floor=float(eta_floor),
            eta_max=float(eta_max),
            cusp_windows=windows)


def _check_value_shape(value_real: np.ndarray, value_imag: np.ndarray,
                       expected: tuple[int, int, int, int]) -> None:
    """Raise if the real/imag value tensors do not match ``expected``."""
    for name, array in (('envelope_real', value_real),
                        ('envelope_imag', value_imag)):
        if np.shape(array) != expected:
            raise ValueError(
                f'{name} has shape {np.shape(array)}; expected {expected} '
                f'from the training grids.')


# ---- Chart selection (deterministic guard stack) ----------------------


def _in_exclusion_ball(chart: 'FarFieldChart | LobeInteriorChart',
                       gamma: float, rho: float, theta_c: float) -> bool:
    """Whether the chart's spatial ``(gamma, rho, theta_c)`` is within a
    refusal ball.

    ``refused_points`` and ``param_spacing`` are both in the chart's own
    spatial coordinate -- caustic-fixed ``(gamma, rho, theta_c)`` for a
    `FarFieldChart` (Build 8h-b3) or lobe-local ``(gamma, rho_lobe,
    theta_local)`` for a `LobeInteriorChart`, which shares this exact
    normalized-ball form -- so the test is coordinate-agnostic.  Tiles are
    sub-arcs in the angular axis (they never wrap), so no angular-wrap
    handling is needed: refused points and queries share the tile's range.
    """
    refused = chart.refused_points
    if refused.shape[0] == 0:
        return False
    query = np.array([gamma, rho, theta_c])
    normalized = (refused - query) / chart.param_spacing
    distances = np.sqrt(np.sum(normalized ** 2, axis=1))
    return bool(np.min(distances) <= _EXCLUSION_RADIUS)


def _log_w_band_inside(chart, log_w_min: float, log_w_max: float) -> bool:
    """Whether the query ``ln w`` band lies inside the chart's ``w`` band."""
    return (chart.log_w_grid[0] <= log_w_min
            and log_w_max <= chart.log_w_grid[-1])


def _theta_into_frame(theta: float, frame_lo: float) -> float:
    """Unwrap a ``[0, 2*pi)`` caustic angle into a chart's theta frame.

    `geometry.nearest_caustic_point` reports ``theta`` in ``[0, 2*pi)``,
    while a chart's ``theta_grid`` lives in the arc's wedge frame and may
    span negative angles (e.g. a deltoid arc at ``[-0.39, -0.09]``).  The
    two differ by a multiple of ``2*pi`` for the same physical angle, so
    every range test, cusp-window test and spline coordinate must first
    map the query into the chart's frame.
    """
    return frame_lo + (theta - frame_lo) % (2.0 * np.pi)


def _tube_serves(chart: TubeChart, gamma: float, log_w_min: float,
                 log_w_max: float, eta: float, theta: float,
                 image_count: int) -> bool:
    """Whether a tube chart serves this candidate (guard-stack steps 1,5,6)."""
    # (1) certified-box containment on gamma and log w.
    if not (chart.gamma_grid[0] <= gamma <= chart.gamma_grid[-1]):
        return False
    if not _log_w_band_inside(chart, log_w_min, log_w_max):
        return False
    # (5) image-count guard.
    if chart.image_count is not None and image_count != chart.image_count:
        return False
    # (6) cusp exclusion / eta floor and (7) tube band membership.
    if not (chart.eta_floor <= eta <= chart.eta_max):
        return False
    u = float(np.sqrt(eta))
    if not (chart.u_grid[0] <= u <= chart.u_grid[-1]):
        return False
    theta = _theta_into_frame(theta, float(chart.theta_grid[0]))
    if not (chart.theta_grid[0] <= theta <= chart.theta_grid[-1]):
        return False
    two_pi = 2.0 * np.pi
    for theta_cusp, delta_theta in chart.cusp_windows:
        # Shrink each 8c cusp-exclusion window to the COMPLEMENT of the
        # Pearcey arm's certified angular coverage: the arm serves the
        # outer part of the cusp neighbourhood, so only the residual
        # near-vertex core still excludes the tube.  A query inside this
        # residual window returns False and falls through to the arm, then
        # the engine.  ``_CUSP_ARM_COVERAGE`` defaults to 0.0, so the
        # window is unchanged (byte-identical) until the census pins a
        # nonzero coverage.  The chart schema is untouched -- the shrink is
        # applied at query time from the module constant, not stored.
        residual = max(0.0, delta_theta - _CUSP_ARM_COVERAGE)
        if abs((theta - theta_cusp + np.pi) % two_pi - np.pi) < residual:
            return False
    return True


def _farfield_serves(chart: FarFieldChart, gamma: float, log_w_min: float,
                     log_w_max: float, eta: float, image_count: int,
                     rho: float, theta_c: float) -> bool:
    """Whether a far-field chart serves this candidate (steps 1,3,5,7).

    The source containment test is in the chart's caustic-fixed
    ``(rho, theta_c)`` axes. The caller and trainer both route through
    `_to_caustic_fixed` / `_from_caustic_fixed`, so the piecewise
    positive-parity coordinate (and the saddle scalar fallback) agree.
    """
    # (1) certified-box containment on gamma, log w, and the source.
    if not (chart.gamma_grid[0] <= gamma <= chart.gamma_grid[-1]):
        return False
    if not _log_w_band_inside(chart, log_w_min, log_w_max):
        return False
    if not (chart.rho_grid[0] <= rho <= chart.rho_grid[-1]
            and chart.theta_c_grid[0] <= theta_c <= chart.theta_c_grid[-1]):
        return False
    # (3) inherited engine-refusal exclusion ball.
    if _in_exclusion_ball(chart, gamma, rho, theta_c):
        return False
    # (5) image-count guard.
    if chart.image_count is not None and image_count != chart.image_count:
        return False
    # (7) far-field priority: only away from the caustic.
    if eta <= chart.eta_overlap_min:
        return False
    return True


def _lobe_serves(chart: 'LobeInteriorChart', gamma: float, log_w_min: float,
                 log_w_max: float, eta: float, image_count: int,
                 y1_eig: float, y2_eig: float) -> bool:
    """Whether a macro-saddle lobe-interior chart serves this candidate.

    A SEPARATE guard from `_farfield_serves`: the far-field chart lives in
    origin-centred caustic-fixed axes, whereas a lobe chart lives in
    LOBE-LOCAL ``(rho_lobe, theta_local)`` axes centred on its own
    source-plane deltoid centroid, so the containment test must first place
    the eigenframe source in the chart's stored frame.

    Gate order (Professor Q1/Q5), each an independently observable abstention
    reason: (a) gamma box containment; (b) ``ln w`` band inside; (c) the
    inter-lobe CORRIDOR test in the chart's stored frame -- this lobe serves
    only when the query is closer to THIS centroid than to the other lobe's
    by the corridor half-width margin, so a source in the inter-lobe corridor
    fails (c) for BOTH lobes and no lobe chart serves it (the documented,
    named fall-through to the exact-engine ladder, Professor Q5); (d)
    lobe-local box containment on ``(rho_lobe, theta_local)``; (e) the
    inherited engine-refusal exclusion balls in lobe-local coordinates; (f)
    the image-count guard; (g) the interior ``eta`` floor.  The frozen 9-probe
    winding admission is NOT replicated here (Professor Q1 + Simplifier): its
    no-false-admit guarantee is inherited via the identical interpolated
    ``r_deltoid`` boundary used to normalise ``rho_lobe``.

    Parameters
    ----------
    chart : LobeInteriorChart
        The lobe-interior chart under test.
    gamma : float
        External shear magnitude (``> 1`` for a macro-saddle lobe).
    log_w_min, log_w_max : float
        Bounds of the query's ``ln w`` band.
    eta : float
        Caustic distance ``partition.caustic_distance``.
    image_count : int
        Real-image count ``int(partition.real_mask.sum())``.
    y1_eig, y2_eig : float
        Eigenframe source position (dimensionless); placed in the chart's
        lobe-local frame for the corridor and box-containment tests.

    Returns
    -------
    bool
        ``True`` when this lobe chart serves the candidate.
    """
    # Precondition: a usable eigenframe source.  A non-finite coordinate
    # (e.g. a caller that did not thread the source) declines cleanly rather
    # than relying on NaN comparison semantics downstream.
    if not (np.isfinite(y1_eig) and np.isfinite(y2_eig)):
        return False
    # (a) certified gamma box containment.
    if not (chart.gamma_grid[0] <= gamma <= chart.gamma_grid[-1]):
        return False
    # (b) ln w band inside.
    if not _log_w_band_inside(chart, log_w_min, log_w_max):
        return False
    # (c) inter-lobe corridor test in the chart's stored source-plane frame:
    # serve only when strictly closer to THIS centroid than to the other by
    # the corridor half-width.  A corridor source fails this for BOTH lobes.
    near_this = float(np.hypot(y1_eig - float(chart.centroid[0]),
                               y2_eig - float(chart.centroid[1])))
    near_other = float(np.hypot(y1_eig - float(chart.other_centroid[0]),
                                y2_eig - float(chart.other_centroid[1])))
    if near_this + chart.corridor_half > near_other:
        return False
    # (d) lobe-local box containment.  A query exactly at the centroid has an
    # undefined theta_local (`_to_lobe_fixed` raises); that degenerate point
    # is refused rather than served an arbitrary angle (Professor Q4).
    try:
        rho_lobe, theta_local = _to_lobe_fixed(
            chart.centroid, chart.boundary_theta, chart.boundary_r,
            y1_eig, y2_eig)
    except ValueError:
        return False
    if not (chart.rho_lobe_grid[0] <= rho_lobe <= chart.rho_lobe_grid[-1]
            and chart.theta_local_grid[0] <= theta_local
            <= chart.theta_local_grid[-1]):
        return False
    # (e) inherited engine-refusal exclusion ball, in lobe-local coordinates
    # (refused_points and param_spacing are both lobe-local for this chart).
    if _in_exclusion_ball(chart, gamma, rho_lobe, theta_local):
        return False
    # (f) image-count guard.
    if chart.image_count is not None and image_count != chart.image_count:
        return False
    # (g) interior eta floor.
    if eta <= chart.eta_overlap_min:
        return False
    return True


def select_chart(charts, *, gamma: float, log_w_min: float, log_w_max: float,
                 eta: float, theta: float, image_count: int, rho: float,
                 theta_c: float, y1_eig: float = float('nan'),
                 y2_eig: float = float('nan')):
    """Deterministically pick the chart to serve a candidate, or ``None``.

    The guard stack (Professor Q7), executed in order, keys ONLY on the
    certified physical quantities ``gamma``, ``eta`` and ``image_count``
    -- never on the gauge angle ``theta`` except for the cusp-window
    exclusion test (F017).  Any fall-through returns ``None`` so the
    caller uses the exact engine.

    Order: (2) gamma guard band near ``gamma = 1`` -> fall through; then
    TUBE charts have priority over FAR-FIELD charts, and FAR-FIELD over
    LOBE-INTERIOR charts (step 7).  Box containment is mutually exclusive
    across the three kinds -- a positive-parity far-field box and a
    macro-saddle lobe box never overlap in ``gamma`` -- so the scan order
    is deterministic, not arbitrating overlap.  Per chart: (1) certified-box
    containment on ``(gamma, log w)``; (3) far-field engine-refusal exclusion
    balls; (5) image-count match; (6) cusp exclusion / ``eta`` floor; (7)
    tube when ``eta in [eta_floor, eta_max]``, else far-field when ``eta >
    eta_overlap_min``, else a lobe chart when the source falls in that lobe.

    Parameters
    ----------
    charts : sequence of TubeChart, FarFieldChart or LobeInteriorChart
        The surrogate's charts.
    gamma : float
        External shear magnitude.
    log_w_min, log_w_max : float
        Bounds of the query's ``ln w`` band.
    eta : float
        Caustic distance ``partition.caustic_distance``.
    theta : float
        Caustic arc angle ``partition.caustic_theta`` (gauge; used only
        for the cusp-window test).
    image_count : int
        Real-image count ``int(partition.real_mask.sum())``.
    rho, theta_c : float
        Piecewise caustic-fixed source coordinate and eigenframe polar angle
        from `_to_caustic_fixed` (the far-field query axes).
    y1_eig, y2_eig : float, optional
        Eigenframe source position, threaded to the lobe-interior guard
        `_lobe_serves` so a macro-saddle lobe chart can place the query in
        its own lobe-local frame.  Defaults are non-finite: a caller that
        does not thread the source declines every lobe chart cleanly (only
        the far-field/tube dispatch is exercised), preserving the legacy
        call sites unchanged.

    Returns
    -------
    TubeChart, FarFieldChart, LobeInteriorChart or None
        The selected chart, or ``None`` to fall through to the engine.
    """
    # (2) gamma guard band around the det-A = 0 parity boundary.
    if abs(gamma - 1.0) < _GAMMA_GUARD_BAND:
        return None
    # (7) priority: tube charts first, then far-field, then lobe-interior.
    for chart in charts:
        if isinstance(chart, TubeChart) and _tube_serves(
                chart, gamma, log_w_min, log_w_max, eta, theta, image_count):
            return chart
    for chart in charts:
        if isinstance(chart, FarFieldChart) and _farfield_serves(
                chart, gamma, log_w_min, log_w_max, eta, image_count,
                rho, theta_c):
            return chart
    for chart in charts:
        if isinstance(chart, LobeInteriorChart) and _lobe_serves(
                chart, gamma, log_w_min, log_w_max, eta, image_count,
                y1_eig, y2_eig):
            return chart
    return None


def _evaluate_chart(chart, gamma: float, rho: float, theta_c: float,
                    eta: float, theta: float,
                    log_w_query: np.ndarray,
                    y1_eig: float = float('nan'),
                    y2_eig: float = float('nan')) -> np.ndarray:
    """Evaluate the selected chart's complex envelope over ``log_w_query``.

    A tube chart contracts on ``(sqrt(eta), theta-into-frame)``; a
    far-field chart contracts on its caustic-fixed spatial axes
    ``(rho, theta_c)`` (Build 8h-b3); a lobe-interior chart contracts on
    the LOBE-LOCAL ``(rho_lobe, theta_local)`` computed from the chart's
    OWN stored frame via `_to_lobe_fixed` (so the eigenframe source is
    placed at its true lobe-local position).  ``y1_eig`` / ``y2_eig`` are
    required for a `LobeInteriorChart` and ignored for the other kinds;
    their non-finite defaults keep the legacy far-field/tube call sites
    unchanged.
    """
    if isinstance(chart, TubeChart):
        v1 = float(np.sqrt(eta))
        v2 = _theta_into_frame(theta, float(chart.theta_grid[0]))
    elif isinstance(chart, LobeInteriorChart):
        v1, v2 = _to_lobe_fixed(chart.centroid, chart.boundary_theta,
                                chart.boundary_r, y1_eig, y2_eig)
    else:
        v1, v2 = rho, theta_c
    real = _contract_tensor_spline(chart.real_coeffs, chart.knots,
                                   gamma, v1, v2, log_w_query)
    imag = _contract_tensor_spline(chart.imag_coeffs, chart.knots,
                                   gamma, v1, v2, log_w_query)
    return real + 1j * imag


class LensAmplificationSurrogate:
    """Global multi-chart tensor-cubic-spline emulator of ``E(w)``.

    Holds a flat list of `FarFieldChart` and `TubeChart` objects plus a
    provenance dict.  Serve the full guard stack with `serve` (the query
    the likelihood uses, fed the geometry partition's ``eta``, ``theta``
    and image count); the legacy single-box `envelope` / `in_domain`
    (raw-eigenframe far-field lookup only) are preserved for the 8a API.

    Construct via `from_engine` (single-box far-field training) or `load`
    (deserialize a saved artifact).  For direct multi-chart construction
    build charts with `FarFieldChart.from_values` / `TubeChart.from_values`
    and pass the list here.

    Parameters
    ----------
    charts : sequence of TubeChart or FarFieldChart
        The surrogate's charts (at least one).
    provenance : dict
        Training metadata (grid spec, engine version, training hash,
        prior-box definition, per-chart exclusion data, chart
        count/types).  Stored verbatim and re-serialized on `save`.

    Raises
    ------
    ValueError
        If ``charts`` is empty or holds an object that is neither a
        `FarFieldChart` nor a `TubeChart`.
    """

    def __init__(self, charts, provenance: dict) -> None:
        charts = list(charts)
        if not charts:
            raise ValueError('A surrogate needs at least one chart.')
        for chart in charts:
            if not isinstance(chart, (FarFieldChart, TubeChart,
                                      LobeInteriorChart)):
                raise ValueError(
                    'charts must be FarFieldChart, TubeChart or '
                    'LobeInteriorChart instances; '
                    f'got {type(chart).__name__}.')
        self.charts = charts
        self.provenance = dict(provenance)

    # ---- Backward-compatible single-box attribute shims ---------------
    # The 8a API exposed the box axes directly.  For a single-chart
    # surrogate these delegate to that chart; multi-chart callers use the
    # per-chart axes / `serve` instead.

    @property
    def gamma_grid(self) -> np.ndarray:
        """External-shear axis of the first chart (8a shim)."""
        return self.charts[0].gamma_grid

    @property
    def log_w_grid(self) -> np.ndarray:
        """``ln w`` axis of the first chart (8a shim)."""
        return self.charts[0].log_w_grid

    @property
    def rho_grid(self) -> np.ndarray:
        """Caustic-fixed ``rho`` axis of the first (far-field) chart (8a
        shim)."""
        return self.charts[0].rho_grid

    @property
    def theta_c_grid(self) -> np.ndarray:
        """Caustic-fixed ``theta_c`` axis of the first (far-field) chart."""
        return self.charts[0].theta_c_grid

    @property
    def refused_points(self) -> np.ndarray:
        """Refused training points of the first (far-field) chart (8a shim)."""
        return self.charts[0].refused_points

    # ---- Construction from the exact engine ---------------------------

    @classmethod
    def from_engine(cls, *, gamma_range: tuple[float, float],
                    rho_range: tuple[float, float],
                    theta_c_range: tuple[float, float],
                    w_range: tuple[float, float],
                    n_gamma: int = _DEFAULT_PARAM_NODES,
                    n_rho: int = _DEFAULT_PARAM_NODES,
                    n_theta: int = _DEFAULT_PARAM_NODES,
                    w_nodes_per_decade: int = _DEFAULT_W_NODES_PER_DECADE,
                    max_order: int | None = None,
                    definition: str = _FARFIELD_ENVELOPE_DEFINITION
                    ) -> 'LensAmplificationSurrogate':
        """Train a single-box far-field surrogate on a dense engine grid.

        Evaluates `ChangRefsdalChannels.evaluate` at ``beta = 0``,
        ``kappa = 0`` on the full dense ``w`` grid for every parameter
        grid point (no LOO / adaptive logic -- unlimited offline engine
        calls), taking the far-field label
        ``E_ff = F - sum_{a real} H_a e^{1j w tau_a}``
        (`farfield_envelope_from_partition`) rather than the caustic-region
        ``partition.envelope`` -- in the exterior the switch/carrier the
        caustic-region envelope needs flip lobes on the astroid diagonals
        and leave a resolved image un-subtracted (Build 8g-b).

        Interior charts (Build S2-3).  When ``definition`` is the interior
        SACR-C tag (`_INTERIOR_ENVELOPE_DEFINITION`), the node value is
        instead the caustic-region ``partition.envelope`` -- the full
        ``tau_c``-demodulated SACR-C envelope ``E`` with the switch ON --
        because INSIDE the caustic the far-field label subtracts the
        near-merged image kernels (which individually diverge as their
        separation from the critical carrier shrinks) and fails
        generically.  The SACR-C label switches that pair INTO the bounded
        envelope instead.  The coordinate is unchanged (caustic-fixed
        ``(rho, theta_c)``); only the ENVELOPE LABEL differs, and the tag
        is stamped on the chart so the serve mirror dispatches the SACR-C
        reconstruction (`channels.reconstruct_from_envelope`).  Interior
        node builds also collect the parked carrier ``critical_source`` and
        assert basin continuity across the tile (`_assert_carrier_continuity`),
        so a tile straddling a medial ridge is rejected for subdivision
        rather than fitted with a phase-kinked envelope.

        Caustic-fixed grid (Build 8h-b3). The two spatial axes are the
        caustic-fixed coordinates ``rho`` and ``theta_c``; each positive-
        parity grid node ``(gamma, rho, theta_c)`` is mapped to a physical
        eigenframe source via `_from_caustic_fixed` BEFORE the engine call.
        Thus the caustic is the fixed surface ``rho = 1`` while the exterior
        coordinate remains an additive physical radial offset. Macro-saddle
        exterior
        charts use the documented scalar fallback because the disconnected
        deltoids have no origin-centred directional radius on every ray. A
        parameter
        point that refuses at any ``w`` node (or returns a non-finite
        envelope) is recorded as refused (in caustic-fixed coordinates) and
        left as zeros in the value arrays.

        Domain contract (exterior-only): the far-field label subtracts the
        resolved geometric-optics images with the switch forced on for
        every real channel, so it is small and smooth ONLY where the box
        lies wholly in the caustic EXTERIOR (``rho > 1`` with the exterior
        exclusion margin).  Near the caustic an image is not fully
        resolved; forcing its switch on leaves an un-subtracted oscillatory
        term, ``E_ff`` grows and a coarse spline fits it poorly.
        Near-caustic domains therefore belong to TUBE charts, not to a
        far-field chart built here; production tiling enforces this by
        admitting only tiles wholly outside the caustic disk.  This method
        applies the far-field label unconditionally and does NOT itself
        guard the exterior contract -- callers must supply an exterior box.

        Parameters
        ----------
        gamma_range : tuple[float, float]
            External-shear axis bounds ``(low, high)``.
        rho_range, theta_c_range : tuple[float, float]
            Caustic-fixed spatial axis bounds ``(low, high)``: at positive
            parity ``rho`` is a directional ratio below one and an additive
            physical radial offset above one; ``theta_c`` is the eigenframe
            polar angle (radians). Macro-saddle exterior charts use the
            scalar-reach fallback.
        w_range : tuple[float, float]
            Dimensionless-frequency bounds ``(w_min, w_max)``, both
            strictly positive.
        n_gamma, n_rho, n_theta : int, optional
            Nodes per parameter axis (default 7; Professor Q2 sizing).
        w_nodes_per_decade : int, optional
            Density of the dense log-w training axis (default 15).
        max_order : int, optional
            Operator-series order cap forwarded to `ChangRefsdalChannels`.
        definition : str, optional
            Envelope-definition tag the chart is trained on (default the
            far-field kernel-sum label).  Pass the interior SACR-C tag
            (`_INTERIOR_ENVELOPE_DEFINITION`) to build a whole-interior
            chart on the caustic-region ``partition.envelope`` (Build
            S2-3).
        Returns
        -------
        LensAmplificationSurrogate
            The trained single-chart surrogate.

        Raises
        ------
        ValueError
            If ``definition`` is not a known envelope-definition tag.
        CarrierDiscontinuityError
            For an interior chart whose tile straddles a critical-basin
            flip, or for an exterior chart whose frame-invariant far-field
            label winds faster than the Nyquist ``pi/2`` per node gap
            (`_assert_farfield_carrier_continuity`); the tile must be
            subdivided.
        """
        definition = _validate_farfield_definition(definition, 'chart build')
        interior = definition in _KNOWN_INTERIOR_DEFINITIONS
        log_w_grid = _log_w_grid(w_range, w_nodes_per_decade)
        gamma_grid = _uniform_axis(gamma_range, n_gamma, 'gamma')
        rho_grid = _uniform_axis(rho_range, n_rho, 'rho')
        theta_c_grid = _uniform_axis(theta_c_range, n_theta, 'theta_c')
        # Positive-parity charts: union exact spline nodes onto the
        # source-plane astroid cusp angles {0, +/-pi/2, pi} that fall in
        # range, so cusp columns (C2 curvature kinks) are charted ON a node
        # rather than smoothed across.  Parity is deterministic in the box-
        # centre ``gamma`` (mirrors `_box_region_labels`: +1 below the
        # ``gamma = 1`` parity wall).  The macro-saddle path keeps the plain
        # uniform axis -- its disconnected deltoids have no single origin-
        # centred cusp-angle set.
        gamma_mid = 0.5 * float(gamma_grid[0] + gamma_grid[-1])
        if gamma_mid < 1.0:
            theta_c_grid = _union_cusp_nodes(theta_c_grid, theta_c_range)
        w_grid = np.exp(log_w_grid)

        shape = (log_w_grid.size, gamma_grid.size, rho_grid.size,
                 theta_c_grid.size)
        envelope_real = np.zeros(shape, dtype=float)
        envelope_imag = np.zeros(shape, dtype=float)
        refused: list[tuple[float, float, float]] = []
        # Parked-carrier ``critical_source`` per node (interior only), for
        # the basin-continuity guard; NaN marks a refused/unfilled node.
        carrier = np.full((gamma_grid.size, rho_grid.size,
                           theta_c_grid.size, 2), np.nan, dtype=float)

        channels_kwargs = {} if max_order is None else {'max_order': max_order}
        for i_g, gamma in enumerate(gamma_grid):
            for i_rho, rho in enumerate(rho_grid):
                for i_th, theta_c in enumerate(theta_c_grid):
                    # Fresh tracker per point -> deterministic initial
                    # labeling; the envelope is well-defined per point and
                    # independent of label continuation.
                    channels = ChangRefsdalChannels(w_grid, **channels_kwargs)
                    try:
                        # Caustic-fixed node -> physical eigenframe source.
                        # This conversion calls `_caustic_reach`, which
                        # raises `LensDomainError` at the ``gamma = 1``
                        # parity wall, so it must sit INSIDE the refusal
                        # guard: such a node is recorded refused (the
                        # documented `from_engine` contract) instead of
                        # crashing chart construction.
                        y1_eig, y2_eig = _from_caustic_fixed(
                            float(gamma), float(rho), float(theta_c))
                        partition = channels.evaluate(
                            gamma=float(gamma),
                            y=(y1_eig, y2_eig),
                            beta=0.0, kappa=0.0)
                    except _REFUSAL_ERRORS:
                        refused.append((float(gamma), float(rho),
                                        float(theta_c)))
                        continue
                    # Interior tiles store the SACR-C ``tau_c``-demodulated
                    # envelope (switch ON, near-merged images switched in);
                    # exterior tiles store the far-field kernel-sum label.
                    env = (partition.envelope if interior
                           else farfield_envelope_from_partition(
                               partition, definition))
                    if not np.all(np.isfinite(env)):
                        # Conservative: a non-finite envelope is treated as
                        # a refusal rather than served as a value (F005).
                        refused.append((float(gamma), float(rho),
                                        float(theta_c)))
                        continue
                    envelope_real[:, i_g, i_rho, i_th] = env.real
                    envelope_imag[:, i_g, i_rho, i_th] = env.imag
                    carrier[i_g, i_rho, i_th] = partition.critical_source

        if interior:
            # Interpolator hygiene: the tau_c-demodulated envelope is smooth
            # only within one nearest-caustic basin.  Reject a tile that
            # straddles a medial ridge (basin flip) for subdivision.
            _assert_carrier_continuity(
                carrier, gamma_grid,
                (gamma_grid.size, rho_grid.size, theta_c_grid.size))
        else:
            # Interpolator hygiene, exterior twin: the stored far-field label
            # is the frame-invariant demodulated ``E_tilde``.  Reject for
            # subdivision a tile whose label JUMPS by more than the chart's
            # peak magnitude across one node gap -- gross aliasing a cubic
            # spline cannot represent.  Measured as a re/im increment, not a
            # phase step, so amplitude nulls (where ``arg`` swings but the
            # splined fields stay smooth) are not false-positived into
            # ladder-served gaps (F022).
            _assert_farfield_carrier_continuity(
                envelope_real + 1j * envelope_imag, float(w_grid[-1]),
                gamma_grid,
                (gamma_grid.size, rho_grid.size, theta_c_grid.size))

        refused_points = (np.array(refused, dtype=float) if refused
                          else np.empty((0, 3), dtype=float))
        image_count, parity = cls._box_region_labels(gamma_grid, rho_grid,
                                                      theta_c_grid)
        chart = FarFieldChart.from_values(
            gamma_grid=gamma_grid, rho_grid=rho_grid,
            theta_c_grid=theta_c_grid, log_w_grid=log_w_grid,
            envelope_real=envelope_real, envelope_imag=envelope_imag,
            image_count=image_count, parity=parity,
            eta_overlap_min=_DEFAULT_CAUSTIC_FLOOR,
            refused_points=refused_points,
            envelope_definition=definition)
        provenance = cls._build_provenance(
            gamma_range, rho_range, theta_c_range, w_range, shape,
            envelope_real, envelope_imag)
        return cls([chart], provenance)

    @classmethod
    def from_lobe_engine(cls, *, admission: '_SaddleLobeAdmission',
                         gamma_range: tuple[float, float],
                         rho_lobe_range: tuple[float, float],
                         theta_local_range: tuple[float, float],
                         w_range: tuple[float, float],
                         n_gamma: int = _DEFAULT_PARAM_NODES,
                         n_rho: int = _DEFAULT_PARAM_NODES,
                         n_theta: int = _DEFAULT_PARAM_NODES,
                         w_nodes_per_decade: int = _DEFAULT_W_NODES_PER_DECADE,
                         max_order: int | None = None
                         ) -> 'LensAmplificationSurrogate':
        """Train a macro-saddle lobe-interior surrogate on a dense engine grid.

        The macro-saddle (``gamma > 1``) counterpart of `from_engine`'s
        interior branch, but in LOBE-LOCAL coordinates.  For each grid node
        ``(gamma, rho_lobe, theta_local)`` the lobe frame maps the node to a
        physical eigenframe source via `_from_lobe_fixed` (NOT the
        origin-centred `_from_caustic_fixed`), evaluates
        `ChangRefsdalChannels.evaluate` at ``beta = 0``, ``kappa = 0`` on the
        full dense ``w`` grid, and stores the ``tau_c``-demodulated
        INTERIOR_SACR_C ``partition.envelope`` (the interior label).  A node
        that refuses at any ``w`` or returns a non-finite envelope is recorded
        refused (in lobe-local coordinates) and left as zeros.

        Image count (Professor Q2).  The lobe interior is a single
        four-real-image region: the image count is read from
        ``partition.real_mask.sum()`` at the FIRST successful node and asserted
        equal to `_MACRO_SADDLE_IMAGE_COUNT`; a later node reporting a
        different count straddles a region boundary and is recorded refused
        rather than fitted.  Interior interpolator hygiene: the collected
        parked-carrier ``critical_source`` grid is passed to the SAME
        `_assert_carrier_continuity` guard the interior branch of `from_engine`
        uses (unchanged, F022), so a tile straddling a critical-basin flip is
        rejected for subdivision.

        The lobe frame (`centroid`, `other_centroid`, `corridor_half`,
        `boundary_theta`, `boundary_r`) is read straight off the passed
        `_SaddleLobeAdmission` and persisted on the `LobeInteriorChart` so the
        node maps to its true physical source at serve time.

        Parameters
        ----------
        admission : _SaddleLobeAdmission
            The frozen per-lobe admission carrying the lobe frame.
        gamma_range : tuple[float, float]
            External-shear axis bounds ``(low, high)``; both above one for a
            macro-saddle lobe.
        rho_lobe_range, theta_local_range : tuple[float, float]
            Lobe-local spatial axis bounds ``(low, high)``: ``rho_lobe`` is
            the directional radius ratio (``rho_lobe = 1`` on the deltoid
            boundary) and ``theta_local`` the lobe-local polar angle (radians).
        w_range : tuple[float, float]
            Dimensionless-frequency bounds ``(w_min, w_max)``, both positive.
        n_gamma, n_rho, n_theta : int, optional
            Nodes per parameter axis (default 7).
        w_nodes_per_decade : int, optional
            Density of the dense log-w training axis (default 15).
        max_order : int, optional
            Operator-series order cap forwarded to `ChangRefsdalChannels`.

        Returns
        -------
        LensAmplificationSurrogate
            The trained single-chart lobe-interior surrogate.

        Raises
        ------
        ValueError
            If the first evaluated node is not a
            `_MACRO_SADDLE_IMAGE_COUNT`-image lobe interior.
        CarrierDiscontinuityError
            If a lobe tile straddles a critical-basin flip
            (`_assert_carrier_continuity`); the tile must be subdivided.
        """
        centroid = np.ascontiguousarray(
            admission.centroid, dtype=float).reshape(2)
        other_centroid = np.ascontiguousarray(
            admission.other_centroid, dtype=float).reshape(2)
        corridor_half = float(admission.corridor_half)
        boundary_theta = np.ascontiguousarray(
            admission.boundary_theta, dtype=float)
        boundary_r = np.ascontiguousarray(admission.boundary_r, dtype=float)

        log_w_grid = _log_w_grid(w_range, w_nodes_per_decade)
        gamma_grid = _uniform_axis(gamma_range, n_gamma, 'gamma')
        rho_lobe_grid = _uniform_axis(rho_lobe_range, n_rho, 'rho_lobe')
        theta_local_grid = _uniform_axis(
            theta_local_range, n_theta, 'theta_local')
        w_grid = np.exp(log_w_grid)

        shape = (log_w_grid.size, gamma_grid.size, rho_lobe_grid.size,
                 theta_local_grid.size)
        envelope_real = np.zeros(shape, dtype=float)
        envelope_imag = np.zeros(shape, dtype=float)
        refused: list[tuple[float, float, float]] = []
        # Parked-carrier ``critical_source`` per node for the basin-continuity
        # guard; NaN marks a refused/unfilled node.
        carrier = np.full((gamma_grid.size, rho_lobe_grid.size,
                           theta_local_grid.size, 2), np.nan, dtype=float)
        image_count: int | None = None

        channels_kwargs = {} if max_order is None else {'max_order': max_order}
        for i_g, gamma in enumerate(gamma_grid):
            for i_rho, rho_lobe in enumerate(rho_lobe_grid):
                for i_th, theta_local in enumerate(theta_local_grid):
                    channels = ChangRefsdalChannels(w_grid, **channels_kwargs)
                    try:
                        # Lobe-local node -> physical eigenframe source (NOT
                        # origin-centred). Inside the refusal guard so an
                        # engine refusal records the node instead of crashing.
                        y1_eig, y2_eig = _from_lobe_fixed(
                            centroid, boundary_theta, boundary_r,
                            float(rho_lobe), float(theta_local))
                        partition = channels.evaluate(
                            gamma=float(gamma), y=(y1_eig, y2_eig),
                            beta=0.0, kappa=0.0)
                    except _REFUSAL_ERRORS:
                        refused.append((float(gamma), float(rho_lobe),
                                        float(theta_local)))
                        continue
                    env = partition.envelope
                    if not np.all(np.isfinite(env)):
                        # Conservative: a non-finite envelope is a refusal.
                        refused.append((float(gamma), float(rho_lobe),
                                        float(theta_local)))
                        continue
                    count = int(partition.real_mask.sum())
                    if image_count is None:
                        image_count = count
                        if image_count != _MACRO_SADDLE_IMAGE_COUNT:
                            raise ValueError(
                                'from_lobe_engine expects a '
                                f'{_MACRO_SADDLE_IMAGE_COUNT}-real-image '
                                'macro-saddle lobe interior, but the first '
                                f'evaluated node reports {image_count} real '
                                'images; the requested box is not a lobe '
                                'interior.')
                    elif count != image_count:
                        # A node with a different image count straddles a
                        # region boundary; record it refused, do not fit it.
                        refused.append((float(gamma), float(rho_lobe),
                                        float(theta_local)))
                        continue
                    envelope_real[:, i_g, i_rho, i_th] = env.real
                    envelope_imag[:, i_g, i_rho, i_th] = env.imag
                    carrier[i_g, i_rho, i_th] = partition.critical_source

        # Interior interpolator hygiene (SAME guard as `from_engine`'s
        # interior branch, unchanged -- F022): reject a tile that straddles a
        # medial ridge (critical-basin flip) for subdivision.
        _assert_carrier_continuity(
            carrier, gamma_grid,
            (gamma_grid.size, rho_lobe_grid.size, theta_local_grid.size))

        refused_points = (np.array(refused, dtype=float) if refused
                          else np.empty((0, 3), dtype=float))
        parity = (1 if 0.5 * float(gamma_grid[0] + gamma_grid[-1]) < 1.0
                  else -1)
        chart = LobeInteriorChart.from_lobe_values(
            gamma_grid=gamma_grid, rho_lobe_grid=rho_lobe_grid,
            theta_local_grid=theta_local_grid, log_w_grid=log_w_grid,
            envelope_real=envelope_real, envelope_imag=envelope_imag,
            image_count=image_count, parity=parity,
            centroid=centroid, other_centroid=other_centroid,
            corridor_half=corridor_half, boundary_theta=boundary_theta,
            boundary_r=boundary_r, eta_overlap_min=_DEFAULT_CAUSTIC_FLOOR,
            refused_points=refused_points,
            envelope_definition=_INTERIOR_ENVELOPE_DEFINITION)
        provenance = cls._build_lobe_provenance(
            gamma_range, rho_lobe_range, theta_local_range, w_range, shape,
            envelope_real, envelope_imag, centroid, other_centroid,
            corridor_half)
        return cls([chart], provenance)

    @staticmethod
    def _box_region_labels(gamma_grid: np.ndarray, rho_grid: np.ndarray,
                           theta_c_grid: np.ndarray
                           ) -> tuple[int | None, int | None]:
        """Real-image count and parity of the box's single region.

        The box lies inside one image-count region, so the region label is
        read once from a cheap ``w``-independent
        `ChangRefsdalChannels.geometry_partition` at the box centre.  The
        centre is a caustic-fixed ``(rho, theta_c)`` node, mapped to a
        physical eigenframe source at the central ``gamma``
        (`_from_caustic_fixed`) before the geometry call.  Parity is
        deterministic in ``gamma`` (``+1`` for ``gamma < 1``, ``-1`` for
        ``gamma > 1``).

        Returns ``(None, None)`` when the box-centre map refuses -- e.g. a box
        whose centre ``gamma`` is exactly ``1.0`` hits the ``_caustic_reach``
        parity wall (`LensDomainError`).  The chart then records unknown labels
        (handled conservatively downstream) instead of crashing construction.
        """
        gamma_c = 0.5 * float(gamma_grid[0] + gamma_grid[-1])
        rho_c = 0.5 * float(rho_grid[0] + rho_grid[-1])
        theta_cc = 0.5 * float(theta_c_grid[0] + theta_c_grid[-1])
        try:
            y1_c, y2_c = _from_caustic_fixed(gamma_c, rho_c, theta_cc)
            geom = ChangRefsdalChannels(
                np.array([1.0, 2.0])).geometry_partition(
                    gamma=gamma_c, y=(y1_c, y2_c), beta=0.0, kappa=0.0)
        except _REFUSAL_ERRORS:
            return None, None
        parity = 1 if gamma_c < 1.0 else -1
        return int(geom.real_mask.sum()), parity

    @staticmethod
    def _build_provenance(gamma_range: tuple[float, float],
                          rho_range: tuple[float, float],
                          theta_c_range: tuple[float, float],
                          w_range: tuple[float, float],
                          shape: tuple[int, int, int, int],
                          envelope_real: np.ndarray,
                          envelope_imag: np.ndarray) -> dict:
        """Build the minimal provenance dict, including a short train hash.

        The spatial ranges are the caustic-fixed ``(rho, theta_c)`` axis
        bounds (Build 8h-b3); the ``axis_schema`` tag records the
        coordinate convention so a stale raw-eigenframe artifact is
        distinguishable at load.
        """
        hasher = hashlib.sha1()
        hasher.update(np.ascontiguousarray(envelope_real).tobytes())
        hasher.update(np.ascontiguousarray(envelope_imag).tobytes())
        n_w, n_gamma, n_rho, n_theta = shape
        return {
            'gamma_range': [float(gamma_range[0]), float(gamma_range[1])],
            'rho_range': [float(rho_range[0]), float(rho_range[1])],
            'theta_c_range': [float(theta_c_range[0]),
                              float(theta_c_range[1])],
            'axis_schema': _FARFIELD_AXIS_SCHEMA,
            'w_range': [float(w_range[0]), float(w_range[1])],
            'resolution': {'n_w': int(n_w), 'n_gamma': int(n_gamma),
                           'n_rho': int(n_rho), 'n_theta': int(n_theta)},
            'beta': 0.0,
            'kappa': 0.0,
            'chart_count': 1,
            'chart_types': ['farfield'],
            'training_hash': hasher.hexdigest()[:12]}

    @staticmethod
    def _build_lobe_provenance(gamma_range: tuple[float, float],
                               rho_lobe_range: tuple[float, float],
                               theta_local_range: tuple[float, float],
                               w_range: tuple[float, float],
                               shape: tuple[int, int, int, int],
                               envelope_real: np.ndarray,
                               envelope_imag: np.ndarray,
                               centroid: np.ndarray,
                               other_centroid: np.ndarray,
                               corridor_half: float) -> dict:
        """Build the provenance dict for a macro-saddle lobe-interior chart.

        The lobe counterpart of `_build_provenance`.  The spatial ranges are
        the LOBE-LOCAL ``(rho_lobe, theta_local)`` axis bounds (WP1); the
        ``axis_schema`` tag records the lobe-local convention so a stale
        origin-centred or old-tag artifact is distinguishable (and hard-
        refused) at load.  The lobe frame (`centroid`, `other_centroid`,
        `corridor_half`) is stamped so the training source geometry is
        recoverable from the provenance alone.
        """
        hasher = hashlib.sha1()
        hasher.update(np.ascontiguousarray(envelope_real).tobytes())
        hasher.update(np.ascontiguousarray(envelope_imag).tobytes())
        n_w, n_gamma, n_rho, n_theta = shape
        return {
            'gamma_range': [float(gamma_range[0]), float(gamma_range[1])],
            'rho_lobe_range': [float(rho_lobe_range[0]),
                               float(rho_lobe_range[1])],
            'theta_local_range': [float(theta_local_range[0]),
                                  float(theta_local_range[1])],
            'axis_schema': _LOBE_AXIS_SCHEMA,
            'w_range': [float(w_range[0]), float(w_range[1])],
            'resolution': {'n_w': int(n_w), 'n_gamma': int(n_gamma),
                           'n_rho_lobe': int(n_rho),
                           'n_theta_local': int(n_theta)},
            'beta': 0.0,
            'kappa': 0.0,
            'centroid': [float(centroid[0]), float(centroid[1])],
            'other_centroid': [float(other_centroid[0]),
                               float(other_centroid[1])],
            'corridor_half': float(corridor_half),
            'chart_count': 1,
            'chart_types': ['lobe'],
            'training_hash': hasher.hexdigest()[:12]}

    # ---- Query --------------------------------------------------------

    def may_serve(self, gamma: float, log_w_min: float,
                  log_w_max: float) -> bool:
        """Cheap pre-check: could any chart serve this ``gamma`` / ``w`` band?

        A candidate that fails this cannot be served by any chart, so the
        caller may skip building the (more expensive) geometry partition
        the full `serve` guard stack needs.  Checks only the gamma guard
        band and per-chart ``(gamma, log w)`` box containment -- no
        ``eta`` / image-count / source coordinates.

        Parameters
        ----------
        gamma : float
            External shear magnitude.
        log_w_min, log_w_max : float
            Bounds of the query's ``ln w`` band.

        Returns
        -------
        bool
            ``True`` if some chart's ``(gamma, log w)`` box could contain
            the candidate.
        """
        if abs(gamma - 1.0) < _GAMMA_GUARD_BAND:
            return False
        for chart in self.charts:
            if (chart.gamma_grid[0] <= gamma <= chart.gamma_grid[-1]
                    and _log_w_band_inside(chart, log_w_min, log_w_max)):
                return True
        return False

    def serve(self, w_array: np.ndarray, *, gamma: float, y1: float,
              y2: float, beta: float, eta: float, theta: float,
              image_count: int) -> tuple[np.ndarray, bool, str | None]:
        """Emulated envelope ``E(w)`` via the full multi-chart guard stack.

        Rotates ``(y1, y2)`` into the shear eigenframe, runs `select_chart`
        on the certified physical ``(gamma, eta, image_count)`` (with
        ``theta`` used only for cusp exclusion), and -- if a chart is
        selected -- evaluates its real/imag splines over ``ln w``.  The
        caustic distance ``eta``, arc angle ``theta`` and ``image_count``
        come from the caller's `ChangRefsdalChannels.geometry_partition`;
        this method recomputes NO geometry.

        Parameters
        ----------
        w_array : np.ndarray
            Dimensionless frequencies; the output matches its shape.
        gamma : float
            External shear magnitude.
        y1, y2 : float
            Source position in the shear frame at orientation ``beta``.
        beta : float
            External shear orientation, radians.
        eta : float
            Caustic distance ``partition.caustic_distance``.
        theta : float
            Caustic arc angle ``partition.caustic_theta`` (gauge).
        image_count : int
            Real-image count ``int(partition.real_mask.sum())``.

        Returns
        -------
        E_array : np.ndarray
            Complex emulated envelope shaped like ``w_array`` (zeros when
            not served).
        served : bool
            ``True`` if the surrogate emulated the envelope; ``False`` if
            the caller must fall back to the exact engine.
        definition : str or None
            The served chart's envelope-definition tag when a
            `FarFieldChart` or `LobeInteriorChart` is served (the
            serving-side reconstruction dispatches on it -- a lobe chart's
            INTERIOR_SACR_C label reconstructs by the interior mirror in the
            query geometry's ``tau_c`` frame, identical to an origin-centred
            interior chart, Professor Q3), ``None`` for a `TubeChart` or
            when not served.  The persisted tag is the single dispatch
            signal -- no parallel flag.
        """
        w = np.asarray(w_array, dtype=float)
        w_flat = np.atleast_1d(w).ravel()
        if w_flat.size == 0 or not np.all(w_flat > 0.0):
            return np.zeros(w.shape, dtype=complex), False, None

        log_w = np.log(w_flat)
        y1_eig, y2_eig = _rotate_to_eigenframe(y1, y2, beta)
        # Caustic-fixed source coordinate for the far-field chart query
        # (Build 8h-b3): the SAME scalar-reach normalisation the map/serve
        # side uses, so train-time and serve-time rho agree exactly.  The
        # caustic reach is undefined exactly on the det-A = 0 parity
        # boundary (inside the gamma guard band select_chart declines
        # anyway); decline cleanly rather than propagate the refusal.
        try:
            rho, theta_c = _to_caustic_fixed(gamma, y1_eig, y2_eig)
        except LensDomainError:
            return np.zeros(w.shape, dtype=complex), False, None
        chart = select_chart(
            self.charts, gamma=gamma, log_w_min=float(log_w.min()),
            log_w_max=float(log_w.max()), eta=eta, theta=theta,
            image_count=image_count, rho=rho, theta_c=theta_c,
            y1_eig=y1_eig, y2_eig=y2_eig)
        if chart is None:
            return np.zeros(w.shape, dtype=complex), False, None

        env_flat = _evaluate_chart(chart, gamma, rho, theta_c, eta, theta,
                                   log_w, y1_eig, y2_eig)
        definition = (chart.envelope_definition
                      if isinstance(chart, (FarFieldChart, LobeInteriorChart))
                      else None)
        return env_flat.reshape(w.shape), True, definition

    # ---- Legacy single-box (far-field) query --------------------------

    def in_domain(self, gamma: float, y1: float, y2: float,
                  beta: float) -> bool:
        """Whether a far-field chart serves ``(gamma, y1, y2, beta)`` by
        caustic-fixed coordinates (the 8a domain gate).

        Rotates the source into the shear eigenframe, maps it to the
        caustic-fixed coordinate ``(rho, theta_c)`` (Build 8h-b3), and
        tests box containment plus the exclusion balls over the far-field
        charts -- the exact 8a single-box gate, generalized to the
        far-field charts of a global surrogate.  It does NOT consult
        ``eta`` / image count (use `serve` for the full guard stack).

        Parameters
        ----------
        gamma : float
            External shear magnitude.
        y1, y2 : float
            Source position in the shear frame at orientation ``beta``.
        beta : float
            External shear orientation, radians.

        Returns
        -------
        bool
            ``True`` if some far-field chart contains the eigenframe point.
        """
        y1_eig, y2_eig = _rotate_to_eigenframe(y1, y2, beta)
        try:
            rho, theta_c = _to_caustic_fixed(gamma, y1_eig, y2_eig)
        except LensDomainError:
            return False
        return self._farfield_raw_chart(gamma, rho, theta_c) is not None

    def _farfield_raw_chart(self, gamma: float, rho: float,
                            theta_c: float):
        """First far-field chart whose box contains the caustic-fixed point.

        The far-field charts are gridded over the caustic-fixed source
        coordinate ``(rho, theta_c)`` (Build 8h-b3), so containment is
        tested on ``rho_grid`` / ``theta_c_grid`` rather than the retired
        raw eigenframe axes.
        """
        for chart in self.charts:
            if not isinstance(chart, FarFieldChart):
                continue
            if not (chart.gamma_grid[0] <= gamma <= chart.gamma_grid[-1]
                    and chart.rho_grid[0] <= rho <= chart.rho_grid[-1]
                    and chart.theta_c_grid[0] <= theta_c
                    <= chart.theta_c_grid[-1]):
                continue
            if _in_exclusion_ball(chart, gamma, rho, theta_c):
                continue
            return chart
        return None

    def envelope(self, w_array: np.ndarray, gamma: float, y1: float,
                 y2: float, beta: float) -> tuple[np.ndarray, bool]:
        """Legacy 8a far-field envelope query (caustic-fixed lookup only).

        Preserves the 8a call signature: rotates ``(y1, y2)`` into the
        eigenframe, maps to the caustic-fixed coordinate ``(rho,
        theta_c)`` (Build 8h-b3), selects the first far-field chart whose
        box contains the point (exclusion balls honoured), and evaluates
        its splines over ``w``.  Returns ``served=False`` (zeros) when no
        far-field chart contains the point or any ``w`` is outside that
        chart's band.  Does NOT run the tube/eta guard stack -- use
        `serve` for the full global query.

        Parameters
        ----------
        w_array : np.ndarray
            Dimensionless frequencies; the output matches its shape.
        gamma : float
            External shear magnitude.
        y1, y2 : float
            Source position in the shear frame at orientation ``beta``.
        beta : float
            External shear orientation, radians.

        Returns
        -------
        E_array : np.ndarray
            Complex emulated envelope shaped like ``w_array``.
        served : bool
            ``True`` if a far-field chart emulated the envelope.
        """
        w = np.asarray(w_array, dtype=float)
        y1_eig, y2_eig = _rotate_to_eigenframe(y1, y2, beta)
        try:
            rho, theta_c = _to_caustic_fixed(gamma, y1_eig, y2_eig)
        except LensDomainError:
            return np.zeros(w.shape, dtype=complex), False
        chart = self._farfield_raw_chart(gamma, rho, theta_c)
        if chart is None:
            return np.zeros(w.shape, dtype=complex), False

        w_flat = np.atleast_1d(w).ravel()
        w_min = float(np.exp(chart.log_w_grid[0]))
        w_max = float(np.exp(chart.log_w_grid[-1]))
        if w_flat.size == 0 or not np.all(
                (w_flat >= w_min) & (w_flat <= w_max)):
            return np.zeros(w.shape, dtype=complex), False

        env_flat = _evaluate_chart(chart, gamma, rho, theta_c,
                                   float('nan'), float('nan'),
                                   np.log(w_flat))
        return env_flat.reshape(w.shape), True

    # ---- Serialization ------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the surrogate to a single ``.npz`` file (flat ndarrays).

        Stores every chart flat -- ``chart{i}_re_coeffs`` /
        ``chart{i}_im_coeffs`` / ``chart{i}_knots_0..3`` /
        ``chart{i}_axis0..3`` plus a JSON-encoded ``chart{i}_meta`` (kind,
        image count, parity, exclusion data; and ``chart{i}_refused`` for
        far-field charts) -- alongside the chart count and the
        JSON-encoded provenance scalar.  There is NO bespoke manifest.
        The interpolant round-trips bit-for-bit and the file loads without
        pickle.

        Parameters
        ----------
        path : str or Path
            Destination path; ``.npz`` is appended by numpy if absent.
        """
        arrays: dict = {
            'n_charts': np.array(len(self.charts)),
            'provenance': np.array(json.dumps(self.provenance))}
        for i, chart in enumerate(self.charts):
            arrays.update(_chart_to_npz(chart, i))
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path | None = None
             ) -> 'LensAmplificationSurrogate':
        """Load a surrogate from a saved ``.npz`` artifact.

        Two load paths:

        - ``path`` given (str or Path): load that explicit file.  This is
          also the path used for a cluster-hosted artifact.
        - ``path`` omitted (``None``): resolve the shipped package-data
          default under ``cogwheel/data/`` via `importlib.resources`.

        An 8a single-box artifact (no ``n_charts`` key) loads as a
        one-chart `FarFieldChart` for backward compatibility.

        Parameters
        ----------
        path : str or Path, optional
            Explicit artifact path; ``None`` resolves the package default.

        Returns
        -------
        LensAmplificationSurrogate
            The reconstructed surrogate.
        """
        if path is None:
            path = cls._default_artifact_path()
            # TODO(build8c-deferred): if the package-data default is
            # absent, fall back to a `data_registry.yaml` cluster path
            # here (deferred per Simplifier; the explicit-``path`` override
            # already covers cluster-hosted artifacts today).
        with np.load(path, allow_pickle=False) as data:
            if 'n_charts' not in data.files:
                return cls._load_legacy_single_box(data)
            provenance = json.loads(str(data['provenance']))
            charts = [_chart_from_npz(data, i)
                      for i in range(int(data['n_charts']))]
            return cls(charts, provenance)

    @staticmethod
    def _default_artifact_path() -> Path:
        """Resolve the shipped package-data artifact path under
        cogwheel/data."""
        return Path(str(files('cogwheel').joinpath('data',
                                                    _DEFAULT_ARTIFACT_NAME)))

    @classmethod
    def _load_legacy_single_box(cls, data
                                ) -> 'LensAmplificationSurrogate':
        """Load an 8a single-box artifact as a one-chart far-field surrogate.

        The 8a box carried no region labels, so ``image_count`` is left
        ``None`` (its image-count guard is skipped) and ``parity`` is
        inferred from the box-centre ``gamma``; the caustic floor becomes
        the chart's ``eta_overlap_min`` to preserve the 8a serving
        boundary.
        """
        gamma_grid = data['gamma_grid']
        # A legacy single-box artifact predates the far-field
        # envelope-definition tag (Build 8g-b), so it carries the OLD
        # caustic-region label and must be refused rather than served under
        # the new reconstruction (an unknown/absent tag hard-refuses).
        tag = (str(data['envelope_definition'])
               if 'envelope_definition' in data.files else None)
        definition = _validate_farfield_definition(
            tag, 'legacy single-box artifact')
        # A legacy single-box artifact predates the caustic-fixed axis
        # schema (Build 8h-b3) -- its spatial axes are raw eigenframe
        # ``(y1_eig, y2_eig)``, so it carries no ``axis_schema`` tag and is
        # hard-refused here (it would never survive the definition refuse
        # above either).  Refuse loudly rather than serve at the wrong
        # coordinate.
        axis_tag = (str(data['axis_schema'])
                    if 'axis_schema' in data.files else None)
        _validate_farfield_axis_schema(
            axis_tag, 'legacy single-box artifact')
        parity = (1 if 0.5 * float(gamma_grid[0] + gamma_grid[-1]) < 1.0
                  else -1)
        chart = FarFieldChart._assemble(
            gamma_grid=gamma_grid, rho_grid=data['rho_grid'],
            theta_c_grid=data['theta_c_grid'], log_w_grid=data['log_w_grid'],
            real_coeffs=data['real_coeffs'], imag_coeffs=data['imag_coeffs'],
            knots=(data['knot_log_w'], data['knot_gamma'], data['knot_rho'],
                   data['knot_theta_c']),
            image_count=None, parity=parity,
            eta_overlap_min=_DEFAULT_CAUSTIC_FLOOR,
            refused_points=data['refused_points'],
            envelope_definition=definition)
        provenance = json.loads(str(data['provenance']))
        return cls([chart], provenance)


# ---- Per-chart (de)serialization helpers ------------------------------


def _validate_farfield_definition(tag, artifact_label: str) -> str:
    """Hard-refuse a far-field chart with an absent or unknown definition tag.

    Build 8g-b redefined the far-field envelope label; a chart trained
    before the tag existed (v1/v2 partial artifacts) encodes the OLD
    lobe-flipping caustic-region envelope and would reconstruct a
    finite-but-wrong ``F`` under the new serving mirror.  Refuse rather
    than serve it.

    Parameters
    ----------
    tag : str or None
        The ``envelope_definition`` read from the artifact meta.
    artifact_label : str
        Human-readable identifier (e.g. ``'chart 3'``) for the error.

    Returns
    -------
    str
        The validated tag.

    Raises
    ------
    ValueError
        If ``tag`` is ``None`` or not in `_KNOWN_ENVELOPE_DEFINITIONS`
        (the union of the far-field window-class labels and the interior
        SACR-C label added in Build S2-3).
    """
    if tag is None or tag not in _KNOWN_ENVELOPE_DEFINITIONS:
        raise ValueError(
            f'Chart {artifact_label} carries envelope-definition tag '
            f'{tag!r}, which is absent or unknown (known: '
            f'{sorted(_KNOWN_ENVELOPE_DEFINITIONS)}).  This artifact '
            f'predates the Build 8g-b far-field envelope redefinition (or '
            f'the Build S2-3 interior SACR-C relabel) and must not serve '
            f'under the new reconstruction; rebuild the surrogate.')
    return str(tag)


def _validate_axis_schema(tag, known_set: frozenset[str],
                          artifact_label: str) -> str:
    """Hard-refuse a chart with an absent or unknown axis schema.

    Generic axis-schema gate shared by every chart kind.  A chart trained
    on a stale or wrong-frame coordinate convention (raw eigenframe,
    scalar-reach rho, multiplicative directional rho, or an origin-centred
    axis on a lobe chart) would be queried at the WRONG coordinate and could
    return a finite-but-wrong amplification.  Refusing at load -- naming the
    artifact, the offending tag, and the known set, and instructing a
    rebuild -- turns that silent mis-serve into a loud, unmissable failure.

    Parameters
    ----------
    tag : str or None
        The ``axis_schema`` read from the artifact meta.
    known_set : frozenset[str]
        The axis schemas accepted for this chart kind (e.g.
        `_KNOWN_FARFIELD_AXIS_SCHEMAS` or `_KNOWN_LOBE_AXIS_SCHEMAS`).
    artifact_label : str
        Human-readable identifier (e.g. ``'chart 3'``) for the error.

    Returns
    -------
    str
        The validated tag.

    Raises
    ------
    ValueError
        If ``tag`` is ``None`` or not in ``known_set``.
    """
    if tag is None or tag not in known_set:
        raise ValueError(
            f'{artifact_label} carries axis-schema tag {tag!r}, which is '
            f'absent or unknown (known: {sorted(known_set)}). This artifact '
            f'may use a stale or wrong-frame coordinate convention and must '
            f'not serve under the current reconstruction; rebuild '
            f'the surrogate.')
    return str(tag)


def _validate_farfield_axis_schema(tag, artifact_label: str) -> str:
    """Hard-refuse a far-field chart with an absent or unknown axis schema.

    Thin wrapper over `_validate_axis_schema` binding the far-field known
    set.  Positive-parity far-field charts use piecewise caustic-fixed
    ``(rho, theta_c)`` coordinates; a chart trained on raw eigenframe axes,
    scalar-reach rho, or multiplicative directional rho would be queried at
    the wrong coordinate and could return a finite-but-wrong amplification.
    """
    return _validate_axis_schema(
        tag, _KNOWN_FARFIELD_AXIS_SCHEMAS, f'Far-field {artifact_label}')


def _validate_lobe_axis_schema(tag, artifact_label: str) -> str:
    """Hard-refuse a lobe-interior chart with an absent or unknown schema.

    Thin wrapper over `_validate_axis_schema` binding the lobe known set.
    Macro-saddle lobe-interior charts are queried on lobe-local
    ``(rho_lobe, theta_local)`` coordinates centred on the lobe centroid; a
    chart stamped with the far-field caustic-fixed tag, an origin-centred
    axis, or an old lobe tag would be reconstructed at the wrong coordinate
    and must hard-refuse at load.
    """
    return _validate_axis_schema(
        tag, _KNOWN_LOBE_AXIS_SCHEMAS, f'Lobe-interior {artifact_label}')


def _chart_to_npz(chart, index: int) -> dict:
    """Flatten one chart into ``chart{index}_*`` npz arrays."""
    prefix = f'chart{index}_'
    if isinstance(chart, TubeChart):
        meta = {'kind': 'tube', 'image_count': chart.image_count,
                'parity': chart.parity, 'eta_floor': chart.eta_floor,
                'eta_max': chart.eta_max,
                'cusp_windows': [[tc, dt] for tc, dt in chart.cusp_windows]}
        axes = (chart.log_w_grid, chart.gamma_grid, chart.u_grid,
                chart.theta_grid)
        arrays = {}
    elif isinstance(chart, LobeInteriorChart):
        # Additive lobe branch (WP1): the persisted record carries the lobe
        # frame (centroid, other_centroid, boundary_theta/boundary_r as
        # arrays; corridor_half scalar in meta) alongside the interior spline.
        # The lobe axis-schema tag makes a mislabeled/old artifact hard-refuse
        # at load rather than reconstruct a finite-but-wrong F.
        meta = {'kind': 'lobe', 'image_count': chart.image_count,
                'parity': chart.parity,
                'eta_overlap_min': chart.eta_overlap_min,
                'envelope_definition': chart.envelope_definition,
                'corridor_half': float(chart.corridor_half),
                'axis_schema': _LOBE_AXIS_SCHEMA}
        axes = (chart.log_w_grid, chart.gamma_grid, chart.rho_lobe_grid,
                chart.theta_local_grid)
        arrays = {prefix + 'refused': chart.refused_points,
                  prefix + 'centroid': chart.centroid,
                  prefix + 'other_centroid': chart.other_centroid,
                  prefix + 'boundary_theta': chart.boundary_theta,
                  prefix + 'boundary_r': chart.boundary_r}
    else:
        meta = {'kind': 'farfield', 'image_count': chart.image_count,
                'parity': chart.parity,
                'eta_overlap_min': chart.eta_overlap_min,
                'envelope_definition': chart.envelope_definition,
                'axis_schema': _FARFIELD_AXIS_SCHEMA}
        axes = (chart.log_w_grid, chart.gamma_grid, chart.rho_grid,
                chart.theta_c_grid)
        arrays = {prefix + 'refused': chart.refused_points}
    arrays[prefix + 'meta'] = np.array(json.dumps(meta))
    arrays[prefix + 're_coeffs'] = chart.real_coeffs
    arrays[prefix + 'im_coeffs'] = chart.imag_coeffs
    for axis_index, (axis, knot) in enumerate(zip(axes, chart.knots)):
        arrays[f'{prefix}axis{axis_index}'] = axis
        arrays[f'{prefix}knots_{axis_index}'] = knot
    return arrays


def _chart_from_npz(data, index: int):
    """Reconstruct one chart from ``chart{index}_*`` npz arrays."""
    prefix = f'chart{index}_'
    meta = json.loads(str(data[prefix + 'meta']))
    axes = [data[f'{prefix}axis{j}'] for j in range(4)]
    knots = tuple(data[f'{prefix}knots_{j}'] for j in range(4))
    real_coeffs = data[prefix + 're_coeffs']
    imag_coeffs = data[prefix + 'im_coeffs']
    log_w_grid, gamma_grid, p1_grid, p2_grid = axes
    if meta['kind'] == 'tube':
        return TubeChart._assemble(
            gamma_grid=gamma_grid, u_grid=p1_grid, theta_grid=p2_grid,
            log_w_grid=log_w_grid, real_coeffs=real_coeffs,
            imag_coeffs=imag_coeffs, knots=knots,
            image_count=meta['image_count'], parity=meta['parity'],
            eta_floor=meta['eta_floor'], eta_max=meta['eta_max'],
            cusp_windows=[tuple(win) for win in meta['cusp_windows']])
    if meta['kind'] == 'lobe':
        # Additive lobe branch (WP1): a lobe chart demands the lobe axis
        # schema, so a mislabeled/old artifact hard-refuses here rather than
        # reconstructing at the wrong (origin-centred or far-field) coordinate.
        _validate_lobe_axis_schema(meta.get('axis_schema'), f'chart {index}')
        definition = _validate_farfield_definition(
            meta.get('envelope_definition'), f'chart {index}')
        return LobeInteriorChart._assemble(
            gamma_grid=gamma_grid, rho_lobe_grid=p1_grid,
            theta_local_grid=p2_grid, log_w_grid=log_w_grid,
            real_coeffs=real_coeffs, imag_coeffs=imag_coeffs, knots=knots,
            image_count=meta['image_count'], parity=meta['parity'],
            eta_overlap_min=meta['eta_overlap_min'],
            refused_points=data[prefix + 'refused'],
            centroid=data[prefix + 'centroid'],
            other_centroid=data[prefix + 'other_centroid'],
            corridor_half=meta['corridor_half'],
            boundary_theta=data[prefix + 'boundary_theta'],
            boundary_r=data[prefix + 'boundary_r'],
            envelope_definition=definition)
    definition = _validate_farfield_definition(
        meta.get('envelope_definition'), f'chart {index}')
    _validate_farfield_axis_schema(
        meta.get('axis_schema'), f'chart {index}')
    return FarFieldChart._assemble(
        gamma_grid=gamma_grid, rho_grid=p1_grid, theta_c_grid=p2_grid,
        log_w_grid=log_w_grid, real_coeffs=real_coeffs,
        imag_coeffs=imag_coeffs, knots=knots,
        image_count=meta['image_count'], parity=meta['parity'],
        eta_overlap_min=meta['eta_overlap_min'],
        refused_points=data[prefix + 'refused'],
        envelope_definition=definition)
