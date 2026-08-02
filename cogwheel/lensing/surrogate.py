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
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import BSpline, make_interp_spline

from cogwheel.lensing.chang_refsdal import (
    ChangRefsdalChannels, farfield_envelope_from_partition, geometry)
from cogwheel.lensing.chang_refsdal.channels import (
    FARFIELD_KERNEL_SUM, KNOWN_FARFIELD_DEFINITIONS, INTERIOR_SACR_C,
    KNOWN_INTERIOR_DEFINITIONS)
from cogwheel.lensing.chang_refsdal.geometry import LensDomainError
from cogwheel.lensing.chang_refsdal._schwinger import (
    SchwingerCertificationError)

if TYPE_CHECKING:  # typing-only; NEVER a runtime import (surrogate is the
    # lower module -- ``surrogate_training`` imports it, not vice versa).
    from cogwheel.lensing.surrogate_training import _SaddleLobeAdmission

# The engine's named refusals.  Any of these at ANY w node marks the whole
# parameter grid point refused (per-w refusal propagation, Professor Q4).
_REFUSAL_ERRORS = (LensDomainError, SchwingerCertificationError)

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

# --- Far-field smooth interpolation coordinate (Build 1e-farfield WP1) ------
# These constants back the additive caustic arc-length primitives
# (`_caustic_arclength_map`, `_to_farfield_smooth`, `_from_farfield_smooth`
# and their rejection guards).  The primitives are UNREFERENCED by existing
# serve/train/containment code; WP2 wires them in as the sole far-field
# coordinate.  Nothing here changes existing behaviour.

# Samples per gamma row in the caustic arc-length map; matches the shipped
# 1e-tube precedent (`surrogate_training._TUBE_ARC_MAP_SIZE = 2001`), whose
# measured round-trip error at 2001 is ~1e-7 (< the 1e-6 tolerance).
_FARFIELD_ARC_MAP_SIZE = 2001

# Lobe-interior wedge-edge sqrt-coordinate map density.  Closed-form
# (no engine calls), so the same 2001-node density is sufficient.
_LOBE_ARC_MAP_SIZE = 2001

# Fixed medial-axis / near-tied-foot tolerance in caustic-source (``y``)
# units: a source whose two nearest caustic feet are closer than this in
# source-plane distance sits on the medial axis, where the arc foot -- and
# hence the smooth coordinate -- is ambiguous, so the tile is rejected
# (Professor: a fixed tolerance, NOT measured-and-decided).
_FARFIELD_MEDIAL_AXIS_TOL = 1e-6

# Polar-angle nodes of the coarse INDEPENDENT tie detector backing the
# near-tied-foot guard (over the full circle for the astroid, per wedge for
# each macro-saddle lobe/branch).  It only brackets local minima to compare
# the two nearest feet; it never sets the served coordinate (that is
# `geometry.nearest_caustic_point`).
_FARFIELD_MEDIAL_SCAN_NODES = 361

# Polar-angle nodes of the cusp-span guard's tangent-reversal scan.  A cusp
# (``|y'| = 0``) reverses the caustic tangent, so consecutive scan tangents
# have a non-positive dot product -- a tolerance-FREE detector; the node
# count only needs to place a cusp strictly between two samples.
_FARFIELD_CUSP_SCAN_NODES = 257

# Source-plane distance below which a foot rebuilt from `critical_point` on
# the arc's own branch is deemed identical to the `nearest_caustic_point`
# foot (a branch/wedge-consistency check for the macro saddle; the two calls
# are byte-identical when the branch matches, so this only rejects a foot
# that actually sits on a different deltoid edge).
_FARFIELD_BRANCH_MATCH_TOL = 1e-9

# Float-comparison slack (radians) for placing a ``[0, 2*pi)`` nearest-foot
# angle into an arc whose endpoints live in another 2*pi window.  Pure
# rounding slack, not a physical margin.
_FARFIELD_ANGLE_SLACK = 1e-12

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
# charts stay in the SAME far-field-smooth ``(s, d)`` coordinate used by
# positive-parity exterior far-field charts; only the ENVELOPE LABEL differs,
# so they are still
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
# 1e-farfield WP2). A far-field EXTERIOR chart stores FAR-FIELD-SMOOTH spatial
# axes ``(s, d)``: ``s`` is the caustic ARC LENGTH of the source's nearest
# foot (tangential) and ``d`` its SIGNED perpendicular distance to the caustic
# (radial, ``+`` outside / ``-`` inside).  ``s`` is smooth across the C2
# curvature kinks the raw ``theta_c`` angle produced, so a cubic spline in
# ``s`` needs no cusp-node bolt-on.  The transform is defined by the chart's
# gamma-resolved arc-length map (`_FarFieldArcMap`), which is persisted
# alongside the axes; the axes are meaningless without it, which is why they
# carry their own schema tag.  The certified-ppGO map retains its separate
# scalar annulus coordinate.  The loader hard-refuses a far-field chart whose
# axis-schema tag is absent or unknown (mirroring the 8g-b envelope-definition
# hard-refuse): a stale caustic-fixed, raw-coordinate or scalar-reach artifact
# fails loudly.
#
# The STORED far-field label is frame-invariant
# (`channels.farfield_envelope_from_partition` demodulates by
# ``exp(+1j w t_min)``); the tag carries a ``_framewinv`` suffix and the loader
# hard-refuses any frame-dependent-label artifact rather than serving a
# finite-but-wrong reconstruction.
_FARFIELD_AXIS_SCHEMA = 'farfield_arclength_s_perp_d_framewinv'
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
_LOBE_AXIS_SCHEMA_V1 = 'lobe_local_offset_rholobe_thetalocal_framewinv'
_LOBE_AXIS_SCHEMA = 'lobe_local_offset_rholobe_thetalocal_sqrtedge_framewinv'
_KNOWN_LOBE_AXIS_SCHEMAS = frozenset({_LOBE_AXIS_SCHEMA_V1, _LOBE_AXIS_SCHEMA})

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


@dataclass(frozen=True, eq=False)
class _FarFieldArcMap:
    """Baked gamma-resolved caustic arc-length map ``s(theta, gamma)``.

    The far-field smooth interpolation coordinate replaces the raw
    ``theta_c`` axis of an exterior chart by the caustic ARC LENGTH ``s``
    (Build 1e-farfield WP1), the same move
    `surrogate_training._tube_arc_length_map` made for the tube charts'
    ``theta`` axis.  ``s`` is smooth across the C2 curvature kinks the raw
    angle produced, so a cubic spline in ``s`` needs no cusp-node bolt-on.

    The map is 2-D: one cumulative-arc-length row per gamma node, because a
    single representative-gamma map is O(10-25%) wrong at a 0.2-wide band
    edge and would corrupt the spline's gamma axis at build time (Professor
    ruling).  ``gamma_nodes`` ARE the 4-D spline's own gamma grid -- the map
    does NOT invent a second gamma axis.

    Units: ``theta`` is the eigenframe (``beta = 0``) caustic polar angle in
    radians; ``s`` is caustic arc length in dimensionless source-plane
    ``y`` units.  For a macro saddle the map spans ONE deltoid edge (one
    ``branch`` over one wedge) and ``s`` resets to ``0`` at ``theta_lo`` per
    wedge, so saddle wedges are charted independently of each other and of
    the astroid.

    Attributes
    ----------
    gamma_nodes : np.ndarray
        Shape ``(n_gamma,)`` strictly ascending shear magnitudes; the
        spline's own gamma grid.
    theta_fine : np.ndarray
        Shape ``(n_map,)`` shared, strictly ascending polar-angle grid over
        ``[theta_lo, theta_hi]``.
    s_table : np.ndarray
        Shape ``(n_gamma, n_map)`` cumulative arc length; ``s_table[i, 0]``
        is ``0`` and every row is strictly increasing.
    branch : int
        Square-root branch (``+1`` positive-parity astroid, ``+-1`` a
        macro-saddle deltoid edge) the map was integrated on.
    theta_lo, theta_hi : float
        Arc endpoints (radians); ``theta_lo < theta_hi``.
    """

    gamma_nodes: np.ndarray
    theta_fine: np.ndarray
    s_table: np.ndarray
    branch: int
    theta_lo: float
    theta_hi: float


def _caustic_arclength_map(gamma_nodes, theta_lo: float, theta_hi: float,
                           branch: int, n_map: int = _FARFIELD_ARC_MAP_SIZE
                           ) -> _FarFieldArcMap:
    """Build the gamma-resolved caustic arc-length map for one exterior arc.

    For every gamma node the exact caustic parametric speed ``|y'(theta)|``
    (`geometry.caustic_speed`, closed form -- NOT a finite difference) is
    integrated by the trapezoidal rule
    (`scipy.integrate.cumulative_trapezoid`) over the shared grid
    ``theta_fine = linspace(theta_lo, theta_hi, n_map)``, giving the
    cumulative arc length ``s`` from ``0`` at ``theta_lo``.  Trapezoid is
    MANDATORY: every increment ``(h/2)(f_i + f_{i+1})`` of a positive
    integrand is positive, so each row is strictly increasing BY
    CONSTRUCTION, which the `np.interp` inversion in `_from_farfield_smooth`
    depends on (`cumulative_simpson` also needs scipy >= 1.12; this env is
    1.11.4).

    ``gamma_nodes`` ARE the 4-D spline's own gamma grid; no second gamma
    axis is invented.  The macro-saddle path integrates ONE deltoid edge
    (one ``branch`` over one wedge); its ``s`` resets to ``0`` at
    ``theta_lo`` per wedge, so a saddle arc is charted independently and is
    NOT routed through the astroid arc (mirroring how `_to_caustic_fixed`
    branches ``gamma < 1`` vs ``> 1``).  Astroid arcs use ``branch = +1``
    and span between two ADJACENT cusps.

    Parameters
    ----------
    gamma_nodes : array_like
        Strictly ascending shear magnitudes (the spline's gamma grid); at
        least two nodes.
    theta_lo, theta_hi : float
        Eigenframe caustic polar-angle endpoints of the arc, radians, with
        ``theta_lo < theta_hi``.  For a macro saddle both must lie strictly
        inside the deltoid wedge (else `geometry.caustic_speed` refuses).
    branch : int
        Square-root branch ``+-1`` (``+1`` for the positive-parity astroid).
    n_map : int
        Samples per gamma row (map resolution); defaults to the shipped
        1e-tube value ``2001``.

    Returns
    -------
    _FarFieldArcMap
        The baked map; round-trips ``(gamma_nodes, theta_fine, s_table)``.

    Raises
    ------
    ValueError
        If the inputs are malformed, or a row is non-finite or not strictly
        increasing (a ``|y'| = 0`` cusp lies inside the arc -- run
        `_reject_if_cusp_spanning` to reject the tile before building).
    LensDomainError
        Propagated from `geometry.caustic_speed` for an out-of-domain
        ``(gamma, theta, branch)`` (parity boundary, over-critical, or a
        macro-saddle arc that reaches the wedge edge).
    """
    gamma_nodes = np.asarray(gamma_nodes, dtype=float)
    if gamma_nodes.ndim != 1 or gamma_nodes.size < 2:
        raise ValueError(
            f'gamma_nodes must be a 1-D array of at least two nodes; got '
            f'shape {gamma_nodes.shape}.')
    if not np.all(np.diff(gamma_nodes) > 0.0):
        raise ValueError('gamma_nodes must be strictly ascending.')
    theta_lo = float(theta_lo)
    theta_hi = float(theta_hi)
    if not theta_hi > theta_lo:
        raise ValueError(
            f'theta_hi must exceed theta_lo; got [{theta_lo}, {theta_hi}].')
    branch = int(branch)
    if branch not in (1, -1):
        raise ValueError(f'branch must be +1 or -1; got {branch}.')
    n_map = int(n_map)
    if n_map < 2:
        raise ValueError(f'n_map must be at least 2; got {n_map}.')

    theta_fine = np.linspace(theta_lo, theta_hi, n_map)
    s_table = np.empty((gamma_nodes.size, n_map), dtype=float)
    for i, gamma in enumerate(gamma_nodes):
        speed = np.asarray(
            geometry.caustic_speed(float(gamma), theta_fine, branch=branch),
            dtype=float)
        row = cumulative_trapezoid(speed, theta_fine, initial=0.0)
        if not np.isfinite(row).all():
            raise ValueError(
                f'Caustic arc-length map is non-finite for gamma={gamma}, '
                f'branch={branch} over [{theta_lo}, {theta_hi}]; the arc '
                f'reaches a cusp or a macro-saddle wedge edge.')
        if not np.all(np.diff(row) > 0.0):
            raise ValueError(
                f'Caustic arc-length map is not strictly increasing for '
                f'gamma={gamma}, branch={branch}; the caustic speed vanishes '
                f"inside the arc (a |y'|=0 cusp lies in [{theta_lo}, "
                f'{theta_hi}]). Reject the tile with _reject_if_cusp_spanning '
                f'before building the map.')
        s_table[i] = row
    return _FarFieldArcMap(gamma_nodes, theta_fine, s_table, branch,
                           theta_lo, theta_hi)


def _farfield_gamma_weights(gamma_nodes: np.ndarray, gamma: float
                            ) -> tuple[int, float]:
    """Lower bracketing index and linear weight of ``gamma`` in ``gamma_nodes``.

    Returns ``(idx, frac)`` with ``gamma_nodes[idx] <= gamma <=
    gamma_nodes[idx + 1]`` and ``frac = (gamma - gamma_nodes[idx]) /
    (gamma_nodes[idx + 1] - gamma_nodes[idx])`` in ``[0, 1]`` -- the weight
    for linear-in-gamma interpolation between the two bracketing rows of an
    arc-length map.  A ``gamma`` outside the node range is a named refusal:
    the arc map does not cover it.

    Raises
    ------
    LensDomainError
        If ``gamma`` lies outside ``[gamma_nodes[0], gamma_nodes[-1]]``.
    """
    gamma = float(gamma)
    lo = float(gamma_nodes[0])
    hi = float(gamma_nodes[-1])
    if gamma < lo or gamma > hi:
        raise LensDomainError(
            f'gamma={gamma} is outside the arc-length map gamma range '
            f'[{lo}, {hi}]; this exterior chart does not cover it.')
    idx = int(np.searchsorted(gamma_nodes, gamma, side='right')) - 1
    idx = min(max(idx, 0), gamma_nodes.size - 2)
    g0 = float(gamma_nodes[idx])
    g1 = float(gamma_nodes[idx + 1])
    return idx, (gamma - g0) / (g1 - g0)


def _arclength_at(arc_map: _FarFieldArcMap, theta: float, gamma: float
                  ) -> float:
    """Arc length ``s`` at ``(theta, gamma)`` from a gamma-resolved map.

    ``np.interp`` in ``theta`` within each of the two bracketing gamma rows,
    then linear-in-gamma between them, so ``s`` is evaluated at the query's
    OWN gamma (serve direction, `_to_farfield_smooth`).
    """
    idx, frac = _farfield_gamma_weights(arc_map.gamma_nodes, gamma)
    s_lo = np.interp(theta, arc_map.theta_fine, arc_map.s_table[idx])
    s_hi = np.interp(theta, arc_map.theta_fine, arc_map.s_table[idx + 1])
    return float(s_lo + (s_hi - s_lo) * frac)


def _theta_at_arclength(arc_map: _FarFieldArcMap, s: float, gamma: float
                        ) -> float:
    """Polar angle ``theta`` at arc length ``s`` and ``gamma`` (map inverse).

    Inverts the monotone ``s`` column of each bracketing gamma row by
    ``np.interp`` (valid because every row is strictly increasing), then
    linear-in-gamma between the two, so ``theta`` is evaluated at the node's
    OWN gamma (train direction, `_from_farfield_smooth`).
    """
    idx, frac = _farfield_gamma_weights(arc_map.gamma_nodes, gamma)
    theta_lo_row = np.interp(s, arc_map.s_table[idx], arc_map.theta_fine)
    theta_hi_row = np.interp(s, arc_map.s_table[idx + 1], arc_map.theta_fine)
    return float(theta_lo_row + (theta_hi_row - theta_lo_row) * frac)


def _wrap_into_arc(theta: float, theta_lo: float, theta_hi: float) -> float:
    """Representative of ``theta`` (mod 2*pi) inside ``[theta_lo, theta_hi]``.

    `geometry.nearest_caustic_point` returns ``theta`` in ``[0, 2*pi)``, but
    an arc's endpoints may be given in another 2*pi window (e.g. a
    ``[-pi/2, 0]`` astroid arc).  Returns the unique ``theta + 2*pi*k`` in
    the arc; raises when the foot lies outside the arc so the caller never
    silently serves the wrong arc.

    Raises
    ------
    LensDomainError
        If no ``2*pi`` shift of ``theta`` lands in the arc.
    """
    for k in (-1, 0, 1):
        candidate = theta + 2.0 * np.pi * k
        if (theta_lo - _FARFIELD_ANGLE_SLACK <= candidate
                <= theta_hi + _FARFIELD_ANGLE_SLACK):
            return candidate
    raise LensDomainError(
        f'Nearest caustic foot theta={theta} lies outside this chart arc '
        f'[{theta_lo}, {theta_hi}]; reject and defer to the exact engine.')


def _outward_caustic_normal(gamma: float, theta: float, branch: int
                            ) -> np.ndarray:
    """Unit source-plane OUTWARD normal of the caustic at ``theta``.

    Built from the EXACT closed-form caustic tangent ``y'(theta)``
    (`geometry.caustic_derivatives`) rotated by ``-90`` deg, so it is
    exactly perpendicular to the tangent.  That exactness is what makes the
    smooth coordinate an exact inverse: at the nearest caustic foot the
    source-to-foot vector is perpendicular to the tangent, hence parallel to
    this normal, so ``foot + d * n_hat`` reconstructs the source (the
    residual is only the along-tangent stationarity error of
    `nearest_caustic_point`).  The sign is fixed OUTWARD (away from the
    fold's two-image / interior side) using
    `geometry.fold_opening_direction`, which points toward that interior
    side; nothing is re-derived.

    Consequently ``d > 0`` means the source is OUTSIDE the caustic and
    ``d < 0`` inside -- the sign convention of the perpendicular distance in
    `_to_farfield_smooth` / `_from_farfield_smooth`.

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    theta : float
        Eigenframe caustic polar angle, radians.
    branch : int
        Square-root branch ``+-1``.

    Returns
    -------
    np.ndarray
        Shape ``(2,)`` unit outward normal.

    Raises
    ------
    LensDomainError
        Propagated from the geometry primitives out of domain, or if the
        tangent vanishes (a cusp -- excluded by `_reject_if_cusp_spanning`).
    """
    tangent, _ = geometry.caustic_derivatives(
        float(gamma), float(theta), branch=int(branch))
    tangent = np.asarray(tangent, dtype=float)
    speed = float(np.hypot(tangent[0], tangent[1]))
    if not speed > 0.0:
        raise LensDomainError(
            f"Caustic tangent vanishes at theta={theta} (a |y'|=0 cusp) for "
            f'gamma={gamma}, branch={branch}; the outward normal is '
            f'undefined there.')
    normal = np.array([tangent[1], -tangent[0]]) / speed
    fold_inward = geometry.fold_opening_direction(
        float(gamma), float(theta), branch=int(branch))
    if float(np.dot(normal, np.asarray(fold_inward, dtype=float))) > 0.0:
        normal = -normal
    return normal


def _to_farfield_smooth(gamma: float, y1_eig: float, y2_eig: float,
                        arc_map: _FarFieldArcMap, branch: int
                        ) -> tuple[float, float]:
    """Far-field smooth coordinates ``(s, d)`` of an eigenframe source (SERVE).

    Serve direction of the far-field smooth frame: an eigenframe source
    ``(y1_eig, y2_eig)`` maps to the caustic arc length ``s`` of its nearest
    foot and the SIGNED perpendicular distance ``d`` to the caustic
    (``d > 0`` outside, ``d < 0`` inside; see `_outward_caustic_normal`),
    both in dimensionless source-plane ``y`` units.

    Steps: `geometry.nearest_caustic_point` (at ``beta = 0``,
    ``kappa = 0``) gives the foot angle ``theta*`` and the unsigned foot
    distance; the foot is rebuilt from `critical_point` on this arc's own
    ``branch`` (identical to the nearest-point foot when the branch matches,
    which is asserted so a source belonging to a different deltoid edge/lobe
    is rejected).  ``d`` is the source-to-foot displacement projected onto
    the outward normal; ``s`` interpolates the 2-D table at ``(theta*,
    gamma)`` -- ``np.interp`` in ``theta`` within the two bracketing gamma
    rows, then linear-in-gamma -- so ``s`` is evaluated at the query's OWN
    gamma.  The near-tied-foot (medial-axis) guard runs first: a source with
    two near-equidistant feet has an ambiguous coordinate and is rejected.

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    y1_eig, y2_eig : float
        Eigenframe source position (dimensionless).
    arc_map : _FarFieldArcMap
        The baked arc-length map for this arc.
    branch : int
        Square-root branch of this arc; must equal ``arc_map.branch``.

    Returns
    -------
    tuple[float, float]
        ``(s, d)`` -- arc length and signed perpendicular distance.

    Raises
    ------
    LensDomainError
        On a medial-axis ambiguity, a foot outside this arc, a foot on a
        different branch/lobe, a gamma outside the map, or any refusal
        propagated from the geometry primitives.
    ValueError
        If ``branch != arc_map.branch``.
    """
    branch = int(branch)
    if branch != arc_map.branch:
        raise ValueError(
            f'branch {branch} does not match arc_map.branch '
            f'{arc_map.branch}.')
    gamma = float(gamma)
    source = np.array([float(y1_eig), float(y2_eig)])

    _reject_if_near_tied_foot(gamma, float(y1_eig), float(y2_eig))

    nearest = geometry.nearest_caustic_point(gamma, 0.0, source, kappa=0.0)
    theta_star = _wrap_into_arc(float(nearest.theta),
                                arc_map.theta_lo, arc_map.theta_hi)

    foot = np.asarray(
        geometry.critical_point(gamma, theta_star, 0.0, 0.0, branch).source,
        dtype=float)
    if float(np.hypot(*(foot - np.asarray(nearest.source, dtype=float)))) \
            > _FARFIELD_BRANCH_MATCH_TOL:
        raise LensDomainError(
            f'Nearest caustic foot at theta={theta_star} is not on this '
            f'chart branch {branch}; the source belongs to a different '
            f'deltoid edge/lobe. Reject and defer to the exact engine.')

    n_hat = _outward_caustic_normal(gamma, theta_star, branch)
    delta = source - foot
    d = float(np.copysign(np.hypot(delta[0], delta[1]),
                          float(np.dot(delta, n_hat))))
    s = _arclength_at(arc_map, theta_star, gamma)
    return s, d


def _from_farfield_smooth(gamma: float, s: float, d: float,
                          arc_map: _FarFieldArcMap, branch: int
                          ) -> tuple[float, float]:
    """Eigenframe source of a far-field smooth ``(s, d)`` node (TRAIN; inverse).

    Train direction and exact inverse of `_to_farfield_smooth`: invert the
    monotone arc-length column at the node's OWN gamma to recover the foot
    angle ``theta*`` (`_theta_at_arclength`), rebuild the caustic foot
    ``P(theta*)`` from `critical_point` on this arc's ``branch``, and step
    off it by ``d`` along the SAME closed-form outward normal
    (`_outward_caustic_normal`).  Because the normal is exactly
    perpendicular to the caustic tangent, this is the exact inverse of the
    serve map (up to the along-tangent stationarity residual of
    `nearest_caustic_point`).

    Units: ``s`` is caustic arc length and ``d`` the signed perpendicular
    distance (``+`` outside, ``-`` inside), both dimensionless source-plane
    ``y`` units.  For a macro saddle ``s`` is measured from ``0`` at the
    wedge's ``theta_lo`` (per-wedge reset).

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    s : float
        Caustic arc length of the node.
    d : float
        Signed perpendicular distance of the node.
    arc_map : _FarFieldArcMap
        The baked arc-length map for this arc.
    branch : int
        Square-root branch of this arc; must equal ``arc_map.branch``.

    Returns
    -------
    tuple[float, float]
        The eigenframe source position ``(y1_eig, y2_eig)``.

    Raises
    ------
    LensDomainError
        On a gamma outside the map or any refusal propagated from the
        geometry primitives.
    ValueError
        If ``branch != arc_map.branch``.
    """
    branch = int(branch)
    if branch != arc_map.branch:
        raise ValueError(
            f'branch {branch} does not match arc_map.branch '
            f'{arc_map.branch}.')
    gamma = float(gamma)
    theta_star = _theta_at_arclength(arc_map, float(s), gamma)
    foot = np.asarray(
        geometry.critical_point(gamma, theta_star, 0.0, 0.0, branch).source,
        dtype=float)
    n_hat = _outward_caustic_normal(gamma, theta_star, branch)
    source = foot + float(d) * n_hat
    return float(source[0]), float(source[1])


def _reject_if_cusp_spanning(gamma: float, theta_lo: float, theta_hi: float,
                             branch: int,
                             n_scan: int = _FARFIELD_CUSP_SCAN_NODES) -> None:
    """Reject an arc that contains a caustic cusp (``|y'| = 0``).

    A cusp reverses the caustic tangent, so two consecutive scan tangents
    straddling it have a NON-POSITIVE dot product -- a tolerance-free
    detector (`geometry.caustic_derivatives` supplies the exact tangent).
    A non-finite tangent (a macro-saddle wedge edge inside the arc, F044) is
    also rejected.  The caller runs this at tile construction and catches
    `LensDomainError` to reject or subdivide the tile; the arc-length map is
    only built for cusp-free arcs (single inter-cusp astroid arc / per-wedge
    saddle edge), so `np.interp` never inverts a non-monotone column.

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    theta_lo, theta_hi : float
        Eigenframe caustic polar-angle endpoints, radians,
        ``theta_lo < theta_hi``.
    branch : int
        Square-root branch ``+-1``.
    n_scan : int
        Tangent-reversal scan resolution; need only place a cusp strictly
        between two samples.

    Raises
    ------
    LensDomainError
        If a cusp (or a macro-saddle wedge edge) lies inside the arc.
    ValueError
        If ``theta_hi <= theta_lo`` or ``n_scan < 2``.
    """
    gamma = float(gamma)
    theta_lo = float(theta_lo)
    theta_hi = float(theta_hi)
    branch = int(branch)
    n_scan = int(n_scan)
    if not theta_hi > theta_lo:
        raise ValueError(
            f'theta_hi must exceed theta_lo; got [{theta_lo}, {theta_hi}].')
    if n_scan < 2:
        raise ValueError(f'n_scan must be at least 2; got {n_scan}.')
    thetas = np.linspace(theta_lo, theta_hi, n_scan)
    tangent, _ = geometry.caustic_derivatives(gamma, thetas, branch=branch)
    tangent = np.asarray(tangent, dtype=float)
    if not np.isfinite(tangent).all():
        raise LensDomainError(
            f'Caustic tangent is non-finite inside [{theta_lo}, {theta_hi}] '
            f'for gamma={gamma}, branch={branch} (a macro-saddle wedge edge '
            f'lies in the arc); reject/subdivide the tile.')
    consecutive_dot = (tangent[0, :-1] * tangent[0, 1:]
                       + tangent[1, :-1] * tangent[1, 1:])
    if np.any(consecutive_dot <= 0.0):
        raise LensDomainError(
            f"A |y'|=0 caustic cusp lies inside the arc [{theta_lo}, "
            f'{theta_hi}] for gamma={gamma}, branch={branch} (the tangent '
            f'reverses); reject/subdivide the tile.')


def _reject_if_near_tied_foot(
        gamma: float, y1_eig: float, y2_eig: float, *,
        tol: float = _FARFIELD_MEDIAL_AXIS_TOL,
        n_scan: int = _FARFIELD_MEDIAL_SCAN_NODES) -> None:
    """Reject a source on the medial axis (two near-equidistant caustic feet).

    A coarse INDEPENDENT scan of the source-plane distance to the caustic
    (`_caustic_foot_distance_minima`) brackets the two nearest feet; if they
    are within ``tol`` the foot -- and hence the smooth coordinate -- is
    ambiguous, so the tile is rejected.  ``tol`` is the fixed
    ``1e-6``-caustic-source-unit medial-axis tolerance (Professor: fixed,
    not measured).  The scan is a tie detector only; the served foot itself
    is `geometry.nearest_caustic_point`.

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    y1_eig, y2_eig : float
        Eigenframe source position (dimensionless).
    tol : float
        Fixed medial-axis tolerance (source-plane ``y`` units).
    n_scan : int
        Coarse scan resolution.

    Raises
    ------
    LensDomainError
        If the two nearest caustic feet are within ``tol``.
    """
    minima = _caustic_foot_distance_minima(
        float(gamma), np.array([float(y1_eig), float(y2_eig)]), int(n_scan))
    if len(minima) >= 2 and (minima[1] - minima[0]) < float(tol):
        raise LensDomainError(
            f'Source ({y1_eig}, {y2_eig}) at gamma={gamma} has two caustic '
            f'feet within {tol} (nearest {minima[0]:.6g}, next '
            f'{minima[1]:.6g}); it lies on the medial axis where the arc '
            f'foot is ambiguous. Reject and defer to the exact engine.')


def _caustic_foot_distance_minima(gamma: float, source: np.ndarray,
                                  n_scan: int) -> list[float]:
    """Sorted source-plane distances of the caustic's local nearest feet.

    A coarse closed-form tie detector, NOT the precise nearest-point finder
    (`geometry.nearest_caustic_point` is): it scans the caustic point
    ``P(theta) = critical_point(...).source`` and returns the distances of
    the strict local minima of ``|P(theta) - source|``, sorted ascending.
    Positive parity scans the full circle on ``branch = +1`` (periodic);
    the macro saddle scans each of the two deltoid wedges on both branches
    (non-periodic, wedge-edge refusals skipped), so a source between the two
    lobes -- or between the two edges of one lobe -- yields two comparable
    minima.

    Parameters
    ----------
    gamma : float
        External shear magnitude.
    source : np.ndarray
        Shape ``(2,)`` eigenframe source position.
    n_scan : int
        Scan resolution (per circle / per wedge).

    Returns
    -------
    list[float]
        Ascending distances of the local nearest feet.
    """
    lam = 1.0
    magnitude = abs(float(gamma))
    if magnitude < lam:
        segments = ((1, np.linspace(0.0, 2.0 * np.pi, n_scan,
                                    endpoint=False)),)
        periodic = True
    else:
        theta_max = 0.5 * np.arcsin(lam / magnitude)
        segments = tuple(
            (branch, np.linspace(center - theta_max, center + theta_max,
                                 n_scan))
            for center in (0.0, np.pi) for branch in (1, -1))
        periodic = False
    minima: list[float] = []
    for branch, thetas in segments:
        distances = np.full(thetas.size, np.inf)
        for j, theta in enumerate(thetas):
            try:
                foot = geometry.critical_point(
                    float(gamma), float(theta), 0.0, 0.0, branch).source
            except LensDomainError:
                continue
            distances[j] = float(np.hypot(*(np.asarray(foot, dtype=float)
                                            - source)))
        minima.extend(_strict_local_minima(distances, periodic))
    minima.sort()
    return minima


def _strict_local_minima(values: np.ndarray, periodic: bool) -> list[float]:
    """Finite values that are strict local minima of a 1-D sequence.

    Non-finite entries (skipped scan points, e.g. macro-saddle wedge-edge
    refusals) never count as minima and never mask a neighbour.  Endpoints
    of a non-periodic sequence are treated as bounded by ``+inf`` outside.
    """
    n = values.size
    result: list[float] = []
    for j in range(n):
        value = values[j]
        if not np.isfinite(value):
            continue
        if periodic:
            left = values[(j - 1) % n]
            right = values[(j + 1) % n]
        else:
            left = values[j - 1] if j > 0 else np.inf
            right = values[j + 1] if j < n - 1 else np.inf
        if value < left and value < right:
            result.append(float(value))
    return result


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

    Why log-w is the correct collocation axis
    ------------------------------------------
    The tube chart splines the DEMODULATED envelope ``E(w)`` — the slowly
    varying complex weight remaining after the analytic carrier
    ``exp(i w Delta_tau)`` has been factored out.  While the raw
    amplification ``F(w)`` oscillates on the scale ``Delta w ~ 1/Delta_tau``
    (linear in ``w``), the demodulated envelope varies on a LOGARITHMIC
    scale: the Airy diffraction pattern governing a fold caustic
    modulates a geometric series of fringes whose successive widths grow
    by a constant factor in ``w`` (each fringe spans a fixed interval
    in ``ln w``).  Consequently:

    * Uniform ``ln w`` spacing gives each fringe-envelope feature the
      same number of collocation nodes — resolution tracks structure.
    * The LOO refinement oracle (`_leave_one_out_errors` in
      ``likelihood.py``) uses ``np.log(node_w)`` as its abscissa and
      inserts geometric midpoints ``sqrt(w_i * w_j)`` (arithmetic
      midpoints in ``ln w``), so the oracle's self-certified node
      placement is natively log-uniform.

    The chart's fixed ``nodes_per_decade`` density and the oracle's
    adaptive placement therefore share the same underlying scale by
    construction: both measure interpolation quality per unit ``ln w``.

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


def _log_reach_gamma_axis(gamma_range: tuple[float, float], n_nodes: int,
                          name: str) -> np.ndarray:
    """Build a gamma axis with nodes equispaced in log-caustic-reach.

    Compared to a uniform gamma grid, this concentrates nodes where the
    caustic reach varies most steeply (near gamma → 0 or gamma → 1),
    improving interpolation fidelity for a fixed node budget.

    Parameters
    ----------
    gamma_range : tuple[float, float]
        ``(low, high)`` with ``low < high``.
    n_nodes : int
        Number of nodes; must be at least 4 for cubic interpolation.
    name : str
        Axis name, used in error messages and downstream validation.

    Returns
    -------
    np.ndarray
        1-D strictly increasing gamma axis of length ``n_nodes`` with exact
        endpoint values ``gamma_range[0]`` and ``gamma_range[1]``.

    Raises
    ------
    ValueError
        If the range is not increasing or ``n_nodes < 4``.
    """
    lo, hi = float(gamma_range[0]), float(gamma_range[1])
    if not lo < hi:
        raise ValueError(
            f'{name} range must satisfy low < high; got {gamma_range}.')
    if n_nodes < 4:
        raise ValueError(
            f'{name} needs at least 4 nodes for cubic interpolation; '
            f'got {n_nodes}.')

    # Fine uniform gamma sweep to tabulate log(caustic_reach).
    g_fine = np.linspace(lo, hi, 200)
    t_fine = np.log(np.array([_caustic_reach(g) for g in g_fine]))

    # Place n_nodes uniformly in log-reach space.
    t_grid = np.linspace(t_fine[0], t_fine[-1], n_nodes)

    # Invert: np.interp requires ascending xp.  log-reach is increasing
    # for positive parity (gamma < 1) and decreasing for saddle (gamma > 1).
    if t_fine[-1] >= t_fine[0]:
        gamma_grid = np.interp(t_grid, t_fine, g_fine)
    else:
        gamma_grid = np.interp(t_grid, t_fine[::-1], g_fine[::-1])

    # Defensive sort (should already be ascending from the interp).
    gamma_grid.sort()

    # Pin exact endpoints to avoid floating-point drift.
    gamma_grid[0] = lo
    gamma_grid[-1] = hi

    return _validate_axis(gamma_grid, name)


def _validate_farfield_arc_map(arc_map: _FarFieldArcMap,
                               gamma_grid: np.ndarray) -> _FarFieldArcMap:
    """Return a far-field arc map validated against its chart gamma grid.

    The arc-length rows are a coordinate transform for the chart's spline,
    not an independently sampled approximation. Consequently their gamma
    nodes must be the chart's own gamma axis exactly.
    """
    if not isinstance(arc_map, _FarFieldArcMap):
        raise TypeError(
            'FarFieldChart requires a _FarFieldArcMap arc_map; got '
            f'{type(arc_map).__name__}.')

    gamma_nodes = np.ascontiguousarray(arc_map.gamma_nodes, dtype=float)
    theta_fine = np.ascontiguousarray(arc_map.theta_fine, dtype=float)
    s_table = np.ascontiguousarray(arc_map.s_table, dtype=float)
    if gamma_nodes.ndim != 1 or gamma_nodes.size != gamma_grid.size:
        raise ValueError(
            'arc_map.gamma_nodes must be a 1-D array with the same number '
            'of nodes as gamma_grid.')
    if not np.isfinite(gamma_nodes).all():
        raise ValueError('arc_map.gamma_nodes must be finite.')
    if not np.all(np.diff(gamma_nodes) > 0.0):
        raise ValueError('arc_map.gamma_nodes must be strictly increasing.')
    if not np.array_equal(gamma_nodes, gamma_grid):
        raise ValueError(
            'arc_map.gamma_nodes must equal gamma_grid; a far-field arc map '
            'cannot define a second gamma lattice.')
    if theta_fine.ndim != 1 or theta_fine.size < 2:
        raise ValueError(
            'arc_map.theta_fine must be a 1-D array with at least 2 nodes.')
    if not np.isfinite(theta_fine).all():
        raise ValueError('arc_map.theta_fine must be finite.')
    if not np.all(np.diff(theta_fine) > 0.0):
        raise ValueError('arc_map.theta_fine must be strictly increasing.')
    if s_table.shape != (gamma_nodes.size, theta_fine.size):
        raise ValueError(
            'arc_map.s_table must have shape '
            '(arc_map.gamma_nodes.size, arc_map.theta_fine.size); got '
            f'{s_table.shape}.')
    if not np.isfinite(s_table).all():
        raise ValueError('arc_map.s_table must be finite.')
    if not np.all(s_table[:, 0] == 0.0):
        raise ValueError('Every arc_map.s_table row must start at zero.')
    if not np.all(np.diff(s_table, axis=1) > 0.0):
        raise ValueError(
            'Every arc_map.s_table row must be strictly increasing.')

    branch = arc_map.branch
    if isinstance(branch, bool) or not isinstance(branch, (int, np.integer)):
        raise ValueError('arc_map.branch must be either +1 or -1.')
    branch = int(branch)
    if branch not in (-1, 1):
        raise ValueError('arc_map.branch must be either +1 or -1.')
    theta_lo = float(arc_map.theta_lo)
    theta_hi = float(arc_map.theta_hi)
    if not (np.isfinite(theta_lo) and np.isfinite(theta_hi)
            and theta_lo < theta_hi):
        raise ValueError(
            'arc_map endpoints must be finite with theta_lo < theta_hi.')
    if theta_fine[0] != theta_lo or theta_fine[-1] != theta_hi:
        raise ValueError(
            'arc_map.theta_fine endpoints must equal arc_map.theta_lo and '
            'arc_map.theta_hi.')
    return _FarFieldArcMap(gamma_nodes, theta_fine, s_table, branch,
                           theta_lo, theta_hi)


def _validate_theta_to_s(theta_to_s: np.ndarray,
                         theta_grid: np.ndarray) -> np.ndarray:
    """Return a validated ``(2, N_map)`` theta->arc-length axis map.

    Row 0 is ``theta_fine`` (strictly ascending, starting at the arc's
    lower bound ``theta_grid[0]``); row 1 is ``s_fine`` (cumulative arc
    length, strictly increasing from ~0).  Both rows must be finite.
    """
    arr = np.ascontiguousarray(theta_to_s, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != 2 or arr.shape[1] < 2:
        raise ValueError(
            f'theta_to_s must have shape (2, N_map) with N_map >= 2; '
            f'got shape {arr.shape}.')
    if not np.isfinite(arr).all():
        raise ValueError('theta_to_s must be finite.')
    theta_fine, s_fine = arr[0], arr[1]
    if not np.all(np.diff(theta_fine) > 0.0):
        raise ValueError(
            'theta_to_s row 0 (theta_fine) must be strictly increasing.')
    if not np.isclose(theta_fine[0], theta_grid[0]):
        raise ValueError(
            f'theta_to_s row 0 must start at theta_grid[0]={theta_grid[0]!r}; '
            f'got {theta_fine[0]!r}.')
    if not np.all(np.diff(s_fine) > 0.0):
        raise ValueError(
            'theta_to_s row 1 (s_fine) must be strictly increasing.')
    if not np.isclose(s_fine[0], 0.0, atol=1e-9):
        raise ValueError(
            f'theta_to_s row 1 (s_fine) must start at ~0; got {s_fine[0]!r}.')
    return arr


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
        Shape ``(n_gamma, n_s, n_d, 2)`` parked-carrier
        ``critical_source`` per node; ``NaN`` rows mark refused nodes and
        are skipped (a refused neighbour cannot certify continuity but is
        not itself a flip).
    gamma_grid : np.ndarray
        The ``n_gamma`` gamma axis, for the per-gamma caustic reach.
    shape : tuple[int, int, int]
        The ``(n_gamma, n_s, n_d)`` node-grid shape.

    Raises
    ------
    CarrierDiscontinuityError
        If a basin flip is detected between adjacent nodes.
    """
    n_gamma, n_s, n_d = shape
    grid = np.asarray(critical_sources, dtype=float).reshape(*shape, 2)
    # Per-gamma caustic reach, broadcast to the full node grid (the reach
    # varies with gamma only).
    reach = np.array([_caustic_reach(float(g)) for g in gamma_grid])
    reach_grid = np.broadcast_to(
        reach[:, None, None], (n_gamma, n_s, n_d))
    # Compare adjacent nodes along each spatial axis (gamma, s, d).
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
        Complex far-field label per node, shape ``(n_w, n_gamma, n_s,
        n_d)``.  Refused/unfilled nodes are exactly zero (`from_engine`
        leaves the value arrays zero there) and are skipped: a refused
        neighbour is a hole in the grid, not a discontinuity.
    w_max : float
        Top-of-band dimensionless frequency; the check is applied on this
        (last, highest) ``w`` slice and the value is reported in the error.
    gamma_grid : np.ndarray
        The ``n_gamma`` gamma axis, carried for parallelism with the interior
        guard; length-checked against ``shape``.
    shape : tuple[int, int, int]
        The ``(n_gamma, n_s, n_d)`` spatial node-grid shape.

    Raises
    ------
    CarrierDiscontinuityError
        If the normalized adjacent-node increment reaches
        `_FARFIELD_CARRIER_STEP_MAX` along any spatial axis.
    ValueError
        If ``gamma_grid`` length disagrees with ``shape[0]``.
    """
    n_gamma, _n_s, _n_d = shape
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
    # Compare adjacent nodes along each spatial axis (gamma, s, d).
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
    """Far-field-smooth-coordinate envelope chart, valid away from a caustic.

    Interpolates ``E(w)`` over ``(log w, gamma, s, d)`` for one image-count
    region, where the two spatial axes are the FAR-FIELD SMOOTH coordinates
    (Build 1e-farfield): the caustic ARC LENGTH ``s`` (tangential) of the
    source's nearest caustic foot and the SIGNED PERPENDICULAR DISTANCE ``d``
    to the caustic (``d > 0`` outside, ``d < 0`` inside).  ``s`` is smooth
    across the C2 curvature kinks the raw ``theta_c`` angle produced, so a
    cubic spline in ``s`` needs no cusp-node bolt-on.
    The chart carries the gamma-resolved arc-length map ``arc_map`` that
    defines the ``theta <-> s`` transform at every gamma; serve maps a query
    eigenframe source to ``(s, d)`` at the query's OWN gamma via
    `_to_farfield_smooth`.  Serve only where ``eta > eta_overlap_min``
    (bounded away from the caustic) and the candidate matches ``image_count``.

    Attributes
    ----------
    gamma_grid, s_grid, d_grid, log_w_grid : np.ndarray
        1-D strictly increasing training axes.  ``s_grid`` (arc length) and
        ``d_grid`` (signed perpendicular distance) are uniform ``linspace``
        grids, so their mean spacing equals their grid spacing and the
        exclusion-ball metric is meaningful by construction.
    real_coeffs, imag_coeffs : np.ndarray
        Cubic B-spline coefficient tensors, axes ``(log w, gamma, s, d)``.
    knots : tuple of np.ndarray
        Knot vectors ``(t_logw, t_gamma, t_s, t_d)``.
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
        Shape ``(n, 3)`` far-field-smooth ``(gamma, s, d)`` training points
        the engine refused; the exclusion-ball gate rejects queries within
        one grid spacing of any of them, measured in the SAME ``(gamma, s,
        d)`` coordinate the grid lives in (Build 1e-farfield WP2).
    param_spacing : np.ndarray
        Shape ``(3,)`` mean spacing of ``(gamma, s, d)`` for the
        exclusion-ball normalization.
    arc_map : _FarFieldArcMap
        REQUIRED gamma-resolved caustic arc-length map ``s(theta, gamma)``
        defining this chart's ``theta <-> s`` transform (its ``branch`` /
        ``theta_lo`` / ``theta_hi`` fix the arc).  Persisted in the npz and
        reloaded; serve/train map to/from ``(s, d)`` through it.
    envelope_definition : str
        Tag naming the label the chart's envelope encodes (Build 8g-b).
        Persisted in the npz meta and checked on load; the serving side
        dispatches the reconstruction on it.  Fresh charts default to
        `_FARFIELD_ENVELOPE_DEFINITION`.
    """

    gamma_grid: np.ndarray
    s_grid: np.ndarray
    d_grid: np.ndarray
    log_w_grid: np.ndarray
    real_coeffs: np.ndarray
    imag_coeffs: np.ndarray
    knots: tuple
    image_count: int | None
    parity: int | None
    eta_overlap_min: float
    refused_points: np.ndarray
    param_spacing: np.ndarray
    arc_map: _FarFieldArcMap
    envelope_definition: str

    @classmethod
    def from_values(cls, *, gamma_grid: np.ndarray, s_grid: np.ndarray,
                    d_grid: np.ndarray, log_w_grid: np.ndarray,
                    envelope_real: np.ndarray, envelope_imag: np.ndarray,
                    arc_map: _FarFieldArcMap,
                    image_count: int | None, parity: int | None,
                    eta_overlap_min: float = _DEFAULT_CAUSTIC_FLOOR,
                    refused_points: np.ndarray | None = None,
                    envelope_definition: str = _FARFIELD_ENVELOPE_DEFINITION
                    ) -> 'FarFieldChart':
        """Build a far-field chart by fitting splines to a value tensor.

        This is the engine-free construction seam (Build 1e-farfield WP2): a
        synthetic-oracle test supplies the ``(s_grid, d_grid)`` axes, a
        precomputed value tensor and an ``arc_map`` with NO engine call.

        Parameters
        ----------
        gamma_grid, s_grid, d_grid, log_w_grid : np.ndarray
            1-D strictly increasing training axes (the two spatial axes are
            the far-field-smooth arc length ``s`` and signed perpendicular
            distance ``d``).
        envelope_real, envelope_imag : np.ndarray
            Shape ``(n_w, n_gamma, n_s, n_d)`` real/imag envelope values.
        arc_map : _FarFieldArcMap
            REQUIRED gamma-resolved arc-length map defining the ``theta <->
            s`` transform (its ``branch`` / ``theta_lo`` / ``theta_hi`` fix
            the arc).
        image_count, parity : int or None
            Region labels (``None`` if unrecorded).
        eta_overlap_min : float, optional
            Minimum caustic distance served (default the caustic floor).
        refused_points : np.ndarray, optional
            Refused far-field-smooth ``(gamma, s, d)`` training points.
        envelope_definition : str, optional
            Tag naming the label the chart's envelope encodes (default the
            far-field kernel-sum label).  Interior charts pass the SACR-C
            interior tag so the serve side dispatches the SACR-C
            reconstruction (Build S2-3).
        """
        gamma_grid = _validate_axis(gamma_grid, 'gamma_grid')
        s_grid = _validate_axis(s_grid, 's_grid')
        d_grid = _validate_axis(d_grid, 'd_grid')
        log_w_grid = _validate_axis(log_w_grid, 'log_w_grid')
        expected = (log_w_grid.size, gamma_grid.size, s_grid.size,
                    d_grid.size)
        _check_value_shape(envelope_real, envelope_imag, expected)
        real_c, imag_c, knots = _fit_tensor_spline(
            (log_w_grid, gamma_grid, s_grid, d_grid),
            envelope_real, envelope_imag)
        return cls._assemble(
            gamma_grid, s_grid, d_grid, log_w_grid, real_c, imag_c,
            knots, image_count, parity, eta_overlap_min, refused_points,
            arc_map, envelope_definition=envelope_definition)

    @classmethod
    def _assemble(cls, gamma_grid, s_grid, d_grid, log_w_grid,
                  real_coeffs, imag_coeffs, knots, image_count, parity,
                  eta_overlap_min, refused_points, arc_map,
                  envelope_definition=_FARFIELD_ENVELOPE_DEFINITION
                  ) -> 'FarFieldChart':
        """Assemble a chart from prebuilt coefficient tensors and knots.

        param_spacing is the mean spacing of the (gamma, s, d) grids;
        because s_grid / d_grid are uniform linspaces the mean spacing equals
        the grid spacing, so the exclusion-ball metric -- refused_points and
        queries are both in (gamma, s, d) -- is meaningful by construction
        (Build 1e-farfield WP2).
        """
        gamma_grid = _validate_axis(gamma_grid, 'gamma_grid')
        s_grid = _validate_axis(s_grid, 's_grid')
        d_grid = _validate_axis(d_grid, 'd_grid')
        log_w_grid = _validate_axis(log_w_grid, 'log_w_grid')
        arc_map = _validate_farfield_arc_map(arc_map, gamma_grid)
        param_spacing = np.array([
            float(np.mean(np.diff(gamma_grid))),
            float(np.mean(np.diff(s_grid))),
            float(np.mean(np.diff(d_grid)))])
        return cls(
            gamma_grid=gamma_grid,
            s_grid=s_grid,
            d_grid=d_grid,
            log_w_grid=log_w_grid,
            real_coeffs=np.ascontiguousarray(real_coeffs, dtype=float),
            imag_coeffs=np.ascontiguousarray(imag_coeffs, dtype=float),
            knots=tuple(np.ascontiguousarray(t, dtype=float) for t in knots),
            image_count=None if image_count is None else int(image_count),
            parity=None if parity is None else int(parity),
            eta_overlap_min=float(eta_overlap_min),
            refused_points=_normalize_refused(refused_points),
            param_spacing=param_spacing,
            arc_map=arc_map,
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
    theta_to_s : np.ndarray or None
        Optional ``(2, N_map)`` theta→s axis reparametrization map.
        Row 0 is the dense ``theta_local`` grid; row 1 is the corresponding
        ``s = sqrt(span) - sqrt(theta_max - theta_local)`` coordinate.
        When ``None``, the spline is on raw ``theta_local`` (identity,
        backward-compatible with fixtures not built via the wedge-edge
        coordinate).
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
    theta_to_s: np.ndarray | None

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
                         = _INTERIOR_ENVELOPE_DEFINITION,
                         theta_to_s: np.ndarray | None = None,
                         s_grid: np.ndarray | None = None
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
        theta_to_s : np.ndarray or None, optional
            ``(2, N_map)`` theta→s axis reparametrization map.  When
            provided together with ``s_grid``, the spline's fourth axis
            is ``s`` (not raw ``theta_local``).
        s_grid : np.ndarray or None, optional
            1-D strictly increasing s-coordinate nodes (same length as
            ``theta_local_grid``).  Required when ``theta_to_s`` is given.
        """
        gamma_grid = _validate_axis(gamma_grid, 'gamma_grid')
        rho_lobe_grid = _validate_axis(rho_lobe_grid, 'rho_lobe_grid')
        theta_local_grid = _validate_axis(theta_local_grid,
                                          'theta_local_grid')
        log_w_grid = _validate_axis(log_w_grid, 'log_w_grid')
        expected = (log_w_grid.size, gamma_grid.size, rho_lobe_grid.size,
                    theta_local_grid.size)
        _check_value_shape(envelope_real, envelope_imag, expected)
        # When both theta_to_s and s_grid are provided, the spline's fourth
        # axis is s (the wedge-edge coordinate) instead of raw theta_local.
        if theta_to_s is not None and s_grid is not None:
            theta_to_s = _validate_theta_to_s(theta_to_s, theta_local_grid)
            s_grid = _validate_axis(s_grid, 's_grid')
            if s_grid.size != theta_local_grid.size:
                raise ValueError(
                    f's_grid length ({s_grid.size}) must equal '
                    f'theta_local_grid length ({theta_local_grid.size}).')
            spline_axes = (log_w_grid, gamma_grid, rho_lobe_grid, s_grid)
        elif theta_to_s is None and s_grid is None:
            # Identity path: byte-identical to HEAD.
            spline_axes = (log_w_grid, gamma_grid, rho_lobe_grid,
                           theta_local_grid)
        else:
            raise ValueError(
                'theta_to_s and s_grid must both be None or both provided.')
        real_c, imag_c, knots = _fit_tensor_spline(
            spline_axes, envelope_real, envelope_imag)
        return cls._assemble(
            gamma_grid, rho_lobe_grid, theta_local_grid, log_w_grid,
            real_c, imag_c, knots, image_count, parity, eta_overlap_min,
            refused_points, centroid, other_centroid, corridor_half,
            boundary_theta, boundary_r,
            envelope_definition=envelope_definition,
            theta_to_s=theta_to_s)

    @classmethod
    def _assemble(cls, gamma_grid, rho_lobe_grid, theta_local_grid, log_w_grid,
                  real_coeffs, imag_coeffs, knots, image_count, parity,
                  eta_overlap_min, refused_points, centroid, other_centroid,
                  corridor_half, boundary_theta, boundary_r,
                  envelope_definition=_INTERIOR_ENVELOPE_DEFINITION,
                  theta_to_s=None
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
            boundary_r=np.ascontiguousarray(boundary_r, dtype=float),
            theta_to_s=(np.ascontiguousarray(theta_to_s, dtype=float)
                        if theta_to_s is not None else None))


@dataclass(frozen=True, eq=False)
class TubeChart:
    """Near-caustic envelope chart in caustic-adapted coordinates.

    Interpolates ``E(w)`` over ``(log w, gamma, u = sqrt(eta), s)``,
    where ``eta`` is the source-plane distance to the caustic, ``theta``
    its arc position, and ``s`` the cumulative ARC LENGTH along the fold
    (``ds = |y'| d theta``) measured from the arc's lower bound.  Fitting
    in ``u = sqrt(eta)`` linearizes the fold's square-root branch so the
    interpolant is smooth through the near-caustic transition; the tube
    covers only the image-pair-present side ``eta > 0``.  ``theta`` is
    BOUNDED and NON-PERIODIC (a single inter-cusp fold arc); cusp
    neighbourhoods are excluded.

    The fourth interpolation axis is ARC LENGTH ``s`` (not raw ``theta``):
    a query ``theta`` is mapped to ``s`` through the stored ``theta_to_s``
    axis map at serve time before the spline is contracted.  Only the
    interpolation coordinate changed -- all membership, range and
    cusp-window tests still operate in ``theta``.

    Attributes
    ----------
    gamma_grid, u_grid, theta_grid, log_w_grid : np.ndarray
        1-D strictly increasing training axes (``u = sqrt(eta)``).
        ``theta_grid`` holds the fold-arc angle nodes (placed uniformly in
        arc length ``s``); the spline itself is fit against ``s``.
    theta_to_s : np.ndarray
        Arc-length axis map of shape ``(2, N_map)``: row 0 ``theta_fine``
        (strictly ascending, ``row0[0] == theta_grid[0]``, in the arc's
        wedge frame) and row 1 ``s_fine`` (arc length from 0, strictly
        increasing).  A query ``theta`` is mapped to the spline's ``s``
        coordinate by ``np.interp(theta, theta_fine, s_fine)``.
    real_coeffs, imag_coeffs : np.ndarray
        Cubic B-spline coefficient tensors, axes ``(log w, gamma, u, s)``.
    knots : tuple of np.ndarray
        Knot vectors ``(t_logw, t_gamma, t_u, t_s)`` (the fourth built in
        arc length ``s``).
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
    theta_to_s: np.ndarray
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
                    cusp_windows: tuple | None = None,
                    s_grid: np.ndarray | None = None,
                    theta_to_s: np.ndarray | None = None) -> 'TubeChart':
        """Build a tube chart by fitting splines to a value tensor.

        Parameters
        ----------
        gamma_grid, u_grid, theta_grid, log_w_grid : np.ndarray
            1-D strictly increasing training axes (``u = sqrt(eta)``).
            ``theta_grid`` holds the fold-arc angle nodes.
        envelope_real, envelope_imag : np.ndarray
            Shape ``(n_w, n_gamma, n_u, n_theta)`` real/imag envelope
            values (sampled at the ``theta_grid`` nodes).
        image_count, parity : int or None
            Region labels.
        eta_floor, eta_max : float
            Caustic-distance band served ``[eta_floor, eta_max]``.
        cusp_windows : tuple of (float, float), optional
            ``(theta_cusp, delta_theta)`` exclusion windows.
        s_grid : np.ndarray, optional
            The arc-length coordinates of the ``theta_grid`` nodes, used as
            the fourth spline axis.  Required when ``theta_to_s`` is given.
        theta_to_s : np.ndarray, optional
            The ``(2, N_map)`` arc-length axis map ``[theta_fine, s_fine]``.
            When omitted, an IDENTITY map is built (``theta_fine =
            theta_grid``, ``s_fine = theta_grid - theta_grid[0]``) and the
            spline is fit against that shifted-theta axis, so a chart built
            without an explicit map serves identically to a raw-theta spline.
        """
        gamma_grid = _validate_axis(gamma_grid, 'gamma_grid')
        u_grid = _validate_axis(u_grid, 'u_grid')
        theta_grid = _validate_axis(theta_grid, 'theta_grid')
        log_w_grid = _validate_axis(log_w_grid, 'log_w_grid')
        expected = (log_w_grid.size, gamma_grid.size, u_grid.size,
                    theta_grid.size)
        _check_value_shape(envelope_real, envelope_imag, expected)
        if theta_to_s is None:
            # Identity map: interpolate in arc length s = theta - theta_lo, a
            # constant shift of the raw-theta axis.  Fitting and serving in the
            # shifted coordinate is translation-equivalent to the raw-theta
            # spline, so charts built without an explicit map are unaffected.
            s_grid = theta_grid - theta_grid[0]
            theta_to_s = np.vstack([theta_grid, s_grid])
        if s_grid is None:
            raise ValueError(
                'from_values requires s_grid when theta_to_s is provided; the '
                'spline is fit against the arc-length node coordinates.')
        s_grid = _validate_axis(s_grid, 's_grid')
        if s_grid.size != theta_grid.size:
            raise ValueError(
                f's_grid size {s_grid.size} must equal theta_grid size '
                f'{theta_grid.size}.')
        real_c, imag_c, knots = _fit_tensor_spline(
            (log_w_grid, gamma_grid, u_grid, s_grid),
            envelope_real, envelope_imag)
        return cls._assemble(
            gamma_grid, u_grid, theta_grid, log_w_grid, real_c, imag_c, knots,
            image_count, parity, eta_floor, eta_max, cusp_windows, theta_to_s)

    @classmethod
    def _assemble(cls, gamma_grid, u_grid, theta_grid, log_w_grid,
                  real_coeffs, imag_coeffs, knots, image_count, parity,
                  eta_floor, eta_max, cusp_windows,
                  theta_to_s=None) -> 'TubeChart':
        """Assemble a chart from prebuilt coefficient tensors and knots.

        ``theta_to_s`` is the ``(2, N_map)`` arc-length axis map; when
        ``None`` an identity map (``s = theta - theta_grid[0]``) derived from
        ``theta_grid`` is used, so ``knots[3]`` (built in ``s``) and the map
        agree.  ``s_grid`` is NOT stored separately -- ``knots[3]`` already
        encodes it.
        """
        gamma_grid = _validate_axis(gamma_grid, 'gamma_grid')
        theta_grid = _validate_axis(theta_grid, 'theta_grid')
        if theta_to_s is None:
            theta_to_s = np.vstack([theta_grid, theta_grid - theta_grid[0]])
        windows = tuple((float(tc), float(dt))
                        for tc, dt in (cusp_windows or ()))
        return cls(
            gamma_grid=gamma_grid,
            u_grid=_validate_axis(u_grid, 'u_grid'),
            theta_grid=theta_grid,
            log_w_grid=_validate_axis(log_w_grid, 'log_w_grid'),
            theta_to_s=_validate_theta_to_s(theta_to_s, theta_grid),
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
                       gamma: float, p1: float, p2: float) -> bool:
    """Whether the chart's spatial ``(gamma, p1, p2)`` is within a refusal
    ball.

    ``refused_points`` and ``param_spacing`` are both in the chart's own
    spatial coordinate -- far-field-smooth ``(gamma, s, d)`` for a
    `FarFieldChart` (Build 1e-farfield WP2) or lobe-local ``(gamma,
    rho_lobe, theta_local)`` for a `LobeInteriorChart`, which shares this
    exact normalized-ball form -- so the test is coordinate-agnostic:
    ``(p1, p2)`` are the chart's own two spatial coordinates.  Tiles are
    sub-arcs in the tangential axis (they never wrap), so no angular-wrap
    handling is needed: refused points and queries share the tile's range.
    """
    refused = chart.refused_points
    if refused.shape[0] == 0:
        return False
    query = np.array([gamma, p1, p2])
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
                     y1_eig: float, y2_eig: float) -> bool:
    """Whether a far-field chart serves this candidate (steps 1,3,5,7).

    Far-field ``(s, d)`` charts are certified only for positive parity.
    A manually assembled or loaded macro-saddle-labelled `FarFieldChart`
    therefore DECLINES here, even if its array bounds happen to match; the
    caller falls through to the exact engine.  This is a safe compatibility
    response for a stale artifact, while construction remains available for
    inspection/migration rather than turning artifact loading into a crash.

    The source containment test is in the chart's far-field-smooth
    ``(s, d)`` axes (Build 1e-farfield WP2): the query eigenframe source is
    mapped to ``(s, d)`` at the query's OWN gamma through the chart's stored
    ``arc_map`` via `_to_farfield_smooth`.  A source whose smooth coordinate
    is ambiguous or off this arc (a medial-axis tie, a foot on a different
    deltoid edge/lobe, or a gamma outside the map) raises `LensDomainError`
    and the chart cleanly DECLINES -- the caller defers to the exact engine.
    """
    if chart.parity != 1:
        return False
    # (1a) gamma / log-w box containment (cheap, coordinate-free) first.
    if not (chart.gamma_grid[0] <= gamma <= chart.gamma_grid[-1]):
        return False
    if not _log_w_band_inside(chart, log_w_min, log_w_max):
        return False
    # (1b) map the source to the chart's far-field-smooth (s, d) at the
    # query's own gamma; decline on any refusal from the smooth map.
    try:
        s, d = _to_farfield_smooth(gamma, y1_eig, y2_eig, chart.arc_map,
                                   chart.arc_map.branch)
    except LensDomainError:
        return False
    if not (chart.s_grid[0] <= s <= chart.s_grid[-1]
            and chart.d_grid[0] <= d <= chart.d_grid[-1]):
        return False
    # (3) inherited engine-refusal exclusion ball, measured in (gamma, s, d).
    if _in_exclusion_ball(chart, gamma, s, d):
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
                 eta: float, theta: float, image_count: int,
                 y1_eig: float = float('nan'),
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
    containment on ``(gamma, log w)`` and the far-field-smooth source
    ``(s, d)``; (3) far-field engine-refusal exclusion balls; (5)
    image-count match; (6) cusp exclusion / ``eta`` floor; (7) tube when
    ``eta in [eta_floor, eta_max]``, else far-field when ``eta >
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
    y1_eig, y2_eig : float, optional
        Eigenframe source position.  Threaded to the far-field guard
        `_farfield_serves` (mapped to far-field-smooth ``(s, d)`` at the
        query's own gamma, Build 1e-farfield WP2) and to the lobe-interior
        guard `_lobe_serves`.  Defaults are non-finite: a caller that does
        not thread the source declines every far-field and lobe chart
        cleanly (only the tube dispatch is exercised).

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
                y1_eig, y2_eig):
            return chart
    for chart in charts:
        if isinstance(chart, LobeInteriorChart) and _lobe_serves(
                chart, gamma, log_w_min, log_w_max, eta, image_count,
                y1_eig, y2_eig):
            return chart
    return None


def _evaluate_chart(chart, gamma: float, eta: float, theta: float,
                    log_w_query: np.ndarray,
                    y1_eig: float = float('nan'),
                    y2_eig: float = float('nan')) -> np.ndarray:
    """Evaluate the selected chart's complex envelope over ``log_w_query``.

    A tube chart contracts on ``(sqrt(eta), s)`` where ``s`` is the query
    theta mapped into the chart frame and then to arc length via the
    chart's stored ``theta_to_s`` map; a FAR-FIELD chart contracts on its
    far-field-smooth spatial axes ``(s, d)`` computed from the eigenframe
    source at the query's OWN gamma via `_to_farfield_smooth` on the chart's
    stored ``arc_map`` (Build 1e-farfield WP2); a lobe-interior chart
    contracts on the LOBE-LOCAL ``(rho_lobe, v2)`` where ``v2`` is either
    the raw ``theta_local`` (when ``theta_to_s is None``) or the wedge-edge
    ``s`` coordinate mapped from ``theta_local`` via the chart's stored
    ``theta_to_s`` map (same pattern as tube charts).  ``y1_eig`` / ``y2_eig``
    are the eigenframe source, required for a `FarFieldChart` or
    `LobeInteriorChart` and ignored for a tube chart.

    A caller reaches this only after `select_chart` picked ``chart``; for a
    far-field chart that means `_to_farfield_smooth` already succeeded on
    this deterministic input, so the recomputation here does not raise.
    """
    if isinstance(chart, TubeChart):
        v1 = float(np.sqrt(eta))
        # The tube spline's fourth axis is ARC LENGTH s, so map the query
        # theta (already gated inside the arc) into the chart frame and then
        # onto s via the stored theta_to_s map before contracting.
        theta_inframe = _theta_into_frame(theta, float(chart.theta_grid[0]))
        v2 = float(np.interp(theta_inframe, chart.theta_to_s[0],
                             chart.theta_to_s[1]))
    elif isinstance(chart, LobeInteriorChart):
        rho_lobe, theta_local = _to_lobe_fixed(
            chart.centroid, chart.boundary_theta, chart.boundary_r,
            y1_eig, y2_eig)
        v1 = rho_lobe
        if chart.theta_to_s is not None:
            # Wedge-edge s-coordinate: map theta_local -> s via the stored
            # dense map before contracting the spline (same pattern as the
            # tube chart's arc-length mapping).
            v2 = float(np.interp(theta_local, chart.theta_to_s[0],
                                 chart.theta_to_s[1]))
        else:
            v2 = theta_local
    else:
        # Far-field chart: far-field-smooth (s, d) at the query's own gamma.
        v1, v2 = _to_farfield_smooth(gamma, y1_eig, y2_eig, chart.arc_map,
                                     chart.arc_map.branch)
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
    def s_grid(self) -> np.ndarray:
        """Caustic arc-length ``s`` axis of the first (far-field) chart (8a
        shim)."""
        return self.charts[0].s_grid

    @property
    def d_grid(self) -> np.ndarray:
        """Signed perpendicular-distance ``d`` axis of the first (far-field)
        chart (8a shim)."""
        return self.charts[0].d_grid

    @property
    def refused_points(self) -> np.ndarray:
        """Refused training points of the first (far-field) chart (8a shim)."""
        return self.charts[0].refused_points

    # ---- Construction from the exact engine ---------------------------

    @classmethod
    def from_engine(cls, *, gamma_range: tuple[float, float],
                    s_range: tuple[float, float],
                    d_range: tuple[float, float],
                    w_range: tuple[float, float],
                    arc_theta_lo: float, arc_theta_hi: float,
                    arc_branch: int,
                    n_gamma: int = _DEFAULT_PARAM_NODES,
                    n_s: int = _DEFAULT_PARAM_NODES,
                    n_d: int = _DEFAULT_PARAM_NODES,
                    w_nodes_per_decade: int = _DEFAULT_W_NODES_PER_DECADE,
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
        ``tau_c``-demodulated SACR-C envelope ``E`` with the switch ON.  The
        coordinate is unchanged (far-field-smooth ``(s, d)``); only the
        ENVELOPE LABEL differs, and the tag is stamped on the chart so the
        serve mirror dispatches the SACR-C reconstruction
        (`channels.reconstruct_from_envelope`).  Interior node builds also
        collect the parked carrier ``critical_source`` and assert basin
        continuity across the tile (`_assert_carrier_continuity`).

        Far-field-smooth grid (Build 1e-farfield WP2). The two spatial axes
        are the caustic ARC LENGTH ``s`` (tangential) and the SIGNED
        perpendicular distance ``d`` (radial, ``+`` outside / ``-`` inside);
        the transform is the gamma-resolved arc-length map
        (`_caustic_arclength_map`) over the arc ``[arc_theta_lo,
        arc_theta_hi]`` on branch ``arc_branch``.  Each grid node
        ``(gamma, s, d)`` is mapped to a physical eigenframe source via
        `_from_farfield_smooth` at that node's OWN gamma BEFORE the engine
        call.  ``s`` is smooth across the C2 curvature kinks of the raw
        ``theta_c`` angle, so no cusp-node union is needed.  A parameter
        point that refuses at any ``w`` node (or returns a non-finite
        envelope, or whose reconstructed source lies on the medial axis) is
        recorded refused (in ``(gamma, s, d)`` coordinates) and left as
        zeros in the value arrays.

        Cusp/tie rejection (WP1 guards).  The arc is rejected outright
        (`LensDomainError` propagated to the caller) if it spans a caustic
        cusp at any gamma node (`_reject_if_cusp_spanning`), so the
        arc-length map inverts a strictly monotone column.  A per-node
        near-tied-foot (medial-axis) guard (`_reject_if_near_tied_foot`)
        rejects an individual node whose reconstructed source has an
        ambiguous foot.

        Domain contract (exterior-only): the far-field label subtracts the
        resolved geometric-optics images with the switch forced on for
        every real channel, so it is small and smooth ONLY where the box
        lies wholly in the caustic EXTERIOR.  Near-caustic domains belong to
        TUBE charts; production tiling enforces the exterior contract, which
        this method does NOT itself guard -- callers must supply an exterior
        arc.

        Parameters
        ----------
        gamma_range : tuple[float, float]
            External-shear axis bounds ``(low, high)``.
        s_range, d_range : tuple[float, float]
            Far-field-smooth spatial axis bounds ``(low, high)``: ``s`` is
            caustic arc length (dimensionless source-plane ``y`` units) and
            ``d`` the signed perpendicular distance (``+`` outside the
            caustic, ``-`` inside).
        w_range : tuple[float, float]
            Dimensionless-frequency bounds ``(w_min, w_max)``, both
            strictly positive.
        arc_theta_lo, arc_theta_hi : float
            Eigenframe caustic polar-angle endpoints of the arc (radians),
            ``arc_theta_lo < arc_theta_hi``, defining the arc-length map.
        arc_branch : int
            Square-root branch ``+-1`` of the arc (``+1`` positive-parity
            astroid; ``+-1`` a macro-saddle deltoid edge).
        n_gamma, n_s, n_d : int, optional
            Nodes per parameter axis (default 7; Professor Q2 sizing).
        w_nodes_per_decade : int, optional
            Density of the dense log-w training axis (default 15).
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
        LensDomainError
            If the arc spans a caustic cusp (or a macro-saddle wedge edge)
            at any gamma node (`_reject_if_cusp_spanning`); the caller must
            subdivide the tile.
        CarrierDiscontinuityError
            For an interior chart whose tile straddles a critical-basin
            flip, or for an exterior chart whose frame-invariant far-field
            label winds faster than the Nyquist ``pi/2`` per node gap
            (`_assert_farfield_carrier_continuity`); the tile must be
            subdivided.
        """
        definition = _validate_farfield_definition(definition, 'chart build')
        interior = definition in _KNOWN_INTERIOR_DEFINITIONS
        arc_branch = int(arc_branch)
        arc_theta_lo = float(arc_theta_lo)
        arc_theta_hi = float(arc_theta_hi)
        log_w_grid = _log_w_grid(w_range, w_nodes_per_decade)
        gamma_grid = _log_reach_gamma_axis(gamma_range, n_gamma, 'gamma')
        s_grid = _uniform_axis(s_range, n_s, 's')
        d_grid = _uniform_axis(d_range, n_d, 'd')
        # Cusp/wedge-edge rejection: the arc-length coordinate is only
        # single-valued on a cusp-free arc.  Reject the whole arc (propagate
        # LensDomainError to the tiler) if any gamma node's arc spans a
        # caustic cusp -- s is smooth across the C2 kinks the raw theta_c
        # angle produced, so there is NO cusp-node union to add here.
        for gamma in gamma_grid:
            _reject_if_cusp_spanning(float(gamma), arc_theta_lo,
                                     arc_theta_hi, arc_branch)
        arc_map = _caustic_arclength_map(gamma_grid, arc_theta_lo,
                                         arc_theta_hi, arc_branch)
        w_grid = np.exp(log_w_grid)

        shape = (log_w_grid.size, gamma_grid.size, s_grid.size, d_grid.size)
        envelope_real = np.zeros(shape, dtype=float)
        envelope_imag = np.zeros(shape, dtype=float)
        refused: list[tuple[float, float, float]] = []
        # Parked-carrier ``critical_source`` per node (interior only), for
        # the basin-continuity guard; NaN marks a refused/unfilled node.
        carrier = np.full((gamma_grid.size, s_grid.size,
                           d_grid.size, 2), np.nan, dtype=float)

        for i_g, gamma in enumerate(gamma_grid):
            for i_s, s in enumerate(s_grid):
                for i_d, d in enumerate(d_grid):
                    # Fresh tracker per point -> deterministic initial
                    # labeling; the envelope is well-defined per point and
                    # independent of label continuation.
                    channels = ChangRefsdalChannels(w_grid)
                    try:
                        # Far-field-smooth node -> physical eigenframe
                        # source at the node's OWN gamma.  The medial-axis
                        # guard rejects a node whose foot is ambiguous; the
                        # inverse map / engine may refuse at the parity wall
                        # -- either records the node refused (the documented
                        # `from_engine` contract) instead of crashing.
                        y1_eig, y2_eig = _from_farfield_smooth(
                            float(gamma), float(s), float(d), arc_map,
                            arc_branch)
                        _reject_if_near_tied_foot(
                            float(gamma), float(y1_eig), float(y2_eig))
                        partition = channels.evaluate(
                            gamma=float(gamma),
                            y=(y1_eig, y2_eig),
                            beta=0.0, kappa=0.0)
                    except _REFUSAL_ERRORS:
                        refused.append((float(gamma), float(s), float(d)))
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
                        refused.append((float(gamma), float(s), float(d)))
                        continue
                    envelope_real[:, i_g, i_s, i_d] = env.real
                    envelope_imag[:, i_g, i_s, i_d] = env.imag
                    carrier[i_g, i_s, i_d] = partition.critical_source

        if interior:
            # Interpolator hygiene: the tau_c-demodulated envelope is smooth
            # only within one nearest-caustic basin.  Reject a tile that
            # straddles a medial ridge (basin flip) for subdivision.
            _assert_carrier_continuity(
                carrier, gamma_grid,
                (gamma_grid.size, s_grid.size, d_grid.size))
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
                (gamma_grid.size, s_grid.size, d_grid.size))

        refused_points = (np.array(refused, dtype=float) if refused
                          else np.empty((0, 3), dtype=float))
        image_count, parity = cls._box_region_labels(
            gamma_grid, s_grid, d_grid, arc_map, arc_branch)
        chart = FarFieldChart.from_values(
            gamma_grid=gamma_grid, s_grid=s_grid,
            d_grid=d_grid, log_w_grid=log_w_grid,
            envelope_real=envelope_real, envelope_imag=envelope_imag,
            arc_map=arc_map,
            image_count=image_count, parity=parity,
            eta_overlap_min=_DEFAULT_CAUSTIC_FLOOR,
            refused_points=refused_points,
            envelope_definition=definition)
        provenance = cls._build_provenance(
            gamma_range, s_range, d_range, w_range, shape,
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
                         w_nodes_per_decade: int = _DEFAULT_W_NODES_PER_DECADE
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
        gamma_grid = _log_reach_gamma_axis(gamma_range, n_gamma, 'gamma')
        rho_lobe_grid = _uniform_axis(rho_lobe_range, n_rho, 'rho_lobe')

        # Wedge-edge s-coordinate: theta_local nodes are placed as images of
        # a uniform s = sqrt(span) - sqrt(theta_max - theta) grid, which
        # concentrates nodes near the wedge edge (theta_max) and makes the
        # cusp a regular point of the interpolation coordinate.
        theta_min, theta_max = theta_local_range
        span = theta_max - theta_min
        s_total = float(np.sqrt(span))
        # Dense theta -> s map for serve-time interpolation.
        theta_fine = np.linspace(theta_min, theta_max, _LOBE_ARC_MAP_SIZE)
        s_fine = s_total - np.sqrt(theta_max - theta_fine)
        theta_to_s = np.vstack([theta_fine, s_fine])
        # Place theta_local nodes as images of uniform s.
        s_grid_nodes = np.linspace(0.0, s_total, n_theta)
        theta_local_grid = theta_max - (s_total - s_grid_nodes) ** 2
        # Force exact endpoints (guard against FP drift).
        theta_local_grid[0] = theta_min
        theta_local_grid[-1] = theta_max

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

        for i_g, gamma in enumerate(gamma_grid):
            for i_rho, rho_lobe in enumerate(rho_lobe_grid):
                for i_th, theta_local in enumerate(theta_local_grid):
                    channels = ChangRefsdalChannels(w_grid)
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
            envelope_definition=_INTERIOR_ENVELOPE_DEFINITION,
            theta_to_s=theta_to_s, s_grid=s_grid_nodes)
        provenance = cls._build_lobe_provenance(
            gamma_range, rho_lobe_range, theta_local_range, w_range, shape,
            envelope_real, envelope_imag, centroid, other_centroid,
            corridor_half)
        return cls([chart], provenance)

    @staticmethod
    def _box_region_labels(gamma_grid: np.ndarray, s_grid: np.ndarray,
                           d_grid: np.ndarray, arc_map: _FarFieldArcMap,
                           branch: int) -> tuple[int | None, int | None]:
        """Real-image count and parity of the box's single region.

        The box lies inside one image-count region, so the region label is
        read once from a cheap ``w``-independent
        `ChangRefsdalChannels.geometry_partition` at the box centre.  The
        centre is a far-field-smooth ``(s, d)`` node, mapped to a physical
        eigenframe source at the central ``gamma`` (`_from_farfield_smooth`)
        before the geometry call.  Parity is deterministic in ``gamma``
        (``+1`` for ``gamma < 1``, ``-1`` for ``gamma > 1``).

        Returns ``(None, None)`` when the box-centre map refuses -- e.g. a box
        whose centre ``gamma`` is exactly ``1.0`` hits the ``_caustic_reach``
        parity wall (`LensDomainError`).  The chart then records unknown labels
        (handled conservatively downstream) instead of crashing construction.
        """
        gamma_c = 0.5 * float(gamma_grid[0] + gamma_grid[-1])
        s_c = 0.5 * float(s_grid[0] + s_grid[-1])
        d_c = 0.5 * float(d_grid[0] + d_grid[-1])
        try:
            y1_c, y2_c = _from_farfield_smooth(gamma_c, s_c, d_c, arc_map,
                                               int(branch))
            geom = ChangRefsdalChannels(
                np.array([1.0, 2.0])).geometry_partition(
                    gamma=gamma_c, y=(y1_c, y2_c), beta=0.0, kappa=0.0)
        except _REFUSAL_ERRORS:
            return None, None
        parity = 1 if gamma_c < 1.0 else -1
        return int(geom.real_mask.sum()), parity

    @staticmethod
    def _build_provenance(gamma_range: tuple[float, float],
                          s_range: tuple[float, float],
                          d_range: tuple[float, float],
                          w_range: tuple[float, float],
                          shape: tuple[int, int, int, int],
                          envelope_real: np.ndarray,
                          envelope_imag: np.ndarray) -> dict:
        """Build the minimal provenance dict, including a short train hash.

        The spatial ranges are the far-field-smooth ``(s, d)`` axis bounds
        (arc length ``s`` and signed perpendicular distance ``d``; Build
        1e-farfield WP2); the ``axis_schema`` tag records the coordinate
        convention so a stale caustic-fixed / raw-eigenframe artifact is
        distinguishable at load.
        """
        hasher = hashlib.sha1()
        hasher.update(np.ascontiguousarray(envelope_real).tobytes())
        hasher.update(np.ascontiguousarray(envelope_imag).tobytes())
        n_w, n_gamma, n_s, n_d = shape
        return {
            'gamma_range': [float(gamma_range[0]), float(gamma_range[1])],
            's_range': [float(s_range[0]), float(s_range[1])],
            'd_range': [float(d_range[0]), float(d_range[1])],
            'axis_schema': _FARFIELD_AXIS_SCHEMA,
            'w_range': [float(w_range[0]), float(w_range[1])],
            'resolution': {'n_w': int(n_w), 'n_gamma': int(n_gamma),
                           'n_s': int(n_s), 'n_d': int(n_d)},
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
        # The far-field chart query coordinate is the far-field-smooth
        # (s, d) (Build 1e-farfield WP2); it is chart-specific (it depends
        # on each chart's arc_map) so it is computed per chart INSIDE
        # select_chart / _evaluate_chart from the eigenframe source, not
        # once here.  A source whose smooth coordinate is ambiguous makes
        # the far-field charts decline and the caller uses the exact engine.
        chart = select_chart(
            self.charts, gamma=gamma, log_w_min=float(log_w.min()),
            log_w_max=float(log_w.max()), eta=eta, theta=theta,
            image_count=image_count, y1_eig=y1_eig, y2_eig=y2_eig)
        if chart is None:
            return np.zeros(w.shape, dtype=complex), False, None

        env_flat = _evaluate_chart(
            chart, gamma=gamma, eta=eta, theta=theta, log_w_query=log_w,
            y1_eig=y1_eig, y2_eig=y2_eig)
        definition = (chart.envelope_definition
                      if isinstance(chart, (FarFieldChart, LobeInteriorChart))
                      else None)
        return env_flat.reshape(w.shape), True, definition

    # ---- Legacy single-box (far-field) query --------------------------

    def in_domain(self, gamma: float, y1: float, y2: float,
                  beta: float) -> bool:
        """Whether a far-field chart serves ``(gamma, y1, y2, beta)`` in the
        caustic arc-length coordinate ``(s, d)`` (the 8a domain gate).

        Rotates the source into the shear eigenframe and tests, over the
        far-field charts, whether some chart's gamma / arc-length ``s`` /
        perpendicular-distance ``d`` box contains the eigenframe point and
        the point clears that chart's exclusion balls -- the exact 8a
        single-box gate, generalized to the far-field charts of a global
        surrogate.  It does NOT consult ``eta`` / image count (use `serve`
        for the full guard stack).

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
        return self._farfield_raw_chart(gamma, y1_eig, y2_eig) is not None

    def _farfield_raw_chart(self, gamma: float, y1_eig: float,
                            y2_eig: float):
        """First far-field chart whose box contains the eigenframe source.

        The far-field charts are gridded over the far-field-smooth source
        coordinate ``(s, d)`` (Build 1e-farfield WP2), so each chart maps the
        eigenframe source to its own ``(s, d)`` through its stored
        ``arc_map`` (declining on any smooth-map refusal) and tests
        containment on ``s_grid`` / ``d_grid`` plus the exclusion balls.
        """
        for chart in self.charts:
            if not isinstance(chart, FarFieldChart):
                continue
            # Macro-saddle far-field charts are exact-engine-only, including
            # stale manual/loaded records that carry otherwise valid arrays.
            if chart.parity != 1:
                continue
            if not (chart.gamma_grid[0] <= gamma <= chart.gamma_grid[-1]):
                continue
            try:
                s, d = _to_farfield_smooth(gamma, y1_eig, y2_eig,
                                           chart.arc_map, chart.arc_map.branch)
            except LensDomainError:
                continue
            if not (chart.s_grid[0] <= s <= chart.s_grid[-1]
                    and chart.d_grid[0] <= d <= chart.d_grid[-1]):
                continue
            if _in_exclusion_ball(chart, gamma, s, d):
                continue
            return chart
        return None

    def envelope(self, w_array: np.ndarray, gamma: float, y1: float,
                 y2: float, beta: float) -> tuple[np.ndarray, bool]:
        """Legacy 8a far-field envelope query (arc-length ``(s, d)`` lookup).

        Preserves the 8a call signature: rotates ``(y1, y2)`` into the
        eigenframe, selects the first far-field chart whose gamma /
        arc-length ``s`` / perpendicular-distance ``d`` box contains the
        point (exclusion balls honoured), and evaluates its splines over
        ``w``.  Returns ``served=False`` (zeros) when no far-field chart
        contains the point or any ``w`` is outside that chart's band.  Does
        NOT run the tube/eta guard stack -- use `serve` for the full global
        query.

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
        chart = self._farfield_raw_chart(gamma, y1_eig, y2_eig)
        if chart is None:
            return np.zeros(w.shape, dtype=complex), False

        w_flat = np.atleast_1d(w).ravel()
        w_min = float(np.exp(chart.log_w_grid[0]))
        w_max = float(np.exp(chart.log_w_grid[-1]))
        if w_flat.size == 0 or not np.all(
                (w_flat >= w_min) & (w_flat <= w_max)):
            return np.zeros(w.shape, dtype=complex), False

        env_flat = _evaluate_chart(
            chart, gamma=gamma, eta=float('nan'), theta=float('nan'),
            log_w_query=np.log(w_flat), y1_eig=y1_eig, y2_eig=y2_eig)
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
        """Hard-refuse an 8a single-box artifact (no ``n_charts`` key).

        A pre-multichart single-box artifact stores raw-eigenframe /
        caustic-fixed spatial axes and the OLD far-field envelope label; it
        cannot be reconstructed under the far-field-smooth ``(s, d)`` serve
        mirror (Build 1e-farfield WP2) and is refused loudly rather than
        served at the wrong coordinate.  The definition and axis-schema
        validators below both hard-refuse an absent/unknown tag, so control
        never reaches the trailing guard; it is kept as a defensive backstop.
        """
        # A legacy single-box artifact predates the far-field
        # envelope-definition tag (Build 8g-b) and the far-field-smooth axis
        # schema (Build 1e-farfield WP2): both validators hard-refuse an
        # absent/unknown tag, so a legacy artifact fails loudly here.
        tag = (str(data['envelope_definition'])
               if 'envelope_definition' in data.files else None)
        _validate_farfield_definition(tag, 'legacy single-box artifact')
        axis_tag = (str(data['axis_schema'])
                    if 'axis_schema' in data.files else None)
        _validate_farfield_axis_schema(
            axis_tag, 'legacy single-box artifact')
        raise ValueError(
            'Legacy single-box artifact carries no per-chart schema and '
            'cannot be reconstructed as a far-field-smooth (s, d) chart; '
            'retrain the surrogate (Build 1e-farfield WP2).')


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
    set. Positive-parity far-field charts require
    ``'farfield_arclength_s_perp_d_framewinv'`` and use the gamma-resolved
    far-field-smooth ``(s, d)`` transform. A chart trained on raw eigenframe
    or retired caustic-fixed axes, or with a frame-dependent stored label,
    would be queried or reconstructed in the wrong convention and could
    return a finite-but-wrong amplification.
    """
    return _validate_axis_schema(
        tag, _KNOWN_FARFIELD_AXIS_SCHEMAS, f'Far-field {artifact_label}')


def _validate_lobe_axis_schema(tag, artifact_label: str) -> str:
    """Hard-refuse a lobe-interior chart with an absent or unknown schema.

    Thin wrapper over `_validate_axis_schema` binding the lobe known set.
    Macro-saddle lobe-interior charts are queried on lobe-local
    ``(rho_lobe, theta_local)`` coordinates centred on the lobe centroid; a
    chart stamped with the far-field smooth tag, an origin-centred axis, or
    an old lobe tag would be reconstructed at the wrong coordinate and must
    hard-refuse at load.
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
        arrays = {prefix + 'theta_to_s': chart.theta_to_s}
    elif isinstance(chart, LobeInteriorChart):
        # Additive lobe branch (WP1): the persisted record carries the lobe
        # frame (centroid, other_centroid, boundary_theta/boundary_r as
        # arrays; corridor_half scalar in meta) alongside the interior spline.
        # The lobe axis-schema tag makes a mislabeled/old artifact hard-refuse
        # at load rather than reconstruct a finite-but-wrong F.
        lobe_schema = (_LOBE_AXIS_SCHEMA if chart.theta_to_s is not None
                       else _LOBE_AXIS_SCHEMA_V1)
        meta = {'kind': 'lobe', 'image_count': chart.image_count,
                'parity': chart.parity,
                'eta_overlap_min': chart.eta_overlap_min,
                'envelope_definition': chart.envelope_definition,
                'corridor_half': float(chart.corridor_half),
                'axis_schema': lobe_schema}
        axes = (chart.log_w_grid, chart.gamma_grid, chart.rho_lobe_grid,
                chart.theta_local_grid)
        arrays = {prefix + 'refused': chart.refused_points,
                  prefix + 'centroid': chart.centroid,
                  prefix + 'other_centroid': chart.other_centroid,
                  prefix + 'boundary_theta': chart.boundary_theta,
                  prefix + 'boundary_r': chart.boundary_r}
        if chart.theta_to_s is not None:
            arrays[prefix + 'theta_to_s'] = chart.theta_to_s
    else:
        # Far-field exterior branch (Build 1e-farfield WP2): the spatial axes
        # are the far-field-smooth arc length ``s`` and signed perpendicular
        # distance ``d``, and the gamma-resolved arc-length map that defines
        # the ``theta <-> s`` transform is persisted alongside them (the axes
        # are meaningless without it).  The arc's branch and angular endpoints
        # ride in meta; the 2-D table (gamma_nodes, theta_fine, s_table) rides
        # in arrays.
        meta = {'kind': 'farfield', 'image_count': chart.image_count,
                'parity': chart.parity,
                'eta_overlap_min': chart.eta_overlap_min,
                'envelope_definition': chart.envelope_definition,
                'axis_schema': _FARFIELD_AXIS_SCHEMA,
                'arc_branch': int(chart.arc_map.branch),
                'arc_theta_lo': float(chart.arc_map.theta_lo),
                'arc_theta_hi': float(chart.arc_map.theta_hi)}
        axes = (chart.log_w_grid, chart.gamma_grid, chart.s_grid,
                chart.d_grid)
        arrays = {prefix + 'refused': chart.refused_points,
                  prefix + 'arc_gamma_nodes': chart.arc_map.gamma_nodes,
                  prefix + 'arc_theta_fine': chart.arc_map.theta_fine,
                  prefix + 'arc_s_table': chart.arc_map.s_table}
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
            cusp_windows=[tuple(win) for win in meta['cusp_windows']],
            theta_to_s=data[prefix + 'theta_to_s'])
    if meta['kind'] == 'lobe':
        # Additive lobe branch (WP1): a lobe chart demands the lobe axis
        # schema, so a mislabeled/old artifact hard-refuses here rather than
        # reconstructing at the wrong (origin-centred or far-field) coordinate.
        _validate_lobe_axis_schema(meta.get('axis_schema'), f'chart {index}')
        definition = _validate_farfield_definition(
            meta.get('envelope_definition'), f'chart {index}')
        # Schema-dependent theta_to_s loading: the V1 schema (raw theta_local
        # spline) has no map — tolerate absence.  The current schema (sqrt-
        # edge coordinate) REQUIRES the map; a missing key hard-refuses.
        schema = meta.get('axis_schema')
        if schema == _LOBE_AXIS_SCHEMA_V1:
            # V1 schema (raw theta_local spline) may lack the map.
            key = prefix + 'theta_to_s'
            theta_to_s = data[key] if key in data else None
        else:
            theta_to_s = data[prefix + 'theta_to_s']
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
            envelope_definition=definition,
            theta_to_s=theta_to_s)
    definition = _validate_farfield_definition(
        meta.get('envelope_definition'), f'chart {index}')
    _validate_farfield_axis_schema(
        meta.get('axis_schema'), f'chart {index}')
    # Rebuild the gamma-resolved arc-length map from the persisted table and
    # arc metadata; a far-field chart is meaningless without it, so a missing
    # key raises loudly here (no legacy caustic-fixed branch -- Build
    # 1e-farfield WP2).
    arc_map = _FarFieldArcMap(
        gamma_nodes=np.ascontiguousarray(data[prefix + 'arc_gamma_nodes'],
                                         dtype=float),
        theta_fine=np.ascontiguousarray(data[prefix + 'arc_theta_fine'],
                                        dtype=float),
        s_table=np.ascontiguousarray(data[prefix + 'arc_s_table'],
                                     dtype=float),
        branch=int(meta['arc_branch']),
        theta_lo=float(meta['arc_theta_lo']),
        theta_hi=float(meta['arc_theta_hi']))
    return FarFieldChart._assemble(
        gamma_grid=gamma_grid, s_grid=p1_grid, d_grid=p2_grid,
        log_w_grid=log_w_grid, real_coeffs=real_coeffs,
        imag_coeffs=imag_coeffs, knots=knots,
        image_count=meta['image_count'], parity=meta['parity'],
        eta_overlap_min=meta['eta_overlap_min'],
        refused_points=data[prefix + 'refused'],
        arc_map=arc_map,
        envelope_definition=definition)
