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

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

from cogwheel.lensing.chang_refsdal import (ChangRefsdalChannels,
                                            farfield_envelope_from_partition)
from cogwheel.lensing.chang_refsdal.geometry import LensDomainError
from cogwheel.lensing.chang_refsdal.operator import CancellationError
from cogwheel.lensing.chang_refsdal._schwinger import (
    SchwingerCertificationError)

# The engine's named refusals.  Any of these at ANY w node marks the whole
# parameter grid point refused (per-w refusal propagation, Professor Q4).
_REFUSAL_ERRORS = (LensDomainError, CancellationError,
                   SchwingerCertificationError)

# Default training resolutions (Professor Q2 sizing).
_DEFAULT_W_NODES_PER_DECADE = 15
_DEFAULT_PARAM_NODES = 7

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
# A far-field chart is trained on the label
# ``E_ff = F - sum_{a real} H_a e^{1j w tau_a}``
# (`channels.farfield_envelope_from_partition`, Build 8g-b): the full
# post-geometric-optics remainder with the criticality switch forced to 1
# on every real channel and NO ``tau_c`` demodulation carrier.  The
# serving side must mirror this exactly (add the full real-channel kernel
# sum back with ``critical_delay = 0``) or the reconstructed ``F`` would
# not match the label.  The loader hard-refuses a far-field chart whose
# tag is absent or unknown: the v1/v2 partial artifacts predate the tag
# and were trained on the OLD (lobe-flipping) caustic-region envelope, so
# reconstructing them under the new definition would be finite-but-wrong.
_FARFIELD_ENVELOPE_DEFINITION = 'farfield_full_kernel_sum'
_KNOWN_FARFIELD_DEFINITIONS = frozenset({_FARFIELD_ENVELOPE_DEFINITION})

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


# ---- Charts -----------------------------------------------------------


@dataclass(frozen=True, eq=False)
class FarFieldChart:
    """Raw-eigenframe-coordinate envelope chart, valid away from a caustic.

    Interpolates ``E(w)`` over ``(log w, gamma, y1_eig, y2_eig)`` for one
    image-count region.  This is the single-box interpolant the 8a
    surrogate shipped; a global surrogate holds one per region.  Serve
    only where ``eta > eta_overlap_min`` (bounded away from the caustic)
    and the candidate matches ``image_count``.

    Attributes
    ----------
    gamma_grid, y1_grid, y2_grid, log_w_grid : np.ndarray
        1-D strictly increasing training axes.
    real_coeffs, imag_coeffs : np.ndarray
        Cubic B-spline coefficient tensors, axes ``(log w, gamma, y1,
        y2)``.
    knots : tuple of np.ndarray
        Knot vectors ``(t_logw, t_gamma, t_y1, t_y2)``.
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
        Shape ``(n, 3)`` eigenframe ``(gamma, y1_eig, y2_eig)`` training
        points the engine refused; the exclusion-ball gate rejects
        queries within one grid spacing of any of them.
    param_spacing : np.ndarray
        Shape ``(3,)`` mean spacing of ``(gamma, y1, y2)`` for the
        exclusion-ball normalization.
    envelope_definition : str
        Tag naming the label the chart's envelope encodes (Build 8g-b).
        Persisted in the npz meta and checked on load; the serving side
        dispatches the reconstruction on it.  Fresh charts default to
        `_FARFIELD_ENVELOPE_DEFINITION`.
    """

    gamma_grid: np.ndarray
    y1_grid: np.ndarray
    y2_grid: np.ndarray
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
    def from_values(cls, *, gamma_grid: np.ndarray, y1_grid: np.ndarray,
                    y2_grid: np.ndarray, log_w_grid: np.ndarray,
                    envelope_real: np.ndarray, envelope_imag: np.ndarray,
                    image_count: int | None, parity: int | None,
                    eta_overlap_min: float = _DEFAULT_CAUSTIC_FLOOR,
                    refused_points: np.ndarray | None = None
                    ) -> 'FarFieldChart':
        """Build a far-field chart by fitting splines to a value tensor.

        Parameters
        ----------
        gamma_grid, y1_grid, y2_grid, log_w_grid : np.ndarray
            1-D strictly increasing training axes.
        envelope_real, envelope_imag : np.ndarray
            Shape ``(n_w, n_gamma, n_y1, n_y2)`` real/imag envelope values.
        image_count, parity : int or None
            Region labels (``None`` if unrecorded).
        eta_overlap_min : float, optional
            Minimum caustic distance served (default the caustic floor).
        refused_points : np.ndarray, optional
            Refused eigenframe ``(gamma, y1, y2)`` training points.
        """
        gamma_grid = _validate_axis(gamma_grid, 'gamma_grid')
        y1_grid = _validate_axis(y1_grid, 'y1_grid')
        y2_grid = _validate_axis(y2_grid, 'y2_grid')
        log_w_grid = _validate_axis(log_w_grid, 'log_w_grid')
        expected = (log_w_grid.size, gamma_grid.size, y1_grid.size,
                    y2_grid.size)
        _check_value_shape(envelope_real, envelope_imag, expected)
        real_c, imag_c, knots = _fit_tensor_spline(
            (log_w_grid, gamma_grid, y1_grid, y2_grid),
            envelope_real, envelope_imag)
        return cls._assemble(
            gamma_grid, y1_grid, y2_grid, log_w_grid, real_c, imag_c, knots,
            image_count, parity, eta_overlap_min, refused_points)

    @classmethod
    def _assemble(cls, gamma_grid, y1_grid, y2_grid, log_w_grid, real_coeffs,
                  imag_coeffs, knots, image_count, parity, eta_overlap_min,
                  refused_points,
                  envelope_definition=_FARFIELD_ENVELOPE_DEFINITION
                  ) -> 'FarFieldChart':
        """Assemble a chart from prebuilt coefficient tensors and knots."""
        param_spacing = np.array([
            float(np.mean(np.diff(gamma_grid))),
            float(np.mean(np.diff(y1_grid))),
            float(np.mean(np.diff(y2_grid)))])
        return cls(
            gamma_grid=_validate_axis(gamma_grid, 'gamma_grid'),
            y1_grid=_validate_axis(y1_grid, 'y1_grid'),
            y2_grid=_validate_axis(y2_grid, 'y2_grid'),
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


def _in_exclusion_ball(chart: FarFieldChart, gamma: float, y1_eig: float,
                       y2_eig: float) -> bool:
    """Whether ``(gamma, y1_eig, y2_eig)`` is within a refusal ball."""
    refused = chart.refused_points
    if refused.shape[0] == 0:
        return False
    query = np.array([gamma, y1_eig, y2_eig])
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
    """Whether a far-field chart serves this candidate (steps 1,3,5,7)."""
    # (1) certified-box containment on gamma, log w, and the source.
    if not (chart.gamma_grid[0] <= gamma <= chart.gamma_grid[-1]):
        return False
    if not _log_w_band_inside(chart, log_w_min, log_w_max):
        return False
    if not (chart.y1_grid[0] <= y1_eig <= chart.y1_grid[-1]
            and chart.y2_grid[0] <= y2_eig <= chart.y2_grid[-1]):
        return False
    # (3) inherited engine-refusal exclusion ball.
    if _in_exclusion_ball(chart, gamma, y1_eig, y2_eig):
        return False
    # (5) image-count guard.
    if chart.image_count is not None and image_count != chart.image_count:
        return False
    # (7) far-field priority: only away from the caustic.
    if eta <= chart.eta_overlap_min:
        return False
    return True


def select_chart(charts, *, gamma: float, log_w_min: float, log_w_max: float,
                 eta: float, theta: float, image_count: int, y1_eig: float,
                 y2_eig: float):
    """Deterministically pick the chart to serve a candidate, or ``None``.

    The guard stack (Professor Q7), executed in order, keys ONLY on the
    certified physical quantities ``gamma``, ``eta`` and ``image_count``
    -- never on the gauge angle ``theta`` except for the cusp-window
    exclusion test (F017).  Any fall-through returns ``None`` so the
    caller uses the exact engine.

    Order: (2) gamma guard band near ``gamma = 1`` -> fall through; then
    TUBE charts have priority over FAR-FIELD charts (step 7).  Per chart:
    (1) certified-box containment on ``(gamma, log w)``; (3) far-field
    engine-refusal exclusion balls; (5) image-count match; (6) cusp
    exclusion / ``eta`` floor; (7) tube when ``eta in [eta_floor,
    eta_max]``, else far-field when ``eta > eta_overlap_min``.

    Parameters
    ----------
    charts : sequence of TubeChart or FarFieldChart
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
    y1_eig, y2_eig : float
        Source position rotated into the shear eigenframe.

    Returns
    -------
    TubeChart or FarFieldChart or None
        The selected chart, or ``None`` to fall through to the engine.
    """
    # (2) gamma guard band around the det-A = 0 parity boundary.
    if abs(gamma - 1.0) < _GAMMA_GUARD_BAND:
        return None
    # (7) priority: tube charts first, then far-field charts.
    for chart in charts:
        if isinstance(chart, TubeChart) and _tube_serves(
                chart, gamma, log_w_min, log_w_max, eta, theta, image_count):
            return chart
    for chart in charts:
        if isinstance(chart, FarFieldChart) and _farfield_serves(
                chart, gamma, log_w_min, log_w_max, eta, image_count,
                y1_eig, y2_eig):
            return chart
    return None


def _evaluate_chart(chart, gamma: float, y1_eig: float, y2_eig: float,
                    eta: float, theta: float,
                    log_w_query: np.ndarray) -> np.ndarray:
    """Evaluate the selected chart's complex envelope over ``log_w_query``."""
    if isinstance(chart, TubeChart):
        v1 = float(np.sqrt(eta))
        v2 = _theta_into_frame(theta, float(chart.theta_grid[0]))
    else:
        v1, v2 = y1_eig, y2_eig
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
            if not isinstance(chart, (FarFieldChart, TubeChart)):
                raise ValueError(
                    'charts must be FarFieldChart or TubeChart instances; '
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
    def y1_grid(self) -> np.ndarray:
        """Eigenframe ``y1`` axis of the first (far-field) chart (8a shim)."""
        return self.charts[0].y1_grid

    @property
    def y2_grid(self) -> np.ndarray:
        """Eigenframe ``y2`` axis of the first (far-field) chart (8a shim)."""
        return self.charts[0].y2_grid

    @property
    def refused_points(self) -> np.ndarray:
        """Refused training points of the first (far-field) chart (8a shim)."""
        return self.charts[0].refused_points

    # ---- Construction from the exact engine ---------------------------

    @classmethod
    def from_engine(cls, *, gamma_range: tuple[float, float],
                    y1_range: tuple[float, float],
                    y2_range: tuple[float, float],
                    w_range: tuple[float, float],
                    n_gamma: int = _DEFAULT_PARAM_NODES,
                    n_y1: int = _DEFAULT_PARAM_NODES,
                    n_y2: int = _DEFAULT_PARAM_NODES,
                    w_nodes_per_decade: int = _DEFAULT_W_NODES_PER_DECADE,
                    max_order: int | None = None
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
        and leave a resolved image un-subtracted (Build 8g-b).  A parameter
        point that refuses at any ``w`` node (or returns a non-finite
        envelope) is recorded as refused and left as zeros in the value
        arrays.  The box must lie wholly inside one image-count region
        with caustic distance bounded away from zero; the region's real
        image count is read once at the box centre and stored as the
        chart's `image_count` label.

        Domain contract (exterior-only): the far-field label subtracts the
        resolved geometric-optics images with the switch forced on for
        every real channel, so it is small and smooth ONLY where the box
        lies wholly in the caustic EXTERIOR -- every corner outside
        ``caustic_reach + eta_max`` for its gamma range.  Near the caustic
        an image is not fully resolved; forcing its switch on leaves an
        un-subtracted oscillatory term, ``E_ff`` grows and a coarse spline
        fits it poorly.  Near-caustic domains therefore belong to TUBE
        charts (the caustic-region ``partition.envelope`` label), not to a
        far-field chart built here; production tiling enforces this by
        admitting only tiles wholly outside ``caustic_reach + eta_max``.
        This method applies the far-field label unconditionally and does
        NOT itself guard the exterior contract -- callers must supply an
        exterior box.

        Parameters
        ----------
        gamma_range : tuple[float, float]
            External-shear axis bounds ``(low, high)``.
        y1_range, y2_range : tuple[float, float]
            Eigenframe source-position axis bounds ``(low, high)``.
        w_range : tuple[float, float]
            Dimensionless-frequency bounds ``(w_min, w_max)``, both
            strictly positive.
        n_gamma, n_y1, n_y2 : int, optional
            Nodes per parameter axis (default 7; Professor Q2 sizing).
        w_nodes_per_decade : int, optional
            Density of the dense log-w training axis (default 15).
        max_order : int, optional
            Operator-series order cap forwarded to `ChangRefsdalChannels`.

        Returns
        -------
        LensAmplificationSurrogate
            The trained single-chart surrogate.
        """
        log_w_grid = _log_w_grid(w_range, w_nodes_per_decade)
        gamma_grid = _uniform_axis(gamma_range, n_gamma, 'gamma')
        y1_grid = _uniform_axis(y1_range, n_y1, 'y1')
        y2_grid = _uniform_axis(y2_range, n_y2, 'y2')
        w_grid = np.exp(log_w_grid)

        shape = (log_w_grid.size, gamma_grid.size, y1_grid.size, y2_grid.size)
        envelope_real = np.zeros(shape, dtype=float)
        envelope_imag = np.zeros(shape, dtype=float)
        refused: list[tuple[float, float, float]] = []

        channels_kwargs = {} if max_order is None else {'max_order': max_order}
        for i_g, gamma in enumerate(gamma_grid):
            for i_y1, y1_eig in enumerate(y1_grid):
                for i_y2, y2_eig in enumerate(y2_grid):
                    # Fresh tracker per point -> deterministic initial
                    # labeling; the envelope is well-defined per point and
                    # independent of label continuation.
                    channels = ChangRefsdalChannels(w_grid, **channels_kwargs)
                    try:
                        partition = channels.evaluate(
                            gamma=float(gamma),
                            y=(float(y1_eig), float(y2_eig)),
                            beta=0.0, kappa=0.0)
                    except _REFUSAL_ERRORS:
                        refused.append((float(gamma), float(y1_eig),
                                        float(y2_eig)))
                        continue
                    env = farfield_envelope_from_partition(partition)
                    if not np.all(np.isfinite(env)):
                        # Conservative: a non-finite envelope is treated as
                        # a refusal rather than served as a value (F005).
                        refused.append((float(gamma), float(y1_eig),
                                        float(y2_eig)))
                        continue
                    envelope_real[:, i_g, i_y1, i_y2] = env.real
                    envelope_imag[:, i_g, i_y1, i_y2] = env.imag

        refused_points = (np.array(refused, dtype=float) if refused
                          else np.empty((0, 3), dtype=float))
        image_count, parity = cls._box_region_labels(gamma_grid, y1_grid,
                                                      y2_grid)
        chart = FarFieldChart.from_values(
            gamma_grid=gamma_grid, y1_grid=y1_grid, y2_grid=y2_grid,
            log_w_grid=log_w_grid, envelope_real=envelope_real,
            envelope_imag=envelope_imag, image_count=image_count,
            parity=parity, eta_overlap_min=_DEFAULT_CAUSTIC_FLOOR,
            refused_points=refused_points)
        provenance = cls._build_provenance(
            gamma_range, y1_range, y2_range, w_range, shape,
            envelope_real, envelope_imag)
        return cls([chart], provenance)

    @staticmethod
    def _box_region_labels(gamma_grid: np.ndarray, y1_grid: np.ndarray,
                           y2_grid: np.ndarray) -> tuple[int, int]:
        """Real-image count and parity of the box's single region.

        The box lies inside one image-count region, so the region label is
        read once from a cheap ``w``-independent
        `ChangRefsdalChannels.geometry_partition` at the box centre (no
        exact total).  Parity is deterministic in ``gamma`` (``+1`` for
        ``gamma < 1``, ``-1`` for ``gamma > 1``).
        """
        gamma_c = 0.5 * float(gamma_grid[0] + gamma_grid[-1])
        y1_c = 0.5 * float(y1_grid[0] + y1_grid[-1])
        y2_c = 0.5 * float(y2_grid[0] + y2_grid[-1])
        geom = ChangRefsdalChannels(np.array([1.0, 2.0])).geometry_partition(
            gamma=gamma_c, y=(y1_c, y2_c), beta=0.0, kappa=0.0)
        parity = 1 if gamma_c < 1.0 else -1
        return int(geom.real_mask.sum()), parity

    @staticmethod
    def _build_provenance(gamma_range: tuple[float, float],
                          y1_range: tuple[float, float],
                          y2_range: tuple[float, float],
                          w_range: tuple[float, float],
                          shape: tuple[int, int, int, int],
                          envelope_real: np.ndarray,
                          envelope_imag: np.ndarray) -> dict:
        """Build the minimal provenance dict, including a short train hash."""
        hasher = hashlib.sha1()
        hasher.update(np.ascontiguousarray(envelope_real).tobytes())
        hasher.update(np.ascontiguousarray(envelope_imag).tobytes())
        n_w, n_gamma, n_y1, n_y2 = shape
        return {
            'gamma_range': [float(gamma_range[0]), float(gamma_range[1])],
            'y1_range': [float(y1_range[0]), float(y1_range[1])],
            'y2_range': [float(y2_range[0]), float(y2_range[1])],
            'w_range': [float(w_range[0]), float(w_range[1])],
            'resolution': {'n_w': int(n_w), 'n_gamma': int(n_gamma),
                           'n_y1': int(n_y1), 'n_y2': int(n_y2)},
            'beta': 0.0,
            'kappa': 0.0,
            'chart_count': 1,
            'chart_types': ['farfield'],
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
            `FarFieldChart` is served (the serving-side reconstruction
            dispatches on it), ``None`` for a `TubeChart` or when not
            served.  The persisted tag is the single dispatch signal --
            no parallel flag.
        """
        w = np.asarray(w_array, dtype=float)
        w_flat = np.atleast_1d(w).ravel()
        if w_flat.size == 0 or not np.all(w_flat > 0.0):
            return np.zeros(w.shape, dtype=complex), False, None

        log_w = np.log(w_flat)
        y1_eig, y2_eig = _rotate_to_eigenframe(y1, y2, beta)
        chart = select_chart(
            self.charts, gamma=gamma, log_w_min=float(log_w.min()),
            log_w_max=float(log_w.max()), eta=eta, theta=theta,
            image_count=image_count, y1_eig=y1_eig, y2_eig=y2_eig)
        if chart is None:
            return np.zeros(w.shape, dtype=complex), False, None

        env_flat = _evaluate_chart(chart, gamma, y1_eig, y2_eig, eta, theta,
                                   log_w)
        definition = (chart.envelope_definition
                      if isinstance(chart, FarFieldChart) else None)
        return env_flat.reshape(w.shape), True, definition

    # ---- Legacy single-box (far-field) query --------------------------

    def in_domain(self, gamma: float, y1: float, y2: float,
                  beta: float) -> bool:
        """Whether a far-field chart serves ``(gamma, y1, y2, beta)`` by raw
        coordinates (the 8a domain gate).

        Rotates the source into the shear eigenframe and tests raw-box
        containment plus the exclusion balls over the far-field charts --
        the exact 8a single-box gate, generalized to the far-field charts
        of a global surrogate.  It does NOT consult ``eta`` / image count
        (use `serve` for the full guard stack).

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
        """First far-field chart whose raw box contains the eigenframe point."""
        for chart in self.charts:
            if not isinstance(chart, FarFieldChart):
                continue
            if not (chart.gamma_grid[0] <= gamma <= chart.gamma_grid[-1]
                    and chart.y1_grid[0] <= y1_eig <= chart.y1_grid[-1]
                    and chart.y2_grid[0] <= y2_eig <= chart.y2_grid[-1]):
                continue
            if _in_exclusion_ball(chart, gamma, y1_eig, y2_eig):
                continue
            return chart
        return None

    def envelope(self, w_array: np.ndarray, gamma: float, y1: float,
                 y2: float, beta: float) -> tuple[np.ndarray, bool]:
        """Legacy 8a far-field envelope query (raw-eigenframe lookup only).

        Preserves the 8a call signature: rotates ``(y1, y2)`` into the
        eigenframe, selects the first far-field chart whose raw box
        contains the point (exclusion balls honoured), and evaluates its
        splines over ``w``.  Returns ``served=False`` (zeros) when no
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
        chart = self._farfield_raw_chart(gamma, y1_eig, y2_eig)
        if chart is None:
            return np.zeros(w.shape, dtype=complex), False

        w_flat = np.atleast_1d(w).ravel()
        w_min = float(np.exp(chart.log_w_grid[0]))
        w_max = float(np.exp(chart.log_w_grid[-1]))
        if w_flat.size == 0 or not np.all(
                (w_flat >= w_min) & (w_flat <= w_max)):
            return np.zeros(w.shape, dtype=complex), False

        env_flat = _evaluate_chart(chart, gamma, y1_eig, y2_eig,
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
        """Resolve the shipped package-data artifact path under cogwheel/data."""
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
        parity = (1 if 0.5 * float(gamma_grid[0] + gamma_grid[-1]) < 1.0
                  else -1)
        chart = FarFieldChart._assemble(
            gamma_grid=gamma_grid, y1_grid=data['y1_grid'],
            y2_grid=data['y2_grid'], log_w_grid=data['log_w_grid'],
            real_coeffs=data['real_coeffs'], imag_coeffs=data['imag_coeffs'],
            knots=(data['knot_log_w'], data['knot_gamma'], data['knot_y1'],
                   data['knot_y2']),
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
        If ``tag`` is ``None`` or not in `_KNOWN_FARFIELD_DEFINITIONS`.
    """
    if tag is None or tag not in _KNOWN_FARFIELD_DEFINITIONS:
        raise ValueError(
            f'Far-field {artifact_label} carries envelope-definition tag '
            f'{tag!r}, which is absent or unknown (known: '
            f'{sorted(_KNOWN_FARFIELD_DEFINITIONS)}).  This artifact '
            f'predates the Build 8g-b far-field envelope redefinition and '
            f'must not serve under the new reconstruction; rebuild the '
            f'surrogate.')
    return str(tag)


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
    else:
        meta = {'kind': 'farfield', 'image_count': chart.image_count,
                'parity': chart.parity,
                'eta_overlap_min': chart.eta_overlap_min,
                'envelope_definition': chart.envelope_definition}
        axes = (chart.log_w_grid, chart.gamma_grid, chart.y1_grid,
                chart.y2_grid)
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
    definition = _validate_farfield_definition(
        meta.get('envelope_definition'), f'chart {index}')
    return FarFieldChart._assemble(
        gamma_grid=gamma_grid, y1_grid=p1_grid, y2_grid=p2_grid,
        log_w_grid=log_w_grid, real_coeffs=real_coeffs,
        imag_coeffs=imag_coeffs, knots=knots,
        image_count=meta['image_count'], parity=meta['parity'],
        eta_overlap_min=meta['eta_overlap_min'],
        refused_points=data[prefix + 'refused'],
        envelope_definition=definition)
