"""
Fast tensor-cubic-spline emulator of the Chang-Refsdal envelope ``E(w)``.

WHAT
----
`LensAmplificationSurrogate` is an offline-trained emulator of the SACR-C
transition envelope ``E(w)`` -- the single beat-free, smooth object the
lensed relative-binning likelihood already interpolates
(`cogwheel.lensing.chang_refsdal.channels.ChangRefsdalPartition.envelope`).
It stores two real tensor-product cubic splines (the real and imaginary
parts of ``E`` interpolated *separately* -- never magnitude/phase, which
aliases under phase wrap) over a fixed 4-D box

    (log w, gamma, y1_eig, y2_eig)

and answers a query in well under a millisecond, with a conservative,
refusal-aware domain gate.

Evaluation mechanism (precomputed tensor cubic B-spline)
--------------------------------------------------------
Evaluation is the SAME interpolant the class shipped with -- the
tensor-product cubic B-spline with not-a-knot boundary conditions that
`scipy.interpolate.make_interp_spline` (and hence the former per-call
`scipy.interpolate.RegularGridInterpolator(method='cubic')`) produces --
but the spline is built ONCE at construction instead of being re-solved
on every query.  At construction the real/imag value tensors are turned
into cubic B-spline coefficient tensors (plus one knot vector per axis)
by successive 1-D `make_interp_spline` fits along each of the four axes.

A query exploits the fact that ``gamma``, ``y1_eig`` and ``y2_eig`` are
FIXED for a given envelope call and only ``ln w`` varies: the coefficient
tensor is contracted at the three fixed parameter coordinates down to a
single 1-D B-spline in ``ln w``, which is then evaluated at every ``w``
node.  This is a handful of de Boor evaluations over small arrays -- no
per-call linear solve -- so a served query is deterministic and well
under 0.1 ms even for hundreds of ``w`` nodes.

Because the interpolant is identical to the former per-call cubic RGI
(reproduced to ~1e-13), the emulator's accuracy is unchanged by this
speed refactor; it is gated against the exact ENGINE on held-out configs
(the reconstruction tests), never against the evaluation mechanism.

WHY
---
The exact engine is the ground truth and the fallback: it is certified
(FINDINGS F005/F013) but costs tens of milliseconds per envelope node.
The surrogate is a purely *additive* speed layer -- it never overrides a
refusal and never serves outside its validated domain.  Queries that fall
outside the trained box, or too close to a training point the engine
refused, return ``served=False`` so the caller re-evaluates with the exact
engine.  A surrogate answer where the engine would refuse would be the
F005 failure mode and is guarded against by the exclusion-ball gate.

Coordinate conventions
----------------------
- ``w`` is the dimensionless lensing frequency
  ``w = 8*pi*G*M_lens*(1+z_lens)*f/c^3``; the log-w training axis is
  ``ln w`` (natural log).  Delays ``tau`` are in seconds (they enter the
  envelope only through the demodulated phase and are not axes here).
- ``kappa`` is fixed at 0 (sampled space): the mass-sheet degeneracy is
  eliminated upstream, so there is no convergence axis.
- BETA ELIMINATION (exact): the engine reduces the source into the shear
  eigenframe via ``z_eig = exp(-i*beta) * (y1 + i*y2)``, i.e. a rotation
  ``R(-beta)``.  The envelope is invariant under this rotation (delays,
  the critical delay ``tau_c``, saddle kernels and the switch are all
  rotation-invariant scalars), so training is done once at ``beta = 0``
  over ``(log w, gamma, y1_eig, y2_eig)`` and every query rotates its
  source by ``-beta`` into the eigenframe before lookup.

The full u1/u2 reflection symmetry of the Fermat potential is deliberately
NOT exploited to shrink training in this MVP: the full box is trained
directly (reflection is validated by a test, not used as a mechanism).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

from cogwheel.lensing.chang_refsdal import ChangRefsdalChannels
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
# boundary conditions -- the exact interpolant `make_interp_spline` (and
# hence the former per-call `RegularGridInterpolator(method='cubic')`)
# produces, precomputed once here.
_SPLINE_DEGREE = 3


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


class LensAmplificationSurrogate:
    """Tensor-cubic-spline emulator of the SACR-C envelope ``E(w)``.

    The envelope is emulated over a fixed 4-D box
    ``(log w, gamma, y1_eig, y2_eig)`` at ``beta = 0``, ``kappa = 0``.
    The box is chosen (by the training call) to lie wholly inside one
    image-count region with caustic distance bounded away from zero, so a
    single interpolant serves the whole box -- no per-region partitioning.

    Construct via `from_engine` (offline dense-grid engine training) or
    `load` (deserialize a saved box).  Query via `envelope`; test domain
    membership via `in_domain`.

    Parameters
    ----------
    log_w_grid : np.ndarray
        1-D strictly increasing ``ln w`` training axis (w dimensionless).
    gamma_grid : np.ndarray
        1-D strictly increasing external-shear axis.
    y1_grid, y2_grid : np.ndarray
        1-D strictly increasing eigenframe source-position axes.
    envelope_real, envelope_imag : np.ndarray
        Shape ``(n_w, n_gamma, n_y1, n_y2)`` real and imaginary parts of
        the trained envelope ``E(w)``.  Refused grid points are stored as
        zeros and never served (they are gated out by `in_domain`).
    refused_points : np.ndarray
        Shape ``(n_refused, 3)`` eigenframe ``(gamma, y1_eig, y2_eig)``
        training points at which the engine raised a named refusal at any
        ``w`` node.  May be empty with shape ``(0, 3)``.
    provenance : dict
        Minimal training metadata (box bounds, grid resolution, a short
        training hash).  Stored verbatim and re-serialized on `save`.

    Raises
    ------
    ValueError
        If the grids are not strictly increasing 1-D axes or the value
        arrays do not match the grid shape.
    """

    log_w_grid: np.ndarray
    gamma_grid: np.ndarray
    y1_grid: np.ndarray
    y2_grid: np.ndarray
    envelope_real: np.ndarray
    envelope_imag: np.ndarray
    refused_points: np.ndarray
    provenance: dict

    def __init__(self, *, log_w_grid: np.ndarray, gamma_grid: np.ndarray,
                 y1_grid: np.ndarray, y2_grid: np.ndarray,
                 envelope_real: np.ndarray, envelope_imag: np.ndarray,
                 refused_points: np.ndarray, provenance: dict,
                 real_coeffs: np.ndarray | None = None,
                 imag_coeffs: np.ndarray | None = None,
                 knots: list | None = None) -> None:
        self.log_w_grid = self._validate_axis(log_w_grid, 'log_w_grid')
        self.gamma_grid = self._validate_axis(gamma_grid, 'gamma_grid')
        self.y1_grid = self._validate_axis(y1_grid, 'y1_grid')
        self.y2_grid = self._validate_axis(y2_grid, 'y2_grid')

        expected = (self.log_w_grid.size, self.gamma_grid.size,
                    self.y1_grid.size, self.y2_grid.size)
        self.envelope_real = np.ascontiguousarray(envelope_real, dtype=float)
        self.envelope_imag = np.ascontiguousarray(envelope_imag, dtype=float)
        for name, array in (('envelope_real', self.envelope_real),
                            ('envelope_imag', self.envelope_imag)):
            if array.shape != expected:
                raise ValueError(
                    f'{name} has shape {array.shape}; expected {expected} '
                    f'from the training grids.')

        refused = np.asarray(refused_points, dtype=float)
        if refused.size == 0:
            refused = np.empty((0, 3), dtype=float)
        if refused.ndim != 2 or refused.shape[1] != 3:
            raise ValueError(
                f'refused_points must have shape (n, 3); got {refused.shape}.')
        self.refused_points = refused
        self.provenance = dict(provenance)

        self._build_coefficients(real_coeffs, imag_coeffs, knots)

    @staticmethod
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

    def _build_coefficients(self, real_coeffs: np.ndarray | None,
                            imag_coeffs: np.ndarray | None,
                            knots: list | None) -> None:
        """Build (or adopt) the tensor cubic B-spline coefficients + knots.

        The real/imag value tensors are turned into cubic B-spline
        coefficient tensors -- plus one knot vector per axis -- by
        successive 1-D `make_interp_spline` fits (not-a-knot boundary
        conditions), giving the SAME tensor-product interpolant the former
        per-call `RegularGridInterpolator(method='cubic')` evaluated, but
        computed once.  ``real_coeffs`` / ``imag_coeffs`` / ``knots``, when
        supplied (deserialization), are adopted verbatim so `save`/`load`
        and pickle round-trip the interpolant bit-for-bit; otherwise
        (training) they are computed here.

        The coefficient axes are kept in ``(log_w, gamma, y1, y2)`` order:
        each 1-D fit returns its coefficient axis leading, so it is rotated
        to the back, and after four fits the layout returns to the original
        axis order.  The remaining cached state -- the mean parameter
        spacing for the exclusion ball and the ``w`` band -- is derived and
        rebuilt on unpickle.
        """
        if real_coeffs is None or imag_coeffs is None or knots is None:
            grids = (self.log_w_grid, self.gamma_grid, self.y1_grid,
                     self.y2_grid)
            real_c, imag_c = self.envelope_real, self.envelope_imag
            knot_list: list[np.ndarray] = []
            for axis_grid in grids:
                spl_r = make_interp_spline(axis_grid, real_c,
                                           k=_SPLINE_DEGREE, axis=0)
                spl_i = make_interp_spline(axis_grid, imag_c,
                                           k=_SPLINE_DEGREE, axis=0)
                # Rotate the just-fitted (leading) coefficient axis to the
                # back so the four axes cycle back to their original order.
                real_c = np.moveaxis(spl_r.c, 0, -1)
                imag_c = np.moveaxis(spl_i.c, 0, -1)
                knot_list.append(np.ascontiguousarray(spl_r.t, dtype=float))
            self._real_coeffs = np.ascontiguousarray(real_c, dtype=float)
            self._imag_coeffs = np.ascontiguousarray(imag_c, dtype=float)
            self._knots = knot_list
        else:
            self._real_coeffs = np.ascontiguousarray(real_coeffs, dtype=float)
            self._imag_coeffs = np.ascontiguousarray(imag_coeffs, dtype=float)
            self._knots = [np.ascontiguousarray(t, dtype=float)
                           for t in knots]
        if self._real_coeffs.shape != self.envelope_real.shape:
            raise ValueError(
                f'real_coeffs has shape {self._real_coeffs.shape}; expected '
                f'{self.envelope_real.shape} from the value tensors.')
        if self._imag_coeffs.shape != self.envelope_real.shape:
            raise ValueError(
                f'imag_coeffs has shape {self._imag_coeffs.shape}; expected '
                f'{self.envelope_real.shape} from the value tensors.')

        # Mean spacing per parameter axis, for exclusion-ball normalization.
        self._param_spacing = np.array([
            float(np.mean(np.diff(self.gamma_grid))),
            float(np.mean(np.diff(self.y1_grid))),
            float(np.mean(np.diff(self.y2_grid)))])
        self._w_min = float(np.exp(self.log_w_grid[0]))
        self._w_max = float(np.exp(self.log_w_grid[-1]))

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
        """Train an envelope surrogate on a dense engine grid (offline).

        Evaluates `ChangRefsdalChannels.evaluate` at ``beta = 0``,
        ``kappa = 0`` on the full dense ``w`` grid for every parameter
        grid point (no LOO / adaptive logic -- unlimited offline engine
        calls), taking ``partition.envelope`` as the label.  Each engine
        call is wrapped for the named refusals; a parameter point that
        refuses at any ``w`` node (or returns a non-finite envelope) is
        recorded as refused and left as zeros in the value arrays.

        Choose the box to lie wholly inside one image-count region with
        caustic distance bounded away from zero; the region/parity is then
        implicit in the box and a single interpolant serves it.

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
            ``None`` (default) uses the engine default.

        Returns
        -------
        LensAmplificationSurrogate
            The trained surrogate.
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
                    env = np.asarray(partition.envelope)
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
        provenance = cls._build_provenance(
            gamma_range, y1_range, y2_range, w_range, shape,
            envelope_real, envelope_imag)
        return cls(log_w_grid=log_w_grid, gamma_grid=gamma_grid,
                   y1_grid=y1_grid, y2_grid=y2_grid,
                   envelope_real=envelope_real, envelope_imag=envelope_imag,
                   refused_points=refused_points, provenance=provenance)

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
            'training_hash': hasher.hexdigest()[:12]}

    # ---- Query --------------------------------------------------------

    def in_domain(self, gamma: float, y1: float, y2: float,
                  beta: float) -> bool:
        """Whether ``(gamma, y1, y2, beta)`` is inside the served domain.

        A point is served iff, after rotating the source into the shear
        eigenframe, it is (a) axis-aligned contained inside the certified
        training box and (b) outside the exclusion ball of radius one
        grid spacing around every refused training point.  This is a fixed
        geometric gate, not a learned mask: a false negative merely defers
        to the exact engine, whereas a false positive would serve a value
        where the engine refuses (the F005 bug), so the gate is
        deliberately conservative.

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
            ``True`` if the surrogate may serve this parameter point.
        """
        y1_eig, y2_eig = _rotate_to_eigenframe(y1, y2, beta)

        if not (self.gamma_grid[0] <= gamma <= self.gamma_grid[-1]
                and self.y1_grid[0] <= y1_eig <= self.y1_grid[-1]
                and self.y2_grid[0] <= y2_eig <= self.y2_grid[-1]):
            return False

        if self.refused_points.shape[0] > 0:
            query = np.array([gamma, y1_eig, y2_eig])
            normalized = (self.refused_points - query) / self._param_spacing
            distances = np.sqrt(np.sum(normalized ** 2, axis=1))
            if np.min(distances) <= _EXCLUSION_RADIUS:
                return False

        return True

    def envelope(self, w_array: np.ndarray, gamma: float, y1: float,
                 y2: float, beta: float) -> tuple[np.ndarray, bool]:
        """Emulated envelope ``E(w)`` and whether the surrogate served it.

        Rotates ``(y1, y2)`` into the shear eigenframe, checks the domain
        gate, and -- if served -- evaluates the real and imaginary cubic
        splines separately at ``(ln w, gamma, y1_eig, y2_eig)``.  Returns
        ``served=False`` (with a zero array) whenever the parameter point
        is out of domain or any requested ``w`` falls outside the trained
        band, signalling the caller to fall back to the exact engine.

        Parameters
        ----------
        w_array : np.ndarray
            Dimensionless frequencies at which to emulate ``E(w)``.  May
            be a scalar or any shape; the output matches its shape.
        gamma : float
            External shear magnitude.
        y1, y2 : float
            Source position in the shear frame at orientation ``beta``.
        beta : float
            External shear orientation, radians.

        Returns
        -------
        E_array : np.ndarray
            Complex emulated envelope, shaped like ``w_array`` (zeros when
            not served).
        served : bool
            ``True`` if the surrogate emulated the envelope; ``False`` if
            the caller must fall back to the exact engine.
        """
        w = np.asarray(w_array, dtype=float)

        if not self.in_domain(gamma, y1, y2, beta):
            return np.zeros(w.shape, dtype=complex), False

        w_flat = np.atleast_1d(w).ravel()
        if w_flat.size == 0 or not np.all(
                (w_flat >= self._w_min) & (w_flat <= self._w_max)):
            return np.zeros(w.shape, dtype=complex), False

        y1_eig, y2_eig = _rotate_to_eigenframe(y1, y2, beta)
        env_flat = (self._contract(self._real_coeffs, gamma, y1_eig, y2_eig,
                                   w_flat)
                    + 1j * self._contract(self._imag_coeffs, gamma, y1_eig,
                                          y2_eig, w_flat))
        return env_flat.reshape(w.shape), True

    def _contract(self, coeffs: np.ndarray, gamma: float, y1_eig: float,
                  y2_eig: float, w_flat: np.ndarray) -> np.ndarray:
        """Evaluate the tensor cubic B-spline at fixed parameters over ``w``.

        The three parameter axes are fixed for one query, so the 4-D
        coefficient tensor is contracted at ``(gamma, y1_eig, y2_eig)`` --
        collapsing the gamma, then y1, then y2 axes -- down to a single
        1-D B-spline in ``ln w``, which is evaluated at every ``w`` node.
        """
        t_w, t_g, t_y1, t_y2 = self._knots
        cc = BSpline(t_g, coeffs, _SPLINE_DEGREE, axis=1)(gamma)
        cc = BSpline(t_y1, cc, _SPLINE_DEGREE, axis=1)(y1_eig)
        cc = BSpline(t_y2, cc, _SPLINE_DEGREE, axis=1)(y2_eig)
        return BSpline(t_w, cc, _SPLINE_DEGREE, axis=0)(np.log(w_flat))

    # ---- Serialization ------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the surrogate to a single ``.npz`` file (flat ndarrays).

        Stores the training grids, the raw real/imag value tensors (kept
        for provenance/retraining and the training hash), the tensor cubic
        B-spline coefficient tensors and per-axis knot vectors that a
        served query evaluates (round-tripped bit-for-bit so `envelope` is
        byte-identical after a reload), the box bounds (implicit in the
        grids), the refused-point array, and the provenance dict
        (JSON-encoded).  The file loads without pickle.

        Parameters
        ----------
        path : str or Path
            Destination path; ``.npz`` is appended by numpy if absent.
        """
        np.savez(
            path,
            log_w_grid=self.log_w_grid,
            gamma_grid=self.gamma_grid,
            y1_grid=self.y1_grid,
            y2_grid=self.y2_grid,
            envelope_real=self.envelope_real,
            envelope_imag=self.envelope_imag,
            real_coeffs=self._real_coeffs,
            imag_coeffs=self._imag_coeffs,
            knot_log_w=self._knots[0],
            knot_gamma=self._knots[1],
            knot_y1=self._knots[2],
            knot_y2=self._knots[3],
            refused_points=self.refused_points,
            provenance=np.array(json.dumps(self.provenance)))

    @classmethod
    def load(cls, path: str | Path) -> 'LensAmplificationSurrogate':
        """Load a surrogate saved by `save`.

        Parameters
        ----------
        path : str or Path
            Path to the ``.npz`` file.

        Returns
        -------
        LensAmplificationSurrogate
            The reconstructed surrogate.
        """
        with np.load(path, allow_pickle=False) as data:
            provenance = json.loads(str(data['provenance']))
            return cls(
                log_w_grid=data['log_w_grid'],
                gamma_grid=data['gamma_grid'],
                y1_grid=data['y1_grid'],
                y2_grid=data['y2_grid'],
                envelope_real=data['envelope_real'],
                envelope_imag=data['envelope_imag'],
                real_coeffs=data['real_coeffs'],
                imag_coeffs=data['imag_coeffs'],
                knots=[data['knot_log_w'], data['knot_gamma'],
                       data['knot_y1'], data['knot_y2']],
                refused_points=data['refused_points'],
                provenance=provenance)

    # ---- Pickle support (flat ndarrays only) --------------------------

    def __getstate__(self) -> dict:
        """Return picklable state: flat ndarrays plus provenance.

        The raw value tensors AND the tensor B-spline coefficient tensors
        with their knot vectors ride along as flat ndarrays (round-tripped
        bit-for-bit, so a query is byte-identical after unpickle); the mean
        parameter spacing and ``w`` band are derived and rebuilt on
        `__setstate__`.
        """
        return {
            'log_w_grid': self.log_w_grid,
            'gamma_grid': self.gamma_grid,
            'y1_grid': self.y1_grid,
            'y2_grid': self.y2_grid,
            'envelope_real': self.envelope_real,
            'envelope_imag': self.envelope_imag,
            'real_coeffs': self._real_coeffs,
            'imag_coeffs': self._imag_coeffs,
            'knots': self._knots,
            'refused_points': self.refused_points,
            'provenance': self.provenance}

    def __setstate__(self, state: dict) -> None:
        """Restore flat state and adopt the saved coefficients verbatim."""
        self.log_w_grid = state['log_w_grid']
        self.gamma_grid = state['gamma_grid']
        self.y1_grid = state['y1_grid']
        self.y2_grid = state['y2_grid']
        self.envelope_real = state['envelope_real']
        self.envelope_imag = state['envelope_imag']
        self.refused_points = state['refused_points']
        self.provenance = state['provenance']
        self._build_coefficients(state['real_coeffs'], state['imag_coeffs'],
                                 state['knots'])
