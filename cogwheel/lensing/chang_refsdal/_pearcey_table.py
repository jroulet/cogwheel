"""
Precomputed spline table for the universal Pearcey primitive ``P(x, y)``.

WHAT
----
`PearceyTable` is a load-time-reconstructed bicubic-spline emulator of the
Fresnel-*demodulated* Pearcey primitive

    P(x, y) = Int_{-inf}^{inf} exp[i (t^4 + x t^2 + y t)] dt

over a bounded, build-time-derived ``(x, y)`` box.  ``P`` is
lens-independent, so a single table amortizes the ~45 ms certified
quadrature (`_pearcey_cusp.pearcey`) down to a spline evaluation
(<= 50 us) inside the box.  The cusp arm consults this table first and
falls back to the live certified quadrature outside the box or on any
load / hash anomaly (the never-serve-where-wrong backstop, see
`_pearcey_cusp`).

WHY DEMODULATE BEFORE SPLINING
------------------------------
``P`` oscillates with the fast cusp Fresnel carrier ``exp(i phi_sp)``,
``phi_sp = t*^4 + x t*^2 + y t*`` at the dominant real stationary point
``t*`` (a root of ``4 t*^3 + 2 x t* + y = 0``).  A raw oscillatory ``P``
cannot reach 1e-8 at feasible spline resolution; dividing the carrier out
leaves a slowly varying demodulated amplitude whose real and imaginary
parts are splined separately on a graded grid (denser near the caustic
``27 y^2 = -8 x^3`` where the residual curvature peaks).

The stationary point used for the carrier is selected by the
*continuous* rule: among the real roots of the cubic, take the one with
the largest phase curvature ``|phi''| = |12 t*^2 + 2 x|``.  That root is
the isolated (non-merging) branch; it stays real and smooth across the
fold caustic ``27 y^2 = -8 x^3`` (where the *other* two roots coalesce
with vanishing curvature), so the carrier -- and hence the demodulated
amplitude -- is continuous across the caustic, which is the property the
table build requires.  (Across the interior Maxwell seam ``y = 0`` the
selected root switches between two branches of equal phase, so the
carrier stays continuous in value there too; the graded grid absorbs the
mild derivative kink.)

SERIALIZATION AND INTEGRITY
---------------------------
The artifact is a single ``.npz`` of plain float64 arrays (the graded
grid axes and the demodulated Re / Im grids) plus a JSON provenance
scalar carrying the derived box edges, the overlap margin, the oracle
tolerance and a SHA1 content hash.  `PearceyTable.load` reads it with
``allow_pickle=False``, recomputes the content hash and refuses (raises
``ValueError``) on mismatch; the splines are re-fitted deterministically
at load.  No new exception class is introduced -- integrity failures are
plain ``ValueError`` / ``OSError`` that the arm turns into a
live-quadrature fall-through.
"""
from __future__ import annotations

import cmath
import hashlib
import json
import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

__all__ = ['PearceyTable', 'derive_box', 'build_table', 'held_out_error',
           'save_table']

#: Shipped package-data artifact name (under ``cogwheel/data/``).
_DEFAULT_TABLE_NAME = 'pearcey_table.npz'

#: Imaginary-part tolerance (relative to ``1 + |Re|``) for accepting a
#: numpy cubic root as a real stationary point.
_IMAG_TOL = 1e-9

#: Bicubic spline degree in each axis (not-a-knot bicubic).
_SPLINE_DEGREE = 3

#: Provenance schema version for the artifact.
_SCHEMA_VERSION = '0.1.0'


# ----------------------------------------------------------------------
# Fresnel carrier (demodulation phase).
# ----------------------------------------------------------------------

def _dominant_stationary_point(x: float, y: float) -> float:
    """Continuous dominant real stationary point ``t*`` of the cusp phase.

    ``t*`` is a real root of ``phi'(t) = 4 t^3 + 2 x t + y = 0`` selected
    by maximal phase curvature ``|12 t^2 + 2 x|``; that isolated branch
    stays real across the fold caustic ``27 y^2 = -8 x^3`` (see the module
    docstring), giving a carrier continuous across the caustic.
    """
    roots = np.roots([4.0, 0.0, 2.0 * x, y])
    real: list[float] = []
    for root in roots:
        value = complex(root)
        if abs(value.imag) < _IMAG_TOL * (1.0 + abs(value.real)):
            real.append(value.real)
    if not real:
        # No real stationary point (does not occur for finite real
        # controls); use the least-imaginary root so the carrier is
        # still defined rather than raising.
        least_imag = min(roots, key=lambda r: abs(complex(r).imag))
        return float(complex(least_imag).real)
    real.sort()
    # ``max`` returns the first argmax; with ``real`` sorted ascending a
    # curvature tie resolves deterministically to the smaller root.
    return max(real, key=lambda t: abs(12.0 * t * t + 2.0 * x))


def _carrier_phase(x: float, y: float) -> float:
    """Fresnel demodulation phase ``phi_sp = t*^4 + x t*^2 + y t*``."""
    t = _dominant_stationary_point(x, y)
    return t ** 4 + x * t ** 2 + y * t


def demodulate(value: complex, x: float, y: float) -> complex:
    """Remove the cusp Fresnel carrier: ``P -> P exp(-i phi_sp)``."""
    return value * cmath.exp(-1j * _carrier_phase(x, y))


def remodulate(demod_value: complex, x: float, y: float) -> complex:
    """Restore the cusp Fresnel carrier.

    ``P_demod -> P_demod exp(i phi_sp)``.
    """
    return demod_value * cmath.exp(1j * _carrier_phase(x, y))


# ----------------------------------------------------------------------
# Content hash (integrity).
# ----------------------------------------------------------------------

def _content_hash(x_grid: np.ndarray, y_grid: np.ndarray,
                  demod_real: np.ndarray, demod_imag: np.ndarray,
                  x_max: float, y_max: float, margin: float,
                  oracle_tol: float) -> str:
    """SHA1 over the stored arrays and box scalars (exact float64 bytes)."""
    hasher = hashlib.sha1()
    for array in (x_grid, y_grid, demod_real, demod_imag):
        hasher.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    box = np.asarray([x_max, y_max, margin, oracle_tol], dtype=np.float64)
    hasher.update(box.tobytes())
    return hasher.hexdigest()


# ----------------------------------------------------------------------
# The table.
# ----------------------------------------------------------------------

@dataclass
class PearceyTable:
    """Bicubic-spline table of the demodulated Pearcey primitive.

    Instances are built by `from_grid` (which fits the load-time splines);
    construct one from a shipped artifact with `PearceyTable.load`.

    Attributes
    ----------
    x_grid, y_grid : ndarray
        Strictly increasing graded knot axes spanning
        ``[-x_max, x_max]`` and ``[-y_max, y_max]``.
    demod_real, demod_imag : ndarray, shape (nx, ny)
        Real / imaginary parts of the demodulated primitive on the grid.
    provenance : dict
        Box edges (``x_max``, ``y_max``), ``margin``, ``oracle_tol``,
        ``content_hash`` and build metadata.
    spline_real, spline_imag : RectBivariateSpline
        Load-time-fitted bicubic interpolants (not serialized).
    """

    x_grid: np.ndarray
    y_grid: np.ndarray
    demod_real: np.ndarray
    demod_imag: np.ndarray
    provenance: dict
    spline_real: RectBivariateSpline
    spline_imag: RectBivariateSpline

    # -- construction -------------------------------------------------

    @classmethod
    def from_grid(cls, x_grid: np.ndarray, y_grid: np.ndarray,
                  demod_real: np.ndarray, demod_imag: np.ndarray,
                  provenance: dict) -> 'PearceyTable':
        """Fit the bicubic splines and assemble a `PearceyTable`."""
        x_grid = np.ascontiguousarray(x_grid, dtype=np.float64)
        y_grid = np.ascontiguousarray(y_grid, dtype=np.float64)
        demod_real = np.ascontiguousarray(demod_real, dtype=np.float64)
        demod_imag = np.ascontiguousarray(demod_imag, dtype=np.float64)
        if x_grid.ndim != 1 or y_grid.ndim != 1:
            raise ValueError('Pearcey table grid axes must be 1-D.')
        if demod_real.shape != (x_grid.size, y_grid.size):
            raise ValueError(
                f'demod_real shape {demod_real.shape} does not match grid '
                f'({x_grid.size}, {y_grid.size}).')
        if demod_imag.shape != (x_grid.size, y_grid.size):
            raise ValueError(
                f'demod_imag shape {demod_imag.shape} does not match grid '
                f'({x_grid.size}, {y_grid.size}).')
        spline_real = RectBivariateSpline(
            x_grid, y_grid, demod_real,
            kx=_SPLINE_DEGREE, ky=_SPLINE_DEGREE, s=0.0)
        spline_imag = RectBivariateSpline(
            x_grid, y_grid, demod_imag,
            kx=_SPLINE_DEGREE, ky=_SPLINE_DEGREE, s=0.0)
        return cls(x_grid, y_grid, demod_real, demod_imag, dict(provenance),
                   spline_real, spline_imag)

    # -- load ---------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> 'PearceyTable':
        """Load and hash-verify a table artifact.

        Parameters
        ----------
        path : str or Path, optional
            Explicit artifact path; ``None`` resolves the shipped
            package-data default under ``cogwheel/data/``.

        Returns
        -------
        PearceyTable
            The reconstructed, hash-verified table.

        Raises
        ------
        ValueError
            If the recomputed content hash does not match the stored one
            (corrupt / stale artifact).  The cusp arm treats this -- and
            any ``OSError`` from a missing file -- as a fall-through to
            live certified quadrature.
        """
        if path is None:
            path = cls._default_artifact_path()
        with np.load(path, allow_pickle=False) as data:
            x_grid = np.asarray(data['x_grid'], dtype=np.float64)
            y_grid = np.asarray(data['y_grid'], dtype=np.float64)
            demod_real = np.asarray(data['demod_real'], dtype=np.float64)
            demod_imag = np.asarray(data['demod_imag'], dtype=np.float64)
            provenance = json.loads(str(data['provenance']))

        expected = provenance.get('content_hash')
        actual = _content_hash(x_grid, y_grid, demod_real, demod_imag,
                               float(provenance['x_max']),
                               float(provenance['y_max']),
                               float(provenance['margin']),
                               float(provenance['oracle_tol']))
        if expected != actual:
            raise ValueError(
                f'Pearcey table content-hash mismatch: stored {expected!r}, '
                f'recomputed {actual!r}. The artifact is corrupt or stale; '
                f'regenerate with scripts/train_pearcey_table.py.')
        return cls.from_grid(x_grid, y_grid, demod_real, demod_imag,
                             provenance)

    @staticmethod
    def _default_artifact_path() -> Path:
        """Resolve the shipped package-data artifact under cogwheel/data."""
        return Path(str(files('cogwheel').joinpath('data',
                                                    _DEFAULT_TABLE_NAME)))

    # -- box edges ----------------------------------------------------

    @property
    def x_max(self) -> float:
        """Half-width of the box in the along-cusp control ``x``."""
        return float(self.provenance['x_max'])

    @property
    def y_max(self) -> float:
        """Half-width of the box in the transverse control ``y``."""
        return float(self.provenance['y_max'])

    def contains(self, x: float, y: float) -> bool:
        """Whether ``(x, y)`` lies inside the served box."""
        return abs(x) <= self.x_max and abs(y) <= self.y_max

    # -- evaluation ---------------------------------------------------

    def evaluate(self, x: float, y: float) -> complex | None:
        """Table value of ``P(x, y)`` inside the box, else ``None``.

        Returns ``None`` (decline -- caller falls back to live quadrature)
        when the controls are non-finite, lie outside the box, or the
        remodulated value is not finite.  Never raises, never serves a
        value outside its certified box.
        """
        x = float(x)
        y = float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        if not self.contains(x, y):
            return None
        real = float(self.spline_real(x, y)[0, 0])
        imag = float(self.spline_imag(x, y)[0, 0])
        value = remodulate(complex(real, imag), x, y)
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            return None
        return value


# ----------------------------------------------------------------------
# Build helpers (offline; used by scripts/train_pearcey_table.py).
# ----------------------------------------------------------------------

def _graded_axis(half_width: float, n_points: int, power: float) -> np.ndarray:
    """Symmetric graded axis on ``[-half_width, half_width]``.

    ``power > 1`` clusters knots near ``0`` (the caustic / cusp), where
    the demodulated residual curvature peaks.
    """
    if n_points < _SPLINE_DEGREE + 1:
        raise ValueError(f'need >= {_SPLINE_DEGREE + 1} knots per axis, '
                         f'got {n_points}.')
    unit = np.linspace(-1.0, 1.0, n_points)
    axis = np.sign(unit) * np.abs(unit) ** power * half_width
    axis[0] = -half_width
    axis[-1] = half_width
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError('graded axis is not strictly increasing; reduce '
                         'the grading power or add knots.')
    return axis


def derive_box(*, oracle_tol: float = 1e-8, margin: float = 0.15,
               n_rays: int = 180, r_start: float = 0.5, r_stop: float = 24.0,
               n_radial: int = 240) -> dict:
    """Derive the box edges where the asymptotics take over from ``P``.

    March outward on a fan of rays through the origin (dense enough to
    include near-caustic directions); on each ray record the first radius
    beyond which ``|pearcey_asymptotic - pearcey| < oracle_tol`` and
    stays below out to ``r_stop``.  ``X_MAX, Y_MAX`` are the axis-aligned
    half-widths enclosing that handoff contour, inflated by ``margin`` so
    the spline-inside and asymptotic-outside regions overlap with no serve
    gap.  The served semicubical-caustic segment ``27 y^2 = -8 x^3`` over
    the full ``x`` range of the box is then forced strictly inside.

    Returns
    -------
    dict
        ``x_max``, ``y_max``, ``margin``, ``oracle_tol`` and the raw
        (pre-inflation) handoff half-widths for provenance.
    """
    from cogwheel.lensing.chang_refsdal._pearcey_cusp import (
        pearcey, pearcey_asymptotic)

    phis = np.linspace(0.0, 2.0 * math.pi, n_rays, endpoint=False)
    radii = np.linspace(r_start, r_stop, n_radial)
    x_handoff = 0.0
    y_handoff = 0.0
    for phi in phis:
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        handoff_r = r_stop
        # Walk inward from the outer edge: the handoff radius is the
        # smallest radius from which the error stays below tol outward.
        for radius in radii[::-1]:
            x = radius * cos_phi
            y = radius * sin_phi
            exact = pearcey(x, y)
            if exact is None:
                break
            error = abs(pearcey_asymptotic(x, y) - exact)
            if error < oracle_tol:
                handoff_r = radius
            else:
                break
        x_handoff = max(x_handoff, abs(handoff_r * cos_phi))
        y_handoff = max(y_handoff, abs(handoff_r * sin_phi))

    x_max = x_handoff * (1.0 + margin)
    y_max = y_handoff * (1.0 + margin)

    # Force the served semicubical-caustic segment strictly inside: at the
    # left box edge the caustic reaches |y| = sqrt(8/27) x_max^{3/2}.
    y_caustic = math.sqrt(8.0 / 27.0) * x_max ** 1.5
    y_max = max(y_max, y_caustic * (1.0 + margin))

    return {'x_max': float(x_max), 'y_max': float(y_max),
            'margin': float(margin), 'oracle_tol': float(oracle_tol),
            'x_handoff': float(x_handoff), 'y_handoff': float(y_handoff)}


def build_table(box: dict, *, n_x: int = 161, n_y: int = 161,
                grading_power: float = 1.6) -> PearceyTable:
    """Build a `PearceyTable` by sampling the certified quadrature.

    Parameters
    ----------
    box : dict
        Output of `derive_box` (must carry ``x_max``, ``y_max``,
        ``margin``, ``oracle_tol``).
    n_x, n_y : int
        Grid sizes along each axis.
    grading_power : float
        Knot-grading exponent (``> 1`` clusters knots near the caustic).

    Returns
    -------
    PearceyTable
        The fitted table with a provenance dict and content hash.

    Raises
    ------
    ValueError
        If the certified quadrature cannot certify a grid node (the table
        must have no holes).
    """
    from cogwheel.lensing.chang_refsdal._pearcey_cusp import pearcey

    x_max = float(box['x_max'])
    y_max = float(box['y_max'])
    x_grid = _graded_axis(x_max, n_x, grading_power)
    y_grid = _graded_axis(y_max, n_y, grading_power)

    demod_real = np.empty((n_x, n_y), dtype=np.float64)
    demod_imag = np.empty((n_x, n_y), dtype=np.float64)
    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            exact = pearcey(float(x), float(y))
            if exact is None:
                raise ValueError(
                    f'certified quadrature declined at grid node '
                    f'(x={x:.6g}, y={y:.6g}); cannot build a table hole.')
            demod = demodulate(exact, float(x), float(y))
            demod_real[i, j] = demod.real
            demod_imag[i, j] = demod.imag

    content_hash = _content_hash(x_grid, y_grid, demod_real, demod_imag,
                                 x_max, y_max, float(box['margin']),
                                 float(box['oracle_tol']))
    provenance = {
        'schema_version': _SCHEMA_VERSION,
        'x_max': x_max,
        'y_max': y_max,
        'margin': float(box['margin']),
        'oracle_tol': float(box['oracle_tol']),
        'x_handoff': float(box.get('x_handoff', float('nan'))),
        'y_handoff': float(box.get('y_handoff', float('nan'))),
        'n_x': int(n_x),
        'n_y': int(n_y),
        'grading_power': float(grading_power),
        'content_hash': content_hash,
    }
    return PearceyTable.from_grid(x_grid, y_grid, demod_real, demod_imag,
                                 provenance)


def held_out_error(table: PearceyTable, *, n_samples: int = 4000,
                   seed: int = 0) -> float:
    """Max absolute table-vs-quadrature error on random in-box points.

    Draws ``n_samples`` uniform points strictly inside the box (avoiding
    the grid nodes) and returns the worst absolute error against the
    certified quadrature; ``inf`` if any drawn point cannot be certified.
    The regeneration script asserts this is below ``oracle_tol`` so an
    uncertified table is never shipped.
    """
    from cogwheel.lensing.chang_refsdal._pearcey_cusp import pearcey

    rng = np.random.default_rng(seed)
    xs = rng.uniform(-table.x_max, table.x_max, n_samples)
    ys = rng.uniform(-table.y_max, table.y_max, n_samples)
    worst = 0.0
    for x, y in zip(xs, ys):
        exact = pearcey(float(x), float(y))
        if exact is None:
            return math.inf
        served = table.evaluate(float(x), float(y))
        if served is None:
            return math.inf
        worst = max(worst, abs(served - exact))
    return worst


def save_table(table: PearceyTable, path: str | Path) -> None:
    """Write a table to ``path`` as an ``allow_pickle=False`` ``.npz``.

    The provenance dict (including the content hash) is stored as a JSON
    scalar so the artifact carries only plain arrays and a unicode scalar.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        x_grid=table.x_grid,
        y_grid=table.y_grid,
        demod_real=table.demod_real,
        demod_imag=table.demod_imag,
        provenance=np.asarray(json.dumps(table.provenance)),
    )
