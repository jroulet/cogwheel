"""
Trained Born-exterior residual interpolation artifact.

WHAT
----
`BornResidualChart` is the frozen dataclass holding a trained 3-D
tensor-product interpolation of the Born (weak-deflection) residual

    R(w; gamma, rho) = F_exact_demod(w) - F_carrier_demod(w)

over the (gamma, rho, log w) grid.  It is produced by the training driver
(``scripts/train_born_residual.py``, not yet implemented) and consumed by
`LensedRelativeBinningLikelihood._surrogate_coefficients` (the fact-4 slot).

Frame convention: stored values represent R(w) in the MIN-RELATIVE DELAY
frame — the same frame as ``ChangRefsdalPartition.exact_total`` and
``born_carrier_from_partition``.  The driver subtracts the carrier from the
exact total directly (both are in the min-relative frame), so no additional
frame rotation is needed at serve time.

WHY
---
The exact Chang--Refsdal engine is certifiable over the Born exterior
(``w * |y| <= 60``) but expensive.  The carrier alone captures the leading
behaviour; this chart interpolates the smooth, bounded RESIDUAL so the sum
``carrier + residual`` reproduces the exact amplification to chart accuracy
without running the engine, giving a serve-time speed-up analogous to the
far-field surrogate charts.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

#: Shipped package-data artifact name (under ``cogwheel/data/``).
_DEFAULT_CHART_NAME = 'born_residual_chart.npz'

#: Artifact schema tag.  The loader hard-refuses (``ValueError``) an
#: artifact whose ``schema`` key is absent or does not match this value,
#: so a stale/foreign npz can never be silently deserialized.
_SCHEMA = 'born_residual_v1'


def _content_hash(gamma_grid: np.ndarray, rho_grid: np.ndarray,
                  log_w_grid: np.ndarray, real_coeffs: np.ndarray,
                  imag_coeffs: np.ndarray) -> str:
    """SHA1 over the stored grids and coefficient arrays (float64).

    This duplicates the ~5-line primitive in
    ``cogwheel.lensing.ppgo_map`` rather than importing it, deliberately:
    importing would introduce an intra-``lensing`` module edge for a
    trivial helper, and the ppGO variant folds certification scalars this
    chart does not have into its signature.  Duplication is the smaller
    cost (DRY-vs-coupling tradeoff).
    """
    hasher = hashlib.sha1()
    for array in (gamma_grid, rho_grid, log_w_grid, real_coeffs, imag_coeffs):
        hasher.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return hasher.hexdigest()


@dataclass(frozen=True)
class BornResidualChart:
    """Frozen 3-D interpolation chart for the Born residual.

    Attributes
    ----------
    gamma_grid : ndarray
        1-D ascending grid of shear parameter gamma values.
    rho_grid : ndarray
        1-D ascending grid of caustic-distance rho values (all > 1.0,
        exterior to the caustic).
    log_w_grid : ndarray
        1-D ascending grid of log(w) values (natural logarithm of
        dimensionless frequency).
    real_coeffs : ndarray
        3-D array shape ``(n_gamma, n_rho, n_w)`` holding the real part
        of the residual at the grid nodes.
    imag_coeffs : ndarray
        3-D array shape ``(n_gamma, n_rho, n_w)`` holding the imaginary
        part of the residual at the grid nodes.
    provenance : dict
        Optional metadata dictionary (training date, driver version,
        accuracy stats, etc.).  Not used by the serve path; preserved
        for audit/reproducibility.
    """

    gamma_grid: np.ndarray
    rho_grid: np.ndarray
    log_w_grid: np.ndarray
    real_coeffs: np.ndarray
    imag_coeffs: np.ndarray
    provenance: dict = field(default_factory=dict)

    def covers(self, gamma: float, rho: float,
               w: np.ndarray | None = None) -> bool:
        """Axis-aligned box containment check.

        Parameters
        ----------
        gamma : float
            Shear parameter of the candidate.
        rho : float
            Caustic distance of the candidate.
        w : ndarray, optional
            Served dimensionless-frequency band.  When provided (and
            non-empty), containment additionally requires the whole band to
            lie within the trained ``log_w_grid`` range, so that
            :meth:`evaluate` never cubic-extrapolates off the frequency
            axis.  When ``None`` (default) only the (gamma, rho) box is
            checked, preserving the original two-argument contract.

        Returns
        -------
        bool
            True if (gamma, rho) lies within the grid's bounding box
            (inclusive on both ends) and, when ``w`` is given, the served
            band lies within the trained log-w range.
        """
        in_box = (self.gamma_grid[0] <= gamma <= self.gamma_grid[-1]
                  and self.rho_grid[0] <= rho <= self.rho_grid[-1])
        if not in_box:
            return False
        if w is None:
            return True
        w = np.asarray(w, dtype=float)
        if w.size == 0:
            return True
        log_w = np.log(w)
        return (float(log_w.min()) >= self.log_w_grid[0]
                and float(log_w.max()) <= self.log_w_grid[-1])

    def evaluate(self, w: np.ndarray, gamma: float, rho: float) -> np.ndarray:
        """Interpolate the residual R(w; gamma, rho).

        Uses 3-D tensor-product cubic interpolation over
        (gamma, rho, log w).  The interpolator is lazily constructed and
        cached on first call (stored via ``object.__setattr__`` on the
        frozen dataclass).

        Parameters
        ----------
        w : ndarray
            1-D array of dimensionless frequencies (positive).
        gamma : float
            Shear parameter of the candidate.
        rho : float
            Caustic distance of the candidate.

        Returns
        -------
        ndarray
            Complex residual array, shape matching ``w``.  Values are in
            the min-relative delay frame BY CONTRACT.
        """
        w = np.asarray(w, dtype=float)
        log_w = np.log(w)

        # Lazy-build and cache the interpolators on first call.
        # Use object.__setattr__ to bypass the frozen guard.
        if not hasattr(self, '_real_interp'):
            real_interp = RegularGridInterpolator(
                (self.gamma_grid, self.rho_grid, self.log_w_grid),
                self.real_coeffs, method='cubic',
                bounds_error=False, fill_value=None)
            imag_interp = RegularGridInterpolator(
                (self.gamma_grid, self.rho_grid, self.log_w_grid),
                self.imag_coeffs, method='cubic',
                bounds_error=False, fill_value=None)
            object.__setattr__(self, '_real_interp', real_interp)
            object.__setattr__(self, '_imag_interp', imag_interp)

        # Build the query points: shape (n_w, 3) with columns
        # (gamma, rho, log_w).
        points = np.empty((log_w.size, 3), dtype=float)
        points[:, 0] = gamma
        points[:, 1] = rho
        points[:, 2] = log_w

        real_part = self._real_interp(points)
        imag_part = self._imag_interp(points)
        return real_part + 1j * imag_part

    @classmethod
    def load(cls, path: str | Path | None = None) -> 'BornResidualChart':
        """Load and hash-verify a Born-residual chart artifact.

        Parameters
        ----------
        path : str or Path, optional
            Explicit artifact path; ``None`` resolves the shipped
            package-data default under ``cogwheel/data/``.

        Returns
        -------
        BornResidualChart
            The reconstructed, schema- and hash-verified chart.

        Raises
        ------
        ValueError
            If the ``schema`` key is absent or does not match
            ``born_residual_v1``, or if the recomputed content hash does
            not match the stored one (corrupt / stale artifact).  The
            message names ``scripts/train_born_residual.py`` as the
            regeneration script.
        """
        if path is None:
            path = cls._default_artifact_path()
        with np.load(path, allow_pickle=False) as data:
            if 'schema' not in data.files:
                raise ValueError(
                    'Born-residual chart artifact is missing the `schema` '
                    'key; it is pre-schema, stale, or corrupt. Regenerate '
                    'with scripts/train_born_residual.py.')
            schema = str(data['schema'])
            if schema != _SCHEMA:
                raise ValueError(
                    f'Born-residual chart schema mismatch: stored '
                    f'{schema!r}, expected {_SCHEMA!r}. The artifact is '
                    f'stale or foreign; regenerate with '
                    f'scripts/train_born_residual.py.')
            if 'content_hash' not in data.files:
                raise ValueError(
                    'Born-residual chart artifact is missing the '
                    '`content_hash` key; it is stale or corrupt. Regenerate '
                    'with scripts/train_born_residual.py.')
            expected = str(data['content_hash'])
            gamma_grid = np.asarray(data['gamma_grid'], dtype=np.float64)
            rho_grid = np.asarray(data['rho_grid'], dtype=np.float64)
            log_w_grid = np.asarray(data['log_w_grid'], dtype=np.float64)
            real_coeffs = np.asarray(data['real_coeffs'], dtype=np.float64)
            imag_coeffs = np.asarray(data['imag_coeffs'], dtype=np.float64)
            provenance = json.loads(str(data['provenance']))

        actual = _content_hash(gamma_grid, rho_grid, log_w_grid,
                               real_coeffs, imag_coeffs)
        if expected != actual:
            raise ValueError(
                f'Born-residual chart content-hash mismatch: stored '
                f'{expected!r}, recomputed {actual!r}. The artifact is '
                f'corrupt or stale; regenerate with '
                f'scripts/train_born_residual.py.')
        return cls(gamma_grid=gamma_grid, rho_grid=rho_grid,
                   log_w_grid=log_w_grid, real_coeffs=real_coeffs,
                   imag_coeffs=imag_coeffs, provenance=provenance)

    @staticmethod
    def _default_artifact_path() -> Path:
        """Resolve the shipped package-data artifact under cogwheel/data."""
        return Path(str(files('cogwheel').joinpath('data',
                                                    _DEFAULT_CHART_NAME)))
