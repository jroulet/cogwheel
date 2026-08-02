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
The exact Chang--Refsdal engine is certifiable over the Born annulus
(``w * |y| <= 60``) but expensive.  The carrier alone captures the leading
behaviour; this chart interpolates the smooth, bounded RESIDUAL so the sum
``carrier + residual`` reproduces the exact amplification to chart accuracy
without running the engine, giving a serve-time speed-up analogous to the
far-field surrogate charts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True)
class BornResidualChart:
    """Frozen 3-D interpolation chart for the Born-annulus residual.

    Fields
    ------
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

    def covers(self, gamma: float, rho: float) -> bool:
        """Axis-aligned box containment check.

        Parameters
        ----------
        gamma : float
            Shear parameter of the candidate.
        rho : float
            Caustic distance of the candidate.

        Returns
        -------
        bool
            True if (gamma, rho) lies within the grid's bounding box
            (inclusive on both ends).
        """
        return (self.gamma_grid[0] <= gamma <= self.gamma_grid[-1]
                and self.rho_grid[0] <= rho <= self.rho_grid[-1])

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
