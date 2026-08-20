"""
Trained low-w diffractive residual interpolation artifact.

WHAT
----
``LowWDiffractiveChart`` is the frozen dataclass holding a trained 4-D
tensor-product interpolation of the low-w diffractive residual

    r_pure(w; gamma', rho, theta) = f_pure / (sqrt(mu_pure) * prefactor_c(w))

over the reduced-coordinate grid ``(gamma', rho, theta, log w)``.  It is the
loader + serve-side interpolation object ONLY (no engine, no training); it is
produced by ``scripts/train_low_w_diffractive_chart.py`` and consumed by the
diffractive low-w serve in the likelihood (Rung P).

Representation: the stored residual strips BOTH known analytic factors from
the exact pure-shear engine value ``f_pure = f_schwinger(w, y_eig, gamma')``:
the macro amplitude ``sqrt(mu_pure) = 1 / sqrt(1 - gamma'^2)`` that diverges
at the parity wall, and ``prefactor_c(w) = C(w)``, the exact point-mass
``w*ln(w)`` diffraction phase.  Stripping both leaves a smooth, bounded
residual (measured ``|r_pure| ~ 0.6-1.0`` across the band), NOT the raw
amplification.  The served value is reconstructed by re-modulation
``F_serve = mass_sheet_phase * prefactor_c(w) * sqrt_mu_full * r_pure``
(``mass_sheet_phase`` and ``sqrt_mu_full`` as defined in the likelihood
serve), which lives in the likelihood serve -- this class returns the
REDUCED-frame residual ``r_pure`` only.

Coordinates are REDUCED / caustic-relative, never raw lens-plane: ``rho =
|y'| / |y_c(theta)|`` (``geometry.caustic_point`` -- the same discriminator as
the fit fence), ``gamma'`` the reduced shear, ``theta`` the eigenframe angle
(folded to ``[0, pi/2]`` by D2 symmetry), and ``log w`` the natural logarithm
of the dimensionless frequency.

WHY
---
The near-fold shell (``RHO_LO <= rho <= RHO_HI``) and the wall-approach band
(``gamma' > _WALL_GAMMA_PRIME``) have no analytic serve at low w: the
order-16 shear-operator series has a convergence-radius collapse there, the
uniform Airy fold arm refuses at low w, and ``w_low_fit`` declines the shell.
This chart interpolates the residual so those draws can be served without
calling the exact Schwinger engine at serve time.  Schwinger is an OFFLINE
oracle only (used by the training script); this module never calls it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cogwheel.lensing.chang_refsdal._diffractive import (
    _DIFFRACTIVE_FIT_FENCE_DELTA,
    _DIFFRACTIVE_FIT_FENCE_RHO_LO,
)

#: Shipped package-data artifact name (under ``cogwheel/data/``).
_DEFAULT_CHART_NAME = 'low_w_diffractive_chart.npz'

#: Artifact schema tag.  The loader hard-refuses (``ValueError``) an
#: artifact whose ``schema`` key is absent or does not match this value,
#: so a stale/foreign npz can never be silently deserialized.
_SCHEMA = 'low_w_diffractive_v1'

#: Inner shell boundary ``RHO_LO`` -- single-sourced from the diffractive-fit
#: fence (the same discriminator ``w_low_fit`` uses to decline the near-fold
#: shell).  Imported, NOT re-typed, so the chart's shell and the fit's fence
#: can never drift apart.
RHO_LO = _DIFFRACTIVE_FIT_FENCE_RHO_LO

#: Outer shell boundary ``RHO_HI = 1.0 + DELTA`` (``DELTA = 0.4`` from the
#: same fence).
RHO_HI = 1.0 + _DIFFRACTIVE_FIT_FENCE_DELTA

#: Wall-approach ``gamma'`` ceiling.  Matches the calibrated gamma ceiling of
#: the fit script (``np.linspace(0.05, 0.5, 6)``) -- the wall band
#: (``gamma' > _WALL_GAMMA_PRIME``) is served by the chart, so a
#: re-calibration that moves that ceiling must move this constant too.
_WALL_GAMMA_PRIME = 0.5


def _content_hash(*arrays: np.ndarray) -> str:
    """SHA1 over each float64-contiguous array (exact float64 bytes).

    This mirrors the ~5-line primitive in
    ``cogwheel.lensing.born_residual_chart`` (and ``cogwheel.lensing.ppgo_map``)
    rather than importing it: a variadic form keeps the helper local to this
    module while hashing every stored grid, both coefficient arrays,
    ``derate`` (passed as a bare float, which ``ascontiguousarray`` folds to a
    0-d float64 array) and the per-cell ``declined_mask`` (a boolean array,
    folded to float64 0.0/1.0 bytes) so tampering with any of them -- a
    stale/tampered all-False mask would silently un-decline the near-fold
    resonance cells -- is detected on load.
    """
    hasher = hashlib.sha1()
    for array in arrays:
        hasher.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return hasher.hexdigest()


@dataclass(frozen=True)
class LowWDiffractiveChart:
    """Frozen 4-D interpolation chart for the low-w diffractive residual.

    Attributes
    ----------
    gamma_prime_grid : ndarray
        1-D ascending grid of reduced-shear ``gamma'`` values.
    rho_grid : ndarray
        1-D ascending grid of caustic-relative distance ``rho`` values.
    theta_grid : ndarray
        1-D ascending grid of eigenframe angle ``theta`` values (radians,
        covering the folded ``[0, pi/2]`` domain).
    log_w_grid : ndarray
        1-D ascending grid of ``log(w)`` values (natural logarithm of the
        dimensionless frequency).
    real_coeffs : ndarray
        4-D array shape ``(n_gp, n_rho, n_theta, n_w)`` holding the real
        part of the residual at the grid nodes.
    imag_coeffs : ndarray
        4-D array shape ``(n_gp, n_rho, n_theta, n_w)`` holding the
        imaginary part of the residual at the grid nodes.
    derate : float
        Scalar de-rate factor applied by the serve path (not here), which
        reads ``self.derate`` and scales the re-modulated residual.  Default
        1.0 (no de-rate).
    declined_mask : ndarray
        3-D boolean array, shape ``(n_gp, n_rho, n_theta)``, flagging the
        cells the training oracle measured as unable to meet the served
        two-sided certification bar (near-fold resonance-limited cells).
        The serve falls through to the exact engine for a covered draw in a
        declined cell (never an amplitude scale); default is empty (no
        declines), the pre-mask artifact shape.
    provenance : dict
        Optional metadata dictionary (training date, driver version,
        accuracy stats, etc.).  Not used by the serve path; preserved for
        audit/reproducibility.
    """

    gamma_prime_grid: np.ndarray
    rho_grid: np.ndarray
    theta_grid: np.ndarray
    log_w_grid: np.ndarray
    real_coeffs: np.ndarray
    imag_coeffs: np.ndarray
    derate: float = 1.0
    declined_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, 0), dtype=bool))
    provenance: dict = field(default_factory=dict)

    def covers(self, gamma_prime: float, rho: float,
               w: np.ndarray | None = None) -> bool:
        """Coverage predicate: box containment AND the shell/wall UNION band.

        Parameters
        ----------
        gamma_prime : float
            Reduced shear of the candidate.
        rho : float
            Caustic-relative distance of the candidate.
        w : ndarray, optional
            Served dimensionless-frequency band.  When provided (and
            non-empty), containment additionally requires the whole band to
            lie within the trained ``log_w_grid`` range, so that
            :meth:`evaluate` never cubic-extrapolates off the frequency
            axis.

        Returns
        -------
        bool
            True if (gamma_prime, rho) lies within the grid's bounding box
            (inclusive on both ends) AND inside the union band
            ``(RHO_LO <= rho <= RHO_HI) or (gamma_prime > _WALL_GAMMA_PRIME)``
            and, when ``w`` is given, the served band lies within the trained
            log-w range.
        """
        in_box = (self.gamma_prime_grid[0] <= gamma_prime
                  <= self.gamma_prime_grid[-1]
                  and self.rho_grid[0] <= rho <= self.rho_grid[-1])
        if not in_box:
            return False
        in_band = ((RHO_LO <= rho <= RHO_HI)
                   or (gamma_prime > _WALL_GAMMA_PRIME))
        if not in_band:
            return False
        if w is None:
            return True
        w = np.asarray(w, dtype=float)
        if w.size == 0:
            return True
        log_w = np.log(w)
        return (float(log_w.min()) >= self.log_w_grid[0]
                and float(log_w.max()) <= self.log_w_grid[-1])

    def declined(self, gamma_prime: float, rho: float,
                 theta: float) -> bool:
        """Whether (gamma_prime, rho, theta) falls in a declined cell.

        Returns ``True`` for a point whose containing grid cell the training
        oracle flagged as unable to meet the served two-sided certification
        bar (the sup-over-``w`` value ``|derate * r - r_engine| / |r_engine|``
        exceeds the bar anywhere in the cell's neighborhood), so the serve
        must fall through to the exact engine instead of serving the
        interpolated residual.  ``theta`` is folded to ``[0, pi/2]`` exactly
        as in :meth:`evaluate`; the empty (pre-mask) artifact reports no
        declines.
        """
        if self.declined_mask.size == 0:
            return False
        theta_folded = abs(theta) % np.pi
        if theta_folded > np.pi / 2:
            theta_folded = np.pi - theta_folded
        i_gp = self._cell_index(self.gamma_prime_grid, gamma_prime)
        i_rho = self._cell_index(self.rho_grid, rho)
        i_theta = self._cell_index(self.theta_grid, theta_folded)
        return bool(self.declined_mask[i_gp, i_rho, i_theta])

    @staticmethod
    def _cell_index(grid: np.ndarray, x: float) -> int:
        """Index of the grid node at or just below ``x``, clamped to bounds."""
        idx = int(np.searchsorted(grid, x, side='right') - 1)
        if idx < 0:
            idx = 0
        elif idx > grid.size - 1:
            idx = grid.size - 1
        return idx

    def evaluate(self, w: np.ndarray, gamma_prime: float, rho: float,
                 theta: float) -> np.ndarray:
        """Interpolate the residual r(w; gamma', rho, theta).

        Folds ``theta`` to ``[0, pi/2]`` via the D2 symmetry
        ``theta -> |theta| mod pi -> pi - theta if > pi/2``, then uses 4-D
        tensor-product cubic interpolation over
        (gamma_prime, rho, theta, log w).  The interpolator is lazily
        constructed and cached on first call (stored via
        ``object.__setattr__`` on the frozen dataclass).

        Parameters
        ----------
        w : ndarray
            1-D array of dimensionless frequencies (positive).
        gamma_prime : float
            Reduced shear of the candidate.
        rho : float
            Caustic-relative distance of the candidate.
        theta : float
            Eigenframe angle (radians) of the candidate; folded to
            [0, pi/2].

        Returns
        -------
        ndarray
            Complex residual array, shape matching ``w``.  These are the
            REDUCED-frame values ``r_pure``; the re-modulation
            (``F_serve = mass_sheet_phase * prefactor_c(w) * sqrt_mu_full *
            r_pure``) is applied by the likelihood serve, NOT here.
        """
        w = np.asarray(w, dtype=float)
        log_w = np.log(w)

        theta_folded = abs(theta) % np.pi
        if theta_folded > np.pi / 2:
            theta_folded = np.pi - theta_folded

        # Lazy-build and cache the interpolators on first call.
        # Use object.__setattr__ to bypass the frozen guard.
        if not hasattr(self, '_real_interp'):
            real_interp = RegularGridInterpolator(
                (self.gamma_prime_grid, self.rho_grid, self.theta_grid,
                 self.log_w_grid),
                self.real_coeffs, method='cubic',
                bounds_error=False, fill_value=None)
            imag_interp = RegularGridInterpolator(
                (self.gamma_prime_grid, self.rho_grid, self.theta_grid,
                 self.log_w_grid),
                self.imag_coeffs, method='cubic',
                bounds_error=False, fill_value=None)
            object.__setattr__(self, '_real_interp', real_interp)
            object.__setattr__(self, '_imag_interp', imag_interp)

        # Build the query points: shape (n_w, 4) with columns
        # (gamma_prime, rho, theta, log_w).
        points = np.empty((log_w.size, 4), dtype=float)
        points[:, 0] = gamma_prime
        points[:, 1] = rho
        points[:, 2] = theta_folded
        points[:, 3] = log_w

        real_part = self._real_interp(points)
        imag_part = self._imag_interp(points)
        return real_part + 1j * imag_part

    @classmethod
    def load(cls, path: str | Path | None = None) -> 'LowWDiffractiveChart':
        """Load and hash-verify a low-w diffractive chart artifact.

        Parameters
        ----------
        path : str or Path, optional
            Explicit artifact path; ``None`` resolves the shipped
            package-data default under ``cogwheel/data/``.

        Returns
        -------
        LowWDiffractiveChart
            The reconstructed, schema- and hash-verified chart.

        Raises
        ------
        ValueError
            If the ``schema`` key is absent or does not match
            ``low_w_diffractive_v1``, or if the recomputed content hash does
            not match the stored one (corrupt / stale artifact).  The
            message names ``scripts/train_low_w_diffractive_chart.py`` as
            the regeneration script.
        """
        if path is None:
            path = cls._default_artifact_path()
        with np.load(path, allow_pickle=False) as data:
            if 'schema' not in data.files:
                raise ValueError(
                    'Low-w diffractive chart artifact is missing the '
                    '`schema` key; it is pre-schema, stale, or corrupt. '
                    'Regenerate with '
                    'scripts/train_low_w_diffractive_chart.py.')
            schema = str(data['schema'])
            if schema != _SCHEMA:
                raise ValueError(
                    f'Low-w diffractive chart schema mismatch: stored '
                    f'{schema!r}, expected {_SCHEMA!r}. The artifact is '
                    f'stale or foreign; regenerate with '
                    f'scripts/train_low_w_diffractive_chart.py.')
            if 'content_hash' not in data.files:
                raise ValueError(
                    'Low-w diffractive chart artifact is missing the '
                    '`content_hash` key; it is stale or corrupt. Regenerate '
                    'with scripts/train_low_w_diffractive_chart.py.')
            expected = str(data['content_hash'])
            gamma_prime_grid = np.asarray(data['gamma_prime_grid'],
                                          dtype=np.float64)
            rho_grid = np.asarray(data['rho_grid'], dtype=np.float64)
            theta_grid = np.asarray(data['theta_grid'], dtype=np.float64)
            log_w_grid = np.asarray(data['log_w_grid'], dtype=np.float64)
            real_coeffs = np.asarray(data['real_coeffs'], dtype=np.float64)
            imag_coeffs = np.asarray(data['imag_coeffs'], dtype=np.float64)
            derate = (float(data['derate'])
                      if 'derate' in data.files else 1.0)
            if 'declined_mask' in data.files:
                declined_mask = np.asarray(data['declined_mask'], dtype=bool)
            else:
                declined_mask = np.zeros(
                    (gamma_prime_grid.size, rho_grid.size, theta_grid.size),
                    dtype=bool)
            provenance = json.loads(str(data['provenance']))

        actual = _content_hash(gamma_prime_grid, rho_grid, theta_grid,
                               log_w_grid, real_coeffs, imag_coeffs, derate,
                               declined_mask)
        if expected != actual:
            raise ValueError(
                f'Low-w diffractive chart content-hash mismatch: stored '
                f'{expected!r}, recomputed {actual!r}. The artifact is '
                f'corrupt or stale; regenerate with '
                f'scripts/train_low_w_diffractive_chart.py.')
        return cls(gamma_prime_grid=gamma_prime_grid, rho_grid=rho_grid,
                   theta_grid=theta_grid, log_w_grid=log_w_grid,
                   real_coeffs=real_coeffs, imag_coeffs=imag_coeffs,
                   derate=derate, declined_mask=declined_mask,
                   provenance=provenance)

    @staticmethod
    def _default_artifact_path() -> Path:
        """Resolve the shipped package-data artifact under cogwheel/data."""
        return Path(str(files('cogwheel').joinpath('data',
                                                    _DEFAULT_CHART_NAME)))
