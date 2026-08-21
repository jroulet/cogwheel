"""
Trained low-w near-fold-shell residual interpolation artifact.

WHAT
----
``LowWShellChart`` is the frozen dataclass holding a trained 4-D
tensor-product interpolation of the MACRO-LEAD demodulated-difference
residual

    R(w; gamma', rho, theta) = F_exact_demod(w) - F_carrier_demod(w)

over the reduced-coordinate grid ``(gamma', rho, theta, log w)``.  The
carrier is the macro lead ``born_lead_carrier`` =
``sqrt(mu_macro) * exp(1j w phi_geo)`` (imported from ``_born``, never
re-implemented); ``F_exact`` is the reduced-eigenframe pure-shear kernel
``f_pure = f_schwinger(w, y_eig, gamma')``.  Both sides carry the SAME
carrier phase, so the difference has no poles -- the carrier's beating
zeros cancel identically because ``F_exact`` carries the same beat.  This
is the settled BornResidualChart representation (subtract the shared
carrier phase, never divide by an oscillatory field); the quotient form it
replaces produced 5800x poles.

The chart owns the smooth low-w shell only: ``rho`` in
``[RHO_LO, RHO_HI]`` (the near-fold shell, where the fold/cusp structure
has not yet developed -- ``w * delta_min < 1``, the smooth regime) and
``theta`` folded to ``[0, pi/2]`` by the D2 symmetry.  Cells where the
fold/cusp structure develops (``w * delta_min >= 1``) decline to the fold
arm / tube chart / exact engine: the chart is not authoritative there.
The serve path reconstructs ``F = mass_sheet_phase * (carrier + R) / lam``
and re-modulates through the ``FARFIELD_DIFFRACTIVE`` gauge; this module
interpolates ``R`` only.

Coordinates are REDUCED / caustic-relative, never raw lens-plane: ``rho =
|y'| / |y_c(theta)|`` (``geometry.caustic_point`` -- the same discriminator
as the diffractive-fit fence), ``gamma'`` the reduced shear, ``theta`` the
eigenframe angle (folded to ``[0, pi/2]``), and ``log w`` the natural
logarithm of the dimensionless frequency.  The chart is produced by
``scripts/train_low_w_shell_chart.py`` and consumed by the likelihood's
low-w diffractive serve (Rung P).

WHY
---
The near-fold shell (``RHO_LO <= rho <= RHO_HI``) has no analytic serve at
low w: the order-16 shear-operator series has a convergence-radius collapse
there, the uniform Airy fold arm refuses at low w, and ``w_low_fit``
declines the shell.  At low w (``w * delta_min < 1``) the fold/cusp
structure has not developed and ``F -> sqrt(mu_macro)`` regardless of rho,
so the residual ``R = f_pure - sqrt(mu_macro) * exp(1j w phi_geo)`` is a
smooth O(1) field.  This chart interpolates that residual so shell draws
can be served without calling the exact Schwinger engine at serve time.
Schwinger is an OFFLINE oracle only (used by the training script); this
module never calls it.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal._diffractive import (
    _DIFFRACTIVE_FIT_FENCE_DELTA,
    _DIFFRACTIVE_FIT_FENCE_RHO_LO,
)
from cogwheel.lensing.chang_refsdal.operator import _real_delay_min_separation

#: Shipped package-data artifact name (under ``cogwheel/data/``).
_DEFAULT_CHART_NAME = 'low_w_shell_chart.npz'

#: Artifact schema tag.  The loader hard-refuses (``ValueError``) an
#: artifact whose ``schema`` key is absent or does not match this value,
#: so a stale/foreign npz can never be silently deserialized.
_SCHEMA = 'low_w_shell_v1'

#: Inner shell boundary ``RHO_LO`` -- single-sourced from the diffractive-fit
#: fence (the same discriminator ``w_low_fit`` uses to decline the near-fold
#: shell).  Imported, NOT re-typed, so the chart's shell and the fit's fence
#: can never drift apart.
RHO_LO = _DIFFRACTIVE_FIT_FENCE_RHO_LO

#: Outer shell boundary ``RHO_HI = 1.0 + DELTA`` (``DELTA`` from the same
#: fence).
RHO_HI = 1.0 + _DIFFRACTIVE_FIT_FENCE_DELTA


def _content_hash(*arrays: np.ndarray) -> str:
    """SHA1 over each float64-contiguous array (exact float64 bytes).

    This mirrors the ~5-line primitive in
    ``cogwheel.lensing.born_residual_chart`` (and ``cogwheel.lensing.ppgo_map``)
    rather than importing it: a variadic form keeps the helper local to this
    module while hashing every stored grid and both coefficient arrays.
    """
    hasher = hashlib.sha1()
    for array in arrays:
        hasher.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return hasher.hexdigest()


def reduced_source(gamma_prime: float, rho: float,
                   theta: float) -> np.ndarray:
    """Reconstruct the reduced eigenframe source ``y_eig`` from chart coords.

    Inverts the fence discriminator the trainer and serve both use:
    ``|y'| = rho * |caustic_point(gamma', theta)|`` (``geometry.caustic_point``,
    the ``hypot`` of the returned 2-vector), then
    ``y_eig = |y'| (cos theta, sin theta)`` -- never a numerical root-find.
    Single-sources the trainer's inline inversion so trainer and serve cannot
    drift.

    Parameters
    ----------
    gamma_prime : float
        Reduced shear.
    rho : float
        Caustic-relative distance ``|y'| / |y_c(theta)|``.
    theta : float
        Eigenframe angle (radians).

    Returns
    -------
    ndarray
        Shape ``(2,)`` reduced eigenframe source ``y_eig``.
    """
    caustic = geometry.caustic_point(gamma_prime, theta)
    y_c = math.hypot(caustic[0], caustic[1])
    r_prime = rho * y_c
    return np.array([r_prime * math.cos(theta),
                     r_prime * math.sin(theta)], dtype=float)


def _reduced_min_delay_separation(gamma_prime: float,
                                  source: np.ndarray) -> float:
    """Smallest pairwise real-image delay gap in the reduced frame.

    Wraps `operator._real_delay_min_separation` on the reduced-eigenframe
    macro matrix ``macro_matrix(gamma_prime, 0, 0)`` -- the same ``kappa = 0``
    convention the fold/cusp forms use -- returning the minimum pairwise
    Fermat-delay separation among the REAL images in the ABSOLUTE frame.
    Fewer than two real images means nothing is resolved, so ``0.0`` is
    returned (the ``w * delta_min >= 1`` resolution condition then fails).
    Single-sources the reduced-frame matrix construction so the trainer and
    serve resolve/unresolve split on the same geometry.

    Parameters
    ----------
    gamma_prime : float
        Reduced shear.
    source : ndarray
        Shape ``(2,)`` reduced eigenframe source position.

    Returns
    -------
    float
        Minimum pairwise real-image delay separation, or ``0.0`` if fewer
        than two real images exist.
    """
    matrix = geometry.macro_matrix(gamma_prime, 0.0, 0.0)
    return _real_delay_min_separation(np.asarray(source, dtype=float), matrix)


@dataclass(frozen=True)
class LowWShellChart:
    """Frozen 4-D interpolation chart for the low-w shell residual.

    Attributes
    ----------
    gamma_prime_grid : ndarray
        1-D ascending grid of reduced-shear ``gamma'`` values.
    rho_grid : ndarray
        1-D ascending grid of caustic-relative distance ``rho`` values,
        spanning ``[RHO_LO, RHO_HI]``.
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
    provenance: dict = field(default_factory=dict)

    @staticmethod
    def _fold_theta(theta: float) -> float:
        """Fold ``theta`` to ``[0, pi/2]`` via the D2 symmetry."""
        theta_folded = abs(theta) % np.pi
        if theta_folded > np.pi / 2:
            theta_folded = np.pi - theta_folded
        return float(theta_folded)

    def covers(self, gamma_prime: float, rho: float, theta: float,
               w: np.ndarray | None = None) -> bool:
        """Axis-aligned box containment check.

        Parameters
        ----------
        gamma_prime : float
            Reduced shear of the candidate.
        rho : float
            Caustic-relative distance of the candidate.
        theta : float
            Eigenframe angle (radians) of the candidate; folded to
            [0, pi/2] via the D2 symmetry before the box check.
        w : ndarray, optional
            Served dimensionless-frequency band.  When provided (and
            non-empty), containment additionally requires the whole band to
            lie within the trained ``log_w_grid`` range, so that
            :meth:`evaluate` never cubic-extrapolates off the frequency
            axis.  When ``None`` (default) only the (gamma_prime, rho,
            theta) box is checked.

        Returns
        -------
        bool
            True if (gamma_prime, rho, folded theta) lies within the grid's
            bounding box (inclusive on both ends) and, when ``w`` is given,
            the served band lies within the trained log-w range.
        """
        theta_folded = self._fold_theta(theta)
        in_box = (self.gamma_prime_grid[0] <= gamma_prime
                  <= self.gamma_prime_grid[-1]
                  and self.rho_grid[0] <= rho <= self.rho_grid[-1]
                  and self.theta_grid[0] <= theta_folded
                  <= self.theta_grid[-1])
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

    def evaluate(self, w: np.ndarray, gamma_prime: float, rho: float,
                 theta: float) -> np.ndarray:
        """Interpolate the residual R(w; gamma_prime, rho, theta).

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
            demodulated-difference values ``R = F_exact_demod -
            F_carrier_demod``; the re-modulation (add the carrier and the
            mass-sheet phase) is applied by the likelihood serve, NOT here.
        """
        w = np.asarray(w, dtype=float)
        log_w = np.log(w)
        theta_folded = self._fold_theta(theta)

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
    def load(cls, path: str | Path | None = None) -> 'LowWShellChart':
        """Load and hash-verify a low-w shell chart artifact.

        Parameters
        ----------
        path : str or Path, optional
            Explicit artifact path; ``None`` resolves the shipped
            package-data default under ``cogwheel/data/``.

        Returns
        -------
        LowWShellChart
            The reconstructed, schema- and hash-verified chart.

        Raises
        ------
        ValueError
            If the ``schema`` key is absent or does not match
            ``low_w_shell_v1``, or if the recomputed content hash does not
            match the stored one (corrupt / stale artifact).  The message
            names ``scripts/train_low_w_shell_chart.py`` as the
            regeneration script.
        """
        if path is None:
            path = cls._default_artifact_path()
        with np.load(path, allow_pickle=False) as data:
            if 'schema' not in data.files:
                raise ValueError(
                    'Low-w shell chart artifact is missing the `schema` '
                    'key; it is pre-schema, stale, or corrupt. Regenerate '
                    'with scripts/train_low_w_shell_chart.py.')
            schema = str(data['schema'])
            if schema != _SCHEMA:
                raise ValueError(
                    f'Low-w shell chart schema mismatch: stored '
                    f'{schema!r}, expected {_SCHEMA!r}. The artifact is '
                    f'stale or foreign; regenerate with '
                    f'scripts/train_low_w_shell_chart.py.')
            if 'content_hash' not in data.files:
                raise ValueError(
                    'Low-w shell chart artifact is missing the '
                    '`content_hash` key; it is stale or corrupt. Regenerate '
                    'with scripts/train_low_w_shell_chart.py.')
            expected = str(data['content_hash'])
            gamma_prime_grid = np.asarray(data['gamma_prime_grid'],
                                          dtype=np.float64)
            rho_grid = np.asarray(data['rho_grid'], dtype=np.float64)
            theta_grid = np.asarray(data['theta_grid'], dtype=np.float64)
            log_w_grid = np.asarray(data['log_w_grid'], dtype=np.float64)
            real_coeffs = np.asarray(data['real_coeffs'], dtype=np.float64)
            imag_coeffs = np.asarray(data['imag_coeffs'], dtype=np.float64)
            provenance = json.loads(str(data['provenance']))

        actual = _content_hash(gamma_prime_grid, rho_grid, theta_grid,
                               log_w_grid, real_coeffs, imag_coeffs)
        if expected != actual:
            raise ValueError(
                f'Low-w shell chart content-hash mismatch: stored '
                f'{expected!r}, recomputed {actual!r}. The artifact is '
                f'corrupt or stale; regenerate with '
                f'scripts/train_low_w_shell_chart.py.')
        return cls(gamma_prime_grid=gamma_prime_grid, rho_grid=rho_grid,
                   theta_grid=theta_grid, log_w_grid=log_w_grid,
                   real_coeffs=real_coeffs, imag_coeffs=imag_coeffs,
                   provenance=provenance)

    @staticmethod
    def _default_artifact_path() -> Path:
        """Resolve the shipped package-data artifact under cogwheel/data."""
        return Path(str(files('cogwheel').joinpath('data',
                                                    _DEFAULT_CHART_NAME)))
