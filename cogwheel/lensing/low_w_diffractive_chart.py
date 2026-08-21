"""
Trained low-w diffractive residual interpolation artifact.

WHAT
----
``LowWDiffractiveChart`` is the frozen dataclass holding a trained 4-D
tensor-product interpolation of the low-w diffractive residual

    r_new(w; gamma', rho, theta) = f_pure * sqrt(1 - gamma'^2) / F_ref(w)

over the reduced-coordinate grid ``(gamma', rho, theta, w^{2/3})``.  It is the
loader + serve-side interpolation object ONLY (no engine, no training); it is
produced by ``scripts/train_low_w_diffractive_chart.py`` and consumed by the
diffractive low-w serve in the likelihood (Rung P).

Representation: the stored residual is the exact pure-shear engine value
``f_pure = f_schwinger(w, y_eig, gamma')`` stripped of the RHO-PARTITIONED
uniform reference ``F_ref = partitioned_reference(w_grid, gamma', rho,
source)`` and scaled by ``sqrt(1 - gamma'^2)``::

    r_new = f_pure * sqrt(1 - gamma'^2) / F_ref .

``partitioned_reference`` chooses the carrier by ``rho``: the caustic
neighborhood (``RHO_LO <= rho <= RHO_HI``) uses the Airy fold q=p Wronskian
form ``|F_ref|^2 ~ w^{1/3} Ai^2 + w^{-1/3} Ai'^2`` (never vanishes,
magnitude-renormalized to the macro lead at low ``w``) or, only where the
fold degenerates (``b3 -> 0``), the uniform Pearcey cusp form; the off-caustic
bands (``rho < RHO_LO`` or ``rho > RHO_HI``, the deep interior and the
far-exterior wall band) split the band at the resolution boundary
``w_split = RHO_END / delta_tau``, using the macro lead carrier
``sqrt(mu_macro) exp(1j w phi_geo)`` (`born_lead_carrier`) on the unresolved
nodes and the two-image geometric-optics sum (the ``w -> inf`` asymptote) on
the resolved nodes.

``F_ref`` replaces the exact point-mass prefactor ``C(w)`` ONLY; the
``sqrt(1 - gamma'^2)`` factor (the macro amplitude ``1 / sqrt(mu_pure)`` that
diverges at the parity wall) STAYS in the residual.  Stripping ``F_ref``
leaves a smooth, bounded residual, NOT the raw amplification.  The served
value is reconstructed by re-modulation in the likelihood serve, which returns
the REDUCED-frame residual ``r_new`` only.

Coordinates are REDUCED / caustic-relative, never raw lens-plane: ``rho =
|y'| / |y_c(theta)|`` (``geometry.caustic_point`` -- the same discriminator as
the fit fence), ``gamma'`` the reduced shear, ``theta`` the eigenframe angle
(folded to ``[0, pi/2]`` by D2 symmetry), and ``w^{2/3}`` the two-thirds
power of the dimensionless frequency.

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
import math
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal._airy_fold import (
    _B3_MIN,
    _fold_amplitudes,
    _merging_fold_pair,
    _soft_axis_cubic,
    airy_fold_value,
)
from cogwheel.lensing.chang_refsdal._born import born_lead_carrier
from cogwheel.lensing.chang_refsdal._diffractive import (
    _DIFFRACTIVE_FIT_FENCE_DELTA,
    _DIFFRACTIVE_FIT_FENCE_RHO_LO,
)
from cogwheel.lensing.chang_refsdal._gauge import smootherstep
from cogwheel.lensing.chang_refsdal._pearcey_cusp import (
    cusp_uniform_reference_grid,
)
from cogwheel.lensing.chang_refsdal.operator import (
    RHO_END,
    RHO_START,
    _real_delay_min_separation,
    geometric_amplification,
)

#: Shipped package-data artifact name (under ``cogwheel/data/``).
_DEFAULT_CHART_NAME = 'low_w_diffractive_chart.npz'

#: Artifact schema tag.  The loader hard-refuses (``ValueError``) an
#: artifact whose ``schema`` key is absent or does not match this value,
#: so a stale/foreign npz can never be silently deserialized.
_SCHEMA = 'low_w_diffractive_v3'

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


def _renormalize_macro_lead(f_ref: np.ndarray, w_grid: np.ndarray,
                            gamma_prime: float, delta_tau: float
                            ) -> np.ndarray:
    """Magnitude-renormalize a uniform reference to the macro lead at low w.

    The uniform fold/cusp forms have a non-macro low-``w`` magnitude (the
    q=p fold form diverges like ``w^{-1/6}``), so the residual
    ``r = f_pure * sqrt(1 - gamma'^2) / F_ref`` blows up or drifts at the
    band bottom.  This real, positive, phase-preserving switch renormalizes
    ``|F_ref|`` to the macro lead ``sqrt(mu_macro) = 1/sqrt(1 - gamma'^2)``
    (the ``w -> 0`` limit of ``f_pure``) at low ``w`` and leaves the raw
    form at resolved ``w``::

        h = smootherstep(w * |delta_tau|, RHO_START, RHO_END)
        f_ref *= h + (1 - h) * sqrt_mu / |f_ref|

    At ``h ~ 0`` (low ``w``) ``|f_ref| -> sqrt_mu``, so the residual
    ``r -> sqrt(1 - gamma'^2)`` is genuinely O(1) at the band bottom; at
    ``h ~ 1`` (resolved ``w``) the raw form survives where it is already
    adequate.  ``delta_tau`` is the delay scale of the transition -- the
    merging-pair gap for the fold, the smallest pairwise real-image delay
    separation (`_reduced_min_delay_separation`) for the cusp.  Both uniform
    forms renormalize through this ONE helper so their low-``w`` magnitudes
    asymptote to the SAME ``sqrt_mu`` (no step at the fold->cusp handoff).
    """
    sqrt_mu = 1.0 / math.sqrt(1.0 - gamma_prime * gamma_prime)
    h = smootherstep(w_grid * abs(delta_tau), RHO_START, RHO_END)
    return f_ref * (h + (1.0 - h) * (sqrt_mu / np.abs(f_ref)))


def _airy_fold_form(w_grid: np.ndarray, gamma_prime: float,
                    source: np.ndarray) -> tuple[np.ndarray | None, bool]:
    """Uniform Airy fold reference ``F_ref`` (q=p Wronskian form).

    Builds the q=p uniform Airy fold form (the Wronskian
    ``|F_ref|^2 ~ w^{1/3} Ai^2 + w^{-1/3} Ai'^2`` never vanishes, unlike the
    leading-order q=0 form) from the merging fold pair and the nearest-caustic
    fold frame, in the ABSOLUTE frame -- no ``t_min`` subtraction and no
    ``exp(-1j w * critical_delay)`` re-referencing, since ``f_pure`` is raw and
    the mean carrier cancels exactly.

    Deliberately does NOT call `fold_amplification`, whose q=0 +
    ``_ETA_MAX_FOLD`` certificate would wrongly refuse wall-band nodes.

    Returns ``(f_ref, cusp_transition)``.  ``f_ref`` is ``None`` on any
    refusal: a geometry ``LensDomainError`` from the solve, no merging fold
    pair, an image at the point mass (a non-finite soft-axis cubic), a
    degenerate fold amplitude, or a non-finite value.
    ``cusp_transition`` is ``True`` exactly when the refusal was the
    degenerate-fold ``b3 -> 0`` transition (`_fold_amplitudes` refused the
    ``abs(b3) <= _B3_MIN`` case) -- the caller's signal to fall back to the
    Pearcey cusp form -- and ``False`` for every other refusal (a genuine
    unbuildable).

    When buildable, the built Airy form is magnitude-renormalized so that
    its low-``w`` limit matches the macro lead ``sqrt(mu_macro) =
    1/sqrt(1 - gamma'^2)`` instead of diverging like ``w^{-1/6}``: the real,
    positive, phase-preserving switch
    ``f_ref *= h + (1 - h) * sqrt_mu / |f_ref|`` with
    ``h = smootherstep(w * |delta_tau|, RHO_START, RHO_END)``: at low ``w``
    ``h ~ 0`` and ``|f_ref| -> sqrt_mu`` (so the residual ``r ->
    sqrt(1 - gamma'^2)`` is genuinely O(1) at the band bottom, not
    ``w^{-1/6}``-blown), while at resolved ``w`` ``h ~ 1`` and the raw fold
    form survives where it is already adequate.
    """
    try:
        matrix = geometry.macro_matrix(gamma_prime, 0.0, 0.0)
        images = geometry.find_images(source, matrix)
        nearest = geometry.nearest_caustic_point(gamma_prime, 0.0, source,
                                                 kappa=0.0)
    except geometry.LensDomainError:
        return None, False

    pair = _merging_fold_pair(images, source, matrix)
    if pair is None:
        return None, False
    tau_plus, tau_minus = pair

    b3 = _soft_axis_cubic(nearest.image, nearest.soft_axis)
    if b3 is None:
        # Image at the point mass (``p <= 0``) or a non-finite cubic
        # coefficient: a genuine unbuildable, NOT the fold->cusp transition.
        return None, False

    amplitudes = _fold_amplitudes(nearest.hard_eigenvalue, b3)
    if amplitudes is None:
        # A fold-amplitude refusal.  Only the degenerate-fold ``b3 -> 0``
        # (``abs(b3) <= _B3_MIN``) case is the genuine fold->cusp transition
        # (fall back to the Pearcey cusp form); a vanished/non-finite hard
        # eigenvalue or a non-finite amplitude is a genuine unbuildable.
        if math.isfinite(b3) and abs(b3) <= _B3_MIN:
            return None, True
        return None, False
    p_amplitude, _, sigma = amplitudes

    tau_bar = 0.5 * (tau_plus + tau_minus)
    delta_tau = tau_minus - tau_plus
    xi = (3.0 * w_grid * delta_tau / 4.0) ** (2.0 / 3.0)

    f_ref = np.empty(w_grid.size, dtype=complex)
    for index, w_value in enumerate(w_grid):
        f_ref[index] = airy_fold_value(float(w_value), tau_bar,
                                       float(xi[index]), p_amplitude,
                                       p_amplitude, sigma)
    if not np.all(np.isfinite(f_ref)):
        return None, False

    f_ref = _renormalize_macro_lead(f_ref, w_grid, gamma_prime, delta_tau)
    return f_ref, False


def _pearcey_cusp_reference(w_grid: np.ndarray, gamma_prime: float,
                            source: np.ndarray) -> np.ndarray | None:
    """Uniform Pearcey cusp reference ``F_ref``, geometry shared across w.

    Fallback form for `partitioned_reference` when the Airy fold reference is
    unbuildable (``b3 -> 0``, the fold->cusp transition).  Builds the
    cluster-only uniform Pearcey form (live certified quadrature, no Pearcey
    table) via `cusp_uniform_reference_grid` with ``beta = 0.0`` and
    ``kappa = 0.0`` -- the same reduced-eigenframe ``kappa = 0`` convention
    the Airy path uses via ``macro_matrix(gamma_prime, 0, 0)`` -- in the
    ABSOLUTE frame.  The w-independent geometry/controls are solved once per
    cell and reused across w nodes, mirroring `_airy_fold_form`.

    The built form is magnitude-renormalized to the macro lead at low ``w``
    by `_renormalize_macro_lead` (``delta_tau`` = the smallest pairwise
    real-image delay separation, `_reduced_min_delay_separation`) -- the
    SAME low-``w`` normalization `_airy_fold_form` applies, so the fold and
    cusp references asymptote to the SAME ``sqrt(mu_macro)`` at the band
    bottom and do not step at the ``b3 -> 0`` handoff.

    Returns ``None`` if ANY w-node is refused (``cusp_uniform_reference_grid``
    returns ``None``) or if the sampled array is non-finite.
    """
    f_ref = cusp_uniform_reference_grid(
        np.asarray(w_grid, dtype=float), np.asarray(source, dtype=float),
        gamma_prime)
    if f_ref is None:
        return None
    delta_tau = _reduced_min_delay_separation(gamma_prime, source)
    return _renormalize_macro_lead(f_ref, np.asarray(w_grid, dtype=float),
                                   gamma_prime, delta_tau)


def partitioned_reference(w_grid: np.ndarray, gamma_prime: float,
                          rho: float, source: np.ndarray
                          ) -> tuple[np.ndarray | None, str]:
    """Rho-partitioned uniform reference ``F_ref`` on ``w_grid``.

    The chart's single reference builder, cell-partitioned by ``rho`` (the
    caustic-relative distance) and shared verbatim between the trainer and
    the serve.  Returns ``(F_ref, kind)`` with ``kind`` naming the carrier
    that built ``F_ref``: ``'airy_fold'`` (the q=p Wronskian fold form),
    ``'pearcey_cusp'`` (the uniform Pearcey cusp form, only on the genuine
    fold->cusp ``b3 -> 0`` transition), ``'macro'`` (the `born_lead_carrier`
    macro lead ``sqrt(mu_macro) exp(1j w phi_geo)``), or ``'geometric'``
    (an off-caustic cell whose band is split at ``w_split = RHO_END /
    delta_tau``: the macro lead below the split, the two-image
    geometric-optics sum above it).

    Partition (single-sourced from the diffractive-fit fence constants
    ``RHO_LO`` / ``RHO_HI``, no new literal):

    * CAUSTIC NEIGHBORHOOD (``RHO_LO <= rho <= RHO_HI``): the Airy fold form
      is primary.  A fold refusal on the ``b3 -> 0`` transition falls back
      to the Pearcey cusp form; any OTHER fold refusal is a genuine
      unbuildable (no Pearcey fallback).  ``kind`` is ``'airy_fold'``
      (possibly with ``f_ref=None`` on a genuine unbuildable) or
      ``'pearcey_cusp'``.
    * OFF-CAUSTIC (``rho < RHO_LO`` or ``rho > RHO_HI``, including the deep
      interior and the far-exterior wall band): the band is split at
      ``w_split = RHO_END / delta_tau`` (``delta_tau`` the smallest pairwise
      real-image delay separation, single-sourced from
      `_reduced_min_delay_separation`).  Unresolved nodes (``w < w_split``)
      carry the macro lead carrier; resolved nodes (``w >= w_split``) carry
      the two-image geometric-optics sum -- the ``w -> inf`` asymptote, so
      the residual against it is the smooth O(1) diffractive correction.
      A cell with any resolved node returns ``kind == 'geometric'``; a cell
      with none (``delta_tau == 0``, or the whole band below the split)
      returns ``kind == 'macro'``.  The macro lead is always buildable; a
      geometric census failure (``LensDomainError``) declines with
      ``f_ref=None``.

    ``F_ref`` is in the ABSOLUTE frame -- no ``t_min`` subtraction and no
    ``exp(-1j w * critical_delay)`` re-referencing -- since ``f_pure`` is
    raw and the mean carrier cancels exactly.  The residual this anchors is
    ``r = f_pure * sqrt(1 - gamma'^2) / F_ref``: ``F_ref`` replaces
    ``prefactor_c`` ONLY, the ``sqrt(1 - gamma'^2)`` stays in the residual.

    Parameters
    ----------
    w_grid : ndarray
        1-D array of dimensionless frequencies (positive).
    gamma_prime : float
        Reduced shear.
    rho : float
        Caustic-relative distance ``|y'| / |y_c(theta)|`` (the fence
        discriminator that selects the carrier partition).
    source : ndarray
        Shape ``(2,)`` reduced eigenframe source position.

    Returns
    -------
    tuple
        ``(f_ref, kind)``: complex ``F_ref`` sampled on ``w_grid`` (or
        ``None`` to decline), and the carrier kind string.
    """
    w_grid = np.asarray(w_grid, dtype=float)
    if RHO_LO <= rho <= RHO_HI:
        f_ref, cusp_transition = _airy_fold_form(w_grid, gamma_prime, source)
        if f_ref is not None:
            return f_ref, 'airy_fold'
        if cusp_transition:
            return (_pearcey_cusp_reference(w_grid, gamma_prime, source),
                    'pearcey_cusp')
        return None, 'airy_fold'

    f_ref = np.array(
        [born_lead_carrier(float(w_value), source[0], source[1], gamma_prime,
                           beta=0.0, kappa=0.0) for w_value in w_grid],
        dtype=complex)
    delta_tau = _reduced_min_delay_separation(gamma_prime, source)
    w_split = RHO_END / delta_tau if delta_tau > 0.0 else math.inf
    resolved = w_grid >= w_split
    if resolved.any():
        # Resolved nodes: the two-image geometric-optics sum (the ``w -> inf``
        # asymptote); unresolved nodes keep the macro lead.  The split
        # ``w_split = RHO_END / delta_tau`` is the SAME single-sourced
        # resolved/unresolved boundary the trainer and serve share.
        try:
            f_ref[resolved] = geometric_amplification(
                w_grid[resolved], source, gamma_prime)
        except geometry.LensDomainError:
            return None, 'geometric'
        return f_ref, 'geometric'
    return f_ref, 'macro'


def _reduced_min_delay_separation(gamma_prime: float,
                                  source: np.ndarray) -> float:
    """Smallest pairwise real-image delay gap in the reduced frame.

    Wraps `operator._real_delay_min_separation` on the reduced-eigenframe
    macro matrix ``macro_matrix(gamma_prime, 0, 0)`` -- the same ``kappa = 0``
    convention the fold/cusp forms use -- returning the minimum pairwise
    Fermat-delay separation among the REAL images in the ABSOLUTE frame.
    Fewer than two real images means nothing is resolved, so ``0.0`` is
    returned (the ``w * delta_min >= RHO_END`` resolution condition then
    fails).  Single-sources the reduced-frame matrix construction so the
    trainer and serve resolve/unresolve split on the same geometry.

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
    w23_grid : ndarray
        1-D ascending grid of ``w**(2/3)`` values (the two-thirds power of
        the dimensionless frequency).
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
    w23_grid: np.ndarray
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
            lie within the trained ``w23_grid`` range, so that
            :meth:`evaluate` never cubic-extrapolates off the frequency
            axis.

        Returns
        -------
        bool
            True if (gamma_prime, rho) lies within the grid's bounding box
            (inclusive on both ends) AND inside the union band
            ``(RHO_LO <= rho <= RHO_HI) or (gamma_prime > _WALL_GAMMA_PRIME)``
            and, when ``w`` is given, the served band lies within the trained
            ``w**(2/3)`` range.
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
        w23 = w ** (2.0 / 3.0)
        return (float(w23.min()) >= self.w23_grid[0]
                and float(w23.max()) <= self.w23_grid[-1])

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
        (gamma_prime, rho, theta, w**(2/3)).  The interpolator is lazily
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
            REDUCED-frame values ``r_new = f_pure * sqrt(1 - gamma'^2) /
            F_ref``; the re-modulation (multiply by ``F_ref`` and the
            mass-sheet phase) is applied by the likelihood serve, NOT here.
        """
        w = np.asarray(w, dtype=float)
        w23 = w ** (2.0 / 3.0)

        theta_folded = abs(theta) % np.pi
        if theta_folded > np.pi / 2:
            theta_folded = np.pi - theta_folded

        # Lazy-build and cache the interpolators on first call.
        # Use object.__setattr__ to bypass the frozen guard.
        if not hasattr(self, '_real_interp'):
            real_interp = RegularGridInterpolator(
                (self.gamma_prime_grid, self.rho_grid, self.theta_grid,
                 self.w23_grid),
                self.real_coeffs, method='cubic',
                bounds_error=False, fill_value=None)
            imag_interp = RegularGridInterpolator(
                (self.gamma_prime_grid, self.rho_grid, self.theta_grid,
                 self.w23_grid),
                self.imag_coeffs, method='cubic',
                bounds_error=False, fill_value=None)
            object.__setattr__(self, '_real_interp', real_interp)
            object.__setattr__(self, '_imag_interp', imag_interp)

        # Build the query points: shape (n_w, 4) with columns
        # (gamma_prime, rho, theta, w**(2/3)).
        points = np.empty((w23.size, 4), dtype=float)
        points[:, 0] = gamma_prime
        points[:, 1] = rho
        points[:, 2] = theta_folded
        points[:, 3] = w23

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
            ``low_w_diffractive_v3``, or if the recomputed content hash does
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
            w23_grid = np.asarray(data['w23_grid'], dtype=np.float64)
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
                               w23_grid, real_coeffs, imag_coeffs, derate,
                               declined_mask)
        if expected != actual:
            raise ValueError(
                f'Low-w diffractive chart content-hash mismatch: stored '
                f'{expected!r}, recomputed {actual!r}. The artifact is '
                f'corrupt or stale; regenerate with '
                f'scripts/train_low_w_diffractive_chart.py.')
        return cls(gamma_prime_grid=gamma_prime_grid, rho_grid=rho_grid,
                   theta_grid=theta_grid, w23_grid=w23_grid,
                   real_coeffs=real_coeffs, imag_coeffs=imag_coeffs,
                   derate=derate, declined_mask=declined_mask,
                   provenance=provenance)

    @staticmethod
    def _default_artifact_path() -> Path:
        """Resolve the shipped package-data artifact under cogwheel/data."""
        return Path(str(files('cogwheel').joinpath('data',
                                                    _DEFAULT_CHART_NAME)))
