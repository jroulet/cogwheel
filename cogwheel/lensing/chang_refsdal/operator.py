"""Contour-free Chang-Refsdal amplification operator ``F_op``.

WHAT
----
`F_op` evaluates the exact wave-optics amplification of an aligned
Chang-Refsdal lens (point mass plus external convergence ``kappa`` and
shear ``gamma``) at a single dimensionless frequency ``w``, without a
lens-plane contour integral.  Writing ``F = exp(i*w*|y|**2/2) * G``,

    G_CR = exp[i*gamma*D_beta / (2*w)] G_PM,

where ``D_beta`` is the rotated traceless second-derivative operator and
``G_PM`` together with all its radial ``s``-derivatives is supplied by
the double-double point-mass kernel `_hyp1f1.point_mass_g_derivatives`.
The operator exponential is summed as a power series in
``i*gamma/(2*w)``; `geometric_amplification` provides the ``w -> inf``
stationary-phase alternative, and `select_branch` is the authoritative
gate that decides between them.

WHY ONE REAL TABLE
------------------
The prototype rebuilds a shear-orientation-dependent representation of
``D_beta**n`` for every ``beta`` and caches it behind an ``lru_cache``
on a float, which hands callers mutable dicts (a silent
cache-corruption hazard).  Instead we ROTATE INTO THE SHEAR EIGENFRAME.
With ``z = y_1 + i*y_2`` the beta=0 operator is
``D_0 = 2*d_z**2 + 2*d_zbar**2`` with REAL coefficients, and a monomial
``z**a * zbar**b * G^(k)(s)`` reached by ``p`` applications of
``d_z**2`` and ``q`` of ``d_zbar**2`` picks up ``exp(i*beta*(b - a))``
relative to beta=0.  Evaluating the beta=0 table at the rotated point
``z_eig = exp(-i*beta) * z`` reproduces the full beta dependence
exactly, because ``F`` is a scalar and rigidly rotating the whole
configuration leaves it invariant.  So there is exactly ONE
integer-keyed real table, stored as a dense array; no float ever keys a
cache and no mutable container is returned to a caller.

Each application of ``D_0`` lowers ``a + b - 2*k`` by exactly two, so
starting from ``(a, b, k) = (0, 0, 0)`` every order-``n`` monomial obeys
``k = (a + b)//2 + n``.  The radial-derivative index is therefore a
function of ``(a, b, n)`` and the table needs only the ``(n, a, b)``
axes; the ladder length handed to the kernel is ``2 * max_order``.

CERTIFIED DOMAIN AND THE BRANCH GATE
------------------------------------
The wave branch inherits the kernel's cancellation law: the series'
partial terms reach ``e**(w*Y)`` with ``Y = |y'|`` while the sum is
O(1), so double-double holds the relative error near ~1e-10 out to
``w*Y ~ 50`` and degrades to ~1e-6 at the kernel ceiling ``w*Y = 60``.
Above ``w = _hyp1f1.W_MAX_CERTIFIED`` the kernel raises
`_hyp1f1.HypergeometricDomainError`; that is allowed to propagate here
rather than being caught and re-wrapped.

The geometric branch is legitimate only once two INDEPENDENT conditions
hold, and `select_branch` requires BOTH:

* Resolution ``w * delta_min >= RHO_END`` -- neighbouring Fermat delays
  are separated by at least one wave cycle, ``delta_min`` being the
  smallest pairwise delay separation.  ``RHO_END`` is the upper edge of
  the smooth switch window ``[RHO_START, RHO_END]`` used by the channel
  tracker: the switch hands each channel gauge to its physical target
  over that window, and geometric optics is legitimate exactly once
  every channel has been fully handed over, i.e. at ``rho >= RHO_END``.
  That is why the certified onset ``rho1`` equals ``RHO_END`` and not
  some looser threshold -- resolution alone does not license the
  asymptote.
* Strong cancellation ``L > L_MAX`` -- the wave-kernel cancellation
  exponent ``L ~ w*Y`` (see `cancellation_exponent`) has grown past
  ``L_MAX = 48``, just below the kernel ceiling of 60 and above the
  geometric onset near 50, so the two branches overlap.  Cancellation
  alone does not license the asymptote either: an unresolved cluster
  still needs the wave evaluation regardless of how deep the
  cancellation is.

The smooth switch is an error-free smoothness device owned by the
channel algebra; it blends channel gauges while preserving the exact
total.  It is NOT this gate: the switch chooses a gauge, the gate
chooses an evaluation method, and the two thresholds must not leak into
each other.

External convergence enters through the exact mass-sheet rescaling
``x' = sqrt(lam)*x``, ``y' = y/sqrt(lam)`` with ``lam = 1 - kappa`` and
effective shear ``gamma/lam``, implemented ONCE in `_mass_sheet_map` and
routed through by every kappa-dependent path of the wave branch; the
geometric branch obtains convergence directly from
`geometry.macro_matrix`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal._hyp1f1 import (
    point_mass_g_derivatives)

__all__ = [
    'RHO_START', 'RHO_END', 'L_MAX', 'MAX_ORDER',
    'OperatorDiagnostics', 'CancellationError',
    'F_op', 'geometric_amplification', 'select_branch',
    'cancellation_exponent',
]

#: Lower edge of the smooth-switch window shared with the channel
#: tracker.  Defined here so the switch and the gate have ONE home; the
#: gate itself does not use it (the switch does).
RHO_START = 0.5

#: Upper edge of the smooth-switch window and the geometric-optics
#: resolution onset ``rho1`` (see the module docstring).
RHO_END = 4.0

#: Cancellation-exponent threshold above which, once resolved, the
#: geometric branch is certified.  Sits below the kernel ceiling of 60
#: and above the geometric onset near 50 so the branches overlap.
L_MAX = 48

#: Operator-series order cap.  The kernel derivative ladder handed to
#: `_hyp1f1.point_mass_g_derivatives` is ``2 * MAX_ORDER`` because each
#: application of ``D_0`` raises the radial index by up to two.
MAX_ORDER = 42

#: Refuse and raise `CancellationError` when the measured operator-series
#: cancellation ratio ``max_partial_term / |total|`` exceeds this: past
#: ~13 digits the double-double substrate no longer protects the sum.
_CANCELLATION_REFUSAL = 1e13

# Operator-series convergence policy (not per-call knobs).
_MIN_ORDER = 6
_CONSECUTIVE_SMALL = 4
_SERIES_TOLERANCE = 2e-12

#: Cache of dense operator tables keyed by integer ``max_order``.  The
#: arrays are marked read-only and never returned to callers.
_TABLE_CACHE: dict[int, np.ndarray] = {}


class CancellationError(RuntimeError):
    """Raised when two channels cancel past the certified depth.

    The runtime expression of the two-channel cancellation law: the
    operator series has lost so many digits to cancellation that the
    result can no longer be trusted, and refusing is safer than
    returning a plausible-but-wrong amplification.
    """


@dataclass(frozen=True)
class OperatorDiagnostics:
    """Frozen report on one `F_op` evaluation.

    Frozen because a report that a caller can edit is a report that gets
    believed after being edited.

    Attributes
    ----------
    order_used : int
        Highest operator order actually summed.
    converged : bool
        Whether the operator series met the small-term stopping rule
        before reaching ``max_order``.
    estimated_relative_tail : float
        MEASURED truncation estimate: the larger of the operator
        series' last-term ratio and the kernel's worst per-order
        relative tail.  Correctness rests on this measurement, not on
        any order heuristic being tight.
    cancellation_ratio : float
        MEASURED ``max_partial_term / |total|`` over the operator
        summation; the refusal in `F_op` triggers off this quantity.
    """

    order_used: int
    converged: bool
    estimated_relative_tail: float
    cancellation_ratio: float


def _build_operator_table(max_order: int) -> np.ndarray:
    """Dense real table of ``D_0**n`` in the shear eigenframe.

    Parameters
    ----------
    max_order : int
        Highest operator order to tabulate.

    Returns
    -------
    np.ndarray
        Read-only array of shape ``(max_order + 1, dim, dim)`` with
        ``dim = 2*max_order + 1``.  Entry ``[n, a, b]`` is the real
        coefficient of the monomial ``z**a * zbar**b * G^(k)(s)`` in
        ``D_0**n``, with ``k = (a + b)//2 + n`` implied.
    """
    dim = 2 * max_order + 1
    table = np.zeros((max_order + 1, dim, dim), dtype=float)
    table[0, 0, 0] = 1.0
    for order in range(1, max_order + 1):
        prev = table[order - 1]
        cur = table[order]
        for a in range(dim):
            for b in range(dim):
                coeff = prev[a, b]
                if coeff == 0.0:
                    continue
                # D_0 = 2 d_z^2 + 2 d_zbar^2, real in the eigenframe.
                if a >= 2:
                    cur[a - 2, b] += 2.0 * coeff * a * (a - 1)
                if a >= 1:
                    cur[a - 1, b + 1] += 4.0 * coeff * a
                cur[a, b + 2] += 2.0 * coeff
                if b >= 2:
                    cur[a, b - 2] += 2.0 * coeff * b * (b - 1)
                if b >= 1:
                    cur[a + 1, b - 1] += 4.0 * coeff * b
                cur[a + 2, b] += 2.0 * coeff
    table.flags.writeable = False
    return table


def _operator_table(max_order: int) -> np.ndarray:
    """Return the cached read-only operator table for ``max_order``."""
    table = _TABLE_CACHE.get(max_order)
    if table is None:
        table = _build_operator_table(max_order)
        _TABLE_CACHE[max_order] = table
    return table


def _mass_sheet_map(y: np.ndarray, gamma: float, kappa: float
                    ) -> tuple[float, np.ndarray, float]:
    """Exact mass-sheet rescaling to the pure-shear problem.

    ``x' = sqrt(lam)*x``, ``y' = y/sqrt(lam)``, ``gamma' = gamma/lam``
    with ``lam = 1 - kappa``.  This is the SINGLE implementation of the
    map; every kappa-dependent path of the wave branch routes through
    it.

    Parameters
    ----------
    y : np.ndarray
        Shape ``(2,)`` source position.
    gamma : float
        External shear magnitude.
    kappa : float
        External convergence.

    Returns
    -------
    lam : float
        ``1 - kappa``.
    y_scaled : np.ndarray
        Rescaled source ``y / sqrt(lam)``.
    gamma_scaled : float
        Effective shear ``gamma / lam``.

    Raises
    ------
    geometry.LensDomainError
        If ``1 - kappa <= abs(gamma)`` (outside the positive-parity
        macro-image regime).
    """
    gamma = float(gamma)
    lam = 1.0 - float(kappa)
    if not lam > abs(gamma):
        raise geometry.LensDomainError(
            f'Cannot rescale the mass sheet for (kappa, gamma) = '
            f'({kappa}, {gamma}): the positive-parity condition '
            f'1 - kappa > |gamma| requires |gamma| < {lam}. Macro '
            f'saddles are out of scope; restrict to the '
            f'positive-parity regime.')
    y = np.asarray(y, dtype=float)
    if y.shape != (2,):
        raise ValueError(
            f'Source position must have shape (2,), got {y.shape}.')
    return lam, y / np.sqrt(lam), gamma / lam


def _series_length(w: float, s: float) -> int:
    """Adaptive kernel series length ``ceil(zeta + 5*sqrt(zeta) + 10)``.

    ``zeta = |z| = w*s/2`` is the magnitude of the confluent argument.
    This is a HEURISTIC: it is handed to the kernel, whose MEASURED tail
    is what the diagnostics report, so correctness never rests on the
    formula being tight.
    """
    zeta = 0.5 * w * s
    return int(np.ceil(zeta + 5.0 * np.sqrt(zeta) + 10.0))


def F_op(w: float, y: np.ndarray, gamma: float, *,
         beta: float = 0.0, kappa: float = 0.0,
         max_order: int = MAX_ORDER
         ) -> tuple[complex, OperatorDiagnostics]:
    """Contour-free Chang-Refsdal amplification at one frequency.

    Parameters
    ----------
    w : float
        Dimensionless frequency, ``0 < w <= _hyp1f1.W_MAX_CERTIFIED``.
    y : np.ndarray
        Shape ``(2,)`` source position (physical frame).
    gamma : float
        External shear magnitude.
    beta : float, optional
        External shear orientation, radians.  Rotated away into the
        eigenframe; the amplification is invariant.
    kappa : float, optional
        External convergence; enters through `_mass_sheet_map`.
    max_order : int, optional
        Operator-series order cap; also fixes the kernel ladder length
        ``2 * max_order``.

    Returns
    -------
    value : complex
        The amplification ``F``.
    diagnostics : OperatorDiagnostics
        MEASURED convergence and cancellation report.

    Raises
    ------
    geometry.LensDomainError
        If ``1 - kappa <= abs(gamma)``.
    CancellationError
        If the measured operator-series cancellation ratio exceeds
        ``_CANCELLATION_REFUSAL``.
    _hyp1f1.HypergeometricDomainError
        Propagated from the kernel above its certified ``w`` ceiling or
        cancellation-exponent ceiling.
    """
    w = float(w)
    lam, y_scaled, gamma_scaled = _mass_sheet_map(y, gamma, kappa)
    s = float(y_scaled @ y_scaled)

    table = _operator_table(max_order)
    dim = 2 * max_order + 1
    n_terms = _series_length(w, s)
    derivs, relative_tail = point_mass_g_derivatives(
        w, s, 2 * max_order, n_terms)

    # Evaluate the beta=0 table at the eigenframe-rotated source; the
    # exp(-1j*beta) rotation reproduces the full shear-orientation
    # dependence (see module docstring).
    z_eig = np.exp(-1j * beta) * complex(y_scaled[0], y_scaled[1])
    powers = np.arange(dim)
    z_powers = z_eig ** powers
    zbar_powers = np.conjugate(z_eig) ** powers
    half_sum = (np.add.outer(powers, powers) // 2)

    total = 0.0 + 0.0j
    coeff = 1.0 + 0.0j
    max_term = 0.0
    small_count = 0
    converged = False
    order_used = 0
    last_ratio = np.inf
    for order in range(max_order + 1):
        if order:
            coeff *= 1j * gamma_scaled / (2.0 * w * order)
        radial = derivs[half_sum + order]
        contribution = z_powers @ (table[order] * radial) @ zbar_powers
        term = coeff * contribution
        total += term
        order_used = order
        term_abs = abs(term)
        max_term = max(max_term, term_abs)
        scale = max(abs(total), 1e-300)
        last_ratio = term_abs / scale
        if order >= _MIN_ORDER and term_abs <= _SERIES_TOLERANCE * scale:
            small_count += 1
            if small_count >= _CONSECUTIVE_SMALL:
                converged = True
                break
        else:
            small_count = 0

    total_abs = max(abs(total), 1e-300)
    cancellation_ratio = max_term / total_abs
    if cancellation_ratio > _CANCELLATION_REFUSAL:
        raise CancellationError(
            f'Refusing F_op at w = {w}, y = {np.asarray(y).tolist()}, '
            f'gamma = {gamma}, kappa = {kappa}: the two channels cancel '
            f'to a measured ratio max_partial_term / |total| = '
            f'{cancellation_ratio:.3e}, past the certified '
            f'{_CANCELLATION_REFUSAL:.0e}. The result cannot be trusted; '
            f'use the geometric branch or a coherent multi-image sum.')

    estimated_tail = max(last_ratio, float(np.max(relative_tail)))
    diagnostics = OperatorDiagnostics(
        order_used=order_used,
        converged=converged,
        estimated_relative_tail=estimated_tail,
        cancellation_ratio=cancellation_ratio)

    # Reconstruct F from G and undo the mass-sheet rescaling.  Because
    # y_scaled = y / sqrt(lam), the physical |y|**2 / lam is exactly s.
    phase_scaled = np.exp(0.5j * w * s)
    mass_sheet_phase = np.exp(
        0.5j * w * np.log(lam) - 0.5j * w * float(kappa) * s)
    value = complex(mass_sheet_phase * phase_scaled * total / lam)
    return value, diagnostics


def geometric_amplification(w, y: np.ndarray, gamma: float, *,
                            beta: float = 0.0, kappa: float = 0.0):
    """Stationary-phase (``w -> inf``) amplification, as glue.

    Sums ``exp(1j*w*tau_a) * geometry.image_kernel(w, image_a, matrix)``
    over `geometry.find_images`, with ``tau_a`` from `geometry.delay`.
    `geometry.image_kernel` already carries the ``w -> inf`` asymptote
    including the C1/C2 corrections; this function re-derives none of
    that physics.

    Parameters
    ----------
    w : float or np.ndarray
        Dimensionless frequency; broadcasts over the carrier.
    y : np.ndarray
        Shape ``(2,)`` source position.
    gamma : float
        External shear magnitude.
    beta : float, optional
        External shear orientation, radians.
    kappa : float, optional
        External convergence.

    Returns
    -------
    complex or np.ndarray
        The geometric-optics amplification, shaped like ``w``.

    Raises
    ------
    geometry.LensDomainError
        If ``1 - kappa <= abs(gamma)`` (from `geometry.macro_matrix`).
    """
    source = np.asarray(y, dtype=float)
    if source.shape != (2,):
        raise ValueError(
            f'Source position must have shape (2,), got {source.shape}.')
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    total = np.zeros_like(np.asarray(w, dtype=float), dtype=complex)
    for image in geometry.find_images(source, matrix):
        tau = geometry.delay(image, source, matrix)
        total = total + (np.exp(1j * np.asarray(w, dtype=float) * tau)
                         * geometry.image_kernel(w, image, matrix))
    return total[()] if total.ndim == 0 else total


def cancellation_exponent(w: float, y: np.ndarray, gamma: float = 0.0,
                          kappa: float = 0.0) -> float:
    """Wave-kernel cancellation exponent ``L = w * |y'|``.

    ``L`` is the depth of the alternating-series cancellation in the
    wave branch (partial terms reach ``e**L``) and the quantity
    `select_branch` compares against ``L_MAX``.  It depends on the
    convergence through the mass-sheet rescaling of ``y`` but not on the
    shear; ``gamma`` is required only for the positive-parity guard.

    Parameters
    ----------
    w : float
        Dimensionless frequency.
    y : np.ndarray
        Shape ``(2,)`` source position.
    gamma : float, optional
        External shear magnitude (guard only).
    kappa : float, optional
        External convergence.

    Returns
    -------
    float
        ``w * sqrt(|y'|**2)`` with ``y' = y / sqrt(1 - kappa)``.

    Raises
    ------
    geometry.LensDomainError
        If ``1 - kappa <= abs(gamma)``.
    """
    _, y_scaled, _ = _mass_sheet_map(y, gamma, kappa)
    return float(w) * float(np.sqrt(y_scaled @ y_scaled))


def select_branch(w: float, delta_min: float,
                  cancellation_exp: float) -> str:
    """Authoritative wave/geometric branch gate.

    THIS module owns the single implementation; the channel tracker
    imports it rather than redefining the thresholds.  Returns
    ``'geometric'`` only when BOTH the resolution and the cancellation
    conditions hold, and ``'wave'`` otherwise.  Neither condition alone
    licenses the asymptote (see the module docstring).

    Parameters
    ----------
    w : float
        Dimensionless frequency.
    delta_min : float
        Smallest pairwise Fermat-delay separation among the channels.
    cancellation_exp : float
        Measured cancellation exponent ``L`` (see
        `cancellation_exponent`).

    Returns
    -------
    str
        ``'geometric'`` if ``w*delta_min >= RHO_END`` and
        ``cancellation_exp > L_MAX``; otherwise ``'wave'``.
    """
    resolved = float(w) * float(delta_min) >= RHO_END
    strongly_cancelling = float(cancellation_exp) > L_MAX
    if resolved and strongly_cancelling:
        return 'geometric'
    return 'wave'
