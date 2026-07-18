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

The kernel is accurate, but the float64 OPERATOR CONTRACTION that sums
its derivatives has its own cancellation limit.  The radial derivatives
span a huge dynamic range, so `F_op` first factors their peak magnitude
out as an exact power of two before the matmuls (otherwise the products
overflow to a silent ``nan`` near ``L ~ 40``), which pushes the usable
contraction out to ``L ~ 45``.  `F_op` then refuses an uncertifiable
input through EITHER of two measured cuts, for two INDEPENDENT error
sources: a TRUNCATION cut on ``estimated_relative_tail`` (the operator-
series last-term ratio and the kernel's per-order tail), binding when
``max_order`` is too small for the shear series to converge; and a
CONTRACTION round-off cut on ``eps * (sum|term| / |total|)`` past
``_CONTRACTION_GUARD``, binding once the series has converged and the
float64 derivative-ladder cancellation dominates near ``L ~ 45``.
Neither cut alone suffices: at the high ``max_order`` the deep-band
sweeps use, the truncation tail goes blind (the series converges) while
the contraction round-off is the live limit.  Every uncertifiable input
in the wave branch therefore exits through a named exception -- never a
``nan`` and never a finite-but-wrong number (FINDINGS F005).

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

NORMALIZATION AND THE w -> 0 MACRO LIMIT
----------------------------------------
``F`` is normalized to NO LENS AT ALL, not to the macro image.  Both
branches agree: `F_op` reconstructs by dividing by ``lam = 1 - kappa``
(a pure convergence gives ``F = 1/(1 - kappa)`` at EVERY ``w``), and
`geometric_amplification` sums ``sqrt(|mu_a|)`` with ``mu_a`` from
`geometry.macro_matrix`, which carries the macro magnification.

As ``w -> 0`` at fixed ``y, gamma, kappa`` the point-mass diffraction
switches off and only the smooth quadratic (shear plus convergence)
part of the Fermat potential survives; a quadratic potential makes the
diffraction integral an exact Gaussian, so

    F -> 1 / sqrt((1 - kappa)**2 - gamma**2) = sqrt(mu_macro),

a real, positive, FREQUENCY- and MASS-INDEPENDENT constant -- NOT 1.
``|F| - 1`` vanishes as ``w -> 0`` only when ``gamma = kappa = 0``.
So the flat, mass-independent ``|F| - 1`` at tiny ``w`` (2.06207e-2 at
``gamma = 0.20``, ``kappa = 0``) IS that exact limit, not a numerical
singularity of the prefactor ``gamma/(2*w)``, and must not be "fixed"
by a small-``w`` short-circuit returning ``1 + O(w)`` -- that would
inject a real 2% discontinuity at the crossover and destroy the exact
pure-shear limit.  The operator suite pins this closed form as a direct
``w -> 0`` gate.

External convergence enters through the exact mass-sheet rescaling
``x' = sqrt(lam)*x``, ``y' = y/sqrt(lam)`` with ``lam = 1 - kappa`` and
effective shear ``gamma/lam``, implemented ONCE in `_mass_sheet_map` and
routed through by every kappa-dependent path of the wave branch; the
geometric branch obtains convergence directly from
`geometry.macro_matrix`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba
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

#: First-order float64 round-off unit for the operator CONTRACTION
#: (machine epsilon).  The contraction stays in complex128 -- the
#: double-double substrate lives only in the 1F1 kernel, never here
#: (FINDINGS F001) -- so its accuracy is bounded by this epsilon times
#: the measured cancellation condition ``sum|term| / |total|``.
_CONTRACTION_UNIT_ROUNDOFF = float(np.finfo(np.float64).eps)

#: Relative-accuracy target the wave-branch contraction must certify
#: (FINDINGS F005).  When the measured round-off estimate
#: ``_CONTRACTION_UNIT_ROUNDOFF * (sum|term| / |total|)`` exceeds this,
#: `F_op` raises `CancellationError` rather than returning a
#: finite-but-uncertified amplification.  This is the certification cut
#: that replaces the former silent-``nan`` overflow near ``L ~ 40``.
_CONTRACTION_TARGET = 1e-10

#: Round-off certification cut for the CONTRACTION error source, applied
#: to the measured bound ``eps * (sum|term| / |total|)``.  Set at 2e-9
#: from direct 70-dps-oracle calibration on the wave band: every config
#: the suite must return sits at or below ~1.1e-9 on this bound (largest:
#: CERT y=(0.9,0) L=43.5 at 1.12e-9; large-shear w=40 at 7.4e-10), while
#: every config whose TRUE error breaches 1e-10 sits at or above ~3e-9
#: (CERT L>=45).  The bound is a WORST-CASE upper bound, loose by
#: ~20-30x and NOT a rigorous 1e-10 proof (it can invert across shear);
#: it is a measured coarse net for the CONTRACTION blow-up the truncation
#: cut cannot see, sound inside the wave band ``L <= L_MAX`` where the
#: kernel is itself certified.  (The former 1e-8 was calibrated on
#: conflated max_order=42 measurements and let L~45-48 leak as finite-
#: but-wrong; see the refusal site.)
_CONTRACTION_GUARD = 2e-9

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
    """Adaptive kernel series length from the cancellation exponent.

    The shared-numerator terms ``P_n = (a')_n * zz**n / (n!)**2`` peak
    near ``n ~ w*sqrt(s)/2`` and reach magnitude ``e**(w*sqrt(s))``, so
    the length that resolves them scales with the CANCELLATION EXPONENT
    ``L = w*sqrt(s)`` -- NOT with ``|zz| = w*s/2``, which is smaller by a
    factor ``sqrt(s)`` and under-sizes the series whenever ``|y'| < 1``
    (the common case), leaving the whole derivative ladder truncated far
    short of the 1e-10 target well inside the certified domain.  This is
    a HEURISTIC: the kernel's MEASURED tail is what the diagnostics
    report, so correctness never rests on the formula being tight, only
    on its being long enough.
    """
    cancellation = float(w) * np.sqrt(float(s))
    return int(np.ceil(cancellation + 8.0 * np.sqrt(cancellation)
                       + 20.0))


def _refusal_message(w: float, y: np.ndarray, gamma: float,
                     kappa: float, reason: str) -> str:
    """Uniform `CancellationError` message naming the configuration.

    Every wave-branch refusal -- cancellation-ratio, contraction
    magnitude spread, or a non-finite (overflow) result -- reports the
    same ``(w, y, gamma, kappa)`` context and the same remedy, so a
    caller can always tell which configuration was refused and why.
    """
    return (
        f'Refusing F_op at w = {w}, y = {np.asarray(y).tolist()}, '
        f'gamma = {gamma}, kappa = {kappa}: {reason}. The result cannot '
        f'be trusted; use the geometric branch or a coherent '
        f'multi-image sum.')


@numba.njit(cache=True, fastmath=False)
def _contract_orders(table, derivs_scaled, z_powers, zbar_powers,
                     abs_powers, half_sum, gamma_scaled, w, max_order,
                     dim):
    """Sum the operator power series ``sum_n coeff_n * <z|D_0^n|zbar>``.

    The njit hot core of `F_op`: it accumulates the order-``n``
    contraction ``z_powers @ (table[n] * radial) @ zbar_powers`` (with
    ``radial`` the clamped radial-derivative lookup), its all-positive
    companion (the honest ``sum|term|`` that measures the contraction's
    cancellation condition), and the small-term convergence test.  The
    (dim x dim) matmuls are written as explicit loops so the kernel
    stays in nopython mode; the two-stage ``column then row`` reduction
    mirrors ``z_powers @ M @ zbar_powers`` so the accumulation order
    tracks numpy's.  It stays complex128 -- the double-double substrate
    lives only in the 1F1 kernel (FINDINGS F001) -- and factors nothing
    out itself; the caller does the exact power-of-two rescaling of
    ``derivs_scaled`` and owns every threshold, refusal, and
    reconstruction (FINDINGS F005).

    Parameters
    ----------
    table : np.ndarray
        Read-only ``(max_order + 1, dim, dim)`` operator table.
    derivs_scaled : np.ndarray
        ``(dim,)`` complex radial derivatives, already rescaled by an
        exact power of two by the caller.
    z_powers, zbar_powers : np.ndarray
        ``(dim,)`` complex monomial powers of the eigenframe source and
        its conjugate.
    abs_powers : np.ndarray
        ``(dim,)`` float magnitudes ``|z_powers|``.
    half_sum : np.ndarray
        ``(dim, dim)`` int table ``(a + b) // 2`` selecting the radial
        derivative index (before the per-order offset and clamp).
    gamma_scaled, w : float
        Effective shear and dimensionless frequency.
    max_order, dim : int
        Series order cap and monomial dimension ``2*max_order + 1``.

    Returns
    -------
    total : complex
        The summed contraction, in the caller's ``2**scale_exp`` units.
    positive_total : float
        ``sum|term|`` over the summed orders, same units.
    max_term : float
        Largest per-order ``|term|`` seen.
    order_used : int
        Highest order actually summed.
    last_ratio : float
        Last per-order ``|term| / |total|``.
    converged : bool
        Whether the small-term stopping rule fired before ``max_order``.
    """
    total = 0.0 + 0.0j          # in units of 2**scale_exp
    coeff = 1.0 + 0.0j
    max_term = 0.0
    positive_total = 0.0        # sum of |summand| magnitudes, same units
    small_count = 0
    converged = False
    order_used = 0
    last_ratio = np.inf
    for order in range(max_order + 1):
        if order:
            coeff = coeff * (1j * gamma_scaled / (2.0 * w * order))
        tbl = table[order]
        # Two-stage contraction mirroring z_powers @ (tbl * radial) @
        # zbar_powers: for each column b sum over rows a, then over b.
        # A nonzero monomial at this order obeys half_sum + order <=
        # 2*max_order = dim - 1, so out-of-range cells carry a zero table
        # coefficient and the clamped lookup is multiplied away -- but
        # the scalar index must not run off the end of ``derivs_scaled``.
        contribution = 0.0 + 0.0j
        abs_contribution = 0.0
        for b in range(dim):
            col = 0.0 + 0.0j
            abs_col = 0.0
            for a in range(dim):
                idx = half_sum[a, b] + order
                if idx > dim - 1:
                    idx = dim - 1
                rad = derivs_scaled[idx]
                coefficient = tbl[a, b]
                col += z_powers[a] * (coefficient * rad)
                abs_col += abs_powers[a] * (abs(coefficient) * abs(rad))
            contribution += col * zbar_powers[b]
            abs_contribution += abs_col * abs_powers[b]
        term = coeff * contribution
        total += term
        # All-positive contraction: the magnitude actually summed in
        # float64 for this order.  Accumulated over orders it is the
        # honest condition number ``sum|term| / |total|`` of the
        # cancellation that limits the contraction's accuracy, which
        # ``estimated_relative_tail`` (a kernel-series quantity) does not
        # bound (FINDINGS F005).
        positive_total += abs(coeff) * abs_contribution
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
    return (total, positive_total, max_term, order_used,
            last_ratio, converged)


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
        If the wave-branch contraction cannot be certified to the
        ``1e-10`` target.  This covers four refusals, all raised as
        named errors rather than returning a ``nan`` or a
        finite-but-uncertified amplification (FINDINGS F005):

        * the scaled contraction still overflows to a non-finite total;
        * the measured operator-series cancellation ratio
          ``max_partial_term / |total|`` exceeds ``_CANCELLATION_REFUSAL``
          (the gamma-channel refusal, FINDINGS F001);
        * the measured truncation tail ``estimated_relative_tail``
          exceeds ``_CONTRACTION_TARGET`` (the operator-series / kernel
          truncation cut, binding at small ``max_order``);
        * the measured contraction round-off bound
          ``_CONTRACTION_UNIT_ROUNDOFF * (sum|term| / |total|)`` exceeds
          ``_CONTRACTION_GUARD`` (the float64 derivative-ladder
          cancellation cut that replaces the former silent-``nan``
          overflow near ``L ~ 45``).
    _hyp1f1.HypergeometricDomainError
        Propagated from the kernel above its certified ``w`` ceiling or
        cancellation-exponent ceiling.

    Notes
    -----
    ``F`` is normalized to no lens at all, not to the macro image, so
    ``F(w -> 0) = 1/sqrt((1 - kappa)**2 - gamma**2) = sqrt(mu_macro)``,
    not 1.  The flat ``|F| - 1`` at tiny ``w`` is that exact limit and
    not a ``gamma/(2*w)`` prefactor singularity; see the module
    docstring before "fixing" it.
    """
    w = float(w)
    lam, y_scaled, gamma_scaled = _mass_sheet_map(y, gamma, kappa)
    s = float(y_scaled @ y_scaled)

    table = _operator_table(max_order)
    dim = 2 * max_order + 1
    n_terms = _series_length(w, s)
    derivs, relative_tail = point_mass_g_derivatives(
        w, s, 2 * max_order, n_terms)

    # Overflow-safe contraction (FINDINGS F005).  At high cancellation
    # exponent ``L = w*|y'|`` the radial derivatives span a huge dynamic
    # range, so the raw products ``table[order] * derivs`` overflow
    # float64 near ``L ~ 40`` -- poisoning the matmul with a SILENT
    # ``nan`` -- even though the physical per-order contribution is O(1).
    # Factor the dominant magnitude out as an EXACT power of two before
    # the matmuls (``np.frexp`` / ``np.ldexp`` introduce no rounding), so
    # every scaled entry is O(1) and no intermediate overflows.  The
    # whole summation then runs in units of ``2**scale_exp`` and the
    # final total is scaled back exactly.  This is NOT extended
    # precision: the contraction stays complex128, honouring the
    # two-channel error model that places double-double only in the 1F1
    # kernel (FINDINGS F001).
    max_abs = float(np.max(np.abs(derivs)))
    _, scale_exp = np.frexp(max_abs)  # max_abs == frac * 2**scale_exp
    scale_exp = int(scale_exp)
    derivs_scaled = (np.ldexp(derivs.real, -scale_exp)
                     + 1j * np.ldexp(derivs.imag, -scale_exp))

    # Evaluate the beta=0 table at the eigenframe-rotated source; the
    # exp(-1j*beta) rotation reproduces the full shear-orientation
    # dependence (see module docstring).
    z_eig = np.exp(-1j * beta) * complex(y_scaled[0], y_scaled[1])
    powers = np.arange(dim)
    z_powers = z_eig ** powers
    zbar_powers = np.conjugate(z_eig) ** powers
    abs_powers = np.abs(z_powers)
    half_sum = (np.add.outer(powers, powers) // 2)

    # The order-accumulation loop -- the (dim x dim) contraction, its
    # all-positive companion, and the small-term convergence test -- runs
    # in the njit kernel `_contract_orders`.  Everything that raises,
    # thresholds, or reconstructs stays here in Python: the non-finite
    # backstop, the cancellation-ratio refusal, both certification cuts,
    # and the mass-sheet reconstruction below are unchanged (FINDINGS
    # F005).  ``half_sum + order`` clamping now happens per-element inside
    # the kernel.
    (total, positive_total, max_term, order_used,
     last_ratio, converged) = _contract_orders(
         table, derivs_scaled, z_powers, zbar_powers, abs_powers,
         half_sum, gamma_scaled, w, max_order, dim)

    # A non-finite running total means the scaled contraction still
    # overflowed; refuse instead of letting a ``nan`` slip past the ratio
    # gates below (``nan > threshold`` is False, so those gates would NOT
    # fire on it).  This is the primary closure of the silent-nan bug.
    if not (np.isfinite(total.real) and np.isfinite(total.imag)):
        raise CancellationError(_refusal_message(
            w, y, gamma, kappa,
            'the scaled operator contraction is non-finite (overflow)'))

    total_abs = max(abs(total), 1e-300)
    cancellation_ratio = max_term / total_abs
    if cancellation_ratio > _CANCELLATION_REFUSAL:
        raise CancellationError(_refusal_message(
            w, y, gamma, kappa,
            f'the two channels cancel to a measured ratio '
            f'max_partial_term / |total| = {cancellation_ratio:.3e}, '
            f'past the certified {_CANCELLATION_REFUSAL:.0e}'))

    # TRUNCATION certification (FINDINGS F005).  ``estimated_relative_
    # tail`` is the max of the operator series' last-term ratio and the
    # kernel's worst per-order relative tail; it bounds the error from
    # stopping the shear series early (small ``max_order``) or an under-
    # resolved kernel ladder.  It is the ONLY cut that sees the
    # truncation source, and it is the BINDING one at the default cap
    # where the shear series has not converged.  Measured 2026-07-16 vs
    # the 70-dps oracle at the default cap ``max_order = MAX_ORDER = 42``,
    # where HEAD silently returned a truncated value:
    #     large-shear (w=40): tail 1.27e-4 vs true 1.26e-4, converged=
    #                         False -- the pre-existing silent hole, now
    #                         refused here.
    # At the high ``max_order`` the suite's deep-band sweeps use, this
    # tail collapses to ~1e-14 (the series converges) and goes BLIND to
    # the float64 contraction cancellation; the round-off GUARD below is
    # what certifies that regime.  The two cuts cover two independent
    # error sources and neither alone is sufficient.
    estimated_tail = max(last_ratio, float(np.max(relative_tail)))
    if estimated_tail > _CONTRACTION_TARGET:
        raise CancellationError(_refusal_message(
            w, y, gamma, kappa,
            f'the series truncation cannot certify the '
            f'{_CONTRACTION_TARGET:.0e} target: estimated relative tail '
            f'= {estimated_tail:.3e} at max_order = {max_order} '
            f'(converged = {converged})'))

    # Round-off CERTIFICATION for the CONTRACTION source.  The first-
    # order float64 round-off of the cancelling summation is
    # ``eps * (sum|term| / |total|)``.  This is the ONLY cut that sees the
    # contraction blow-up: at the tested ``max_order`` the shear series
    # converges, so ``estimated_relative_tail`` above goes blind (flat
    # ~3e-14) while the TRUE error climbs past 1e-10 near ``L ~ 45`` on
    # the deep-cancellation configs, driven by the float64 derivative-
    # ladder cancellation this bound measures.  Calibrated 2026-07-16 vs
    # the suite's 70-dps oracle (guard = _CONTRACTION_GUARD = 2e-9):
    #     CERT L=43.5:      bound 1.12e-9, true 3.1e-11  -> return
    #     large-shear w=40: bound 7.4e-10, true 2.7e-11  -> return
    #     CERT L=45:        bound 5.1e-9,  true 1.3e-10  -> refuse
    #     CERT L=48:        bound 2.1e-8,  true 7.7e-10  -> refuse
    # The bound is a WORST-CASE upper bound (loose ~20-30x, and it can
    # invert across shear), so this is a measured coarse net, not a
    # rigorous 1e-10 proof; it is sound inside the wave band L <= L_MAX,
    # where the kernel is itself certified.  The scale-invariant ratio is
    # unperturbed by the power-of-two rescaling above.
    contraction_condition = positive_total / total_abs
    contraction_error = _CONTRACTION_UNIT_ROUNDOFF * contraction_condition
    if contraction_error > _CONTRACTION_GUARD:
        raise CancellationError(_refusal_message(
            w, y, gamma, kappa,
            f'the wave-branch contraction round-off guard tripped: '
            f'eps * (sum|term| / |total|) = {contraction_error:.3e} '
            f'(guard {_CONTRACTION_GUARD:.0e}) from a magnitude spread '
            f'sum|term| / |total| = {contraction_condition:.3e}'))
    diagnostics = OperatorDiagnostics(
        order_used=order_used,
        converged=converged,
        estimated_relative_tail=estimated_tail,
        cancellation_ratio=cancellation_ratio)

    # Undo the power-of-two rescaling EXACTLY (``ldexp`` by the same
    # exponent), then reconstruct F from G and undo the mass-sheet
    # rescaling.  Because y_scaled = y / sqrt(lam), the physical
    # |y|**2 / lam is exactly s.
    total = complex(np.ldexp(total.real, scale_exp),
                    np.ldexp(total.imag, scale_exp))
    phase_scaled = np.exp(0.5j * w * s)
    mass_sheet_phase = np.exp(
        0.5j * w * np.log(lam) - 0.5j * w * float(kappa) * s)
    value = complex(mass_sheet_phase * phase_scaled * total / lam)
    if not (np.isfinite(value.real) and np.isfinite(value.imag)):
        raise CancellationError(_refusal_message(
            w, y, gamma, kappa,
            'the reconstructed amplification is non-finite (overflow)'))
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
