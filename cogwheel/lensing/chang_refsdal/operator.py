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

MACRO-SADDLE (NEGATIVE-PARITY) DISPATCH
---------------------------------------
`F_op` and `F_op_grid` classify parity from ``lam = 1 - kappa`` versus
``|gamma|``.  Since Build 8d BOTH parities are served on the wave
branch by the exact 1D Schwinger-parameter quadrature
`_schwinger.f_schwinger`: a positive-parity host ``lam > |gamma|`` with
reduced shear ``gamma' = gamma/lam > 0`` reduces / rotates /
reconstructs through the SAME evaluator as the macro saddle (below),
and the legacy operator / 1F1 contraction is RETIRED to the shear-free
``gamma' == 0`` point-lens exit (`_grid_certified`, its F001/F005
refusal constants unchanged) and to test-only oracle duty
(`legacy_operator_oracle`).  A macro saddle ``0 < lam < |gamma|`` has
NO convergent shear operator series (the series diverges past the
parity branch point), so those configs route instead to the exact 1D
Schwinger-parameter quadrature `_schwinger.f_schwinger`, evaluated in
the shear eigenframe with the reduced shear ``gamma' = gamma/lam > 1``
and reconstructed with the SAME mass-sheet identity as the operator
path, ``F = (1/lam) * exp[i*w*(ln(lam)/2 - kappa*|y'|**2/2)] *
F_{0,gamma'}(w, y_eig)``.  The saddle mass-sheet reduction lives in a
SEPARATE `_saddle_mass_sheet_map` (opposite valid-domain assumption);
the byte-frozen `_mass_sheet_map` is left untouched.  ``lam <= 0`` (Type
III) and ``lam == |gamma|`` (the degenerate parity boundary) are named
`geometry.LensDomainError` refusals -- never a silent ``nan``.

The saddle geometric-vs-wave decision is made per node directly in the
dispatch (not through `select_branch`, whose positive-parity call stays
byte-identical): a saddle node takes the stationary-phase
`geometric_amplification` (over the real images of the indefinite
matrix) ONLY when it is BOTH resolved (``w * delta_min >= RHO_END``,
``delta_min`` the real-image delay separation) AND above the wave
ceiling ``w > _schwinger.W_CEILING_SCHWINGER`` (= 60); otherwise it
takes the Schwinger wave branch.  Because `_schwinger.f_schwinger` also
hard-refuses ``w > 60`` internally, an UNRESOLVED saddle above the
ceiling propagates `_schwinger.SchwingerCertificationError` rather than
returning a wrong value.  ``cancellation_exponent`` is untouched and
still refuses saddles: the saddle wave path takes ``w`` directly and
never computes ``L = w*|y'|``.

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

import math
from dataclasses import dataclass

import numba
import numpy as np

from cogwheel.lensing.chang_refsdal import (
    geometry, _schwinger, _airy_fold, _pearcey_cusp)
from cogwheel.lensing.chang_refsdal._dd import dd_complex_sub
from cogwheel.lensing.chang_refsdal._hyp1f1 import (
    point_mass_g_derivatives)
# Bare module-global alias for the Schwinger raw-integral njit core so the
# node-parallel driver `_schwinger_raw_integral_map` can call it inside
# `numba.prange` AND its `.py_func` chain can be patched by the F010
# self-falsification tests (the same discipline `_fused_contraction` uses
# for its module-global references).  The evaluator body itself is UNTOUCHED
# (Build 8f lever 3 restructures only the calling loop).
from cogwheel.lensing.chang_refsdal._schwinger import (
    _raw_t_integral_core as _schwinger_raw_t_integral_core)

__all__ = [
    'RHO_START', 'RHO_END', 'L_MAX', 'MAX_ORDER',
    'OperatorDiagnostics', 'CancellationError',
    'F_op', 'F_op_grid', 'geometric_amplification', 'select_branch',
    'cancellation_exponent',
]

#: Lower edge of the smooth-switch window shared with the channel
#: tracker.  Defined here so the switch and the gate have ONE home; the
#: gate itself does not use it (the switch does).
RHO_START = 0.5

#: Upper edge of the smooth-switch window and the geometric-optics
#: resolution onset ``rho1`` (see the module docstring).
RHO_END = 4.0

#: Cancellation-exponent HANDOFF threshold: above it, once resolved, the
#: geometric branch is certified.  L_MAX is a handoff exponent INSIDE the
#: certified wave/geometric overlap, NOT a one-sided accuracy floor.  The
#: wave operator series is accurate to ``L ~ 45-46`` (FINDINGS F005); the
#: geometric asymptote is accurate above its ``~50`` onset at resolved
#: clusters (FINDINGS F013, governed by ``w*delta`` NOT ``L``); ``48`` is
#: the census-(b) 13.9%-calibrated crossover; the refusal band
#: ``[46, 48]`` exits by named `CancellationError`.  ``50`` is the ceiling
#: of any defensible raise, gated by the enforcement bracket (the
#: Test-Developer's graduated audit test).  Raising L_MAX past ~48 would
#: push previously-geometric-served nodes onto the wave path past its
#: 1e-10 accuracy ceiling (~L45-46), where they refuse -- so it stays 48.
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


def _saddle_mass_sheet_map(y: np.ndarray, gamma: float, kappa: float
                           ) -> tuple[float, np.ndarray, float]:
    """Exact mass-sheet rescaling for the macro-SADDLE (negative-parity).

    Identical algebra to `_mass_sheet_map` -- ``y' = y/sqrt(lam)``,
    ``gamma' = gamma/lam`` with ``lam = 1 - kappa`` -- but a SEPARATE
    implementation with the OPPOSITE valid-domain assumption: the
    positive-parity map requires ``lam > |gamma|`` (so ``gamma' < 1``),
    whereas the saddle map requires ``0 < lam < |gamma|`` (so the reduced
    shear ``gamma' > 1`` the Schwinger evaluator consumes).  The two are
    deliberately kept apart; the byte-frozen `_mass_sheet_map` must not be
    reshaped to straddle both parities.

    Parameters
    ----------
    y : np.ndarray
        Shape ``(2,)`` source position (physical frame).
    gamma : float
        External shear magnitude.
    kappa : float
        External convergence.

    Returns
    -------
    lam : float
        ``1 - kappa`` (strictly positive, ``< |gamma|``).
    y_scaled : np.ndarray
        Rescaled source ``y / sqrt(lam)``.
    gamma_prime : float
        Reduced shear ``gamma / lam`` (``> 1`` for a genuine saddle).

    Raises
    ------
    geometry.LensDomainError
        If ``1 - kappa <= 0`` (over-critical / Type III, where the
        mass-sheet reduction ``sqrt(lam)`` is not real), or if
        ``|gamma| <= 1 - kappa`` (``lam == |gamma|`` is the degenerate
        parity boundary ``det A = 0``; ``lam > |gamma|`` is positive
        parity and belongs to the operator path).  Both are named
        refusals -- never a silent ``nan``.
    ValueError
        If ``y`` does not have shape ``(2,)``.
    """
    gamma = float(gamma)
    lam = 1.0 - float(kappa)
    if lam <= 0.0:
        raise geometry.LensDomainError(
            f'Cannot rescale the mass sheet for (kappa, gamma) = '
            f'({kappa}, {gamma}): 1 - kappa = {lam} <= 0 (kappa >= 1). '
            f'The mass-sheet reduction sqrt(1 - kappa) is not real and '
            f'over-critical / Type III configurations are out of scope.')
    if not abs(gamma) > lam:
        raise geometry.LensDomainError(
            f'Cannot apply the macro-saddle mass-sheet map for '
            f'(kappa, gamma) = ({kappa}, {gamma}): the saddle domain '
            f'requires 1 - kappa < |gamma| (|gamma| > {lam}). '
            f'|gamma| == 1 - kappa is the degenerate parity boundary '
            f'(det A = 0) and |gamma| < 1 - kappa is positive parity '
            f'(use the operator wave branch); both are named refusals.')
    y = np.asarray(y, dtype=float)
    if y.shape != (2,):
        raise ValueError(
            f'Source position must have shape (2,), got {y.shape}.')
    return lam, y / np.sqrt(lam), gamma / lam


def _real_delay_min_separation(source: np.ndarray, matrix: np.ndarray
                               ) -> float:
    """Smallest pairwise Fermat-delay separation among the real images.

    The saddle resolution measure ``delta_min`` for the geometric-branch
    gate ``w * delta_min >= RHO_END``.  Mirrors the channel tracker's
    ``_min_delay_separation`` but keys directly on `geometry.find_images`
    (which returns the real images only, so the real-only convention is
    automatic).  Fewer than two real images means nothing is resolved, so
    ``0.0`` is returned and the resolution condition fails -- keeping the
    wave branch (where the Schwinger evaluator hard-refuses ``w > 60``).

    Parameters
    ----------
    source : np.ndarray
        Shape ``(2,)`` source position (physical frame).
    matrix : np.ndarray
        Shape ``(2, 2)`` macro matrix (the indefinite saddle matrix).

    Returns
    -------
    float
        Minimum pairwise real-image delay separation, or ``0.0`` if
        fewer than two real images exist.
    """
    images = geometry.find_images(source, matrix)
    if len(images) < 2:
        return 0.0
    delays = np.array([geometry.delay(image, source, matrix)
                       for image in images])
    differences = np.abs(delays[:, None] - delays[None, :])
    upper = differences[np.triu_indices(delays.size, k=1)]
    return float(np.min(upper))


def _uniform_arm_value(w: float, y: np.ndarray, gamma: float, *,
                       beta: float = 0.0, kappa: float = 0.0
                       ) -> complex | None:
    """Uniform-asymptotic rung of the per-node serving ladder, or ``None``.

    Offered at a node the geometric and Schwinger paths have already
    declined -- ``w > _schwinger.W_CEILING_SCHWINGER`` and not
    geometric-resolved -- BEFORE the existing named
    `_schwinger.SchwingerCertificationError` refusal fires.  Tries the
    near-fold uniform Airy arm (`_airy_fold.fold_amplification`) first,
    then the near-cusp uniform Pearcey arm
    (`_pearcey_cusp.cusp_amplification`), and returns the FIRST that
    certifies a finite value; if neither certifies it returns ``None`` so
    the caller lets the existing NAMED refusal stand (refusal-conservative
    -- no swallowing, no new exception class).

    The order (fold then cusp) is a pure, deterministic function of the
    node, so the serving ladder is reproducible.  Each arm runs its own
    local caustic classification and returns ``None`` when the node is not
    of its type, so trying both in a fixed order is safe: a fold-type node
    is served by the fold arm (the cusp arm refuses it) and vice versa.

    Parameters
    ----------
    w : float
        Dimensionless frequency of the refusing node (``w > 60``).
    y : np.ndarray
        Shape ``(2,)`` source position in the physical (un-rotated) frame;
        the arms rotate into their own frames internally.
    gamma : float
        External shear magnitude.
    beta : float, optional
        External shear orientation, radians.
    kappa : float, optional
        External convergence.

    Returns
    -------
    complex or None
        The first certified uniform amplification, or ``None`` if neither
        arm certifies.
    """
    value = _airy_fold.fold_amplification(w, y, gamma, beta=beta, kappa=kappa)
    if value is not None:
        return complex(value)
    value = _pearcey_cusp.cusp_amplification(w, y, gamma, beta=beta,
                                             kappa=kappa)
    if value is not None:
        return complex(value)
    return None


@numba.njit(parallel=True, cache=True, fastmath=False)
def _schwinger_raw_integral_map(
        w_nodes, a, b, y1, y2, u_lo, u_mid, u_hi, n_panels,
        xk_hi, xk_lo, wk_hi, wk_lo):
    """Node-parallel PURE MAP of the raw Schwinger ``t``-integral.

    THE parallel hot core of the node-parallel exact wave path (Build 8f
    lever 3).  For each of the ``m`` independent nodes it evaluates the
    coarse (``N``-panel) and refined (``2N``-panel) raw ``t``-integrals via
    the byte-frozen `_schwinger._raw_t_integral_core` -- the SAME njit core,
    with the SAME per-node float64 arguments the serial `f_schwinger`
    passes -- and stores the two dd-complex results in disjoint rows.  The
    ``w``-independent eigenframe reduction (``a``, ``b``, ``y1``, ``y2``,
    the shared Gauss-Legendre rule) is computed ONCE by the Python wrapper
    and broadcast in; each ``prange`` iteration only INDEXES the per-node
    setup arrays and writes ``int_n[i]`` / ``int_2n[i]``.

    This is a pure map by construction (Professor rules for lever 3):

    * NO cross-node reduction lives in the ``prange`` (a parallel
      ``sum``/``max`` would reassociate and break bit-exactness); the
      certification magnitudes and the named-refusal AND are reduced in the
      Python wrapper `_schwinger_wave_grid_values`.
    * ``fastmath`` is OFF, so no reassociation is introduced relative to the
      serial `f_schwinger` path.
    * Each node's `_raw_t_integral_core` call is self-contained and
      deterministic in its inputs, so the result is INDEPENDENT of which
      thread runs it or of the node ordering -- the two returned arrays are
      byte-for-byte identical to a serial loop over the same nodes.
    * No per-thread scratch buffer changes the within-node accumulation
      order (the accumulation lives entirely inside the untouched core).

    The named `_schwinger.SchwingerCertificationError` is NEVER raised here
    -- it would have to cross a thread boundary; the refusal is a boolean
    reduced and raised by the Python wrapper AFTER this map completes.

    Parameters
    ----------
    w_nodes : np.ndarray
        ``(m,)`` float dimensionless frequencies, each ``0 < w <=
        W_CEILING_SCHWINGER`` (the wave-branch nodes gathered by the
        wrapper).
    a, b, y1, y2 : float
        The ``w``-independent eigenframe scalars ``1 - gamma'``,
        ``1 + gamma'`` and the two eigenframe source components, computed
        ONCE by the wrapper.
    u_lo, u_mid, u_hi : np.ndarray
        ``(m,)`` float ``ln t`` integration range ends per node
        (``u_mid == log_t_cap``), computed in the wrapper with CPython
        ``math`` so they reach the core bit-identical to `f_schwinger`.
    n_panels : np.ndarray
        ``(m,)`` int coarse composite-panel count per node; the refined
        rule uses ``2 * n_panels``.
    xk_hi, xk_lo, wk_hi, wk_lo : np.ndarray
        The shared double-double Gauss-Legendre nodes and weights.

    Returns
    -------
    int_n, int_2n : np.ndarray
        ``(m, 4)`` float dd-complex ``(re_hi, re_lo, im_hi, im_lo)`` raw
        ``t``-integrals from the ``N`` and ``2N`` rules, BEFORE any
        prefactor -- the wrapper certifies and reconstructs them.
    """
    m = w_nodes.shape[0]
    int_n = np.empty((m, 4), dtype=np.float64)
    int_2n = np.empty((m, 4), dtype=np.float64)
    for i in numba.prange(m):
        w = w_nodes[i]
        # Same call order as `f_schwinger`'s ``for n_side in (n_panels,
        # 2 * n_panels)`` loop, so int_n / int_2n match the serial path.
        rn0, rn1, rn2, rn3 = _schwinger_raw_t_integral_core(
            w, a, b, y1, y2, u_lo[i], u_mid[i], u_hi[i], n_panels[i],
            xk_hi, xk_lo, wk_hi, wk_lo)
        int_n[i, 0] = rn0
        int_n[i, 1] = rn1
        int_n[i, 2] = rn2
        int_n[i, 3] = rn3
        r20, r21, r22, r23 = _schwinger_raw_t_integral_core(
            w, a, b, y1, y2, u_lo[i], u_mid[i], u_hi[i], 2 * n_panels[i],
            xk_hi, xk_lo, wk_hi, wk_lo)
        int_2n[i, 0] = r20
        int_2n[i, 1] = r21
        int_2n[i, 2] = r22
        int_2n[i, 3] = r23
    return int_n, int_2n


def _schwinger_wave_grid_values(
        w_nodes: np.ndarray, y_eig: np.ndarray, gamma_prime: float,
        lam: float, kappa: float, s: float
        ) -> tuple[np.ndarray, np.ndarray]:
    """Byte-identical node-parallel batch of the ``w <= ceiling`` wave nodes.

    The Python wrapper around `_schwinger_raw_integral_map`.  Given the
    ``w <= W_CEILING_SCHWINGER`` wave-branch frequencies of a single lens
    config (all sharing the eigenframe ``y_eig`` and reduced shear
    ``gamma'``), it reproduces `_schwinger.f_schwinger` node by node with
    the expensive raw ``t``-integrals evaluated in PARALLEL, and returns
    the mass-sheet-reconstructed grid values together with a per-node
    certification flag.  It does NOT raise: the refusal is reduced by the
    GRID caller over the full node ordering (so the authentic named
    exception carries the lowest-index refuser's message, matching the
    serial first-refuser).

    Byte-identity is by construction: the setup scalars (``t_cap``,
    ``margin``, ``u_lo``/``u_mid``/``u_hi``, ``n_panels``) are computed here
    in CPython exactly as `f_schwinger` computes them (same ``math.log``,
    same `_schwinger._panel_count`), the raw integrals come from the SAME
    frozen core, and the certification + `_schwinger._reconstruct` +
    mass-sheet reconstruction reuse the SAME frozen helpers and tolerance.
    The only re-sequencing is that the per-node quadrature now runs across
    threads; each node's arithmetic sequence is untouched.

    NOTE (maintenance coupling): the setup, certification and
    reconstruction below MIRROR the body of `_schwinger.f_schwinger`; they
    are byte-identity-critical.  If `f_schwinger`'s setup or certification
    ever changes, this wrapper must change in lockstep (the byte-identity
    test suite is the guard).  The heavy `_raw_t_integral_core` math itself
    is reused, not reimplemented.

    Parameters
    ----------
    w_nodes : np.ndarray
        ``(m,)`` float frequencies with ``0 < w <= W_CEILING_SCHWINGER``.
    y_eig : np.ndarray
        Shape ``(2,)`` eigenframe source position (soft/hard axes).
    gamma_prime : float
        Reduced external shear ``gamma' > 0``.
    lam : float
        Mass-sheet scale ``lam = 1 - kappa`` for the reconstruction.
    kappa : float
        External convergence (the mass-sheet phase).
    s : float
        Rescaled ``|y'|**2`` for the mass-sheet phase.

    Returns
    -------
    values : np.ndarray
        ``(m,)`` complex grid amplifications; entries flagged uncertified
        are left unspecified (the caller refuses instead of serving them).
    certified : np.ndarray
        ``(m,)`` bool per-node paired-rule certification outcome.
    """
    m = w_nodes.shape[0]
    values = np.empty(m, dtype=complex)
    certified = np.zeros(m, dtype=bool)
    if m == 0:
        return values, certified

    a = 1.0 - gamma_prime
    b = 1.0 + gamma_prime
    y1 = float(y_eig[0])
    y2 = float(y_eig[1])

    # Per-node ``ln t`` range and panel count, computed in PLAIN CPython
    # exactly as `f_schwinger` does, so the float64 arguments reach the
    # frozen core bit-identical to the serial path (numba's libm is NOT
    # assumed to match CPython's to the last ULP -- byte-identity by
    # construction rather than by hope).
    u_lo = np.empty(m, dtype=float)
    u_mid = np.empty(m, dtype=float)
    u_hi = np.empty(m, dtype=float)
    n_panels = np.empty(m, dtype=np.int64)
    for k in range(m):
        w = float(w_nodes[k])
        t_cap = 0.5 * w * (abs(a) + abs(b) + 2.0)
        log_t_cap = math.log(t_cap)
        margin = _schwinger._CANCEL_SCALE * w + _schwinger._U_MARGIN_CONST
        u_lo[k] = log_t_cap - margin
        u_mid[k] = log_t_cap
        u_hi[k] = log_t_cap + margin
        n_panels[k] = _schwinger._panel_count(margin, w)

    xk_hi, xk_lo, wk_hi, wk_lo = _schwinger._dd_gl_rule(
        _schwinger._PANEL_ORDER)

    int_n, int_2n = _schwinger_raw_integral_map(
        np.ascontiguousarray(w_nodes, dtype=float), a, b, y1, y2,
        u_lo, u_mid, u_hi, n_panels, xk_hi, xk_lo, wk_hi, wk_lo)

    # Per-node certification + reconstruction, byte-identical to the tail of
    # `f_schwinger` (same dd_complex_sub, magnitude, _CERTIFICATION_TOL,
    # _reconstruct) and the grid mass-sheet identity.
    for k in range(m):
        rn = int_n[k]
        r2 = int_2n[k]
        difference = dd_complex_sub(
            rn[0], rn[1], rn[2], rn[3], r2[0], r2[1], r2[2], r2[3])
        reference_magnitude = _schwinger._dd_complex_magnitude(
            (r2[0], r2[1], r2[2], r2[3]))
        difference_magnitude = _schwinger._dd_complex_magnitude(difference)
        if (reference_magnitude == 0.0
                or difference_magnitude
                > _schwinger._CERTIFICATION_TOL * reference_magnitude):
            certified[k] = False
            continue
        integral = complex(r2[0] + r2[1], r2[2] + r2[3])
        w = float(w_nodes[k])
        f_pure = _schwinger._reconstruct(w, y_eig, integral)
        mass_sheet_phase = np.exp(
            0.5j * w * np.log(lam) - 0.5j * w * float(kappa) * s)
        values[k] = complex(mass_sheet_phase * f_pure / lam)
        certified[k] = True
    return values, certified


def _measure_node_parallel_speedup(
        gamma: float = 0.4, kappa: float = 0.0,
        y: tuple[float, float] = (0.3, 0.2),
        n_nodes: int = 32, repeats: int = 3) -> dict[str, float]:
    """Measure the node-parallel exact-path speedup on a small grid.

    A MEASUREMENT-ONLY diagnostic (never touches evaluator control flow),
    mirroring `_schwinger._measure_warm_cost`.  It prices the serial
    per-node `_schwinger.f_schwinger` loop against the node-parallel
    `_schwinger_wave_grid_values` batch on a positive-parity config whose
    grid sits entirely below ``W_CEILING_SCHWINGER`` (so every node takes
    the exact wave branch).  Returns a summary dict and prints one line.

    Parameters
    ----------
    gamma, kappa : float
        Positive-parity lens parameters (``1 - kappa > |gamma|``).
    y : tuple of float
        Source position.
    n_nodes : int
        Grid size (kept small so the measurement stays bounded).
    repeats : int
        Best-of timing repeats.

    Returns
    -------
    dict of float
        ``n_nodes``, ``serial_ms``, ``parallel_ms``, ``speedup``.
    """
    import time  # local: keep the timing dependency out of the hot module

    y_arr = np.asarray(y, dtype=float)
    lam, y_scaled, gamma_prime = _mass_sheet_map(y_arr, gamma, kappa)
    z_eig = np.exp(-1j * 0.0) * complex(y_scaled[0], y_scaled[1])
    y_eig = np.array([z_eig.real, z_eig.imag])
    s = float(y_scaled @ y_scaled)
    w_nodes = np.linspace(
        1.0, 0.9 * _schwinger.W_CEILING_SCHWINGER, n_nodes)

    # Warm up: trigger numba compilation of the core, the parallel map, and
    # the lru_cache population once for both paths.
    for w in w_nodes:
        _schwinger.f_schwinger(float(w), y_eig, gamma_prime)
    _schwinger_wave_grid_values(w_nodes, y_eig, gamma_prime, lam, kappa, s)

    serial_best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        for w in w_nodes:
            _schwinger.f_schwinger(float(w), y_eig, gamma_prime)
        serial_best = min(serial_best, time.perf_counter() - start)

    parallel_best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        _schwinger_wave_grid_values(w_nodes, y_eig, gamma_prime, lam, kappa, s)
        parallel_best = min(parallel_best, time.perf_counter() - start)

    summary = {
        'n_nodes': float(n_nodes),
        'serial_ms': 1e3 * serial_best,
        'parallel_ms': 1e3 * parallel_best,
        'speedup': (serial_best / parallel_best
                    if parallel_best > 0.0 else math.inf),
    }
    print(
        f'[node-parallel speedup] {summary["n_nodes"]:.0f} nodes | '
        f'serial {summary["serial_ms"]:.1f} ms | '
        f'parallel {summary["parallel_ms"]:.1f} ms | '
        f'{summary["speedup"]:.2f}x')
    return summary


def _saddle_grid(w_array: np.ndarray, y: np.ndarray, gamma: float, *,
                 beta: float = 0.0, kappa: float = 0.0) -> np.ndarray:
    """Macro-saddle amplification over a ``w`` grid, node by node.

    The negative-parity counterpart of `_grid_certified`.  A saddle host
    (``0 < 1 - kappa < |gamma|``) has no convergent operator power series
    (the shear series diverges past the parity boundary), so each node is
    evaluated by the exact 1D Schwinger-parameter quadrature
    `_schwinger.f_schwinger` in the shear eigenframe and reconstructed
    with the same mass-sheet identity the operator path uses.

    Per node the geometric-vs-wave decision is ``(resolved AND
    w > W_CEILING_SCHWINGER) -> geometric``, else the Schwinger wave
    branch.  ``resolved`` uses the frequency-independent real-image
    ``delta_min`` (computed once, only when some node exceeds the
    ceiling).  Because `_schwinger.f_schwinger` ALSO hard-refuses
    ``w > W_CEILING_SCHWINGER`` internally, an unresolved saddle above the
    ceiling is first offered to the uniform arms (`_uniform_arm_value`:
    fold Airy then cusp Pearcey); only if BOTH arms refuse does the node
    propagate `_schwinger.SchwingerCertificationError` from the wave
    branch rather than returning a wrong value.  The arm intercept fires
    ONLY on this previously-refusing branch, so resolved (geometric) and
    ``w <= W_CEILING_SCHWINGER`` nodes are byte-identical to the exact
    path.

    NODE-PARALLEL EXACT EVALUATION (Build 8f lever 3).  A Python pre-pass
    classifies each node into its branch (geometric / arm / exact wave /
    refuse) and gathers the independent ``w <= ceiling`` exact wave nodes;
    those are evaluated across cores by `_schwinger_wave_grid_values` (an
    njit ``prange`` PURE MAP over `_schwinger_raw_integral_map`, the frozen
    `_schwinger._raw_t_integral_core` unchanged).  The per-node value is
    byte-identical to the serial `f_schwinger` path (fastmath off, no
    cross-node reduction in the parallel region, the ``w``-independent
    eigenframe reduction done once here); the named
    `_schwinger.SchwingerCertificationError` never crosses a thread
    boundary -- it is reduced over the full node ordering and raised by the
    Python wrapper with the lowest-index refuser's authentic message
    (serial first-refuser identity; scheduling-independent).

    Parameters
    ----------
    w_array : np.ndarray
        ``(n_nodes,)`` dimensionless frequencies, ``w > 0``.
    y : np.ndarray
        Shape ``(2,)`` source position (physical frame).
    gamma : float
        External shear magnitude.
    beta : float, optional
        External shear orientation, radians (rotated into the eigenframe).
    kappa : float, optional
        External convergence.

    Returns
    -------
    np.ndarray
        ``(n_nodes,)`` complex amplifications ``F``.

    Raises
    ------
    geometry.LensDomainError
        If ``1 - kappa <= 0`` (Type III) or the config is not a genuine
        saddle (``|gamma| <= 1 - kappa``), via `geometry.macro_matrix`
        and `_saddle_mass_sheet_map`.
    _schwinger.SchwingerCertificationError
        If any node cannot be certified by the paired Gauss-Legendre
        rules, or an unresolved node exceeds ``W_CEILING_SCHWINGER``.
    ValueError
        If ``y`` does not have shape ``(2,)`` or ``w_array`` is not 1-D.
    """
    w_array = np.asarray(w_array, dtype=float)
    if w_array.ndim != 1:
        raise ValueError(
            f'w_array must be one-dimensional, got shape {w_array.shape}.')

    # `macro_matrix` is the parity classifier and the named Type III /
    # parity-boundary refusal (never a silent nan); it also supplies the
    # indefinite matrix the geometric branch and resolution need.
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    lam, y_scaled, gamma_prime = _saddle_mass_sheet_map(y, gamma, kappa)
    source = np.asarray(y, dtype=float)

    # w-INDEPENDENT eigenframe reduction: rotate the rescaled source into
    # the shear eigenframe (e1 = soft axis, e2 = hard axis), exactly as
    # the operator path does; the exp(-1j*beta) rotation carries the full
    # shear-orientation dependence and |y_eig|**2 == |y_scaled|**2 == s.
    z_eig = np.exp(-1j * float(beta)) * complex(y_scaled[0], y_scaled[1])
    y_eig = np.array([z_eig.real, z_eig.imag])
    s = float(y_scaled @ y_scaled)

    # Resolution is frequency-independent; compute delta_min once and only
    # if any node could take the geometric branch (w > ceiling).
    delta_min = 0.0
    if np.any(w_array > _schwinger.W_CEILING_SCHWINGER):
        delta_min = _real_delay_min_separation(source, matrix)

    n_nodes = w_array.shape[0]
    values = np.empty(n_nodes, dtype=complex)

    # Python PRE-PASS over the nodes in index order (Build 8f lever 3):
    # classify each into its serving branch and GATHER the expensive
    # ``w <= ceiling`` exact wave nodes for the node-parallel batch.  The
    # geometric and arm branches stay in Python; only the pure Schwinger
    # inner map is parallelized.  `select_branch` is NOT the saddle
    # authority (it stays byte-frozen for the positive-parity operator
    # path); the saddle takeover is owned here (channels.py / Build 7).
    batch_index: list[int] = []
    ceiling_refusers: list[int] = []
    for node in range(n_nodes):
        w_node = float(w_array[node])
        if (w_node > _schwinger.W_CEILING_SCHWINGER
                and w_node * delta_min >= RHO_END):
            # Resolved and above the wave ceiling: stationary-phase sum
            # over the real images of the indefinite matrix.
            values[node] = complex(geometric_amplification(
                w_node, y, gamma, beta=beta, kappa=kappa))
        elif w_node > _schwinger.W_CEILING_SCHWINGER:
            # Unresolved above the ceiling: `f_schwinger` would refuse
            # (SchwingerCertificationError).  Offer the uniform-asymptotic
            # rung of the serving ladder (fold then cusp arm) first; only
            # if BOTH arms refuse does the node become a refuser.
            arm_value = _uniform_arm_value(
                w_node, source, gamma, beta=beta, kappa=kappa)
            if arm_value is not None:
                values[node] = arm_value
            else:
                ceiling_refusers.append(node)
        else:
            # w <= ceiling exact wave node: the parallel batch (byte-
            # identical to the serial `f_schwinger` path per node).
            batch_index.append(node)

    batch_index_arr = np.array(batch_index, dtype=np.int64)
    batch_values, batch_cert = _schwinger_wave_grid_values(
        w_array[batch_index_arr], y_eig, gamma_prime, lam, kappa, s)

    # Reduce the named refusal ACROSS THE FULL node ordering in the Python
    # wrapper (never across a thread boundary): any node refuses -> the
    # whole grid refuses, raised with the authentic message of the
    # LOWEST-index refuser (serial first-refuser identity).
    refusers = list(ceiling_refusers)
    for pos, node in enumerate(batch_index):
        if batch_cert[pos]:
            values[node] = batch_values[pos]
        else:
            refusers.append(node)
    if refusers:
        first = int(min(refusers))
        # Re-run the lowest-index refuser through `f_schwinger` to raise
        # the exact named exception (ceiling or paired-rule); identical
        # inputs -> identical decision and message as the serial path.
        _schwinger.f_schwinger(float(w_array[first]), y_eig, gamma_prime)
        raise _schwinger.SchwingerCertificationError(  # unreachable guard
            f'Node-parallel batch flagged node {first} '
            f'(w = {float(w_array[first])}) as refused, but the serial '
            f're-evaluation certified it; refusing rather than serving an '
            f'unverified value.')
    return values


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
def _fused_contraction(table, z_powers, zbar_powers, abs_powers, half_sum,
                       derivs_scaled, w_array, gamma_scaled, max_order, dim):
    """Fused w-independent weight-vector build + batched node contraction.

    THE single njit hot core of `F_op_grid`, merging the two formerly
    separate stages -- the ``w``-INDEPENDENT per-order weight-vector build
    and the per-node operator-series contraction -- into one dispatch.
    The fusion is a DISPATCH-ONLY merge: the two loop nests below are the
    former ``_weight_vectors`` and ``_contract_grid`` bodies inlined
    verbatim, in the identical iteration order, so every float64
    add/multiply happens in the SAME sequence as before.  The weight
    vectors ``v`` / ``v_abs`` are now internal per-call temporaries rather
    than arrays handed across an njit boundary; nothing is re-associated,
    nothing switches to ``np.dot``/BLAS, and the ``(order, a, b)`` scatter
    and ``(node, order, j)`` contraction are byte-for-byte unchanged.  The
    win is eliminating the intermediate materialization/handoff of
    ``v`` / ``v_abs`` and the second njit dispatch -- NOT any arithmetic
    restructuring; the returned 6-tuple is bit-identical to the former
    ``_weight_vectors`` -> ``_contract_grid`` pipeline (FINDINGS F005).

    Stage 1 -- w-independent weight vectors.  Within one `F_op_grid` call
    the lens parameters are fixed and only ``w`` varies over the node
    grid, so ``z_powers``, ``zbar_powers``, the operator ``table`` and the
    radial-index selector ``half_sum`` are all ``w``-INDEPENDENT.  The
    order-``n`` contraction ``sum_{a,b} z_powers[a] * table[n,a,b] *
    zbar_powers[b] * derivs[idx(a,b,n)]`` (with ``idx = min(half_sum[a,b]
    + n, dim - 1)``, the SAME clamp the scalar kernel used) is regrouped
    by the radial index ``j`` into a single length-``dim`` weight vector
    ``v[n, j]`` and its all-positive companion ``v_abs[n, j]`` (built from
    ``|z_powers[a]| * |table[n,a,b]| * |zbar_powers[b]|`` for the
    ``sum|term|`` / ``max_term`` cancellation bookkeeping).  Built ONCE
    per call and reused across every node.  GATHER-INDEX INVARIANT: a
    nonzero monomial at order ``n`` obeys ``half_sum + n <= 2*max_order =
    dim - 1``, so any index that would clamp carries a zero table
    coefficient and is skipped -- the clamp never scatters a spurious
    contribution into ``v[n, dim-1]``.

    Stage 2 -- per-node contraction.  For every node it sums the operator
    power series ``sum_n coeff_n * (v[n] . derivs)`` -- one length-``dim``
    dot of the weight vector built above against that node's rescaled
    radial derivatives -- accumulates the all-positive companion
    ``sum|term|`` (the honest cancellation condition the round-off
    certification measures), and runs the same small-term convergence test
    the scalar kernel used, per node.  It stays complex128 -- the
    double-double substrate lives only in the 1F1 kernel (FINDINGS F001) --
    factors nothing out itself (the caller does the exact per-node
    power-of-two rescaling of ``derivs_scaled``), and owns no threshold,
    refusal, or reconstruction.

    F010 note: ``half_sum`` stays an explicit ARGUMENT and
    ``_SERIES_TOLERANCE`` / ``_CONSECUTIVE_SMALL`` / ``_MIN_ORDER`` are
    referenced by name as MODULE GLOBALS, so the py_func-chain
    self-falsification tests can still patch the gather index and the
    convergence tolerance and drive the accuracy gate red.

    Parameters
    ----------
    table : np.ndarray
        Read-only ``(max_order + 1, dim, dim)`` operator table.
    z_powers, zbar_powers : np.ndarray
        ``(dim,)`` complex monomial powers of the eigenframe source and
        its conjugate.
    abs_powers : np.ndarray
        ``(dim,)`` float magnitudes ``|z_powers|``.
    half_sum : np.ndarray
        ``(dim, dim)`` int table ``(a + b) // 2``.
    derivs_scaled : np.ndarray
        ``(n_nodes, dim)`` complex radial derivatives, each row already
        rescaled by that node's exact power of two by the caller.
    w_array : np.ndarray
        ``(n_nodes,)`` float dimensionless frequencies.
    gamma_scaled : float
        Effective shear ``gamma / (1 - kappa)``.
    max_order, dim : int
        Series order cap and monomial dimension ``2*max_order + 1``.

    Returns
    -------
    totals : np.ndarray
        ``(n_nodes,)`` complex summed contractions, each in its node's
        ``2**scale_exp`` units.
    positive_totals : np.ndarray
        ``(n_nodes,)`` float ``sum|term|`` per node, same units.
    max_terms : np.ndarray
        ``(n_nodes,)`` float largest per-order ``|term|`` per node.
    orders_used : np.ndarray
        ``(n_nodes,)`` int highest order actually summed per node.
    last_ratios : np.ndarray
        ``(n_nodes,)`` float last per-order ``|term| / |total|`` per node.
    converged : np.ndarray
        ``(n_nodes,)`` bool small-term-stop flag per node.
    """
    # --- Stage 1: w-independent per-order weight vectors ------------------
    # (formerly ``_weight_vectors``; loop copied verbatim so the (order, a,
    # b) scatter accumulates into v/v_abs in the identical float64 order.)
    v = np.zeros((max_order + 1, dim), dtype=np.complex128)
    v_abs = np.zeros((max_order + 1, dim), dtype=np.float64)
    for order in range(max_order + 1):
        tbl = table[order]
        vn = v[order]
        vabs = v_abs[order]
        for a in range(dim):
            za = z_powers[a]
            aa = abs_powers[a]
            for b in range(dim):
                coefficient = tbl[a, b]
                if coefficient == 0.0:
                    continue
                idx = half_sum[a, b] + order
                if idx > dim - 1:
                    idx = dim - 1
                vn[idx] += za * (coefficient * zbar_powers[b])
                vabs[idx] += aa * (abs(coefficient) * abs_powers[b])

    # --- Stage 2: batched per-node operator-series contraction -----------
    # (formerly ``_contract_grid``; loop copied verbatim so the (node,
    # order, j) contraction and small-term stop are byte-for-byte unchanged.)
    n_nodes = w_array.shape[0]
    totals = np.zeros(n_nodes, dtype=np.complex128)
    positive_totals = np.zeros(n_nodes, dtype=np.float64)
    max_terms = np.zeros(n_nodes, dtype=np.float64)
    orders_used = np.zeros(n_nodes, dtype=np.int64)
    last_ratios = np.zeros(n_nodes, dtype=np.float64)
    converged = np.zeros(n_nodes, dtype=np.bool_)
    for node in range(n_nodes):
        w = w_array[node]
        derivs = derivs_scaled[node]
        total = 0.0 + 0.0j          # in units of 2**scale_exp
        coeff = 1.0 + 0.0j
        max_term = 0.0
        positive_total = 0.0        # sum of |summand| magnitudes
        small_count = 0
        node_converged = False
        order_used = 0
        last_ratio = np.inf
        for order in range(max_order + 1):
            if order:
                coeff = coeff * (1j * gamma_scaled / (2.0 * w * order))
            vn = v[order]
            vabs = v_abs[order]
            contribution = 0.0 + 0.0j
            abs_contribution = 0.0
            for j in range(dim):
                contribution += vn[j] * derivs[j]
                abs_contribution += vabs[j] * abs(derivs[j])
            term = coeff * contribution
            total += term
            positive_total += abs(coeff) * abs_contribution
            order_used = order
            term_abs = abs(term)
            max_term = max(max_term, term_abs)
            scale = max(abs(total), 1e-300)
            last_ratio = term_abs / scale
            if (order >= _MIN_ORDER
                    and term_abs <= _SERIES_TOLERANCE * scale):
                small_count += 1
                if small_count >= _CONSECUTIVE_SMALL:
                    node_converged = True
                    break
            else:
                small_count = 0
        totals[node] = total
        positive_totals[node] = positive_total
        max_terms[node] = max_term
        orders_used[node] = order_used
        last_ratios[node] = last_ratio
        converged[node] = node_converged
    return (totals, positive_totals, max_terms, orders_used,
            last_ratios, converged)


def _grid_certified(w_array: np.ndarray, y: np.ndarray, gamma: float, *,
                    beta: float = 0.0, kappa: float = 0.0,
                    max_order: int = MAX_ORDER
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray]:
    """Shared contraction + certification for the whole ``w`` node grid.

    THE single wave-branch contraction and certification path.  Both the
    lean public `F_op_grid` and the scalar `F_op` delegate here, so there
    is exactly ONE contraction implementation and ONE application of the
    four F005 refusals -- a value returned by either entry point and the
    diagnostics reported alongside it can never disagree.

    The operator table is built ONCE and the point-mass kernel is
    evaluated per node (its series length varies with ``w``); the single
    fused njit core `_fused_contraction` builds the ``w``-independent
    weight vectors once and sums the operator series for every node, and
    then each node is certified-or-refused with the four thresholds
    BYTE-UNCHANGED from the former scalar path (FINDINGS F005/F001).

    Parameters
    ----------
    w_array : np.ndarray
        ``(n_nodes,)`` dimensionless frequencies,
        ``0 < w <= _hyp1f1.W_MAX_CERTIFIED``.
    y : np.ndarray
        Shape ``(2,)`` source position (physical frame).
    gamma : float
        External shear magnitude.
    beta : float, optional
        External shear orientation, radians (rotated into the eigenframe).
    kappa : float, optional
        External convergence; enters through `_mass_sheet_map`.
    max_order : int, optional
        Operator-series order cap; fixes the kernel ladder length
        ``2 * max_order``.

    Returns
    -------
    values : np.ndarray
        ``(n_nodes,)`` complex amplifications ``F``.
    orders_used : np.ndarray
        ``(n_nodes,)`` int highest operator order summed per node.
    converged : np.ndarray
        ``(n_nodes,)`` bool small-term-stop flag per node.
    estimated_tails : np.ndarray
        ``(n_nodes,)`` float measured truncation estimate per node.
    cancellation_ratios : np.ndarray
        ``(n_nodes,)`` float measured ``max_term / |total|`` per node.

    Raises
    ------
    geometry.LensDomainError
        If ``1 - kappa <= abs(gamma)``.
    CancellationError
        If any node cannot be certified to the ``1e-10`` target (the four
        F005 refusals, per node).
    _hyp1f1.HypergeometricDomainError
        Propagated from the kernel above its certified ceilings.
    """
    w_array = np.asarray(w_array, dtype=float)
    if w_array.ndim != 1:
        raise ValueError(
            f'w_array must be one-dimensional, got shape {w_array.shape}.')
    lam, y_scaled, gamma_scaled = _mass_sheet_map(y, gamma, kappa)
    s = float(y_scaled @ y_scaled)

    table = _operator_table(max_order)
    dim = 2 * max_order + 1

    # Every quantity feeding the w-INDEPENDENT weight vectors is fixed
    # within one grid call; the fused contraction below builds those
    # vectors ONCE internally from these inputs.  Evaluate the beta=0
    # table at the eigenframe-rotated source; the exp(-1j*beta) rotation
    # reproduces the full shear-orientation dependence (see the module
    # docstring).
    z_eig = np.exp(-1j * beta) * complex(y_scaled[0], y_scaled[1])
    powers = np.arange(dim)
    z_powers = z_eig ** powers
    zbar_powers = np.conjugate(z_eig) ** powers
    abs_powers = np.abs(z_powers)
    half_sum = (np.add.outer(powers, powers) // 2).astype(np.int64)

    # Per-node kernel evaluation and overflow-safe rescaling (FINDINGS
    # F005).  The kernel is NOT batched -- its series length varies with
    # w -- so each node's derivatives are computed and rescaled here, then
    # stacked for the single batched contraction call.  At high
    # cancellation exponent ``L = w*|y'|`` the radial derivatives span a
    # huge dynamic range, so factor each node's peak magnitude out as an
    # EXACT power of two (``np.frexp`` / ``np.ldexp`` introduce no
    # rounding) before the contraction; the total is scaled back exactly
    # below.  This is NOT extended precision (FINDINGS F001).
    n_nodes = w_array.shape[0]
    derivs_scaled = np.empty((n_nodes, dim), dtype=complex)
    scale_exps = np.empty(n_nodes, dtype=np.int64)
    kernel_tails = np.empty(n_nodes, dtype=float)
    for node in range(n_nodes):
        w_node = float(w_array[node])
        n_terms = _series_length(w_node, s)
        derivs, relative_tail = point_mass_g_derivatives(
            w_node, s, 2 * max_order, n_terms)
        max_abs = float(np.max(np.abs(derivs)))
        _, scale_exp = np.frexp(max_abs)  # max_abs == frac * 2**scale_exp
        scale_exp = int(scale_exp)
        scale_exps[node] = scale_exp
        derivs_scaled[node] = (np.ldexp(derivs.real, -scale_exp)
                               + 1j * np.ldexp(derivs.imag, -scale_exp))
        kernel_tails[node] = float(np.max(relative_tail))

    # The w-independent weight-vector build and the order-accumulation
    # loop -- the length-dim weight-vector dot, its all-positive companion,
    # and the small-term convergence test -- run in the single fused njit
    # core `_fused_contraction` for all nodes at once.  Everything that
    # raises, thresholds, or reconstructs stays here in Python.
    (totals, positive_totals, max_terms, orders_used,
     last_ratios, converged) = _fused_contraction(
         table, z_powers, zbar_powers, abs_powers, half_sum,
         derivs_scaled, w_array, gamma_scaled, max_order, dim)

    values = np.empty(n_nodes, dtype=complex)
    estimated_tails = np.empty(n_nodes, dtype=float)
    cancellation_ratios = np.empty(n_nodes, dtype=float)
    for node in range(n_nodes):
        w_node = float(w_array[node])
        total = totals[node]

        # A non-finite running total means the scaled contraction still
        # overflowed; refuse instead of letting a ``nan`` slip past the
        # ratio gates below (``nan > threshold`` is False, so those gates
        # would NOT fire on it).  Primary closure of the silent-nan bug.
        if not (np.isfinite(total.real) and np.isfinite(total.imag)):
            raise CancellationError(_refusal_message(
                w_node, y, gamma, kappa,
                'the scaled operator contraction is non-finite '
                '(overflow)'))

        total_abs = max(abs(total), 1e-300)
        cancellation_ratio = max_terms[node] / total_abs
        if cancellation_ratio > _CANCELLATION_REFUSAL:
            raise CancellationError(_refusal_message(
                w_node, y, gamma, kappa,
                f'the two channels cancel to a measured ratio '
                f'max_partial_term / |total| = {cancellation_ratio:.3e}, '
                f'past the certified {_CANCELLATION_REFUSAL:.0e}'))

        # TRUNCATION certification (FINDINGS F005): the max of the operator
        # series' last-term ratio and the kernel's worst per-order tail.
        # The BINDING cut at small ``max_order`` where the shear series has
        # not converged; it goes blind (~1e-14) once it has, where the
        # round-off GUARD below is what certifies.
        estimated_tail = max(float(last_ratios[node]), kernel_tails[node])
        if estimated_tail > _CONTRACTION_TARGET:
            raise CancellationError(_refusal_message(
                w_node, y, gamma, kappa,
                f'the series truncation cannot certify the '
                f'{_CONTRACTION_TARGET:.0e} target: estimated relative '
                f'tail = {estimated_tail:.3e} at max_order = {max_order} '
                f'(converged = {bool(converged[node])})'))

        # Round-off CERTIFICATION for the CONTRACTION source: the first-
        # order float64 round-off ``eps * (sum|term| / |total|)``.  The
        # ONLY cut that sees the contraction blow-up near ``L ~ 45`` once
        # the shear series has converged (FINDINGS F005).  The scale-
        # invariant ratio is unperturbed by the power-of-two rescaling.
        contraction_condition = positive_totals[node] / total_abs
        contraction_error = (_CONTRACTION_UNIT_ROUNDOFF
                             * contraction_condition)
        if contraction_error > _CONTRACTION_GUARD:
            raise CancellationError(_refusal_message(
                w_node, y, gamma, kappa,
                f'the wave-branch contraction round-off guard tripped: '
                f'eps * (sum|term| / |total|) = {contraction_error:.3e} '
                f'(guard {_CONTRACTION_GUARD:.0e}) from a magnitude '
                f'spread sum|term| / |total| = '
                f'{contraction_condition:.3e}'))

        # Undo the power-of-two rescaling EXACTLY (``ldexp`` by the same
        # exponent), then reconstruct F from G and undo the mass-sheet
        # rescaling.  Because y_scaled = y / sqrt(lam), the physical
        # |y|**2 / lam is exactly s.
        scale_exp = int(scale_exps[node])
        total = complex(np.ldexp(total.real, scale_exp),
                        np.ldexp(total.imag, scale_exp))
        phase_scaled = np.exp(0.5j * w_node * s)
        mass_sheet_phase = np.exp(
            0.5j * w_node * np.log(lam) - 0.5j * w_node * float(kappa) * s)
        value = complex(mass_sheet_phase * phase_scaled * total / lam)
        if not (np.isfinite(value.real) and np.isfinite(value.imag)):
            raise CancellationError(_refusal_message(
                w_node, y, gamma, kappa,
                'the reconstructed amplification is non-finite '
                '(overflow)'))
        values[node] = value
        estimated_tails[node] = estimated_tail
        cancellation_ratios[node] = cancellation_ratio

    return (values, orders_used, converged, estimated_tails,
            cancellation_ratios)


#: Test-only oracle exposing the LEGACY positive-parity operator-series
#: contraction (the certified dd/1F1 wave path).  Since Build 8d the
#: production positive-parity evaluator is Schwinger (`_positive_parity_grid`
#: for ``gamma' > 0``); this alias lets the overlap-domain regression
#: harness import the legacy evaluator to certify Schwinger against it on
#: the certified overlap.  NOT a production path (see Build 8d).
legacy_operator_oracle = _grid_certified


def _positive_parity_grid(
        w_array: np.ndarray, y: np.ndarray, gamma: float, *,
        beta: float = 0.0, kappa: float = 0.0,
        max_order: int = MAX_ORDER
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                   np.ndarray]:
    """Positive-parity wave-branch grid: Schwinger for ``gamma' > 0``.

    The single positive-parity wave-branch entry point shared by
    `F_op_grid` and `F_op`.  Since Build 8d the exact evaluator is the
    same 1D Schwinger-parameter quadrature `_schwinger.f_schwinger` the
    macro-saddle arm (`_saddle_grid`) and the former Build-7a strong-
    shear rescue used (``lam = 1 - kappa``, IDENTICAL reduce / rotate /
    reconstruct), so BOTH parities now run through ONE exact wave
    evaluator; the legacy operator-series contraction is retired to the
    single shear-free exit below (and to test-only oracle duty via
    `legacy_operator_oracle`).

    Dispatch is on the reduced shear ``gamma' = gamma / (1 - kappa)``:

    * ``gamma' > 0`` (every sheared positive-parity host -- the whole
      sampled prior box): each node is evaluated by `f_schwinger` in the
      pure-shear eigenframe and reconstructed with the mass-sheet
      identity.  Before the named refusal fires for a
      ``w > _schwinger.W_CEILING_SCHWINGER`` node (the previously-
      refusing set), the uniform-asymptotic rung is offered first
      (`_uniform_arm_value`: fold Airy then cusp Pearcey); the first arm
      that certifies serves the node.  Only if BOTH arms refuse does the
      node raise `_schwinger.SchwingerCertificationError` -- the named
      refusal still stands; there is NO legacy fallback catch (that would
      re-introduce a parallel production path).  A ``w <= ceiling`` node
      never reaches the arm intercept, so it is byte-identical to the
      exact path.

    * ``gamma' == 0`` (the shear-free point lens; measure-zero in the
      prior but reachable in unit tests and by direct callers): the 1D
      Schwinger representation requires ``gamma' > 0``, so the legacy
      operator-series contraction `_grid_certified` is the SOLE serving
      route, with its `CancellationError` semantics unchanged.  This is
      the only remaining production path through the legacy contraction.

    The Schwinger nodes carry no operator-series diagnostics, so their
    ``orders`` / ``estimated_tails`` / ``cancellation_ratios`` are
    reported as zero and ``converged`` as ``True`` (mirroring the saddle
    arm); the ``gamma' == 0`` legacy nodes keep their measured
    diagnostics.

    NODE-PARALLEL EXACT EVALUATION (Build 8f lever 3).  On the
    ``gamma' > 0`` route a Python pre-pass gathers the independent
    ``w <= ceiling`` exact wave nodes and evaluates them across cores via
    `_schwinger_wave_grid_values` (the njit ``prange`` PURE MAP over the
    frozen `_schwinger._raw_t_integral_core`); each per-node value is
    byte-identical to the serial `f_schwinger` path, and the named
    refusal is reduced over the full node ordering and raised by the
    Python wrapper (never across a thread boundary) with the lowest-index
    refuser's authentic message.

    Parameters and returns match `_grid_certified`.  The caller
    guarantees positive parity (``1 - kappa > |gamma|``), so the
    mass-sheet map never refuses here.

    Raises
    ------
    _schwinger.SchwingerCertificationError
        If a ``gamma' > 0`` node cannot certify its paired-rule
        quadrature, or lies above ``_schwinger.W_CEILING_SCHWINGER``.
    CancellationError
        If a ``gamma' == 0`` node's legacy operator contraction cannot
        be certified to the ``1e-10`` target (the four F005 refusals).
    geometry.LensDomainError, _hyp1f1.HypergeometricDomainError
        Propagated from `_grid_certified` on the ``gamma' == 0`` route.
    """
    w_array = np.asarray(w_array, dtype=float)
    if w_array.ndim != 1:
        raise ValueError(
            f'w_array must be one-dimensional, got shape {w_array.shape}.')
    lam, y_scaled, gamma_prime = _mass_sheet_map(y, gamma, kappa)
    if not gamma_prime > 0.0:
        # Shear-free (gamma' = 0, pure point lens): the Schwinger 1D
        # representation requires gamma' > 0, so the legacy operator
        # contraction is the SOLE serving route (its named
        # CancellationError refusals are unchanged).  This is the only
        # remaining production exit through the legacy path (Build 8d).
        return _grid_certified(
            w_array, y, gamma, beta=beta, kappa=kappa, max_order=max_order)

    # gamma' > 0: the exact Schwinger evaluator, reconstructed with the
    # SAME mass-sheet identity the saddle arm and the former Build-7a
    # rescue use.  Reduce ONCE (w-independent) to the pure-shear
    # eigenframe, then evaluate node by node.
    z_eig = np.exp(-1j * float(beta)) * complex(y_scaled[0], y_scaled[1])
    y_eig = np.array([z_eig.real, z_eig.imag])
    s = float(y_scaled @ y_scaled)

    n_nodes = w_array.shape[0]
    values = np.empty(n_nodes, dtype=complex)

    # Python PRE-PASS over the nodes in index order (Build 8f lever 3):
    # gather the expensive ``w <= ceiling`` exact wave nodes for the
    # node-parallel batch; the arm-served / above-ceiling-refusing nodes
    # stay in Python.  A served node carries zero operator-series
    # diagnostics like the Schwinger nodes (the diagnostic arrays below are
    # uniformly zero / True).
    batch_index: list[int] = []
    ceiling_refusers: list[int] = []
    for node in range(n_nodes):
        w_node = float(w_array[node])
        if w_node > _schwinger.W_CEILING_SCHWINGER:
            # Previously-refusing node: offer the uniform-asymptotic rung
            # (fold then cusp arm) before the named refusal; only if BOTH
            # arms refuse does the node become a refuser (NO legacy
            # fallback catch -- that would re-introduce a parallel path).
            arm_value = _uniform_arm_value(
                w_node, y, gamma, beta=beta, kappa=kappa)
            if arm_value is not None:
                values[node] = arm_value
            else:
                ceiling_refusers.append(node)
        else:
            # w <= ceiling exact wave node: the parallel batch (byte-
            # identical to the serial `f_schwinger` path per node).
            batch_index.append(node)

    batch_index_arr = np.array(batch_index, dtype=np.int64)
    batch_values, batch_cert = _schwinger_wave_grid_values(
        w_array[batch_index_arr], y_eig, gamma_prime, lam, kappa, s)

    # Reduce the named refusal ACROSS THE FULL node ordering in the Python
    # wrapper (never across a thread boundary): any node refuses -> the
    # whole grid refuses, raised with the authentic message of the
    # LOWEST-index refuser (serial first-refuser identity).
    refusers = list(ceiling_refusers)
    for pos, node in enumerate(batch_index):
        if batch_cert[pos]:
            values[node] = batch_values[pos]
        else:
            refusers.append(node)
    if refusers:
        first = int(min(refusers))
        # Re-run the lowest-index refuser through `f_schwinger` to raise
        # the exact named exception (ceiling or paired-rule); identical
        # inputs -> identical decision and message as the serial path.
        _schwinger.f_schwinger(float(w_array[first]), y_eig, gamma_prime)
        raise _schwinger.SchwingerCertificationError(  # unreachable guard
            f'Node-parallel batch flagged node {first} '
            f'(w = {float(w_array[first])}) as refused, but the serial '
            f're-evaluation certified it; refusing rather than serving an '
            f'unverified value.')

    orders = np.zeros(n_nodes, dtype=int)
    converged = np.ones(n_nodes, dtype=bool)
    estimated_tails = np.zeros(n_nodes, dtype=float)
    cancellation_ratios = np.zeros(n_nodes, dtype=float)
    return (values, orders, converged, estimated_tails,
            cancellation_ratios)


def F_op_grid(w_array: np.ndarray, y: np.ndarray, gamma: float, *,
              beta: float = 0.0, kappa: float = 0.0,
              max_order: int = MAX_ORDER
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched contour-free Chang-Refsdal amplification over a ``w`` grid.

    Evaluates `F_op` at every frequency in ``w_array`` in one call.
    Since Build 8d a positive-parity host with ``gamma' > 0`` is served
    by the exact 1D Schwinger evaluator `_schwinger.f_schwinger` (the
    SAME reduce / rotate / reconstruct as the macro-saddle arm, with
    ``lam = 1 - kappa``), evaluated node by node; the legacy batched
    operator-series contraction serves ONLY the shear-free ``gamma' == 0``
    point lens (measure-zero in the prior).  This is the fast wave-branch
    entry point the lensed likelihood evaluates on its coarse kernel-node
    grid.

    The certified-or-named-refusal contract is unchanged and applied PER
    NODE: any frequency that cannot be certified to the ``1e-10`` target
    raises a named refusal, so a single uncertifiable node refuses the
    whole grid rather than returning a finite-but-uncertified value.  For
    ``gamma' > 0`` the refusal is `_schwinger.SchwingerCertificationError`
    (including every ``w > _schwinger.W_CEILING_SCHWINGER``); for
    ``gamma' == 0`` it is `CancellationError` (the four F005 refusals).

    Parameters
    ----------
    w_array : np.ndarray
        One-dimensional array of dimensionless frequencies, each
        ``0 < w <= _hyp1f1.W_MAX_CERTIFIED``.
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
    values : np.ndarray
        ``(n_nodes,)`` complex amplifications ``F``, one per frequency.
    orders : np.ndarray
        ``(n_nodes,)`` int highest operator order summed per node.
    converged : np.ndarray
        ``(n_nodes,)`` bool small-term-stop flag per node.

    Raises
    ------
    geometry.LensDomainError
        If ``1 - kappa <= abs(gamma)``.
    _schwinger.SchwingerCertificationError
        Positive-parity ``gamma' > 0`` route: if a node cannot certify
        its paired-rule Schwinger quadrature, or lies above
        ``_schwinger.W_CEILING_SCHWINGER`` (``w > 60``, where the named
        refusal stands).  This is the production refusal for every
        sheared positive-parity host (`_positive_parity_grid`).
    CancellationError
        Shear-free ``gamma' == 0`` legacy route only: if a node's
        operator contraction cannot be certified to the ``1e-10`` target
        (the four F005 refusals, per node).
    _hyp1f1.HypergeometricDomainError
        Propagated from the kernel above its certified ``w`` or
        cancellation-exponent ceiling.
    """
    lam = 1.0 - float(kappa)
    if not lam > abs(float(gamma)):
        # Negative-parity (macro saddle), Type III, or the parity
        # boundary: the operator power series does not converge, so route
        # to the exact Schwinger wave branch (which raises the named Type
        # III / parity-boundary / certification refusals).  Positive
        # parity (lam > |gamma|) is untouched and byte-identical below.
        values = _saddle_grid(w_array, y, gamma, beta=beta, kappa=kappa)
        orders = np.zeros(values.shape[0], dtype=int)
        converged = np.ones(values.shape[0], dtype=bool)
        return values, orders, converged
    values, orders, converged, _, _ = _positive_parity_grid(
        w_array, y, gamma, beta=beta, kappa=kappa, max_order=max_order)
    return values, orders, converged


def F_op(w: float, y: np.ndarray, gamma: float, *,
         beta: float = 0.0, kappa: float = 0.0,
         max_order: int = MAX_ORDER
         ) -> tuple[complex, OperatorDiagnostics]:
    """Contour-free Chang-Refsdal amplification at one frequency.

    A thin scalar wrapper over the shared grid path
    (`_positive_parity_grid` / `_saddle_grid`, a single-element grid
    call), so the scalar and batched entry points share exactly ONE
    evaluator per branch.  Dispatches on parity.  Since Build 8d BOTH
    parities are served by the exact 1D Schwinger wave evaluator
    `_schwinger.f_schwinger` on the wave branch: the positive-parity
    host (``1 - kappa > |gamma|``) with reduced shear ``gamma' > 0`` and
    the macro saddle (``0 < 1 - kappa < |gamma|``) reduce / rotate /
    reconstruct identically (``lam = 1 - kappa``).  The ONLY remaining
    legacy operator-series exit is the shear-free ``gamma' == 0`` point
    lens (measure-zero in the prior, reached via `_grid_certified`).  On
    the Schwinger branch the operator-series diagnostics do not apply and
    are reported as zero / converged (the Schwinger evaluator
    certifies-or-refuses internally, so a returned value is certified);
    on the ``gamma' == 0`` legacy branch the measured diagnostics stand.

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
        External convergence; enters through `_mass_sheet_map` (positive
        parity) or `_saddle_mass_sheet_map` (macro saddle).
    max_order : int, optional
        Operator-series order cap; also fixes the kernel ladder length
        ``2 * max_order``.  Ignored on the saddle branch.

    Returns
    -------
    value : complex
        The amplification ``F``.
    diagnostics : OperatorDiagnostics
        MEASURED convergence and cancellation report (positive parity).
        On the saddle branch the operator-series fields do not apply and
        are reported as zero with ``converged = True``.

    Raises
    ------
    geometry.LensDomainError
        If ``1 - kappa <= 0`` (Type III) or ``|gamma| == 1 - kappa`` (the
        parity boundary).
    CancellationError
        Shear-free ``gamma' == 0`` legacy route only (the point lens; see
        `_positive_parity_grid`): if the operator contraction cannot be
        certified to the ``1e-10`` target.  Every sheared positive-parity
        host (``gamma' > 0``) is served by Schwinger instead and refuses,
        if at all, with `_schwinger.SchwingerCertificationError`.  The
        underlying contraction covers four refusals, all raised as named
        errors rather than returning a ``nan`` or a finite-but-uncertified
        amplification (FINDINGS F005):

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
    _schwinger.SchwingerCertificationError
        The production wave-branch refusal on BOTH parities: if the
        paired Gauss-Legendre rules cannot certify the Schwinger
        quadrature -- on the saddle branch, or on any sheared
        positive-parity (``gamma' > 0``) node -- or a node (or an
        unresolved saddle) exceeds ``_schwinger.W_CEILING_SCHWINGER``
        (``w > 60``).
    _hyp1f1.HypergeometricDomainError
        Shear-free ``gamma' == 0`` legacy route: propagated from the
        kernel above its certified ``w`` ceiling or cancellation-exponent
        ceiling.

    Notes
    -----
    ``F`` is normalized to no lens at all, not to the macro image, so
    ``F(w -> 0) = 1/sqrt((1 - kappa)**2 - gamma**2) = sqrt(mu_macro)``,
    not 1.  The flat ``|F| - 1`` at tiny ``w`` is that exact limit and
    not a ``gamma/(2*w)`` prefactor singularity; see the module
    docstring before "fixing" it.
    """
    lam = 1.0 - float(kappa)
    if not lam > abs(float(gamma)):
        # Macro saddle (or Type III / parity boundary): the operator
        # series does not converge; the exact Schwinger wave branch
        # returns the value or raises a named refusal.  The
        # operator-series diagnostics do not apply here.
        values = _saddle_grid(
            np.asarray([float(w)], dtype=float), y, gamma,
            beta=beta, kappa=kappa)
        diagnostics = OperatorDiagnostics(
            order_used=0,
            converged=True,
            estimated_relative_tail=0.0,
            cancellation_ratio=0.0)
        return complex(values[0]), diagnostics
    (values, orders, converged, estimated_tails,
     cancellation_ratios) = _positive_parity_grid(
         np.asarray([float(w)], dtype=float), y, gamma,
         beta=beta, kappa=kappa, max_order=max_order)
    diagnostics = OperatorDiagnostics(
        order_used=int(orders[0]),
        converged=bool(converged[0]),
        estimated_relative_tail=float(estimated_tails[0]),
        cancellation_ratio=float(cancellation_ratios[0]))
    return complex(values[0]), diagnostics


def _certify_geometric_census(images: list, matrix: np.ndarray) -> None:
    """Refuse an image census that cannot license the geometric asymptote.

    The stationary-phase (``w -> inf``) sum in `geometric_amplification`
    is legitimate only on a RESOLVED, NON-DEGENERATE image census.  In
    production the resolution gate (`select_branch`) already routes every
    unresolved cluster to the wave branch, so for a valid served config
    BOTH guards below pass silently and the returned amplification is
    byte-identical to the ungated sum.  They fire only on an inconsistent
    census that must never reach geometric optics, and they do so through
    the EXISTING refusal vocabulary -- `geometry.LensDomainError` -- with
    no new exception type (Build 8f lever 5).

    Guard (a), IMAGE-COUNT MATCH against the caustic classification.  A
    non-degenerate Chang-Refsdal source has exactly TWO real images
    outside the caustic and FOUR inside it; the image quartic admits at
    most four real roots, and the Morse index theorem forces the count to
    be EVEN for both the positive parity and the macro saddle, so ``2``
    and ``4`` are the only valid served counts.  Any other count -- an odd
    count from a fold-merged ``(min, saddle)`` pair or a cusp-merged
    triple, or a dropped image -- means the source sits on or across a
    caustic: a degenerate census that belongs on the wave branch.

    Guard (b), MORSE PARITY-SUM.  The signed magnification sum obeys the
    Morse index theorem ``sum_a sign(mu_a) == sign(det A) - 1`` -- ``0``
    for the positive parity (``det A > 0``) and ``-2`` for the macro
    saddle (``det A < 0``).  ``sign(mu_a)`` is ``(-1)`` to the Morse index
    (the number of NEGATIVE Fermat-Hessian eigenvalues, from `eigvalsh`
    with a strict ``< 0`` test; see `geometry.morse_index`).  A violation
    means the quartic solve silently dropped or duplicated an image, so
    the census is unfaithful and the summed amplification would be finite
    but wrong.

    Parameters
    ----------
    images : list of np.ndarray
        The real images from `geometry.find_images`, each shape ``(2,)``,
        for `matrix`.
    matrix : np.ndarray
        Shape ``(2, 2)`` macro matrix the images were solved for.

    Raises
    ------
    geometry.LensDomainError
        If the census fails the image-count match or the Morse
        parity-sum guard.
    """
    image_count = len(images)
    if image_count not in (2, 4):
        raise geometry.LensDomainError(
            f'Geometric-optics census defect: {image_count} real images '
            f'for macro matrix {np.asarray(matrix).tolist()}, but a '
            f'non-degenerate Chang-Refsdal source has exactly 2 images '
            f'(outside the caustic) or 4 (inside). An odd or otherwise '
            f'anomalous count is a fold/cusp-merged or dropped census that '
            f'must be served on the wave branch, not by the stationary-'
            f'phase sum.')

    signed_magnification_sum = sum(
        (-1) ** geometry.morse_index(image, matrix) for image in images)
    expected_sum = (1 if float(np.linalg.det(matrix)) > 0.0 else -1) - 1
    if signed_magnification_sum != expected_sum:
        raise geometry.LensDomainError(
            f'Geometric-optics census defect: the {image_count} images '
            f'give a signed magnification sum sum_a sign(mu_a) = '
            f'{signed_magnification_sum}, but the Morse index theorem '
            f'requires sum_a sign(mu_a) == sign(det A) - 1 = {expected_sum} '
            f'for macro matrix {np.asarray(matrix).tolist()}. A mismatch '
            f'means the quartic solve dropped or duplicated an image, so '
            f'the census is unfaithful and the summed amplification would '
            f'be finite but wrong.')


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
        If ``1 - kappa <= abs(gamma)`` (from `geometry.macro_matrix`),
        or if the solved image census fails the geometric-served handoff
        guards -- image-count match (2 outside the caustic, 4 inside) or
        the Morse parity-sum (see `_certify_geometric_census`).
    """
    source = np.asarray(y, dtype=float)
    if source.shape != (2,):
        raise ValueError(
            f'Source position must have shape (2,), got {source.shape}.')
    matrix = geometry.macro_matrix(gamma, beta, kappa)
    images = geometry.find_images(source, matrix)
    # Build 8f lever 5: refuse an inconsistent census (image count vs the
    # quartic solve, and the Morse parity-sum) before summing.  Value-
    # preserving -- a resolved, non-degenerate served census passes
    # silently and ``total`` below is byte-identical to the ungated sum.
    _certify_geometric_census(images, matrix)
    total = np.zeros_like(np.asarray(w, dtype=float), dtype=complex)
    for image in images:
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

    Notes
    -----
    ``L_MAX`` is a HANDOFF exponent inside the certified wave/geometric
    overlap, not a one-sided accuracy floor (see the `L_MAX` provenance):
    the wave series is accurate to ``L ~ 45-46`` (F005) and the geometric
    asymptote above its ``~50`` onset at resolved clusters (F013), so the
    shipped ``48`` sits in the overlap and the refusal band ``[46, 48]``
    exits by named `CancellationError`.  The geometric-served path itself
    additionally enforces the census guards of
    `_certify_geometric_census` before summing.
    """
    resolved = float(w) * float(delta_min) >= RHO_END
    strongly_cancelling = float(cancellation_exp) > L_MAX
    if resolved and strongly_cancelling:
        return 'geometric'
    return 'wave'
