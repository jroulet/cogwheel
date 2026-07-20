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
``|gamma|``.  Positive parity ``lam > |gamma|`` takes the operator /
1F1 wave branch above unchanged and BYTE-IDENTICAL (every F001/F005
refusal constant is frozen).  A macro saddle ``0 < lam < |gamma|`` has
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

from dataclasses import dataclass

import numba
import numpy as np

from cogwheel.lensing.chang_refsdal import geometry, _schwinger
from cogwheel.lensing.chang_refsdal._hyp1f1 import (
    point_mass_g_derivatives)

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
    ceiling propagates `_schwinger.SchwingerCertificationError` from the
    wave branch rather than returning a wrong value.

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

    values = np.empty(w_array.shape[0], dtype=complex)
    for node in range(w_array.shape[0]):
        w_node = float(w_array[node])
        # NOTE: `select_branch` is NOT the saddle authority in Build S1 --
        # it stays byte-frozen for the positive-parity operator path; the
        # saddle takeover is owned here (channels.py / Build 7 wires it).
        if (w_node > _schwinger.W_CEILING_SCHWINGER
                and w_node * delta_min >= RHO_END):
            # Resolved and above the wave ceiling: stationary-phase sum
            # over the real images of the indefinite matrix.
            values[node] = complex(geometric_amplification(
                w_node, y, gamma, beta=beta, kappa=kappa))
            continue
        # Wave branch.  An unresolved node with w > ceiling reaches here
        # too and `f_schwinger` refuses it (SchwingerCertificationError).
        f_pure = _schwinger.f_schwinger(w_node, y_eig, gamma_prime)
        mass_sheet_phase = np.exp(
            0.5j * w_node * np.log(lam) - 0.5j * w_node * float(kappa) * s)
        values[node] = complex(mass_sheet_phase * f_pure / lam)
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


def _positive_parity_grid_with_fallback(
        w_array: np.ndarray, y: np.ndarray, gamma: float, *,
        beta: float = 0.0, kappa: float = 0.0,
        max_order: int = MAX_ORDER
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                   np.ndarray]:
    """Positive-parity grid with a strong-shear Schwinger fallback.

    The single positive-parity wave-branch entry point shared by
    `F_op_grid` and `F_op`.  It FIRST evaluates the whole grid through
    the byte-frozen operator contraction `_grid_certified`; if that
    returns, its full five-tuple is handed back UNCHANGED, so the common
    all-certified case incurs no per-node loop and stays bit-for-bit
    identical to the legacy path.

    ONLY when `_grid_certified` raises `CancellationError` (strong shear,
    near the operator's cancellation band) does it fall to a per-node
    loop.  Each node is retried on its own single-element grid; a node
    that still refuses is routed, if ``w <= W_CEILING_SCHWINGER``, through
    the exact 1D Schwinger evaluator in the positive-parity mass-sheet-
    reduced eigenframe and reconstructed with the SAME mass-sheet
    identity the operator path uses -- converting an uncertifiable
    operator-series refusal into a certified answer without touching any
    threshold.  A refusing node with ``w > W_CEILING_SCHWINGER`` re-raises
    the original `CancellationError` (the named refusal stands), and a
    `_schwinger.SchwingerCertificationError` from the fallback propagates.

    The fallback nodes carry no operator-series diagnostics, so their
    ``orders``/``estimated_tails``/``cancellation_ratios`` are reported as
    zero and ``converged`` as ``True`` (mirroring the saddle-arm
    convention); certified nodes keep their measured diagnostics.

    Parameters and returns match `_grid_certified`.  The caller
    guarantees positive parity (``1 - kappa > |gamma|``), so the
    mass-sheet map never refuses here.

    Raises
    ------
    CancellationError
        For a refusing node with ``w > W_CEILING_SCHWINGER`` (the named
        refusal above the Schwinger ceiling stands).
    _schwinger.SchwingerCertificationError
        If the Schwinger fallback cannot certify a sub-ceiling node.
    geometry.LensDomainError, _hyp1f1.HypergeometricDomainError
        Propagated from `_grid_certified` as before.
    """
    w_array = np.asarray(w_array, dtype=float)
    try:
        return _grid_certified(
            w_array, y, gamma, beta=beta, kappa=kappa, max_order=max_order)
    except CancellationError as exc:
        original_refusal = exc

    # At least one node's operator contraction refused.  Reduce ONCE
    # (w-independent) to the positive-parity pure-shear eigenframe, then
    # re-evaluate node by node, substituting the Schwinger value wherever
    # the operator path still refuses below the ceiling.
    lam, y_scaled, gamma_prime = _mass_sheet_map(y, gamma, kappa)
    if not gamma_prime > 0.0:
        # A shear-free (gamma' = 0, pure point-lens) refusal has no
        # Schwinger arm: the 1D representation requires gamma' > 0, and
        # calling it would replace the NAMED CancellationError with a
        # raw ValueError (found in production by the channel-layer
        # config sweep and the prior-box smoke: legacy tail refusals at
        # gamma = 0 exist).  The original named refusal stands.
        raise original_refusal
    z_eig = np.exp(-1j * float(beta)) * complex(y_scaled[0], y_scaled[1])
    y_eig = np.array([z_eig.real, z_eig.imag])
    s = float(y_scaled @ y_scaled)

    n_nodes = w_array.shape[0]
    values = np.empty(n_nodes, dtype=complex)
    orders = np.zeros(n_nodes, dtype=int)
    converged = np.ones(n_nodes, dtype=bool)
    estimated_tails = np.zeros(n_nodes, dtype=float)
    cancellation_ratios = np.zeros(n_nodes, dtype=float)
    for node in range(n_nodes):
        w_node = float(w_array[node])
        try:
            (node_values, node_orders, node_converged, node_tails,
             node_ratios) = _grid_certified(
                 w_array[node:node + 1], y, gamma,
                 beta=beta, kappa=kappa, max_order=max_order)
        except CancellationError:
            if w_node > _schwinger.W_CEILING_SCHWINGER:
                # Above the Schwinger ceiling the named refusal stands.
                raise
            # Positive-parity strong-shear fallback: the exact Schwinger
            # amplification, reconstructed with the mass-sheet identity.
            # The prefactor is a single float64 exp/mul/div OUTSIDE the
            # paired-rule certificate and bounded by inspection (FINDINGS
            # F011); `f_schwinger` certifies-or-refuses its own quadrature.
            f_pure = _schwinger.f_schwinger(w_node, y_eig, gamma_prime)
            mass_sheet_phase = np.exp(
                0.5j * w_node * np.log(lam)
                - 0.5j * w_node * float(kappa) * s)
            values[node] = complex(mass_sheet_phase * f_pure / lam)
            continue
        # Certified node: keep the operator path's value and diagnostics
        # bit-for-bit.
        values[node] = node_values[0]
        orders[node] = int(node_orders[0])
        converged[node] = bool(node_converged[0])
        estimated_tails[node] = float(node_tails[0])
        cancellation_ratios[node] = float(node_ratios[0])
    return (values, orders, converged, estimated_tails,
            cancellation_ratios)


def F_op_grid(w_array: np.ndarray, y: np.ndarray, gamma: float, *,
              beta: float = 0.0, kappa: float = 0.0,
              max_order: int = MAX_ORDER
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched contour-free Chang-Refsdal amplification over a ``w`` grid.

    Evaluates `F_op` at every frequency in ``w_array`` in one call.
    Within one lens configuration only ``w`` varies, so the operator
    table and the per-order weight vectors are built ONCE and reused
    across every node, and the operator-series contraction runs as a
    single batched njit sweep instead of one bilinear form per node --
    the fast wave-branch entry point the lensed likelihood evaluates on
    its coarse kernel-node grid.

    The certified-or-named-refusal contract is unchanged and applied PER
    NODE: any frequency that cannot be certified to the ``1e-10`` target
    raises `CancellationError` (FINDINGS F005), so a single uncertifiable
    node refuses the whole grid rather than returning a
    finite-but-uncertified value.

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
    CancellationError
        If a node's operator contraction cannot be certified to the
        ``1e-10`` target (the four F005 refusals, per node) AND its
        strong-shear Schwinger fallback cannot rescue it -- i.e. the
        node lies above ``_schwinger.W_CEILING_SCHWINGER`` (``w > 60``),
        where the named refusal stands.  Sub-ceiling positive-parity
        refusals are instead converted to certified answers by the exact
        1D Schwinger evaluator (`_positive_parity_grid_with_fallback`).
    _schwinger.SchwingerCertificationError
        If a strong-shear Schwinger fallback node cannot certify its own
        paired-rule quadrature.
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
    values, orders, converged, _, _ = _positive_parity_grid_with_fallback(
        w_array, y, gamma, beta=beta, kappa=kappa, max_order=max_order)
    return values, orders, converged


def F_op(w: float, y: np.ndarray, gamma: float, *,
         beta: float = 0.0, kappa: float = 0.0,
         max_order: int = MAX_ORDER
         ) -> tuple[complex, OperatorDiagnostics]:
    """Contour-free Chang-Refsdal amplification at one frequency.

    Dispatches on parity.  For a positive-parity host
    (``1 - kappa > |gamma|``) this is a thin scalar wrapper over
    `F_op_grid`'s shared contraction path (`_grid_certified`, a
    single-element grid call), so the scalar and batched entry points
    share exactly ONE contraction implementation and ONE certification
    (FINDINGS F005): the value and the diagnostics can never disagree.
    For a macro saddle (``0 < 1 - kappa < |gamma|``) the operator series
    does not converge and the value comes from the exact Schwinger wave
    branch `_saddle_grid` instead; the operator-series diagnostics fields
    are then not applicable and reported as zero / converged (the
    Schwinger evaluator certifies-or-refuses internally, so a returned
    value is certified).

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
        Positive-parity wave branch only: if the contraction cannot be
        certified to the ``1e-10`` target AND the strong-shear Schwinger
        fallback cannot rescue it (``w > _schwinger.W_CEILING_SCHWINGER``,
        i.e. ``w > 60``, where the named refusal stands).  A sub-ceiling
        positive-parity refusal is instead converted to a certified
        answer by the exact 1D Schwinger evaluator (see
        `_positive_parity_grid_with_fallback`).  The underlying
        contraction covers four refusals, all raised as named errors
        rather than returning a ``nan`` or a finite-but-uncertified
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
        If the paired Gauss-Legendre rules cannot certify the Schwinger
        quadrature -- on the saddle branch, or on a positive-parity
        strong-shear fallback node -- or an unresolved saddle exceeds
        ``_schwinger.W_CEILING_SCHWINGER`` (``w > 60``).
    _hyp1f1.HypergeometricDomainError
        Positive parity: propagated from the kernel above its certified
        ``w`` ceiling or cancellation-exponent ceiling.

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
     cancellation_ratios) = _positive_parity_grid_with_fallback(
         np.asarray([float(w)], dtype=float), y, gamma,
         beta=beta, kappa=kappa, max_order=max_order)
    diagnostics = OperatorDiagnostics(
        order_used=int(orders[0]),
        converged=bool(converged[0]),
        estimated_relative_tail=float(estimated_tails[0]),
        cancellation_ratio=float(cancellation_ratios[0]))
    return complex(values[0]), diagnostics


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
