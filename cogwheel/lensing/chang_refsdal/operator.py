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
contraction out to ``L ~ 45``.  Every uncertifiable input
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
    'RHO_START', 'RHO_END', 'L_MAX',
    'OperatorDiagnostics',
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
#: ``[46, 48]`` exits by a named refusal
#: (`_schwinger.SchwingerCertificationError`).  ``50`` is the ceiling
#: of any defensible raise, gated by the enforcement bracket (the
#: Test-Developer's graduated audit test).  Raising L_MAX past ~48 would
#: push previously-geometric-served nodes onto the wave path past its
#: 1e-10 accuracy ceiling (~L45-46), where they refuse -- so it stays 48.
L_MAX = 48

#: Minimum distance from the source to the caustic for the geometric
#: asymptote to be admitted, in Einstein-radius units.  Third leg of
#: `select_branch`; `L_MAX` cannot substitute for it, because below this
#: the error is flat in ``L`` -- near a fold the annihilated pair are
#: undamped complex saddles a real-image sum omits.  Refusal-increasing by
#: design.  Measured p90: 1.17 below 0.1, 7.65e-5 above 0.3 (F029, F031).
ETA_MIN_GEOMETRIC = 0.3

#: First-order float64 round-off unit for the operator CONTRACTION
#: (machine epsilon).  The contraction stays in complex128 -- the
#: double-double substrate lives only in the 1F1 kernel, never here
#: (FINDINGS F001) -- so its accuracy is bounded by this epsilon times
#: the measured cancellation condition ``sum|term| / |total|``.
_CONTRACTION_UNIT_ROUNDOFF = float(np.finfo(np.float64).eps)

#: Relative-accuracy target the wave branch must certify (FINDINGS
#: F005): the 1e-10 bar every returned amplification is held to, above
#: which the evaluator refuses by name rather than returning a
#: finite-but-uncertified value.  The operator-series contraction that
#: once measured itself against this target is retired; the target
#: survives as the shared accuracy bar the certified paths quote.
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
        before reaching its order cap.
    estimated_relative_tail : float
        MEASURED truncation estimate: the larger of the operator
        series' last-term ratio and the kernel's worst per-order
        relative tail.  Correctness rests on this measurement, not on
        any order heuristic being tight.
    cancellation_ratio : float
        MEASURED ``max_partial_term / |total|`` over the operator
        summation.  Reported as zero on every current serving route --
        the operator-series contraction that produced it is retired --
        and kept for the test-only legacy oracle.
    """

    order_used: int
    converged: bool
    estimated_relative_tail: float
    cancellation_ratio: float


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

    Per node the geometric-vs-wave decision above the ceiling is routed
    through the authoritative `select_branch`, called with an INFINITE
    cancellation exponent (``math.inf``).  That makes its
    ``strongly_cancelling`` leg vacuously true so ONLY the resolution
    leg is live: ``select_branch(w, delta_min, inf) == 'geometric'``
    holds iff ``w*delta_min >= RHO_END``.  This PRESERVES the historical
    saddle boundary ``(resolved AND w > W_CEILING_SCHWINGER) ->
    geometric`` EXACTLY -- the routing is behaviour-preserving and the
    boundary did not move.  Whether the saddle ADDITIONALLY needs a
    geometric-onset gate (the positive-parity ``L > L_MAX`` accuracy leg
    that `select_branch` normally enforces) is an OPEN, UNMEASURED
    question: every driver sweep behind F028 and the gate comparison was
    positive-parity only, so there is no saddle data on that leg.  Note
    that ceiling exhaustion above ``w = 60`` explains only wave
    UNAVAILABILITY, not geometric accuracy, so it does not settle the
    accuracy question.  ``resolved`` uses the frequency-independent
    real-image ``delta_min`` (computed once, only when some node exceeds
    the ceiling).  Because `_schwinger.f_schwinger` ALSO hard-refuses
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

    # Resolution and caustic distance are both frequency-independent;
    # compute each once and only if some node could take the geometric
    # branch (w > ceiling).  A refusing caustic search gives eta = 0.0 ->
    # 'wave', the conservative direction.
    delta_min = 0.0
    eta = 0.0
    if np.any(w_array > _schwinger.W_CEILING_SCHWINGER):
        delta_min = _real_delay_min_separation(source, matrix)
        try:
            eta = float(geometry.nearest_caustic_point(
                gamma, beta, source, kappa=kappa).distance)
        except geometry.LensDomainError:
            eta = 0.0

    n_nodes = w_array.shape[0]
    values = np.empty(n_nodes, dtype=complex)

    # Python PRE-PASS over the nodes in index order (Build 8f lever 3):
    # classify each into its serving branch and GATHER the expensive
    # ``w <= ceiling`` exact wave nodes for the node-parallel batch.  The
    # geometric and arm branches stay in Python; only the pure Schwinger
    # inner map is parallelized.  The geometric-vs-wave choice above the
    # ceiling is routed through the authoritative `select_branch` with an
    # infinite cancellation exponent, so ONLY its resolution leg is live
    # and the historical ``w > 60 AND resolved`` saddle boundary is
    # preserved exactly (see the function docstring).
    batch_index: list[int] = []
    ceiling_refusers: list[int] = []
    for node in range(n_nodes):
        w_node = float(w_array[node])
        if w_node > _schwinger.W_CEILING_SCHWINGER:
            # Above the wave ceiling.  The cancellation exponent is
            # positive-parity bookkeeping and has no saddle analogue, so
            # `inf` leaves that leg vacuously true; the resolution and
            # `eta` legs are live.  The `eta` leg is NOT inherited from
            # positive parity -- it is measured on the saddle (F034).
            if select_branch(w_node, delta_min, math.inf,
                             eta) == 'geometric':
                # Resolved and above the wave ceiling: stationary-phase sum
                # over the real images of the indefinite matrix.
                values[node] = complex(geometric_amplification(
                    w_node, y, gamma, beta=beta, kappa=kappa))
            else:
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


def _positive_parity_grid(
        w_array: np.ndarray, y: np.ndarray, gamma: float, *,
        beta: float = 0.0, kappa: float = 0.0
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
      identity.  A ``w > _schwinger.W_CEILING_SCHWINGER`` node (the set
      the exact wave evaluator refuses) is routed by the AUTHORITATIVE
      gate `select_branch` -- the SAME predicate `channels._exact_total`
      and `_saddle_grid` use, so the wave/geometric decision has one home:

      - `select_branch` returns ``'geometric'`` (resolved AND strongly
        cancelling): the node is served by `geometric_amplification`, the
        stationary-phase asymptote.  F028 measured the uniform fold arm at
        60%-267% relative error on exactly these well-resolved,
        strongly-cancelling configs, which the geometric serve replaces.
        This is the BEST AVAILABLE serve under the authoritative gate,
        with a measured ~1% O(1) residual tail (driver sweep,
        2026-07-28); it is NOT certified or exact.
      - `select_branch` returns ``'wave'``: the uniform-asymptotic rung is
        offered (`_uniform_arm_value`: fold Airy then cusp Pearcey); the
        first arm that certifies serves the node.  Only if BOTH arms
        refuse does the node raise `_schwinger.SchwingerCertificationError`
        -- the named refusal still stands; there is NO legacy fallback
        catch (that would re-introduce a parallel production path).

      A ``w <= ceiling`` node never reaches the ceiling classifier, so it
      is byte-identical to the exact path.

    * ``gamma' == 0`` (the shear-free point lens; measure-zero in the
      prior but reachable in unit tests and by direct callers): the 1D
      Schwinger representation requires ``gamma' > 0``, so this route is
      served by the CLOSED FORM below -- at ``gamma' == 0`` the shear
      operator is the identity and the series collapses to the
      point-mass kernel's zeroth term.  Its only refusal is the kernel's
      own `_hyp1f1.HypergeometricDomainError`.

    The Schwinger nodes carry no operator-series diagnostics, so their
    ``orders`` / ``estimated_tails`` / ``cancellation_ratios`` are
    reported as zero and ``converged`` as ``True`` (mirroring the saddle
    arm); the ``gamma' == 0`` closed form likewise reports no order and
    no cancellation ratio, but does report the kernel's MEASURED
    truncation tail.

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
    geometry.LensDomainError, _hyp1f1.HypergeometricDomainError
        Propagated from the point-mass kernel on the ``gamma' == 0``
        closed-form route.
    """
    w_array = np.asarray(w_array, dtype=float)
    if w_array.ndim != 1:
        raise ValueError(
            f'w_array must be one-dimensional, got shape {w_array.shape}.')
    lam, y_scaled, gamma_prime = _mass_sheet_map(y, gamma, kappa)
    if not gamma_prime > 0.0:
        # Shear-free (gamma' = 0, pure point lens) closed form.  The
        # Schwinger 1D representation requires gamma' > 0; at gamma' = 0 the
        # shear operator is the identity, so the amplification is just the
        # point-mass kernel times the mass-sheet prefactor.  `beta` is
        # correctly absent: with no shear there is no preferred axis.
        # `HypergeometricDomainError` propagates from the kernel unchanged.
        source_scaled = np.asarray(y, dtype=float) / np.sqrt(lam)
        s_shear_free = float(source_scaled @ source_scaled)
        n_nodes = w_array.shape[0]
        values = np.empty(n_nodes, dtype=complex)
        tails = np.zeros(n_nodes, dtype=float)
        for node in range(n_nodes):
            w_node = float(w_array[node])
            kernel, kernel_tail = point_mass_g_derivatives(
                w_node, s_shear_free, 0,
                _series_length(w_node, s_shear_free))
            # Two separate exponentials, then `* total / lam`.  Folding the
            # phase terms into one `exp` is mathematically identical and
            # 1 ULP different in float64; the byte-identity tests hold this
            # association.
            phase_scaled = np.exp(0.5j * w_node * s_shear_free)
            mass_sheet_phase = np.exp(
                0.5j * w_node * np.log(lam)
                - 0.5j * w_node * float(kappa) * s_shear_free)
            values[node] = complex(
                mass_sheet_phase * phase_scaled * complex(kernel[0]) / lam)
            tails[node] = float(kernel_tail[0])
        # Diagnostics follow the Schwinger convention (no operator series
        # ran, so no order and no cancellation ratio), except that the
        # kernel's MEASURED truncation tail is reported rather than zero.
        return (values,
                np.zeros(n_nodes, dtype=int),
                np.ones(n_nodes, dtype=bool),
                tails,
                np.zeros(n_nodes, dtype=float))

    # gamma' > 0: the exact Schwinger evaluator, reconstructed with the
    # SAME mass-sheet identity the saddle arm and the former Build-7a
    # rescue use.  Reduce ONCE (w-independent) to the pure-shear
    # eigenframe, then evaluate node by node.
    z_eig = np.exp(-1j * float(beta)) * complex(y_scaled[0], y_scaled[1])
    y_eig = np.array([z_eig.real, z_eig.imag])
    s = float(y_scaled @ y_scaled)

    # Geometric-vs-wave routing constants for the above-ceiling nodes
    # (F028), computed once per grid call.  `y_prime_norm = |y'|` is the
    # frequency-independent part of the cancellation exponent
    # (`cancellation_exponent(w, y, gamma, kappa) == w * |y'|` with
    # `|y'| = sqrt(y_scaled @ y_scaled)`), so `w_node * y_prime_norm`
    # reproduces ``L`` exactly WITHOUT re-running `_mass_sheet_map` per
    # node.  `delta_min` is the frequency-independent real-image
    # resolution measure; `_real_delay_min_separation` solves the image
    # quartic, so -- exactly as `_saddle_grid` guards its own delta_min --
    # both the macro matrix and the quartic solve are skipped entirely
    # when no node exceeds the ceiling.  FRAME DISCIPLINE: the geometric
    # gate feeds the PHYSICAL source / matrix, never the eigenframe
    # ``y_eig`` (`geometric_amplification` rebuilds `macro_matrix`
    # internally from the physical ``y`` / ``beta``).
    source = np.asarray(y, dtype=float)
    y_prime_norm = float(np.sqrt(y_scaled @ y_scaled))
    delta_min = 0.0
    # `eta` (distance to the caustic) is the third leg of the authoritative
    # gate and, like `delta_min`, is w-INDEPENDENT -- so it is computed once
    # per grid call and only when some node exceeds the ceiling.  A refusing
    # caustic search means no geometric admission: `eta = 0.0` sends every
    # node to 'wave', which is the conservative direction.
    eta = 0.0
    if np.any(w_array > _schwinger.W_CEILING_SCHWINGER):
        matrix = geometry.macro_matrix(gamma, beta, kappa)
        delta_min = _real_delay_min_separation(source, matrix)
        try:
            eta = float(geometry.nearest_caustic_point(
                gamma, beta, source, kappa=kappa).distance)
        except geometry.LensDomainError:
            eta = 0.0

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
            # Above the wave ceiling: the AUTHORITATIVE gate decides
            # geometric vs wave, so the predicate has ONE home shared with
            # `channels._exact_total` and `_saddle_grid` (F028).  ``L`` is
            # reconstructed as ``w_node * |y'|`` from the cached
            # w-independent norm; `select_branch` returns 'geometric' only
            # when the node is BOTH resolved (``w * delta_min >= RHO_END``)
            # and strongly cancelling (``L > L_MAX``).
            branch = select_branch(
                w_node, delta_min, w_node * y_prime_norm, eta)
            if branch == 'geometric':
                # Resolved and strongly cancelling: served by the
                # stationary-phase asymptote instead of the uniform fold
                # arm, which F028 measured at 60%-267% relative error on
                # exactly these well-resolved configs.  This is the best
                # available serve under the authoritative gate, with a
                # measured ~1% O(1) tail (driver sweep, 2026-07-28) -- it
                # is NOT certified or exact.
                values[node] = complex(geometric_amplification(
                    w_node, y, gamma, beta=beta, kappa=kappa))
            else:
                # Gate says 'wave': offer the uniform-asymptotic rung
                # (fold then cusp arm) before the named refusal; only if
                # BOTH arms refuse does the node become a refuser (NO
                # legacy fallback catch -- that would re-introduce a
                # parallel production path).
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
              beta: float = 0.0, kappa: float = 0.0
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
    (including every ``w > _schwinger.W_CEILING_SCHWINGER``); the
    shear-free ``gamma' == 0`` closed form refuses only through the
    kernel's own `_hyp1f1.HypergeometricDomainError`.

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
        w_array, y, gamma, beta=beta, kappa=kappa)
    return values, orders, converged


def F_op(w: float, y: np.ndarray, gamma: float, *,
         beta: float = 0.0, kappa: float = 0.0
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
         beta=beta, kappa=kappa)
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
                  cancellation_exp: float,
                  eta: float = np.inf) -> str:
    """Authoritative wave/geometric branch gate.

    THIS module owns the single implementation; the channel tracker
    imports it rather than redefining the thresholds.  Returns
    ``'geometric'`` only when ALL THREE conditions hold -- resolution,
    cancellation, and distance from the caustic -- and ``'wave'``
    otherwise.  No condition alone licenses the asymptote.

    Parameters
    ----------
    w : float
        Dimensionless frequency.
    delta_min : float
        Smallest pairwise Fermat-delay separation among the channels.
    cancellation_exp : float
        Measured cancellation exponent ``L`` (see
        `cancellation_exponent`).
    eta : float, optional
        Distance from the source to the caustic
        (`geometry.nearest_caustic_point`).  Defaults to ``inf``, which
        satisfies the leg vacuously and reproduces the two-condition
        gate.  A caller that omits it is DISABLING a measured accuracy
        condition; positive parity must supply it.  The macro saddle
        passes ``inf`` deliberately -- see the Notes.

    Returns
    -------
    str
        ``'geometric'`` if ``w*delta_min >= RHO_END`` and
        ``cancellation_exp > L_MAX`` and ``eta >= ETA_MIN_GEOMETRIC``;
        otherwise ``'wave'``.

    Notes
    -----
    ``L_MAX`` is a geometric-optics ONSET threshold, re-derived on its own
    terms in F031 after the legacy operator series it was originally
    calibrated against was retired: at FIXED ``eta``, the geometric error
    falls monotonically with ``L``, 100x to 280x across the range, measured
    against the Schwinger quadrature.

    The ``eta`` leg exists because ``L`` alone is NOT sufficient, also
    measured in F031: at ``eta < 0.1`` the error is FLAT in ``L`` and the
    two-condition gate still admitted nodes at p90 = 1.17 -- 117% relative
    error.  No amount of ``L`` rescues the near-caustic regime, because
    geometric optics has no validity there (F029: just outside a fold the
    annihilated image pair are undamped complex saddles that a real-image
    sum omits).  Adding this leg moves worst-case p90 from 1.17 to 7.65e-5.

    SADDLE: F031 is POSITIVE PARITY ONLY -- there is no saddle sweep. The
    macro-saddle path therefore passes ``eta = inf``, preserving its
    boundary exactly rather than extrapolating a positive-parity threshold
    onto an unmeasured branch.  Whether the saddle needs its own ``eta``
    floor is OPEN.
    """
    resolved = float(w) * float(delta_min) >= RHO_END
    strongly_cancelling = float(cancellation_exp) > L_MAX
    far_from_caustic = float(eta) >= ETA_MIN_GEOMETRIC
    if resolved and strongly_cancelling and far_from_caustic:
        return 'geometric'
    return 'wave'
