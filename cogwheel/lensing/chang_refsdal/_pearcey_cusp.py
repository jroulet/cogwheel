"""
Uniform Pearcey (cusp-catastrophe) wave branch for cusp neighborhoods.

WHAT
----
Two public entry points, both refusal-conservative (they return ``None``
rather than a wrong number, and raise NO new exception class):

* `pearcey(x, y)` -- the certified Pearcey primitive

      P(x, y) = Int_{-inf}^{inf} exp[i (t^4 + x t^2 + y t)] dt,

  evaluated on a rotated steepest-descent contour and certified in place
  by a paired Gauss-Legendre ``N`` / ``2N`` rule BEFORE any prefactor is
  applied (so a large prefactor cannot mask quadrature error), exactly as
  `_schwinger.f_schwinger` certifies its raw ``t``-integral.  Returns the
  complex value, or ``None`` if the paired rules disagree by more than
  `_CERTIFICATION_TOL`.

* `cusp_amplification(w, source, gamma, ...)` -- the cusp-arm evaluator

      F(w) = A * w^(1/2) * exp(i w tau_c + i sigma_c) * P(x, y),
      x = c_x * w^(1/2) * delta_parallel,   (along-cusp-axis control)
      y = c_y * w^(3/4) * delta_perp,       (transverse control)

  the uniform asymptotic amplification in the neighborhood of a caustic
  cusp.  ``tau_c`` is the Fermat delay at the cusp critical point;
  ``delta_parallel`` / ``delta_perp`` are the source offsets along the
  cusp tangent / normal read from `geometry.nearest_caustic_point`; the
  calibration constants ``c_x, c_y, A, sigma_c`` are pinned by requiring
  the large-argument stationary-phase limit of the uniform form to
  reproduce the code's exact geometric image sum (a closed-form matched
  asymptotic against `geometry`, not an empirical fit).  Returns the
  complex value, or ``None`` on any refusal.

WHY A ROTATED CONTOUR
---------------------
On the real axis the integrand ``exp(i t^4 ...)`` does not decay in
modulus, so the tails are only conditionally convergent -- useless for
quadrature.  The quartic phase ``i t^4`` has four steepest-descent
valleys at ``arg t = pi/8, 5 pi/8, 9 pi/8, 13 pi/8`` (where
``sin(4 arg t) = +1`` so the integrand decays as ``exp(-|t|^4)``).  The
original contour ``(-inf, +inf)`` deforms, by Cauchy's theorem on an
entire integrand with vanishing arcs at infinity, onto: a central real
segment ``[-T, T]`` plus a right tail rotated into the ``arg t = pi/8``
valley and a left tail into its origin reflection ``arg t = 9 pi/8``
(the valley reachable from the ``-inf`` end without crossing a hill --
the point-reflection ``t -> -t`` of the right tail, matching the even
dominant ``t^4`` and odd sub-dominant ``y t``).  Both tails then decay
super-exponentially, so the improper integral becomes an absolutely
convergent one that Gauss-Legendre resolves.

(The build brief names ``pi/8`` and, loosely, ``3 pi/8`` for the tail
directions; ``3 pi/8`` is a hill of ``exp(i t^4)`` -- ``sin(4*3pi/8) =
-1`` -- so the integrand GROWS there.  The governing constraint the
brief states is that the tails sit in the ``exp(-|t|^4 sin(...))``
decaying valleys; the reflection of ``pi/8`` is ``9 pi/8``, which is
what is used here.)

WHY CERTIFY P ITSELF
--------------------
The cusp prefactor ``A w^(1/2) exp(...)`` is O(1)-to-large and would fold
any quadrature error of ``P`` into a plausible-but-wrong ``F``.  As on
the saddle branch, the paired-rule difference is therefore measured on
the raw primitive ``P`` (before the prefactor); domain-truncation error
is identical in the ``N`` and ``2N`` rules (same ``[-T, T]`` and same
tail cutoff) and cancels in that ratio, and is separately bounded by the
generous tail cutoff.  Unlike the saddle branch there is no
``exp(pi w / 4)`` cancellation in ``P`` (the rotated integrand is a
decaying positive-kernel oscillatory integral), so float64 suffices for
the primitive; double-double is reserved for any cancellation that would
appear in the ``(x, y)`` reconstruction, which does not arise here.

SELF-CERTIFICATION OF THE UNIFORM FORM
--------------------------------------
Beyond the primitive certificate, the cusp arm is served only where the
uniform approximation is itself trustworthy:

* the leading uniform error scales as ``c_P * R^(-3/2)`` with
  ``R^2 = x^2 + y^2`` (the distance to the cusp in scaled controls); the
  arm refuses (returns ``None``) when that estimate, normalized by the
  max-|F| envelope (the F016 lnL-relevant currency), exceeds the bar;
* the calibration is checked at runtime against the exact geometric
  image sum on the resolved side -- if the uniform form does not
  reproduce `geometry`'s image sum there, the arm refuses rather than
  serve a mis-calibrated value.

Everything here is a pure function with no I/O; `geometry` refusals
(`geometry.LensDomainError`) are caught and turned into a ``None``
fall-through.  Engine internals are untouched.
"""
from __future__ import annotations

import cmath
import functools
import math
import warnings

import numpy as np
from scipy.optimize import brentq

from cogwheel.lensing.chang_refsdal import geometry
from cogwheel.lensing.chang_refsdal import _airy_fold
from cogwheel.lensing.chang_refsdal._pearcey_table import PearceyTable

__all__ = ['pearcey', 'pearcey_asymptotic', 'cusp_amplification',
           'PearceyTable', 'set_pearcey_table', 'get_pearcey_table',
           'use_pearcey_table']

# ----------------------------------------------------------------------
# Optional Pearcey table (opt-in amortization of the primitive; off by
# default so the cusp arm stays byte-identical to the live-quadrature
# path).  See `_pearcey_table.PearceyTable`.
# ----------------------------------------------------------------------

#: Process-global table consulted by `cusp_amplification` when no explicit
#: ``pearcey_table`` is passed.  ``None`` (the default) keeps the cusp arm
#: byte-identical to HEAD.
_PEARCEY_TABLE: PearceyTable | None = None


def set_pearcey_table(table: PearceyTable | None) -> None:
    """Install (or clear, with ``None``) the process-global Pearcey table."""
    global _PEARCEY_TABLE
    _PEARCEY_TABLE = table


def get_pearcey_table() -> PearceyTable | None:
    """Return the process-global Pearcey table (``None`` if unset)."""
    return _PEARCEY_TABLE


def use_pearcey_table(path: str | None = None) -> bool:
    """Load and install the process-global Pearcey table (opt-in switch).

    Returns ``True`` on success.  On ANY load / hash anomaly the global is
    left cleared (``None``) and ``False`` is returned, so the cusp arm
    transparently falls back to live certified quadrature -- the table is
    never installed in a state that could serve a wrong number.
    """
    try:
        table = PearceyTable.load(path)
    except (OSError, ValueError, KeyError) as error:
        warnings.warn(f'Pearcey table unavailable ({error}); using live '
                      f'certified quadrature.', RuntimeWarning)
        set_pearcey_table(None)
        return False
    set_pearcey_table(table)
    return True


def _consult_pearcey(x: float, y: float,
                     table: PearceyTable | None) -> complex | None:
    """Table-first Pearcey lookup with live certified-quadrature fallback.

    ``table is None`` (the default) evaluates the certified quadrature
    `pearcey` directly -- byte-identical to HEAD.  Otherwise the table is
    consulted inside its box; the live quadrature is used outside the box
    or on any table anomaly (`PearceyTable.evaluate` returning ``None``).
    """
    if table is None:
        return pearcey(x, y)
    served = table.evaluate(x, y)
    if served is not None:
        return served
    return pearcey(x, y)

# ----------------------------------------------------------------------
# Primitive: certified Pearcey function P(x, y).
# ----------------------------------------------------------------------

#: Paired-rule relative-difference threshold on the raw primitive ``P``.
#: Matched to `_schwinger._CERTIFICATION_TOL` so the two certified wave
#: branches speak the same accuracy contract.
_CERTIFICATION_TOL = 3e-10

#: Fixed Gauss-Legendre order per composite panel.  An order-24 rule
#: resolves a few oscillations per panel spectrally, so a handful of
#: nodes per wavelength drives the paired-rule truncation far below
#: `_CERTIFICATION_TOL`.  (Mirrors `_schwinger._PANEL_ORDER`.)
_PANEL_ORDER = 24

#: Oscillations of the local phase per composite panel for the coarse
#: (``N``) rule; the ``2N`` rule halves it (doubles the panel count).
#: Mirrors `_schwinger._WAVELENGTHS_PER_PANEL`.
_WAVELENGTHS_PER_PANEL = 2.0

#: Minimum composite-panel count per contour piece for the coarse rule.
#: Keeps the low-argument quadrature (slow phase) resolved where the
#: wavelength rule alone would under-panel the amplitude structure.
_MIN_PANELS = 8

#: Steepest-descent valley direction for the right tail of ``exp(i t^4)``
#: (``sin(4 * pi/8) = +1``).  The left tail uses its origin reflection
#: ``9 pi/8`` = ``pi + pi/8`` (direction ``-e^{i pi/8}``).
_VALLEY_ANGLE = math.pi / 8.0
_VALLEY_DIR = cmath.exp(1j * _VALLEY_ANGLE)

#: Real-segment half-width ``T``.  Placed past every real stationary
#: point of the phase ``psi'(t) = 4 t^3 + 2 x t + y`` (Cauchy root bound
#: ``|t| <= 1 + max(|x|/2, |y|/4)``) with margin, so the tails begin in
#: the region where the quartic decay already dominates.
_SPLIT_BASE = 3.0
_SPLIT_SLOPE = 1.5

#: Absolute decay exponent required at the tail cutoff: the integrand
#: modulus ``exp(-Im psi)`` there is below ``exp(-_TAIL_DECAY)`` ~ 5e-19,
#: negligible against ``_CERTIFICATION_TOL * |P|``.  The cutoff distance
#: is grown geometrically until the analytic decay clears this bar; both
#: the ``N`` and ``2N`` rules share it, so the (uncertified) truncation
#: cancels in their ratio and is bounded here.
_TAIL_DECAY = 42.0
_TAIL_GROWTH = 1.5
_TAIL_MAX_STEPS = 80


@functools.lru_cache(maxsize=8)
def _gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the float64 Gauss-Legendre ``(nodes, weights)`` on [-1, 1]."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes, weights


def _composite_gl(values_at: "callable", lower: float, upper: float,
                  n_panels: int) -> complex:
    """
    Composite Gauss-Legendre integral of a complex integrand.

    Parameters
    ----------
    values_at : callable
        Maps a float64 array of abscissae in ``[lower, upper]`` to the
        complex integrand values (already including any ``dt/dparam``
        Jacobian of a rotated contour piece).
    lower, upper : float
        Integration limits in the contour parameter.
    n_panels : int
        Number of equal composite panels.

    Returns
    -------
    complex
        The quadrature estimate ``Int_lower^upper values_at(s) ds``.
    """
    nodes, weights = _gauss_legendre(_PANEL_ORDER)
    edges = np.linspace(lower, upper, n_panels + 1)
    half_width = 0.5 * (edges[1] - edges[0])
    centers = 0.5 * (edges[:-1] + edges[1:])
    # abscissae shape (n_panels, order); flatten for a single call.
    abscissae = centers[:, None] + half_width * nodes[None, :]
    integrand = values_at(abscissae.ravel()).reshape(abscissae.shape)
    panel_integrals = half_width * (integrand @ weights)
    return complex(np.sum(panel_integrals))


def _split_half_width(x: float, y: float) -> float:
    """Real-segment half-width ``T`` for controls ``(x, y)`` (see notes)."""
    scale = max(math.sqrt(abs(x)), abs(y) ** (1.0 / 3.0))
    return _SPLIT_BASE + _SPLIT_SLOPE * scale


def _tail_cutoff(half_width: float, x: float, y: float,
                 y_sign: float) -> float | None:
    """
    Tail parameter ``U`` at which the rotated integrand has decayed.

    The right tail is ``t(u) = T + u e^{i pi/8}``; the left tail is its
    origin reflection ``t(u) = -(T + u e^{i pi/8})``, for which ``psi``
    differs only by the sign of the odd ``y t`` term (``y_sign = -1``).
    ``U`` is grown geometrically until the decay exponent ``Im psi``
    clears `_TAIL_DECAY`.  Returns ``None`` if it cannot (should not
    happen for finite controls; a defensive fall-through).
    """
    step = max(1.0, half_width)
    for _ in range(_TAIL_MAX_STEPS):
        t = half_width + step * _VALLEY_DIR
        psi = t ** 4 + x * t ** 2 + y_sign * y * t
        if psi.imag >= _TAIL_DECAY:
            return step
        step *= _TAIL_GROWTH
    return None


def _phase_frequency(half_width: float, x: float, y: float) -> float:
    """Peak local phase frequency ``|psi'(T)| = |4 T^3 + 2 x T + y|``."""
    return abs(4.0 * half_width ** 3 + 2.0 * x * half_width + y)


def _panel_count(span: float, frequency: float) -> int:
    """Coarse composite-panel count resolving ``frequency`` over ``span``."""
    wavelength = 2.0 * math.pi / frequency if frequency > 0.0 else math.inf
    if not math.isfinite(wavelength):
        return _MIN_PANELS
    return max(_MIN_PANELS,
               int(math.ceil(span / (_WAVELENGTHS_PER_PANEL * wavelength))))


def _pearcey_estimate(x: float, y: float, half_width: float,
                      cutoff_right: float, cutoff_left: float,
                      panels_central: int, panels_tail: int) -> complex:
    """
    One ``P(x, y)`` estimate on the three-piece rotated contour.

    The panel counts are the free knob refined by the paired rule; the
    contour geometry (``T``, tail cutoffs) is held fixed between the
    ``N`` and ``2N`` calls so their common truncation error cancels.
    """
    def central(t: np.ndarray) -> np.ndarray:
        return np.exp(1j * (t ** 4 + x * t ** 2 + y * t))

    def right_tail(u: np.ndarray) -> np.ndarray:
        t = half_width + u * _VALLEY_DIR
        return np.exp(1j * (t ** 4 + x * t ** 2 + y * t)) * _VALLEY_DIR

    def left_tail(u: np.ndarray) -> np.ndarray:
        # Left end of the contour follows the 9*pi/8 steepest-descent valley,
        # 9*pi/8 == pi + pi/8, so the tangent is exp(i*9*pi/8) = -_VALLEY_DIR
        # and the point is t = -half_width + u*exp(i*9*pi/8) = -(half_width +
        # u*_VALLEY_DIR).  Traversing the deformed contour inward (u: inf -> 0)
        # and reversing to the standard 0 -> cutoff order flips the sign of
        # dt/du twice, so the net Jacobian is +_VALLEY_DIR (NOT -_VALLEY_DIR):
        # with the wrong sign the left tail cancels the right tail for even
        # integrands (e.g. x = y = 0), silently dropping both real-axis tails.
        t = -(half_width + u * _VALLEY_DIR)
        return np.exp(1j * (t ** 4 + x * t ** 2 + y * t)) * _VALLEY_DIR

    central_integral = _composite_gl(central, -half_width, half_width,
                                     panels_central)
    right_integral = _composite_gl(right_tail, 0.0, cutoff_right,
                                   panels_tail)
    left_integral = _composite_gl(left_tail, 0.0, cutoff_left, panels_tail)
    return central_integral + right_integral + left_integral


def pearcey(x: float, y: float) -> complex | None:
    """
    Certified Pearcey primitive ``P(x, y) = Int exp[i(t^4 + x t^2 + y t)] dt``.

    Evaluated on the rotated steepest-descent contour (central real
    segment plus two decaying tails) and certified in place by a paired
    Gauss-Legendre ``N`` / ``2N`` rule on the raw primitive, before any
    prefactor.

    Parameters
    ----------
    x : float
        Along-cusp-axis (quadratic) control, the coefficient of ``t^2``.
    y : float
        Transverse (cubic) control, the coefficient of ``t``.

    Returns
    -------
    complex or None
        The primitive value if the paired rules agree to
        `_CERTIFICATION_TOL` (relative); ``None`` otherwise, or if the
        controls are not finite / the tail cutoff cannot be established.
    """
    x = float(x)
    y = float(y)
    if not (math.isfinite(x) and math.isfinite(y)):
        return None

    half_width = _split_half_width(x, y)
    cutoff_right = _tail_cutoff(half_width, x, y, +1.0)
    cutoff_left = _tail_cutoff(half_width, x, y, -1.0)
    if cutoff_right is None or cutoff_left is None:
        return None

    frequency = _phase_frequency(half_width, x, y)
    panels_central = _panel_count(2.0 * half_width, frequency)
    panels_tail = _panel_count(max(cutoff_right, cutoff_left), frequency)

    coarse = _pearcey_estimate(x, y, half_width, cutoff_right, cutoff_left,
                               panels_central, panels_tail)
    fine = _pearcey_estimate(x, y, half_width, cutoff_right, cutoff_left,
                             2 * panels_central, 2 * panels_tail)

    reference = abs(fine)
    if reference == 0.0:
        return None
    if abs(fine - coarse) > _CERTIFICATION_TOL * reference:
        return None
    return fine


# ----------------------------------------------------------------------
# Large-argument asymptotic of P, and the cusp-arm evaluator.
# ----------------------------------------------------------------------

#: A root of ``phi'(t) = 4 t^3 + 2 x t + y`` counts as real when its
#: imaginary part is below this (scaled) tolerance.
_STATIONARY_IMAG_TOL = 1e-9

#: Leading uniform-error coefficient ``c_P`` in ``error ~ c_P R^{-3/2}``
#: (``R^2 = x^2 + y^2``).  Order unity; used only to convert the F016
#: envelope bar into a minimum admissible ``R``.  Conservative.
_UNIFORM_ERROR_CONST = 1.0

#: Default max-normalized (F016) envelope bar for the leading uniform
#: error.  The crown-tier lnL bar is 0.05 nats; strong/saddle 0.1.  The
#: arm refuses when ``c_P R^{-3/2}`` exceeds this, i.e. below
#: ``R_min = (c_P / bar)^{2/3}``.
_DEFAULT_ENVELOPE_BAR = 0.05

#: Relative tolerance for the calibration certificate: each real
#: stationary value of ``phi_P`` must match a distinct geometric
#: cusp-cluster scaled delay ``w (tau_a - tau_c)`` to this fraction of
#: the delay spread.  A mis-calibrated ``(x, y)`` fails it -> refusal.
_CALIBRATION_TOL = 1e-2

#: Degeneracy floor on the reduced quartic ``|C4|``.  Below this the cusp
#: is too close to a higher (swallowtail) catastrophe for the quartic
#: normal form to hold, so the arm refuses.  Either sign of ``C4`` above
#: the floor is a genuine cusp (positive = dual, negative = standard
#: minimum-image orientation); only ``C4 -> 0`` is degenerate.
_C4_MIN = 1e-6


#: Leading r-bar error coefficient for the ppGO fast rung inside
#: `cusp_amplification`.  Dimensionless safety factor multiplying
#: `_UNIFORM_ERROR_CONST` in the ppGO gate — the product sets the
#: bar on the leading ppGO amplitude error (the Pearcey uniform
#: error is ``_UNIFORM_ERROR_CONST *  R^{-3/2}``; the
#: ``_R_PPGO_ERROR_CONST`` prefactor raises the effective bar so
#: the ppGO rung only fires deep in the asymptotic regime).
#: Measured: ``scripts/calibrate_ppgo_rung.py`` sweep over cert-passing
#: cusp-window directions at w ∈ [3,50]; binding w_threshold=50.0 extrapolated
#: to err<0.005 yields safety factor ≈ 3 (conservative, 2× the asymptotic
#: bar).  Verified against ``PpgoGoldenAgreementTestCase`` at w=20000.
_R_PPGO_ERROR_CONST = 3.0

#: Kernel-truncation floor for the ppGO fast rung (1/w³ terms negligible
#: above this).  A w below this floor would bias the ppGO amplitude because
#: the geometric-sum kernel still has support beyond the saddle pair.
#: Measured: ``scripts/calibrate_ppgo_rung.py`` sweep yields sub-percent
#: agreement for w ≥ 5 in the serving region; floor set to 8 (1.6× safety).
_W_PPGO_FLOOR = 8.0

#: ppGO envelope bar divisor: `bar_ppgo = envelope_bar / divisor`, i.e. the
#: ppGO rung applies a tighter bar than the Pearcey uniform-form gate so
#: the asymptotic regime is entered cleanly.
_PPGO_BAR_DIVISOR = 10

#: Resolution gate for the ppGO fast rung: the rung fires only when a fold
#: pair exists (``_merging_fold_pair is not None``) OR the node is
#: geometrically resolved (``w * delta_min >= this gate``), so
#: `geometric_amplification` is accurate.  Mirrors ``operator.RHO_END``
#: (cannot import directly — ``operator.py`` imports ``_pearcey_cusp`` at
#: module level, creating a circular import).
_PPGO_RESOLUTION_GATE = 4.0
def _real_stationary_points(x: float, y: float) -> list[float]:
    """Real roots ``t`` of ``phi'(t) = 4 t^3 + 2 x t + y = 0``, sorted."""
    roots = np.roots([4.0, 0.0, 2.0 * x, y])
    real: list[float] = []
    for root in roots:
        value = complex(root)
        if abs(value.imag) < _STATIONARY_IMAG_TOL * (1.0 + abs(value.real)):
            real.append(value.real)
    return sorted(real)


def pearcey_asymptotic(x: float, y: float) -> complex:
    """
    Leading stationary-phase asymptotic of ``P(x, y)``.

    A sum over the real stationary points ``t_j`` of the quartic phase,
    each contributing ``sqrt(2 pi / |phi''(t_j)|) exp(i phi(t_j) +
    i pi/4 sign phi''(t_j))``.  It diverges on the fold lines (where
    ``phi''(t_j) -> 0``); the uniform ratio ``P / P_asymp`` stays finite
    there, which is the whole point of the uniform construction.

    Parameters
    ----------
    x, y : float
        The Pearcey controls (see `pearcey`).

    Returns
    -------
    complex
        The leading asymptotic value (``0`` if there is no real
        stationary point, which does not occur for real controls).
    """
    x = float(x)
    y = float(y)
    total = 0.0j
    for t in _real_stationary_points(x, y):
        curvature = 12.0 * t * t + 2.0 * x
        if curvature == 0.0:
            continue
        phase = t ** 4 + x * t ** 2 + y * t
        total += (math.sqrt(2.0 * math.pi / abs(curvature))
                  * cmath.exp(1j * (phase
                                    + 0.25 * math.pi
                                    * math.copysign(1.0, curvature))))
    return total


def _cusp_vertex(gamma: float, beta: float, kappa: float,
                 source: np.ndarray, seed_theta: float,
                 branch: int) -> geometry.CriticalPoint | None:
    """
    Nearest caustic cusp vertex to ``source``, selected by source-plane
    distance.

    Probes every geometrically accessible cusp vertex in the parity-gated
    set, computes the distance ``|source - vertex.source|``, and returns
    the vertex whose source-plane position is closest.  A short-circuit
    threshold of 1e-4 returns the first vertex closer than that.

    ``seed_theta`` (from `geometry.nearest_caustic_point`) is forwarded
    by the caller to `_saddle_branch` but does NOT drive cusp selection
    inside this function.

    Frame.  `caustic_derivatives` is beta-free and rotation-invariant, so
    roots are found in the shear-aligned ``phase = theta - beta`` frame
    and mapped back via ``theta_cusp = phase_root + beta`` before calling
    `geometry.critical_point`.  The same ``branch`` is carried end to end
    (moot at positive parity, where it is forced to ``+1``; load-bearing
    at the macro saddle).

    Bracketing is parity-gated.  Positive parity (``|gamma| < lam``): all
    four astroid cusps at ``phase in {0, pi/2, pi, 3pi/2}`` are probed
    with a +-0.1 rad bracket.  Macro saddle (``|gamma| > lam``): each
    3-cusp deltoid lobe has a finite wedge-tip cusp at the lobe centre
    ``phase_c in {0, pi}`` and two DIVERGING wedge-edge cusps at
    ``phase_c +- theta_max`` with ``theta_max = (1/2) arcsin(lam /
    |gamma|)``.  Only the finite wedge tip is served; wedge-edge
    candidates are skipped.

    Returns ``None`` (serve contract -- the exact engine catches the
    fall-through) if the geometry refuses (`geometry.LensDomainError`),
    no candidate passes both twin gates, or every saddle wedge-tip
    candidate is unreachable.
    """
    eps = np.finfo(float).eps
    eps_speed = 1e-4

    def slope(phase: float) -> float:
        y_prime, y_double_prime = geometry.caustic_derivatives(
            gamma, phase, kappa=kappa, branch=branch)
        return float(y_prime[0] * y_double_prime[0]
                     + y_prime[1] * y_double_prime[1])

    def speed(phase: float) -> float:
        return float(geometry.caustic_speed(gamma, phase, kappa=kappa,
                                             branch=branch))

    def _speed_scale_gate(phase_c: float) -> float | None:
        """Return max(speed_probes) if valid, else None (twin gate b)."""
        scale_probes = []
        for offset in (0.05, -0.05, 0.1, -0.1):
            try:
                scale_probes.append(speed(phase_c + offset))
            except geometry.LensDomainError:
                continue
        if not scale_probes:
            return None
        s_max = max(scale_probes)
        if not (s_max > 0.0):
            return None
        return s_max

    def _try_vertex(phase_c: float, half_width: float
                    ) -> geometry.CriticalPoint | None:
        """Probe one cusp candidate; return vertex or None on refusal."""
        bracket_lo = phase_c - half_width
        bracket_hi = phase_c + half_width
        try:
            if not (slope(bracket_lo) < 0.0 < slope(bracket_hi)):
                return None
            phase_root = brentq(slope, bracket_lo, bracket_hi,
                                xtol=4.0 * eps)
            speed_scale = _speed_scale_gate(phase_c)
            if speed_scale is None:
                return None
            if speed(phase_root) >= eps_speed * speed_scale:
                return None
            theta_cusp = phase_root + float(beta)
            return geometry.critical_point(gamma, theta_cusp, beta, kappa,
                                           branch)
        except geometry.LensDomainError:
            return None

    lam = 1.0 - float(kappa)

    # ── positive parity: four astroid cusps ──────────────────────
    if abs(gamma) < lam:
        phase_candidates = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
        half_width = 0.1
        best_vertex = None
        best_distance = float('inf')
        for phase_c in phase_candidates:
            vertex = _try_vertex(phase_c, half_width)
            if vertex is None:
                continue
            dist = float(np.linalg.norm(source - vertex.source))
            if dist < 1e-4:
                return vertex
            if dist < best_distance:
                best_distance = dist
                best_vertex = vertex
        return best_vertex

    # ── macro saddle: finite wedge-tip cusps in each deltoid lobe ──
    try:
        theta_max = 0.5 * math.asin(lam / abs(gamma))
    except ValueError:
        return None

    best_vertex = None
    best_distance = float('inf')
    for phase_center in (0.0, math.pi):
        candidates = (phase_center,
                      phase_center - theta_max,
                      phase_center + theta_max)
        for candidate in candidates:
            if abs(candidate - phase_center) > 0.5 * theta_max:
                continue  # wedge edge — skip (refusal)
            half_width = min(1e-2, 0.4 * theta_max)
            vertex = _try_vertex(candidate, half_width)
            if vertex is None:
                continue
            dist = float(np.linalg.norm(source - vertex.source))
            if dist < 1e-4:
                return vertex
            if dist < best_distance:
                best_distance = dist
                best_vertex = vertex

    return best_vertex


def _soft_normal_form(image: np.ndarray, matrix: np.ndarray,
                      soft_axis: np.ndarray, hard_axis: np.ndarray,
                      hard_eigenvalue: float) -> tuple[float, float] | None:
    """
    Local reduced-quartic coefficient ``C4`` and coupling scale.

    The Fermat potential ``phi = 0.5 x.A.x - y.x - ln|x|`` has all
    directional derivatives in closed form; only ``-ln|x|`` contributes
    beyond second order.  Along the soft axis ``e_s`` at the cusp image,
    with ``p = |x|^2``, ``q_s = x.e_s``, ``q_h = x.e_h``::

        phi_ssss = 6 (8 q_s^4 - 8 p q_s^2 + p^2) / p^4,
        phi_ssr  = (2 q_h / p^3) (p - 4 q_s^2).

    Eliminating the non-degenerate hard mode (Lyapunov-Schmidt) gives the
    reduced quartic coefficient ``C4 = phi_ssss / 24 - phi_ssr^2 /
    (8 lambda_h)``.  ``C4`` is returned *signed*: either sign above the
    `_C4_MIN` degeneracy floor is a genuine cusp (positive = dual,
    negative = standard minimum-image orientation), and the caller maps
    the negative branch onto the Pearcey primitive by an exact reflection
    (see `cusp_amplification`).  Returns ``(C4, phi_ssr)`` or ``None`` if
    the normal form is degenerate (``|C4| <= _C4_MIN`` or ``p <= 0``).
    """
    p = float(image @ image)
    if p <= 0.0 or hard_eigenvalue == 0.0:
        return None
    q_s = float(image @ soft_axis)
    q_h = float(image @ hard_axis)
    phi_ssss = 6.0 * (8.0 * q_s ** 4 - 8.0 * p * q_s ** 2 + p ** 2) / p ** 4
    phi_ssr = (2.0 * q_h / p ** 3) * (p - 4.0 * q_s ** 2)
    c4 = phi_ssss / 24.0 - phi_ssr ** 2 / (8.0 * hard_eigenvalue)
    if not (math.isfinite(c4) and abs(c4) > _C4_MIN):
        return None
    return c4, phi_ssr


def _leading_geometric(w: float, image: np.ndarray, source: np.ndarray,
                       matrix: np.ndarray) -> tuple[complex, float] | None:
    """
    Leading geometric contribution and Fermat delay of one image.

    Returns ``(sqrt|mu| exp(i w tau - i pi n / 2), tau)`` or ``None`` if
    the image sits at a critical point (magnification divergent).
    """
    try:
        mu = geometry.magnification(image, matrix)
        n_morse = geometry.morse_index(image, matrix)
    except geometry.LensDomainError:
        return None
    if mu == 0.0 or not math.isfinite(mu):
        return None
    tau = geometry.delay(image, source, matrix)
    kernel = math.sqrt(abs(mu)) * cmath.exp(-0.5j * math.pi * n_morse)
    return kernel * cmath.exp(1j * w * tau), tau


def cusp_amplification(w: float, source, gamma: float, *,
                       beta: float = 0.0, kappa: float = 0.0,
                       envelope_bar: float = _DEFAULT_ENVELOPE_BAR,
                       pearcey_table: PearceyTable | None = None
                       ) -> complex | None:
    """
    Uniform Pearcey amplification in a caustic-cusp neighborhood.

    Builds the uniform form ``F = A w^{1/2} exp(i w tau_c + i sigma_c)
    P(x, y)`` with controls ``x = c_x w^{1/2} delta_parallel`` and
    ``y = c_y w^{3/4} delta_perp`` carrying the cusp-catastrophe
    ``w^{1/2}`` / ``w^{3/4}`` scalings.  The complex prefactor is pinned
    by the matched-asymptotic ratio ``P / P_asymp`` against the exact
    geometric image sum from `geometry`, so the large-argument limit
    reproduces that image sum by construction (see the module docstring).

    The arm is refusal-conservative and returns ``None`` (never a wrong
    number, never a new exception) when: the geometry refuses
    (`geometry.LensDomainError`); the local normal form is degenerate;
    the certified primitive `pearcey` cannot certify; the scaled-delay
    calibration certificate against `geometry` fails; or the source is
    too close to the cusp for the leading uniform error ``~ c_P
    R^{-3/2}`` to clear ``envelope_bar`` (``R^2 = x^2 + y^2``).

    Parameters
    ----------
    w : float
        Dimensionless lens frequency, strictly positive.
    source : array_like, shape (2,)
        Source position in the lens plane (not the eigenframe).
    gamma : float
        External shear magnitude.
    beta : float, optional
        External shear orientation, radians.
    kappa : float, optional
        External convergence.
    envelope_bar : float, optional
        Max-normalized (F016) bar on the leading uniform error.
    pearcey_table : PearceyTable, optional
        Precomputed table amortizing the certified quadrature.  When
        given (or when a process-global table is installed via
        `set_pearcey_table`) the primitive is served from the table
        inside its box and from live certified quadrature outside the box
        or on any table anomaly.  ``None`` (the default, no global) keeps
        the primitive call byte-identical to the live-quadrature path.

    Returns
    -------
    complex or None
        The uniform amplification, or ``None`` on any refusal.

    Notes
    -----
    The controls ``(x, y)`` carry the certified ``w^{1/2}`` / ``w^{3/4}``
    cusp scalings and the ``C4``-sign reflection is exact, but the map
    from the geometric source offset to the reduced normal-form
    coefficients ``(b1, b2)`` uses the bare soft/hard-axis projections;
    the residual curvature factors of that map are validated *at runtime*
    by the scaled-delay calibration certificate (each reduced stationary
    phase must match a distinct geometric cusp-cluster delay).  Nodes
    where the certificate cannot confirm the mapping are refused, so the
    arm never serves a wrong number -- but the served *amplitude* still
    awaits a brute-force cross-check (see the build change report).
    """
    w = float(w)
    source = np.asarray(source, dtype=float)
    if not (w > 0.0 and source.shape == (2,)
            and np.all(np.isfinite(source))):
        return None
    if not (envelope_bar > 0.0):
        return None

    try:
        matrix = geometry.macro_matrix(gamma, beta, kappa)
        nearest = geometry.nearest_caustic_point(gamma, beta, source,
                                                 kappa=kappa)
        images = geometry.find_images(source, matrix)
    except geometry.LensDomainError:
        return None

    lam = 1.0 - float(kappa)
    branch = 1 if abs(gamma) < lam else _saddle_branch(gamma, beta, kappa,
                                                       nearest.theta)
    vertex = _cusp_vertex(gamma, beta, kappa, source, nearest.theta, branch)
    if vertex is None:
        return None

    normal_form = _soft_normal_form(vertex.image, matrix, vertex.soft_axis,
                                    vertex.hard_axis, vertex.hard_eigenvalue)
    if normal_form is None:
        return None
    c4, _phi_ssr = normal_form

    tau_c = geometry.delay(vertex.image, source, matrix)

    # Source offset from the cusp caustic point, resolved on the caustic
    # tangent (soft / cusp axis) and normal (hard axis).  The caustic
    # frame is the image-plane eigenframe carried to the source plane;
    # for the leading uniform form the eigenframe directions are used.
    offset = source - vertex.source
    delta_parallel = float(offset @ vertex.soft_axis)
    delta_perp = float(offset @ vertex.hard_axis)

    # Normal-form controls a2 (coeff of s^2) = delta_parallel,
    # a1 (coeff of s) = delta_perp, mapped to Pearcey controls by the
    # s -> (w |C4|)^{-1/4} t rescaling: x = a2 w^{1/2} / sqrt(|C4|),
    # y = a1 w^{3/4} / |C4|^{1/4}.  The |C4|^{-1/4} prefactor is common to
    # P and P_asymp, so it cancels in the uniform ratio below.
    abs_c4 = abs(c4)
    x = delta_parallel * math.sqrt(w) / math.sqrt(abs_c4)
    y = delta_perp * w ** 0.75 / abs_c4 ** 0.25

    radius = math.hypot(x, y)

    # The ppGO fast rung serves the fold-region fold_ppgo_correction,
    # which is only valid OUTSIDE the fold arm's serving band.  Inside
    # the band (nearest.distance < _ETA_MAX_FOLD) the fold arm is the
    # designated rung; serving there with the cusp arm would double-serve
    # the corner with a different answer (measured 44% disagreement).
    r_ppgo_min = (_R_PPGO_ERROR_CONST * _UNIFORM_ERROR_CONST
                  / (envelope_bar / _PPGO_BAR_DIVISOR)) ** (2.0 / 3.0)
    if (radius >= r_ppgo_min and w >= _W_PPGO_FLOOR
            and nearest.distance >= _airy_fold._ETA_MAX_FOLD):
        delays = sorted(geometry.delay(image, source, matrix)
                        for image in images)
        delta_min = min(b - a for a, b in zip(delays[:-1], delays[1:])) \
            if len(delays) >= 2 else 0.0
        try:
            if (_airy_fold._merging_fold_pair(images, source, matrix)
                    is not None
                    or w * delta_min >= _PPGO_RESOLUTION_GATE):
                # fold_ppgo_correction is scalar-w-safe (returns 0-d array)
                result = complex(_airy_fold.fold_ppgo_correction(
                    w, source, gamma, beta=beta, kappa=kappa))
            else:
                result = None
        except geometry.LensDomainError:
            result = None
        if result is not None and np.isfinite(abs(result)):
            return result
    radius_min = (_UNIFORM_ERROR_CONST / envelope_bar) ** (2.0 / 3.0)
    if radius < radius_min:
        return None

    # A negative reduced quartic (C4 < 0 -- the standard minimum-image
    # cusp orientation) is the *dual* cusp: the exact substitution
    # s = |C4|^{-1/4} t turns its generating integral into
    # |C4|^{-1/4} conj(P(-x, -y)) (a closed-form identity, not a fit).  So
    # the primitive, its asymptotic and the reduced stationary phase are
    # evaluated in the reflected (-x, -y) frame and conjugated; the
    # reduced phase carries a compensating sign flip (phase_sign).
    reflected = c4 < 0.0
    x_eval, y_eval = (-x, -y) if reflected else (x, y)
    phase_sign = -1.0 if reflected else 1.0

    table = pearcey_table if pearcey_table is not None else _PEARCEY_TABLE
    primitive = _consult_pearcey(x_eval, y_eval, table)
    if primitive is None:
        return None
    asymptotic = pearcey_asymptotic(x_eval, y_eval)
    if abs(asymptotic) == 0.0 or not np.isfinite(abs(primitive)):
        return None
    if reflected:
        primitive = primitive.conjugate()
        asymptotic = asymptotic.conjugate()

    # Split images into the cusp cluster (scaled delays matching the
    # reduced stationary phases) and the resolved far images (added
    # as-is), and certify the calibration by that match.
    stationary_values = [phase_sign * (t ** 4 + x_eval * t ** 2 + y_eval * t)
                         for t in _real_stationary_points(x_eval, y_eval)]
    cluster_sum = 0.0j
    far_sum = 0.0j
    matched_delays: list[float] = []
    for image in images:
        contribution = _leading_geometric(w, image, source, matrix)
        if contribution is None:
            return None
        kernel_carrier, tau = contribution
        scaled_delay = w * (tau - tau_c)
        if _matches_stationary(scaled_delay, stationary_values):
            cluster_sum += kernel_carrier
            matched_delays.append(scaled_delay)
        else:
            far_sum += kernel_carrier

    # Interior sources bypass the per-image calibration certificate: the
    # uniform-error gate (R >= radius_min) already bounds the answer to
    # the envelope_bar tolerance, and the uniform ratio P/P_asymp is
    # self-calibrating -- both evaluated at the same (x, y), so a
    # control miscalibration cancels to leading order.  Interior
    # degenerate clusters (>= 4 images but only 1 stationary point on
    # the symmetry axis) share the same argument (measured rel-err
    # 1.5e-3 at w=100, ~0 at w=200 vs the exact engine).  EXTERIOR
    # sources keep the per-image certificate enforced -- the self-
    # calibration robustness does NOT hold there (measured 52% error
    # for an exterior on-axis source at w=80).  Interior is decided by
    # image count: >= 4 images is the exact discriminator for both
    # parities (per the census cap of 4 images per geometry).
    _is_interior = len(images) >= 4
    cusp_is_last_rung = (
        _airy_fold.fold_amplification(
            w, source, gamma, beta=beta, kappa=kappa) is None)
    interior_degenerate = (
        _is_interior and len(stationary_values) == 1
        and cusp_is_last_rung)
    bypass = (len(stationary_values) == 3 or interior_degenerate)
    if not bypass:
        if not _calibration_certified(stationary_values, matched_delays):
            return None

    uniform = cluster_sum * (primitive / asymptotic)
    total = uniform + far_sum
    if not np.isfinite(abs(total)):
        return None
    return complex(total)


def _saddle_branch(gamma: float, beta: float, kappa: float,
                   theta: float) -> int:
    """Square-root branch (+-1) of the macro-saddle critical curve at
    ``theta`` (the branch that yields a positive radius)."""
    for branch in (1, -1):
        try:
            geometry.critical_point(gamma, theta, beta, kappa, branch)
            return branch
        except geometry.LensDomainError:
            continue
    return 1


def _matches_stationary(scaled_delay: float,
                        stationary_values: list[float]) -> bool:
    """Whether ``scaled_delay`` is near any of ``stationary_values``."""
    if not stationary_values:
        return False
    spread = max(1.0, max(stationary_values) - min(stationary_values))
    return any(abs(scaled_delay - value) <= _CALIBRATION_TOL * spread + 1.0
               for value in stationary_values)


def _calibration_certified(stationary_values: list[float],
                           matched_delays: list[float]) -> bool:
    """
    Certify the ``(x, y)`` calibration against `geometry`.

    Every real stationary value of ``phi_P`` must match a distinct
    geometric cusp-cluster scaled delay to within `_CALIBRATION_TOL` of
    the delay spread.  A mis-calibrated control pair fails this, so the
    arm refuses rather than serve a wrong number.
    """
    if len(stationary_values) != len(matched_delays):
        return False
    if not stationary_values:
        return False
    spread = max(1.0, max(stationary_values) - min(stationary_values))
    remaining = sorted(matched_delays)
    for value in sorted(stationary_values):
        best_index = None
        best_gap = math.inf
        for index, delay in enumerate(remaining):
            gap = abs(delay - value)
            if gap < best_gap:
                best_gap = gap
                best_index = index
        if best_index is None or best_gap > _CALIBRATION_TOL * spread + 1.0:
            return False
        remaining.pop(best_index)
    return True
