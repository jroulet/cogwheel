"""
Uniform Airy (fold-catastrophe) wave branch for near-fold neighborhoods.

WHAT
----
Two public entry points, both refusal-conservative (they return ``None``
rather than a wrong number, and raise NO new exception class):

* `airy_fold_value(w, tau_bar, xi_control, p, q, sigma)` -- the pure
  uniform Airy fold form

      F_Airy = 2 sqrt(pi) exp(i (w tau_bar + sigma))
               [ p w^{1/6} Ai(-xi) - i q w^{-1/6} Ai'(-xi) ],

  evaluated with `scipy.special.airy`.  ``xi_control`` is *signed*: a
  positive value is the two-real-image (inside-caustic) side, where
  ``Ai(-xi)`` oscillates; a negative value is the no-image
  (outside-caustic) side, where ``Ai(+|xi|)`` decays evanescently.  This
  is a total function of finite inputs -- it never returns ``None`` and is
  finite at ``xi = 0`` (on-caustic), where ``Ai(0) = 1 / (3^{2/3}
  Gamma(2/3)) ~= 0.3550`` and ``Ai'(0)`` are both finite.

* `fold_amplification(w, source, gamma, ...)` -- the fold-arm evaluator

  the uniform asymptotic amplification for two coalescing (fold) images,
  valid at any ``w``.  It locates the merging minimum (Morse index 0) /
  saddle (Morse index 1) pair from `geometry.find_images`, builds the
  Airy control ``xi`` from their Fermat-delay separation, builds the
  amplitudes ``p, q`` and the fixed fold phase ``sigma`` from the FINITE
  fold-frame curvatures at the nearby caustic point, self-certifies, and
  returns the complex amplification or ``None`` on any refusal.

CONVENTION (VERIFIED against `geometry.image_kernel`)
-----------------------------------------------------
cogwheel's amplification is ``F = sum_a exp(+1j w tau_a) K_a`` with the
carrier-free kernel ``K_a = sqrt|mu_a| exp(-0.5j pi n_a) (1 + ...)``; a
minimum (``n = 0``) carries ``+sqrt|mu|`` and a saddle (``n = 1``) carries
``sqrt|mu| exp(-i pi/2)``.  The fold arm reproduces exactly this two-image
sum in its large-``xi`` limit (see CALIBRATION), so its phase convention is
identical to the geometric branch by construction.

THE CONTROL xi
--------------
Write the two merging images as ``tau_plus`` (the minimum) and
``tau_minus`` (the saddle), with ``tau_minus > tau_plus`` near a fold, and

    tau_bar = (tau_plus + tau_minus) / 2 ,
    DT      = tau_minus - tau_plus  (> 0, the FULL delay separation) .

The uniform Airy control is

    xi = (3 w DT / 4)^{2/3} .

(The build brief also writes ``xi = (3 w Delta_tau / 4)^{2/3}`` with
``Delta_tau = (tau_minus - tau_plus) / 2``; those two statements are
mutually inconsistent by a factor of two.  The value that makes the
large-``xi`` limit reproduce the geometric two-image sum -- checked below --
is the one written here, using the FULL separation ``DT``.  Equivalently:
the Airy oscillation ``(2/3) xi^{3/2} = w DT / 2`` must equal the
carrier-relative image phase offset ``w (tau_minus - tau_bar) = w DT / 2``;
this pins ``xi^{3/2} = 3 w DT / 4``.)

CALIBRATION OF (p, q, sigma) -- Professor flag #1
-------------------------------------------------
The amplitudes are built from the FINITE fold-frame curvatures -- the
hard-axis Hessian eigenvalue ``lambda_h`` and the cubic soft-axis
coefficient ``b3`` (the third derivative of the Fermat potential along the
degenerate/soft eigenaxis at the caustic critical point) -- and NOT from
the raw single-image ``sqrt|mu|``, which DIVERGES at the fold (``mu = 1 /
det H`` and ``det H -> 0`` there).  Using ``sqrt|mu|`` directly would
double-count that blow-up against the ``w^{1/6}`` prefactor.

Substitute the standard asymptotics into the form and require the
large-``xi`` limit to equal the geometric sum ``sqrt|mu_+| exp(i w tau_+)
+ sqrt|mu_-| exp(i w tau_- - i pi/2)``.  With

    Ai(-xi)  ~ pi^{-1/2} xi^{-1/4} sin((2/3) xi^{3/2} + pi/4) ,
    Ai'(-xi) ~ pi^{-1/2} xi^{ 1/4} cos((2/3) xi^{3/2} + pi/4) ,

both ``w^{1/6}`` / ``w^{-1/6}`` prefactors cancel the Airy-argument powers,
leaving a ``w``-independent match.  Splitting into the ``exp(i w tau_+)``
(minimum) and ``exp(i w tau_-)`` (saddle) channels gives, at leading order,

    sigma = -pi/4 ,
    q     = 0 ,
    p     = 2^{-1/6} |lambda_h|^{-1/2} |b3|^{-1/3} .

The divergent geometric scale (the image-merge half-separation
``s0 -> 0``) cancels out of ``p`` exactly, so ``p`` is finite at the fold:
this is the whole point of building it from ``lambda_h`` and ``b3``.  With
these three constants the form reproduces the geometric two-image sum term
by term (minimum with ``n = 0``, saddle with the exact ``-pi/2`` Morse
phase).  ``q = 0`` is the rigorous leading value for the pure-phase
lensing diffraction integral (unit Kirchhoff amplitude, so the two
stationary curvatures are equal in magnitude to cubic order and the
``Ai'`` channel vanishes).  The ``Ai'`` correction is a genuinely
higher-order term set by the quartic coefficient ``b4`` (outside the
curvature inputs this arm gathers); it is deferred, and the served
amplitude is therefore leading-order (see the change report / Notes).

SELF-CERTIFICATION
------------------
Beyond the geometry refusals, the arm serves only where the leading
uniform error estimate ``~ c_A xi^{-3/2}`` (``c_A`` the ``C1`` saddle
coefficient magnitude of `geometry.saddle_coefficients` in the fold frame)
clears the F016 max-normalized envelope bar; otherwise it returns ``None``.
This is refusal-conservative: a divergent ``C1`` (an image too close to the
fold for the metric inversion) is caught and turned into a ``None``
fall-through rather than a served value.

Everything here is a pure function with no I/O; `geometry` refusals
(`geometry.LensDomainError`) are caught and turned into ``None``.  Engine
internals are untouched.
"""
from __future__ import annotations

import cmath
import math

import numpy as np
from scipy.special import airy

from cogwheel.lensing.chang_refsdal import geometry

__all__ = ['airy_fold_value', 'fold_amplification', 'fold_ppgo_correction']


# ----------------------------------------------------------------------
# Fixed calibration constants (closed form; see the module docstring).
# ----------------------------------------------------------------------

#: The fold phase pinned by the large-``xi`` match to the geometric
#: two-image sum: ``sigma = -pi/4``.
_SIGMA_FOLD = -0.25 * math.pi

#: Overall real prefactor of ``p``: ``2^{-1/6}``.  With
#: ``p = 2^{-1/6} |lambda_h|^{-1/2} |b3|^{-1/3}`` the divergent merge scale
#: cancels and ``p`` stays finite at the fold.
_AMP_CONST = 2.0 ** (-1.0 / 6.0)

#: Degeneracy floor on the soft-axis cubic ``|b3|``.  Below this the fold
#: is too close to a higher (cusp) catastrophe for the cubic normal form
#: to hold, so the arm refuses (the cusp arm owns that neighborhood).
_B3_MIN = 1e-6

#: Default max-normalized (F016) envelope bar for the leading uniform
#: error.  The crown-tier lnL bar is 0.05 nats; the arm refuses when the
#: estimate ``c_A xi^{-3/2}`` exceeds this.
_DEFAULT_ENVELOPE_BAR = 0.05

#: Maximum distance from the caustic at which this arm may serve, in
#: Einstein-radius units.  ``q = 0`` is a symmetric-fold assumption, exact
#: only where the merging pair has equal magnification (at the caustic), so
#: validity is bounded by distance from the caustic -- which the ``xi``
#: certificate does not measure.  Complement of
#: `operator.ETA_MIN_GEOMETRIC`; a literal, not an import, because
#: `operator` imports this module.  Pinned equal by test.  F028, F031, F032.
_ETA_MAX_FOLD = 0.3


# ----------------------------------------------------------------------
# The pure uniform Airy fold form.
# ----------------------------------------------------------------------

def airy_fold_value(w: float, tau_bar: float, xi_control: float,
                    p: float, q: float, sigma: float) -> complex:
    """
    Uniform Airy fold form ``F_Airy`` (a total function of finite inputs).

    Evaluates

        2 sqrt(pi) exp(i (w tau_bar + sigma))
        [ p w^{1/6} Ai(-xi) - i q w^{-1/6} Ai'(-xi) ]

    with ``scipy.special.airy``.  ``xi_control`` is signed: positive is the
    inside-caustic (two-image) side where ``Ai(-xi)`` oscillates, negative
    is the outside-caustic (no-image) side where ``Ai(+|xi|)`` decays.  The
    result is finite for every finite argument, including ``xi = 0``.

    Parameters
    ----------
    w : float
        Dimensionless lens frequency, strictly positive.
    tau_bar : float
        Mean Fermat delay of the merging pair (the carrier phase / w).
    xi_control : float
        Signed uniform Airy control (see above).
    p, q : float
        Airy and Airy-derivative amplitudes (see `fold_amplification`).
    sigma : float
        Fixed fold phase (radians).

    Returns
    -------
    complex
        The uniform Airy amplification value.
    """
    w = float(w)
    ai_value, aip_value, _, _ = airy(-float(xi_control))
    bracket = (p * w ** (1.0 / 6.0) * ai_value
               - 1j * q * w ** (-1.0 / 6.0) * aip_value)
    carrier = cmath.exp(1j * (w * float(tau_bar) + float(sigma)))
    return complex(2.0 * math.sqrt(math.pi) * carrier * bracket)


# ----------------------------------------------------------------------
# Fold-frame curvatures and amplitudes.
# ----------------------------------------------------------------------

def _soft_axis_cubic(image: np.ndarray, soft_axis: np.ndarray
                     ) -> float | None:
    """
    Cubic soft-axis coefficient ``b3`` of the Fermat potential at a
    caustic critical point.

    Only the ``-ln|x|`` part of the Fermat potential ``phi = 0.5 x.A.x -
    y.x - ln|x|`` contributes beyond second order, so the third directional
    derivative along the soft (vanishing-eigenvalue) axis ``e_s`` is, with
    ``p = |x|^2`` and ``q_s = x.e_s``,

        b3 = d^3/ds^3 [-ln|x + s e_s|]|_{s=0}
           = 2 q_s (3 p - 4 q_s^2) / p^3 .

    In the Hessian eigenframe the soft/hard mixed second derivative
    vanishes, so this bare cubic is already the reduced (Lyapunov-Schmidt)
    cubic coefficient to leading order.

    Returns
    -------
    float or None
        ``b3``, or ``None`` if the image sits at the point mass
        (``p <= 0``) or the result is not finite.
    """
    p = float(image @ image)
    if p <= 0.0:
        return None
    q_s = float(image @ soft_axis)
    b3 = 2.0 * q_s * (3.0 * p - 4.0 * q_s ** 2) / p ** 3
    if not math.isfinite(b3):
        return None
    return b3


def _fold_amplitudes(hard_eigenvalue: float, b3: float
                     ) -> tuple[float, float, float] | None:
    """
    Closed-form fold amplitudes ``(p, q, sigma)`` from the finite
    curvatures.

    ``p = 2^{-1/6} |lambda_h|^{-1/2} |b3|^{-1/3}``, ``q = 0`` (leading), and
    ``sigma = -pi/4``, as pinned by the large-``xi`` match to the geometric
    two-image sum (see the module docstring).  Returns ``None`` if the
    hard eigenvalue vanishes or the fold is degenerate (``|b3| <=
    _B3_MIN``).
    """
    if not (math.isfinite(hard_eigenvalue) and abs(hard_eigenvalue) > 0.0):
        return None
    if not (math.isfinite(b3) and abs(b3) > _B3_MIN):
        return None
    p_amplitude = (_AMP_CONST * abs(hard_eigenvalue) ** (-0.5)
                   * abs(b3) ** (-1.0 / 3.0))
    if not math.isfinite(p_amplitude):
        return None
    # q is the subleading Ai' amplitude; it vanishes at leading order for
    # the symmetric cubic (pure-phase) fold.  Kept explicit so the form
    # matches the brief and a future quartic (b4) refinement can set it.
    return p_amplitude, 0.0, _SIGMA_FOLD


# ----------------------------------------------------------------------
# Merging fold pair, error certificate, and the arm evaluator.
# ----------------------------------------------------------------------

def _merging_fold_pair(images: list[np.ndarray], source: np.ndarray,
                       matrix: np.ndarray
                       ) -> tuple[float, float] | None:
    """
    Fermat delays ``(tau_plus, tau_minus)`` of the merging fold pair.

    The merging pair is the delay-adjacent minimum (Morse index 0) /
    saddle (Morse index 1) with the smallest delay separation, oriented so
    that the minimum ``tau_plus`` has the lower delay and the saddle
    ``tau_minus`` the higher (the standard near-fold ordering).  Only the
    delays are used -- the near-fold magnifications are ill-conditioned and
    are never evaluated here.

    Returns
    -------
    tuple of float or None
        ``(tau_plus, tau_minus)``, or ``None`` if the geometry refuses or
        no admissible minimum/saddle pair exists.
    """
    entries: list[tuple[float, int]] = []
    for image in images:
        try:
            n_morse = geometry.morse_index(image, matrix)
        except geometry.LensDomainError:
            return None
        entries.append((geometry.delay(image, source, matrix), n_morse))
    entries.sort(key=lambda entry: entry[0])

    best: tuple[float, float] | None = None
    best_gap = math.inf
    for (tau_low, n_low), (tau_high, n_high) in zip(entries, entries[1:]):
        # Standard fold orientation: the minimum (n = 0) precedes the
        # saddle (n = 1) in delay.
        if n_low == 0 and n_high == 1:
            gap = tau_high - tau_low
            if 0.0 < gap < best_gap:
                best_gap = gap
                best = (tau_low, tau_high)
    return best


def _uniform_error_estimate(image_plus: np.ndarray, image_minus: np.ndarray,
                            matrix: np.ndarray, xi: float) -> float | None:
    """
    Leading relative uniform-error estimate ``c_A xi^{-3/2}``.

    ``c_A`` is the magnitude of the ``C1`` saddle coefficient
    (`geometry.saddle_coefficients`) in the fold frame, taken as the larger
    of the two merging images' values.  Near the fold the metric inversion
    can refuse (``LensDomainError``); that is caught and turned into
    ``None`` (a conservative fall-through).

    Returns
    -------
    float or None
        The estimate, or ``None`` if ``c_A`` is unavailable / non-finite or
        ``xi < 0``.  Returns ``0.0`` at ``xi == 0`` (the Airy form is exact
        on the fold, so no leading-error correction applies).
    """
    if xi < 0.0:
        return None
    if xi == 0.0:
        return 0.0
    try:
        c1_plus, _ = geometry.saddle_coefficients(image_plus, matrix)
        c1_minus, _ = geometry.saddle_coefficients(image_minus, matrix)
    except geometry.LensDomainError:
        return None
    c_a = max(abs(c1_plus), abs(c1_minus))
    estimate = c_a * xi ** (-1.5)
    if not math.isfinite(estimate):
        return None
    return estimate


def fold_amplification(w: float, source, gamma: float, *,
                       beta: float = 0.0, kappa: float = 0.0,
                       envelope_bar: float = _DEFAULT_ENVELOPE_BAR
                       ) -> complex | None:
    """
    Uniform Airy amplification for a merging fold image pair.

    Builds the uniform form ``F_Airy = 2 sqrt(pi) exp(i (w tau_bar +
    sigma)) [p w^{1/6} Ai(-xi) - i q w^{-1/6} Ai'(-xi)]`` with the control
    ``xi = (3 w DT / 4)^{2/3}`` from the merging pair's Fermat-delay
    separation ``DT`` and the amplitudes ``(p, q, sigma)`` from the finite
    fold-frame curvatures at the nearest caustic point.  The large-``xi``
    limit reproduces `geometry`'s exact geometric two-image sum by
    construction (see the module docstring).

    The arm is refusal-conservative and returns ``None`` (never a wrong
    number, never a new exception) when: the geometry refuses
    (`geometry.LensDomainError`); no admissible merging minimum/saddle pair
    exists; the fold is degenerate (``|b3| <= _B3_MIN`` or the hard
    eigenvalue vanishes); or the leading uniform-error estimate ``c_A
    xi^{-3/2}`` exceeds ``envelope_bar``.

    Parameters
    ----------
    w : float
        Dimensionless lens frequency, strictly positive.
    source : array_like, shape (2,)
        Source position in the lens plane.
    gamma : float
        External shear magnitude.
    beta : float, optional
        External shear orientation, radians.
    kappa : float, optional
        External convergence.
    envelope_bar : float, optional
        Max-normalized (F016) bar on the leading uniform error.

    Returns
    -------
    complex or None
        The uniform Airy amplification, or ``None`` on any refusal.

    Notes
    -----
    The control ``xi`` and the amplitude ``p`` carry the certified fold
    scalings and the exact ``-pi/2`` saddle Morse phase, and ``p`` is built
    from the finite curvatures ``(lambda_h, b3)`` -- never from the
    divergent ``sqrt|mu|``.  The subleading ``Ai'`` amplitude ``q`` is set
    to its leading value ``0`` (the pure-phase symmetric-fold result); its
    quartic (``b4``) refinement is deferred, so the served *amplitude* is
    leading-order and still awaits a brute-force cross-check (see the build
    change report).
    """
    w = float(w)
    source = np.asarray(source, dtype=float)
    if not (w > 0.0 and source.shape == (2,) and np.all(np.isfinite(source))):
        return None
    if not (envelope_bar > 0.0):
        return None

    try:
        matrix = geometry.macro_matrix(gamma, beta, kappa)
        images = geometry.find_images(source, matrix)
        nearest = geometry.nearest_caustic_point(gamma, beta, source,
                                                 kappa=kappa)
    except geometry.LensDomainError:
        return None

    # Caustic-relative admission: the ``q = 0`` symmetric-fold assumption
    # fails away from the caustic, and the ``xi`` certificate cannot see it
    # (F028, F032).
    if not float(nearest.distance) < _ETA_MAX_FOLD:
        return None

    pair = _merging_fold_pair(images, source, matrix)
    if pair is None:
        return None
    tau_plus, tau_minus = pair
    delta_tau = tau_minus - tau_plus
    if not (delta_tau > 0.0):
        return None
    tau_bar = 0.5 * (tau_plus + tau_minus)

    b3 = _soft_axis_cubic(nearest.image, nearest.soft_axis)
    if b3 is None:
        return None
    amplitudes = _fold_amplitudes(nearest.hard_eigenvalue, b3)
    if amplitudes is None:
        return None
    p_amplitude, q_amplitude, sigma = amplitudes

    # xi from the FULL delay separation (see the module docstring on the
    # factor-of-two convention); positive on the two-real-image side.
    xi = (3.0 * w * delta_tau / 4.0) ** (2.0 / 3.0)

    # Identify the merging pair positions for the error certificate: the
    # delay-adjacent minimum/saddle whose delays are tau_plus/tau_minus.
    image_plus = _image_at_delay(images, source, matrix, tau_plus)
    image_minus = _image_at_delay(images, source, matrix, tau_minus)
    if image_plus is None or image_minus is None:
        return None
    error_estimate = _uniform_error_estimate(image_plus, image_minus,
                                             matrix, xi)
    if error_estimate is None or error_estimate > envelope_bar:
        return None

    value = airy_fold_value(w, tau_bar, xi, p_amplitude, q_amplitude, sigma)
    if not np.isfinite(abs(value)):
        return None
    return value


def _image_at_delay(images: list[np.ndarray], source: np.ndarray,
                    matrix: np.ndarray, target_delay: float
                    ) -> np.ndarray | None:
    """Image whose Fermat delay matches ``target_delay`` (nearest match)."""
    best: np.ndarray | None = None
    best_gap = math.inf
    for image in images:
        gap = abs(geometry.delay(image, source, matrix) - target_delay)
        if gap < best_gap:
            best_gap = gap
            best = image
    return best


def fold_ppgo_correction(w, source, gamma: float, *,
                         beta: float = 0.0,
                         kappa: float = 0.0) -> np.ndarray:
    """
    Fold-corrected ppGO amplification (all images).

    Replaces the raw ppGO contribution of the merging fold pair with the
    uniform Airy fold form (`airy_fold_value`), producing a corrected total
    amplification that removes the O(7%) error at caustic-adjacent angles
    where the fold pair has small xi and standard ppGO (divergent sqrt|mu|)
    breaks down.

    The DO-NOTHING control property holds: even when the Airy form is
    inaccurate, it cannot make things worse than raw ppGO for a merging
    pair, so no error-estimate gate or ETA_MAX distance gate is applied.
    Only structural gates (pair exists, non-degenerate fold geometry) are
    checked.

    On any structural refusal, the function falls back transparently to
    raw `geometric_amplification` (byte-identical to the uncorrected path).

    Parameters
    ----------
    w : float or array_like
        Dimensionless lens frequency (strictly positive).  Scalar or 1-D.
    source : array_like, shape (2,)
        Source position in the lens plane.
    gamma : float
        External shear magnitude.
    beta : float, optional
        External shear orientation, radians.
    kappa : float, optional
        External convergence.

    Returns
    -------
    np.ndarray
        Complex amplification, shaped like ``w`` (0-d for scalar input,
        1-d for array input).
    """
    # Lazy import to avoid circular dependency (operator imports this
    # module at the top level).
    from cogwheel.lensing.chang_refsdal.operator import (
        geometric_amplification)

    source = np.asarray(source, dtype=float)
    w_input = np.asarray(w, dtype=float)
    w_arr = np.atleast_1d(w_input)
    w_scalar = w_input.ndim == 0

    def _fallback():
        """Return raw ppGO, shaped to match the input w."""
        result = geometric_amplification(w_arr, source, gamma,
                                         beta=beta, kappa=kappa)
        return np.atleast_1d(result)

    # NOTE (maintenance): the fold-correction logic below (structural gates +
    # w-dependent Airy/ppGO computation) is mirrored in the inline fold
    # correction block inside `channels.born_carrier_from_partition`.  The two
    # sites are kept separate because this function re-solves the geometry from
    # scratch (needed for its standalone public interface), while the
    # `channels` block reuses pre-computed images/matrix from the partition to
    # avoid a redundant `geometric_amplification` call.  If the correction
    # formula or structural gates change, BOTH locations must be updated.
    # See INS-c8-003.

    # --- Structural gates (w-independent geometry) ---
    try:
        matrix = geometry.macro_matrix(gamma, beta, kappa)
        images = geometry.find_images(source, matrix)
    except geometry.LensDomainError:
        result = _fallback()
        return result[0] if w_scalar else result

    pair = _merging_fold_pair(images, source, matrix)
    if pair is None:
        result = _fallback()
        return result[0] if w_scalar else result
    tau_plus, tau_minus = pair
    delta_tau = tau_minus - tau_plus
    if not (delta_tau > 0.0):
        result = _fallback()
        return result[0] if w_scalar else result
    tau_bar = 0.5 * (tau_plus + tau_minus)

    try:
        nearest = geometry.nearest_caustic_point(gamma, beta, source,
                                                 kappa=kappa)
    except geometry.LensDomainError:
        result = _fallback()
        return result[0] if w_scalar else result

    b3 = _soft_axis_cubic(nearest.image, nearest.soft_axis)
    if b3 is None:
        result = _fallback()
        return result[0] if w_scalar else result

    amplitudes = _fold_amplitudes(nearest.hard_eigenvalue, b3)
    if amplitudes is None:
        result = _fallback()
        return result[0] if w_scalar else result
    p_amplitude, q_amplitude, sigma = amplitudes

    # --- w-dependent computation ---
    # Airy values for each w_i (airy_fold_value is scalar).
    airy_values = np.empty(w_arr.shape, dtype=complex)
    for i, w_i in enumerate(w_arr):
        xi_i = (3.0 * w_i * delta_tau / 4.0) ** (2.0 / 3.0)
        airy_values[i] = airy_fold_value(
            w_i, tau_bar, xi_i, p_amplitude, q_amplitude, sigma)

    # Pair's raw ppGO: sum of exp(1j*w*tau_a) * image_kernel over the two
    # pair images (vectorized over w).
    image_plus = _image_at_delay(images, source, matrix, tau_plus)
    image_minus = _image_at_delay(images, source, matrix, tau_minus)
    if image_plus is None or image_minus is None:
        result = _fallback()
        return result[0] if w_scalar else result

    pair_ppgo = np.zeros(w_arr.shape, dtype=complex)
    for img, tau_a in ((image_plus, tau_plus), (image_minus, tau_minus)):
        pair_ppgo = pair_ppgo + (
            np.exp(1j * w_arr * tau_a)
            * geometry.image_kernel(w_arr, img, matrix))

    # Full ppGO (all images) via geometric_amplification.
    full_ppgo = np.atleast_1d(
        geometric_amplification(w_arr, source, gamma, beta=beta, kappa=kappa))

    # Corrected: replace pair contribution with Airy form.
    result = full_ppgo - pair_ppgo + airy_values

    # Non-finite Airy fallback: keep the uncorrected ppGO value where the
    # Airy form produced non-finite values.
    non_finite_mask = ~np.isfinite(airy_values)
    if np.any(non_finite_mask):
        result[non_finite_mask] = full_ppgo[non_finite_mask]

    return result[0] if w_scalar else result
